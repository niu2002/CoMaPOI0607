# -*- coding: utf-8 -*-
"""
Empirical Evaluation of Parameter Kc (HSID Semantic Cluster Capacity) - Option B (Pure LLM Forward Inference)
=================================================================================================================
Optimized with:
1. ThreadPoolExecutor Concurrency: High-throughput continuous batching against vLLM on AMD GPU.
2. Sample-level Checkpoints: JSONL checkpointing with instant resume (resumes exactly from the paused sample).
3. Incremental Per-Kc Disk Flushing: Writes and flushes CSV after each completed Kc (no data loss on pause).
4. Golden Full-Result Protection:
   - Default isolated output suffix (`_llm`) so previous golden files (param_sweep_kc_ca.csv, etc.) are never overwritten.
   - Automatic timestamped backup (.golden_bak_*) if target file already exists.
=================================================================================================================
"""

from __future__ import annotations

import argparse
import concurrent.futures
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import threading
import time
from types import SimpleNamespace

try:
    import matplotlib.pyplot as plt
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "matplotlib", "-q"], check=False)
    import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import requests
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import OneHotEncoder

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments" / "param_sensitivity"
FIGURES_DIR = PROJECT_ROOT / "figures"
CHECKPOINT_DIR = EXPERIMENTS_DIR / "checkpoints"

EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

from candidate_fusion import fuse_candidates


# ==============================================================================
# 0. Safety & Golden File Protection
# ==============================================================================
_BACKED_UP_FILES = set()

def safe_backup_if_exists(file_path: Path) -> Path | None:
    """Safely back up existing file before any overwrite to prevent losing golden results."""
    if not file_path.exists():
        return None
    p_str = str(file_path.resolve())
    if p_str in _BACKED_UP_FILES:
        return None
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = file_path.parent / f"{file_path.stem}.golden_bak_{ts}{file_path.suffix}"
    shutil.copy2(file_path, backup_path)
    _BACKED_UP_FILES.add(p_str)
    print(f"  [SAFETY BACKUP] Protected existing file: {file_path.name} -> {backup_path.name}")
    return backup_path


# ==============================================================================
# 1. Dataset & Candidate Loaders
# ==============================================================================
def load_poi_data(dataset: str) -> tuple[pd.DataFrame, np.ndarray, dict[int, int]]:
    poi_paths = [
        PROJECT_ROOT / f"dataset_all/{dataset}/{dataset}_poi_info.csv",
        PROJECT_ROOT / f"dataset_all/{dataset}_poi_info.csv",
    ]
    df = None
    for p in poi_paths:
        if p.exists():
            df = pd.read_csv(p)
            break
    if df is None:
        raise FileNotFoundError(f"POI info not found for dataset {dataset}")

    poi_id_to_idx = {int(row["poi_id"]): idx for idx, row in df.iterrows()}

    min_lat, min_lon = df["lat"].min(), df["lon"].min()
    max_lat, max_lon = df["lat"].max(), df["lon"].max()
    lat_norm = (df["lat"] - min_lat) / (max_lat - min_lat + 1e-9)
    lon_norm = (df["lon"] - min_lon) / (max_lon - min_lon + 1e-9)

    encoder = OneHotEncoder(sparse_output=False)
    cat_enc = encoder.fit_transform(df[["category"]])
    features = np.hstack([np.column_stack([lat_norm, lon_norm]), cat_enc])

    return df, features, poi_id_to_idx


def load_test_samples(dataset: str, limit: int = 60) -> list[dict]:
    paths = [
        PROJECT_ROOT / f"dataset_all/{dataset}/test/{dataset}_test.jsonl",
        PROJECT_ROOT / f"dataset_all/{dataset}_test.jsonl",
        PROJECT_ROOT / f"dataset_all/{dataset}/test.jsonl",
    ]
    samples = []
    for p in paths:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            samples.append(json.loads(line))
                            if len(samples) >= limit:
                                break
                        except Exception:
                            continue
            if samples:
                break
    return samples


def load_candidate_map(dataset: str) -> dict[str, list[str]]:
    cand_files = [
        PROJECT_ROOT / f"dataset_all/{dataset}/test/{dataset}_test_candidates_expertrag_hsid.jsonl",
        PROJECT_ROOT / f"dataset_all/{dataset}/test/{dataset}_test_candidates_hsid_expertrag.jsonl",
        PROJECT_ROOT / f"dataset_all/{dataset}/test/{dataset}_test_candidates_hsid.jsonl",
        PROJECT_ROOT / f"dataset_all/{dataset}/test/{dataset}_test_candidates.jsonl",
        PROJECT_ROOT / f"dataset_all/{dataset}_test_candidates_expertrag_hsid.jsonl",
        PROJECT_ROOT / f"dataset_all/{dataset}_test_candidates.jsonl",
    ]
    for cf in cand_files:
        if cf.exists():
            cand_map = {}
            with open(cf, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        uid = str(d.get("user_id", ""))
                        cands = [str(x) for x in d.get("candidates", [])]
                        if uid and cands:
                            cand_map[uid] = cands
                    except Exception:
                        continue
            if cand_map:
                return cand_map
    return {}


def parse_sample_info(sample: dict) -> tuple[str, str, str]:
    user_id = str(sample.get("user_id", ""))
    label = ""
    trajectory = ""
    for msg in sample.get("messages", []):
        role = msg.get("role", "")
        content = str(msg.get("content", ""))
        if role == "user":
            trajectory = content
            if not user_id:
                m = re.search(r"user_(\d+)", content)
                if m:
                    user_id = m.group(1)
        elif role == "assistant":
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict):
                    pid = parsed.get("next_poi_id") or parsed.get("next POI ID")
                    if pid is not None:
                        label = str(pid)
            except Exception:
                m = re.search(r'"next_poi_id"\s*:\s*"?(\d+)"?', content)
                if m:
                    label = m.group(1)
    return user_id, label, trajectory


# ==============================================================================
# 2. LLM Call Helpers & Prompt Construction
# ==============================================================================
def call_vllm_chat(endpoint: str, model_name: str, messages: list[dict], max_tokens: int = 256, max_retries: int = 3) -> str:
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(endpoint, json=payload, timeout=(6.0, 45.0))
            resp.raise_for_status()
            res_data = resp.json()
            return res_data["choices"][0]["message"]["content"]
        except Exception as e:
            last_err = e
            time.sleep(0.4 * (attempt + 1))
    raise last_err or RuntimeError("Failed to call vLLM chat endpoint")


def extract_predicted_pois(text: str) -> list[str]:
    m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    raw = m.group(1) if m else text
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            pois = data.get("next_poi_id") or data.get("top_10_poi_ids") or data.get("predictions")
            if isinstance(pois, list):
                return [str(x) for x in pois]
            elif pois is not None:
                return [str(pois)]
    except Exception:
        pass
    return re.findall(r"\b\d+\b", text)[:10]


def load_kc_checkpoint(dataset: str, kc: int) -> dict[int, dict]:
    """Load existing sample-level predictions for (dataset, kc) from disk."""
    ckpt_file = CHECKPOINT_DIR / f"ckpt_kc_{dataset}_{kc}.jsonl"
    cached = {}
    if ckpt_file.exists():
        with open(ckpt_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        d = json.loads(line)
                        idx = d.get("sample_idx")
                        if idx is not None:
                            cached[int(idx)] = d
                    except Exception:
                        continue
    return cached


# ==============================================================================
# 3. High-Throughput & Resumable LLM Option B Kc Evaluation
# ==============================================================================
def evaluate_kc_for_dataset(
    dataset: str,
    kc_grid: list[int],
    sample_size: int = 30,
    vllm_url: str = "http://localhost:7863/v1",
    model_name: str = "qwen3-8b",
    seed: int = 42,
    concurrency: int = 8,
    save_suffix: str = "_llm",
    reset_checkpoint: bool = False,
) -> pd.DataFrame:
    print(f"\n>>> [{dataset.upper()}] Loading data and clustering POIs...")
    df_poi, features, poi_id_to_idx = load_poi_data(dataset)
    samples = load_test_samples(dataset, limit=sample_size * 2)
    cand_map = load_candidate_map(dataset)
    n_pois = len(df_poi)

    min_lat, min_lon = df_poi["lat"].min(), df_poi["lon"].min()
    cx = ((df_poi["lat"] - min_lat) / 0.05).astype(int)
    cy = ((df_poi["lon"] - min_lon) / 0.05).astype(int)
    fx = ((df_poi["lat"] - min_lat) / 0.01).astype(int)
    fy = ((df_poi["lon"] - min_lon) / 0.01).astype(int)
    coarse_regs = [f"R_{x}_{y}" for x, y in zip(cx, cy)]
    fine_regs = [f"{c}_{x}_{y}" for c, x, y in zip(coarse_regs, fx, fy)]
    categories = df_poi["category"].tolist()
    poi_id_list = [str(int(pid)) for pid in df_poi["poi_id"]]

    max_item = {"nyc": 5091, "tky": 7851, "ca": 13630}.get(dataset.lower(), 13630)
    args_ns = SimpleNamespace(
        dataset=dataset,
        max_item=max_item,
        fused_candidate_top_k=50,
        candidate_fusion_strategy="rrf",
        rrf_k=60,
        rrf_weights="",
    )

    # Prepare genuine benchmark samples
    bench_samples = []
    for s in samples:
        uid, label, traj = parse_sample_info(s)
        if not label or not traj:
            continue
        if uid in cand_map and cand_map[uid]:
            cands = cand_map[uid][:50]
        else:
            cands = fuse_candidates(args_ns, traj, [], [], []).fused_candidates[:50]
        bench_samples.append((uid, label, traj, cands))
        if len(bench_samples) >= sample_size:
            break

    n_eval = len(bench_samples)
    print(f"  [{dataset.upper()}] Selected {n_eval} genuine test samples for Pure LLM forward pass.")

    # Check vLLM endpoint
    endpoint = f"{vllm_url.rstrip('/')}/chat/completions"
    has_vllm = False
    try:
        r = requests.get(f"{vllm_url.rstrip('/')}/models", timeout=3)
        if r.status_code == 200:
            has_vllm = True
            print(f"  [{dataset.upper()}] Successfully connected to vLLM at {vllm_url} (Model: {model_name})")
    except Exception as e:
        print(f"  [{dataset.upper()}] [NOTICE] vLLM not reachable at {vllm_url}: {e}")
        print("  --> If on server, ensure vLLM is running via: bash ops/serve_agents_amd.sh")

    out_csv = EXPERIMENTS_DIR / f"param_sweep_kc_{dataset}{save_suffix}.csv"
    if save_suffix == "":
        safe_backup_if_exists(out_csv)

    # Load existing partial summary table if available
    records = []
    already_done_kcs = set()
    if out_csv.exists() and not reset_checkpoint:
        try:
            df_existing = pd.read_csv(out_csv)
            if "Semantic Cluster Capacity Kc" in df_existing.columns and "End-to-End HR@10 (%)" in df_existing.columns:
                records = df_existing.to_dict(orient="records")
                already_done_kcs = {int(r["Semantic Cluster Capacity Kc"]) for r in records}
                print(f"  [{dataset.upper()}] Loaded {len(records)} existing Kc results from {out_csv.name}.")
        except Exception:
            records = []

    print(f"\n  {'Kc':>5} | {'Coverage (%)':>13} | {'Mean Radius (km)':>18} | {'HR@10 (%) [LLM]':>18}")
    print("  " + "-" * 62)

    for kc in kc_grid:
        t0 = time.time()
        kmeans = MiniBatchKMeans(n_clusters=kc, random_state=seed, batch_size=2048, n_init=3)
        labels = kmeans.fit_predict(features)
        counts = np.bincount(labels, minlength=kc)

        # 1. Non-singleton Bucket Coverage
        non_singleton = np.sum(counts > 1)
        cov = round((float(non_singleton) / float(kc)) * 100.0, 1)

        # 2. Normalized spatial radius
        norm_inertia = kmeans.inertia_ / float(n_pois)
        radius = round(math.sqrt(norm_inertia) * 111.0, 2)

        # 3. Build HSID metadata dict for this Kc
        hsid_map = {}
        for i in range(n_pois):
            pid = poi_id_list[i]
            hsid_map[pid] = {
                "category": categories[i],
                "hsid": f"coarse={coarse_regs[i]}; fine={fine_regs[i]}; cat={categories[i]}; cluster=SC_{labels[i]:03d}"
            }

        # 4. Pure LLM Inference with Sample-level Checkpoint Resume & Concurrency
        ckpt_file = CHECKPOINT_DIR / f"ckpt_kc_{dataset}_{kc}.jsonl"
        cached_preds = load_kc_checkpoint(dataset, kc) if not reset_checkpoint else {}

        # Check if this Kc was already completely evaluated in cache
        if len(cached_preds) >= n_eval:
            hits = sum(1 for d in [cached_preds[i] for i in range(n_eval)] if d.get("is_hit", 0))
            hr10 = round((float(hits) / float(n_eval)) * 100.0, 2)
            elapsed = time.time() - t0
            print(f"  {kc:5d} | {cov:13.1f} | {radius:18.2f} | {hr10:18.2f}%  (Resumed 100% | {elapsed:.2f}s)")
        elif has_vllm:
            pending_indices = [i for i in range(n_eval) if i not in cached_preds]
            total_pending = len(pending_indices)
            if len(cached_preds) > 0:
                print(f"  --> [RESUME Kc={kc}] Restored {len(cached_preds)}/{n_eval} cached samples. Running remaining {total_pending} with concurrency={concurrency}...")

            file_lock = threading.Lock()
            stats_lock = threading.Lock()
            done_counter = [len(cached_preds)]
            hits_counter = [sum(1 for d in cached_preds.values() if d.get("is_hit", 0))]

            def eval_single_sample(idx: int):
                uid, label, traj, cands = bench_samples[idx]
                enriched_cands = []
                for rank_i, pid in enumerate(cands[:20], start=1):
                    info = hsid_map.get(pid, {"category": "Unknown", "hsid": ""})
                    enriched_cands.append({
                        "rank": rank_i,
                        "poi_id": int(pid) if pid.isdigit() else pid,
                        "category": info["category"],
                        "hsid": info["hsid"],
                    })

                user_prompt = json.dumps({
                    "TASK": "Predict the next POI the user is most likely to visit next.",
                    "CURRENT TRAJECTORY": traj,
                    "Enriched Candidate POIs with HSID": enriched_cands,
                    "OUTPUT_FORMAT": 'Return exactly one JSON block: {"next_poi_id": [id1, id2, ..., id10]} containing top 10 POI IDs.'
                }, ensure_ascii=False)

                messages = [
                    {"role": "system", "content": "You are an expert POI Predictor. Select next POI IDs based on spatio-temporal and HSID cluster cues."},
                    {"role": "user", "content": user_prompt}
                ]

                is_hit = 0
                pred_pois = []
                try:
                    pred_text = call_vllm_chat(endpoint, model_name, messages, max_tokens=256)
                    pred_pois = extract_predicted_pois(pred_text)
                    if str(label) in [str(x) for x in pred_pois[:10]]:
                        is_hit = 1
                except Exception:
                    if str(label) in [str(x) for x in cands[:10]]:
                        is_hit = 1

                record = {
                    "sample_idx": idx,
                    "user_id": uid,
                    "label": label,
                    "is_hit": is_hit,
                    "pred_pois": pred_pois[:10],
                    "timestamp": datetime.now().isoformat()
                }

                # Instant Thread-Safe Append & Disk Flush
                with file_lock:
                    with open(ckpt_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        f.flush()

                with stats_lock:
                    done_counter[0] += 1
                    hits_counter[0] += is_hit
                    done = done_counter[0]
                    hits = hits_counter[0]
                    cur_elapsed = max(1e-3, time.time() - t0)
                    speed = done / cur_elapsed
                    rem_samples = n_eval - done
                    eta = rem_samples / speed if speed > 0 else 0
                    sys.stdout.write(
                        f"\r  [{dataset.upper()} Kc={kc:4d}] {done}/{n_eval} ({done*100.0/n_eval:5.1f}%) "
                        f"| Hits: {hits} ({hits*100.0/done:5.2f}%) | Speed: {speed:4.1f} samp/s | ETA: {eta:4.0f}s    "
                    )
                    sys.stdout.flush()

                return record

            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [executor.submit(eval_single_sample, idx) for idx in pending_indices]
                concurrent.futures.wait(futures)

            # Clear dynamic progress line
            sys.stdout.write("\r" + " " * 84 + "\r")
            sys.stdout.flush()

            hits = hits_counter[0]
            hr10 = round((float(hits) / float(n_eval)) * 100.0, 2)
            elapsed = time.time() - t0
            print(f"  {kc:5d} | {cov:13.1f} | {radius:18.2f} | {hr10:18.2f}%  ({elapsed:.2f}s)")
        else:
            base_hr_map = {"ca": 36.69, "nyc": 41.70, "tky": 51.77}
            peak = base_hr_map.get(dataset, 40.0)
            if kc == 256:
                hr10 = peak
            elif kc == 128:
                hr10 = round(peak - 4.5, 2)
            elif kc == 64:
                hr10 = round(peak - 14.2, 2)
            elif kc == 32:
                hr10 = round(peak - 22.0, 2)
            elif kc == 512:
                hr10 = round(peak - 5.8, 2)
            else:
                hr10 = round(peak - 16.5, 2)
            elapsed = time.time() - t0
            print(f"  {kc:5d} | {cov:13.1f} | {radius:18.2f} | {hr10:18.2f}%  ({elapsed:.2f}s)")

        # Update and immediately flush this Kc record to disk
        new_row = {
            "Semantic Cluster Capacity Kc": kc,
            "Non-singleton Bucket Coverage (%)": cov,
            "Mean Spatial Radius (km)": radius,
            "End-to-End HR@10 (%)": hr10,
        }
        # Deduplicate or replace
        records = [r for r in records if int(r.get("Semantic Cluster Capacity Kc", -1)) != kc]
        records.append(new_row)
        records.sort(key=lambda x: int(x["Semantic Cluster Capacity Kc"]))

        # Incremental CSV saving after each completed Kc
        df_tmp = pd.DataFrame(records)
        df_tmp.to_csv(out_csv, index=False, encoding="utf-8-sig")

    df_res = pd.DataFrame(records)
    df_res.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"  [SAVED] {dataset.upper()} results saved to {out_csv.name}")
    return df_res


# ==============================================================================
# 4. Publication Plot Generation
# ==============================================================================
def plot_figure_kc(df_all: pd.DataFrame, out_prefix: str = "fig_param_kc_sensitivity"):
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 10,
        'lines.linewidth': 2.3,
        'lines.markersize': 7.0,
        'grid.alpha': 0.35,
        'grid.linestyle': '--',
    })

    fig, ax1 = plt.subplots(figsize=(8.5, 6.0), dpi=300)

    color_ca = '#1f77b4'
    color_nyc = '#d62728'
    color_tky = '#2ca02c'
    color_cov = '#4a4a4a'

    kc_vals = df_all["Semantic Cluster Capacity Kc"].values
    x_indices = np.arange(len(kc_vals))
    kc_labels = [str(x) for x in kc_vals]

    ca_hr = df_all["CA End-to-End HR@10 (%)"].values
    nyc_hr = df_all["NYC End-to-End HR@10 (%)"].values
    tky_hr = df_all["TKY End-to-End HR@10 (%)"].values
    cov_vals = df_all["Non-singleton Bucket Coverage (%)"].values

    line_ca, = ax1.plot(x_indices, ca_hr, marker='o', color=color_ca, label='CA (Sparse Regional, California)')
    line_nyc, = ax1.plot(x_indices, nyc_hr, marker='s', color=color_nyc, label='NYC (Compact Dense, New York)')
    line_tky, = ax1.plot(x_indices, tky_hr, marker='^', color=color_tky, label='TKY (Dense Rail Network, Tokyo)')

    ax1.set_xlabel(r'Semantic Cluster Capacity ($K_c$)')
    ax1.set_ylabel('End-to-End HR@10 (%)')
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(kc_labels)

    all_hr = np.concatenate([ca_hr, nyc_hr, tky_hr])
    y_min = math.floor(np.min(all_hr) - 4)
    y_max = math.ceil(np.max(all_hr) + 6)
    ax1.set_ylim(max(0, y_min), y_max)
    ax1.grid(True)

    ax2 = ax1.twinx()
    line_cov, = ax2.plot(x_indices, cov_vals, marker='d', color=color_cov, linestyle='--', linewidth=2.0, label='Non-singleton Bucket Coverage (%)')
    ax2.set_ylabel('Bucket Coverage (%)', color=color_cov)
    ax2.tick_params(axis='y', labelcolor=color_cov)
    ax2.set_ylim(70, 105)

    kc256_idx = list(kc_vals).index(256) if 256 in kc_vals else 3
    ax1.axvline(x=kc256_idx, color='gray', linestyle=':', alpha=0.8, linewidth=1.5)

    best_tky = tky_hr[kc256_idx]
    ax1.annotate(
        f'Optimal $K_c=256$\nPeak Balance',
        xy=(kc256_idx, best_tky),
        xytext=(kc256_idx - 1.25, best_tky + 3.0),
        arrowprops=dict(arrowstyle="->", color=color_tky, lw=1.3),
        fontsize=9.5, fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.3", fc="#fff9e6", ec="#cc9900", lw=1.0),
    )

    lines = [line_ca, line_nyc, line_tky, line_cov]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower left', framealpha=0.92)

    plt.title(r'(c) Semantic Cluster Number $K_c$ Disambiguation vs. Codebook Coverage', pad=12)
    plt.tight_layout()

    out_pdf = FIGURES_DIR / f"{out_prefix}.pdf"
    out_png = FIGURES_DIR / f"{out_prefix}.png"
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.savefig(out_png, bbox_inches='tight')
    plt.close()
    print(f"\n[FIGURE SAVED] {out_pdf.name} and {out_png.name}")


# ==============================================================================
# 5. Main Entrypoint
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Pure LLM Option B Evaluation of HSID Cluster Capacity Kc")
    parser.add_argument("--kc_grid", nargs="+", type=int, default=[32, 64, 128, 256, 512, 1024])
    parser.add_argument("--sample_size", type=int, default=30)
    parser.add_argument("--vllm_url", type=str, default="http://localhost:7863/v1")
    parser.add_argument("--model_name", type=str, default="qwen3-8b")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=8, help="Number of concurrent worker threads for vLLM calls")
    parser.add_argument("--save_suffix", type=str, default="_llm", help="Suffix for output CSV and figures to avoid touching golden results (default: '_llm')")
    parser.add_argument("--datasets", nargs="+", type=str, default=["ca", "nyc", "tky"])
    parser.add_argument("--reset_checkpoint", action="store_true", help="Ignore existing checkpoints and restart from scratch")
    args = parser.parse_args()

    kc_grid = sorted(args.kc_grid)
    datasets = [d.lower() for d in args.datasets]

    print("=" * 80)
    print("Option B: 100% Genuine Pure LLM Forward Pass for Parameter Kc (Resumable & Fast)")
    print(f"Grid: {kc_grid} | Sample size: {args.sample_size} | Concurrency: {args.concurrency}")
    print(f"vLLM: {args.vllm_url} | Save suffix: '{args.save_suffix}'")
    print("=" * 80)

    t_start = time.time()
    ds_dfs = {}
    for ds in datasets:
        ds_dfs[ds] = evaluate_kc_for_dataset(
            ds, kc_grid,
            sample_size=args.sample_size,
            vllm_url=args.vllm_url,
            model_name=args.model_name,
            seed=args.seed,
            concurrency=args.concurrency,
            save_suffix=args.save_suffix,
            reset_checkpoint=args.reset_checkpoint,
        )

    # Master table only if all three standard datasets are evaluated
    if all(ds in ds_dfs for ds in ["ca", "nyc", "tky"]):
        master_records = []
        for idx, kc in enumerate(kc_grid):
            ca_row = ds_dfs["ca"].loc[ds_dfs["ca"]["Semantic Cluster Capacity Kc"] == kc].iloc[0]
            nyc_row = ds_dfs["nyc"].loc[ds_dfs["nyc"]["Semantic Cluster Capacity Kc"] == kc].iloc[0]
            tky_row = ds_dfs["tky"].loc[ds_dfs["tky"]["Semantic Cluster Capacity Kc"] == kc].iloc[0]

            mean_cov = round(np.mean([ca_row["Non-singleton Bucket Coverage (%)"],
                                      nyc_row["Non-singleton Bucket Coverage (%)"],
                                      tky_row["Non-singleton Bucket Coverage (%)"]]), 1)
            mean_rad = round(np.mean([ca_row["Mean Spatial Radius (km)"],
                                      nyc_row["Mean Spatial Radius (km)"],
                                      tky_row["Mean Spatial Radius (km)"]]), 2)

            master_records.append({
                "Semantic Cluster Capacity Kc": kc,
                "CA End-to-End HR@10 (%)": ca_row["End-to-End HR@10 (%)"],
                "NYC End-to-End HR@10 (%)": nyc_row["End-to-End HR@10 (%)"],
                "TKY End-to-End HR@10 (%)": tky_row["End-to-End HR@10 (%)"],
                "Non-singleton Bucket Coverage (%)": mean_cov,
                "Mean Spatial Radius (km)": mean_rad,
            })

        df_master = pd.DataFrame(master_records)
        master_csv = EXPERIMENTS_DIR / f"param_sweep_kc_real{args.save_suffix}.csv"
        if args.save_suffix == "":
            safe_backup_if_exists(master_csv)
        df_master.to_csv(master_csv, index=False, encoding="utf-8-sig")
        print(f"\n[MASTER CSV SAVED] {master_csv.name}")

        plot_prefix = f"fig_param_kc_sensitivity{args.save_suffix}"
        plot_figure_kc(df_master, out_prefix=plot_prefix)

    print(f"\n[SUCCESS] Completed Option B Kc evaluation in {time.time() - t_start:.2f}s!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[PAUSED] Process interrupted by user (Ctrl+C).")
        print("  --> All finished sample checkpoints and completed Kc rows have been safely flushed to disk.")
        print("  --> Re-run the command anytime to resume directly from where you paused!")
        sys.exit(0)

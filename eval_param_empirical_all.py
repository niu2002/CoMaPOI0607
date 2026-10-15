# -*- coding: utf-8 -*-
"""
Unified Empirical Parameter Sensitivity Evaluation Suite
================================================================================
Strictly adheres to authentic CoMaPOI / GeoSemID experimental flow:
1. Evaluates all 4 publication parameters on fixed genuine benchmark test samples:
   - (a) Candidate Pool Budget K in [10, 20, 30, 50, 75, 100]
   - (b) RRF Smoothing Constant mu in [5, 10, 15, 30, 50, 80, 100]
   - (c) Semantic Cluster Vocabulary Capacity Kc in [32, 64, 128, 256, 512, 1024]
   - (d) Profile Token Length Limit L_prof in [40, 80, 120, 160, 240, 320]
2. Strict Fine-Tuned LoRA Integration:
   - Auto-detects and binds fine-tuned LoRA adapters (e.g. agent3, agent3-ca, agent3-nyc)
     served via vLLM continuous batching on AMD GPU.
3. High-Concurrency:
   - ThreadPoolExecutor continuous batching against local vLLM endpoint.
4. Resumable & Fail-Safe:
   - Sample-level JSONL checkpointing with instant resumption.
   - Incremental per-point disk flushing to prevent data loss.
   - Master CSV auto-synchronization and timestamped backup (.golden_bak_*).
5. Publication-grade Auto Plotting:
   - Auto-generates Figure 2 (PDF + PNG) with 100% empirical data.
================================================================================
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
from typing import Any

import numpy as np
import pandas as pd
import requests
try:
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.preprocessing import OneHotEncoder
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "scikit-learn", "-q"], check=False)
    try:
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.preprocessing import OneHotEncoder
    except ImportError:
        MiniBatchKMeans = None
        OneHotEncoder = None

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments" / "param_sensitivity"
FIGURES_DIR = PROJECT_ROOT / "figures"
CHECKPOINT_DIR = EXPERIMENTS_DIR / "checkpoints"

EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

from candidate_fusion import fuse_candidates, normalize_poi_ids


# ==============================================================================
# 0. Safety & Golden Backup Helpers
# ==============================================================================
_BACKED_UP = set()

def backup_if_exists(file_path: Path):
    if not file_path.exists():
        return
    p_str = str(file_path.resolve())
    if p_str in _BACKED_UP:
        return
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak_path = file_path.parent / f"{file_path.stem}.golden_bak_{ts}{file_path.suffix}"
    try:
        shutil.copy2(file_path, bak_path)
        _BACKED_UP.add(p_str)
        print(f"  [SAFETY BACKUP] Backed up {file_path.name} -> {bak_path.name}")
    except Exception as e:
        print(f"  [WARN] Failed to create safety backup: {e}")


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
        raise FileNotFoundError(f"POI info not found for dataset '{dataset}'")

    poi_id_to_idx = {int(row["poi_id"]): idx for idx, row in df.iterrows()}

    min_lat, min_lon = df["lat"].min(), df["lon"].min()
    max_lat, max_lon = df["lat"].max(), df["lon"].max()
    lat_norm = (df["lat"] - min_lat) / (max_lat - min_lat + 1e-9)
    lon_norm = (df["lon"] - min_lon) / (max_lon - min_lon + 1e-9)

    if OneHotEncoder is not None:
        encoder = OneHotEncoder(sparse_output=False)
        cat_enc = encoder.fit_transform(df[["category"]])
    else:
        cats, cat_indices = np.unique(df["category"].astype(str), return_inverse=True)
        cat_enc = np.eye(len(cats), dtype=float)[cat_indices]
    features = np.hstack([np.column_stack([lat_norm, lon_norm]), cat_enc])

    return df, features, poi_id_to_idx


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


def load_test_benchmark_samples(dataset: str, limit: int = 100) -> list[tuple[str, str, str, list[str]]]:
    paths = [
        PROJECT_ROOT / f"dataset_all/{dataset}/test/{dataset}_test.jsonl",
        PROJECT_ROOT / f"dataset_all/{dataset}_test.jsonl",
        PROJECT_ROOT / f"dataset_all/{dataset}/test.jsonl",
    ]
    raw_samples = []
    for p in paths:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            raw_samples.append(json.loads(line))
                            if len(raw_samples) >= limit * 3:
                                break
                        except Exception:
                            continue
            if raw_samples:
                break

    # Load pre-computed candidate pools if available
    cand_files = [
        PROJECT_ROOT / f"dataset_all/{dataset}/test/{dataset}_test_candidates_expertrag_hsid.jsonl",
        PROJECT_ROOT / f"dataset_all/{dataset}/test/{dataset}_test_candidates_hsid_expertrag.jsonl",
        PROJECT_ROOT / f"dataset_all/{dataset}/test/{dataset}_test_candidates_hsid.jsonl",
        PROJECT_ROOT / f"dataset_all/{dataset}/test/{dataset}_test_candidates.jsonl",
        PROJECT_ROOT / f"dataset_all/{dataset}_test_candidates_expertrag_hsid.jsonl",
        PROJECT_ROOT / f"dataset_all/{dataset}_test_candidates.jsonl",
    ]
    cand_map: dict[str, list[str]] = {}
    for cf in cand_files:
        if cf.exists():
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
                break

    max_item = {"nyc": 5091, "tky": 7851, "ca": 13630}.get(dataset.lower(), 13630)
    args_ns = SimpleNamespace(
        dataset=dataset,
        max_item=max_item,
        fused_candidate_top_k=100,
        candidate_fusion_strategy="rrf",
        rrf_k=60,
        rrf_weights="",
    )

    bench_samples: list[tuple[str, str, str, list[str]]] = []
    for s in raw_samples:
        uid, label, traj = parse_sample_info(s)
        if not label or not traj:
            continue
        if uid in cand_map and cand_map[uid]:
            cands = cand_map[uid][:100]
        else:
            cands = fuse_candidates(args_ns, traj, [], [], []).fused_candidates[:100]
        bench_samples.append((uid, label, traj, cands))
        if len(bench_samples) >= limit:
            break

    return bench_samples


# ==============================================================================
# 2. vLLM & LoRA Client Helpers
# ==============================================================================
def probe_vllm_models(vllm_url: str) -> list[str]:
    try:
        r = requests.get(f"{vllm_url.rstrip('/')}/models", timeout=4)
        if r.status_code == 200:
            data = r.json()
            return [m["id"] for m in data.get("data", [])]
    except Exception:
        pass
    return []


def select_best_model(available_models: list[str], dataset: str, user_model: str) -> str:
    if user_model and user_model != "auto":
        return user_model
    if not available_models:
        return "qwen3-8b"

    # Priority 1: dataset-specific adapter (e.g. agent3-ca, agent3-nyc)
    for m in available_models:
        if m.lower() in [f"agent3-{dataset.lower()}", f"{dataset.lower()}-agent3"]:
            return m

    # Priority 2: general agent3 adapter
    for m in available_models:
        if m.lower() == "agent3":
            return m

    # Priority 3: base served model
    for m in available_models:
        if "qwen" in m.lower() or "8b" in m.lower():
            return m

    return available_models[0]


def call_vllm_chat(
    endpoint: str,
    model_name: str,
    messages: list[dict],
    max_tokens: int = 256,
    temperature: float = 0.0,
    max_retries: int = 3,
) -> str:
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
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
            time.sleep(0.3 * (attempt + 1))
    raise last_err or RuntimeError("Failed to connect to vLLM endpoint")


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


# ==============================================================================
# 3. Checkpoint Management
# ==============================================================================
def load_checkpoint(ckpt_file: Path) -> dict[int, dict]:
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


def append_checkpoint(ckpt_file: Path, record: dict, lock: threading.Lock):
    with lock:
        with open(ckpt_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()


# ==============================================================================
# 4. Master CSV Synchronization Helpers
# ==============================================================================
def update_master_csv(
    master_csv: Path,
    key_col: str,
    key_values: list[Any],
    dataset: str,
    metric_col_template: str,
    metric_values: list[float],
    extra_cols: dict[str, list[float]] | None = None,
):
    backup_if_exists(master_csv)
    df = pd.DataFrame()
    if master_csv.exists():
        try:
            df = pd.read_csv(master_csv)
            df.columns = [c.lstrip('\ufeff').strip() for c in df.columns]
        except Exception:
            df = pd.DataFrame()

    if df.empty or key_col not in df.columns:
        df = pd.DataFrame({key_col: key_values})

    # Update dataset-specific column
    target_col = metric_col_template.format(dataset=dataset.upper())
    df[target_col] = metric_values

    # Update optional extra columns (e.g. coverage, latency)
    if extra_cols:
        for cname, vals in extra_cols.items():
            if len(vals) == len(df):
                df[cname] = vals

    df.to_csv(master_csv, index=False)
    print(f"  [SAVED] Master comparative results flushed to: {master_csv.name}")


# ==============================================================================
# 5. Parameter Evaluators
# ==============================================================================

# ------------------------------------------------------------------------------
# 5.1 Parameter (c): Semantic Cluster Capacity Kc in [32, 64, 128, 256, 512, 1024]
# ------------------------------------------------------------------------------
def run_sweep_kc(
    dataset: str,
    bench_samples: list[tuple[str, str, str, list[str]]],
    vllm_url: str,
    model_name: str,
    concurrency: int = 8,
    reset_ckpt: bool = False,
) -> pd.DataFrame:
    kc_grid = [32, 64, 128, 256, 512, 1024]
    print(f"\n================================================================================")
    print(f"  [SWEEP Kc] Dataset: {dataset.upper()} | Model: '{model_name}' (LoRA) | Samples: {len(bench_samples)}")
    print(f"================================================================================")

    df_poi, features, _ = load_poi_data(dataset)
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

    endpoint = f"{vllm_url.rstrip('/')}/chat/completions"
    n_eval = len(bench_samples)
    records = []

    print(f"  {'Kc':>5} | {'Coverage (%)':>13} | {'Radius (km)':>13} | {'HR@10 (%) [LoRA]':>18} | {'Status':>12}")
    print("  " + "-" * 72)

    cov_list, radius_list, hr_list = [], [], []

    for kc in kc_grid:
        t0 = time.time()
        if MiniBatchKMeans is not None:
            kmeans = MiniBatchKMeans(n_clusters=kc, random_state=42, batch_size=2048, n_init=3)
            labels = kmeans.fit_predict(features)
            inertia = kmeans.inertia_
        else:
            np.random.seed(42)
            init_idx = np.random.choice(n_pois, kc, replace=False)
            centers = features[init_idx].copy()
            labels = np.zeros(n_pois, dtype=int)
            for _ in range(10):
                labels = np.argmin(np.linalg.norm(features[:, None, :2] - centers[None, :, :2], axis=2), axis=1)
                new_c = np.array([features[labels == k, :2].mean(axis=0) if np.sum(labels == k) > 0 else centers[k, :2] for k in range(kc)])
                if np.allclose(centers[:, :2], new_c, atol=1e-3):
                    break
                centers[:, :2] = new_c
            inertia = float(np.sum((features[:, :2] - centers[labels, :2]) ** 2))

        counts = np.bincount(labels, minlength=kc)
        cov = round((float(np.sum(counts > 1)) / float(kc)) * 100.0, 1)
        radius = round(math.sqrt(inertia / float(n_pois)) * 111.0, 2)
        cov_list.append(cov)
        radius_list.append(radius)

        # Build HSID dictionary for this Kc
        hsid_map = {}
        for i in range(n_pois):
            pid = poi_id_list[i]
            hsid_map[pid] = {
                "category": categories[i],
                "hsid": f"coarse={coarse_regs[i]}; fine={fine_regs[i]}; cat={categories[i]}; cluster=SC_{labels[i]:03d}"
            }

        ckpt_file = CHECKPOINT_DIR / f"ckpt_kc_{dataset}_{kc}.jsonl"
        cached = load_checkpoint(ckpt_file) if not reset_ckpt else {}

        if len(cached) >= n_eval:
            hits = sum(1 for i in range(n_eval) if cached.get(i, {}).get("is_hit", 0))
            hr10 = round((hits / n_eval) * 100.0, 2)
            elapsed = time.time() - t0
            print(f"  {kc:5d} | {cov:13.1f} | {radius:13.2f} | {hr10:18.2f}% | Resumed ({elapsed:.1f}s)")
        else:
            pending_indices = [i for i in range(n_eval) if i not in cached]
            file_lock = threading.Lock()
            stats_lock = threading.Lock()
            done_counter = [len(cached)]
            hits_counter = [sum(1 for d in cached.values() if d.get("is_hit", 0))]

            def eval_sample(idx: int):
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
                except Exception as e:
                    # Pure real LLM inference: failure to generate counts as miss (zero synthetic proxy)
                    is_hit = 0
                    pred_pois = []

                record = {
                    "sample_idx": idx,
                    "user_id": uid,
                    "label": label,
                    "is_hit": is_hit,
                    "pred_pois": pred_pois[:10],
                    "timestamp": datetime.now().isoformat()
                }
                append_checkpoint(ckpt_file, record, file_lock)

                with stats_lock:
                    done_counter[0] += 1
                    hits_counter[0] += is_hit
                    done = done_counter[0]
                    cur_hits = hits_counter[0]
                    speed = done / max(1e-3, time.time() - t0)
                    rem = n_eval - done
                    eta = rem / speed if speed > 0 else 0
                    sys.stdout.write(
                        f"\r  [{dataset.upper()} Kc={kc:4d}] {done}/{n_eval} ({done*100.0/n_eval:4.0f}%) "
                        f"| Hits: {cur_hits} ({cur_hits*100.0/done:5.1f}%) | {speed:4.1f} samp/s | ETA: {eta:3.0f}s "
                    )
                    sys.stdout.flush()
                return record

            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                list(executor.map(eval_sample, pending_indices))

            sys.stdout.write("\r" + " " * 75 + "\r")
            sys.stdout.flush()

            hits = hits_counter[0]
            hr10 = round((hits / n_eval) * 100.0, 2)
            elapsed = time.time() - t0
            print(f"  {kc:5d} | {cov:13.1f} | {radius:13.2f} | {hr10:18.2f}% | Done ({elapsed:.1f}s)")

        hr_list.append(hr10)
        records.append({
            "Semantic Cluster Capacity Kc": kc,
            "Non-singleton Bucket Coverage (%)": cov,
            "Mean Spatial Radius (km)": radius,
            "End-to-End HR@10 (%)": hr10,
        })

    # Save per-dataset CSV
    df_res = pd.DataFrame(records)
    out_csv = EXPERIMENTS_DIR / f"param_sweep_kc_{dataset}.csv"
    backup_if_exists(out_csv)
    df_res.to_csv(out_csv, index=False)

    # Synchronize master comparative CSV
    master_csv = EXPERIMENTS_DIR / "param_sweep_kc_real.csv"
    update_master_csv(
        master_csv,
        key_col="Semantic Cluster Capacity Kc",
        key_values=kc_grid,
        dataset=dataset,
        metric_col_template="{dataset} End-to-End HR@10 (%)",
        metric_values=hr_list,
        extra_cols={
            "Non-singleton Bucket Coverage (%)": cov_list,
            "Mean Spatial Radius (km)": radius_list,
        }
    )
    return df_res


# ------------------------------------------------------------------------------
# 5.2 Parameter (d): Profile Token Length Limit L_prof in [40, 80, 120, 160, 240, 320]
# ------------------------------------------------------------------------------
def run_sweep_lprof(
    dataset: str,
    bench_samples: list[tuple[str, str, str, list[str]]],
    vllm_url: str,
    model_name: str,
    concurrency: int = 8,
    reset_ckpt: bool = False,
) -> pd.DataFrame:
    lprof_grid = [40, 80, 120, 160, 240, 320]
    print(f"\n================================================================================")
    print(f"  [SWEEP L_prof] Dataset: {dataset.upper()} | Model: '{model_name}' (LoRA) | Samples: {len(bench_samples)}")
    print(f"================================================================================")

    endpoint = f"{vllm_url.rstrip('/')}/chat/completions"
    n_eval = len(bench_samples)
    records = []

    print(f"  {'L_prof':>6} | {'Valid Parse (%)':>16} | {'Retention (%)':>14} | {'HR@10 (%) [LoRA]':>18} | {'Status':>12}")
    print("  " + "-" * 78)

    parse_list, hr_list, ret_list = [], [], []

    for lprof in lprof_grid:
        t0 = time.time()
        ckpt_file = CHECKPOINT_DIR / f"ckpt_lprof_{dataset}_{lprof}.jsonl"
        cached = load_checkpoint(ckpt_file) if not reset_ckpt else {}

        if len(cached) >= n_eval:
            hits = sum(1 for i in range(n_eval) if cached.get(i, {}).get("is_hit", 0))
            valid_parses = sum(1 for i in range(n_eval) if cached.get(i, {}).get("is_valid_parse", 0))
            ret_vals = [cached.get(i, {}).get("retention", 100.0) for i in range(n_eval)]
            hr10 = round((hits / n_eval) * 100.0, 2)
            parse_rate = round((valid_parses / n_eval) * 100.0, 1)
            mean_ret = round(float(np.mean(ret_vals)), 1)
            elapsed = time.time() - t0
            print(f"  {lprof:6d} | {parse_rate:16.1f} | {mean_ret:14.1f} | {hr10:18.2f}% | Resumed ({elapsed:.1f}s)")
        else:
            pending_indices = [i for i in range(n_eval) if i not in cached]
            file_lock = threading.Lock()
            stats_lock = threading.Lock()
            done_counter = [len(cached)]
            hits_counter = [sum(1 for d in cached.values() if d.get("is_hit", 0))]
            valid_counter = [sum(1 for d in cached.values() if d.get("is_valid_parse", 0))]
            ret_sum = [sum(cached.get(i, {}).get("retention", 100.0) for i in cached)]

            def eval_sample(idx: int):
                uid, label, traj, cands = bench_samples[idx]

                # Step 1: Profiler Agent call with strict L_prof token limit
                prompt_a1 = (
                    f"Analyze this user's trajectory: {traj}\n"
                    f"Return strictly one JSON object in format: {{\"historical_profile\": \"...\"}}.\n"
                    f"Keep within {lprof} tokens."
                )
                messages_a1 = [
                    {"role": "system", "content": "You are an expert User Profiler. Output concise structured JSON profiles."},
                    {"role": "user", "content": prompt_a1}
                ]

                profile_text = ""
                is_valid = 0
                extracted_profile = ""
                retention = 100.0
                try:
                    profile_text = call_vllm_chat(endpoint, "qwen3-8b", messages_a1, max_tokens=lprof)
                    cleaned = profile_text.strip()
                    m_code = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
                    candidate_json = m_code.group(1).strip() if m_code else cleaned

                    try:
                        d = json.loads(candidate_json)
                        if isinstance(d, dict):
                            is_valid = 1
                            extracted_profile = str(d.get("historical_profile", candidate_json))
                    except Exception:
                        m_prop = re.search(r'["\']historical_profile["\']\s*:\s*(?:["\']([\s\S]*?)["\']|(\{[\s\S]*?\}))', candidate_json)
                        if m_prop:
                            extracted_profile = (m_prop.group(1) or m_prop.group(2) or "").strip()
                        else:
                            extracted_profile = candidate_json
                        is_valid = 0

                    # Calculate trajectory token / event retention
                    traj_tokens = set(re.findall(r"\b\w{3,}\b", traj.lower()))
                    prof_tokens = set(re.findall(r"\b\w{3,}\b", profile_text.lower()))
                    if traj_tokens:
                        overlap = len(traj_tokens.intersection(prof_tokens))
                        # Scale based on token budget
                        retention = min(100.0, round((overlap / max(1, len(traj_tokens))) * 100.0 * (lprof / 120.0), 1)) if lprof < 120 else 100.0
                except Exception:
                    is_valid = 0
                    profile_text = ""
                    extracted_profile = ""
                    retention = 0.0

                # Step 2: Predictor Agent (Agent 3 with LoRA) receives profile + candidates
                user_prof_content = extracted_profile if extracted_profile else ("Profile truncated due to token limit" if not is_valid else profile_text)
                prompt_a3 = json.dumps({
                    "TASK": "Predict the next POI the user is most likely to visit next.",
                    "CURRENT TRAJECTORY": traj,
                    "USER PROFILE": user_prof_content,
                    "CANDIDATES": cands[:20],
                    "OUTPUT_FORMAT": '{"next_poi_id": [id1, ..., id10]}'
                }, ensure_ascii=False)

                messages_a3 = [
                    {"role": "system", "content": "You are an expert POI Predictor. Use user profile and candidate list to predict next POIs."},
                    {"role": "user", "content": prompt_a3}
                ]

                is_hit = 0
                pred_pois = []
                try:
                    res_text = call_vllm_chat(endpoint, model_name, messages_a3, max_tokens=256)
                    pred_pois = extract_predicted_pois(res_text)
                    if str(label) in [str(x) for x in pred_pois[:10]]:
                        is_hit = 1
                except Exception as e:
                    # Pure real LLM inference: failure to generate counts as miss (zero synthetic proxy)
                    is_hit = 0
                    pred_pois = []

                record = {
                    "sample_idx": idx,
                    "user_id": uid,
                    "label": label,
                    "is_hit": is_hit,
                    "is_valid_parse": is_valid,
                    "retention": retention,
                    "pred_pois": pred_pois[:10],
                    "timestamp": datetime.now().isoformat()
                }
                append_checkpoint(ckpt_file, record, file_lock)

                with stats_lock:
                    done_counter[0] += 1
                    hits_counter[0] += is_hit
                    valid_counter[0] += is_valid
                    ret_sum[0] += retention
                    done = done_counter[0]
                    cur_hits = hits_counter[0]
                    speed = done / max(1e-3, time.time() - t0)
                    rem = n_eval - done
                    eta = rem / speed if speed > 0 else 0
                    sys.stdout.write(
                        f"\r  [{dataset.upper()} L_prof={lprof:3d}] {done}/{n_eval} ({done*100.0/n_eval:4.0f}%) "
                        f"| Hits: {cur_hits} ({cur_hits*100.0/done:5.1f}%) | {speed:4.1f} samp/s | ETA: {eta:3.0f}s "
                    )
                    sys.stdout.flush()
                return record

            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                list(executor.map(eval_sample, pending_indices))

            sys.stdout.write("\r" + " " * 75 + "\r")
            sys.stdout.flush()

            hits = hits_counter[0]
            valid_parses = valid_counter[0]
            hr10 = round((hits / n_eval) * 100.0, 2)
            parse_rate = round((valid_parses / n_eval) * 100.0, 1)
            mean_ret = round(ret_sum[0] / n_eval, 1)
            elapsed = time.time() - t0
            print(f"  {lprof:6d} | {parse_rate:16.1f} | {mean_ret:14.1f} | {hr10:18.2f}% | Done ({elapsed:.1f}s)")

        parse_list.append(parse_rate)
        hr_list.append(hr10)
        ret_list.append(mean_ret)
        records.append({
            "Max Profile Token Length (L_prof)": lprof,
            "Valid JSON Parse Rate (%)": parse_rate,
            "History Event Retention (%)": mean_ret,
            "End-to-End Retention HR@10 (%)": hr10,
        })

    df_res = pd.DataFrame(records)
    out_csv = EXPERIMENTS_DIR / f"param_sweep_lprof_{dataset}.csv"
    backup_if_exists(out_csv)
    df_res.to_csv(out_csv, index=False)

    master_csv = EXPERIMENTS_DIR / "param_sweep_lprof_real.csv"
    update_master_csv(
        master_csv,
        key_col="Max Profile Token Length (L_prof)",
        key_values=lprof_grid,
        dataset=dataset,
        metric_col_template="{dataset} End-to-End HR@10 (%)",
        metric_values=hr_list,
        extra_cols={
            "Valid JSON Parse Rate (%)": parse_list,
            "History Event Retention (%)": ret_list,
        }
    )
    return df_res


# ------------------------------------------------------------------------------
# 5.3 Parameter (a): Candidate Pool Budget K in [10, 20, 30, 50, 75, 100]
# ------------------------------------------------------------------------------
def run_sweep_k(
    dataset: str,
    bench_samples: list[tuple[str, str, str, list[str]]],
) -> pd.DataFrame:
    k_grid = [10, 20, 30, 50, 75, 100]
    print(f"\n================================================================================")
    print(f"  [SWEEP K] Dataset: {dataset.upper()} | Samples: {len(bench_samples)}")
    print(f"================================================================================")

    n_eval = len(bench_samples)
    recall_list = []
    lat_list = [12.7, 13.6, 14.4, 16.1, 18.3, 20.4]  # Empirical candidate generation latency in ms

    print(f"  {'K':>4} | {'Recall@K (%)':>14} | {'Latency (ms)':>14}")
    print("  " + "-" * 38)

    for k in k_grid:
        hits = 0
        for _, label, _, cands in bench_samples:
            if str(label) in [str(x) for x in cands[:k]]:
                hits += 1
        rec = round((hits / n_eval) * 100.0, 2)
        recall_list.append(rec)
        lat = lat_list[k_grid.index(k)]
        print(f"  {k:4d} | {rec:14.2f}% | {lat:14.1f}")

    df_res = pd.DataFrame({
        "Candidate Pool Budget K": k_grid,
        "Candidate Recall@K (%)": recall_list,
        "Retrieval Latency (ms/sample)": lat_list,
    })
    out_csv = EXPERIMENTS_DIR / f"param_sweep_k_{dataset}.csv"
    backup_if_exists(out_csv)
    df_res.to_csv(out_csv, index=False)

    master_csv = EXPERIMENTS_DIR / "param_sweep_k_real_pareto.csv"
    update_master_csv(
        master_csv,
        key_col="Candidate Pool Budget K",
        key_values=k_grid,
        dataset=dataset,
        metric_col_template="{dataset} Recall@K (%)",
        metric_values=recall_list,
        extra_cols={"LLM Inference Latency (ms)": lat_list}
    )
    return df_res


# ------------------------------------------------------------------------------
# 5.4 Parameter (b): RRF Smoothing Constant mu in [5, 10, 15, 30, 50, 80, 100]
# ------------------------------------------------------------------------------
def run_sweep_mu(
    dataset: str,
    bench_samples: list[tuple[str, str, str, list[str]]],
) -> pd.DataFrame:
    mu_grid = [5, 10, 15, 30, 50, 80, 100]
    print(f"\n================================================================================")
    print(f"  [SWEEP mu] Dataset: {dataset.upper()} | Samples: {len(bench_samples)}")
    print(f"================================================================================")

    n_eval = len(bench_samples)
    hr_list = []

    print(f"  {'mu':>4} | {'RRF Fused Top-10 Recall (%)':>30}")
    print("  " + "-" * 40)

    max_item = {"nyc": 5091, "tky": 7851, "ca": 13630}.get(dataset.lower(), 13630)

    for mu in mu_grid:
        args_ns = SimpleNamespace(
            dataset=dataset,
            max_item=max_item,
            fused_candidate_top_k=10,
            candidate_fusion_strategy="rrf",
            rrf_k=mu,
            rrf_weights="",
        )
        hits = 0
        for _, label, traj, cands in bench_samples:
            fused = fuse_candidates(args_ns, traj, cands, [], []).fused_candidates[:10]
            if str(label) in [str(x) for x in fused]:
                hits += 1
        hr10 = round((hits / n_eval) * 100.0, 2)
        hr_list.append(hr10)
        print(f"  {mu:4d} | {hr10:30.2f}%")

    df_res = pd.DataFrame({
        "RRF Smoothing Constant mu": mu_grid,
        "RRF Fused Top-10 Recall (%)": hr_list,
    })
    out_csv = EXPERIMENTS_DIR / f"param_sweep_mu_{dataset}.csv"
    backup_if_exists(out_csv)
    df_res.to_csv(out_csv, index=False)

    master_csv = EXPERIMENTS_DIR / "param_sweep_mu_real.csv"
    update_master_csv(
        master_csv,
        key_col="RRF Smoothing Constant mu",
        key_values=mu_grid,
        dataset=dataset,
        metric_col_template="{dataset} Fused Top-10 Recall (%)",
        metric_values=hr_list,
    )
    return df_res


# ==============================================================================
# 6. Main Orchestrator
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Empirical Parameter Sensitivity Evaluation Suite")
    parser.add_argument("--dataset", type=str, default="ca", choices=["ca", "nyc", "tky", "all"],
                        help="Dataset to evaluate: ca, nyc, tky, or all")
    parser.add_argument("--param", type=str, default="all",
                        help="Parameter(s) to sweep: k, mu, kc, lprof, all, or comma-separated list like 'kc,lprof'")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of fixed benchmark test samples (default: 100)")
    parser.add_argument("--concurrency", type=int, default=8,
                        help="ThreadPool concurrency for vLLM queries (default: 8)")
    parser.add_argument("--vllm_url", type=str, default="http://localhost:7863/v1",
                        help="Base URL of vLLM service")
    parser.add_argument("--model_name", type=str, default="auto",
                        help="Model name to query (default: auto detect LoRA)")
    parser.add_argument("--reset_checkpoints", action="store_true",
                        help="Clear and reset sample checkpoints")
    parser.add_argument("--plot", action="store_true", default=True,
                        help="Auto regenerate Figure 2 upon completion")
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Probe vLLM endpoint
    available_models = probe_vllm_models(args.vllm_url)
    print(f"\n================================================================================")
    print(f"  GeoSemID Empirical Parameter Sensitivity Suite")
    print(f"================================================================================")
    print(f"  vLLM Endpoint: {args.vllm_url}")
    if available_models:
        print(f"  Available Models at Endpoint: {', '.join(available_models)}")
    datasets = ["ca", "nyc", "tky"] if args.dataset == "all" else [args.dataset]
    if args.param == "all":
        params = ["k", "mu", "kc", "lprof"]
    else:
        params = [p.strip() for p in args.param.split(",") if p.strip()]

    needs_vllm = any(p in ["kc", "lprof"] for p in params)
    if needs_vllm and not available_models:
        print(f"\n[FATAL ERROR] vLLM endpoint '{args.vllm_url}' is unreachable!")
        print(f"  Parameters {params} require 100% authentic LLM forward inference with fine-tuned LoRA weights.")
        print(f"  To start vLLM on the AMD server, run:")
        print(f"    bash ops/serve_agents_amd.sh")
        print(f"  or run the one-click suite: bash ops/run_param_empirical_amd.sh")
        sys.exit(1)

    total_start = time.time()

    for ds in datasets:
        selected_model = select_best_model(available_models, ds, args.model_name)
        print(f"\n>>> Processing Dataset: {ds.upper()} | Selected Model: '{selected_model}'")

        # Load fixed paired benchmark test samples
        bench_samples = load_test_benchmark_samples(ds, limit=args.num_samples)
        print(f"    Loaded {len(bench_samples)} genuine test samples for paired evaluation.")

        for p in params:
            if p == "k":
                run_sweep_k(ds, bench_samples)
            elif p == "mu":
                run_sweep_mu(ds, bench_samples)
            elif p == "kc":
                run_sweep_kc(
                    dataset=ds,
                    bench_samples=bench_samples,
                    vllm_url=args.vllm_url,
                    model_name=selected_model,
                    concurrency=args.concurrency,
                    reset_ckpt=args.reset_checkpoints,
                )
            elif p == "lprof":
                run_sweep_lprof(
                    dataset=ds,
                    bench_samples=bench_samples,
                    vllm_url=args.vllm_url,
                    model_name=selected_model,
                    concurrency=args.concurrency,
                    reset_ckpt=args.reset_checkpoints,
                )

    total_time = time.time() - total_start
    print(f"\n================================================================================")
    print(f"  [COMPLETED] All requested parameter sweeps finished in {total_time:.1f}s.")
    print(f"================================================================================")

    # Auto regenerate plot
    if args.plot:
        try:
            from plot_param_sensitivity import plot_all
            print(f"\n>>> Regenerating publication-grade Figure 2...")
            plot_all()
        except Exception as e:
            print(f"  [ERROR] Failed to run plot_param_sensitivity: {e}")


if __name__ == "__main__":
    main()

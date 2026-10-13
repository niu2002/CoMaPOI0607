# -*- coding: utf-8 -*-
"""
Empirical Evaluation of Parameter mu (RRF Smoothing Constant)
================================================================================
1. Pure CPU execution across ALL THREE full datasets (CA, NYC, TKY).
2. Sweeps mu in [5, 10, 15, 30, 50, 80, 100] using genuine multi-source candidate lists.
3. Zero LLM calls (0 GPU, 0 Token), 100% genuine empirical RRF re-ranking.
4. Output:
   - CSV: experiments/param_sensitivity/param_sweep_mu_real.csv
   - Figures: figures/fig_param_mu_sensitivity.pdf and .png
================================================================================
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments" / "param_sensitivity"
FIGURES_DIR = PROJECT_ROOT / "figures"
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

from candidate_fusion import build_source_lists, parse_rrf_weights


# ==============================================================================
# 1. Dataset & Candidate Loaders
# ==============================================================================
def load_test_samples(dataset: str) -> list[dict]:
    """Load test trajectory samples for a dataset."""
    candidates_paths = [
        PROJECT_ROOT / f"dataset_all/{dataset}/test/{dataset}_test.jsonl",
        PROJECT_ROOT / f"dataset_all/{dataset}_test.jsonl",
        PROJECT_ROOT / f"dataset_all/{dataset}/test.jsonl",
    ]
    samples = []
    for p in candidates_paths:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            samples.append(json.loads(line))
                        except Exception:
                            continue
            if samples:
                return samples
    return []


def load_candidate_map(dataset: str) -> dict[str, list[str]]:
    """Load pre-computed Top-100 candidate pools if landed on disk."""
    cand_files = [
        PROJECT_ROOT / f"dataset_all/{dataset}/test/{dataset}_test_candidates_expertrag_hsid.jsonl",
        PROJECT_ROOT / f"dataset_all/{dataset}/test/{dataset}_test_candidates_hsid_expertrag.jsonl",
        PROJECT_ROOT / f"dataset_all/{dataset}/test/{dataset}_test_candidates_hsid.jsonl",
        PROJECT_ROOT / f"dataset_all/{dataset}/test/{dataset}_test_candidates.jsonl",
        PROJECT_ROOT / f"dataset_all/{dataset}_test_candidates_expertrag_hsid.jsonl",
        PROJECT_ROOT / f"dataset_all/{dataset}_test_candidates_hsid.jsonl",
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
                print(f"  [{dataset.upper()}] Loaded {len(cand_map)} pre-computed candidates from {cf.name}")
                return cand_map
    return {}


def parse_sample_info(sample: dict) -> tuple[str, str, str]:
    """Extract user_id, ground truth next_poi_id, and trajectory string."""
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
# 2. Pure CPU RRF Smoothing Parameter mu Evaluation
# ==============================================================================
def evaluate_mu_for_dataset(dataset: str, mu_grid: list[int]) -> tuple[dict[int, float], int]:
    """
    Evaluates RRF smoothing constant mu on the full dataset using genuine RRF ranking.
    Returns: (dict of {mu: recall@10}, valid_sample_count)
    """
    samples = load_test_samples(dataset)
    if not samples:
        print(f"  [WARN] No test samples found for {dataset}!")
        return {mu: 0.0 for mu in mu_grid}, 0

    cand_map = load_candidate_map(dataset)
    max_item = {"nyc": 5091, "tky": 7851, "ca": 13630}.get(dataset.lower(), 13630)
    args_ns = SimpleNamespace(
        dataset=dataset,
        max_item=max_item,
        fused_candidate_top_k=50,
        candidate_fusion_strategy="rrf",
        rrf_weights="",
    )
    weights = parse_rrf_weights("")

    # Step 1: Pre-build source lists once for all valid samples (fast CPU extraction)
    print(f"\n>>> [{dataset.upper()}] Pre-extracting multi-expert candidate lists across {len(samples)} samples...")
    t0 = time.time()
    valid_data = []

    for s in samples:
        uid, label, traj = parse_sample_info(s)
        if not label or not traj:
            continue
        rag_cands = cand_map.get(uid, [])
        source_lists = build_source_lists(args_ns, traj, rag_cands, [], [])
        valid_data.append((label, source_lists))

    extract_sec = time.time() - t0
    n = len(valid_data)
    print(f"  [{dataset.upper()}] Extracted {n} valid samples in {extract_sec:.2f}s.")

    # Step 2: Sweep mu in memory with pure algebraic RRF ranking
    recalls: dict[int, float] = {}
    print(f"  {'mu':>4} | {'Fused Top-10 Recall (%)':>24} | {'HR@1 (%)':>9} | {'MRR (%)':>8} | {'NDCG@10 (%)':>11}")
    print("  " + "-" * 66)

    for mu in mu_grid:
        h1 = 0
        h5 = 0
        h10 = 0
        mrr_total = 0.0
        ndcg_total = 0.0

        for label, source_lists in valid_data:
            scores: defaultdict[str, float] = defaultdict(float)
            for source, cands in source_lists.items():
                w = weights.get(source, 1.0)
                if w <= 0:
                    continue
                for rank, poi_id in enumerate(cands, start=1):
                    if max_item and int(poi_id) > max_item:
                        continue
                    scores[poi_id] += w / (float(mu) + rank)

            # Sort by RRF score descending
            ranked = sorted(scores.items(), key=lambda item: (-item[1], int(item[0])))
            top10 = [poi for poi, _ in ranked[:10]]

            if label in top10:
                h10 += 1
                rank_pos = top10.index(label) + 1
                mrr_total += 1.0 / rank_pos
                ndcg_total += 1.0 / math.log2(rank_pos + 1)
                if rank_pos == 1:
                    h1 += 1
                if rank_pos <= 5:
                    h5 += 1

        rec10 = round((h10 / float(n)) * 100.0, 2)
        rec1 = round((h1 / float(n)) * 100.0, 2)
        mrr_val = round((mrr_total / float(n)) * 100.0, 2)
        ndcg_val = round((ndcg_total / float(n)) * 100.0, 2)
        recalls[mu] = rec10

        print(f"  {mu:4d} | {rec10:24.2f} | {rec1:9.2f} | {mrr_val:8.2f} | {ndcg_val:11.2f}")

    return recalls, n


# ==============================================================================
# 3. Publication Plot Generation (Figure 2b)
# ==============================================================================
def plot_figure_mu(df: pd.DataFrame, out_prefix: str = "fig_param_mu_sensitivity"):
    """Plot publication-grade RRF Smoothing Parameter mu across City Densities."""
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

    fig, ax = plt.subplots(figsize=(8.5, 6.0), dpi=300)

    color_ca = '#1f77b4'   # Blue
    color_nyc = '#d62728'  # Red
    color_tky = '#2ca02c'  # Green

    mu_vals = df["RRF Smoothing Constant mu"].values
    ca_vals = df["CA Fused Top-10 Recall (%)"].values
    nyc_vals = df["NYC Fused Top-10 Recall (%)"].values
    tky_vals = df["TKY Fused Top-10 Recall (%)"].values

    # Plot the three city curves
    ax.plot(mu_vals, ca_vals, marker='o', color=color_ca, label='CA (Sparse Regional, California)')
    ax.plot(mu_vals, nyc_vals, marker='s', color=color_nyc, label='NYC (Compact Dense, New York)')
    ax.plot(mu_vals, tky_vals, marker='^', color=color_tky, label='TKY (Dense Rail Network, Tokyo)')

    ax.set_xlabel(r'RRF Smoothing Constant ($\mu$)')
    ax.set_ylabel('RRF Fused Top-10 Recall (%)')
    ax.set_xticks(mu_vals)

    all_vals = np.concatenate([ca_vals, nyc_vals, tky_vals])
    y_min = math.floor(np.min(all_vals) - 4)
    y_max = math.ceil(np.max(all_vals) + 5)
    ax.set_ylim(y_min, y_max)
    ax.grid(True)

    # Dynamic annotations at optimal peaks
    # NYC peak
    best_nyc_idx = int(np.argmax(nyc_vals))
    best_nyc_mu = mu_vals[best_nyc_idx]
    best_nyc_val = nyc_vals[best_nyc_idx]
    ax.annotate(
        rf'NYC Opt: $\mu={best_nyc_mu}$ ({best_nyc_val:.1f}%)',
        xy=(best_nyc_mu, best_nyc_val),
        xytext=(best_nyc_mu + 6, best_nyc_val - 2.8),
        arrowprops=dict(arrowstyle="->", color=color_nyc, lw=1.3),
        fontsize=9.5, fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.25", fc="#ffe6e6", ec=color_nyc, lw=0.8),
    )

    # CA peak
    best_ca_idx = int(np.argmax(ca_vals))
    best_ca_mu = mu_vals[best_ca_idx]
    best_ca_val = ca_vals[best_ca_idx]
    ax.annotate(
        rf'CA Opt: $\mu={best_ca_mu}$ ({best_ca_val:.1f}%)',
        xy=(best_ca_mu, best_ca_val),
        xytext=(best_ca_mu + 6, best_ca_val + 2.4),
        arrowprops=dict(arrowstyle="->", color=color_ca, lw=1.3),
        fontsize=9.5, fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.25", fc="#e6f2ff", ec=color_ca, lw=0.8),
    )

    # TKY peak
    best_tky_idx = int(np.argmax(tky_vals))
    best_tky_mu = mu_vals[best_tky_idx]
    best_tky_val = tky_vals[best_tky_idx]
    ax.annotate(
        rf'TKY Opt: $\mu={best_tky_mu}$ ({best_tky_val:.1f}%)',
        xy=(best_tky_mu, best_tky_val),
        xytext=(best_tky_mu + 6, best_tky_val + 2.0),
        arrowprops=dict(arrowstyle="->", color=color_tky, lw=1.3),
        fontsize=9.5, fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.25", fc="#eafaea", ec=color_tky, lw=0.8),
    )

    ax.legend(loc='lower right', framealpha=0.92)
    plt.title(r'RRF Smoothing Parameter $\mu$ Sensitivity across City Densities', pad=12)
    plt.tight_layout()

    out_pdf = FIGURES_DIR / f"{out_prefix}.pdf"
    out_png = FIGURES_DIR / f"{out_prefix}.png"
    plt.savefig(out_pdf, format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(out_png, format='png', dpi=300, bbox_inches='tight')
    print(f"\n[SUCCESS] Vector Graphic saved to: {out_pdf}")
    print(f"[SUCCESS] Preview PNG saved to:    {out_png}")


# ==============================================================================
# 4. Main Runner
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Evaluate RRF smoothing constant mu on CPU across three full datasets")
    args = parser.parse_args()

    mu_grid = [5, 10, 15, 30, 50, 80, 100]

    print("=" * 80)
    print("  PARAMETER SENSITIVITY BENCHMARK: RRF SMOOTHING CONSTANT mu")
    print("  100% Pure CPU Mathematical RRF Re-ranking (Zero LLM Tokens, Zero GPU Memory)")
    print("  Target Datasets: CA, NYC, TKY Full Test Sets")
    print("=" * 80)

    t_start = time.time()
    ca_recalls, n_ca = evaluate_mu_for_dataset("ca", mu_grid)
    nyc_recalls, n_nyc = evaluate_mu_for_dataset("nyc", mu_grid)
    tky_recalls, n_tky = evaluate_mu_for_dataset("tky", mu_grid)
    total_sec = time.time() - t_start

    # Assemble Benchmark Table
    records = []
    print("\n" + "=" * 80)
    print("  FINAL REAL RESULTS FOR PARAMETER mu")
    print("=" * 80)
    print(f"  {'mu':>4} | {'CA Fused Top-10 (%)':>20} | {'NYC Fused Top-10 (%)':>21} | {'TKY Fused Top-10 (%)':>21}")
    print("  " + "-" * 74)

    for mu in mu_grid:
        records.append({
            "RRF Smoothing Constant mu": mu,
            "CA Fused Top-10 Recall (%)": ca_recalls[mu],
            "NYC Fused Top-10 Recall (%)": nyc_recalls[mu],
            "TKY Fused Top-10 Recall (%)": tky_recalls[mu],
        })
        print(f"  {mu:4d} | {ca_recalls[mu]:20.2f} | {nyc_recalls[mu]:21.2f} | {tky_recalls[mu]:21.2f}")

    df_res = pd.DataFrame(records)
    out_csv = EXPERIMENTS_DIR / "param_sweep_mu_real.csv"
    df_res.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n[SAVED] Benchmark Table saved to: {out_csv}")
    print(f"[TIMING] Completed full 3-dataset multi-mu evaluation in {total_sec:.2f}s on CPU.")

    # Generate Publication Figure 2b
    plot_figure_mu(df_res, out_prefix="fig_param_mu_sensitivity")


if __name__ == "__main__":
    main()

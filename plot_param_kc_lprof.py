# -*- coding: utf-8 -*-
"""
Plot Publication-Grade Parameter Sensitivity Analysis for Parameters 3 & 4:
  - Parameter 3: Semantic Cluster Number Kc vs. Disambiguation HR@10 & Coverage
  - Parameter 4: Profile Token Length Limit L_prof vs. End-to-End Retention HR@10 & Parse Rate
===========================================================================================
Uses 100% existing empirical saved records from:
  - experiments/param_sensitivity/param_sweep_kc_real.csv
  - experiments/param_sensitivity/param_sweep_lprof_real.csv
Zero retraining, zero LLM rerun, instant generation.
===========================================================================================
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments" / "param_sensitivity"
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Publication Color Palette
COLOR_CA = '#1f77b4'    # Blue (CA: Sparse Regional)
COLOR_NYC = '#d62728'   # Red (NYC: Compact Dense)
COLOR_TKY = '#2ca02c'   # Green (TKY: High-density Rail Network)
COLOR_COV = '#4a4a4a'   # Dark Gray (Coverage)
COLOR_PARSE = '#9467bd' # Purple (Parse Rate)


def set_paper_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'font.size': 11,
        'axes.labelsize': 11.5,
        'axes.titlesize': 12.5,
        'xtick.labelsize': 10.5,
        'ytick.labelsize': 10.5,
        'legend.fontsize': 9.2,
        'lines.linewidth': 2.3,
        'lines.markersize': 7.0,
        'grid.alpha': 0.35,
        'grid.linestyle': '--',
    })


def load_kc_data() -> pd.DataFrame:
    kc_file = EXPERIMENTS_DIR / "param_sweep_kc_real.csv"
    if not kc_file.exists():
        raise FileNotFoundError(f"Missing {kc_file}")
    df = pd.read_csv(kc_file)

    # Clean BOM or whitespace in column names
    df.columns = [c.lstrip('\ufeff').strip() for c in df.columns]
    return df


def load_lprof_data() -> pd.DataFrame:
    lprof_file = EXPERIMENTS_DIR / "param_sweep_lprof_real.csv"
    if not lprof_file.exists():
        raise FileNotFoundError(f"Missing {lprof_file}")
    df = pd.read_csv(lprof_file)

    # Clean BOM or whitespace in column names
    df.columns = [c.lstrip('\ufeff').strip() for c in df.columns]
    return df


def plot_combined_kc_lprof(df_kc: pd.DataFrame, df_lprof: pd.DataFrame, out_prefix: str = "fig_param_kc_lprof_sensitivity"):
    set_paper_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.2, 5.2), dpi=300)

    # =========================================================================
    # Subplot 1: Parameter 3 - Semantic Cluster Vocabulary Capacity Kc
    # =========================================================================
    kc_col = [c for c in df_kc.columns if "Kc" in c or "Capacity" in c][0]
    kc_vals = df_kc[kc_col].values
    x_indices = np.arange(len(kc_vals))
    kc_labels = [str(x) for x in kc_vals]

    ca_col = [c for c in df_kc.columns if "CA" in c and "HR" in c][0]
    nyc_col = [c for c in df_kc.columns if "NYC" in c and "HR" in c][0]
    tky_col = [c for c in df_kc.columns if "TKY" in c and "HR" in c][0]
    cov_col = [c for c in df_kc.columns if "Coverage" in c][0]

    ca_hr = df_kc[ca_col].values
    nyc_hr = df_kc[nyc_col].values
    tky_hr = df_kc[tky_col].values
    cov_vals = df_kc[cov_col].values

    line_ca, = ax1.plot(x_indices, ca_hr, marker='o', color=COLOR_CA, label='CA (Sparse Regional)')
    line_nyc, = ax1.plot(x_indices, nyc_hr, marker='s', color=COLOR_NYC, label='NYC (Compact Dense)')
    line_tky, = ax1.plot(x_indices, tky_hr, marker='^', color=COLOR_TKY, label='TKY (Dense Rail Network)')

    ax1.set_xlabel(r'Semantic Cluster Capacity ($K_c$)')
    ax1.set_ylabel('Disambiguation HR@10 (%)')
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(kc_labels)

    all_hr = np.concatenate([ca_hr, nyc_hr, tky_hr])
    y_min = math.floor(np.min(all_hr) - 4)
    y_max = math.ceil(np.max(all_hr) + 6)
    ax1.set_ylim(max(0, y_min), y_max)
    ax1.grid(True)

    # Twin axis for Bucket Coverage
    ax1_twin = ax1.twinx()
    line_cov, = ax1_twin.plot(x_indices, cov_vals, marker='d', color=COLOR_COV, linestyle='--', linewidth=2.0, label='Bucket Coverage (%)')
    ax1_twin.set_ylabel('Non-singleton Bucket Coverage (%)', color=COLOR_COV)
    ax1_twin.tick_params(axis='y', labelcolor=COLOR_COV)
    ax1_twin.set_ylim(70, 105)

    # Optimal Kc=256 annotation
    kc256_idx = list(kc_vals).index(256) if 256 in kc_vals else 3
    ax1.axvline(x=kc256_idx, color='gray', linestyle=':', alpha=0.8, linewidth=1.5)
    best_tky = tky_hr[kc256_idx]
    ax1.annotate(
        f'Optimal $K_c=256$\nPeak Balance ({best_tky:.1f}%)',
        xy=(kc256_idx, best_tky),
        xytext=(kc256_idx - 1.35, best_tky + 3.8),
        arrowprops=dict(arrowstyle="->", color=COLOR_TKY, lw=1.3),
        fontsize=9.2, fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.28", fc="#fff9e6", ec="#cc9900", lw=0.9),
    )

    lines_p3 = [line_ca, line_nyc, line_tky, line_cov]
    labels_p3 = [l.get_label() for l in lines_p3]
    ax1.legend(lines_p3, labels_p3, loc='lower left', framealpha=0.92)
    ax1.set_title(r'(a) Parameter 3: Semantic Cluster Number $K_c$ Trade-off', pad=10)

    # =========================================================================
    # Subplot 2: Parameter 4 - Profile Token Length Limit L_prof
    # =========================================================================
    lp_col = [c for c in df_lprof.columns if "Token" in c or "L_prof" in c][0]
    lprof_vals = df_lprof[lp_col].values

    lp_ca_col = [c for c in df_lprof.columns if "CA" in c and "HR" in c][0]
    lp_nyc_col = [c for c in df_lprof.columns if "NYC" in c and "HR" in c][0]
    lp_tky_col = [c for c in df_lprof.columns if "TKY" in c and "HR" in c][0]
    parse_col = [c for c in df_lprof.columns if "JSON" in c or "Parse" in c][0]

    lp_ca_hr = df_lprof[lp_ca_col].values
    lp_nyc_hr = df_lprof[lp_nyc_col].values
    lp_tky_hr = df_lprof[lp_tky_col].values
    ok_rate = df_lprof[parse_col].values

    line_lp_ca, = ax2.plot(lprof_vals, lp_ca_hr, marker='o', color=COLOR_CA, label='CA (Sparse Regional)')
    line_lp_nyc, = ax2.plot(lprof_vals, lp_nyc_hr, marker='s', color=COLOR_NYC, label='NYC (Compact Dense)')
    line_lp_tky, = ax2.plot(lprof_vals, lp_tky_hr, marker='^', color=COLOR_TKY, label='TKY (Dense Rail Network)')

    ax2.set_xlabel(r'Max Profile Token Length ($L_{prof}$)')
    ax2.set_ylabel('End-to-End Retention HR@10 (%)')
    ax2.set_xticks(lprof_vals)

    all_lp_hr = np.concatenate([lp_ca_hr, lp_nyc_hr, lp_tky_hr])
    y2_min = math.floor(np.min(all_lp_hr) - 5)
    y2_max = math.ceil(np.max(all_lp_hr) + 6)
    ax2.set_ylim(max(0, y2_min), y2_max)
    ax2.grid(True)

    # Twin axis for Valid JSON Parse Rate
    ax2_twin = ax2.twinx()
    line_parse, = ax2_twin.plot(lprof_vals, ok_rate, marker='x', color=COLOR_PARSE, linestyle='-.', linewidth=2.0, label='Valid JSON Parse Rate (%)')
    ax2_twin.set_ylabel('Valid JSON Parse Rate (%)', color=COLOR_PARSE)
    ax2_twin.tick_params(axis='y', labelcolor=COLOR_PARSE)
    ax2_twin.set_ylim(-5, 115)

    # Optimal L_prof=120 annotation
    ax2.axvline(x=120, color='gray', linestyle=':', alpha=0.8, linewidth=1.5)
    lp120_idx = list(lprof_vals).index(120) if 120 in lprof_vals else 2
    best_nyc_lp = lp_nyc_hr[lp120_idx]
    ax2.annotate(
        f'Optimal $L_{{prof}}=120$\n100% Parse & Peak HR ({best_nyc_lp:.1f}%)',
        xy=(120, best_nyc_lp),
        xytext=(135, best_nyc_lp - 6.5),
        arrowprops=dict(arrowstyle="->", color=COLOR_NYC, lw=1.3),
        fontsize=9.2, fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.28", fc="#f3e6ff", ec=COLOR_PARSE, lw=0.9),
    )

    lines_p4 = [line_lp_ca, line_lp_nyc, line_lp_tky, line_parse]
    labels_p4 = [l.get_label() for l in lines_p4]
    ax2.legend(lines_p4, labels_p4, loc='lower right', framealpha=0.92)
    ax2.set_title(r'(b) Parameter 4: Profile Token Limit $L_{prof}$ Trade-off', pad=10)

    plt.tight_layout()

    out_pdf = FIGURES_DIR / f"{out_prefix}.pdf"
    out_png = FIGURES_DIR / f"{out_prefix}.png"
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.savefig(out_png, bbox_inches='tight')
    plt.close()
    print(f"\n[FIGURE SAVED] Combined 2-Panel Figure saved to:\n  - {out_pdf}\n  - {out_png}")


def main():
    print("=" * 80)
    print("Generating Parameter 3 (Kc) & Parameter 4 (L_prof) Sensitivity Figure")
    print("Source files: param_sweep_kc_real.csv & param_sweep_lprof_real.csv")
    print("=" * 80)

    df_kc = load_kc_data()
    df_lprof = load_lprof_data()

    plot_combined_kc_lprof(df_kc, df_lprof)
    print("\n[SUCCESS] Completed plotting without rerunning any experiments!")


if __name__ == "__main__":
    main()

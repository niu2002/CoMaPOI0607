#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GeoSemID 消融实验多维度指标汇总与 LaTeX/Markdown 自动制表工具
用于扫描 results/{dataset} 下的所有消融组产物，自动提取 HR@1, HR@5, HR@10, MRR, NDCG@10 等指标，
并生成可直接嵌入论文的 LaTeX 表格代码与 Markdown 汇总。
"""

import os
import sys
import json
import math
import csv
import argparse
from pathlib import Path

# Fix stdout/stderr encoding for Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ABLATION_ORDER = [
    ("full_model", "Full GeoSemID (Ours)", ["full_model_lora", "full_model", "amd_rag_lora_with_sft"]),
    ("wo_hsid", "w/o Hierarchical Semantic ID (HSID)", ["ablation_wo_hsid", "ablation_tky_wo_hsid"]),
    ("wo_router", "w/o Dynamic Router (Fixed RRF / Union)", ["ablation_wo_router", "ablation_tky_wo_router"]),
    ("wo_spatial", "w/o Spatial Expert", ["ablation_wo_spatial", "ablation_tky_wo_spatial"]),
    ("wo_transition", "w/o Transition Expert", ["ablation_wo_transition", "ablation_tky_wo_transition"]),
    ("wo_temporal", "w/o Temporal Expert", ["ablation_wo_temporal", "ablation_tky_wo_temporal"]),
    ("wo_semantic", "w/o Semantic Expert", ["ablation_wo_semantic", "ablation_tky_wo_semantic"]),
    ("base_only", "w/o SFT (Base-only Qwen3-8B)", ["ablation_base_only", "base_only"]),
]

METRICS_OF_INTEREST = ["HR@1", "HR@5", "HR@10", "MRR", "NDCG@10"]


def evaluate_predictions_file(pred_file_path):
    """Fallback evaluator if metrics.txt is missing."""
    if not os.path.exists(pred_file_path):
        return None
    try:
        with open(pred_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not data or not isinstance(data, list):
            return None
        
        ks = [1, 3, 5, 10]
        total_samples = len(data)
        hit_counts = {k: 0 for k in ks}
        ndcg_sums = {k: 0.0 for k in ks}
        reciprocal_rank_sum = 0.0

        for sample in data:
            true_poi_id = str(sample.get("label"))
            predicted_poi_ids = sample.get("predicted_poi_ids", [])
            if predicted_poi_ids is None:
                predicted_poi_ids = []
            elif not isinstance(predicted_poi_ids, list):
                predicted_poi_ids = [predicted_poi_ids]
            predicted_poi_ids = [str(x) for x in predicted_poi_ids]

            rank = None
            try:
                rank = predicted_poi_ids.index(true_poi_id) + 1
                reciprocal_rank_sum += 1.0 / rank
            except ValueError:
                pass

            for k in ks:
                if rank is not None and rank <= k:
                    hit_counts[k] += 1
                    ndcg_sums[k] += 1.0 / math.log2(rank + 1)

        res = {"total_samples": total_samples}
        for k in ks:
            res[f"HR@{k}"] = hit_counts[k] / total_samples * 100 if total_samples else 0.0
            res[f"NDCG@{k}"] = ndcg_sums[k] / total_samples * 100 if total_samples else 0.0
        res["MRR"] = reciprocal_rank_sum / total_samples * 100 if total_samples else 0.0
        return res
    except Exception as e:
        print(f"[WARN] Error reading {pred_file_path}: {e}")
        return None


def extract_metrics_from_dir(target_dir):
    """Extracts final metrics from a result directory."""
    if not os.path.exists(target_dir):
        return None
    
    # Priority 1: Check metrics.csv (last column)
    csv_path = os.path.join(target_dir, "metrics.csv")
    if os.path.exists(csv_path):
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                rows = list(reader)
            if len(rows) > 1:
                metrics = {}
                for row in rows[1:]:
                    if len(row) >= 2:
                        k = row[0].strip()
                        val_str = row[-1].strip().rstrip("%")
                        try:
                            metrics[k] = float(val_str)
                        except ValueError:
                            pass
                if "HR@10" in metrics and "MRR" in metrics:
                    return metrics
        except Exception:
            pass

    # Priority 2: Check metrics.txt
    txt_path = os.path.join(target_dir, "metrics.txt")
    if os.path.exists(txt_path):
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                content = f.read()
            metrics = {}
            for line in content.splitlines():
                if ":" in line:
                    parts = line.split(":")
                    k = parts[0].strip()
                    val_str = parts[1].strip().rstrip("%")
                    try:
                        metrics[k] = float(val_str)
                    except ValueError:
                        pass
                elif "|" in line:
                    parts = [p.strip() for p in line.split("|")]
                    if len(parts) >= 2:
                        k = parts[0]
                        val_str = parts[-1].rstrip("%")
                        try:
                            metrics[k] = float(val_str)
                        except ValueError:
                            pass
            if "HR@10" in metrics and "MRR" in metrics:
                return metrics
        except Exception:
            pass

    # Priority 3: Fallback to calculate from poi_predictions.json
    pred_path = os.path.join(target_dir, "poi_predictions.json")
    return evaluate_predictions_file(pred_path)


def find_matched_dir(dataset_dir, candidate_names, dataset_name):
    """Finds directory matching any of candidate names."""
    if not os.path.exists(dataset_dir):
        return None
    for cand in candidate_names:
        # 1. Exact match
        p = os.path.join(dataset_dir, cand)
        if os.path.exists(p):
            return p
        # 2. Match with dataset suffix (e.g. ablation_wo_hsid_ca)
        p_ds = os.path.join(dataset_dir, f"{cand}_{dataset_name}")
        if os.path.exists(p_ds):
            return p_ds
    
    # 3. Fuzzy search
    for item in os.listdir(dataset_dir):
        full = os.path.join(dataset_dir, item)
        if os.path.isdir(full):
            for cand in candidate_names:
                if cand in item:
                    return full
    return None


def collect_ablation_results(results_base_dir, dataset):
    dataset_dir = os.path.join(results_base_dir, dataset)
    if not os.path.exists(dataset_dir):
        return {}

    results = {}
    for key, display_name, aliases in ABLATION_ORDER:
        matched_dir = find_matched_dir(dataset_dir, aliases, dataset)
        if matched_dir:
            metrics = extract_metrics_from_dir(matched_dir)
            if metrics:
                results[key] = {
                    "display_name": display_name,
                    "dir": matched_dir,
                    "metrics": metrics
                }
            else:
                results[key] = {
                    "display_name": display_name,
                    "dir": matched_dir,
                    "metrics": None
                }
        else:
            results[key] = {
                "display_name": display_name,
                "dir": None,
                "metrics": None
            }
    return results


def print_markdown_table(dataset, results):
    print(f"\n" + "=" * 80)
    print(f"[REPORT] GeoSemID Ablation Summary (Dataset: {dataset.upper()})")
    print("=" * 80)
    
    headers = ["Setting", "Samples", "HR@1", "HR@5", "HR@10", "MRR", "NDCG@10"]
    row_fmt = "{:<38} | {:>7} | {:>7} | {:>7} | {:>7} | {:>7} | {:>8}"
    
    print(row_fmt.format(*headers))
    print("-" * 39 + "+" + "-" * 9 + "+" + "-" * 9 + "+" + "-" * 9 + "+" + "-" * 9 + "+" + "-" * 9 + "+" + "-" * 10)
    
    for key, display_name, _ in ABLATION_ORDER:
        item = results.get(key)
        if item and item.get("metrics"):
            m = item["metrics"]
            total = str(int(m.get("total_samples", m.get("Total samples", 0))))
            hr1 = f"{m.get('HR@1', 0.0):.2f}%"
            hr5 = f"{m.get('HR@5', 0.0):.2f}%"
            hr10 = f"{m.get('HR@10', 0.0):.2f}%"
            mrr = f"{m.get('MRR', 0.0):.2f}%"
            ndcg = f"{m.get('NDCG@10', 0.0):.2f}%"
            print(row_fmt.format(display_name, total, hr1, hr5, hr10, mrr, ndcg))
        else:
            print(row_fmt.format(display_name, "-", "-", "-", "-", "-", "-"))
    print("=" * 80 + "\n")


def generate_latex_table(results_dict, datasets):
    """
    Generates LaTeX table code matching Table VI in paper.
    """
    latex_lines = []
    latex_lines.append(r"\begin{table*}[!t]")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\caption{GeoSemID 核心模块及检索专家的消融实验对比表}")
    latex_lines.append(r"\label{tab:ablation_studies}")
    latex_lines.append(r"\resizebox{\textwidth}{!}{%")
    latex_lines.append(r"\begin{tabular}{l" + "cc" * len(datasets) + r"}\toprule")
    
    # Header 1
    h1 = ["消融设置"]
    for ds in datasets:
        h1.append(r"\multicolumn{2}{c}{" + ds.upper() + r"}")
    latex_lines.append(" & ".join(h1) + r" \\ ")
    
    # Header 2
    h2 = [""]
    for _ in datasets:
        h2.extend(["HR@10", "MRR"])
    latex_lines.append(" & ".join(h2) + r" \\ \midrule")
    
    # Rows
    for key, display_name, _ in ABLATION_ORDER:
        row = []
        if key == "full_model":
            row.append(r"\textbf{" + display_name + r"}")
        else:
            row.append(display_name)
        
        for ds in datasets:
            res = results_dict.get(ds, {}).get(key)
            if res and res.get("metrics"):
                m = res["metrics"]
                hr10_val = f"{m.get('HR@10', 0.0):.2f}"
                mrr_val = f"{m.get('MRR', 0.0):.2f}"
                if key == "full_model":
                    row.append(r"\textbf{" + hr10_val + r"}")
                    row.append(r"\textbf{" + mrr_val + r"}")
                else:
                    row.append(hr10_val)
                    row.append(mrr_val)
            else:
                row.extend(["--", "--"])
        
        if key == "full_model":
            latex_lines.append(" & ".join(row) + r" \\ \midrule")
        elif key == "wo_semantic":
            latex_lines.append(" & ".join(row) + r" \\ \midrule")
        else:
            latex_lines.append(" & ".join(row) + r" \\")

    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}%")
    latex_lines.append(r"}")
    latex_lines.append(r"\end{table*}")
    
    return "\n".join(latex_lines)


def main():
    parser = argparse.ArgumentParser(description="GeoSemID 消融实验指标汇总与制表工具")
    parser.add_argument("--results_dir", type=str, default="results", help="结果根目录 (默认: results)")
    parser.add_argument("--dataset", type=str, default="all", choices=["ca", "nyc", "tky", "all"], help="指定数据集 (默认: all)")
    parser.add_argument("--save_latex", type=str, default="paper_writing/chinese/ablation_table_generated.tex", help="LaTeX 输出路径")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    results_base = (project_root / args.results_dir).resolve()

    datasets = ["ca", "nyc", "tky"] if args.dataset == "all" else [args.dataset]
    all_results = {}

    for ds in datasets:
        ds_res = collect_ablation_results(str(results_base), ds)
        all_results[ds] = ds_res
        print_markdown_table(ds, ds_res)

    latex_code = generate_latex_table(all_results, datasets)
    
    save_path = project_root / args.save_latex
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(latex_code)
    
    print(f"[SUCCESS] LaTeX 表格代码已自动生成并保存至: {save_path}")
    print("\n--- LaTeX Preview ---")
    print(latex_code)
    print("---------------------\n")


if __name__ == "__main__":
    main()

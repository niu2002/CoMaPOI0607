import os
import json
import math
import csv
import argparse
import sys
import io
from collections import defaultdict

# Force stdout and stderr to use UTF-8 encoding on Windows to prevent GBK encoding crashes
if sys.platform.startswith("win"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees), return in kilometers.
    """
    # Radius of earth in kilometers
    R = 6371.0
    
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [float(lat1), float(lon1), float(lat2), float(lon2)])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

def load_poi_coordinates(poi_csv_path):
    """Load POI coordinates mapped by poi_id string."""
    poi_coords = {}
    if not os.path.exists(poi_csv_path):
        print(f"[WARN] POI metadata CSV not found at: {poi_csv_path}")
        return poi_coords
        
    with open(poi_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            poi_id = str(row['poi_id'])
            poi_coords[poi_id] = (float(row['lat']), float(row['lon']))
    return poi_coords

def analyze_predictions(result_json_path, poi_csv_path, output_md_path=None):
    if not os.path.exists(result_json_path):
        print(f"[ERROR] Result JSON file not found: {result_json_path}")
        return
        
    with open(result_json_path, 'r', encoding='utf-8') as f:
        predictions = json.load(f)
        
    total_samples = len(predictions)
    if total_samples == 0:
        print("[ERROR] Predictions file is empty.")
        return
        
    poi_coords = load_poi_coordinates(poi_csv_path)
    has_coords = len(poi_coords) > 0

    # Metrics computation
    hits = {1: 0, 3: 0, 5: 0, 10: 0}
    ndcgs = {1: 0.0, 3: 0.0, 5: 0.0, 10: 0.0}
    reciprocal_rank_sum = 0.0
    
    # Spatial Metrics
    distance_errors = []  # Top-1 distance error
    spatial_recalls = {1.0: 0, 3.0: 0, 5.0: 0} # Spatial recall within d km
    
    # Constraint verification
    format_errors = 0
    not_in_candidate_count = 0
    total_preds_checked = 0
    length_mismatch_count = 0

    for item in predictions:
        # Normalize target labels and predictions
        label = str(item.get("label"))
        predicted = [str(p) for p in item.get("predicted_poi_ids", [])]
        candidates = [str(c) for c in item.get("candidates", [])]
        
        # Verify formats
        if not predicted:
            format_errors += 1
            continue
            
        if len(predicted) != 10:
            length_mismatch_count += 1
            
        for p in predicted:
            total_preds_checked += 1
            if candidates and p not in candidates:
                not_in_candidate_count += 1

        # Accuracy Metrics
        rank = None
        try:
            rank = predicted.index(label) + 1
            reciprocal_rank_sum += 1.0 / rank
        except ValueError:
            pass

        for k in hits.keys():
            if rank is not None and rank <= k:
                hits[k] += 1
                ndcgs[k] += 1.0 / math.log2(rank + 1)
                
        # Spatial Distance Analysis (using Top-1 prediction vs True Label)
        if has_coords and predicted:
            top_1_pred = predicted[0]
            if top_1_pred in poi_coords and label in poi_coords:
                lat1, lon1 = poi_coords[top_1_pred]
                lat2, lon2 = poi_coords[label]
                dist = haversine(lat1, lon1, lat2, lon2)
                distance_errors.append(dist)
                
                # Check spatial recalls
                for d in spatial_recalls.keys():
                    if dist <= d:
                        spatial_recalls[d] += 1

    # Aggregating metrics
    hr_metrics = {k: (hits[k] / total_samples) * 100 for k in hits.keys()}
    ndcg_metrics = {k: (ndcgs[k] / total_samples) * 100 for k in ndcgs.keys()}
    mrr = (reciprocal_rank_sum / total_samples) * 100
    
    mean_dist_err = sum(distance_errors) / len(distance_errors) if distance_errors else 0.0
    sorted_dists = sorted(distance_errors)
    median_dist_err = sorted_dists[len(sorted_dists)//2] if sorted_dists else 0.0
    
    spatial_recall_metrics = {d: (spatial_recalls[d] / len(distance_errors)) * 100 if distance_errors else 0.0 for d in spatial_recalls.keys()}
    
    # Print Diagnostics Table
    report = []
    report.append(f"# 📊 CoMaPOI 多维度学术评测与空间误差分析报告")
    report.append(f"- **评测数据文件**: `{result_json_path}`")
    report.append(f"- **总评估样本量 (Total Samples)**: `{total_samples}`")
    report.append("\n## 1. 经典推荐指标 (Recommendation Metrics)")
    report.append("| 指标名 | HR (命中率) | NDCG (归一化折现累计增益) |")
    report.append("| :--- | :---: | :---: |")
    for k in sorted(hits.keys()):
        report.append(f"| **@{k}** | {hr_metrics[k]:.2f}% | {ndcg_metrics[k]:.2f}% |")
    report.append(f"| **MRR** | **{mrr:.2f}%** | - |")
    
    if has_coords:
        report.append("\n## 2. 空间合理性分析 (Spatial Feasibility Analysis)")
        report.append("| 空间评估维度 | 指标数值 | 物理意义 |")
        report.append("| :--- | :---: | :--- |")
        report.append(f"| **Mean Distance Error (MDE)** | **{mean_dist_err:.3f} km** | Top-1 预测 POI 距离真实 POI 的平均大圆距离 |")
        report.append(f"| **Median Distance Error** | **{median_dist_err:.3f} km** | Top-1 预测 POI 距离真实 POI 的中位数距离 |")
        report.append(f"| **Spatial Recall @1.0km** | {spatial_recall_metrics[1.0]:.2f}% | Top-1 预测在真实 POI 地理半径 1 公里内的比例 |")
        report.append(f"| **Spatial Recall @3.0km** | {spatial_recall_metrics[3.0]:.2f}% | Top-1 预测在真实 POI 地理半径 3 公里内的比例 |")
        report.append(f"| **Spatial Recall @5.0km** | {spatial_recall_metrics[5.0]:.2f}% | Top-1 预测在真实 POI 地理半径 5 公里内的比例 |")

    report.append("\n## 3. 指令对齐与合规性诊断 (Alignment & Feasibility Diagnostics)")
    report.append("| 诊断维度 | 异常频次 / 总数 | 百分比 | 物理说明 |")
    report.append("| :--- | :---: | :---: | :--- |")
    report.append(f"| **格式解析错误 (Format Error)** | {format_errors} / {total_samples} | {format_errors/total_samples*100:.2f}% | 大模型未能生成合规 JSON 列表而导致解析为空 |")
    report.append(f"| **预测长度溢出 (Length Mismatch)** | {length_mismatch_count} / {total_samples} | {length_mismatch_count/total_samples*100:.2f}% | 推荐列表的推荐项不等于 10 个 |")
    if total_preds_checked > 0:
        report.append(f"| **越界推荐率 (Out-of-Candidates)** | {not_in_candidate_count} / {total_preds_checked} | {not_in_candidate_count/total_preds_checked*100:.2f}% | 推荐的 POI 不在多源融合召回候选池中 (幻觉) |")
    
    report_text = "\n".join(report)
    print(report_text)
    
    if output_md_path:
        os.makedirs(os.path.dirname(output_md_path), exist_ok=True)
        with open(output_md_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"\n[INFO] Complete diagnostics report saved to: {output_md_path}")

    # Generate academic plot if plotting tools are available
    if has_coords and distance_errors:
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Setup plot style
            sns.set_theme(style="whitegrid")
            plt.rcParams.update({
                "font.family": "serif",
                "font.size": 12,
                "axes.labelsize": 14,
                "axes.titlesize": 14,
                "xtick.labelsize": 11,
                "ytick.labelsize": 11
            })
            
            fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
            dists = np.array(distance_errors)
            clipped_dists = np.clip(dists, 0, 50)
            
            sns.histplot(
                clipped_dists, 
                bins=25, 
                kde=True, 
                color="#1f77b4", 
                edgecolor="w", 
                linewidth=1,
                alpha=0.6, 
                ax=ax,
                stat="probability"
            )
            
            # Draw median line
            median_val = np.median(dists)
            ax.axvline(
                median_val, 
                color="#d62728", 
                linestyle="--", 
                linewidth=1.8, 
                label=f"Median Error: {median_val:.2f} km"
            )
            
            ax.set_title("Probability Distribution of POI Prediction Distance Error", pad=15)
            ax.set_xlabel("Great-Circle Distance Error (km)")
            ax.set_ylabel("Probability")
            ax.set_xlim(0, 50)
            ax.legend(frameon=True, facecolor="white", edgecolor="none")
            
            plot_path = os.path.splitext(result_json_path)[0] + "_spatial_error.pdf"
            plt.tight_layout()
            plt.savefig(plot_path, bbox_inches='tight', format='pdf')
            plt.close()
            print(f"[INFO] Spatial error distribution plot saved to: {plot_path}")
        except ImportError:
            print("[INFO] matplotlib or seaborn not found. Skipping plot generation.")
        except Exception as e:
            print(f"[WARN] Failed to generate plot: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CoMaPOI Diagnostics and Spatial Evaluation Tool")
    parser.add_argument('--result_file', type=str, required=True, help="Path to predictions JSON file")
    parser.add_argument('--dataset', type=str, default="ca", choices=["ca", "nyc", "tky"], help="Dataset name")
    args = parser.parse_args()
    
    poi_csv = f"dataset_all/{args.dataset}/{args.dataset}_poi_info.csv"
    output_md = os.path.splitext(args.result_file)[0] + "_diagnostics.md"
    
    analyze_predictions(args.result_file, poi_csv, output_md)

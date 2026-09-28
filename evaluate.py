import csv
import json
import math


DEFAULT_EVAL_KS = (1, 3, 5, 10)


def _prediction_list(sample, key):
    predicted_poi_ids = sample.get(key, [])
    if predicted_poi_ids is None:
        return []
    if not isinstance(predicted_poi_ids, list):
        predicted_poi_ids = [predicted_poi_ids]
    return [str(poi_id) for poi_id in predicted_poi_ids]


def _metric_ks(top_k):
    """Use @1/@3/@5/@10 by default, keeping an explicit top_k if different."""
    ks = list(DEFAULT_EVAL_KS)
    if top_k and top_k not in ks:
        ks.append(top_k)
    return sorted(set(ks))


def evaluate_poi_predictions(args, file_path, top_k, output_file, csv_file, key):
    """
    Evaluate POI prediction results.

    The engineering default is HR/NDCG@1,@3,@5,@10. If a caller passes a
    different top_k, that cutoff is also reported for backward compatibility.
    """
    metric_ks = _metric_ks(top_k)
    total_samples = 0
    hit_counts = {k: 0 for k in metric_ks}
    ndcg_sums = {k: 0.0 for k in metric_ks}
    reciprocal_rank_sum = 0.0

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for sample in data:
        total_samples += 1
        true_poi_id = str(sample.get("label"))
        predicted_poi_ids = _prediction_list(sample, key)

        rank = None
        try:
            rank = predicted_poi_ids.index(true_poi_id) + 1
            reciprocal_rank_sum += 1 / rank
        except ValueError:
            pass

        for k in metric_ks:
            if rank is not None and rank <= k:
                hit_counts[k] += 1
                ndcg_sums[k] += 1 / math.log2(rank + 1)

    metrics = {"total_samples": total_samples}
    for k in metric_ks:
        metrics[f"HR@{k}"] = hit_counts[k] / total_samples * 100 if total_samples else 0

    metrics["MRR"] = reciprocal_rank_sum / total_samples * 100 if total_samples else 0

    for k in metric_ks:
        metrics[f"NDCG@{k}"] = ndcg_sums[k] / total_samples * 100 if total_samples else 0

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"Total samples: {total_samples}\n")
        for k in metric_ks:
            f.write(f"HR@{k}: {metrics[f'HR@{k}']:.2f}\n")
        f.write(f"MRR: {metrics['MRR']:.2f}\n")
        for k in metric_ks:
            f.write(f"NDCG@{k}: {metrics[f'NDCG@{k}']:.2f}\n")

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Total samples", total_samples])
        for k in metric_ks:
            writer.writerow([f"HR@{k}", f"{metrics[f'HR@{k}']:.2f}"])
        writer.writerow(["MRR", f"{metrics['MRR']:.2f}"])
        for k in metric_ks:
            writer.writerow([f"NDCG@{k}", f"{metrics[f'NDCG@{k}']:.2f}"])

    print(f"Evaluation results saved to {output_file} and {csv_file}")
    requested_k = top_k if top_k in metric_ks else metric_ks[-1]
    print(f"HR@{requested_k}: {metrics[f'HR@{requested_k}']:.2f}")
    print(f"MRR: {metrics['MRR']:.2f}")
    print(f"NDCG@{requested_k}: {metrics[f'NDCG@{requested_k}']:.2f}")

    # Automatically maintain cumulative multi-stage metrics across all intervals (200, 400, 600, 800, 900...)
    try:
        import os
        results_dir = os.path.dirname(output_file)
        update_cumulative_metrics(results_dir, top_k=top_k, key=key)
    except Exception as e:
        print(f"[WARN] Failed to update cumulative multi-stage metrics: {e}")

    return metrics


def update_cumulative_metrics(results_path, top_k=10, key='predicted_poi_ids'):
    """
    Scans all interim and final prediction files in results_path and creates/updates
    metrics.csv and metrics.txt containing columns for each interval (e.g. 200, 400, 600, 800, 900).
    """
    import os
    import glob
    import re
    if not results_path or not os.path.exists(results_path):
        return

    pattern = os.path.join(results_path, "interim_poi_predictions_*.json")
    interim_files = glob.glob(pattern)
    
    stage_data = {}
    for fpath in interim_files:
        basename = os.path.basename(fpath)
        m = re.search(r"interim_poi_predictions_(\d+)\.json", basename)
        if m:
            count = int(m.group(1))
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list) and data:
                    stage_data[count] = data
            except Exception:
                pass

    final_path = os.path.join(results_path, "poi_predictions.json")
    if os.path.exists(final_path):
        try:
            with open(final_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list) and data:
                count = len(data)
                stage_data[count] = data
        except Exception:
            pass

    if not stage_data:
        return

    metric_ks = _metric_ks(top_k)
    sorted_counts = sorted(stage_data.keys())
    eval_results = {}

    for count in sorted_counts:
        data = stage_data[count]
        total_samples = len(data)
        hit_counts = {k: 0 for k in metric_ks}
        ndcg_sums = {k: 0.0 for k in metric_ks}
        reciprocal_rank_sum = 0.0

        for sample in data:
            true_poi_id = str(sample.get("label"))
            predicted_poi_ids = _prediction_list(sample, key)

            rank = None
            try:
                rank = predicted_poi_ids.index(true_poi_id) + 1
                reciprocal_rank_sum += 1.0 / rank
            except ValueError:
                pass

            for k in metric_ks:
                if rank is not None and rank <= k:
                    hit_counts[k] += 1
                    ndcg_sums[k] += 1.0 / math.log2(rank + 1)

        m = {"Total samples": total_samples}
        for k in metric_ks:
            m[f"HR@{k}"] = hit_counts[k] / total_samples * 100 if total_samples else 0.0
        m["MRR"] = reciprocal_rank_sum / total_samples * 100 if total_samples else 0.0
        for k in metric_ks:
            m[f"NDCG@{k}"] = ndcg_sums[k] / total_samples * 100 if total_samples else 0.0

        eval_results[count] = m

    metric_rows = ["Total samples"] + [f"HR@{k}" for k in metric_ks] + ["MRR"] + [f"NDCG@{k}" for k in metric_ks]

    # Write metrics.csv
    csv_file = os.path.join(results_path, "metrics.csv")
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        headers = ["Metric"] + [f"{cnt}_samples" for cnt in sorted_counts]
        writer.writerow(headers)
        for m in metric_rows:
            row = [m]
            for cnt in sorted_counts:
                val = eval_results[cnt].get(m, 0.0)
                if m == "Total samples":
                    row.append(str(int(val)))
                else:
                    row.append(f"{val:.2f}")
            writer.writerow(row)

    # Write metrics.txt (Formatted text table)
    txt_file = os.path.join(results_path, "metrics.txt")
    col_widths = [15] + [14] * len(sorted_counts)
    header_strs = ["Metric".ljust(col_widths[0])] + [f"{cnt}_samples".rjust(col_widths[i+1]) for i, cnt in enumerate(sorted_counts)]
    
    with open(txt_file, "w", encoding="utf-8") as f:
        f.write(f"=== Multi-Stage Evaluation Summary: {results_path} ===\n\n")
        f.write(" | ".join(header_strs) + "\n")
        f.write("-" * (sum(col_widths) + 3 * len(sorted_counts)) + "\n")
        for m in metric_rows:
            vals = [m.ljust(col_widths[0])]
            for i, cnt in enumerate(sorted_counts):
                val = eval_results[cnt].get(m, 0.0)
                if m == "Total samples":
                    s = str(int(val))
                else:
                    s = f"{val:.2f}%"
                vals.append(s.rjust(col_widths[i+1]))
            f.write(" | ".join(vals) + "\n")
        f.write("\n")


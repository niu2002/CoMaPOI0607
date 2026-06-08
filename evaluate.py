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

    return metrics

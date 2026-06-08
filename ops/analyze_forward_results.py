#!/usr/bin/env python
"""Offline diagnostics for CoMaPOI forward prediction JSON files."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


DEFAULT_TOP_KS = (1, 3, 5, 10)


def normalize_poi_ids(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, (str, int, float)):
        values = [values]
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        try:
            poi_id = str(int(str(value).strip()))
        except (TypeError, ValueError):
            continue
        if poi_id not in seen:
            result.append(poi_id)
            seen.add(poi_id)
    return result


def load_predictions(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return list(data.values())
    raise TypeError(f"Unsupported predictions JSON root: {type(data).__name__}")


def extract_legacy_candidates(reasoning: str) -> dict[str, list[str]]:
    def between(marker: str, next_marker: str | None) -> list[str]:
        start = reasoning.find(marker)
        if start < 0:
            return []
        start += len(marker)
        end = reasoning.find(next_marker, start) if next_marker else len(reasoning)
        if end < 0:
            end = len(reasoning)
        return normalize_poi_ids(re.findall(r"\b\d+\b", reasoning[start:end]))

    agent1 = between("candidate_poi_list_agent1:", "candidate_poi_list_agent2:")
    agent2 = between("candidate_poi_list_agent2:", "final_prediction:")
    return {
        "rag": [],
        "agent1": agent1,
        "agent2": agent2,
        "union": normalize_poi_ids(agent1 + agent2),
    }


def extract_candidates(reasoning: Any) -> dict[str, list[str]]:
    if isinstance(reasoning, dict):
        candidates = reasoning.get("candidates", {})
        agent1 = normalize_poi_ids(candidates.get("agent1_top25", []))
        agent2 = normalize_poi_ids(candidates.get("agent2_top25", []))
        union = normalize_poi_ids(candidates.get("candidate_union", agent1 + agent2))
        return {
            "rag": normalize_poi_ids(candidates.get("rag_top100", [])),
            "agent1": agent1,
            "agent2": agent2,
            "union": union,
        }
    if isinstance(reasoning, str):
        return extract_legacy_candidates(reasoning)
    return {"rag": [], "agent1": [], "agent2": [], "union": []}


def parse_status(reasoning: Any) -> str:
    if isinstance(reasoning, dict):
        return str(reasoning.get("final_prediction", {}).get("parse_status", "unknown"))
    if isinstance(reasoning, str):
        return "legacy_string"
    return "unknown"


def hit_at(predicted: list[str], label: str, k: int) -> int:
    return int(label in predicted[:k])


def ndcg_at(predicted: list[str], label: str, k: int) -> float:
    for idx, poi_id in enumerate(predicted[:k], start=1):
        if poi_id == label:
            return 1.0 / math.log2(idx + 1)
    return 0.0


def reciprocal_rank(predicted: list[str], label: str) -> float:
    for idx, poi_id in enumerate(predicted, start=1):
        if poi_id == label:
            return 1.0 / idx
    return 0.0


def percentage(value: float) -> float:
    return round(value * 100, 2)


def analyze(predictions: Iterable[dict[str, Any]], top_ks: tuple[int, ...]) -> dict[str, Any]:
    rows = list(predictions)
    total = len(rows)
    length_distribution: Counter[str] = Counter()
    parse_status_counts: Counter[str] = Counter()
    rank_distribution: Counter[str] = Counter()
    candidate_hits: Counter[str] = Counter()
    predicted_total = 0
    predicted_from_union = 0

    hr_sums = {k: 0 for k in top_ks}
    ndcg_sums = {k: 0.0 for k in top_ks}
    mrr_sum = 0.0

    for row in rows:
        label_values = normalize_poi_ids(row.get("label"))
        label = label_values[0] if label_values else ""
        predicted = normalize_poi_ids(row.get("predicted_poi_ids", []))
        reasoning = row.get("reasoning_path")
        candidates = extract_candidates(reasoning)

        length_distribution[str(len(predicted))] += 1
        parse_status_counts[parse_status(reasoning)] += 1
        mrr_sum += reciprocal_rank(predicted, label)

        rank = "miss"
        for idx, poi_id in enumerate(predicted, start=1):
            if poi_id == label:
                rank = str(idx)
                break
        rank_distribution[rank] += 1

        for k in top_ks:
            hr_sums[k] += hit_at(predicted, label, k)
            ndcg_sums[k] += ndcg_at(predicted, label, k)

        if label in candidates["rag"]:
            candidate_hits["rag_top100"] += 1
        if label in candidates["agent1"]:
            candidate_hits["agent1_top25"] += 1
        if label in candidates["agent2"]:
            candidate_hits["agent2_top25"] += 1
        if label in set(candidates["agent1"]) | set(candidates["agent2"]):
            candidate_hits["agent1_or_agent2"] += 1

        union = set(candidates["union"])
        for poi_id in predicted:
            predicted_total += 1
            if poi_id in union:
                predicted_from_union += 1

    metrics = {
        "total_samples": total,
        "hr": {f"HR@{k}": percentage(hr_sums[k] / total) if total else 0.0 for k in top_ks},
        "ndcg": {f"NDCG@{k}": percentage(ndcg_sums[k] / total) if total else 0.0 for k in top_ks},
        "MRR": percentage(mrr_sum / total) if total else 0.0,
    }
    diagnostics = {
        "metrics": metrics,
        "prediction_length_distribution": dict(sorted(length_distribution.items(), key=lambda item: int(item[0]))),
        "top_k_complete_rate": percentage(sum(count for length, count in length_distribution.items() if int(length) >= max(top_ks)) / total) if total else 0.0,
        "parse_status_counts": dict(parse_status_counts),
        "rank_distribution": dict(sorted(rank_distribution.items(), key=lambda item: (item[0] == "miss", int(item[0]) if item[0].isdigit() else 10**9))),
        "candidate_recall": {
            name: {
                "hits": candidate_hits[name],
                "rate": percentage(candidate_hits[name] / total) if total else 0.0,
            }
            for name in ("rag_top100", "agent1_top25", "agent2_top25", "agent1_or_agent2")
        },
        "predicted_from_candidate_union": {
            "hits": predicted_from_union,
            "total_predictions": predicted_total,
            "rate": percentage(predicted_from_union / predicted_total) if predicted_total else 0.0,
        },
    }
    return diagnostics


def print_summary(diagnostics: dict[str, Any]) -> None:
    metrics = diagnostics["metrics"]
    print(f"Total samples: {metrics['total_samples']}")
    for name, value in metrics["hr"].items():
        print(f"{name}: {value:.2f}")
    print(f"MRR: {metrics['MRR']:.2f}")
    for name, value in metrics["ndcg"].items():
        print(f"{name}: {value:.2f}")
    print("")
    print(f"Prediction length distribution: {diagnostics['prediction_length_distribution']}")
    print(f"Top-K complete rate: {diagnostics['top_k_complete_rate']:.2f}")
    print(f"Parse status counts: {diagnostics['parse_status_counts']}")
    print(f"Rank distribution: {diagnostics['rank_distribution']}")
    print("Candidate recall:")
    for name, value in diagnostics["candidate_recall"].items():
        print(f"  {name}: {value['hits']} / {metrics['total_samples']} = {value['rate']:.2f}")
    union = diagnostics["predicted_from_candidate_union"]
    print(f"Predicted from agent candidate union: {union['hits']} / {union['total_predictions']} = {union['rate']:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze CoMaPOI forward prediction results.")
    parser.add_argument("--predictions", required=True, help="Path to poi_predictions.json or interim predictions JSON.")
    parser.add_argument("--output", default="", help="Optional diagnostics JSON output path.")
    parser.add_argument("--top_ks", default="1,3,5,10", help="Comma-separated top-k values.")
    args = parser.parse_args()

    predictions_path = Path(args.predictions)
    top_ks = tuple(sorted({int(item.strip()) for item in args.top_ks.split(",") if item.strip()}))
    diagnostics = analyze(load_predictions(predictions_path), top_ks or DEFAULT_TOP_KS)

    output_path = Path(args.output) if args.output else predictions_path.with_name("diagnostics_offline.json")
    output_path.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    print_summary(diagnostics)
    print(f"\nDiagnostics saved to: {output_path}")


if __name__ == "__main__":
    main()

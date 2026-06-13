#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from candidate_fusion import build_source_lists, fuse_candidates, normalize_poi_ids


DEFAULT_CUTOFFS = (10, 25, 50, 100, 200)


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def extract_user_id(content: str) -> str | None:
    patterns = [
        r'user_(\d+)_subtrajectory_(\d+)',
        r'user_id\s*:\s*"?(\d+)"?',
        r'"user_id"\s*:\s*"?(\d+)"?',
        r'user_(\d+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, content, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def parse_sample(sample: dict[str, Any]) -> tuple[str, str, str]:
    user_id = ""
    label = ""
    trajectory = ""
    for msg in sample.get("messages", []):
        content = str(msg.get("content", ""))
        if msg.get("role") == "user":
            trajectory = content
            user_id = extract_user_id(content) or user_id
        elif msg.get("role") == "assistant":
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict):
                    labels = normalize_poi_ids(parsed.get("next_poi_id"))
                    label = labels[0] if labels else label
            except json.JSONDecodeError:
                match = re.search(r'"next_poi_id"\s*:\s*"?(\d+)"?', content)
                if match:
                    label = match.group(1)
    return user_id, label, trajectory


def load_candidate_map(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    result: dict[str, list[str]] = {}
    for row in iter_jsonl(path):
        user_id = str(row.get("user_id", ""))
        result[user_id] = normalize_poi_ids(row.get("candidates", []))
    return result


def resolve_sample_path(dataset: str, mode: str, explicit_path: str) -> Path:
    if explicit_path:
        return Path(explicit_path)
    candidates = [
        PROJECT_ROOT / "dataset_all" / dataset / mode / f"{dataset}_{mode}.jsonl",
        PROJECT_ROOT / "dataset_all" / f"{dataset}_{mode}.jsonl",
        PROJECT_ROOT / "dataset_all" / dataset / f"{dataset}_{mode}.jsonl",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def resolve_candidate_path(dataset: str, mode: str, explicit_path: str) -> Path:
    if explicit_path:
        return Path(explicit_path)
    candidates = [
        PROJECT_ROOT / "dataset_all" / dataset / mode / f"{dataset}_{mode}_candidates.jsonl",
        PROJECT_ROOT / "dataset_all" / f"{dataset}_{mode}_candidates.jsonl",
        PROJECT_ROOT / "dataset_all" / dataset / f"{dataset}_{mode}_candidates.jsonl",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def load_prediction_agent_candidates(path: Path | None) -> dict[str, dict[str, list[str]]]:
    if path is None or not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data if isinstance(data, list) else list(data.values())
    result: dict[str, dict[str, list[str]]] = {}
    for row in rows:
        user_id = str(row.get("user_id", ""))
        reasoning = row.get("reasoning_path", {})
        candidates = reasoning.get("candidates", {}) if isinstance(reasoning, dict) else {}
        result[user_id] = {
            "agent1": normalize_poi_ids(candidates.get("agent1_top25", [])),
            "agent2": normalize_poi_ids(candidates.get("agent2_top25", [])),
        }
    return result


def update_hits(hit_counts: dict[str, Counter], source_name: str, candidates: list[str], label: str,
                cutoffs: tuple[int, ...]) -> None:
    for cutoff in cutoffs:
        if label in candidates[:cutoff]:
            hit_counts[source_name][f"@{cutoff}"] += 1


def analyze(args) -> dict[str, Any]:
    dataset = args.dataset
    sample_path = resolve_sample_path(dataset, args.mode, args.samples)
    candidate_path = resolve_candidate_path(dataset, args.mode, args.candidates)
    predictions_path = Path(args.predictions) if args.predictions else None

    candidate_map = load_candidate_map(candidate_path)
    prediction_candidates = load_prediction_agent_candidates(predictions_path)

    namespace = SimpleNamespace(
        dataset=dataset,
        max_item={"nyc": 5091, "tky": 7851, "ca": 13630}.get(dataset, 5091),
        num_candidate=args.num_candidate,
        candidate_fusion_strategy=args.strategy,
        fused_candidate_top_k=args.fused_candidate_top_k,
        rrf_k=args.rrf_k,
        rrf_weights=args.rrf_weights,
        history_candidate_k=args.history_candidate_k,
        geo_candidate_k=args.geo_candidate_k,
        category_candidate_k=args.category_candidate_k,
        popular_candidate_k=args.popular_candidate_k,
    )

    cutoffs = tuple(sorted({int(item) for item in args.cutoffs.split(",") if item.strip()}))
    samples = list(iter_jsonl(sample_path))[:args.num_samples if args.num_samples > 0 else None]

    hit_counts: dict[str, Counter] = {}
    source_lengths: dict[str, Counter] = {}
    label_missing = 0
    total = 0

    def counter_for(name: str) -> Counter:
        if name not in hit_counts:
            hit_counts[name] = Counter()
            source_lengths[name] = Counter()
        return hit_counts[name]

    for sample in samples:
        user_id, label, trajectory = parse_sample(sample)
        if not user_id or not label:
            label_missing += 1
            continue
        total += 1

        rag_candidates = candidate_map.get(user_id, [])
        agent_candidates = prediction_candidates.get(user_id, {})
        agent1 = agent_candidates.get("agent1", [])
        agent2 = agent_candidates.get("agent2", [])

        source_lists = build_source_lists(namespace, trajectory, rag_candidates, agent1, agent2)
        fusion_result = fuse_candidates(namespace, trajectory, rag_candidates, agent1, agent2)
        source_lists["fused"] = fusion_result.fused_candidates
        source_lists["agent_union"] = normalize_poi_ids(agent1 + agent2)
        source_lists["all_union"] = normalize_poi_ids(
            fusion_result.fused_candidates
            + source_lists.get("rag_top100", [])
            + source_lists.get("history_recent", [])
            + source_lists.get("history_freq", [])
            + source_lists.get("geo_near", [])
            + source_lists.get("category_similar", [])
            + source_lists.get("popular_global", [])
            + agent1
            + agent2
        )

        for source_name, candidates in source_lists.items():
            counter_for(source_name)
            source_lengths[source_name][str(len(candidates))] += 1
            update_hits(hit_counts, source_name, candidates, label, cutoffs)

    def rate(count: int) -> float:
        return round(count / total * 100, 2) if total else 0.0

    recall = {
        source: {
            cutoff: {
                "hits": counts[cutoff],
                "rate": rate(counts[cutoff]),
            }
            for cutoff in [f"@{k}" for k in cutoffs]
        }
        for source, counts in sorted(hit_counts.items())
    }

    return {
        "dataset": dataset,
        "mode": args.mode,
        "sample_path": str(sample_path),
        "candidate_path": str(candidate_path),
        "predictions_path": str(predictions_path) if predictions_path else "",
        "strategy": args.strategy,
        "total_samples": total,
        "label_missing": label_missing,
        "cutoffs": list(cutoffs),
        "recall": recall,
        "source_length_distribution": {source: dict(counter) for source, counter in sorted(source_lengths.items())},
    }


def print_summary(result: dict[str, Any]) -> None:
    print(f"Dataset: {result['dataset']} mode={result['mode']} strategy={result['strategy']}")
    print(f"Total samples: {result['total_samples']} label_missing={result['label_missing']}")
    print("Candidate recall:")
    for source, values in result["recall"].items():
        parts = [f"{cutoff}={stats['rate']:.2f}" for cutoff, stats in values.items()]
        print(f"  {source}: " + ", ".join(parts))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze candidate recall upper bounds for CoMaPOI.")
    parser.add_argument("--dataset", default="ca", choices=["nyc", "tky", "ca"])
    parser.add_argument("--mode", default="test", choices=["train", "test"])
    parser.add_argument("--samples", default="", help="Optional sample jsonl path")
    parser.add_argument("--candidates", default="", help="Optional RAG candidate jsonl path")
    parser.add_argument("--predictions", default="", help="Optional poi_predictions.json for agent candidates")
    parser.add_argument("--num_samples", type=int, default=101)
    parser.add_argument("--cutoffs", default="10,25,50,100,200")
    parser.add_argument("--strategy", default="rrf", choices=["none", "union", "rrf"])
    parser.add_argument("--num_candidate", type=int, default=25)
    parser.add_argument("--fused_candidate_top_k", type=int, default=100)
    parser.add_argument("--rrf_k", type=float, default=60.0)
    parser.add_argument("--rrf_weights", default="")
    parser.add_argument("--history_candidate_k", type=int, default=30)
    parser.add_argument("--geo_candidate_k", type=int, default=50)
    parser.add_argument("--category_candidate_k", type=int, default=50)
    parser.add_argument("--popular_candidate_k", type=int, default=50)
    parser.add_argument("--output", default="", help="Optional diagnostics JSON output")
    args = parser.parse_args()

    result = analyze(args)
    output_path = Path(args.output) if args.output else PROJECT_ROOT / "artifacts" / "candidate_recall" / f"{args.dataset}_{args.mode}_{args.strategy}_{args.num_samples}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print_summary(result)
    print(f"\nCandidate recall diagnostics saved to: {output_path}")


if __name__ == "__main__":
    main()

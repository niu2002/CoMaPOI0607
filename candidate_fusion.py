from __future__ import annotations

import csv
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


EVENT_RE = re.compile(
    r"POI ID\s+(\d+)\s+\((.*?),\s*([-+]?\d+(?:\.\d+)?),\s*([-+]?\d+(?:\.\d+)?)\)"
)


@dataclass
class FusionResult:
    fused_candidates: list[str]
    source_lists: dict[str, list[str]]
    source_ranks: dict[str, dict[str, int]]
    scores: dict[str, float]


def normalize_poi_ids(values: Any, max_item: int | None = None, limit: int | None = None) -> list[str]:
    if values is None:
        return []
    if not isinstance(values, list):
        values = [values]

    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        try:
            poi_id = int(value)
        except (TypeError, ValueError):
            continue
        if poi_id <= 0:
            continue
        if max_item is not None and poi_id > max_item:
            continue
        poi_id_str = str(poi_id)
        if poi_id_str in seen:
            continue
        seen.add(poi_id_str)
        normalized.append(poi_id_str)
        if limit is not None and len(normalized) >= limit:
            break
    return normalized


def parse_trajectory_events(text: str | None, max_item: int | None = None) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if not text:
        return events

    for match in EVENT_RE.finditer(text):
        poi_id = int(match.group(1))
        if poi_id <= 0:
            continue
        if max_item is not None and poi_id > max_item:
            continue
        events.append(
            {
                "poi_id": str(poi_id),
                "category": match.group(2).strip(),
                "lat": float(match.group(3)),
                "lon": float(match.group(4)),
            }
        )
    return events


def ordered_unique(values: list[str], limit: int | None = None) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
        if limit is not None and len(result) >= limit:
            break
    return result


def resolve_dataset_root(dataset: str) -> Path:
    return Path(__file__).resolve().parent / "dataset_all" / dataset


@lru_cache(maxsize=8)
def load_poi_info(dataset: str) -> dict[str, dict[str, Any]]:
    dataset_root = resolve_dataset_root(dataset)
    candidates = [
        dataset_root / f"{dataset}_poi_info.csv",
        Path(__file__).resolve().parent / "dataset_all" / f"{dataset}_poi_info.csv",
    ]
    poi_info_path = next((path for path in candidates if path.exists()), None)
    if poi_info_path is None:
        return {}

    poi_info: dict[str, dict[str, Any]] = {}
    with poi_info_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                poi_id = str(int(row["poi_id"]))
                poi_info[poi_id] = {
                    "category": str(row.get("category", "")).strip(),
                    "lat": float(row.get("lat", 0.0)),
                    "lon": float(row.get("lon", 0.0)),
                }
            except (KeyError, TypeError, ValueError):
                continue
    return poi_info


def _iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


@lru_cache(maxsize=8)
def load_popular_pois(dataset: str, max_item: int) -> list[str]:
    project_root = Path(__file__).resolve().parent
    candidates = [
        project_root / "dataset_all" / dataset / "train" / f"{dataset}_train.jsonl",
        project_root / "dataset_all" / f"{dataset}_train.jsonl",
        project_root / "dataset_all" / dataset / f"{dataset}_train.jsonl",
    ]
    train_path = next((path for path in candidates if path.exists()), None)
    if train_path is None:
        return []

    counter: Counter[str] = Counter()
    for sample in _iter_jsonl(train_path):
        for msg in sample.get("messages", []):
            if msg.get("role") == "user":
                events = parse_trajectory_events(msg.get("content", ""), max_item=max_item)
                counter.update(event["poi_id"] for event in events)
                break
    return [poi_id for poi_id, _ in counter.most_common()]


def history_recent(events: list[dict[str, Any]], limit: int) -> list[str]:
    return ordered_unique([event["poi_id"] for event in reversed(events)], limit=limit)


def history_frequent(events: list[dict[str, Any]], limit: int) -> list[str]:
    counter = Counter(event["poi_id"] for event in events)
    return [poi_id for poi_id, _ in counter.most_common(limit)]


def geo_near(events: list[dict[str, Any]], poi_info: dict[str, dict[str, Any]], limit: int) -> list[str]:
    if not events or not poi_info:
        return []
    last = events[-1]
    lat = float(last["lat"])
    lon = float(last["lon"])

    ranked: list[tuple[float, str]] = []
    for poi_id, info in poi_info.items():
        dist = (float(info["lat"]) - lat) ** 2 + (float(info["lon"]) - lon) ** 2
        ranked.append((dist, poi_id))
    ranked.sort(key=lambda item: item[0])
    return [poi_id for _, poi_id in ranked[:limit]]


def category_similar(events: list[dict[str, Any]], poi_info: dict[str, dict[str, Any]], limit: int) -> list[str]:
    if not events or not poi_info:
        return []
    category_counts = Counter(event["category"] for event in events[-10:] if event.get("category"))
    if not category_counts:
        return []

    last_lat = float(events[-1]["lat"])
    last_lon = float(events[-1]["lon"])
    category_rank = {category: rank for rank, (category, _) in enumerate(category_counts.most_common(), start=1)}

    ranked: list[tuple[int, float, str]] = []
    for poi_id, info in poi_info.items():
        category = str(info.get("category", "")).strip()
        if category not in category_rank:
            continue
        dist = (float(info["lat"]) - last_lat) ** 2 + (float(info["lon"]) - last_lon) ** 2
        ranked.append((category_rank[category], dist, poi_id))

    ranked.sort(key=lambda item: (item[0], item[1]))
    return [poi_id for _, _, poi_id in ranked[:limit]]


def parse_rrf_weights(raw_weights: str | None) -> dict[str, float]:
    default = {
        "rag_top100": 1.0,
        "history_recent": 1.2,
        "history_freq": 1.0,
        "geo_near": 0.8,
        "category_similar": 0.7,
        "popular_global": 0.3,
        "agent1_candidates": 0.8,
        "agent2_candidates": 0.8,
    }
    if not raw_weights:
        return default

    weights = dict(default)
    for item in raw_weights.split(","):
        if not item.strip() or "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        try:
            weights[key] = float(value.strip())
        except ValueError:
            continue
    return weights


def build_source_lists(args, current_trajectory: str, rag_candidates: Any,
                       agent1_candidates: Any, agent2_candidates: Any) -> dict[str, list[str]]:
    max_item = int(getattr(args, "max_item", 0) or 0)
    dataset = getattr(args, "dataset", "ca")
    fused_top_k = int(getattr(args, "fused_candidate_top_k", 50))
    source_limit = max(100, fused_top_k)

    events = parse_trajectory_events(current_trajectory, max_item=max_item)
    poi_info = load_poi_info(dataset)
    popular = load_popular_pois(dataset, max_item)

    return {
        "rag_top100": normalize_poi_ids(rag_candidates, max_item, limit=source_limit),
        "history_recent": history_recent(events, limit=int(getattr(args, "history_candidate_k", 30))),
        "history_freq": history_frequent(events, limit=int(getattr(args, "history_candidate_k", 30))),
        "geo_near": geo_near(events, poi_info, limit=int(getattr(args, "geo_candidate_k", 50))),
        "category_similar": category_similar(events, poi_info, limit=int(getattr(args, "category_candidate_k", 50))),
        "popular_global": normalize_poi_ids(popular, max_item, limit=int(getattr(args, "popular_candidate_k", 50))),
        "agent1_candidates": normalize_poi_ids(agent1_candidates, max_item, limit=int(getattr(args, "num_candidate", 25))),
        "agent2_candidates": normalize_poi_ids(agent2_candidates, max_item, limit=int(getattr(args, "num_candidate", 25))),
    }


def fuse_candidates(args, current_trajectory: str, rag_candidates: Any,
                    agent1_candidates: Any, agent2_candidates: Any) -> FusionResult:
    strategy = str(getattr(args, "candidate_fusion_strategy", "none")).lower()
    max_item = int(getattr(args, "max_item", 0) or 0)
    fused_top_k = int(getattr(args, "fused_candidate_top_k", 50))

    source_lists = build_source_lists(args, current_trajectory, rag_candidates, agent1_candidates, agent2_candidates)
    source_ranks = {
        source: {poi_id: rank for rank, poi_id in enumerate(candidates, start=1)}
        for source, candidates in source_lists.items()
    }

    if strategy == "none":
        fused = ordered_unique(
            source_lists["agent2_candidates"] + source_lists["agent1_candidates"] + source_lists["rag_top100"],
            limit=max(fused_top_k, int(getattr(args, "num_candidate", 25))),
        )
        return FusionResult(fused, source_lists, source_ranks, {})

    if strategy == "union":
        fused = ordered_unique(
            source_lists["agent2_candidates"]
            + source_lists["agent1_candidates"]
            + source_lists["rag_top100"]
            + source_lists["history_recent"]
            + source_lists["history_freq"]
            + source_lists["geo_near"]
            + source_lists["category_similar"]
            + source_lists["popular_global"],
            limit=fused_top_k,
        )
        return FusionResult(fused, source_lists, source_ranks, {})

    if strategy != "rrf":
        raise ValueError(f"Unsupported candidate_fusion_strategy: {strategy}")

    rrf_k = float(getattr(args, "rrf_k", 60))
    weights = parse_rrf_weights(getattr(args, "rrf_weights", ""))
    scores: defaultdict[str, float] = defaultdict(float)

    for source, candidates in source_lists.items():
        weight = weights.get(source, 1.0)
        if weight <= 0:
            continue
        for rank, poi_id in enumerate(candidates, start=1):
            if max_item and int(poi_id) > max_item:
                continue
            scores[poi_id] += weight / (rrf_k + rank)

    ranked = sorted(scores.items(), key=lambda item: (-item[1], int(item[0])))
    fused = [poi_id for poi_id, _ in ranked[:fused_top_k]]
    return FusionResult(fused, source_lists, source_ranks, dict(scores))


def summarize_fusion_result(result: FusionResult, source_preview_k: int = 20) -> dict[str, Any]:
    top_scores = {
        poi_id: round(result.scores.get(poi_id, 0.0), 8)
        for poi_id in result.fused_candidates[:source_preview_k]
    }
    return {
        "fused_top": result.fused_candidates,
        "source_lists": {
            source: candidates[:source_preview_k]
            for source, candidates in result.source_lists.items()
        },
        "source_ranks": {
            poi_id: {
                source: ranks[poi_id]
                for source, ranks in result.source_ranks.items()
                if poi_id in ranks
            }
            for poi_id in result.fused_candidates[:source_preview_k]
        },
        "scores": top_scores,
    }


def label_recall(source_lists: dict[str, list[str]], label: str, cutoffs: tuple[int, ...]) -> dict[str, dict[str, bool]]:
    label = str(label)
    result: dict[str, dict[str, bool]] = {}
    for source, candidates in source_lists.items():
        result[source] = {f"@{k}": label in candidates[:k] for k in cutoffs}
    return result


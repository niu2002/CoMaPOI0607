from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


POLLUTION_VALUES = set(str(year) for year in range(2000, 2027)) | {"25", "37", "38", "121", "122"}


def parse_args():
    parser = argparse.ArgumentParser(description="Audit CoMaPOI agent SFT JSONL files")
    parser.add_argument("paths", nargs="*", help="JSONL files to audit")
    parser.add_argument("--dataset", default="ca", choices=["nyc", "tky", "ca"], help="Dataset name for default paths")
    parser.add_argument("--max_item", type=int, default=13630, help="Maximum valid POI ID")
    return parser.parse_args()


def extract_json_object(text: str) -> dict[str, Any] | None:
    if not isinstance(text, str):
        return None
    candidates = [text.strip()]
    candidates.extend(block.strip() for block in re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.I))
    for candidate in candidates:
        if not candidate:
            continue
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return None


def flatten_ids(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, int, float)):
        value = [value]
    if not isinstance(value, list):
        return []
    ids: list[str] = []
    for item in value:
        try:
            ids.append(str(int(str(item).strip())))
        except (TypeError, ValueError):
            continue
    return ids


def expected_key(path: Path) -> str | None:
    name = path.name
    if "agent1" in name:
        return "candidate_poi_list_from_profile"
    if "agent2" in name:
        return "refined_candidate_from_rag"
    if "agent3" in name:
        return "next_poi_id"
    return None


def audit_file(path: Path, max_item: int) -> dict[str, Any]:
    key = expected_key(path)
    stats: Counter[str] = Counter()
    length_distribution: Counter[str] = Counter()
    pollution_counter: Counter[str] = Counter()

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stats["total"] += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                stats["invalid_jsonl"] += 1
                continue

            messages = row.get("messages", [])
            assistant = messages[-1].get("content", "") if messages else ""
            if not str(assistant).strip():
                stats["assistant_empty"] += 1
                continue

            data = extract_json_object(assistant)
            if data is None:
                stats["assistant_not_json"] += 1
                raw_ids = re.findall(r"\b\d+\b", str(assistant))
            else:
                stats["assistant_json"] += 1
                if key and key not in data:
                    stats["missing_expected_key"] += 1
                raw_ids = flatten_ids(data.get(key)) if key else []

            if len(raw_ids) != len(set(raw_ids)):
                stats["has_duplicates"] += 1

            valid_ids = []
            for poi_id in raw_ids:
                if poi_id in POLLUTION_VALUES:
                    pollution_counter[poi_id] += 1
                    stats["has_pollution_value"] += 1
                try:
                    poi_int = int(poi_id)
                except ValueError:
                    stats["non_numeric_id"] += 1
                    continue
                if 1 <= poi_int <= max_item:
                    valid_ids.append(poi_id)
                else:
                    stats["out_of_range_id"] += 1

            length_distribution[str(len(valid_ids))] += 1

    return {
        "path": str(path),
        "expected_key": key,
        "stats": dict(stats),
        "length_distribution": dict(sorted(length_distribution.items(), key=lambda item: int(item[0]))),
        "pollution_values": dict(pollution_counter),
    }


def main():
    args = parse_args()
    paths = [Path(path) for path in args.paths]
    if not paths:
        paths = [
            Path("finetune") / "data" / args.dataset / "agent2_train_samples.jsonl",
            Path("finetune") / "data" / args.dataset / "agent3_train_samples.jsonl",
        ]

    results = []
    for path in paths:
        if not path.exists():
            print(f"[audit] missing: {path}")
            continue
        result = audit_file(path, args.max_item)
        results.append(result)
        print(f"\n==== {path} ====")
        print(json.dumps(result, ensure_ascii=False, indent=2))

    if results:
        output = Path("artifacts") / "agent_training_data_audit.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[audit] saved to {output}")


if __name__ == "__main__":
    main()

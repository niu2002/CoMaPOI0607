from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path


POI_PATTERN = re.compile(
    r"POI ID\s+(\d+)\s+\((.+?),\s*([-0-9.]+),\s*([-0-9.]+)\)"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare dataset layout and POI info for multi-agent CoMaPOI runs")
    parser.add_argument("--dataset", type=str, default="ca", choices=["nyc", "tky", "ca"], help="Dataset name")
    parser.add_argument("--force", action="store_true", help="Overwrite generated files")
    return parser.parse_args()


def ensure_split_layout(project_root: Path, dataset: str, split: str, force: bool) -> Path:
    flat_path = project_root / "dataset_all" / f"{dataset}_{split}.jsonl"
    structured_dir = project_root / "dataset_all" / dataset / split
    structured_dir.mkdir(parents=True, exist_ok=True)
    structured_path = structured_dir / f"{dataset}_{split}.jsonl"

    if not flat_path.exists():
        raise FileNotFoundError(f"Missing source dataset file: {flat_path}")

    if force or not structured_path.exists():
        shutil.copy2(flat_path, structured_path)
        print(f"[prepare] copied {flat_path} -> {structured_path}")
    else:
        print(f"[prepare] keep existing structured file: {structured_path}")

    return structured_path


def collect_poi_info(*jsonl_paths: Path) -> dict[int, dict[str, str]]:
    poi_map: dict[int, dict[str, str]] = {}

    for jsonl_path in jsonl_paths:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                sample = json.loads(line)
                for message in sample.get("messages", []):
                    content = str(message.get("content", ""))
                    for poi_id, category, lat, lon in POI_PATTERN.findall(content):
                        poi_key = int(poi_id)
                        poi_map.setdefault(
                            poi_key,
                            {
                                "poi_id": str(poi_key),
                                "category": category.strip(),
                                "lat": lat.strip(),
                                "lon": lon.strip(),
                            },
                        )

    return poi_map


def write_poi_info(output_path: Path, poi_map: dict[int, dict[str, str]], force: bool) -> None:
    if output_path.exists() and not force:
        print(f"[prepare] keep existing poi_info: {output_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["poi_id", "category", "lat", "lon"])
        writer.writeheader()
        for _, record in sorted(poi_map.items(), key=lambda item: item[0]):
            writer.writerow(record)
    print(f"[prepare] wrote POI info CSV: {output_path} ({len(poi_map)} POIs)")


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent

    train_path = ensure_split_layout(project_root, args.dataset, "train", args.force)
    test_path = ensure_split_layout(project_root, args.dataset, "test", args.force)
    poi_map = collect_poi_info(train_path, test_path)

    if not poi_map:
        raise ValueError("Failed to extract any POI metadata from dataset JSONL files.")

    poi_info_path = project_root / "dataset_all" / args.dataset / f"{args.dataset}_poi_info.csv"
    write_poi_info(poi_info_path, poi_map, args.force)


if __name__ == "__main__":
    main()

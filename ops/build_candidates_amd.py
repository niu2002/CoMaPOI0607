from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.RAG import DEFAULT_QUERY_INSTRUCTION, RAG_Finder


def parse_args():
    parser = argparse.ArgumentParser(description="Build candidate POI cache for multi-agent AMD runs")
    parser.add_argument("--dataset", type=str, default="ca", choices=["nyc", "tky", "ca"], help="Dataset name")
    parser.add_argument("--mode", type=str, default="test", choices=["train", "test"], help="Dataset split")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to build candidates for")
    parser.add_argument("--top_k", type=int, default=100, help="Number of candidate POIs to keep per sample")
    parser.add_argument("--embedding_model_path", type=str, default="models/Qwen3-Embedding-4B", help="Local embedding model path")
    parser.add_argument("--embedding_batch_size", type=int, default=8, help="Embedding batch size")
    parser.add_argument("--embedding_max_length", type=int, default=2048, help="Embedding max length")
    parser.add_argument(
        "--embedding_query_instruction",
        type=str,
        default=DEFAULT_QUERY_INSTRUCTION,
        help="Instruction prefix for query embedding",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    namespace = SimpleNamespace(
        embedding_model_path=args.embedding_model_path,
        embedding_batch_size=args.embedding_batch_size,
        embedding_max_length=args.embedding_max_length,
        embedding_query_instruction=args.embedding_query_instruction,
    )
    max_item = {"nyc": 5091, "tky": 7851, "ca": 13630}[args.dataset]
    finder = RAG_Finder(
        data=args.dataset,
        num_test=args.num_samples,
        top_k=args.top_k,
        max_id=max_item,
        args=namespace,
        mode=args.mode,
    )
    finder.generate_candidates()


if __name__ == "__main__":
    main()

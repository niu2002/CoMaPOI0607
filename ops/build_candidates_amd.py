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
    parser.add_argument("--use_hsid", action="store_true", help="Enable HSID embedding text augmentation")
    parser.add_argument("--hsid_path", type=str, default="", help="Path to poi_hsid.json")
    parser.add_argument("--strategy", type=str, default="rag", choices=["rag", "expertrag"], help="Candidate retrieval strategy")
    
    # Cloud embedding and reranker arguments
    parser.add_argument("--use_cloud_embedding", action="store_true", help="Use cloud embedding API instead of local model")
    parser.add_argument("--embedding_api_key", type=str, default="", help="API key for cloud embedding API")
    parser.add_argument("--embedding_base_url", type=str, default="", help="Base URL for cloud embedding API")
    parser.add_argument("--embedding_model_name", type=str, default="text-embedding-v3", help="Model name for cloud embedding API")
    parser.add_argument("--use_reranker", action="store_true", help="Enable reranking of retrieved candidates")
    parser.add_argument("--reranker_model", type=str, default="qwen3-rerank", help="Model name for cloud reranker API")
    return parser.parse_args()


def main():
    args = parse_args()
    namespace = SimpleNamespace(
        embedding_model_path=args.embedding_model_path,
        embedding_batch_size=args.embedding_batch_size,
        embedding_max_length=args.embedding_max_length,
        embedding_query_instruction=args.embedding_query_instruction,
        use_hsid=args.use_hsid,
        hsid_path=args.hsid_path,
        use_cloud_embedding=args.use_cloud_embedding,
        embedding_api_key=args.embedding_api_key,
        embedding_base_url=args.embedding_base_url,
        embedding_model_name=args.embedding_model_name,
        use_reranker=args.use_reranker,
        reranker_model=args.reranker_model,
        strategy=args.strategy,
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
    output_file = finder.generate_candidates()
    
    import os
    if args.use_hsid:
        if "_candidates" in output_file:
            new_output_file = output_file.replace("_candidates", "_candidates_hsid")
            if os.path.exists(output_file):
                os.replace(output_file, new_output_file)
                print(f"Renamed HSID candidates target file to: {new_output_file}")


if __name__ == "__main__":
    main()

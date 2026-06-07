"""Run small-batch local evaluation with a base model plus LoRA adapter."""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.utils import logging as transformers_logging

from evaluate import evaluate_poi_predictions
from run_logging import setup_run_logging


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA small-batch inference evaluation")
    parser.add_argument("--dataset", type=str, default="nyc", help="Dataset name")
    parser.add_argument("--data_path", type=str, default="", help="JSONL file for evaluation input")
    parser.add_argument("--base_model_path", type=str, default="models/Qwen3.5-2B", help="Base model directory")
    parser.add_argument("--adapter_path", type=str, required=True, help="LoRA adapter directory")
    parser.add_argument("--start_index", type=int, default=0, help="0-based start index in the JSONL file")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of test samples to evaluate (0 for all remaining samples)")
    parser.add_argument("--top_k", type=int, default=10, help="Top-k POI predictions to keep per sample")
    parser.add_argument("--num_beams", type=int, default=10, help="Beam count for deterministic multi-candidate generation")
    parser.add_argument("--max_new_tokens", type=int, default=64, help="Maximum generated tokens")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p for generation")
    parser.add_argument("--save_interval", type=int, default=10, help="How often to write interim predictions")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load the base model in 4-bit mode")
    parser.add_argument("--device_map", type=str, default="auto", help="Transformers device_map value")
    return parser.parse_args()


def resolve_data_path(project_root: Path, dataset: str, data_path_arg: str) -> Path:
    if data_path_arg:
        data_path = Path(data_path_arg)
        if not data_path.is_absolute():
            data_path = (project_root / data_path).resolve()
        return data_path
    return (project_root / "dataset_all" / f"{dataset}_test.jsonl").resolve()


def resolve_model_path(project_root: Path, model_path_arg: str) -> Path:
    model_path = Path(model_path_arg)
    if not model_path.is_absolute():
        model_path = (project_root / model_path).resolve()
    return model_path


def load_jsonl_samples(path: Path, start_index: int, num_samples: int) -> list[dict]:
    samples: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx < start_index:
                continue
            if num_samples != 0 and len(samples) >= num_samples:
                break
            samples.append(json.loads(line))
    if not samples:
        raise ValueError(f"No samples loaded from {path} with start_index={start_index}, num_samples={num_samples}")
    return samples


def extract_answer_text(messages: list[dict]) -> str:
    for message in messages:
        if message.get("role") == "assistant":
            return str(message.get("content", ""))
    return ""


def extract_reference_poi_id(text: str) -> str:
    match = re.search(r'["\']next[_ ]?poi[_ ]?id["\']\s*:\s*([0-9]+)', text, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    fallback = re.search(r"\b([0-9]+)\b", text)
    return fallback.group(1) if fallback else "unparsed"


def extract_predicted_poi_id(text: str) -> str:
    match = re.search(r'["\']next[_ ]?poi[_ ]?id["\']\s*:\s*([0-9]+)', text, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    fallback = re.search(r"\b([0-9]+)\b", text)
    return fallback.group(1) if fallback else "unparsed"


def build_generation_kwargs(args, tokenizer):
    do_sample = args.temperature > 0
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": do_sample,
        "top_p": args.top_p,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    if do_sample:
        generation_kwargs["num_return_sequences"] = args.top_k
        generation_kwargs["temperature"] = args.temperature
    else:
        generation_kwargs["num_beams"] = max(args.num_beams, args.top_k)
        generation_kwargs["num_return_sequences"] = args.top_k
        generation_kwargs["early_stopping"] = True

    return generation_kwargs


def save_predictions(predictions: list[dict], output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    transformers_logging.disable_progress_bar()

    data_path = resolve_data_path(project_root, args.dataset, args.data_path)
    base_model_path = resolve_model_path(project_root, args.base_model_path)
    adapter_path = resolve_model_path(project_root, args.adapter_path)

    args.data_path = str(data_path)
    args.base_model_path = str(base_model_path)
    args.adapter_path = str(adapter_path)
    args.log_path = str(setup_run_logging(project_root, "infer_eval", args.dataset, args))

    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir = project_root / "artifacts" / "eval" / args.dataset / f"lora_eval_{run_tag}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = eval_dir / "predictions.json"
    metrics_txt_path = eval_dir / "metrics.txt"
    metrics_csv_path = eval_dir / "metrics.csv"
    summary_path = eval_dir / "summary.json"

    samples = load_jsonl_samples(data_path, args.start_index, args.num_samples)
    print(f"Loaded {len(samples)} samples from: {data_path}")

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    print(f"Loading base model from: {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map=args.device_map,
        quantization_config=quantization_config,
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    generation_kwargs = build_generation_kwargs(args, tokenizer)
    predictions: list[dict] = []
    total_start_time = time.time()

    for sample_offset, sample in enumerate(samples):
        sample_index = args.start_index + sample_offset
        prompt_messages = [msg for msg in sample["messages"] if msg.get("role") != "assistant"]
        reference_label_text = extract_answer_text(sample["messages"])
        label = extract_reference_poi_id(reference_label_text)

        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

        print(f"\n[{sample_offset + 1}/{len(samples)}] Running sample_index={sample_index}, label={label}")
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_kwargs)
        elapsed = time.time() - start_time

        predicted_poi_ids: list[str] = []
        raw_responses: list[str] = []
        for sequence in outputs:
            generated_ids = sequence[inputs["input_ids"].shape[1]:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            raw_responses.append(generated_text)
            predicted_poi_id = extract_predicted_poi_id(generated_text)
            if predicted_poi_id != "unparsed" and predicted_poi_id not in predicted_poi_ids:
                predicted_poi_ids.append(predicted_poi_id)
            if len(predicted_poi_ids) >= args.top_k:
                break

        print(f"Predicted POI IDs: {predicted_poi_ids}")
        print(f"Elapsed seconds: {elapsed:.2f}")

        predictions.append(
            {
                "sample_index": sample_index,
                "label": label,
                "reference_label": reference_label_text,
                "predicted_poi_ids": predicted_poi_ids,
                "raw_responses": raw_responses,
                "elapsed_seconds": elapsed,
            }
        )

        if args.save_interval > 0 and len(predictions) % args.save_interval == 0:
            interim_path = eval_dir / f"interim_predictions_{len(predictions)}.json"
            save_predictions(predictions, interim_path)
            print(f"Saved interim predictions to: {interim_path}")

    total_elapsed = time.time() - total_start_time
    save_predictions(predictions, predictions_path)

    metrics = evaluate_poi_predictions(
        args,
        str(predictions_path),
        args.top_k,
        str(metrics_txt_path),
        str(metrics_csv_path),
        key="predicted_poi_ids",
    )

    peak_mem_mb = None
    if torch.cuda.is_available():
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    summary = {
        "dataset": args.dataset,
        "data_path": str(data_path),
        "base_model_path": str(base_model_path),
        "adapter_path": str(adapter_path),
        "start_index": args.start_index,
        "num_samples": len(samples),
        "top_k": args.top_k,
        "generation": {
            "num_beams": args.num_beams,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "load_in_4bit": args.load_in_4bit,
        },
        "metrics": metrics,
        "elapsed_seconds_total": total_elapsed,
        "peak_cuda_memory_mb": peak_mem_mb,
        "predictions_path": str(predictions_path),
        "metrics_txt_path": str(metrics_txt_path),
        "metrics_csv_path": str(metrics_csv_path),
        "log_path": args.log_path,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nEvaluation run complete.")
    print(f"Predictions saved to: {predictions_path}")
    print(f"Metrics saved to: {metrics_txt_path}")
    print(f"Summary saved to: {summary_path}")
    if peak_mem_mb is not None:
        print(f"Peak CUDA memory allocated: {peak_mem_mb:.2f} MB")


if __name__ == "__main__":
    main()

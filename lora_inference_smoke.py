"""Run a local inference smoke test with a base model plus LoRA adapter."""

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

from run_logging import setup_run_logging


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA inference smoke test")
    parser.add_argument("--dataset", type=str, default="nyc", help="Dataset name")
    parser.add_argument("--data_path", type=str, default="", help="JSONL file for smoke input")
    parser.add_argument("--base_model_path", type=str, default="models/Qwen3.5-2B", help="Base model directory")
    parser.add_argument("--adapter_path", type=str, required=True, help="LoRA adapter directory")
    parser.add_argument("--sample_index", type=int, default=0, help="0-based sample index in the JSONL file")
    parser.add_argument("--max_new_tokens", type=int, default=64, help="Maximum generated tokens")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p for generation")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load the base model in 4-bit mode")
    parser.add_argument("--device_map", type=str, default="auto", help="Transformers device_map value")
    return parser.parse_args()


def load_jsonl_sample(path: Path, sample_index: int) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx == sample_index:
                return json.loads(line)
    raise IndexError(f"Sample index {sample_index} is out of range for {path}")


def extract_answer_text(messages: list[dict]) -> str:
    for message in messages:
        if message.get("role") == "assistant":
            return str(message.get("content", ""))
    return ""


def maybe_extract_poi_id(text: str) -> str:
    match = re.search(r'["\']next[_ ]?poi[_ ]?id["\']\s*:\s*([0-9]+)', text, flags=re.IGNORECASE)
    return match.group(1) if match else "unparsed"


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    transformers_logging.disable_progress_bar()

    if args.data_path:
        data_path = Path(args.data_path)
        if not data_path.is_absolute():
            data_path = (project_root / data_path).resolve()
    else:
        data_path = (project_root / "dataset_all" / f"{args.dataset}_test.jsonl").resolve()

    base_model_path = Path(args.base_model_path)
    if not base_model_path.is_absolute():
        base_model_path = (project_root / base_model_path).resolve()

    adapter_path = Path(args.adapter_path)
    if not adapter_path.is_absolute():
        adapter_path = (project_root / adapter_path).resolve()

    args.data_path = str(data_path)
    args.base_model_path = str(base_model_path)
    args.adapter_path = str(adapter_path)
    args.log_path = str(setup_run_logging(project_root, "infer_smoke", args.dataset, args))

    sample = load_jsonl_sample(data_path, args.sample_index)
    prompt_messages = [msg for msg in sample["messages"] if msg.get("role") != "assistant"]
    label_text = extract_answer_text(sample["messages"])

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    response_prefix = '{"next_poi_id": '
    prompt_text = f"{prompt_text}{response_prefix}"

    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

    print(f"Loading base model from: {base_model_path}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map=args.device_map,
        quantization_config=quantization_config,
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "top_p": args.top_p,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.temperature > 0:
        generation_kwargs["temperature"] = args.temperature

    print(f"Loaded smoke sample from: {data_path}")
    print(f"Prompt roles: {[msg['role'] for msg in prompt_messages]}")
    print(f"Reference label: {label_text}")
    print("Starting LoRA inference smoke...")

    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_kwargs)
    elapsed = time.time() - start_time

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    generated_suffix = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()
    generated_text = f"{response_prefix}{generated_suffix}"
    predicted_poi_id = maybe_extract_poi_id(generated_text)

    peak_mem_mb = None
    if torch.cuda.is_available():
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    print(f"Generated text: {generated_text}")
    print(f"Predicted POI ID: {predicted_poi_id}")
    print(f"Inference elapsed seconds: {elapsed:.2f}")
    if peak_mem_mb is not None:
        print(f"Peak CUDA memory allocated: {peak_mem_mb:.2f} MB")

    smoke_dir = project_root / "artifacts" / "smoke" / args.dataset
    smoke_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = smoke_dir / f"lora_smoke_{args.dataset}_{timestamp}.json"

    result_payload = {
        "dataset": args.dataset,
        "data_path": str(data_path),
        "sample_index": args.sample_index,
        "base_model_path": str(base_model_path),
        "adapter_path": str(adapter_path),
        "reference_label": label_text,
        "generated_text": generated_text,
        "predicted_poi_id": predicted_poi_id,
        "elapsed_seconds": elapsed,
        "peak_cuda_memory_mb": peak_mem_mb,
        "log_path": args.log_path,
    }
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result_payload, f, ensure_ascii=False, indent=2)

    print(f"Smoke result saved to: {result_path}")


if __name__ == "__main__":
    main()

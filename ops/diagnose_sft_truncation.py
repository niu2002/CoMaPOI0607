#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import sys
import re
from transformers import AutoTokenizer

def prepare_sample_text(example):
    # Extract assistant content
    assistant_content = next(
        (msg['content'] for msg in example['messages'] if msg['role'] == 'assistant'),
        None
    )

    if assistant_content:
        # Remove prefixes
        assistant_content = re.sub(
            r'^Answer:\s*"?(recent_mobility_analysis|historical_profile)"?:\s*"?',
            '',
            assistant_content,
            flags=re.DOTALL
        ).strip(' "')

    # Extract user content
    user_content = next(
        (msg['content'] for msg in example['messages'] if msg['role'] == 'user'),
        ""
    )
    user_content = re.sub(r"OUTPUT FORMAT:.*", "", user_content, flags=re.DOTALL).strip()

    # Format the text
    text = f"Question: {user_content}\n\nAnswer: {assistant_content}"
    
    # We also want to know the index where Answer starts
    prompt_only = f"Question: {user_content}\n\nAnswer: "
    return text, prompt_only

def main():
    dataset_path = "finetune/data/ca/agent3_train_samples.jsonl"
    model_path = "/mnt/workspace/comapoilatest/models/Llama-3.1-8B-Instruct"

    if not os.path.exists(dataset_path):
        print(f"[ERROR] SFT dataset file not found at: {dataset_path}")
        sys.exit(1)

    print(f"Loading tokenizer from {model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception as e:
        print(f"[ERROR] Failed to load tokenizer: {e}")
        sys.exit(1)

    print(f"Reading SFT samples from {dataset_path}...")
    samples = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    print(f"Total samples: {len(samples)}")

    lengths = []
    prompt_lengths = []
    fully_truncated_count = 0
    partially_truncated_count = 0
    limit = 2048

    for idx, sample in enumerate(samples):
        full_text, prompt_text = prepare_sample_text(sample)
        
        full_tokens = tokenizer(full_text, return_tensors="pt")["input_ids"].shape[1]
        prompt_tokens = tokenizer(prompt_text, return_tensors="pt")["input_ids"].shape[1]
        
        lengths.append(full_tokens)
        prompt_lengths.append(prompt_tokens)

        if prompt_tokens >= limit:
            # The prompt itself is longer than 2048. So the label is 100% truncated/cut off.
            fully_truncated_count += 1
        elif full_tokens > limit:
            # The prompt fits, but part of the assistant response is truncated
            partially_truncated_count += 1

    max_len = max(lengths)
    avg_len = sum(lengths) / len(lengths)
    min_len = min(lengths)

    print("\n=== Token Length Diagnosis ===")
    print(f"Max sequence length cap during SFT training: {limit}")
    print(f"Min sample tokens: {min_len}")
    print(f"Average sample tokens: {avg_len:.2f}")
    print(f"Max sample tokens: {max_len}")
    print(f"Samples exceeding {limit} sequence limit: {sum(1 for l in lengths if l > limit)} / {len(samples)} ({sum(1 for l in lengths if l > limit) / len(samples) * 100:.1f}%)")
    print(f"  - Fully Truncated (Prompt >= {limit}, Labels 100% missing): {fully_truncated_count} ({fully_truncated_count / len(samples) * 100:.1f}%)")
    print(f"  - Partially Truncated (Prompt < {limit}, but Answer cut off): {partially_truncated_count} ({partially_truncated_count / len(samples) * 100:.1f}%)")

    # Show a detailed sample profile
    print("\n=== Token length distribution ===")
    brackets = [500, 1000, 1500, 2048, 3000, 4000, 5000]
    last_b = 0
    for b in brackets:
        count = sum(1 for l in lengths if last_b < l <= b)
        print(f" {last_b:4d} - {b:4d} tokens: {count:3d} samples")
        last_b = b
    count_more = sum(1 for l in lengths if l > last_b)
    print(f" > {last_b:4d} tokens: {count_more:3d} samples")

if __name__ == "__main__":
    main()

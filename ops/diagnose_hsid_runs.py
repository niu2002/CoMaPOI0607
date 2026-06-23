import json
import os
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="Diagnose CoMaPOI forward inference results")
    parser.add_argument("--file", type=str, default="", help="Path to diagnostics.json")
    args = parser.parse_args()

    file_path = args.file
    if not file_path:
        # Try to find the latest ca predictions
        base_dir = "results/ca/amd-forward-101-top10-agent3-fused"
        if os.path.exists(base_dir):
            subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir)]
            subdirs = [d for d in subdirs if os.path.isdir(d)]
            if subdirs:
                # Get the latest modified subdirectory
                latest_subdir = max(subdirs, key=os.path.getmtime)
                pred_path = os.path.join(latest_subdir, "poi_predictions.json")
                diag_path = os.path.join(latest_subdir, "diagnostics.json")
                if os.path.exists(pred_path):
                    file_path = pred_path
                else:
                    file_path = diag_path
                print(f"[INFO] Auto-resolved latest output path: {file_path}")
            else:
                print(f"[ERROR] No subdirectory found in {base_dir}")
                sys.exit(1)
        else:
            print(f"[ERROR] Base directory {base_dir} does not exist. Please specify --file explicitly.")
            sys.exit(1)

    if not os.path.exists(file_path):
        print(f"[ERROR] Diagnostics file not found at: {file_path}")
        sys.exit(1)

    print(f"=== Diagnosing {file_path} ===")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Check metrics if stored
    metrics = data.get("metrics", {})
    print("\n--- Metrics Summary ---")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    samples = data.get("diagnostics", [])
    if not samples:
        # Fallback to checking list of items directly if structured differently
        if isinstance(data, list):
            samples = data
        elif isinstance(data, dict) and "results" in data:
            samples = data["results"]
        else:
            print(f"[ERROR] Could not extract samples list from file structure. Keys: {list(data.keys())}")
            sys.exit(1)

    print(f"\nTotal samples found: {len(samples)}")

    # Sample audit
    empty_rag_count = 0
    empty_agent1_count = 0
    empty_agent2_count = 0
    empty_fused_count = 0
    empty_pred_count = 0
    parse_failed_count = 0

    print("\n--- Auditing first 2 samples in detail ---")
    for idx, sample in enumerate(samples[:2]):
        print(f"\n[Sample {idx+1}] User ID: {sample.get('user_id')}")
        print(f"Target Label: {sample.get('label')}")
        
        # Check candidate lists
        rag = sample.get("rag_candidates", [])
        a1 = sample.get("candidate_poi_list_agent1", [])
        a2 = sample.get("candidate_poi_list_agent2", [])
        fused = sample.get("fused_candidates", [])
        pred = sample.get("predicted_poi_ids", [])
        
        print(f"  RAG candidates count: {len(rag)} (Sample: {rag[:5]})")
        print(f"  Agent 1 candidates count: {len(a1)} (Sample: {a1[:5]})")
        print(f"  Agent 2 candidates count: {len(a2)} (Sample: {a2[:5]})")
        print(f"  Fused candidates count: {len(fused)} (Sample: {fused[:5]})")
        print(f"  Predicted POI IDs count: {len(pred)} (Sample: {pred[:5]})")

        # Let's inspect the reasoning path messages
        reasoning = sample.get("reasoning_path", {})
        if not reasoning:
            reasoning = sample # fallback if flat
            
        print("\n  [Reasoning Flow]")
        # Check Profiler (Agent 1)
        a1_raw = reasoning.get("candidate_poi_list_agent1_raw", "") or reasoning.get("candidate_poi_list_agent1", "")
        print(f"  Profiler raw response length: {len(str(a1_raw))}")
        
        # Check Forecaster (Agent 2)
        a2_raw = reasoning.get("candidate_poi_list_agent2_raw", "") or reasoning.get("candidate_poi_list_agent2", "")
        print(f"  Forecaster raw response length: {len(str(a2_raw))}")
        if a2_raw and isinstance(a2_raw, str):
            print(f"  Forecaster raw excerpt: {a2_raw[:300]}...")

        # Check Predictor (Agent 3)
        init_pred_raw = reasoning.get("init_prediction", "")
        final_pred_raw = reasoning.get("final_prediction", "")
        print(f"  Predictor init response length: {len(str(init_pred_raw))}")
        if init_pred_raw and isinstance(init_pred_raw, str):
            print(f"  Predictor init raw excerpt: {init_pred_raw[:300]}...")
            
        print(f"  Predictor final response length: {len(str(final_pred_raw))}")

    # Stat aggregation
    for sample in samples:
        rag = sample.get("rag_candidates", [])
        a1 = sample.get("candidate_poi_list_agent1", [])
        a2 = sample.get("candidate_poi_list_agent2", [])
        fused = sample.get("fused_candidates", [])
        pred = sample.get("predicted_poi_ids", [])
        
        if not rag:
            empty_rag_count += 1
        if not a1 or (len(a1) == 1 and str(a1[0]) == '0'):
            empty_agent1_count += 1
        if not a2 or (len(a2) == 1 and str(a2[0]) == '0'):
            empty_agent2_count += 1
        if not fused:
            empty_fused_count += 1
        if not pred:
            empty_pred_count += 1

    print("\n--- Statistical Report ---")
    print(f"Samples with EMPTY RAG candidates: {empty_rag_count} / {len(samples)}")
    print(f"Samples with EMPTY Agent 1 candidates: {empty_agent1_count} / {len(samples)}")
    print(f"Samples with EMPTY Agent 2 candidates: {empty_agent2_count} / {len(samples)}")
    print(f"Samples with EMPTY Fused candidates: {empty_fused_count} / {len(samples)}")
    print(f"Samples with EMPTY final predictions: {empty_pred_count} / {len(samples)}")

if __name__ == "__main__":
    main()

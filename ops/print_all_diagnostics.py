import json
import glob
import os
import sys

def main():
    results_dir = "results"
    if not os.path.exists(results_dir):
        print(f"Error: '{results_dir}' directory not found.")
        sys.exit(1)
        
    diag_files = glob.glob(os.path.join(results_dir, "**", "diagnostics.json"), recursive=True)
    if not diag_files:
        print("No diagnostics.json files found in the results directory.")
        return
        
    print(f"Found {len(diag_files)} diagnostics.json files:")
    print("=" * 100)
    
    for file_path in sorted(diag_files):
        print(f"Path: {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Print metrics if they exist
            metrics = data.get("metrics", {})
            if metrics:
                print("  [Metrics]")
                print(f"    HR@10   : {metrics.get('HR@10', 'N/A')}")
                print(f"    MRR     : {metrics.get('MRR', 'N/A')}")
                print(f"    NDCG@10 : {metrics.get('NDCG@10', 'N/A')}")
            
            # Print candidate recall rates
            recall = data.get("candidate_recall", {})
            if recall:
                print("  [Candidate Recall Rates]")
                for key, val in recall.items():
                    rate = val.get("rate", "N/A")
                    hits = val.get("hits", "N/A")
                    print(f"    {key:<16}: {rate}% ({hits} hits)")
            
            # Print parsing status
            parse_status = data.get("parse_status_counts", {})
            if parse_status:
                print("  [Parse Status Counts]")
                for key, val in parse_status.items():
                    print(f"    {key:<16}: {val}")
                    
        except Exception as e:
            print(f"  Error reading file: {e}")
        print("-" * 100)

if __name__ == "__main__":
    main()

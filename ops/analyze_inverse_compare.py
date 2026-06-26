import json
import os

def analyze_samples(file_path):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found!")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [json.loads(line) for line in f]

    total_samples = len(lines)
    print(f"Total analyzed samples in SFT file: {total_samples}")
    
    if total_samples < 20:
        print(f"Warning: Only {total_samples} samples found. We need at least 20 samples to compare 10 vs 10.")
        plus_samples = lines[:min(10, total_samples)]
        turbo_samples = lines[10:] if total_samples > 10 else []
    else:
        plus_samples = lines[:10]
        turbo_samples = lines[10:20]

    def get_stats(samples_list):
        if not samples_list:
            return None
        
        long_term_lens = []
        short_term_lens = []
        candidate_counts = []
        target_counts = []
        raw_profiles = []
        
        for sample in samples_list:
            user_msg = sample['messages'][1]['content']
            assistant_msg = sample['messages'][2]['content']
            
            try:
                user_data = json.loads(user_msg)
                long_term = user_data.get("Long-Term Profile", "")
                short_term = user_data.get("Short-Term Mobility Profile", "")
                candidates = user_data.get("Candidate POIs from Profile Analysis", [])
                
                long_term_lens.append(len(long_term))
                short_term_lens.append(len(short_term))
                candidate_counts.append(len(candidates))
                raw_profiles.append(long_term)
            except Exception as e:
                long_term_lens.append(0)
                short_term_lens.append(0)
                candidate_counts.append(0)
                raw_profiles.append("Error parsing profile")
                
            try:
                assistant_data = json.loads(assistant_msg)
                targets = assistant_data.get("next_poi_id", [])
                target_counts.append(len(targets))
            except Exception as e:
                target_counts.append(0)
                
        return {
            "avg_long_term_len": sum(long_term_lens) / len(samples_list),
            "avg_short_term_len": sum(short_term_lens) / len(samples_list),
            "avg_candidates": sum(candidate_counts) / len(samples_list),
            "avg_targets": sum(target_counts) / len(samples_list),
            "raw_profiles": raw_profiles
        }

    plus_stats = get_stats(plus_samples)
    turbo_stats = get_stats(turbo_samples)

    print("\n" + "="*60)
    print("      逆向生成效果对比 (qwen-plus vs qwen-turbo)      ")
    print("="*60)
    
    if plus_stats:
        print(f"【qwen-plus (前 10 样本)】:")
        print(f"  - 平均 Long-Term Profile 字符长度  : {plus_stats['avg_long_term_len']:.2f}")
        print(f"  - 平均 Short-Term Profile 字符长度 : {plus_stats['avg_short_term_len']:.2f}")
        print(f"  - 平均生成的 Profile 候选集大小   : {plus_stats['avg_candidates']:.2f}")
        print(f"  - 平均标签/预测目标集大小         : {plus_stats['avg_targets']:.2f}")
        print("  - 样本 1 的 Long-Term Profile 预览: ")
        print(f"    \"{plus_stats['raw_profiles'][0][:150]}...\"")
        
    print("-"*60)
    
    if turbo_stats:
        print(f"【qwen-turbo (后 10 样本)】:")
        print(f"  - 平均 Long-Term Profile 字符长度  : {turbo_stats['avg_long_term_len']:.2f}")
        print(f"  - 平均 Short-Term Profile 字符长度 : {turbo_stats['avg_short_term_len']:.2f}")
        print(f"  - 平均生成的 Profile 候选集大小   : {turbo_stats['avg_candidates']:.2f}")
        print(f"  - 平均标签/预测目标集大小         : {turbo_stats['avg_targets']:.2f}")
        print("  - 样本 1 的 Long-Term Profile 预览: ")
        print(f"    \"{turbo_stats['raw_profiles'][0][:150]}...\"")
    else:
        print("【qwen-turbo (后 10 样本)】: 未生成或无数据。")
        
    print("="*60)

if __name__ == "__main__":
    analyze_samples("finetune/data/ca/agent3_train_samples.jsonl")

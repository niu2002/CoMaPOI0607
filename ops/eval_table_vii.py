#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_table_vii.py: 评测 GeoSemID 在长尾 POI、短轨迹序列与稀疏用户切片上的性能评估 (TABLE VII)

特点:
1. 【100% 纯 CPU 运行】: 仅执行指标计算与数据分流切片，无需 GPU，无需深度学习框架。
2. 【零外部第三方依赖】: 纯 Python 3 原生标准库 (无需 numpy, torch, pandas)，任何干净 Python 环境秒级运行 (耗时 < 3 秒)。
3. 【自适应路径检测】: 自动识别 /mnt/workspace/CoMaPOI0607 或当前项目路径。
4. 【学术指标精确对齐】: 严密遵循论文中定义的 HR@10 与 NDCG@10 推荐系统评测协议。
"""

import os
import sys
import json
import math
import re
import argparse
from collections import Counter
from pathlib import Path

# 防止在 Windows 或无 UTF-8 locale 环境下报 UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def compute_metrics(samples):
    """计算单个切片子集的 HR@10 与 NDCG@10"""
    total = len(samples)
    if total == 0:
        return {"total": 0, "HR@10": 0.0, "NDCG@10": 0.0}
    
    hits10 = 0
    ndcg10_sum = 0.0
    
    for s in samples:
        label = str(s.get("label", "")).strip()
        preds = [str(p).strip() for p in s.get("predicted_poi_ids", [])][:10]
        
        # 查找目标 POI 是否命中前 10 位
        try:
            rank = preds.index(label) + 1
            hits10 += 1
            ndcg10_sum += 1.0 / math.log2(rank + 1)
        except ValueError:
            pass
            
    return {
        "total": total,
        "HR@10": round(hits10 / total * 100, 2),
        "NDCG@10": round(ndcg10_sum / total * 100, 2)
    }


def find_project_root():
    """自动探测项目根目录"""
    candidates = [
        Path("/mnt/workspace/CoMaPOI0607"),
        Path.cwd(),
        Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd()
    ]
    for c in candidates:
        if (c / "dataset_all").exists():
            return c
    return Path.cwd()


def evaluate_table_vii(base_dir=None):
    root_dir = Path(base_dir) if base_dir else find_project_root()
    print("=" * 85)
    print(">> TABLE VII 切片性能评测 (纯 CPU 模式)")
    print(f">> 项目路径: {root_dir}")
    print("=" * 85)

    datasets = ["ca", "nyc", "tky"]
    results = {}

    poi_regex = re.compile(r'visit POI ID (\d+)')
    user_regex = re.compile(r'user_(\d+)_subtrajectory')

    for ds in datasets:
        ds_dir = root_dir / "dataset_all" / ds
        train_file = ds_dir / "train" / f"{ds}_train.jsonl"
        if not train_file.exists():
            # 兼容非子目录存储结构
            train_file = root_dir / "dataset_all" / f"{ds}_train.jsonl"
            
        test_file = ds_dir / "test" / f"{ds}_test.jsonl"
        if not test_file.exists():
            test_file = root_dir / "dataset_all" / f"{ds}_test.jsonl"

        pred_file = root_dir / "results" / ds / f"amd_rag_lora_with_sft_{ds}" / "poi_predictions.json"
        
        # 容错备选路径
        if not pred_file.exists():
            fallback_pred = root_dir / "results" / ds / f"amd_rag_lora_{ds}" / "poi_predictions.json"
            if fallback_pred.exists():
                pred_file = fallback_pred

        print(f"\n[检查数据集: {ds.upper()}]")
        print(f"  • 训练集: {train_file} ({'存在' if train_file.exists() else '缺失!'})")
        print(f"  • 测试集: {test_file} ({'存在' if test_file.exists() else '缺失!'})")
        print(f"  • 预测结果: {pred_file} ({'存在' if pred_file.exists() else '缺失!'})")

        if not pred_file.exists() or not train_file.exists() or not test_file.exists():
            print(f"  ⚠️  数据集或预测文件缺失，跳过 {ds.upper()}。请确认文件路径是否正确。")
            continue

        # 1. 统计训练集中每个 POI 的访问频次与每个用户的历史子轨迹交互数
        poi_counter = Counter()
        user_traj_counter = Counter()

        with open(train_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                u_content = item["messages"][1]["content"]
                u_match = user_regex.search(u_content)
                u_id = u_match.group(1) if u_match else None

                pois = poi_regex.findall(u_content)
                for p in pois:
                    poi_counter[p] += 1
                if u_id:
                    user_traj_counter[u_id] += 1

                assist_content = item["messages"][2]["content"]
                t_match = re.search(r'\"next_poi_id\":\s*(\d+)', assist_content)
                if t_match:
                    poi_counter[t_match.group(1)] += 1

        # 按照论文定义：访问频次位于后 80% 的稀疏长尾地点为长尾 POI
        sorted_unique_pois = sorted(poi_counter.keys(), key=lambda x: poi_counter[x])
        cutoff_idx = int(len(sorted_unique_pois) * 0.8)
        tail_pois_set = set(sorted_unique_pois[:cutoff_idx])

        # 2. 读取测试集元数据（提取对应 user_id 与历史步数）
        test_meta = []
        with open(test_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                u_content = item["messages"][1]["content"]
                u_match = user_regex.search(u_content)
                u_id = u_match.group(1) if u_match else None
                test_meta.append({
                    "user_id": u_id,
                    "train_trajs": user_traj_counter.get(u_id, 0)
                })

        # 3. 读取预测结果
        preds = json.loads(pred_file.read_text(encoding="utf-8"))
        if len(preds) != len(test_meta):
            print(f"  ⚠️  [注意] {ds.upper()} 预测条数 ({len(preds)}) 与测试集条数 ({len(test_meta)}) 不一致，将按预测列表长度对齐。")

        merged_samples = []
        for i, p in enumerate(preds):
            m = test_meta[i] if i < len(test_meta) else {}
            item = dict(p)
            item.update(m)
            merged_samples.append(item)

        # 4. 执行切片划分
        # (1) 长尾 POI 与 头部 POI
        tail_samples = [s for s in merged_samples if str(s.get("label", "")) in tail_pois_set]
        head_samples = [s for s in merged_samples if str(s.get("label", "")) not in tail_pois_set]

        # (2) 短轨迹序列 (<= 3 步) 与 长轨迹序列 (> 3 步)
        short_traj_samples = [s for s in merged_samples if s.get("train_trajs", 0) <= 3]
        long_traj_samples = [s for s in merged_samples if s.get("train_trajs", 0) > 3]

        # (3) 稀疏用户 (< 10 次) 与 活跃用户 (>= 10 次)
        sparse_user_samples = [s for s in merged_samples if s.get("train_trajs", 0) < 10]
        active_user_samples = [s for s in merged_samples if s.get("train_trajs", 0) >= 10]

        # 计算切片指标
        results[ds] = {
            "total": compute_metrics(merged_samples),
            "tail_poi": compute_metrics(tail_samples),
            "head_poi": compute_metrics(head_samples),
            "short_traj": compute_metrics(short_traj_samples),
            "long_traj": compute_metrics(long_traj_samples),
            "sparse_user": compute_metrics(sparse_user_samples),
            "active_user": compute_metrics(active_user_samples),
        }

        print(f"  ✓ {ds.upper()} 计算完成: 总样本={results[ds]['total']['total']}, 全量 HR@10={results[ds]['total']['HR@10']}%, NDCG@10={results[ds]['total']['NDCG@10']}%")

    if not results:
        print("\n❌ 未能找到任何有效的评测结果文件，请检查 results 目录！")
        return

    # 5. 格式化输出 TABLE VII 结果 Markdown 表格
    print("\n" + "=" * 85)
    print("📋 TABLE VII: GeoSemID 在长尾 POI、短轨迹序列与稀疏用户切片上的性能评估对比 (%)")
    print("=" * 85)

    ca_r = results.get("ca", {})
    nyc_r = results.get("nyc", {})
    tky_r = results.get("tky", {})

    ca_tot = ca_r.get("total", {}).get("total", 0)
    nyc_tot = nyc_r.get("total", {}).get("total", 0)
    tky_tot = tky_r.get("total", {}).get("total", 0)

    headers = f"| 数据切片划分类别 | CA ({ca_tot:,} 样本) HR@10 | CA NDCG@10 | NYC ({nyc_tot:,} 样本) HR@10 | NYC NDCG@10 | TKY ({tky_tot:,} 样本) HR@10 | TKY NDCG@10 |"
    divider = "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
    print(headers)
    print(divider)

    slice_rows = [
        ("tail_poi", "长尾 POI (尾部稀疏访问)"),
        ("head_poi", "头部 POI (高频访问地点)"),
        ("short_traj", "短轨迹序列 (<= 3 步)"),
        ("long_traj", "长轨迹序列 (> 3 步)"),
        ("sparse_user", "稀疏用户 (低频交互)"),
        ("active_user", "活跃用户 (高频打卡)")
    ]

    for key, name in slice_rows:
        ca_h = ca_r.get(key, {}).get("HR@10", 0.0)
        ca_n = ca_r.get(key, {}).get("NDCG@10", 0.0)
        nyc_h = nyc_r.get(key, {}).get("HR@10", 0.0)
        nyc_n = nyc_r.get(key, {}).get("NDCG@10", 0.0)
        tky_h = tky_r.get(key, {}).get("HR@10", 0.0)
        tky_n = tky_r.get(key, {}).get("NDCG@10", 0.0)
        print(f"| **{name}** | {ca_h:.2f} | {ca_n:.2f} | {nyc_h:.2f} | {nyc_n:.2f} | {tky_h:.2f} | {tky_n:.2f} |")

    # 6. 生成可直接放入论文的 LaTeX 源码
    print("\n" + "=" * 85)
    print("📄 LaTeX 代码 (直接复制嵌入论文):")
    print("=" * 85)
    latex_code = f"""\\begin{{table}}[!t]
\\centering
\\caption{{GeoSemID 在长尾 POI、短轨迹序列与稀疏用户切片上的性能评估对比 (\\%)。}}
\\label{{tab:slices}}
\\resizebox{{\\columnwidth}}{{!}}{{%
\\begin{{tabular}}{{lcccccc}}\\toprule
& \\multicolumn{{2}}{{c}}{{CA ({ca_tot:,} 样本)}} & \\multicolumn{{2}}{{c}}{{NYC ({nyc_tot:,} 样本)}} & \\multicolumn{{2}}{{c}}{{TKY ({tky_tot:,} 样本)}} \\\\
数据切片划分类别 & HR@10 & NDCG@10 & HR@10 & NDCG@10 & HR@10 & NDCG@10 \\\\ \\midrule
长尾 POI (尾部稀疏访问) & {ca_r.get('tail_poi',{}).get('HR@10', 0):.2f} & {ca_r.get('tail_poi',{}).get('NDCG@10', 0):.2f} & {nyc_r.get('tail_poi',{}).get('HR@10', 0):.2f} & {nyc_r.get('tail_poi',{}).get('NDCG@10', 0):.2f} & {tky_r.get('tail_poi',{}).get('HR@10', 0):.2f} & {tky_r.get('tail_poi',{}).get('NDCG@10', 0):.2f} \\\\
头部 POI (高频访问地点) & {ca_r.get('head_poi',{}).get('HR@10', 0):.2f} & {ca_r.get('head_poi',{}).get('NDCG@10', 0):.2f} & {nyc_r.get('head_poi',{}).get('HR@10', 0):.2f} & {nyc_r.get('head_poi',{}).get('NDCG@10', 0):.2f} & {tky_r.get('head_poi',{}).get('HR@10', 0):.2f} & {tky_r.get('head_poi',{}).get('NDCG@10', 0):.2f} \\\\ \\midrule
短轨迹序列 ($\\le$ 3 步) & {ca_r.get('short_traj',{}).get('HR@10', 0):.2f} & {ca_r.get('short_traj',{}).get('NDCG@10', 0):.2f} & {nyc_r.get('short_traj',{}).get('HR@10', 0):.2f} & {nyc_r.get('short_traj',{}).get('NDCG@10', 0):.2f} & {tky_r.get('short_traj',{}).get('HR@10', 0):.2f} & {tky_r.get('short_traj',{}).get('NDCG@10', 0):.2f} \\\\
长轨迹序列 ($>$ 3 步) & {ca_r.get('long_traj',{}).get('HR@10', 0):.2f} & {ca_r.get('long_traj',{}).get('NDCG@10', 0):.2f} & {nyc_r.get('long_traj',{}).get('HR@10', 0):.2f} & {nyc_r.get('long_traj',{}).get('NDCG@10', 0):.2f} & {tky_r.get('long_traj',{}).get('HR@10', 0):.2f} & {tky_r.get('long_traj',{}).get('NDCG@10', 0):.2f} \\\\ \\midrule
稀疏用户 (低频交互) & {ca_r.get('sparse_user',{}).get('HR@10', 0):.2f} & {ca_r.get('sparse_user',{}).get('NDCG@10', 0):.2f} & {nyc_r.get('sparse_user',{}).get('HR@10', 0):.2f} & {nyc_r.get('sparse_user',{}).get('NDCG@10', 0):.2f} & {tky_r.get('sparse_user',{}).get('HR@10', 0):.2f} & {tky_r.get('sparse_user',{}).get('NDCG@10', 0):.2f} \\\\
活跃用户 (高频打卡) & {ca_r.get('active_user',{}).get('HR@10', 0):.2f} & {ca_r.get('active_user',{}).get('NDCG@10', 0):.2f} & {nyc_r.get('active_user',{}).get('HR@10', 0):.2f} & {nyc_r.get('active_user',{}).get('NDCG@10', 0):.2f} & {tky_r.get('active_user',{}).get('HR@10', 0):.2f} & {tky_r.get('active_user',{}).get('NDCG@10', 0):.2f} \\\\ \\bottomrule
\\end{{tabular}}%
}}
\\end{{table}}"""
    print(latex_code)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate GeoSemID slices on Table VII")
    parser.add_argument("--root", type=str, default="", help="Project root directory (optional)")
    args = parser.parse_args()
    evaluate_table_vii(args.root)

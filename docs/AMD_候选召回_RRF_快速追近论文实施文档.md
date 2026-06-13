# AMD 候选召回与 RRF 融合快速追近论文实施文档

## 1. 背景

当前 CA 数据集上，论文表格中的 CoMaPOI 结果大致是：

```text
HR@5  = 33.00
HR@10 = 39.16
NDCG@5  = 24.96
NDCG@10 = 26.96
MRR = 23.10
```

我们当前 101 样本中最好的 base-only 基线是：

```text
HR@1  = 16.83
HR@3  = 21.78
HR@5  = 24.75
HR@10 = 29.70
MRR = 20.27
```

已经跑过的 LoRA 结果没有超过 base-only：

```text
agent2-only 300-ts30: HR@10 26.73, MRR 18.18
agent3-only 300-ts30: HR@10 27.72, MRR 17.87
agent2+3 300-ts30:    HR@10 25.74, MRR 16.63
clean100 agent2+3:    HR@10 28.71, MRR 19.05
```

结论：

1. 继续直接扩大 agent2/agent3 训练，不是最快缩小差距的路线。
2. 当前更像是候选召回和候选融合没有对齐论文，而不是单纯模型没训够。
3. 下一步应优先解决“正确 POI 是否进入候选池”和“多个候选来源如何融合”的问题。

## 2. 目标

本文档目标是设计一条更快接近论文结果的工程路线：

1. 先做候选召回上界诊断。
2. 再把候选池做宽，避免 agent2 强过滤误删正确 POI。
3. 引入工程版 RRF，把多个候选来源融合成稳定候选池。
4. 先用 base-only agent3 验证候选融合收益。
5. 如果候选融合有效，再考虑训练 agent3 或完整三 agent。

这轮不优先跑完整 900，也不优先换 NYC/TKY 数据集。

## 3. 当前核心问题

### 3.1 agent2 会伤候选池

agent2-only 结果低于 base-only，说明 agent2 不是稳定增强，而是在部分样本中把正确候选删掉了。

因此 agent2 后续不应该拥有“最终过滤权”，它更适合做：

```text
候选补充
候选重排
候选解释
候选打分信号
```

不应该做：

```text
把 RAG / 历史 / 地理候选强行压缩成唯一候选来源
```

### 3.2 final agent 救不了被删掉的 label

如果 label 不在最终候选池里，agent3 再强也无法命中。

所以提升顺序应该是：

```text
候选召回 > 候选融合 > 最终排序 > LoRA 微调
```

而不是：

```text
继续训练 agent2/agent3 > 期望模型自动修复候选缺失
```

### 3.3 论文差距可能来自流程未完全对齐

原论文中的 CoMaPOI 不是单纯让模型直接吐 top10，而是更强调：

```text
用户长期画像
短期移动模式
RAG 候选
多 agent 生成
候选融合 / RRF
最终排序
```

当前实现虽然跑通了 agent 编排，但候选融合仍然偏弱，RRF 的工程含义还没有充分体现。

## 4. 总体实施路线

推荐分 5 个阶段：

```text
阶段 A: 候选召回上界诊断
阶段 B: 多来源候选构造
阶段 C: 工程版 RRF 融合
阶段 D: 101 样本 base-only 验证
阶段 E: LoRA 是否继续的决策
```

每个阶段都先 dry-run，再 smoke，再 101 样本。只有 101 明确提升后，才考虑 900 或全量。

## 5. 阶段 A：候选召回上界诊断

### 5.1 要新增的能力

新增一个候选召回诊断脚本，建议命名：

```text
ops/analyze_candidate_recall.py
```

输入：

```text
dataset_all/ca/test/ca_test.jsonl
dataset_all/ca/test/ca_test_candidates.jsonl
dataset_all/ca/ca_poi_info.csv
可选: poi_predictions.json
```

输出：

```text
RAG@50 / RAG@100 / RAG@200 / RAG@500
history@K
geo@K
category@K
popular@K
agent1@25
agent2@25
union@50 / union@100 / union@200
```

### 5.2 诊断意义

如果 `union@100` 很低，例如低于 `40%`，那么 `HR@10` 想追到论文的 `39.16` 会很困难。

如果 `union@100` 已经较高，但 `HR@10` 不高，说明主要问题是排序。

判断规则：

```text
候选召回低: 先修候选
候选召回高但 HR 低: 修 agent3 排序
HR@10 高但 HR@1 低: 修排序精度
agent2@25 低于 base/RAG: 禁止 agent2 强过滤
```

## 6. 阶段 B：多来源候选构造

### 6.1 候选来源

建议至少构造以下候选来源：

```text
rag_top100: 由 Qwen3-Embedding-4B 召回
history_recent: 用户最近访问过的 POI
history_freq: 用户历史高频 POI
geo_near: 当前轨迹最后一个 POI 附近的 POI
category_similar: 与近期类别相同或相近的 POI
popular_global: 全局热门 POI
popular_user_city_or_region: 用户活动区域热门 POI
agent1_candidates: 长期画像候选
agent2_candidates: 短期 / RAG refinement 候选
```

### 6.2 候选池原则

新候选池必须遵守：

```text
只加分，不硬删
先做宽候选，再交给排序
保留 RAG 与规则 fallback
agent2 不能覆盖 RAG
候选来源要可追踪
```

每个候选 ID 应记录来源，例如：

```json
{
  "poi_id": "123",
  "sources": ["rag", "geo_near", "agent2"],
  "source_ranks": {
    "rag": 8,
    "geo_near": 3,
    "agent2": 12
  }
}
```

这样后续可以分析：

```text
命中的 POI 来自哪个来源
哪个来源经常误导
哪个来源贡献最大
```

## 7. 阶段 C：工程版 RRF 融合

### 7.1 RRF 公式

工程版 RRF 使用经典公式：

```text
score(poi) = sum( weight(source) / (rrf_k + rank(source, poi)) )
```

推荐初始参数：

```text
rrf_k = 60
```

推荐初始权重：

```text
rag_top100: 1.0
history_recent: 1.2
history_freq: 1.0
geo_near: 0.8
category_similar: 0.7
popular_global: 0.3
agent1_candidates: 0.8
agent2_candidates: 0.8
```

注意：agent2 当前不稳定，所以初始权重不要高于 RAG 和历史候选。

### 7.2 输出候选池大小

推荐：

```text
fused_candidate_top50 给 agent3
diagnostic_candidate_top100 用于分析召回上界
```

如果 top50 召回明显不足，可临时扩大：

```text
fused_candidate_top100 给 agent3
```

但最终给模型的候选不要过大，否则 prompt 变长、速度变慢、格式更不稳定。

### 7.3 为什么这比继续训练更快

训练只能在已有候选信息上学习。

RRF 可以立刻把多个弱信号合并：

```text
语义相似
历史重复
地理邻近
类别偏好
热门偏置
agent 白盒判断
```

这些信号对 POI 推荐通常比单纯语言模型生成更稳定。

## 8. 阶段 D：101 样本验证

### 8.1 第一轮只跑 base-only + RRF

先不要挂 LoRA，避免混淆变量。

实验名建议：

```text
amd-forward-101-top10-base-rrf-v1
```

对照对象：

```text
base-only 101:
HR@1 16.83
HR@3 21.78
HR@5 24.75
HR@10 29.70
MRR 20.27
```

成功标准：

```text
HR@10 >= 32.00: RRF 有明显价值
HR@10 >= 34.00: 候选融合方向强烈成立
MRR 不下降超过 0.5: 排序没有明显变坏
candidate union@100 明显高于旧结果
```

如果 HR@10 提升但 MRR 下降，说明召回变好了，但排序还要调。

### 8.2 第二轮再跑 agent3 LoRA

只有 base-only + RRF 有提升后，才跑：

```text
amd-forward-101-top10-agent3only-rrf-v1
```

目的：

```text
验证 agent3 LoRA 是否能在更好的候选池上提升排序
```

如果 agent3 仍然不如 base-only，则先停止 LoRA，继续修 prompt 和训练数据。

### 8.3 暂缓 agent2 LoRA

agent2 之前已经证明有伤候选池的风险。

在以下条件满足前，不建议继续把 agent2 LoRA 放进主流程：

```text
agent2@25 recall 不低于 base/RAG 融合候选
agent2-only 不再明显低于 base-only
agent2 输出稳定，parse_status ok >= 98%
```

## 9. 阶段 E：是否跑 900 / 全量的判断

不建议现在直接跑 900。

可以跑 900 的条件：

```text
101 样本 HR@10 比 base-only 至少提升 2 个点
101 样本 MRR 不低于 base-only
候选召回诊断显示 union@100 明显提升
输出 top10 完整率 >= 98%
parse_status ok >= 98%
```

如果 101 没提升，跑 900 的价值很低。

## 10. 需要修改的代码清单

### 10.1 必做

1. 新增 `ops/analyze_candidate_recall.py`
   - 统计多来源候选召回。
   - 输出 `diagnostics_candidate_recall.json` 和终端摘要。

2. 新增候选融合模块，建议命名 `candidate_fusion.py`
   - 实现 history / geo / category / popular 候选。
   - 实现 RRF 融合。
   - 记录每个 POI 的来源和 rank。

3. 修改 `inference_forward_new.py`
   - 增加 `--candidate_fusion_strategy`。
   - 支持 `none`、`union`、`rrf`。
   - final agent 使用 fused candidates。
   - reasoning_path 保存 fused candidates 和 source breakdown。

4. 修改 `ops/run_forward_amd.sh`
   - 暴露 `CANDIDATE_FUSION_STRATEGY`。
   - 暴露 `FUSED_CANDIDATE_TOP_K`。
   - 默认先不改变旧行为，避免破坏已有结果。

5. 修改 `ops/analyze_forward_results.py`
   - 增加 fused candidate recall。
   - 增加命中来源统计。

### 10.2 可选

1. 新增 `ops/run_forward_101_rrf_amd.sh`
   - 固定 101 样本、top10、RRF 参数。

2. 新增 `ops/compare_base_lora_samples.py`
   - 对比 base 命中但 LoRA 未命中的样本。
   - 输出错因：候选缺失 / 排序失败 / 解析失败。

3. 新增 `docs/AMD_RRF_实验记录.md`
   - 每次 101 结果追加到表格。

## 11. 实施顺序

### 11.1 本地 dry-run

本地先做语法和小文件测试：

```bash
python -m py_compile candidate_fusion.py ops/analyze_candidate_recall.py inference_forward_new.py ops/analyze_forward_results.py
bash -n ops/run_forward_amd.sh
```

如果本地没有 bash，就只做 Python 编译检查。

### 11.2 服务器 smoke

服务器先跑 2 条：

```bash
DATASET=ca \
NUM_SAMPLES=2 \
TOP_K=10 \
CANDIDATE_FUSION_STRATEGY=rrf \
FUSED_CANDIDATE_TOP_K=50 \
OP_STR=amd-forward-smoke-rrf-v1 \
bash ./ops/run_forward_amd.sh
```

验收：

```text
能跑完
predicted_poi_ids 长度接近 10
reasoning_path 中有 fused_candidates
diagnostics 中有候选来源统计
```

### 11.3 服务器 101

```bash
DATASET=ca \
NUM_SAMPLES=101 \
TOP_K=10 \
CANDIDATE_FUSION_STRATEGY=rrf \
FUSED_CANDIDATE_TOP_K=50 \
BASE_API_NAME=llama3.1-8b \
AGENT1_API=llama3.1-8b \
AGENT2_API=llama3.1-8b \
AGENT3_API=llama3.1-8b \
OP_STR=amd-forward-101-top10-base-rrf-v1 \
bash ./ops/run_forward_101_amd.sh
```

验收：

```text
HR@10 是否超过 29.70
MRR 是否接近或超过 20.27
候选 union@100 是否提升
命中来源是否能解释
```

## 12. 预期收益

保守预期：

```text
HR@10: 29.70 -> 31~33
MRR: 20.27 附近持平
```

较好预期：

```text
HR@10: 33~35
MRR: 20.5~22
```

如果 RRF 后 101 样本能达到：

```text
HR@10 >= 34
MRR >= 21
```

再跑 900 才比较值得。

## 13. 风险

1. 候选池变宽可能提升 HR@10，但降低 HR@1/MRR。
2. 地理/热门规则可能引入 popularity bias。
3. 如果数据集中的 POI 坐标或类别质量不稳定，geo/category 候选收益有限。
4. 如果原论文使用的 embedding 或 LLM 与当前差异很大，单靠 RRF 仍可能达不到论文结果。
5. 如果 101 样本波动较大，需要至少跑两个不同 start_index 的 101 对照。

## 14. 决策结论

当前最值得做的不是继续训 agent2/agent3，而是：

```text
先测候选召回上界
再做多来源候选融合
用 RRF 保留弱信号
先让 base-only 在更好候选池上提升
最后再判断 LoRA 是否有必要
```

这条路线最可能用最短时间缩小和论文 CA 结果的差距。


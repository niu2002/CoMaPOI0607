# AMD 论文式 RRF 逆向推理对比实验

## 1. 目的

本实验用于比较三种 agent2/agent3 逆向 SFT 数据生成方式：

```text
A. clean
   当前主线。label 不固定第一，降低 label-first 过拟合风险。

B. paper_label_first
   更接近论文式 Reverse Reasoning Fine-Tuning。
   agent1/agent2 候选列表和 agent3 top10 目标都把真实 label 放第一。

C. paper_label_first_fused
   在 B 的基础上，把当前多元候选融合结果 fused_candidates 也放进 agent3 训练 prompt，
   让训练输入更接近新 forward 里的 RRF 候选融合输入。
```

## 2. 为什么要做这个对比

之前结果显示：

```text
base-only 101:      HR@10 29.70, MRR 20.27
clean100 agent2+3: HR@10 28.71, MRR 19.05
300-ts30 agent2+3: HR@10 25.74, MRR 16.63
```

说明 agent2/agent3 LoRA 没有超过 base-only。

可能原因之一是当前逆向训练数据与论文 RRF 逻辑已经不完全一致：

1. 原始逆向提示词要求真实 next POI ranked first。
2. 当前 clean 逻辑为减少过拟合，把 label 插到前几位之一。
3. 新 forward 已经支持 fused candidates，但旧 inverse SFT 没让 agent3 学这个输入。

因此需要对比：

```text
恢复论文式 label-first 是否改善 agent2/agent3
加入 fused candidates 训练是否进一步改善 agent3 排序
```

## 3. 新增参数

入口：

```text
inference_inverse_new.py
ops/prepare_agent_training_data_amd.sh
```

新增环境变量：

```text
INVERSE_RRF_STYLE=clean
INVERSE_RRF_STYLE=paper_label_first
INVERSE_RRF_STYLE=paper_label_first_fused
```

额外 fused 参数：

```text
INVERSE_FUSION_STRATEGY=rrf
FUSED_CANDIDATE_TOP_K=50
RRF_K=60
RRF_WEIGHTS=""
```

## 4. 推荐实验顺序

先用 101 条逆向数据做快速对比：

```text
B1: paper_label_first, 101 inverse, test_size=10
C1: paper_label_first_fused, 101 inverse, test_size=10
```

如果 C1 优于 base-only 或明显优于 clean100，再扩大到 300。

不建议一开始直接生成 900 或全量逆向数据。

## 5. 验收指标

forward 101 主要看：

```text
HR@1
HR@3
HR@5
HR@10
MRR
NDCG@10
```

诊断指标看：

```text
agent1_top25 recall
agent2_top25 recall
agent1_or_agent2 recall
fused_top recall
parse_status
top10 完整率
```

判断标准：

```text
如果 paper_label_first > clean:
说明论文式 label-first 监督有帮助。

如果 paper_label_first_fused > paper_label_first:
说明 agent3 需要学习 fused candidates 输入。

如果两者都不如 base-only + RRF:
说明主要问题不是逆向数据格式，而是候选来源/排序 prompt/模型能力。
```

## 6. 当前代码默认行为

默认仍是旧主线：

```text
INVERSE_RRF_STYLE=clean
```

所以旧流程不会被破坏。


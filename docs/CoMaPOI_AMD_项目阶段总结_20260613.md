# CoMaPOI AMD 项目阶段总结（截至 2026-06-13）

## 1. 项目目标

当前目标是把 CoMaPOI 在 ModelScope AMD ROCm 环境上跑通，并尽量缩小与原论文 CA 数据集结果的差距。

论文 CA 参考结果：

```text
CoMaPOI:
HR@5  = 33.00
HR@10 = 39.16
NDCG@5  = 24.96
NDCG@10 = 26.96
MRR = 23.10
```

当前我们最好的 101 样本 base-only 基线：

```text
HR@1  = 16.83
HR@3  = 21.78
HR@5  = 24.75
HR@10 = 29.70
MRR = 20.27
```

核心判断：

1. 当前差距主要不是“模型没继续训够”，而是候选召回、候选融合、RRF 流程与论文设定还没完全对齐。
2. agent2/agent3 LoRA 已经跑通过，但没有稳定超过 base-only。
3. 下一阶段应优先修候选池和 RRF 融合，再决定是否继续扩大 LoRA 训练。

## 2. 本地与远端环境

本地仓库：

```text
F:\jjy\project\CoMaPoi0607
当前工作分支: jjy
远程仓库: https://github.com/niu2002/CoMaPOI0607
```

服务器仓库：

```text
/mnt/workspace/comapoilatest/CoMaPOI0607
```

服务器模型目录：

```text
/mnt/workspace/comapoilatest/models
```

当前使用模型：

```text
主模型: /mnt/workspace/comapoilatest/models/Llama-3.1-8B-Instruct
Embedding: /mnt/workspace/comapoilatest/models/Qwen3-Embedding-4B
```

AMD 环境：

```text
ModelScope PAI-DSW AMD GPU 环境
ROCm 7.2.1
torch 2.9.1
显存约 192GB
```

远程控制方式：

```text
SSH/FRP
ssh -i "F:\jjy\LLM4POI\.codex_ssh\id_ed25519" -p 7001 root@114.55.55.110
```

vLLM 服务默认：

```text
PORT=7863
SERVED_MODEL_NAME=llama3.1-8b
OpenAI-compatible endpoint: http://127.0.0.1:7863/v1
```

## 3. 已解决的工程问题

### 3.1 远程控制

先后尝试过 trycloudflare + ttyd，后来切到 SSH/FRP。

最终 SSH/FRP 更稳定，当前使用：

```bash
ssh -i "F:\jjy\LLM4POI\.codex_ssh\id_ed25519" -p 7001 root@114.55.55.110
```

注意：

1. `frpc` 只需要保留一个进程。
2. 如果重复启动，会出现 `proxy [ubuntu-ssh] already exists`。
3. 清理时不要杀掉唯一可用的 `frpc`，否则 SSH 会断。

### 3.2 AMD / ROCm 上的 vLLM

已验证 vLLM 能在 AMD 环境启动 Llama-3.1-8B-Instruct。

base-only 服务启动方式：

```bash
MODEL_ROOT=/mnt/workspace/comapoilatest/models \
MODEL_NAME=Llama-3.1-8B-Instruct \
SERVED_MODEL_NAME=llama3.1-8b \
PORT=7863 \
bash ./ops/serve_agents_amd.sh
```

检查方式：

```bash
curl -m 5 -sS http://127.0.0.1:7863/v1/models
```

### 3.3 依赖问题

遇到过：

```text
ModuleNotFoundError: No module named 'trl'
ModuleNotFoundError: No module named 'agentscope'
ModuleNotFoundError: No module named 'agentscope.parsers'
agentscope 与 loguru 版本冲突
```

目前处理方式：

```text
agentscope==0.1.6
loguru==0.6.0
trl / peft / datasets / accelerate / transformers 正常安装
```

依赖安装脚本：

```bash
bash ./ops/install_amd_deps.sh
```

环境检查：

```bash
python -c "import trl, agentscope, openai, transformers, peft, datasets, accelerate; print('deps_ok')"
```

## 4. 已实现的代码能力

### 4.1 AMD 运行脚本

已实现或修复：

```text
ops/serve_agents_amd.sh
ops/run_forward_amd.sh
ops/run_forward_101_amd.sh
ops/run_forward_101_agent23_amd.sh
ops/train_agent2_agent3_amd.sh
ops/serve_agent2_agent3_amd.sh
ops/prepare_agent_training_data_amd.sh
ops/install_amd_deps.sh
```

能力：

1. 启动 base-only vLLM。
2. 启动 agent2/agent3 LoRA adapter。
3. forward 前等待 `/v1/models` ready。
4. 101 样本固定评估。
5. agent2 + agent3 串行训练。
6. 支持 checkpoint resume。

### 4.2 多 agent 结构化输出

已把 `reasoning_path` 从长字符串逐步改为结构化对象，便于解析和诊断。

结构大致包括：

```json
{
  "long_term_profile": {},
  "short_term_profile": {},
  "candidates": {
    "rag_top100": [],
    "agent1_top25": [],
    "agent2_top25": [],
    "candidate_union": [],
    "fused_top": [],
    "fusion": {}
  },
  "initial_prediction": {},
  "final_prediction": {}
}
```

### 4.3 逆向训练数据清洗

修复过的问题：

1. agent2/agent3 训练输出和 forward 推理 JSON schema 不一致。
2. agent3 曾错误学习 agent2 的候选列表。
3. 训练数据中可能混入年份、坐标、时间等污染数字。
4. label 固定排第一导致过拟合。

相关提交：

```text
48d1dd2 修复AMD逆向训练数据清洗与格式
```

### 4.4 候选召回与 RRF 融合

最新实现：

```text
803aa0d 增加候选召回诊断和RRF融合
```

新增文件：

```text
candidate_fusion.py
ops/analyze_candidate_recall.py
docs/AMD_候选召回_RRF_快速追近论文实施文档.md
```

新增能力：

1. 多来源候选构造。
2. RRF 融合。
3. `fused_top` 写入 reasoning。
4. final agent prompt 支持 `Fused Candidate POIs`。
5. 离线统计 `fused_top` 召回。
6. 不改变旧默认行为，只有显式设置 `CANDIDATE_FUSION_STRATEGY=rrf` 才启用新逻辑。

候选来源：

```text
rag_top100
history_recent
history_freq
geo_near
category_similar
popular_global
agent1_candidates
agent2_candidates
```

## 5. 已跑过的重要实验

### 5.1 base-only 101

当前最好 101 基线：

```text
HR@1  = 16.83
HR@3  = 21.78
HR@5  = 24.75
HR@10 = 29.70
MRR = 20.27
```

这是后续 RRF 和 LoRA 的主要对照。

### 5.2 base-only 900

```text
Total samples: 900
HR@1: 13.78
HR@5: 29.44
HR@10: 29.44
HR@20: 29.44
MRR: 19.61
NDCG@5: 22.06
NDCG@10: 22.06
NDCG@20: 22.06
```

注意：当时输出只有 top5，所以 `HR@5/10/20` 一样，不能直接与 top10 新口径比较。

### 5.3 merged LoRA

CA 全量训练 1 epoch：

```text
BATCH_SIZE=24
耗时约 146.87 分钟
峰值显存约 125GB
```

200 样本评估：

```text
HR@5: 29.50
MRR: 20.03
NDCG@5: 22.41
```

这条路线不是论文完整多 agent / RRF 路线，只能作为工程 baseline。

### 5.4 agent2 / agent3 LoRA 消融

```text
agent2-only 300-ts30:
HR@1 13.86
HR@3 21.78
HR@5 23.76
HR@10 26.73
MRR 18.18

agent3-only 300-ts30:
HR@1 13.86
HR@3 20.79
HR@5 23.76
HR@10 27.72
MRR 17.87

agent2+3 300-ts30:
HR@1 12.87
HR@3 17.82
HR@5 19.80
HR@10 25.74
MRR 16.63

clean100 agent2+3:
HR@1 15.84
HR@3 19.80
HR@5 23.76
HR@10 28.71
MRR 19.05
```

结论：

1. LoRA adapter 确实挂载并生效了。
2. 但当前 agent2/agent3 LoRA 没有超过 base-only。
3. agent2 有伤候选池风险。
4. 继续盲目扩大 agent2/agent3 训练不划算。

## 6. 当前关键判断

### 6.1 为什么 base-only 反而更好

原因不是 LoRA 没挂上，而是：

1. 逆向生成训练数据质量有限。
2. agent2 学到的是“过滤候选”，但过滤会误删正确 POI。
3. agent3 在错误候选池上排序，无法救回已经被删掉的 label。
4. 当前候选融合还没有完全发挥论文 RRF 的作用。
5. 101 样本已经足够暴露趋势，不建议直接跑大规模赌结果。

### 6.2 为什么现在优先修候选召回

如果 label 不在候选池里：

```text
final agent 再强也无法命中
```

所以提升顺序应是：

```text
候选召回 > 候选融合 > 排序 > LoRA
```

而不是：

```text
继续训练 agent2/agent3 > 希望模型自己修正候选缺失
```

## 7. 下一步推荐路线

### 7.1 不先跑 900

当前不建议马上跑 900 或全量。

原因：

1. 101 样本已经显示 LoRA 未超过 base-only。
2. 直接扩大样本大概率只是更稳定地证明“没提升”。
3. 应先用候选召回诊断判断上界。

### 7.2 先跑候选召回诊断

```bash
python ./ops/analyze_candidate_recall.py \
  --dataset ca \
  --mode test \
  --num_samples 101 \
  --strategy rrf \
  --fused_candidate_top_k 100
```

重点看：

```text
rag_top100
history_recent
history_freq
geo_near
category_similar
popular_global
fused
all_union
```

如果 `fused` 和 `all_union` 明显高于旧 RAG，说明 RRF 有价值。

### 7.3 再跑 base-only + RRF 101

```bash
DATASET=ca \
MODEL_NAME=Llama-3.1-8B-Instruct \
PORT=7863 \
BASE_API_NAME=llama3.1-8b \
AGENT1_API=llama3.1-8b \
AGENT2_API=llama3.1-8b \
AGENT3_API=llama3.1-8b \
CANDIDATE_FUSION_STRATEGY=rrf \
FUSED_CANDIDATE_TOP_K=50 \
TOP_K=10 \
NUM_SAMPLES=101 \
TEST_INTERVAL=101 \
FORWARD_WORKERS=1 \
OP_STR=amd-forward-101-top10-base-rrf-v1 \
bash ./ops/run_forward_101_amd.sh
```

成功标准：

```text
HR@10 > 29.70
MRR 接近或超过 20.27
fused_top 召回明显高于 agent1/agent2 union
```

如果 `HR@10 >= 32`，说明 RRF 方向值得继续。

## 8. 仍未完成

1. 完整三 agent LoRA 没有形成稳定超过 base-only 的结果。
2. agent1 LoRA 还没有纳入主实验闭环。
3. RRF 权重还没有调参。
4. 还没有做 base-only vs RRF 的逐样本错因分析。
5. 还没有完全对齐论文的全量设置、模型设置和所有数据集结果。
6. 还没有证明当前 CA 结果能接近论文 `HR@10=39.16`。

## 9. 开新对话时的最短背景

如果开新对话，可以直接给下面这段：

```text
项目是 CoMaPOI0607，分支 jjy，本地路径 F:\jjy\project\CoMaPoi0607，服务器路径 /mnt/workspace/comapoilatest/CoMaPOI0607。目标是在 ModelScope AMD ROCm 上复现/接近 CoMaPOI 论文 CA 结果。已经跑通 base-only、多 agent forward、agent2/agent3 LoRA、逆向训练数据生成和 Qwen3-Embedding-4B RAG。当前最好 base-only 101 是 HR@10=29.70, MRR=20.27；agent2/agent3 LoRA 没超过 base-only。最新提交 803aa0d 增加了 candidate_fusion.py、ops/analyze_candidate_recall.py 和 RRF 融合，下一步应先跑候选召回诊断，再跑 base-only + RRF 101，不建议直接跑 900。
```


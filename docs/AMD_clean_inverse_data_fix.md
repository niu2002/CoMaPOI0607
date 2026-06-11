# AMD 逆向训练数据清洗修复说明

## 修复目标

当前 base-only 好于 agent2/agent3 LoRA 的主要原因不是 LoRA 未生效，而是逆向 SFT 数据存在污染：

1. agent2 训练文件混合画像生成和候选生成，且存在空 assistant。
2. agent2/agent3 assistant 是裸数字列表，不是 forward 推理使用的 JSON schema。
3. 输出中混入 `25`、年份、时间、经纬度等非候选语义数字。
4. agent3 原代码本应写 label，却实际写入了前一个变量，导致 agent3 学到 agent2 候选裸列表。
5. agent3 训练目标和 forward top10 排序目标不一致。

本次修复使新生成的训练数据更接近 forward 推理格式。

## 已实现

1. `inference_inverse_new.py`
   - agent2 训练样本改为严格 JSON：
     - `{"current_profile": "..."}`
     - `{"refined_candidate_from_rag": [...25 ids...]}`
   - agent3 训练样本改为严格 JSON：
     - `{"next_poi_id": [...10 ids...]}`
   - agent3 使用候选并集构造 top10 排序目标，不再误写 agent2 裸列表。
   - 增加候选清洗：
     - 去重。
     - 过滤越界 ID。
     - 过滤明显污染数字，如年份、经纬度片段、非 label 的 `25`。
     - agent2 候选优先限制在 RAG candidates + label。
   - label 不再固定第一，而是在前几位中确定性变化，降低 label-first 过拟合。

2. `ops/audit_agent_training_data.py`
   - 新增训练数据审计脚本。
   - 统计空 assistant、非 JSON、缺 key、长度分布、重复、污染数字等。

3. `ops/prepare_agent_training_data_amd.sh`
   - 传递 `NUM_CANDIDATE` 和 `PROFILE_MAX_TOKENS` 到逆向生成脚本。

4. `ops/serve_agents_amd.sh`
   - 启动 vLLM 后等待 `/v1/models` 中出现期望模型，减少服务未 ready 导致的连接失败。

5. `ops/run_forward_amd.sh`
   - forward 前检查所需 API 名称是否已在 vLLM 中注册。

## 服务器验证建议

先重新生成小规模干净数据，不要直接跑 900：

```bash
cd /mnt/workspace/comapoilatest/CoMaPOI0607
git pull --ff-only origin jjy

MODEL_ROOT=/mnt/workspace/comapoilatest/models \
SERVED_MODEL_NAME=llama3.1-8b \
DATASET=ca \
MODE=train \
PORT=7863 \
INVERSE_WORKERS=1 \
INVERSE_NUM_SAMPLES=100 \
FORCE_AGENT_DATA=1 \
FORCE_CANDIDATES=1 \
bash ./ops/prepare_agent_training_data_amd.sh
```

生成后立刻审计：

```bash
python ./ops/audit_agent_training_data.py --dataset ca
```

重点看：

```text
assistant_empty = 0
assistant_not_json = 0
missing_expected_key = 0
agent2 length_distribution 主要集中在 25
agent3 length_distribution 主要集中在 10
污染数字显著下降
```

审计通过后再训练 agent2 + agent3：

```bash
MODEL_ROOT=/mnt/workspace/comapoilatest/models \
MODEL_NAME=Llama-3.1-8B-Instruct \
DATASET=ca \
OP_STR=amd-agent23-clean100-ts10 \
BATCH_SIZE=16 \
MAX_STEPS=-1 \
AGENT_TEST_SIZE=10 \
FORCE_REPROCESS_AGENT_DATA=1 \
SAVE_FREQ=10 \
SAVE_TOTAL_LIMIT=2 \
bash ./ops/train_agent2_agent3_amd.sh
```

再做 agent2-only、agent3-only、agent2+3 消融评估。

## 仍未解决

1. 逆向生成阶段仍然使用真实 label 来合成监督信号，只是不再把 label 固定第一。
2. RAG@100 召回偏低的问题尚未解决。
3. agent1 仍未纳入干净三 agent LoRA 训练。
4. 还未验证干净数据是否能提升 HR@10/MRR，需要服务器重新跑实验。

# AMD agent2 + agent3 LoRA 执行手册

这份文档补齐实施文档里最关键的工程闭环：先训练 agent2 和 agent3 的 LoRA，再用 base agent1 + LoRA agent2 + LoRA agent3 跑 101 样本。

## 已实现

1. `ops/train_agent2_agent3_amd.sh`
   - 串行训练 agent2 和 agent3。
   - 默认使用 `OP_STR=amd-agent23-ca-v1`、`BATCH_SIZE=16`、`SEQ_LENGTH=2048`、`SAVE_FREQ=50`。
   - 训练完成后写出 `finetune/results/amd-agent23-ca-v1/sft-ca/agent23_paths.env`。
   - 支持断点恢复：`RESUME_AGENT2_FROM_CHECKPOINT=latest`、`RESUME_AGENT3_FROM_CHECKPOINT=latest`。

2. `ops/serve_agent2_agent3_amd.sh`
   - 自动定位 agent2/agent3 adapter。
   - agent1 使用 base model，agent2/agent3 使用 LoRA。
   - 调用现有 `ops/serve_agents_amd.sh` 启动 vLLM OpenAI-compatible 服务。

3. `ops/run_forward_101_agent23_amd.sh`
   - 固定 101 样本、`TOP_K=10`。
   - 默认 `AGENT1_API=llama3.1-8b`、`AGENT2_API=agent2`、`AGENT3_API=agent3`。

4. `ops/analyze_forward_results.py`
   - 可离线分析任意 `poi_predictions.json`。
   - 输出 `HR@1/@3/@5/@10`、`NDCG@1/@3/@5/@10`、`MRR`。
   - 输出预测长度分布、解析状态、命中 rank 分布、候选召回、预测是否来自候选并集。

## 服务器执行顺序

先拉代码：

```bash
cd /mnt/workspace/comapoilatest/CoMaPOI0607
git fetch origin amd
git checkout amd
git pull --ff-only origin amd
```

确认依赖：

```bash
python ./ops/check_multiagent_env.py
python -m py_compile evaluate.py parser_tool.py prompt_provider.py inference_forward_new.py ops/analyze_forward_results.py
bash -n ops/train_agent2_agent3_amd.sh ops/serve_agent2_agent3_amd.sh ops/run_forward_101_agent23_amd.sh
```

构建 101 条候选：

```bash
python ./ops/build_candidates_amd.py \
  --dataset ca \
  --mode test \
  --num_samples 101 \
  --top_k 100 \
  --embedding_model_path /mnt/workspace/comapoilatest/models/Qwen3-Embedding-4B \
  --embedding_batch_size 4
```

训练 agent2 + agent3：

```bash
MODEL_ROOT=/mnt/workspace/comapoilatest/models \
MODEL_NAME=Llama-3.1-8B-Instruct \
DATASET=ca \
OP_STR=amd-agent23-ca-v1 \
BATCH_SIZE=16 \
SAVE_FREQ=50 \
SAVE_TOTAL_LIMIT=3 \
bash ./ops/train_agent2_agent3_amd.sh
```

如果服务器中断，继续训练：

```bash
MODEL_ROOT=/mnt/workspace/comapoilatest/models \
MODEL_NAME=Llama-3.1-8B-Instruct \
DATASET=ca \
OP_STR=amd-agent23-ca-v1 \
BATCH_SIZE=16 \
RESUME_AGENT2_FROM_CHECKPOINT=latest \
RESUME_AGENT3_FROM_CHECKPOINT=latest \
bash ./ops/train_agent2_agent3_amd.sh
```

启动 agent2 + agent3 LoRA 服务：

```bash
source /mnt/workspace/comapoilatest/CoMaPOI0607/finetune/results/amd-agent23-ca-v1/sft-ca/agent23_paths.env

pkill -f "vllm.entrypoints.openai.api_server" || true
MODEL_ROOT=/mnt/workspace/comapoilatest/models \
MODEL_NAME=Llama-3.1-8B-Instruct \
SERVED_MODEL_NAME=llama3.1-8b \
PORT=7863 \
bash ./ops/serve_agent2_agent3_amd.sh

sleep 30
curl -m 5 -sS http://127.0.0.1:7863/v1/models
tail -n 100 /tmp/comapoi-vllm-agents.log
```

101 样本推理：

```bash
DATASET=ca \
MODEL_NAME=Llama-3.1-8B-Instruct \
PORT=7863 \
OP_STR=amd-forward-101-top10-agent23 \
bash ./ops/run_forward_101_agent23_amd.sh
```

离线诊断：

```bash
python ./ops/analyze_forward_results.py \
  --predictions "results/ca/amd-forward-101-top10-agent23/[Forward_Inference_ca_Llama-3.1-8B-Instruct_1_25_agent1_api_llama3.1-8b_agent2_api_agent2_agent3_api_agent3]/poi_predictions.json"
```

## 仍未实现

1. 三 agent 全部 LoRA 的一键训练脚本还没有新增。本轮只优先实现 agent2 + agent3，因为它最可能直接影响候选精炼和最终排序。
2. 自动批量消融实验还没有实现，例如 E0/E1/E2/E3 自动排队、自动汇总对比。
3. RAG query 和 POI 描述增强还没有实现。当前候选召回偏低，如果 agent2+agent3 LoRA 仍提升有限，下一步应该优先改这里。
4. 全流程自动恢复推理还没有实现。训练支持 checkpoint 恢复，推理目前仍建议 101 样本小批量跑。

## 验收口径

agent2 + agent3 这轮不是和旧的 900 条直接比，而是和新的 base-only 101 条比：

```text
base-only 101:
HR@1 16.83
HR@3 21.78
HR@5 24.75
HR@10 29.70
MRR 20.27
```

如果 agent2 + agent3 LoRA 的 `HR@10` 比 base-only 101 提升 3 个百分点以上，说明这条路线有继续投入价值。

# AMD 101样本三Agent LoRA工程实施文档

## 1. 目标

本文档用于把 AMD 分支上的 CoMaPOI 多 agent 流程变成可重复、可对比、能逐步提升效果的工程流程。

固定开发验证口径：

1. 每次先跑 101 个 test 样本。
2. 最终预测列表长度固定为 `TOP_K=10`。
3. 指标固定看 `HR@1`、`HR@3`、`HR@5`、`HR@10`，以及对应的 `NDCG@1/3/5/10` 和 `MRR`。
4. 候选召回单独记录，不和最终预测指标混在一起。
5. 推理必须明确区分 base-only、部分 LoRA、三 LoRA 全量编排三种模式。

这份文档的重点不是只把流程跑通，而是解决当前 900 条实验暴露出来的三个问题：

1. `HR@5/10/20` 一致，因为最终预测只输出了 top5。
2. 当前 900 条是 base-only 多 agent，不是论文设定里的三 agent adapter 协同。
3. 白盒 reasoning 要保留，但需要结构化保存，方便阅读和解析，避免候选列表被解释文字里的数字污染。

## 2. 当前 900条基线结论

本轮 900 条结果可以作为 base-only baseline：

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

关键诊断：

```text
预测长度分布:
5个预测: 884条
4个预测: 12条
3个预测: 2条
2个预测: 2条
超过5个: 0条

命中位置:
rank1: 124条
rank2: 63条
rank3: 31条
rank4: 26条
rank5: 21条
miss: 635条

候选召回:
label in agent1 candidates: 207/900 = 23.00%
label in agent2 candidates: 187/900 = 20.78%
label in either candidates: 209/900 = 23.22%
```

结论：

1. `HR@5/10/20` 一致不是评估函数核心错误，而是输出列表长度只有 5。
2. 当前指标不能代表论文完整多 agent LoRA 效果。
3. 候选阶段召回偏低，是后续提升效果的首要观察点。

## 3. Top-K设置

### 3.1 最终预测列表长度

推荐工程默认：

```text
TOP_K=10
```

理由：

1. POI recommendation 和一般推荐系统里，`@10` 是常用主指标。
2. 当前代码 `inference_forward_new.py` 的 argparse 默认就是 `--top_k 10`。
3. 论文代码文档里更常出现 top10，不需要强行扩到 top20。
4. 只要输出 top10，就能同时计算 `@1/@3/@5/@10`。
5. `TOP_K=10` 比 `TOP_K=20` 更省 token，也更容易让模型保持格式稳定。

因此以后不再用 `TOP_K=5` 做正式效果判断。`TOP_K=5` 只允许用于极短 smoke。

### 3.2 指标口径

正式 101 样本和后续 900 样本统一记录：

```text
HR@1
HR@3
HR@5
HR@10
MRR
NDCG@1
NDCG@3
NDCG@5
NDCG@10
```

当前 `evaluate.py` 已经有 `@1/@5/@10/@20`，但没有 `@3` 和 `NDCG@1`。后续实施时需要修改评估输出，或者新增一个独立评估脚本，按同一份 `poi_predictions.json` 重新计算上述指标。

### 3.3 候选池大小

推荐默认：

```text
RAG候选: 100
agent候选: 25
final预测: 10
```

含义：

1. embedding/RAG 先召回 100 个，尽量保证 label 有机会进入候选池。
2. agent1/agent2 各自压缩成 25 个候选，符合原代码的候选 agent 设定。
3. agent3 最后输出 10 个最终预测，用于 `@1/@3/@5/@10`。

如果 101 样本里 `RAG@100` 召回仍然明显低，需要考虑把 RAG 候选临时扩到 200 做对照，但正式默认先用 100。

## 4. 三Agent LoRA服务模式修正

### 4.1 当前问题

之前启动服务时没有传：

```bash
AGENT1_ADAPTER_PATH
AGENT2_ADAPTER_PATH
AGENT3_ADAPTER_PATH
```

所以 `ops/serve_agents_amd.sh` 进入了 base-only 模式。虽然 agent 编排链路跑通了，但三个 agent 本质上都是同一个未微调 base model。

### 4.2 正式三Agent模式

三 agent 全量 LoRA 模式必须满足：

```bash
AGENT1_ADAPTER_PATH=/path/to/agent1_adapter
AGENT2_ADAPTER_PATH=/path/to/agent2_adapter
AGENT3_ADAPTER_PATH=/path/to/agent3_adapter
AGENT1_API=agent1
AGENT2_API=agent2
AGENT3_API=agent3
```

服务启动后，`curl http://127.0.0.1:7863/v1/models` 应该能看到 base model 和 LoRA 模块可被调用。

### 4.3 部分LoRA模式

为了做消融实验，建议支持部分 LoRA：

1. agent1 base，agent2 LoRA，agent3 LoRA
2. agent1 LoRA，agent2 LoRA，agent3 base
3. agent1 base，agent2 base，agent3 LoRA

其中最值得优先跑的是：

```text
agent2 LoRA + agent3 LoRA
```

原因：

1. agent2 直接影响 RAG 候选精炼。
2. agent3 直接影响最终排序。
3. agent1 是用户长期画像和长期候选，重要但可以先作为第二优先级。

当前 `serve_agents_amd.sh` 只有“三个 adapter 全传才开启 LoRA”的逻辑。后续实施时建议改成“传了哪个 adapter 就挂哪个 LoRA”，并让未传 adapter 的 agent 自动回退 base API。

## 5. 训练方案

### 5.1 三Agent全量训练

训练入口：

```bash
AGENT_TYPE=agent1 ./ops/run_amd_agent.sh
AGENT_TYPE=agent2 ./ops/run_amd_agent.sh
AGENT_TYPE=agent3 ./ops/run_amd_agent.sh
```

推荐稳定参数：

```bash
DATASET=ca
MODEL_ROOT=/mnt/workspace/comapoilatest/models
MODEL_NAME=Llama-3.1-8B-Instruct
DEVICE_MAP=cuda
OP_STR=amd-agent-ca-v1
SEQ_LENGTH=2048
BATCH_SIZE=16
GRAD_ACC=1
MAX_STEPS=-1
NUM_TRAIN_EPOCHS=1
NUM_WORKERS=0
LOG_FREQ=5
SAVE_FREQ=50
SAVE_TOTAL_LIMIT=3
```

说明：

1. 之前 merged LoRA 用 `BATCH_SIZE=24` 跑完 1 epoch，峰值显存约 125GB。
2. 为了稳定和避免 ROCm 突然碎片化，单 agent 训练建议先用 `BATCH_SIZE=16`。
3. 如果显存峰值低于 100GB 且速度太慢，再试 `BATCH_SIZE=20` 或 `24`。
4. 训练过程中保留 checkpoint，服务器断开后可用 `RESUME_FROM_CHECKPOINT=latest` 或手动 checkpoint 路径恢复。

### 5.2 两Agent优先训练

如果想更快验证“真实提升”，可以先训练：

```text
agent2: Forecaster / RAG候选精炼
agent3: Final_Predictor / 最终排序
```

暂时让 agent1 用 base model。

这个路线更适合快速判断：

1. 候选召回是否改善。
2. top10 排序是否改善。
3. prompt 格式收紧后是否能稳定输出。

### 5.3 训练耗时估计

已知数据：

```text
merged LoRA，ca全量，1 epoch，BATCH_SIZE=24:
耗时约 146.9 分钟
峰值显存约 125GB
```

估计：

```text
三Agent串行训练，BATCH_SIZE=16:
约 3.5 到 5 小时

只训练 agent2 + agent3，BATCH_SIZE=16:
约 2.3 到 3.5 小时

如果 BATCH_SIZE=24 稳定:
三Agent约 2.8 到 4 小时
agent2 + agent3约 1.8 到 2.8 小时
```

这些是工程估计，最终以 AMD 机器上的 tokens/sec 和 checkpoint 日志为准。

## 6. 推理方案和耗时估计

### 6.1 101样本默认推理命令

先构建 101 条候选：

```bash
cd /mnt/workspace/comapoilatest/CoMaPOI0607

python ./ops/build_candidates_amd.py \
  --dataset ca \
  --mode test \
  --num_samples 101 \
  --top_k 100 \
  --embedding_model_path /mnt/workspace/comapoilatest/models/Qwen3-Embedding-4B \
  --embedding_batch_size 4
```

三 LoRA 服务启动：

```bash
cd /mnt/workspace/comapoilatest/CoMaPOI0607

pkill -f "vllm.entrypoints.openai.api_server" || true

nohup env \
  MODEL_ROOT=/mnt/workspace/comapoilatest/models \
  MODEL_NAME=Llama-3.1-8B-Instruct \
  SERVED_MODEL_NAME=llama3.1-8b \
  PORT=7863 \
  AGENT1_ADAPTER_PATH=/mnt/workspace/comapoilatest/CoMaPOI0607/finetune/results/amd-agent-ca-v1/sft-ca/AGENT1_SAVE_NAME \
  AGENT2_ADAPTER_PATH=/mnt/workspace/comapoilatest/CoMaPOI0607/finetune/results/amd-agent-ca-v1/sft-ca/AGENT2_SAVE_NAME \
  AGENT3_ADAPTER_PATH=/mnt/workspace/comapoilatest/CoMaPOI0607/finetune/results/amd-agent-ca-v1/sft-ca/AGENT3_SAVE_NAME \
  ./ops/serve_agents_amd.sh >/tmp/serve_agents_amd.log 2>&1 &

sleep 30
curl -m 5 -sS http://127.0.0.1:7863/v1/models
tail -n 80 /tmp/comapoi-vllm-agents.log
```

101 样本推理：

```bash
cd /mnt/workspace/comapoilatest/CoMaPOI0607

DATASET=ca \
MODEL_NAME=Llama-3.1-8B-Instruct \
PORT=7863 \
NUM_SAMPLES=101 \
START_POINT=0 \
TOP_K=10 \
NUM_CANDIDATE=25 \
FORWARD_WORKERS=1 \
TEST_INTERVAL=101 \
AGENT1_API=agent1 \
AGENT2_API=agent2 \
AGENT3_API=agent3 \
AGENT1_MAX_TOKENS=512 \
AGENT2_MAX_TOKENS=512 \
AGENT3_MAX_TOKENS=384 \
TEMPERATURE=0.0 \
TOP_P=1.0 \
OP_STR=amd-forward-101-top10-lora \
./ops/run_forward_amd.sh
```

### 6.2 900样本耗时估计

已知：

```text
base-only，900条，TOP_K=5:
约 1.5 小时
约 6 秒/样本
```

估计：

```text
base-only，101条，TOP_K=10:
约 10 到 15 分钟

三Agent LoRA，101条，TOP_K=10:
约 12 到 20 分钟

三Agent LoRA，900条，TOP_K=10:
约 1.8 到 2.8 小时

agent2 + agent3 LoRA，agent1 base，900条，TOP_K=10:
约 1.7 到 2.5 小时
```

说明：

1. LoRA 推理不是重新加载三份 8B 模型，而是一个 base model 加多个 adapter，所以显存够。
2. 耗时主要来自每个样本多轮 agent 调用和生成 token 数。
3. 如果收紧 profile 输出长度，实际速度可能接近 base-only。
4. 如果 `AGENT1_MAX_TOKENS/AGENT2_MAX_TOKENS` 太大，速度会变慢，但格式更稳。

## 7. 白盒输出结构化

### 7.1 原论文白盒思想要保留

CoMaPOI 的价值不只是输出 POI ID，还要让我们看到：

1. 用户长期画像。
2. 用户短期移动模式。
3. 长期画像候选。
4. RAG 和短期模式候选。
5. 最终排序理由。

所以不能简单删除 reasoning。

### 7.2 当前问题

当前 `reasoning_path` 是一个长字符串，内容类似：

```text
long_term_profile: ..., short_pattern_response: ..., candidate_poi_list_agent1: ..., candidate_poi_list_agent2: ..., final_prediction: ...
```

问题：

1. 阅读困难。
2. 解析困难。
3. 模型解释文字里的年份、时间、用户 ID 可能被候选解析逻辑误当作 POI ID。
4. 结果文件大，后续对比和统计不方便。

### 7.3 推荐保存格式

后续把 `reasoning_path` 从字符串改成结构化对象：

```json
{
  "user_id": "10",
  "label": "123",
  "predicted_poi_ids": ["11", "22", "33", "44", "55", "66", "77", "88", "99", "100"],
  "init_valid_poi_ids": ["..."],
  "reasoning_path": {
    "long_term_profile": {
      "raw": "...",
      "parsed_profile": "..."
    },
    "short_term_profile": {
      "raw": "...",
      "parsed_profile": "..."
    },
    "candidates": {
      "rag_top100": ["..."],
      "agent1_top25": ["..."],
      "agent2_top25": ["..."],
      "candidate_union": ["..."]
    },
    "final_prediction": {
      "raw": "...",
      "parsed_top10": ["..."],
      "parse_status": "ok"
    }
  }
}
```

这样既保留白盒解释，又能稳定做指标和错误分析。

### 7.4 模型输出格式

模型仍然可以生成用户画像和分析，但每个 agent 的可解析字段必须固定：

agent1 profile：

```json
{"historical_profile": "简短长期画像"}
```

agent1 candidates：

```json
{"candidate_poi_list_from_profile": [1, 2, 3]}
```

agent2 profile：

```json
{"current_profile": "简短短期移动画像"}
```

agent2 candidates：

```json
{"refined_candidate_from_rag": [1, 2, 3]}
```

agent3 final：

```json
{"next_poi_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
```

工程原则：

1. 白盒解释存在 profile 字段中。
2. 可评分结果只从固定 JSON key 中提取。
3. 非 JSON 外的解释不参与解析。
4. 候选列表解析失败时要记录 `parse_status`，不要静默混入自然语言数字。

## 8. 真正提升效果的实施顺序

### 阶段A：修正评估口径

目标：

1. `TOP_K=10`。
2. 指标输出 `@1/@3/@5/@10`。
3. 101 样本固定跑法。

成功标准：

```text
101条结果中，大多数 predicted_poi_ids 长度为10
HR@1 <= HR@3 <= HR@5 <= HR@10
NDCG@1 <= NDCG@3 <= NDCG@5 <= NDCG@10
```

### 阶段B：三Agent adapter挂载

目标：

1. `serve_agents_amd.sh` 能挂三 adapter。
2. `run_forward_amd.sh` 明确调用 `agent1/agent2/agent3`。
3. 101 条能跑完。

成功标准：

```text
日志中出现 enable-lora
curl /v1/models 能看到 LoRA model name
101条能正常输出 top10
```

### 阶段C：候选召回诊断

新增每次评估时记录：

```text
RAG@100 label recall
agent1@25 label recall
agent2@25 label recall
agent1_or_agent2@50 label recall
final@10 HR
```

判断逻辑：

1. 如果 `RAG@100` 低，先改 embedding query 和 POI 描述。
2. 如果 `RAG@100` 高但 `agent2@25` 低，优先训练或修正 agent2。
3. 如果候选召回高但 final@10 低，优先训练 agent3。
4. 如果 final@1 低但 final@10 高，说明排序弱，不是召回弱。

### 阶段D：结构化reasoning

目标：

1. 保留白盒解释。
2. 结果 JSON 变清晰。
3. 候选和预测解析更稳。

成功标准：

```text
reasoning_path 是 dict
raw 与 parsed 分开保存
parse_status 可统计
候选列表不再混入年份和时间
```

### 阶段E：101样本对照实验

每轮只跑 101 条，按这个顺序比较：

```text
E0: base-only + TOP_K=10
E1: agent3 LoRA only
E2: agent2 LoRA + agent3 LoRA
E3: agent1 LoRA + agent2 LoRA + agent3 LoRA
```

优先级：

1. E0 用来确认评估口径。
2. E2 用来判断最快提升路径。
3. E3 才是完整论文路线。

如果 E2 明显提升，说明 agent2/agent3 是主增益来源。

如果 E2 不升但 E3 升，说明长期画像 agent1 作用明显。

如果 E3 也不升，优先查训练数据格式、候选召回和 prompt/解析。

## 9. 需要修改的代码清单

### 9.1 必改

1. `evaluate.py`
   - 增加 `HR@3`
   - 增加 `NDCG@1`
   - 增加 `NDCG@3`
   - 默认输出 `@1/@3/@5/@10`

2. `ops/run_forward_amd.sh`
   - 默认 `TOP_K` 从 5 改为 10
   - 101 样本推荐命令写入 run example

3. `ops/serve_agents_amd.sh`
   - 支持部分 LoRA adapter
   - 明确打印每个 agent 使用的是 base 还是 LoRA

4. `inference_forward_new.py`
   - `reasoning_path` 从字符串改为结构化 dict
   - 保存 RAG candidates
   - 保存 parsed candidates 和 parse_status

### 9.2 强烈建议

1. 增加一个 candidate recall 统计脚本。
2. 增加一个 `ops/run_forward_101_amd.sh`，固定 101 样本参数。
3. 增加输出格式检查：每条预测必须长度为 10，否则记录 warning。
4. 增加 `results/.../diagnostics.json`，保存候选召回、预测长度分布、解析失败率。

## 10. 推荐验收标准

101 样本不是最终论文指标，但足够做开发判断。

一次 101 样本实验合格需要满足：

```text
predicted_poi_ids长度为10的比例 >= 95%
parse_status=ok比例 >= 95%
没有 APIConnectionError
没有 KeyError user_id
没有 BrokenProcessPool
候选召回指标被输出
HR@1/@3/@5/@10 被输出
```

效果提升判断：

```text
E2 比 E0 的 HR@10 提升 >= 3个百分点，认为 agent2+agent3 有工程价值
E3 比 E2 的 HR@10 继续提升 >= 1个百分点，认为 agent1 值得纳入正式全流程
如果 HR@10 提升但 HR@1 不升，下一步重点优化排序
如果候选召回不升，下一步重点优化 RAG/agent2
```

## 11. 最终推荐路线

短期先做：

1. 把正式评估改成 `TOP_K=10` 和 `@1/@3/@5/@10`。
2. 固定 101 样本跑法。
3. 支持三 agent adapter path 和部分 adapter path。
4. 结构化保存 reasoning。
5. 先跑 `agent2 + agent3 LoRA`。
6. 再跑三 agent LoRA。

不要马上再跑 900 条。

理由：

1. 当前 900 条已经证明链路能跑，但不是完整论文路线。
2. 101 条足够暴露格式、服务、adapter、候选召回的问题。
3. 等 101 条确认有提升，再跑 900 条才划算。

## 12. 当前实现状态

已实现：

1. `evaluate.py` 已改为默认输出 `HR/NDCG@1,@3,@5,@10`，并保留额外 `top_k` 的兼容输出。
2. `ops/run_forward_amd.sh` 默认 `TOP_K=10`，并新增 `PROFILE_MAX_TOKENS`。
3. 新增 `ops/run_forward_101_amd.sh`，固定 101 样本、top10、单 worker、101 条保存一次。
4. `ops/serve_agents_amd.sh` 已支持部分 LoRA adapter，传哪个 agent adapter 就挂哪个，未传的 agent 回退 base model。
5. `prompt_provider.py` 已收紧白盒 profile 输出，默认 profile 目标长度为 220 tokens，保留行为画像但减少大段散文。
6. `inference_forward_new.py` 已把 `reasoning_path` 改为结构化 dict，保存 raw、parsed、RAG 候选、agent 候选、final parse status。
7. `inference_forward_new.py` 已新增 `diagnostics.json` 和 interim diagnostics，记录预测长度分布、parse status、候选召回、预测是否来自候选 union。
8. `parser_tool.py` 已减少把占位符或自然语言编号误解析成 POI ID 的风险。

未实现或仍需服务器验证：

1. 三个 agent 的 LoRA adapter 还需要在服务器上训练得到真实路径。
2. vLLM 在 ROCm 上同时挂部分或全部 LoRA adapter 的稳定性需要服务器实测确认。
3. 当前未强制 constrained decoding，只是 prompt 与解析更严格；如果 base model 仍输出非 JSON，需要继续加重试或服务端 schema 约束。
4. 当前 diagnostics 只统计候选召回，不会自动调参；RAG query、POI 描述、候选数量仍需根据 101 样本结果迭代。
5. `ops/run_forward_101_amd.sh` 在 Linux 上建议用 `bash ops/run_forward_101_amd.sh` 执行，若要 `./ops/run_forward_101_amd.sh` 直接运行，需要确认 git executable bit。

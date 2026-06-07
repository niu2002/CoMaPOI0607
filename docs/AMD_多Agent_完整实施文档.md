# AMD 多 Agent 完整实施文档

## 1. 目标

本次目标不是先追论文最终指标，而是先在 AMD 机器上尽快打通下面这条主流程：

1. 准备多 Agent 所需的数据目录和 `poi_info.csv`
2. 下载真实 embedding 模型 `Qwen/Qwen3-Embedding-4B`
3. 生成 candidate cache
4. 启动阶段 4 的主方案服务：
   - 1 份 base model
   - 3 个 Agent API 名字
   - 优先尝试多 LoRA 挂载
5. 跑一个 10 样本 forward inference 测试

当前阶段重点是：

1. **先验证流程通不通**
2. **再追求三 Agent 分别训练后的最终质量**

## 2. 当前方案的原则

### 2.1 关于显存

这张 AMD 卡显存足够做：

1. 三个 Agent 串行训练
2. 一份基座 + 三个 LoRA adapter 编排推理

真正的风险点不是显存，而是：

1. `vLLM + LoRA + ROCm`
2. `AgentScope + OpenAI-compatible API`

所以这次实施采用：

1. **主方案**
   - 单服务多 LoRA
2. **后备方案**
   - 如果主方案不稳定，就先用同一个 base model 名称把三 Agent 编排链路跑通

## 3. 关于 embedding：是否需要 API

本次方案默认：

1. **不使用远程 embedding API**
2. 直接在服务器上下载 `Qwen/Qwen3-Embedding-4B`
3. 本地生成 embedding
4. 本地生成 candidate cache

只有当本地 embedding 模型无法正常运行时，才退到 API 方案。

也就是说：

1. 当前默认路线 **不需要 API**
2. 但需要在服务器上下载一个 embedding 模型

## 4. 本次已经补好的脚本

本次在 `amd` 分支上，已经准备了这些脚本：

1. `ops/download_qwen_embedding_modelscope.sh`
   - 从 ModelScope 下载 `Qwen/Qwen3-Embedding-4B`
2. `ops/prepare_multiagent_assets.py`
   - 把现有扁平 JSONL 整理成 multi-agent 代码能读的目录结构
   - 自动构建 `poi_info.csv`
3. `ops/build_candidates_amd.py`
   - 用本地 embedding 模型生成 candidate cache
4. `ops/serve_agents_amd.sh`
   - 启动主方案服务
   - 支持 base-only 和 multi-LoRA 两种模式
5. `ops/run_forward_amd.sh`
   - 跑 `inference_forward_new.py`
6. `ops/run_amd_agent.sh`
   - 为后续 agent1 / agent2 / agent3 单独训练做准备

## 5. 服务器执行顺序

以下命令假设你的服务器仓库目录是：

```bash
/mnt/workspace/comapoilatest/CoMaPOI0607
```

### 步骤 0：先做环境自检

```bash
cd /mnt/workspace/comapoilatest/CoMaPOI0607
python ./ops/check_multiagent_env.py
```

如果缺的是 `modelscope` 或 `agentscope`，优先补：

```bash
python -m pip install -U modelscope agentscope
```

如果缺的是 `vllm`，说明主方案服务还跑不起来，这时先不要直接进入步骤 5。

### 步骤 1：拉取最新 amd 分支

```bash
cd /mnt/workspace/comapoilatest/CoMaPOI0607
git pull --ff-only origin amd
```

### 步骤 2：下载 Qwen3 embedding 4B

如果服务器上已经装好 `modelscope`，直接执行：

```bash
cd /mnt/workspace/comapoilatest/CoMaPOI0607
TARGET_DIR=/mnt/workspace/comapoilatest/models/Qwen3-Embedding-4B \
./ops/download_qwen_embedding_modelscope.sh
```

如果 `modelscope` 还没装，先执行：

```bash
python -m pip install -U modelscope
```

再执行下载脚本。

### 步骤 3：准备 multi-agent 数据资产

这一步会做两件事：

1. 建立 `dataset_all/ca/train/ca_train.jsonl`
2. 建立 `dataset_all/ca/test/ca_test.jsonl`
3. 自动从 JSONL 里抽取并生成 `dataset_all/ca/ca_poi_info.csv`

执行：

```bash
cd /mnt/workspace/comapoilatest/CoMaPOI0607
python ./ops/prepare_multiagent_assets.py --dataset ca
```

### 步骤 4：先生成 10 样本 candidate cache

这一步先只做 `test` split 的前 10 条，目的是验证流程。

执行：

```bash
cd /mnt/workspace/comapoilatest/CoMaPOI0607
python ./ops/build_candidates_amd.py \
  --dataset ca \
  --mode test \
  --num_samples 10 \
  --top_k 100 \
  --embedding_model_path /mnt/workspace/comapoilatest/models/Qwen3-Embedding-4B \
  --embedding_batch_size 4
```

生成结果应该在：

```bash
dataset_all/ca/test/ca_test_candidates.jsonl
```

### 步骤 5：启动阶段 4 主方案服务

#### 方案 A：如果你已经有三个 Agent adapter

如果已经训练好了三个 adapter，可以这样起服务：

```bash
cd /mnt/workspace/comapoilatest/CoMaPOI0607
MODEL_ROOT=/mnt/workspace/comapoilatest/models \
MODEL_NAME=Llama-3.1-8B-Instruct \
PORT=7863 \
AGENT1_ADAPTER_PATH=/mnt/workspace/comapoilatest/CoMaPOI0607/finetune/results/agent1/... \
AGENT2_ADAPTER_PATH=/mnt/workspace/comapoilatest/CoMaPOI0607/finetune/results/agent2/... \
AGENT3_ADAPTER_PATH=/mnt/workspace/comapoilatest/CoMaPOI0607/finetune/results/agent3/... \
./ops/serve_agents_amd.sh
```

这是真正的主方案：

1. 一份基座
2. 三个 LoRA adapter
3. API 名字是：
   - `agent1`
   - `agent2`
   - `agent3`

#### 方案 B：如果还没有三个 Agent adapter

先用同一个 base model 冒烟测试三 Agent 编排链路：

```bash
cd /mnt/workspace/comapoilatest/CoMaPOI0607
MODEL_ROOT=/mnt/workspace/comapoilatest/models \
MODEL_NAME=Llama-3.1-8B-Instruct \
SERVED_MODEL_NAME=llama3.1-8b \
PORT=7863 \
./ops/serve_agents_amd.sh
```

这个模式下没有挂载 LoRA，但可以先验证：

1. 服务是否起来
2. `inference_forward_new.py` 是否能走完整个 multi-agent 编排

### 步骤 6：跑 10 样本 forward inference 冒烟测试

#### 如果你走的是 base-only 冒烟模式

执行：

```bash
cd /mnt/workspace/comapoilatest/CoMaPOI0607
DATASET=ca \
MODEL_NAME=Llama-3.1-8B-Instruct \
PORT=7863 \
NUM_SAMPLES=10 \
FORWARD_WORKERS=1 \
BASE_API_NAME=llama3.1-8b \
AGENT1_API=llama3.1-8b \
AGENT2_API=llama3.1-8b \
AGENT3_API=llama3.1-8b \
OP_STR=amd-forward-smoke \
./ops/run_forward_amd.sh
```

#### 如果你已经挂上了三个 Agent adapter

执行：

```bash
cd /mnt/workspace/comapoilatest/CoMaPOI0607
DATASET=ca \
MODEL_NAME=Llama-3.1-8B-Instruct \
PORT=7863 \
NUM_SAMPLES=10 \
FORWARD_WORKERS=1 \
AGENT1_API=agent1 \
AGENT2_API=agent2 \
AGENT3_API=agent3 \
OP_STR=amd-forward-smoke \
./ops/run_forward_amd.sh
```

## 6. 10 样本测试的成功标准

本次 10 样本测试主要看“流程”，不是先看指标。

成功标准是：

1. `serve_agents_amd.sh` 能把服务起起来
2. `build_candidates_amd.py` 能生成 candidate 文件
3. `run_forward_amd.sh` 能跑完 10 个样本
4. 最终能产出：
   - `results/.../poi_predictions.json`
   - `results/.../metrics.txt`
   - `results/.../metrics.csv`

只要这四件事成立，就说明：

1. 多 Agent 路线的主链路已经打通
2. 后续只需要继续替换为真正的三 Agent adapter 和更高质量 embedding / prompts

## 7. 如果 10 样本测试失败，优先检查什么

### 7.1 服务没起来

先看：

```bash
tail -n 80 /tmp/comapoi-vllm-agents.log
```

### 7.2 candidate 构建失败

优先看：

1. `Qwen3-Embedding-4B` 是否下载成功
2. `dataset_all/ca/ca_poi_info.csv` 是否存在
3. `dataset_all/ca/test/ca_test.jsonl` 是否存在

### 7.3 forward inference 报 user_id / 路径错误

这通常说明：

1. structured dataset 目录没准备好
2. 或者 candidate cache 没生成

先重新执行：

```bash
python ./ops/prepare_multiagent_assets.py --dataset ca
python ./ops/build_candidates_amd.py --dataset ca --mode test --num_samples 10 --top_k 100 --embedding_model_path /mnt/workspace/comapoilatest/models/Qwen3-Embedding-4B
```

## 8. 后续正式进入三 Agent 训练时怎么做

当 10 样本链路跑通以后，再进入真正训练阶段。

训练入口脚本已经准备好：

```bash
AGENT_TYPE=agent1 ./ops/run_amd_agent.sh
AGENT_TYPE=agent2 ./ops/run_amd_agent.sh
AGENT_TYPE=agent3 ./ops/run_amd_agent.sh
```

后续正式训练时，我们再单独定：

1. 每个 Agent 的 batch
2. 每个 Agent 的 step / epoch
3. 输出目录命名
4. checkpoint 恢复策略

## 9. 本次建议

本次最稳的执行策略是：

1. 先下载 Qwen embedding 4B
2. 先准备数据目录和 `poi_info.csv`
3. 先构建 10 条 candidate cache
4. 先起 base-only 服务，验证编排链路
5. 10 样本 forward inference 跑通后
6. 再进入真正的三 Agent adapter 训练和挂载

这样做的原因是：

1. 更容易一次成功
2. 出问题时更容易定位
3. 不会把“模型质量问题”和“工程链路问题”混在一起

## 10. 当前默认结论

1. 这张 AMD 卡 **够**
2. 当前默认 embedding 方案 **不需要 API**
3. 本次优先尝试阶段 4 主方案
4. 但第一次 10 样本测试建议先允许 base-only 编排冒烟，降低失败概率

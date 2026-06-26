# CoMaPOI AMD 远程服务器部署与实验运行手册

本项目目前在 **DSW AMD GPU (ROCm) 远程算力服务器** 上运行。
开发及运行不依赖 conda 环境（直接在系统默认 Python 环境下运行）。

---

## 一、 实验现状

1.  **已完成阶段**：
    *   完成了 **无 RAG 状态下的前向推理评测** (CA 数据集，N=900)。
    *   此时：Agent 1 (Profiler) 和 Agent 2 (Forecaster) 调用云端大模型 API；Agent 3 (Predictor) 为 Base-Only 模型，调用本地部署 of vLLM 服务。
    *   无 RAG 评测结果：`HR@10 = 41.67%`，`MRR = 23.78%`，`NDCG@10 = 28.03%`。

2.  **当前进行阶段**：
    *   准备运行 **不微调的 ExpertRAG 评测** (CA 数据集，N=900)。
    *   大模型基座仍旧保持 Base-Only，但在检索端使用我方升级优化后的 **ExpertRAG 策略**（集成了 HSID 语义匹配、多尺度网格转移统计、强地理空间去噪与 RRF 融合）。

---

## 二、 核心部署与运行指令

请严格按照以下步骤在 AMD 服务器终端中执行。

### 步骤 1. 启动 Base-Only 的 vLLM 服务
在未开启 conda 的系统环境中直接运行，加载 `/mnt/workspace/models/Qwen/Qwen3-8B` 基座：

```bash
# 进入项目根目录
cd /mnt/workspace/CoMaPOI0607

# 限制显存占比为 0.78 并将服务命名为 qwen3-8b
MODEL_ROOT=/mnt/workspace/models \
MODEL_NAME=Qwen/Qwen3-8B \
SERVED_MODEL_NAME=qwen3-8b \
GPU_MEMORY_UTILIZATION=0.78 \
bash ops/serve_agents_amd.sh
```
*   **监控服务启动状态**：`tail -100f /tmp/comapoi-vllm-agents.log`
*   **服务 Ready 标志**：日志中输出 `ready models` 信息，或者运行 `curl http://localhost:7863/v1/models` 有返回。

### 步骤 2. 生成 900 样本的 ExpertRAG 候选池
在运行前向推理前，需通过阿里云百炼的 Embedding 接口和本地 POI 元数据构建召回候选池：

```bash
python ops/build_candidates_amd.py \
  --dataset ca \
  --mode test \
  --num_samples 900 \
  --strategy expertrag \
  --use_hsid \
  --use_cloud_embedding \
  --embedding_api_key "sk-6cfeba9834fd460cbe856c99be17aa74"
```
*   *运行结束后，将在 `dataset_all/ca/test/` 自动生成名为 `ca_test_candidates_expertrag_hsid.jsonl` 的文件。*

### 步骤 3. 运行前向推理评测
当 vLLM 服务已 ready 且候选池已生成完毕，即可拉起 900 样本的前向推理服务：

```bash
python3 inference_forward_new.py --dataset ca --num_samples 900 --batch_size 4 --candidate_fusion_strategy rrf --use_hsid \
  --strategy expertrag \
  --agent1_base_url "https://dashscope.aliyuncs.com/compatible-mode/v1" --agent1_api_key "sk-6cfeba9834fd460cbe856c99be17aa74" --agent1_api "qwen-plus" \
  --agent2_base_url "https://dashscope.aliyuncs.com/compatible-mode/v1" --agent2_api_key "sk-6cfeba9834fd460cbe856c99be17aa74" --agent2_api "qwen-plus" \
  --agent3_base_url "http://localhost:7863/v1" --agent3_api_key "EMPTY" --agent3_api "qwen3-8b" \
  --profile_max_tokens 120 \
  --agent3_max_tokens 1024 \
  --test_interval 50 \
  --save_name amd_rag_lora --store_save_name
```
*   *结果将自动保存并进行指标输出，您可以对比无 RAG 状态下的指标（HR@10 = 41.67%），观察 Recall 和 NDCG 的增幅！*

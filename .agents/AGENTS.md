# CoMaPOI AMD 开发与实验规则 (Project-Scoped Rules)

## 1. 项目背景与目标
* **核心框架**：基于 [README.md](file:///d:/project/jjycomapoi/README.md) 框架（协同多 Agent 下一步 POI 预测）。
* **任务目标**：在 CoMaPOI 基础上加入创新点并进行实验。
* **参考资料**：
  * 原论文核心整理：`rawpaper/paper_summary.md`（若有）
  * 创新点与实验方案：`rawpaper/创新点.md`（含语义 ID/HSID 与 ExpertRAG，具体修改以实际情况为准）。

## 2. 远程算力服务器环境
* **服务器项目路径**：`/mnt/workspace/CoMaPOI0607`
* **基础模型路径**：`/mnt/workspace/models/Qwen/Qwen3-8B` (Jupyter 侧显示为 `models/Qwen/Qwen3-8B`)
* **Embedding 模型路径**：`/mnt/workspace/models/Qwen3-Embedding-4B`
* **运行环境**：ModelScope AMD GPU / ROCm / vLLM OpenAI-compatible server (服务器直接在系统 Python 环境运行，未使用 conda)

## 3. 协作与开发流程 (良性循环)
* **流程**：`Antigravity` 本地轻量修改/测试 -> Git 提交并推送 -> 用户在远程算力服务器 Pull -> 运行 `Antigravity` 提供的 smoke 脚本 -> 跑通后进行 900 样本 (CA 数据集) 评测效果 -> 用户反馈结果 -> `Antigravity` 迭代。

## 4. 当前运行主流程与关键脚本
* **Agent 配置**：
  * **Agent 1** (Profiler): Base model (使用云端 Dashscope `qwen-plus`)
  * **Agent 2** (Forecaster): Base model (使用云端 Dashscope `qwen-plus`)
  * **Agent 3** (Predictor): Base model (使用本地 vLLM 起的 `qwen3-8b`)
* **目前运行阶段与现状**：
  * 已完成 **无 RAG** 前向推理 (HR@10 = 41.67%)。
  * 当前准备跑 **不微调的 ExpertRAG**（即 Agent 3 保持 Base-only，但检索端升级为我方优化后的多专家 Recall RAG 策略，提取包含 HSID 的 900 样本候选池）。
* **融合与评估**：RRF 多源融合 (`CANDIDATE_FUSION_STRATEGY=rrf`, `FUSED_CANDIDATE_TOP_K=50`)，评估指标为 Top-10，已补齐 900 条 test candidates。

## 5. 远程服务器 vLLM 部署与运行指令
为了方便在服务器重启或重新实验时快速启动，请严格执行以下指令。

### 5.1 启动 Base-Only 的 vLLM 模型服务
在服务器未配置 conda 的系统 Python 环境中直接运行：
```bash
# 进入项目目录
cd /mnt/workspace/CoMaPOI0607

# 限制显存并以 qwen3-8b 的名称提供服务
MODEL_ROOT=/mnt/workspace/models \
MODEL_NAME=Qwen/Qwen3-8B \
SERVED_MODEL_NAME=qwen3-8b \
GPU_MEMORY_UTILIZATION=0.78 \
bash ops/serve_agents_amd.sh
```
*   *监控启动状态命令*：`tail -100f /tmp/comapoi-vllm-agents.log`

### 5.2 生成 900 样本的 ExpertRAG 候选池
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

### 5.3 运行前向推理 (900样本测试)
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

## 6. 创新点 1 (HSID) 实验规范与实操要点
* **HSID 配置参数**：RAG 检索及前向推理阶段均支持 `--use_hsid` 开关。开启后程序会自动定位至带有 `_hsid` 的隔离文件并丰富 Prompt 为带 Category/Location/HSID 的 JSON 结构。

## 7. 预置向量复用、命名规范与模型名称定死
* **零 Token 向量复用逻辑**：在 RAG 中，如果本地存在 `.npy` 预提取特征库文件，无论本地还是远程，程序都会直接采用 `np.load` 秒级载入，若是远程有 FAISS（`HAS_FAISS=True`）则在此基础上零 Token 开销秒级自适应建立 FAISS 索引存为 `.bin` 格式。避免重复请求 API 浪费 Token 和时间。
* **云端 API 统一模型定死**：
  * 嵌入模型（Embedding Model）：`text-embedding-v3`
  * 推理模型（Profiler/Forecaster/Predictor）：`qwen-plus`
* **AMD 本地部署模型定死**：
  * **Agent 1 (Profiler)**：`Qwen3-7B-Instruct` (本地 Base-only)
  * **Agent 2 (Forecaster)**：`Qwen3-7B-Instruct` (本地 Base-only)
  * **Agent 3 (Predictor)**：`Qwen3-7B-Instruct` 基座
  * **本地 Rerank 模式**：不开启（即 `use_reranker=False`）。


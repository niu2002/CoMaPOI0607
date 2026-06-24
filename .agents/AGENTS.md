# CoMaPOI AMD 开发与实验规则 (Project-Scoped Rules)

## 1. 项目背景与目标
* **核心框架**：基于 [README.md](file:///d:/project/jjycomapoi/README.md) 框架（协同多 Agent 下一步 POI 预测）。
* **任务目标**：在 CoMaPOI 基础上加入创新点并进行实验。
* **参考资料**：
  * 原论文核心整理：`rawpaper/paper_summary.md`（若有）
  * 创新点与实验方案：`rawpaper/创新点.md`（含语义 ID/HSID 与 ExpertRAG，具体修改以实际情况为准）。

## 2. 远程算力服务器环境
* **服务器项目路径**：`/mnt/workspace/CoMaPOI0607`
* **基础模型路径**：`/mnt/workspace/models/Llama-3.1-8B-Instruct`
* **Embedding 模型路径**：`/mnt/workspace/models/Qwen3-Embedding-4B`
* **运行环境**：ModelScope AMD GPU / ROCm / vLLM OpenAI-compatible server

## 3. 协作与开发流程 (良性循环)
* **流程**：`Antigravity` 本地轻量修改/测试 -> Git 提交并推送 -> 用户在远程算力服务器 Pull -> 运行 `Antigravity` 提供的 smoke 脚本 -> 跑通后进行 900 样本 (CA 数据集) 评测效果 -> 用户反馈结果 -> `Antigravity` 迭代。

## 4. 当前运行主流程与关键脚本
* **Agent 配置**：
  * **Agent 1** (Profiler): Base model
  * **Agent 2** (Forecaster): Base model (不使用微调)
  * **Agent 3** (Predictor): LoRA 微调（300 条逆向推理数据，逆向生成方式为 `paper_label_first_fused`）
* **融合与评估**：RRF 多源融合 (`CANDIDATE_FUSION_STRATEGY=rrf`, `FUSED_CANDIDATE_TOP_K=50`)，评估指标为 Top-10，已补齐 900 条 test candidates。
* **现有关键脚本**：
  * 训练 Agent 3: `ops/train_agent3_amd.sh`
  * 部署 Agent 3: `ops/serve_agent3_amd.sh`
  * 评估前向推理: `ops/run_forward_101_agent3_amd.sh`
* **效果记录**：
  * 目前最新最好效果配置：`amd-forward-900-top10-agent3-300-fused-fullrag`。
  * `agent2-only` (HR@10 = 26.73) 与 `agent2+3` (HR@10 = 25.74) 效果不佳，亟需通过创新点优化。

## 5. 创新点 1 (HSID) 实验规范与实操要点
* **HSID 配置参数**：RAG 检索及前向推理阶段均支持 `--use_hsid` 开关。开启后程序会自动定位至带有 `_hsid` 的隔离文件并丰富 Prompt 为带 Category/Location/HSID 的 JSON 结构。
* **vLLM 显存调优**：为避免显存被 vLLM 完全锁定阻碍 RAG Embedding 的生成，启动大模型服务时应传递 `GPU_MEMORY_UTILIZATION=0.78` 将显存上限压在 150G 左右（基于 192G 显存）。

## 6. 维护与开发规范
* **规划与实施**：每次有重大调整或难度较大的任务，必须先写实施文档/方案规划，并整理出明确的 To-Do List，严格按照 To-Do List 逐步实施。
* **规则维护**：每次修改本项目规则时应简明扼要，仅保留关键增量信息，防止上下文过长。

## 7. 预置向量复用、命名规范与模型名称定死
* **零 Token 向量复用逻辑**：在 RAG 中，如果本地存在 `.npy` 预提取特征库文件，无论本地还是远程，程序都会直接采用 `np.load` 秒级载入，若是远程有 FAISS（`HAS_FAISS=True`）则在此基础上零 Token 开销秒级自适应建立 FAISS 索引存为 `.bin` 格式。避免重复请求 API 浪费 Token 和时间。
* **模型/接口命名规范**：
  * 本地 Embedding 命名：`local_qwen3_embedding`（对应路径：`models/Qwen3-Embedding-4B`）。
  * API Embedding 命名：`cloud_qwen_embedding`（对应百炼 `text-embedding-v3`）。
* **云端 API 统一模型定死**：
  * 嵌入模型（Embedding Model）：`text-embedding-v3`
  * 重排模型（Reranker Model）：`qwen3-rerank`
  * 推理模型（Profiler/Forecaster/Predictor）：`qwen-plus`
* **AMD 本地部署模型定死**：
  * **Agent 1 (Profiler)**：`Qwen3-7B-Instruct` (本地 Base-only)
  * **Agent 2 (Forecaster)**：`Qwen3-7B-Instruct` (本地 Base-only)
  * **Agent 3 (Predictor)**：`Qwen3-7B-Instruct` 基座，附加基于 `paper_label_first_fused` 逆向微调 of LoRA 权重适配器。
  * **本地 Embedding 模型**：`Qwen3-Embedding-4B`
  * **本地 Rerank 模式**：不开启（即 `use_reranker=False`）。

# 🚀 CoMaPOI / GeoSemID 远程算力机实验交接与复现指南 (ModelScope Handover Guide)

本文档专为在新的算力机器（如 ModelScope AMD/NVIDIA GPU 环境）上复现并跑完剩余两个数据集（**NYC** 与 **TKY**）的主实验而编写。文档涵盖了代码同步、当前进度归档、环境拉起、候选池构建、推理运行及指标回填全流程。

---

## 1. 当前研究进度与成果总结

目前，**CA 数据集的主实验与消融实验已全部完成并闭环**。核心结论与评估指标已正式收录在中英文论文手稿（`paper_writing/chinese/GeoSemID_TITS_zh.tex` 与 `paper_writing/english/GeoSemID_TITS.tex`）中：

### 1.1 CA 数据集已完成实验指标 (HR@10 & MRR)

| 实验版本 | 样本数 | HR@1 | HR@3 | HR@5 | **HR@10** | **MRR** | **NDCG@10** | 核心说明 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **无 RAG 基线 (Baseline)** | 900 | 16.00% | 29.00% | 33.67% | **41.67%** | 23.78% | 28.03% | Agent 1 + Agent 2 RRF 融合，无 RAG 注入，截断较多 |
| **未微调 ExpertRAG** | 900 | 12.33% | 26.78% | 33.67% | **40.00%** | 20.99% | 25.56% | 引入带 HSID 的多专家候选，生成正常率 91.8% |
| **微调完整模型 (Full GeoSemID)** | **1050** | **12.57%** | **27.33%** | **33.71%** | **42.48%** | **21.54%** | **26.53%** | **论文正式采纳的 Ours 主指标**，模型具备高判别力 |

### 1.2 空间物理地理误差 (Haversine Distance)
- **平均距离误差 (Mean Distance Error, MDE)**: **38.417 km**
- **中位数距离误差 (Median Distance Error)**: **5.889 km**
- **空间地理召回率**: `@1.0km`: 30.76% | `@3.0km`: 39.81% | `@5.0km`: 47.24%
- **模型合规与无幻觉率**: 格式错误率 **0.00%**，越界候选推荐率 **0.00%**（100% 严格落在多源召回池内）。

### 1.3 剩余任务
在相同框架下跑完 **NYC** 与 **TKY** 两个数据集的候选召回与前向推理，替换论文中当前的 `\mock{...}` 占位符。

---

## 2. 源代码拉取与环境准备

### 2.1 Git 仓库克隆与同步

主项目代码库地址为：`https://github.com/niu2002/CoMaPOI0607.git`
开发分支为：`fromjjy0624-lq`

在新的算力机终端执行：

```bash
# 1. 克隆代码库并检出当前工作分支
git clone -b fromjjy0624-lq https://github.com/niu2002/CoMaPOI0607.git /mnt/workspace/CoMaPOI0607
cd /mnt/workspace/CoMaPOI0607

# 2. 如果之前已有目录，拉取最新代码与文档
git checkout fromjjy0624-lq
git pull origin fromjjy0624-lq
```

### 2.2 目录与大模型权重确认
确认远程算力服务器（ModelScope 实例）系统环境与模型挂载路径：
- **项目目录**：`/mnt/workspace/CoMaPOI0607`
- **本地 LLM 模型目录**：`/mnt/workspace/models/Qwen/Qwen3-8B`（对应本地部署的 Agent 3）
- **本地 Embedding 目录（可选备用）**：`/mnt/workspace/models/Qwen3-Embedding-4B`
- **Python 环境**：直接使用系统 Python 环境即可，或使用已预装 vLLM、PyTorch 的环境。

---

## 3. 运行环境启动：拉起 vLLM 模型服务 (Agent 3)

Agent 1 (Profiler) 和 Agent 2 (Forecaster) 采用阿里云 DashScope 云端 API (`qwen-plus`)；Agent 3 (Predictor) 采用本地 vLLM 拉起 `Qwen3-8B` 提供兼容 OpenAI 的本地服务。

### 3.1 启动 vLLM 服务
在终端中执行启动脚本（已优化显存参数防止 AMD/ROCm OOM）：

```bash
cd /mnt/workspace/CoMaPOI0607

# 限制显存利用率 (0.78~0.85)，以 qwen3-8b 名称提供本地服务 (默认监听 7863 端口)
MODEL_ROOT=/mnt/workspace/models \
MODEL_NAME=Qwen/Qwen3-8B \
SERVED_MODEL_NAME=qwen3-8b \
GPU_MEMORY_UTILIZATION=0.78 \
bash ops/serve_agents_amd.sh
```

### 3.2 监控服务拉起状态
```bash
# 查看实时启动日志
tail -100f /tmp/comapoi-vllm-agents.log
```
当日志出现 `Application startup complete` 或 `Uvicorn running on http://127.0.0.1:7863` 时，说明服务就绪。

### 3.3 验证健康检查接口
```bash
curl http://localhost:7863/v1/models
```
返回包含 `{"id": "qwen3-8b", ...}` 的 JSON 即表示拉起成功。

> **提示**：如需停止或重启 vLLM，可执行：`pkill -f vllm` 或 `kill -9 $(lsof -t -i:7863)`。

---

## 4. 构建候选池：ExpertRAG + HSID 离线召回

在进行前向推理前，需要通过多专家 RAG 检索模型，为测试样本构建候选 POI 集合（含地理空间、类别转移与 HSID 语义编码）。

### 4.1 NYC 数据集候选池构建
NYC 测试集共有 988 条，我们构建 900 条（或全量 988 条）：

```bash
cd /mnt/workspace/CoMaPOI0607

python ops/build_candidates_amd.py \
  --dataset nyc \
  --mode test \
  --num_samples 900 \
  --strategy expertrag \
  --use_hsid \
  --use_cloud_embedding \
  --embedding_api_key "sk-6cfeba9834fd460cbe856c99be17aa74"
```

*生成产物位置*：`dataset_all/nyc/test/nyc_test_candidates_hsid.jsonl`

### 4.2 TKY 数据集候选池构建
TKY 测试集共有 2206 条，推荐首批构建 900 条保持与 CA 评估基准严格对齐：

```bash
cd /mnt/workspace/CoMaPOI0607

python ops/build_candidates_amd.py \
  --dataset tky \
  --mode test \
  --num_samples 900 \
  --strategy expertrag \
  --use_hsid \
  --use_cloud_embedding \
  --embedding_api_key "sk-6cfeba9834fd460cbe856c99be17aa74"
```

*生成产物位置*：`dataset_all/tky/test/tky_test_candidates_hsid.jsonl`

---

## 5. 运行三 Agent 协同前向推理 (Inference)

### 5.1 NYC 前向推理实验 (900 样本)

```bash
cd /mnt/workspace/CoMaPOI0607

python3 inference_forward_new.py \
  --dataset nyc \
  --num_samples 900 \
  --batch_size 4 \
  --candidate_fusion_strategy rrf \
  --use_hsid \
  --strategy expertrag \
  --agent1_base_url "https://dashscope.aliyuncs.com/compatible-mode/v1" \
  --agent1_api_key "sk-6cfeba9834fd460cbe856c99be17aa74" \
  --agent1_api "qwen-plus" \
  --agent2_base_url "https://dashscope.aliyuncs.com/compatible-mode/v1" \
  --agent2_api_key "sk-6cfeba9834fd460cbe856c99be17aa74" \
  --agent2_api "qwen-plus" \
  --agent3_base_url "http://localhost:7863/v1" \
  --agent3_api_key "EMPTY" \
  --agent3_api "qwen3-8b" \
  --profile_max_tokens 120 \
  --agent3_max_tokens 1024 \
  --test_interval 50 \
  --save_name amd_rag_lora_nyc \
  --store_save_name
```

### 5.2 TKY 前向推理实验 (900 样本)

```bash
cd /mnt/workspace/CoMaPOI0607

python3 inference_forward_new.py \
  --dataset tky \
  --num_samples 900 \
  --batch_size 4 \
  --candidate_fusion_strategy rrf \
  --use_hsid \
  --strategy expertrag \
  --agent1_base_url "https://dashscope.aliyuncs.com/compatible-mode/v1" \
  --agent1_api_key "sk-6cfeba9834fd460cbe856c99be17aa74" \
  --agent1_api "qwen-plus" \
  --agent2_base_url "https://dashscope.aliyuncs.com/compatible-mode/v1" \
  --agent2_api_key "sk-6cfeba9834fd460cbe856c99be17aa74" \
  --agent2_api "qwen-plus" \
  --agent3_base_url "http://localhost:7863/v1" \
  --agent3_api_key "EMPTY" \
  --agent3_api "qwen3-8b" \
  --profile_max_tokens 120 \
  --agent3_max_tokens 1024 \
  --test_interval 50 \
  --save_name amd_rag_lora_tky \
  --store_save_name
```

> **参数说明**：
> - `--test_interval 50`: 保证每推理 50 条样本就落盘一次 `interim_diagnostics_*.json` 与 `interim_poi_predictions_*.json`。即使网络波动中断，之前已跑的数据也不会丢失。
> - `--batch_size 4`: 在保证 DashScope API 吞吐量和防限流（429）之间取得最佳平衡。

---

## 6. 结果查看与指标提取

推理完成后，实验结果会自动归档于：
- NYC: `results/nyc/amd_rag_lora_nyc/`
- TKY: `results/tky/amd_rag_lora_tky/`

### 6.1 查看主指标
```bash
cat results/nyc/amd_rag_lora_nyc/metrics.txt
cat results/tky/amd_rag_lora_tky/metrics.txt
```
指标包括：`HR@1`, `HR@3`, `HR@5`, `HR@10`, `MRR`, `NDCG@10`。

---

## 7. 论文回填指南

跑完后，将 NYC 与 TKY 得到的指标直接回填到以下两份论文源码中：
1. **中文手稿**：`paper_writing/chinese/GeoSemID_TITS_zh.tex`
   - 定位 `Table IV` (主对比表) 与 `Table V` (消融表)
   - 替换 `\mock{...}` 占位数字为实测值。
2. **英文手稿**：`paper_writing/english/GeoSemID_TITS.tex`
   - 定位对应表格，填入相同实测数据。

---

## 8. 常见问题排查 (FAQ)

1. **vLLM 报端口 7863 冲突**：
   - 解决：`kill -9 $(lsof -t -i:7863)` 然后重新执行 `serve_agents_amd.sh`。
2. **DashScope API 报错 429 Too Many Requests**：
   - 解决：在 `inference_forward_new.py` 中将 `--batch_size` 调整为 `2`。
3. **输出截断或 JSON 解析异常**：
   - 脚本内置了智能自动清洗器，若模型偶尔产生 markdown 代码块包裹，也会被提取并回退，不会导致进程崩溃。

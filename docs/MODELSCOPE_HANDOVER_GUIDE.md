# 🚀 CoMaPOI / GeoSemID 本地客户端换机交接与实验推进指南
> **使用场景界定**：
> * **远程服务器**：**保持不变**，依然是同一台远程 ModelScope 算力服务器（GPU、模型权重、已有的环境与目录均在）。
> * **本地客户端**：**更换新电脑**。用户将在新电脑上使用 Antigravity AI 继续发布指令、监控与协同推进。

---

## 一、 新电脑（本地客户端）准备工作

在新电脑上克隆代码库并打开项目，即可立即恢复与 AI 的无缝协同：

### 1. 新电脑拉取代码
在您的新电脑终端（支持 Git Bash / PowerShell / Terminal）执行：

```bash
git clone -b fromjjy0624-lq https://github.com/niu2002/CoMaPOI0607.git
cd CoMaPOI0607
```

### 2. 本地项目已包含的资产与上下文
当前分支（`fromjjy0624-lq`）已同步好所有上下文，打开即可直接使用：
* **项目与远程环境规则**：[.agents/AGENTS.md](file:///d:/project/jjycomapoi/.agents/AGENTS.md)（内置了 ModelScope AMD GPU 环境、API 配置、vLLM 规范，新电脑上的 AI 会自动遵循）。
* **CA 已完成成果与诊断分析**：
  * [docs/ca_full_model_1050_diagnostics.md](file:///d:/project/jjycomapoi/docs/ca_full_model_1050_diagnostics.md)（1050 样本物理距离误差与合规分析报告）
  * [docs/ca_baseline_vs_rag_comparison.md](file:///d:/project/jjycomapoi/docs/ca_baseline_vs_rag_comparison.md)（900 样本差分分析报告）
* **中英文最新论文手稿与图表**：`paper_writing/chinese/` 与 `paper_writing/english/`。

---

## 二、 当前实验进展状态

| 数据集 | 实验阶段 | 样本量 | HR@10 | MRR | 状态与进展 |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **CA** | **Baseline (无 RAG)** | 900 | 41.67% | 23.78% | 已完成 |
| **CA** | **未微调 ExpertRAG** | 900 | 40.00% | 20.99% | 已完成（差分诊断完成） |
| **CA** | **微调完整版 (Full GeoSemID)** | **1050** | **42.48%** | **21.54%** | **已完成并写入论文 Table IV/V**（MDE=38.4km, 5km召回=47.24%） |
| **NYC** | **Full GeoSemID 评测** | 900 | - | - | **待跑（剩余任务 1）** |
| **TKY** | **Full GeoSemID 评测** | 900 | - | - | **待跑（剩余任务 2）** |

---

## 三、 远程 ModelScope 算力机同步与运行状态检查

由于远程还是原先同一台 ModelScope 实例，因此**模型权重（`/mnt/workspace/models`）与已有环境均完好保留**，只需要同步一下最新脚本：

### 1. 远程服务器同步最新代码
在 ModelScope 实例终端执行：
```bash
cd /mnt/workspace/CoMaPOI0607
git checkout fromjjy0624-lq
git pull origin fromjjy0624-lq
```

### 2. 检查或拉起 vLLM 服务 (Agent 3)
检查后台是否已有 vLLM 在运行：
```bash
curl http://localhost:7863/v1/models
```
* **若返回包含 `qwen3-8b` 的 JSON**：说明服务依然存活，可直接跳过启动，直接进行候选池与推理！
* **若未运行或实例曾重启**：执行以下命令一键拉起：
  ```bash
  cd /mnt/workspace/CoMaPOI0607
  MODEL_ROOT=/mnt/workspace/models \
  MODEL_NAME=Qwen/Qwen3-8B \
  SERVED_MODEL_NAME=qwen3-8b \
  GPU_MEMORY_UTILIZATION=0.78 \
  bash ops/serve_agents_amd.sh
  ```
  监控日志：`tail -100f /tmp/comapoi-vllm-agents.log`

---

## 四、 剩余两个实验的具体跑法 (NYC & TKY)

### 任务 1：NYC 数据集（900 样本）

#### 步骤 1.1：在 ModelScope 上生成 NYC 候选池（含 HSID）
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
*产物确认*：`dataset_all/nyc/test/nyc_test_candidates_hsid.jsonl`

#### 步骤 1.2：运行 NYC 三 Agent 前向推理
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

---

### 任务 2：TKY 数据集（900 样本）

#### 步骤 2.1：在 ModelScope 上生成 TKY 候选池（含 HSID）
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
*产物确认*：`dataset_all/tky/test/tky_test_candidates_hsid.jsonl`

#### 步骤 2.2：运行 TKY 三 Agent 前向推理
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

---

## 五、 在新电脑上开启对话时对 AI 的“开箱即用”提示词 (Prompt 模板)

当您在新电脑上打开 Antigravity 时，可以直接复制以下这段话发给 AI，AI 会瞬间获悉全部背景并接管下一步工作：

```text
你好！我换到了这台新电脑上继续和你结对编程。
我们的远程算力机（同一台 ModelScope GPU 实例）环境已经就绪，CA 的主实验已完整跑完（HR@10=42.48% 已入论文）。
请查看项目中的 docs/MODELSCOPE_HANDOVER_GUIDE.md，接下来请协助我推进 NYC 和 TKY 两个数据集的候选池生成与前向推理实验！
```

---

## 六、 运行后的指标提取与论文回填

1. **查看主指标**：
   ```bash
   cat results/nyc/amd_rag_lora_nyc/metrics.txt
   cat results/tky/amd_rag_lora_tky/metrics.txt
   ```
2. **回填论文占位符**：
   * 中文手稿：`paper_writing/chinese/GeoSemID_TITS_zh.tex` (定位 Table IV & V)
   * 英文手稿：`paper_writing/english/GeoSemID_TITS.tex` (定位 Table IV & V)
   * 将 `\mock{...}` 替换为实测的 HR@1/5/10 与 MRR、NDCG 指标。

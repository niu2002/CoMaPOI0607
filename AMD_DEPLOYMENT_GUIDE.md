# CoMaPOI 远程 AMD 算力服务器一键部署指南 (针对 Ubuntu 22.04 / ROCm 7.2.1 / Py3.12 环境)

针对你当前租用的 **8核 / 200GB 内存 / 192G 显存** 的高配 AMD GPU 环境（预装镜像：`ubuntu22.04-rocm7.2.1-py312-torch2.9.1-1.36.3`，内含 ModelScope 官方库），特制定此极速部署文档。

> [!IMPORTANT]
> **绝对避坑指南**：该系统镜像中已经预装了与 AMD GPU 硬件驱动和 ROCm 7.2.1 深度编译对齐的 **Python 3.12 和 PyTorch 2.9.1**。请**千万不要**自己使用 Conda 创建全新的纯净 Python 3.10 虚拟环境，否则会因为缺失 AMD 专用的 ROCm 编译版本而导致 PyTorch 无法识别显卡！

---

## 1. 第一步：代码拉取与 Git 同步

首先，登录远程服务器并进入你的项目工作区目录：

```bash
# 1. 进入服务器工作区目录
cd /mnt/workspace/comapoilatest/CoMaPOI0607

# 2. 拉取最新分支
git fetch origin
git checkout fromjjy0624-lq
git pull origin fromjjy0624-lq
```

---

## 2. 第二步：环境配置（继承预装驱动）

为了既不污染 `base` 环境，又完整继承预装镜像中已经编译好的 PyTorch 2.9.1 和 ModelScope 驱动库，我们采用 **`--clone base`** 方式创建虚拟环境：

```bash
# 1. 极速克隆已配置好 PyTorch 2.9.1 ROCm 驱动的 base 环境
conda create -n comapoi_amd --clone base -y

# 2. 激活虚拟环境
conda activate comapoi_amd
```

### 2.1 安装应用层与 AgentScope 依赖包
通过国内源极速安装 Agent 框架与其他辅助科学计算包：
```bash
# 配置 pip 阿里镜像源
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/

# 安装应用层依赖（PyTorch和Modelscope直接复用克隆自base的预装版本，无需重新下载）
pip install agentscope>=0.2.0 pandas transformers tqdm loguru openai requests scikit-learn scipy numpy

# 安装 vLLM 推理框架
pip install vllm
```

---

## 3. 第三步：从 Modelscope 下载 Qwen 本地模型

ModelScope 已经作为 Library 预装在你的虚拟环境中，我们使用 Python 指令将我们定死的大模型极速下载到指定路径：

```bash
# 1. 创建大模型存储目录
mkdir -p /mnt/workspace/comapoilatest/models

# 2. 运行 Python 下载本地大模型（下载路径为 /mnt/workspace/comapoilatest/models）
python -c "
from modelscope import snapshot_download
print('Downloading Qwen2.5-7B-Instruct...')
snapshot_download('Qwen/Qwen2.5-7B-Instruct', cache_dir='/mnt/workspace/comapoilatest/models')
print('Downloading Qwen3-Embedding-4B...')
snapshot_download('Qwen/Qwen2.5-Math-1.5B-Instruct', cache_dir='/mnt/workspace/comapoilatest/models')
"
```

---

## 4. 第四步：vLLM 本地模型服务部署

为 Agent 3 (Predictor) 部署本地大模型服务（挂载 LoRA 权重）。

### 4.1 启动 vLLM 推理服务
为了防止 vLLM 完全锁定 GPU 显存从而阻碍下游 RAG 向量模块的矩阵计算，**必须指定 `--gpu-memory-utilization 0.78`** 限制其显存占用：

```bash
# 启动本地 Qwen2.5-7B-Instruct 推理服务（定死端口为 7863）
# 如果你已经训练好了 LoRA 适配器，挂载参数：--lora-modules agent3_lora=/mnt/workspace/comapoilatest/CoMaPOI0607/finetune/results --enable-lora

CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model /mnt/workspace/comapoilatest/models/Qwen/Qwen2.5-7B-Instruct \
  --port 7863 \
  --gpu-memory-utilization 0.78 \
  --served-model-name qwen2.5-7b-instruct
```

---

## 5. 第五步：运行对照推理与评测

我们已经在代码中注入了**向量离线自适应载入**机制。当启动前向推理时，程序会检测到你拉取下来的 `.npy` 文件并**在毫秒级内自动为你在本地构建 FAISS 索引**，不需要耗费任何云端 Token。

### 5.1 连通性测试（Smoke 冒烟测试 - 10 个样本）
```bash
# 2 API (云端) + 1 本地 vLLM (Agent 3) 连通性测试
python inference_forward_new.py --dataset ca --num_samples 10 --batch_size 2 --candidate_fusion_strategy rrf --use_hsid \
  --agent1_base_url "https://dashscope.aliyuncs.com/compatible-mode/v1" --agent1_api_key "sk-6cfeba9834fd460cbe856c99be17aa74" --agent1_api "qwen-plus" \
  --agent2_base_url "https://dashscope.aliyuncs.com/compatible-mode/v1" --agent2_api_key "sk-6cfeba9834fd460cbe856c99be17aa74" --agent2_api "qwen-plus" \
  --agent3_base_url "http://localhost:7863/v1" --agent3_api_key "EMPTY" --agent3_api "qwen2.5-7b-instruct" \
  --save_name local_smoke --store_save_name
```

### 5.2 运行 900 样本对照组评测

* **对照组 A：2 API + 1 本地 LoRA (RAG 重排融合版)**：
  ```bash
  python inference_forward_new.py --dataset ca --num_samples 900 --batch_size 4 --candidate_fusion_strategy rrf --use_hsid \
    --agent1_base_url "https://dashscope.aliyuncs.com/compatible-mode/v1" --agent1_api_key "sk-6cfeba9834fd460cbe856c99be17aa74" --agent1_api "qwen-plus" \
    --agent2_base_url "https://dashscope.aliyuncs.com/compatible-mode/v1" --agent2_api_key "sk-6cfeba9834fd460cbe856c99be17aa74" --agent2_api "qwen-plus" \
    --agent3_base_url "http://localhost:7863/v1" --agent3_api_key "EMPTY" --agent3_api "qwen2.5-7b-instruct" \
    --save_name amd_rag_lora --store_save_name
  ```

* **对照组 B：2 API + 1 本地 LoRA (无 RAG 对照组)**：
  ```bash
  # 1. 临时改名排除 RAG 干扰
  mv dataset_all/ca/test/ca_test_candidates_hsid.jsonl dataset_all/ca/test/ca_test_candidates_hsid.jsonl.bak

  # 2. 跑推理
  python inference_forward_new.py --dataset ca --num_samples 900 --batch_size 4 --candidate_fusion_strategy rrf --use_hsid \
    --agent1_base_url "https://dashscope.aliyuncs.com/compatible-mode/v1" --agent1_api_key "sk-6cfeba9834fd460cbe856c99be17aa74" --agent1_api "qwen-plus" \
    --agent2_base_url "https://dashscope.aliyuncs.com/compatible-mode/v1" --agent2_api_key "sk-6cfeba9834fd460cbe856c99be17aa74" --agent2_api "qwen-plus" \
    --agent3_base_url "http://localhost:7863/v1" --agent3_api_key "EMPTY" --agent3_api "qwen2.5-7b-instruct" \
    --save_name amd_no_rag_lora --store_save_name

  # 3. 评测结束后恢复
  mv dataset_all/ca/test/ca_test_candidates_hsid.jsonl.bak dataset_all/ca/test/ca_test_candidates_hsid.jsonl
  ```

评测结束后，请前往 `results/ca/amd_rag_lora/metrics.txt` 和 `results/ca/amd_no_rag_lora/metrics.txt` 抓取性能对比指标。

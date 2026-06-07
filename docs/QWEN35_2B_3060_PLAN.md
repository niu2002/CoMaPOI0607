# CoMaPOI + Qwen3.5-2B 在 RTX 3060 12GB 上的实施记录

## 1. 本机配置

- GPU: NVIDIA GeForce RTX 3060 Desktop 12GB
- 当前显存占用观察: `1.5 / 12.0 GB`
- NVIDIA Driver: `591.86`
- `nvidia-smi` 显示 CUDA: `13.1`
- Conda 环境: `co0607`
- Python: `3.12.13`

## 2. 可行性结论

结论: `Qwen/Qwen3.5-2B` 可以作为 CoMaPOI 的基础模型继续推进，但需要区分三件事。

### 2.1 本地下载和单卡推理

可行。

- 已通过 ModelScope 下载到本地: `models/Qwen3.5-2B`
- 已用 `transformers` 在本机 GPU 上完成最小生成 smoke
- 该 smoke 运行时的 `torch.cuda.max_memory_allocated()` 约为 `3617.8 MB`

这说明在 3060 12GB 上，`Qwen3.5-2B` 的基础推理是轻松可跑的。

### 2.2 LoRA / QLoRA 微调

可行，但推荐从小规模实验开始。

- 本机已经装好 `bitsandbytes`
- CoMaPOI 的微调脚本本身使用 `AutoModelForCausalLM + LoRA`
- 我已将 `finetune_sft_new.py` 中原作者机器上的 Linux 绝对路径改为项目相对路径，避免 Windows 下直接报路径错误

建议策略:

- 优先用 `agent1 / agent2 / agent3` 单独微调做通路验证
- 再尝试 `merged`
- 初始参数建议:
  - `--batch_size 1`
  - `--gradient_accumulation_steps 8` 或更大
  - `--fp16`
  - 必要时打开 `--gradient_checkpointing`

### 2.3 完整多智能体 OpenAI-Compatible 服务推理

部分可行。

- CoMaPOI 的 `forward inference` 依赖本地 OpenAI-compatible 服务
- 仓库文档默认使用 `vLLM`
- 在当前原生 Windows 环境中，`vllm` 安装退回源码构建并失败，因此“不建议把原生 Windows + vLLM”作为主路径

推荐路线:

1. Windows 本机负责:
   - ModelScope 拉模型
   - Transformers 本地 smoke
   - LoRA / QLoRA 微调
2. 若后续一定要跑多 Agent API 服务:
   - 优先上 `WSL2 Ubuntu`
   - 或改用其他 OpenAI-compatible 服务端

## 3. 你给的 `models/qwen3-2.5B-intro` 文档怎么用

整体方向可以用，但要做两点纠正。

### 3.1 可以直接沿用的部分

- 用 `ModelScope` 下载模型
- 用本地目录而不是在线仓库路径
- 用 `Transformers + PEFT` 做微调

建议下载命令:

```bash
modelscope download --model "Qwen/Qwen3.5-2B" --local_dir "./models/Qwen3.5-2B"
```

### 3.2 不要直接照搬的部分

文档中的这段:

```python
from transformers import AutoProcessor, AutoModelForImageTextToText
```

不适合作为 CoMaPOI 的默认入口。CoMaPOI 当前代码是纯文本 CausalLM 流程，应优先使用:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
```

并通过 `tokenizer.apply_chat_template(...)` 组织输入。

## 4. 当前环境版本

在 `co0607` 中已确认:

- `torch == 2.11.0+cu128`
- `torch.version.cuda == 12.8`
- `transformers == 5.10.2`
- `tokenizers == 0.22.2`
- `peft == 0.19.1`
- `trl == 1.5.1`
- `accelerate == 1.13.0`
- `modelscope == 1.37.1`
- `bitsandbytes == 0.49.2`

## 5. 已完成的 smoke

### 5.1 GPU / CUDA smoke

已确认:

- `torch.cuda.is_available() == True`
- `torch.cuda.device_count() == 1`
- GPU 名称识别正常

### 5.2 本地模型生成 smoke

已完成:

- 从 `models/Qwen3.5-2B` 加载 tokenizer
- 从 `models/Qwen3.5-2B` 加载 model
- 在 GPU 上生成一条中文回答

说明:

- 主链路 `ModelScope -> local model dir -> transformers -> CUDA generate` 已打通
- 生成内容与项目事实未必完全一致，这次 smoke 的目标是验证链路，不是验证任务效果

## 6. 下一步建议

建议按这个顺序做:

1. 先跑 `inference_inverse_new.py` 生成 `finetune/data/{dataset}` 的三份训练数据
2. 用 `finetune_sft_new.py` 先做 `agent1` 的 10 到 20 step 小样本试跑
3. 再分别试 `agent2`、`agent3`
4. 三个 agent 微调稳定后，再决定是否进入:
   - `merged`
   - WSL2 + vLLM
   - 多 Agent 完整前向推理

## 7. 参考命令

### 7.1 激活环境

```bash
conda activate co0607
```

### 7.2 本地下载模型

```bash
modelscope download --model "Qwen/Qwen3.5-2B" --local_dir "./models/Qwen3.5-2B"
```

### 7.3 单 Agent 微调试跑

```bash
python finetune_sft_new.py ^
  --dataset nyc ^
  --model Qwen3.5-2B ^
  --model_path models ^
  --type agent1 ^
  --batch_size 1 ^
  --gradient_accumulation_steps 8 ^
  --max_steps 10 ^
  --fp16
```

## 8. 当前已知风险

- `vLLM` 在原生 Windows 上未安装成功
- `transformers 5.x` 虽然当前 smoke 正常，但若后续 CoMaPOI 某些训练接口报兼容问题，优先考虑回退到稳定的 `4.x`
- `Qwen3.5-2B` 是小模型，能否复现论文最好指标并不乐观，但作为全流程验证和可运行基线是合适的

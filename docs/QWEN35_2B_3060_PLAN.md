# CoMaPOI + Qwen3.5-2B 在 RTX 3060 12GB 上的实施记录

## 1. 本机配置

- GPU: NVIDIA GeForce RTX 3060 Desktop 12GB
- 当前显存占用观察: `1.5 / 12.0 GB`
- NVIDIA Driver: `591.86`
- `nvidia-smi` 显示 CUDA: `13.1`
- Conda 环境: `co0607`
- Python: `3.12.13`

## 2. 当前结论

结论: `Qwen/Qwen3.5-2B` 可以作为 CoMaPOI 的基础模型继续推进，当前最稳妥的路线是:

1. 本地模型已下载完成，直接使用本地目录
2. 用 `transformers + PEFT/LoRA` 先跑通训练和推理
3. 原生 Windows 暂不把 `vLLM` 作为主路径

## 3. 当前已完成事项

### 3.1 本地模型状态

已完成。

- 本地模型目录: [models/Qwen3.5-2B](F:/jjy/project/CoMaPoi0607/models/Qwen3.5-2B)
- 模型来源: `ModelScope`

说明: 当前机器上已经有可直接使用的本地模型目录，后续训练和推理默认都基于这个目录，不再需要重复下载模型。

### 3.2 环境验证

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

### 3.3 GPU / CUDA smoke

已确认:

- `torch.cuda.is_available() == True`
- `torch.cuda.device_count() == 1`
- GPU 名称识别正常

### 3.4 本地模型生成 smoke

已完成:

- 从本地目录加载 tokenizer
- 从本地目录加载 model
- 在 GPU 上生成一条中文回答

结果:

- `torch.cuda.max_memory_allocated()` 约为 `3617.8 MB`

说明:

- 主链路 `ModelScope -> local model dir -> transformers -> CUDA generate` 已打通
- 这次 smoke 的目标是验证链路，不是验证任务效果

### 3.5 训练与推理产物归档规范

已完成。

- 训练权重固定保存在 `finetune/results/...`
- 运行日志统一保存在 `artifacts/logs/train/` 和 `artifacts/logs/infer_smoke/`
- 推理 smoke 结果统一保存在 `artifacts/smoke/{dataset}/`
- 历史根目录日志已迁移到 `artifacts/logs/legacy/`

## 4. CoMaPOI 对 Qwen3.5-2B 的可行性判断

### 4.1 推理

可行。

`Qwen3.5-2B` 在 3060 12GB 上做单卡推理没有问题，显存余量充足。

### 4.2 微调

可行。

- 当前项目脚本走的是 `AutoModelForCausalLM + LoRA`
- 已安装 `bitsandbytes`，脚本默认优先尝试 `4bit + LoRA`
- 这套组合非常适合 3060 12GB 做单卡实验

### 4.3 多智能体完整服务推理

部分可行。

- CoMaPOI 的 `forward inference` 依赖本地 OpenAI-compatible 服务
- 仓库文档默认示例是 `vLLM`
- 当前原生 Windows 环境中，`vLLM` 安装失败，不建议作为主路线

## 5. 已修复的问题

### 5.1 微调脚本路径兼容

已修复 [finetune_sft_new.py](F:/jjy/project/CoMaPoi0607/finetune_sft_new.py) 中原作者机器上的 Linux 绝对路径，改为项目相对路径。

### 5.2 显式训练文件支持

已支持在命令行中直接传入 `--data_path`。

这意味着现在可以直接用现有文件开训，例如:

- [dataset_all/nyc_train.jsonl](F:/jjy/project/CoMaPoi0607/dataset_all/nyc_train.jsonl)

而不必强依赖作者原来的目录布局。

## 6. 当前推荐路线

推荐按这个顺序推进:

1. 先用现有 `dataset_all/*.jsonl` 做训练 smoke
2. 训练稳定后，再决定是否进入 `agent1 / agent2 / agent3`
3. 若后续一定要跑多 Agent API 服务，再考虑 `WSL2 + vLLM`

## 7. 最新验证结果

### 7.1 训练日志与乱码修复

已验证:

- 新训练日志文件名包含 `dataset + timestamp`
- 日志开头会自动记录启动参数
- 日志不再出现此前的乱码 `鈻堚枅`

参考日志:

- [train_nyc_20260607_131527.log](F:/jjy/project/CoMaPoi0607/artifacts/logs/train/train_nyc_20260607_131527.log)

### 7.2 LoRA 权重推理 smoke

已完成，使用权重:

- [bs2-gas1-ms10-merged-lr0.0001](F:/jjy/project/CoMaPoi0607/finetune/results/4-14/sft-nyc/bs2-gas1-ms10-merged-lr0.0001)

结果:

- 测试集样本 0 的参考标签是 `{"next_poi_id": 29}`
- LoRA smoke 生成结果是 `[ {{ 'next POI ID': 2044 } }]`
- 成功解析出的预测 POI ID 是 `2044`
- 单次推理耗时约 `2.12s`
- 峰值显存约 `2081.15 MB`

参考文件:

- [infer_smoke_nyc_20260607_131800.log](F:/jjy/project/CoMaPoi0607/artifacts/logs/infer_smoke/infer_smoke_nyc_20260607_131800.log)
- [lora_smoke_nyc_20260607_131805.json](F:/jjy/project/CoMaPoi0607/artifacts/smoke/nyc/lora_smoke_nyc_20260607_131805.json)

## 8. 当前参考命令

### 8.1 激活环境

```bash
conda activate co0607
```

### 8.2 单次本地推理 smoke

```bash
python lora_inference_smoke.py ^
  --dataset nyc ^
  --data_path dataset_all/nyc_test.jsonl ^
  --adapter_path finetune/results/4-14/sft-nyc/bs2-gas1-ms10-merged-lr0.0001 ^
  --load_in_4bit
```

### 8.3 使用本地模型直接做微调 smoke

```bash
python finetune_sft_new.py ^
  --dataset nyc ^
  --model Qwen3.5-2B ^
  --model_path models ^
  --data_path dataset_all/nyc_train.jsonl ^
  --batch_size 2 ^
  --gradient_accumulation_steps 1 ^
  --max_steps 10 ^
```

## 9. 风险与注意事项

- `vLLM` 在原生 Windows 上当前未安装成功
- 由于脚本默认尝试 `4bit + LoRA`，显存可能不会自然占满 10GB，需要通过 `batch_size` 和序列长度调节
- `Qwen3.5-2B` 是小模型，适合作为全流程验证和可运行基线，但未必能复现论文最好指标

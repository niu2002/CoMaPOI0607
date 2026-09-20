# Sanitized AMD GPU PyTorch Demo

这是一个经过脱敏的 AMD GPU 通用工程演示：它使用合成分类数据和独立实现的小型 MLP，展示 PyTorch 的设备选择、张量迁移、混合精度、计时和显存统计。

**This repository is a sanitized engineering demonstration and does not contain the unpublished research implementation.**

本仓库不包含未公开研究实现、论文方法、私有数据、权重、实验配置、实验结果或内部服务信息。

## 文件结构

```text
public_amd_demo/
├── amd_gpu_demo.ipynb      # 按顺序运行的主示例
├── README.md               # 使用说明
├── requirements.txt        # 最小 Python 依赖
├── src/
│   ├── __init__.py
│   └── amd_runtime.py      # 独立的运行时与性能工具
├── assets/
│   └── README.md
├── DISCLOSURE_REVIEW.md    # 脱敏审查记录
└── RUN_CHECKLIST.md        # 运行前检查清单
```

## 安装与运行

1. 在 AMD ROCm 主机上安装与 ROCm 版本匹配的 PyTorch wheel。请以 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/) 和 AMD ROCm 文档为准；不要将本仓库的 `requirements.txt` 视为 ROCm wheel 的替代品。
2. 安装 Notebook 依赖：`python -m pip install -r requirements.txt`
3. 启动：`jupyter notebook amd_gpu_demo.ipynb`
4. 按顺序执行所有单元格。运行输出位置保留为空，真实信息会在本机执行时显示。

## AMD ROCm 说明

在 ROCm 版 PyTorch 中，AMD GPU 通常仍通过 `torch.cuda` 命名空间访问，例如 `torch.cuda.is_available()` 和 `torch.cuda.get_device_name()`。这是 PyTorch 的兼容接口命名，**不表示正在使用 NVIDIA GPU**。Notebook 同时打印 `torch.version.hip`，以帮助识别 HIP/ROCm 运行时。

## CPU fallback

没有可用的 PyTorch 加速器时，Notebook 自动选择 `cpu`，跳过 GPU 显存统计与 float16 autocast，并仍可完成小规模合成数据训练和计时。CPU 数值仅用于验证流程，不应作为 AMD GPU 性能结论。

## ModelScope Gallery 上传建议

- 上传该目录本身，不要从父项目复制任何文件。
- 保持 Notebook 输出为空，或仅保留上传者在公开 AMD ROCm 环境中真实生成的输出。
- 在作品说明中注明硬件、ROCm、PyTorch 版本和实际执行日期。
- 不上传 checkpoints、数据集、日志、环境变量文件或截图中可识别的路径与账户信息。

## 隐私与脱敏声明

该示例从零实现通用 PyTorch 工程步骤，不导入父目录或任何研究模块，不读取外部数据、权重或配置，也不访问网络服务。它不构成完整论文实现，也不能用于复现实验结论。


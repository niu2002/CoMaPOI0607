#!/usr/bin/env bash
# ==============================================================================
# GeoSemID 加州 (CA) 全量 1,818 样本高速并发消融实验脚本
# ==============================================================================
# 特性:
# 1. 样本量: 1,818 (CA 全量测试集);
# 2. 并发度: 16 并发 (充分吃满 vLLM continuous batching 与 AMD GPU 算力);
# 3. 间隔评估: 每 100 样本落盘并输出一次实时阶段指标 (TEST_INTERVAL=100);
# 4. 生成约束: Agent 3 最大生成 512 tokens (杜绝长尾 CoT 拖慢推理);
# 5. 断点续跑: 自动识别已完成的样本并秒级跳过 (已跑的 900 样本直接续跑到 1818).
# ==============================================================================

set -eo pipefail

# 自动激活 Conda 环境 (若存在)
for conda_sh in /root/miniconda3/etc/profile.d/conda.sh /opt/conda/etc/profile.d/conda.sh /root/anaconda3/etc/profile.d/conda.sh; do
    if [ -f "$conda_sh" ]; then
        source "$conda_sh"
        break
    fi
done

if command -v conda >/dev/null 2>&1; then
    conda activate comapoi_amd 2>/dev/null || conda activate base 2>/dev/null || true
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    PYTHON_BIN="python3"
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=============================================================================="
echo "⚡ GeoSemID 加州 (CA) 全量 1,818 样本高速并发消融矩阵启动"
echo "  * 样本总数: 1818 (全量测试集)"
echo "  * 并发批大小: 16"
echo "  * 输出评估间隔: 每 100 样本"
echo "  * Agent 3 最大 Token: 512"
echo "=============================================================================="

# 1. 检查并确保 vLLM 正在挂载 CA LoRA (若未启动则自动拉起)
if ! curl -s "http://localhost:7863/v1/models" | grep -q "agent3"; then
    echo "🔄 [VLLM] 正在启动 CA LoRA 服务..."
    pkill -f "vllm.entrypoints.openai.api_server" || true
    sleep 3
    DATASET=ca \
    MODEL_ROOT=/mnt/workspace/models \
    MODEL_NAME=Qwen/Qwen3-8B \
    SERVED_MODEL_NAME=qwen3-8b \
    OP_STR=amd-agent3-ca-lora-v1 \
    GPU_MEMORY_UTILIZATION=0.78 \
    PYTHON_BIN="$PYTHON_BIN" \
    bash ops/serve_agent3_amd.sh
fi

echo "✅ [VLLM] CA LoRA 服务已就绪！"

# 2. 启动 CA 全量 1818 样本消融流水线
BATCH_SIZE=16 \
TEST_INTERVAL=100 \
AGENT3_MAX_TOKENS=512 \
PYTHON_BIN="$PYTHON_BIN" \
bash ops/run_ablation_matrix_lora.sh ca 1818

echo ""
echo "=============================================================================="
echo "🎉 加州 (CA) 全量 1,818 样本消融实验全部完成！"
echo "=============================================================================="

# 3. 输出 CA 完整消融对比总表
"$PYTHON_BIN" ops/summarize_ablation_table.py --dataset ca || true

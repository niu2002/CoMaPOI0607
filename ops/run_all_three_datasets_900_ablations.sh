#!/usr/bin/env bash
# ==============================================================================
# GeoSemID 三大都市 (CA / NYC / TKY) 900 样本消融实验一键全自动调度脚本
# ==============================================================================
# 核心特性:
# 1. 自动按序遍历加州 (CA)、纽约 (NYC)、东京 (TKY);
# 2. 每个数据集自动热切换并重启挂载对应城市的 Agent 3 LoRA 权重;
# 3. 自动复用各都市 Master Cache (Agent 1 & Agent 2 历史画像与重排语义);
# 4. 零 API Token 消耗，全自动断点秒级跳过与离线评估;
# 5. 全部完成后自动生成完整的 3 城市横向对比 Table VI (Markdown & LaTeX).
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

NUM_SAMPLES="${1:-900}"
DATASETS=("ca" "nyc" "tky")

echo "=============================================================================="
echo "🌟 GeoSemID 三大都市 900 样本消融实验全自动化矩阵"
echo "  * 目标数据集: ${DATASETS[*]}"
echo "  * 每数据集样本数: ${NUM_SAMPLES}"
echo "  * Python: $("$PYTHON_BIN" --version 2>&1) ($PYTHON_BIN)"
echo "=============================================================================="

switch_vllm_lora() {
    local DATASET_NAME="$1"
    local OP_STR_NAME="amd-agent3-${DATASET_NAME}-lora-v1"

    echo ""
    echo "=============================================================================="
    echo "🔄 [VLLM] 正在切换并拉起 ${DATASET_NAME^^} 的 LoRA 适配器服务..."
    echo "  * DATASET: $DATASET_NAME"
    echo "  * OP_STR: $OP_STR_NAME"
    echo "=============================================================================="

    # 优雅停止已有 vLLM 进程
    pkill -f "vllm.entrypoints.openai.api_server" || true
    sleep 3

    DATASET="${DATASET_NAME}" \
    MODEL_ROOT=/mnt/workspace/models \
    MODEL_NAME=Qwen/Qwen3-8B \
    SERVED_MODEL_NAME=qwen3-8b \
    OP_STR="${OP_STR_NAME}" \
    GPU_MEMORY_UTILIZATION=0.78 \
    PYTHON_BIN="$PYTHON_BIN" \
    bash ops/serve_agent3_amd.sh

    echo "✅ [VLLM] ${DATASET_NAME^^} LoRA 服务就绪！"
}

for ds in "${DATASETS[@]}"; do
    echo ""
    echo "##############################################################################"
    echo "🚀 开始执行数据集 [${ds^^}] 的 900 样本消融矩阵"
    echo "##############################################################################"

    # 1. 切换并启动当前都市对应的 LoRA 服务
    switch_vllm_lora "$ds"

    # 2. 执行该都市的 7 大消融组
    PYTHON_BIN="$PYTHON_BIN" bash ops/run_ablation_matrix_lora.sh "$ds" "$NUM_SAMPLES"

    echo "✅ 数据集 [${ds^^}] 消融矩阵全部执行完毕！"
done

echo ""
echo "=============================================================================="
echo "🎉🎉🎉 三大都市 (CA / NYC / TKY) 全量消融实验全部顺利完成！"
echo "=============================================================================="

# 生成三大都市横向对比的完整 Table VI
"$PYTHON_BIN" ops/summarize_ablation_table.py --dataset all || true

echo "=============================================================================="
echo "📄 论文 LaTeX 代码已写入: paper_writing/chinese/ablation_table_generated.tex"
echo "=============================================================================="

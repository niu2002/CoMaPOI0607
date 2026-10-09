#!/usr/bin/env bash
# ==============================================================================
# GeoSemID 纽约 (NYC) 消融实验 1 & 2 (900 样本稳健并发流水线)
# ==============================================================================
# 包含组别:
# 1. ablation_nyc_wo_hsid   (去除分层语义标识 HSID，退回离散 POI ID)
# 2. ablation_nyc_wo_router (去除动态路由 RRF，采用 Union 候选池)
# 
# 特性:
# - 样本量: 900 样本 (黄金子集)
# - 稳健并发: 默认 BATCH_SIZE=8 (充分发挥 AMD 算力，防爆显存)
# - 权重挂载: 全程调用 NYC 微调后的 Agent 3 LoRA 权重 (agent3)
# - 每 100 样本自动落盘并评估阶段指标 (TEST_INTERVAL=100)
# - 自动输出 NYC 评估结果
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

DATASET="nyc"
NUM_SAMPLES="900"
BATCH_SIZE="${BATCH_SIZE:-8}"
TEST_INTERVAL="${TEST_INTERVAL:-100}"
AGENT3_MAX_TOKENS="${AGENT3_MAX_TOKENS:-512}"
AGENT3_BASE_URL="${AGENT3_BASE_URL:-http://localhost:7863/v1}"
AGENT3_API_KEY="${AGENT3_API_KEY:-EMPTY}"
AGENT3_LORA_MODEL="${AGENT3_LORA_MODEL:-agent3}"
AGENT3_BASE_MODEL="${AGENT3_BASE_MODEL:-qwen3-8b}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=============================================================================="
echo "🚀 GeoSemID 纽约 (NYC) 消融实验 1 & 2 (900 样本稳健矩阵启动)"
echo "  * 数据集: $DATASET"
echo "  * 样本数: $NUM_SAMPLES"
echo "  * 并发批大小: $BATCH_SIZE (防爆显存优化)"
echo "  * 输出评估间隔: 每 $TEST_INTERVAL 样本"
echo "  * Agent 3 最大 Token: $AGENT3_MAX_TOKENS"
echo "  * Agent 3 目标权重: $AGENT3_LORA_MODEL (NYC SFT LoRA)"
echo "=============================================================================="

# 1. 自动定位 Master Cache
MASTER_CACHE="results/${DATASET}/amd_rag_lora_with_sft_${DATASET}/poi_predictions.json"
if [ ! -f "$MASTER_CACHE" ]; then
    if [ -f "results/${DATASET}/full_model_lora/poi_predictions.json" ]; then
        MASTER_CACHE="results/${DATASET}/full_model_lora/poi_predictions.json"
    elif [ -f "results/${DATASET}/full_model/poi_predictions.json" ]; then
        MASTER_CACHE="results/${DATASET}/full_model/poi_predictions.json"
    else
        echo "❌ [ERROR] 未找到 NYC Master Cache: $MASTER_CACHE"
        echo "请确认已生成 NYC 的主实验前向结果 (或使用 Dashscope API 提取 900 样本画像)"
        exit 1
    fi
fi
echo "✅ [CACHE] 成功定位 NYC Master Cache: $MASTER_CACHE"

# 2. 检查 vLLM 服务并确保挂载 NYC LoRA
if ! curl -s "${AGENT3_BASE_URL}/models" >/dev/null 2>&1; then
    echo "⚠️ [WARN] 正在启动 NYC LoRA 服务..."
    pkill -f "vllm.entrypoints.openai.api_server" || true
    sleep 3
    DATASET=nyc \
    MODEL_ROOT=/mnt/workspace/models \
    MODEL_NAME=Qwen/Qwen3-8B \
    SERVED_MODEL_NAME=qwen3-8b \
    OP_STR=amd-agent3-nyc-lora-v1 \
    GPU_MEMORY_UTILIZATION=0.78 \
    PYTHON_BIN="$PYTHON_BIN" \
    bash ops/serve_agent3_amd.sh
fi
echo "✅ [CHECK] vLLM 服务响应正常!"

# 3. 通用消融执行函数
run_ablation_step() {
    SAVE_NAME="$1"
    DESC="$2"
    USE_HSID_FLAG="$3"
    AGENT3_API_TARGET="$4"
    EXTRA_ARGS="$5"

    SAVE_DIR="results/${DATASET}/${SAVE_NAME}"
    mkdir -p "$SAVE_DIR"

    echo ""
    echo "=============================================================================="
    echo ">>> 🚀 [消融组] $SAVE_NAME"
    echo ">>> 说明: $DESC"
    echo ">>> HSID: $USE_HSID_FLAG | Agent3: $AGENT3_API_TARGET"
    echo ">>> 额外参数: $EXTRA_ARGS"
    echo "=============================================================================="

    PRED_FILE="${SAVE_DIR}/poi_predictions.json"
    if [ -f "$PRED_FILE" ]; then
        EXISTING_COUNT=$(grep -o '"predicted_poi_ids"' "$PRED_FILE" | wc -l || echo 0)
        if [ "$EXISTING_COUNT" -ge "$NUM_SAMPLES" ]; then
            echo "⚡ [SKIP] 已完成 $EXISTING_COUNT 样本，跳过推理，执行评估..."
            "$PYTHON_BIN" evaluate.py --prediction_file "$PRED_FILE" --save_dir "$SAVE_DIR" || true
            return 0
        fi
    fi

    # 组装命令
    CMD="\"$PYTHON_BIN\" inference_forward_new.py \
      --dataset ${DATASET} \
      --num_samples ${NUM_SAMPLES} \
      --batch_size ${BATCH_SIZE} \
      --candidate_fusion_strategy rrf \
      --fused_candidate_top_k 50 \
      --strategy expertrag \
      --agent3_base_url \"${AGENT3_BASE_URL}\" \
      --agent3_api_key \"${AGENT3_API_KEY}\" \
      --agent3_api \"${AGENT3_API_TARGET}\" \
      --agent3_max_tokens ${AGENT3_MAX_TOKENS} \
      --test_interval ${TEST_INTERVAL} \
      --load_pf_output \
      --saved_results_path \"${MASTER_CACHE}\" \
      --save_name \"${SAVE_NAME}\" \
      --store_save_name \
      ${USE_HSID_FLAG} \
      ${EXTRA_ARGS}"

    eval "$CMD"

    # 执行离线多阶段与最终评估
    if [ -f "$PRED_FILE" ]; then
        "$PYTHON_BIN" evaluate.py --prediction_file "$PRED_FILE" --save_dir "$SAVE_DIR" || true
    fi
}

# ==============================================================================
# 4. 执行 NYC 消融组 1 & 2
# ==============================================================================

# E1: w/o HSID（去除分层语义标识，退回传统 POI ID 候选池）
run_ablation_step "ablation_wo_hsid" \
  "[E1] 去除分层语义标识 HSID (退回离散 POI ID)" \
  "" \
  "$AGENT3_LORA_MODEL" \
  ""

# E2: w/o Dynamic Router（去除动态路由/RRF，使用朴素 Union 候选召回）
run_ablation_step "ablation_wo_router" \
  "[E2] 去除动态路由 (采用 Union 候选池)" \
  "--use_hsid" \
  "$AGENT3_LORA_MODEL" \
  "--candidate_fusion_strategy union"

# ==============================================================================
# 5. 自动汇总并输出 NYC 指标表格
# ==============================================================================
echo ""
echo "=============================================================================="
echo "🎉 纽约 (NYC) 消融 1 & 2 执行完成！"
echo "=============================================================================="

"$PYTHON_BIN" ops/summarize_ablation_table.py --dataset nyc || true

echo "=============================================================================="
echo "📄 论文 LaTeX 代码已写入: paper_writing/chinese/ablation_table_generated.tex"
echo "=============================================================================="

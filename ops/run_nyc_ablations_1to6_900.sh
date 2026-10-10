#!/usr/bin/env bash
# ==============================================================================
# GeoSemID 全自动化消融实验调度流水线 (LoRA + Master Cache 零 Token 极速复用版)
# ==============================================================================
# 支持数据集: ca / nyc / tky
# 核心机制:
# 1. 自动复用 Master Run (amd_rag_lora_with_sft_*) 中的 Agent 1 & Agent 2 时空画像缓存;
# 2. 仅让微调后的 Agent 3 LoRA 针对不同候选池与专家权重进行重排推导;
# 3. 极速执行, 零额外 DashScope Token 消耗, 支持断点秒级跳过与自动多指标汇总.
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

# 检查 agentscope 依赖，缺失或版本异常则自动补全安装兼容版
if ! "$PYTHON_BIN" -c "import agentscope; from agentscope.message import Msg" >/dev/null 2>&1; then
    echo "⚠️ [DEPENDENCY] 正在自动安装兼容版本 agentscope==0.1.6 与 loguru==0.6.0..."
    "$PYTHON_BIN" -m pip install "agentscope==0.1.6" "loguru==0.6.0" -i https://mirrors.aliyun.com/pypi/simple/ || "$PYTHON_BIN" -m pip install "agentscope==0.1.6" "loguru==0.6.0"
fi

DATASET="${1:-nyc}"
NUM_SAMPLES="${2:-900}"
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
echo "🎯 GeoSemID 纽约 (NYC) 900 样本消融实验矩阵 (E1 ~ E6) 启动"
echo "  * 数据集: $DATASET"
echo "  * 样本量: $NUM_SAMPLES"
echo "  * 批大小: $BATCH_SIZE (防爆显存优化)"
echo "  * Python: $("$PYTHON_BIN" --version 2>&1) ($PYTHON_BIN)"
echo "  * Agent 3 LoRA 终端: $AGENT3_BASE_URL (model: $AGENT3_LORA_MODEL)"
echo "=============================================================================="

# 1. 多候选路径自动定位 Master Cache
find_master_cache() {
    local ds="$1"
    local candidates=(
        "results/${ds}/amd_rag_lora_with_sft_${ds}/poi_predictions.json"
        "results/${ds}/amd_rag_lora_${ds}/poi_predictions.json"
        "results/${ds}/amd_rag_lora/poi_predictions.json"
        "results/${ds}/full_model_lora/poi_predictions.json"
        "results/${ds}/full_model/poi_predictions.json"
    )
    for c in "${candidates[@]}"; do
        if [ -f "$c" ]; then
            echo "$c"
            return 0
        fi
    done
    return 1
}

MASTER_CACHE=$(find_master_cache "$DATASET" || echo "")
if [ -z "$MASTER_CACHE" ]; then
    echo "❌ [ERROR] 未在 results/${DATASET}/ 下找到 Master Cache 预测文件！"
    echo "请确认已完成该数据集的主实验前向推理，预测文件应位于 results/${DATASET}/amd_rag_lora_with_sft_${DATASET}/poi_predictions.json"
    exit 1
fi
echo "✅ [CACHE] 成功定位 Master Cache: $MASTER_CACHE"

# 2. 检查 vLLM 7863 端口连通性
echo "🔍 [CHECK] 检查 vLLM 服务状态..."
if curl -s "${AGENT3_BASE_URL}/models" >/dev/null 2>&1; then
    echo "✅ [CHECK] vLLM 服务响应正常!"
else
    echo "⚠️ [WARN] 无法连接到 ${AGENT3_BASE_URL}/models，请确保已启动 serve_agent3_amd.sh"
fi

# 3. 定义通用消融执行函数
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
# 4. 执行 7 大消融对照实验
# ==============================================================================

# E1: w/o HSID（去除分层语义标识，退回传统 POI ID 候选池）
run_ablation_step "ablation_wo_hsid" \
  "去除分层语义标识 HSID (退回离散 POI ID)" \
  "" \
  "$AGENT3_LORA_MODEL" \
  ""

# E2: w/o Dynamic Router（去除动态路由/RRF，使用朴素 Union 候选召回）
run_ablation_step "ablation_wo_router" \
  "去除动态路由 (采用 Union 候选池)" \
  "--use_hsid" \
  "$AGENT3_LORA_MODEL" \
  "--candidate_fusion_strategy union"

# E3: w/o Spatial Expert（去除空间距离衰减专家）
run_ablation_step "ablation_wo_spatial" \
  "去除地理空间专家 (w_spatial=0)" \
  "--use_hsid" \
  "$AGENT3_LORA_MODEL" \
  "--rrf_weights geo_near=0.0"

# E4: w/o Transition Expert（去除时序马尔可夫转移专家）
run_ablation_step "ablation_wo_transition" \
  "去除转移专家 (w_transition=0)" \
  "--use_hsid" \
  "$AGENT3_LORA_MODEL" \
  "--rrf_weights history_recent=0.0,history_freq=0.0"

# E5: w/o Temporal Expert（去除时段/星期周期与全局热度专家）
run_ablation_step "ablation_wo_temporal" \
  "去除时间与热度专家 (w_temporal=0)" \
  "--use_hsid" \
  "$AGENT3_LORA_MODEL" \
  "--rrf_weights category_similar=0.0,popular_global=0.0"

# E6: w/o Semantic Expert（去除稠密向量语义 RAG 专家）
run_ablation_step "ablation_wo_semantic" \
  "去除语义向量专家 (w_semantic=0)" \
  "--use_hsid" \
  "$AGENT3_LORA_MODEL" \
  "--rrf_weights rag_top100=0.0"

# E7: Base-only Model（去除 LoRA SFT 微调，使用原始 Qwen3-8B 基座）
run_ablation_step "ablation_base_only" \
  "去除 Agent 3 监督微调 (使用 Base-only 基座模型)" \
  "--use_hsid" \
  "$AGENT3_BASE_MODEL" \
  ""

# ==============================================================================
# 5. 自动汇总并输出当前数据集表格
# ==============================================================================
echo ""
echo "=============================================================================="
echo "🎉 数据集 [${DATASET^^}] 所有消融组推导完成！"
echo "=============================================================================="

"$PYTHON_BIN" ops/summarize_ablation_table.py --dataset "$DATASET" || true

echo "=============================================================================="
echo "✅ [${DATASET^^}] 消融实验矩阵执行与汇总完成！"
echo "=============================================================================="

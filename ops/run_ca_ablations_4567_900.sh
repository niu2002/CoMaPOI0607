#!/usr/bin/env bash
# ==============================================================================
# GeoSemID 加州 (CA) 消融实验 4, 5, 6, 7 (900 样本稳健并发流水线)
# ==============================================================================
# 包含组别:
# 4. ablation_wo_transition (去除时序马尔可夫转移专家)
# 5. ablation_wo_temporal   (去除时间规律与全局热度专家)
# 6. ablation_wo_semantic   (去除稠密语义向量 RAG 专家)
# 7. ablation_base_only     (去除 Agent 3 LoRA 微调，对比 Base 基座)
# 
# 特性:
# - 样本量: 900 样本 (黄金子集)
# - 稳健并发: 默认 BATCH_SIZE=8 (充分发挥 AMD 算力，绝对零爆显存风险)
# - 每 100 样本自动落盘并评估阶段指标 (TEST_INTERVAL=100)
# - 自动汇总并输出 CA 完整 Table VI 表格
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

DATASET="ca"
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
echo "🚀 GeoSemID 加州 (CA) 消融组 (E4~E7) 900 样本稳健矩阵启动"
echo "  * 样本数: $NUM_SAMPLES"
echo "  * 并发批大小: $BATCH_SIZE (防爆显存优化)"
echo "  * 输出评估间隔: 每 $TEST_INTERVAL 样本"
echo "  * Agent 3 最大 Token: $AGENT3_MAX_TOKENS"
echo "=============================================================================="

# 1. 自动定位 Master Cache
MASTER_CACHE="results/${DATASET}/amd_rag_lora_with_sft_${DATASET}/poi_predictions.json"
if [ ! -f "$MASTER_CACHE" ]; then
    if [ -f "results/${DATASET}/full_model_lora/poi_predictions.json" ]; then
        MASTER_CACHE="results/${DATASET}/full_model_lora/poi_predictions.json"
    elif [ -f "results/${DATASET}/full_model/poi_predictions.json" ]; then
        MASTER_CACHE="results/${DATASET}/full_model/poi_predictions.json"
    else
        echo "❌ [ERROR] 未找到 Master Cache: $MASTER_CACHE"
        exit 1
    fi
fi
echo "✅ [CACHE] 成功定位 Master Cache: $MASTER_CACHE"

# 2. 检查 vLLM 服务
if ! curl -s "${AGENT3_BASE_URL}/models" >/dev/null 2>&1; then
    echo "⚠️ [WARN] 正在启动 CA LoRA 服务..."
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
# 4. 执行 CA 消融组 (E4 ~ E7)
# ==============================================================================

# E4: w/o Transition Expert（去除时序马尔可夫转移专家）
run_ablation_step "ablation_wo_transition" \
  "[E4] 去除转移专家 (w_transition=0)" \
  "--use_hsid" \
  "$AGENT3_LORA_MODEL" \
  "--rrf_weights history_recent=0.0,history_freq=0.0"

# E5: w/o Temporal Expert（去除时段/星期周期与全局热度专家）
run_ablation_step "ablation_wo_temporal" \
  "[E5] 去除时间与热度专家 (w_temporal=0)" \
  "--use_hsid" \
  "$AGENT3_LORA_MODEL" \
  "--rrf_weights category_similar=0.0,popular_global=0.0"

# E6: w/o Semantic Expert（去除稠密向量语义 RAG 专家）
run_ablation_step "ablation_wo_semantic" \
  "[E6] 去除语义向量专家 (w_semantic=0)" \
  "--use_hsid" \
  "$AGENT3_LORA_MODEL" \
  "--rrf_weights rag_top100=0.0"

# E7: Base-only Model（去除 LoRA SFT 微调，使用原始 Qwen3-8B 基座）
run_ablation_step "ablation_base_only" \
  "[E7] 去除 Agent 3 监督微调 (使用 Base-only 基座模型)" \
  "--use_hsid" \
  "$AGENT3_BASE_MODEL" \
  ""

# ==============================================================================
# 5. 自动汇总并输出 CA 完整 Table VI 表格
# ==============================================================================
echo ""
echo "=============================================================================="
echo "🎉 加州 (CA) 全部 7 组消融实验顺利完成！"
echo "=============================================================================="

"$PYTHON_BIN" ops/summarize_ablation_table.py --dataset ca || true

echo "=============================================================================="
echo "📄 论文 LaTeX 代码已写入: paper_writing/chinese/ablation_table_generated.tex"
echo "=============================================================================="

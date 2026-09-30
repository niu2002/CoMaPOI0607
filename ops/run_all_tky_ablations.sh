#!/bin/bash
set -e
cd /mnt/workspace/GeoSemID

mkdir -p results/tky/full_model

MASTER_CACHE="results/tky/full_model/poi_predictions.json"

echo "=================================================="
echo "🎯 TKY 900 全量 6 大消融实验全自动化执行"
echo "Master Cache: $MASTER_CACHE"
echo "=================================================="

# 通用函数：带 HSID
run_ablation_hsid() {
    NAME=$1
    EXTRA_ARGS=$2
    echo ""
    echo "=================================================="
    echo ">>> Running Ablation: $NAME"
    echo ">>> Args: $EXTRA_ARGS"
    echo "=================================================="
    # If already completed 900 samples, skip
    if [ -f "results/tky/$NAME/poi_predictions.json" ]; then
        count=$(grep -o '"predicted_poi_ids"' "results/tky/$NAME/poi_predictions.json" | wc -l)
        if [ "$count" -ge 900 ]; then
            echo "[INFO] $NAME already completed with $count samples, skipping."
            return 0
        fi
    fi
    python3 inference_forward_new.py \
      --dataset tky \
      --num_samples 900 \
      --batch_size 4 \
      --candidate_fusion_strategy rrf \
      --fused_candidate_top_k 50 \
      --use_hsid \
      --strategy expertrag \
      --agent3_base_url "http://localhost:7863/v1" \
      --agent3_api_key "EMPTY" \
      --agent3_api "qwen3-8b" \
      --agent3_max_tokens 1024 \
      --test_interval 200 \
      --load_pf_output \
      --saved_results_path "$MASTER_CACHE" \
      --save_name "$NAME" \
      --store_save_name \
      $EXTRA_ARGS
}

# 通用函数：不带 HSID
run_ablation_wo_hsid() {
    NAME=$1
    EXTRA_ARGS=$2
    echo ""
    echo "=================================================="
    echo ">>> Running Ablation (w/o HSID): $NAME"
    echo ">>> Args: $EXTRA_ARGS"
    echo "=================================================="
    if [ -f "results/tky/$NAME/poi_predictions.json" ]; then
        count=$(grep -o '"predicted_poi_ids"' "results/tky/$NAME/poi_predictions.json" | wc -l)
        if [ "$count" -ge 900 ]; then
            echo "[INFO] $NAME already completed with $count samples, skipping."
            return 0
        fi
    fi
    python3 inference_forward_new.py \
      --dataset tky \
      --num_samples 900 \
      --batch_size 4 \
      --candidate_fusion_strategy rrf \
      --fused_candidate_top_k 50 \
      --strategy expertrag \
      --agent3_base_url "http://localhost:7863/v1" \
      --agent3_api_key "EMPTY" \
      --agent3_api "qwen3-8b" \
      --agent3_max_tokens 1024 \
      --test_interval 200 \
      --load_pf_output \
      --saved_results_path "$MASTER_CACHE" \
      --save_name "$NAME" \
      --store_save_name \
      $EXTRA_ARGS
}

# ① w/o Spatial（去除空间专家）
run_ablation_hsid "ablation_tky_wo_spatial" "--rrf_weights geo_near=0.0"

# ② w/o Transition（去除转移专家）
run_ablation_hsid "ablation_tky_wo_transition" "--rrf_weights history_recent=0.0,history_freq=0.0"

# ③ w/o Temporal（去除时间/频次专家）
run_ablation_hsid "ablation_tky_wo_temporal" "--rrf_weights category_similar=0.0,popular_global=0.0"

# ④ w/o Semantic（去除语义 RAG 专家）
run_ablation_hsid "ablation_tky_wo_semantic" "--rrf_weights rag_top100=0.0"

# ⑤ w/o Router（去除动态路由，使用朴素 Union 召回）
run_ablation_hsid "ablation_tky_wo_router" "--candidate_fusion_strategy union"

# ⑥ w/o HSID（去除分层语义标识，使用传统 POI ID）
run_ablation_wo_hsid "ablation_tky_wo_hsid" ""

# 汇总生成完整表格
python3 ops/summarize_interval_metrics.py --results_dir results/tky

echo "=================================================="
echo "🎉 所有 6 大 TKY 900 消融实验执行完毕！"
echo "=================================================="

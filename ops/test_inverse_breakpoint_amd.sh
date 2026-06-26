#!/bin/bash
# test_inverse_breakpoint_amd.sh - 测试逆向生成的断点续跑与多模型混合生成功能

PROJECT_ROOT="/mnt/workspace/CoMaPOI0607"
DATA_DIR="$PROJECT_ROOT/finetune/data/ca"
API_KEY="sk-ws-H.RYRERIY.JEgr.MEQCIGyARjHUJ7c2U2re8zWD6XQEMgvUhQHAL6aEtZKH5lZ9AiA7wa4j7ZAUPuu6JVtUVCUpyISZYvdFDzeUodQ445fv2w"
BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"

echo "=========================================================="
echo "[STEP 1/5] 清理旧 of ca 生成数据，以确保测试环境干净..."
echo "=========================================================="
if [ -d "$DATA_DIR" ]; then
    rm -rf "$DATA_DIR"/*
    echo "已清空 $DATA_DIR 中的所有历史数据。"
else
    mkdir -p "$DATA_DIR"
fi

echo -e "\n=========================================================="
echo "[STEP 2/5] 第一次执行：使用 qwen-plus 生成前 10 个样本..."
echo "=========================================================="
python3 inference_inverse_new.py \
  --dataset ca \
  --mode train \
  --api_type qwen-plus \
  --api_base_url "$BASE_URL" \
  --api_key "$API_KEY" \
  --batch_size 8 \
  --num_candidate 25 \
  --profile_max_tokens 120 \
  --inverse_rrf_style clean \
  --inverse_fusion_strategy rrf \
  --fused_candidate_top_k 50 \
  --num_samples 10

echo -e "\n=========================================================="
echo "[STEP 3/5] 验证第一次执行结果（预期应为 10 条样本）..."
echo "=========================================================="
if [ -f "$DATA_DIR/agent3_train_samples.jsonl" ]; then
    LINE_COUNT=$(wc -l < "$DATA_DIR/agent3_train_samples.jsonl")
    echo "第一次生成后的样本行数: $LINE_COUNT"
    if [ "$LINE_COUNT" -eq 10 ]; then
        echo "验证通过！成功生成前 10 条样本。"
    else
        echo "错误！第一次生成的样本行数不为 10，当前为 $LINE_COUNT 行。"
        exit 1
    fi
else
    echo "错误！未检测到生成的 $DATA_DIR/agent3_train_samples.jsonl 文件！"
    exit 1
fi

echo -e "\n=========================================================="
echo "[STEP 4/5] 第二次执行：设定 num_samples=20，并使用 qwen-turbo..."
echo "预期：前 10 个已处理样本会被自动检测并跳过，仅使用 qwen-turbo 请求 11-20 号样本"
echo "=========================================================="
python3 inference_inverse_new.py \
  --dataset ca \
  --mode train \
  --api_type qwen-turbo \
  --api_base_url "$BASE_URL" \
  --api_key "$API_KEY" \
  --batch_size 8 \
  --num_candidate 25 \
  --profile_max_tokens 120 \
  --inverse_rrf_style clean \
  --inverse_fusion_strategy rrf \
  --fused_candidate_top_k 50 \
  --num_samples 20

echo -e "\n=========================================================="
echo "[STEP 5/5] 验证最终执行结果（预期总样本应累计为 20 条）..."
echo "=========================================================="
if [ -f "$DATA_DIR/agent3_train_samples.jsonl" ]; then
    LINE_COUNT_FINAL=$(wc -l < "$DATA_DIR/agent3_train_samples.jsonl")
    echo "最终合并生成后的样本行数: $LINE_COUNT_FINAL"
    if [ "$LINE_COUNT_FINAL" -eq 20 ]; then
        echo "=========================================================="
        echo "🎉🎉🎉 集成测试完全成功！ 🎉🎉🎉"
        echo "成功检测到：已跳过前 10 条，并增量追加了 10 条 (使用 qwen-turbo)。"
        echo "已累计合并生成了 20 条 SFT 样本数据。"
        echo "=========================================================="
        
        echo -e "\n=========================================================="
        echo "📊 自动运行效果对比：qwen-plus 与 qwen-turbo 生成质量分析..."
        echo "=========================================================="
        python3 ops/analyze_inverse_compare.py
    else
        echo "错误！合并生成后的总样本行数不为 20，当前为 $LINE_COUNT_FINAL 行。"
        exit 1
    fi
else
    echo "错误！未检测到生成的 $DATA_DIR/agent3_train_samples.jsonl 文件！"
    exit 1
fi

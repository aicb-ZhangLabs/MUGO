#!/bin/bash

# index csv 所在的目录
INDEX_DIR="/home/dongbos/Combine_optim_Borzoi_SNP/src/run_scripts/job_indices"
SCRIPT_TO_RUN="/home/dongbos/Combine_optim_Borzoi_SNP/src/run_scripts/run_job_array.sh"

# 只跑这两个 Tissue
TISSUES=("blood" "brain")

for tissue in "${TISSUES[@]}"; do
    FILE="${INDEX_DIR}/indices_${tissue}.csv"
    
    if [ ! -f "$FILE" ]; then
        echo "⚠️ Warning: Index file not found for $tissue: $FILE"
        continue
    fi
    
    # 计算 CSV 有多少行数据 (减去 1 行 header)
    TOTAL_LINES=$(wc -l < "$FILE")
    NUM_JOBS=$((TOTAL_LINES - 1))
    
    if [ "$NUM_JOBS" -le 0 ]; then
        echo "⚠️ No genes found in $FILE"
        continue
    fi

    echo "========================================"
    echo "📦 Submitting Batch for Tissue: $tissue"
    echo "📄 File: $FILE"
    echo "📊 Total Genes: $NUM_JOBS"
    echo "========================================"

    # 提交 Array 任务
    # --array=1-100%10 意思是：
    # 任务编号 1 到 100
    # %10 表示最多同时运行 10 个任务 (并发限制，防止占满集群)
    # 你可以根据集群空闲情况调整 % 后面的数字，比如 %20 或 %50
    
    sbatch --array=1-${NUM_JOBS}%20 \
           --job-name="${tissue}_MultiOpt" \
           "$SCRIPT_TO_RUN" "$tissue"
           
    echo "✅ Submitted!"
    echo ""
done
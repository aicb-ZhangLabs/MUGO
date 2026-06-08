#!/bin/bash
#SBATCH --job-name=Random_Search
#SBATCH --array=0-30%5                    # <--- 限制并发数为显卡数量 (例如10张卡)
#SBATCH --output=/home/dongbos/Combine_optim_Borzoi_SNP/jobs/logs/%x_%A_%a.out
#SBATCH --error=/home/dongbos/Combine_optim_Borzoi_SNP/jobs/logs/%x_%A_%a.err
#SBATCH --partition=zhanglab.p
#SBATCH --time=04:00:00                      # 200次推理虽然比遍历所有SNP快，但也需要时间，给4小时比较稳
#SBATCH --gres=gpu:1                         # 【必须】申请 1 张 GPU
#SBATCH --cpus-per-task=4                    # 配套 CPU 核
#SBATCH --mem=32G                            # 模型推理需要较大内存
#SBATCH --nodes=1
#SBATCH --nodelist=laniakea                    # <--- 建议指定 GPU 节点 (galaxy/laniakea)

export HDF5_USE_FILE_LOCKING='FALSE'

echo "Activating Conda environment..."
eval "$(conda shell.bash hook)"
conda activate meta-borzoi

GENE_INDEX=$SLURM_ARRAY_TASK_ID
K_VAL=10
TRIALS=200

# 脚本路径
# 注意：我使用了你第一行提供的路径 random_search.py
SCRIPT_PATH="/home/dongbos/Combine_optim_Borzoi_SNP/src/baseline_benchmark/random_search.py"

# 定义要跑的 Tissue
# TISSUES=("blood" "brain" "liver" "heart" "muscle" "Pancreas")
TISSUES=("blood") # 调试用

echo "=========================================================="
echo "🚀 Starting Random Search Baseline for Gene Index $GENE_INDEX"
echo "   Config: K=$K_VAL, Trials=$TRIALS"
echo "   Node: $SLURMD_NODENAME"
echo "   GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "=========================================================="

# --- 循环运行所有 Tissue ---
for TISSUE in "${TISSUES[@]}"; do
    echo "----------------------------------------------------------"
    echo "   -> [$(date +%H:%M:%S)] Running Random Search for: $TISSUE ..."
    
    # 串行执行，共用 GPU
    python -u $SCRIPT_PATH --index $GENE_INDEX --tissue $TISSUE --k $K_VAL --trials $TRIALS
    
    if [ $? -eq 0 ]; then
        echo "   ✅ $TISSUE finished."
    else
        echo "   ❌ $TISSUE failed."
    fi
done

echo "🎉 All tissues finished for Gene Index $GENE_INDEX at: $(date)"
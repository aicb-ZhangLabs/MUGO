#!/bin/bash
#SBATCH --job-name=Borzoi_Ablation
#SBATCH --array=31-100%1                    # <--- 【注意】并发数受限于显卡数量。如果你有10张空闲卡，就写%10
#SBATCH --output=/home/dongbos/Combine_optim_Borzoi_SNP/jobs/logs/%x_%A_%a.out
#SBATCH --error=/home/dongbos/Combine_optim_Borzoi_SNP/jobs/logs/%x_%A_%a.err
#SBATCH --partition=zhanglab.p               # <--- 确保分区里有GPU节点
#SBATCH --time=04:00:00                      # Ablation比较慢，给4小时以防万一（取决于SNP数量）
#SBATCH --gres=gpu:1                         # 【必须】每个任务申请1张GPU
#SBATCH --cpus-per-task=4                    # GPU任务配套4个CPU核做数据加载
#SBATCH --mem=32G                            # Borzoi模型比较吃内存，32G比较稳
#SBATCH --nodes=1
#SBATCH --nodelist=laniakea                    # <--- 指定GPU节点 (galaxy/laniakea)，或者去掉这行让系统自动分配

export HDF5_USE_FILE_LOCKING='FALSE'

echo "Activating Conda environment..."
eval "$(conda shell.bash hook)"
conda activate meta-borzoi

GENE_INDEX=$SLURM_ARRAY_TASK_ID

# 你的脚本路径
SCRIPT_PATH="/home/dongbos/Combine_optim_Borzoi_SNP/src/baseline_benchmark/feature_ablation.py"

# 定义要跑的 Tissue
# 注意：Ablation 跑所有 Tissue 会比较慢（6倍时间）。
# 如果想快点，可以先只跑 blood，或者把 Time 调大到 12:00:00
# TISSUES=("blood" "brain" "liver" "heart" "muscle" "Pancreas")
TISSUES=("blood") # 调试用

echo "=========================================================="
echo "🚀 Starting Feature Ablation for Gene Index $GENE_INDEX"
echo "   Node: $SLURMD_NODENAME"
echo "   GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "=========================================================="

# --- 循环运行所有 Tissue ---
for TISSUE in "${TISSUES[@]}"; do
    echo "----------------------------------------------------------"
    echo "   -> [$(date +%H:%M:%S)] Running Ablation for: $TISSUE ..."
    
    # 必须串行跑，因为共用这一块 GPU
    python -u $SCRIPT_PATH --index $GENE_INDEX --tissue $TISSUE
    
    if [ $? -eq 0 ]; then
        echo "   ✅ $TISSUE finished."
    else
        echo "   ❌ $TISSUE failed (Check OOM or Index Error)."
    fi
done

echo "🎉 All tissues finished for Gene Index $GENE_INDEX at: $(date)"
#!/bin/bash
#SBATCH --job-name=MVP_3090_Seq
#SBATCH --array=2360-2999%4                        # <--- 一次发6个任务，正好把6张3090占满
#SBATCH --output=/home/dongbos/Combine_optim_Borzoi_SNP/jobs/logs/%x_%A_%a.out
#SBATCH --error=/home/dongbos/Combine_optim_Borzoi_SNP/jobs/logs/%x_%A_%a.err
#SBATCH --partition=zhanglab.p               # <--- 记得确认包含3090的partition名字
#SBATCH --time=3-00:00:00                     # 3090比H100慢，给3小时很稳
#SBATCH --nodelist=galaxy                  # <--- A6000 节点
#SBATCH --gres=gpu:1                         # 每个任务领1张卡
#SBATCH --cpus-per-task=4                    # 3090一般配的CPU核没那么多，给4个够了
#SBATCH --mem=32G                            # 内存给32G
#SBATCH --nodes=1

export HDF5_USE_FILE_LOCKING='FALSE'

echo "Activating Conda environment..."
eval "$(conda shell.bash hook)"
conda activate meta-borzoi

GENE_INDEX=$SLURM_ARRAY_TASK_ID
K_VAL=10 

echo "Running on Node: $SLURMD_NODENAME"
echo "GPU Info: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

SCRIPT_PATH="/home/dongbos/Combine_optim_Borzoi_SNP/src/train_model/MVP_multi_head.py"

# 定义要跑的 6 个 Tissue
# TISSUES=("blood" "brain" "liver" "heart" "muscle" "Pancreas")
TISSUES=("kidney" "lung")

echo "=========================================================="
echo "🚀 Starting 3090 Execution for Gene $GENE_INDEX"
echo "=========================================================="

# --- ♻️ 必须串行 (3090塞不下两个) ---
for TISSUE in "${TISSUES[@]}"; do
    echo "----------------------------------------------------------"
    echo "   -> [$(date +%H:%M:%S)] Processing: $TISSUE ..."
    
    # ⚠️ 必须去掉 &，必须串行！
    python -u $SCRIPT_PATH --index $GENE_INDEX --k $K_VAL --tissue $TISSUE
    
    if [ $? -eq 0 ]; then
        echo "   ✅ $TISSUE finished."
    else
        echo "   ❌ $TISSUE failed (OOM or Error)."
    fi
done

echo "🎉 All tissues finished for Gene Index $GENE_INDEX at: $(date)"
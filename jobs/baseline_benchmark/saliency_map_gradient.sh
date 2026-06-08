#!/bin/bash
#SBATCH --job-name=Saliency_Grad
#SBATCH --array=31-3000%4                    # <--- 限制并发数为显卡数量 (例如10张卡)
#SBATCH --output=/home/dongbos/Combine_optim_Borzoi_SNP/jobs/logs/%x_%A_%a.out
#SBATCH --error=/home/dongbos/Combine_optim_Borzoi_SNP/jobs/logs/%x_%A_%a.err
#SBATCH --partition=zhanglab.p
#SBATCH --time=03:00:00                      # 梯度计算通常比完整的组合优化快，但比单纯推理慢，3小时足够
#SBATCH --gres=gpu:1                         # 【必须】申请 1 张 GPU 用于反向传播
#SBATCH --cpus-per-task=4                    # 配套 CPU 核
#SBATCH --mem=32G                            # 梯度计算需要存储计算图，内存不能太小
#SBATCH --nodes=1
#SBATCH --nodelist=voyager                    # <--- 建议指定 GPU 节点 (galaxy/laniakea)

export HDF5_USE_FILE_LOCKING='FALSE'

echo "Activating Conda environment..."
eval "$(conda shell.bash hook)"
conda activate meta-borzoi

GENE_INDEX=$SLURM_ARRAY_TASK_ID

# 脚本路径
SCRIPT_PATH="/home/dongbos/Combine_optim_Borzoi_SNP/src/baseline_benchmark/saliency_map_gradient_based.py"

# 定义要跑的 Tissue
# TISSUES=("blood" "brain" "liver" "heart" "muscle" "Pancreas")
# TISSUES=("blood" "brain") # 调试用
TISSUES=("blood" "brain" "liver" "heart" "muscle" "Pancreas" "lung" "kidney")

echo "=========================================================="
echo "🚀 Starting Gradient Saliency Map for Gene Index $GENE_INDEX"
echo "   Node: $SLURMD_NODENAME"
echo "   GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "=========================================================="

# --- 循环运行所有 Tissue ---
for TISSUE in "${TISSUES[@]}"; do
    echo "----------------------------------------------------------"
    echo "   -> [$(date +%H:%M:%S)] Calculating Gradients for: $TISSUE ..."
    
    # 串行执行，共用 GPU
    python -u $SCRIPT_PATH --index $GENE_INDEX --tissue $TISSUE --modality RNA  # (choose from 'RNA', 'ATAC', 'CAGE', 'DNAse', 'ChIP')
    
    if [ $? -eq 0 ]; then
        echo "   ✅ $TISSUE finished."
    else
        echo "   ❌ $TISSUE failed."
    fi
done

echo "🎉 All tissues finished for Gene Index $GENE_INDEX at: $(date)"
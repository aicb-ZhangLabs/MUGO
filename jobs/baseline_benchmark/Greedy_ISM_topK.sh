#!/bin/bash
#SBATCH --job-name=Greedy_Search
#SBATCH --array=31-100%4                    # <--- 限制并发数为显卡数量 (例如10张卡)
#SBATCH --output=/home/dongbos/Combine_optim_Borzoi_SNP/jobs/logs/%x_%A_%a.out
#SBATCH --error=/home/dongbos/Combine_optim_Borzoi_SNP/jobs/logs/%x_%A_%a.err
#SBATCH --partition=zhanglab.p
#SBATCH --time=04:00:00                      # Greedy Search 计算量较大，给4小时比较稳妥
#SBATCH --gres=gpu:1                         # 【必须】申请 1 张 GPU
#SBATCH --cpus-per-task=4                    # 配套 CPU 核
#SBATCH --mem=32G                            # 模型推理需要较大内存
#SBATCH --nodes=1
#SBATCH --nodelist=voyager                    # <--- 建议指定 GPU 节点 (galaxy/laniakea)

export HDF5_USE_FILE_LOCKING='FALSE'

echo "Activating Conda environment..."
eval "$(conda shell.bash hook)"
conda activate meta-borzoi

GENE_INDEX=$SLURM_ARRAY_TASK_ID

# 脚本路径
SCRIPT_PATH="/home/dongbos/Combine_optim_Borzoi_SNP/src/baseline_benchmark/Greedy_ISM_topK_search.py"

# 定义要跑的 Tissue
# TISSUES=("blood" "brain" "liver" "heart" "muscle" "Pancreas")
TISSUES=("blood") # 调试用

echo "=========================================================="
echo "🚀 Starting Greedy ISM Search for Gene Index $GENE_INDEX"
echo "   Node: $SLURMD_NODENAME"
echo "   GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "=========================================================="

# --- 循环运行所有 Tissue ---
for TISSUE in "${TISSUES[@]}"; do
    echo "----------------------------------------------------------"
    echo "   -> [$(date +%H:%M:%S)] Running Greedy Search for: $TISSUE ..."
    
    # 串行执行，共用 GPU
    python -u $SCRIPT_PATH --index $GENE_INDEX --tissue $TISSUE
    
    if [ $? -eq 0 ]; then
        echo "   ✅ $TISSUE finished."
    else
        echo "   ❌ $TISSUE failed."
    fi
done

echo "🎉 All tissues finished for Gene Index $GENE_INDEX at: $(date)"
#!/bin/bash
#SBATCH --job-name=H100ATAC_Seq              # <--- 改名了，区分 RNA 和 ATAC
#SBATCH --array=10-1300%3                     # <--- Array 范围，根据需要调整
#SBATCH --output=/home/dongbos/Combine_optim_Borzoi_SNP/jobs/logs/%x_%A_%a.out
#SBATCH --error=/home/dongbos/Combine_optim_Borzoi_SNP/jobs/logs/%x_%A_%a.err
#SBATCH --partition=zhanglab.p
#SBATCH --nodelist=voyager                  # <--- H100 节点
#SBATCH --time=3-00:00:00                    # <--- ATAC 计算量和 RNA 差不多，时间管够
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --nodes=1

# --- 错误处理 ---
# set -e 

export HDF5_USE_FILE_LOCKING='FALSE'

# --- 加载环境 ---
echo "Activating Conda environment..."
eval "$(conda shell.bash hook)"
conda activate meta-borzoi

# --- 获取当前的 Gene Index ---
GENE_INDEX=$SLURM_ARRAY_TASK_ID
K_VAL=10 

# --- 调试信息 ---
echo "Master Job ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID (Gene Index): $GENE_INDEX"
echo "Running on Node: $SLURMD_NODENAME"

# --- ⚠️⚠️⚠️ 核心修改：ATAC 脚本路径 ⚠️⚠️⚠️ ---
SCRIPT_PATH="/home/dongbos/Combine_optim_Borzoi_SNP/src/train_model/multi_head_borzoi_ATAC.py"

# --- 定义要跑的 6 个 Tissue ---
# 既然是新的一轮 ATAC 实验，建议 6 个全跑。
# 如果只想跑特定几个，像之前一样注释掉即可。
TISSUES=("blood" "brain" "liver" "heart" "muscle" "Pancreas") 

echo "=========================================================="
echo "🚀 Starting ATAC Sequential Execution for Gene $GENE_INDEX"
echo "=========================================================="

# --- ♻️ 串行循环 ---
for TISSUE in "${TISSUES[@]}"; do
    echo "----------------------------------------------------------"
    echo "   -> [$(date +%H:%M:%S)] Processing ATAC: $TISSUE ..."
    
    # 运行 Python 脚本
    python -u $SCRIPT_PATH --index $GENE_INDEX --k $K_VAL --tissue $TISSUE
    
    # 检查状态
    if [ $? -eq 0 ]; then
        echo "   ✅ $TISSUE (ATAC) finished successfully."
    else
        echo "   ❌ $TISSUE (ATAC) failed!"
    fi
done

echo "=========================================================="
echo "🎉 All tissues (ATAC) finished for Gene Index $GENE_INDEX at: $(date)"
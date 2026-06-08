#!/bin/bash
#SBATCH --job-name=MVP_3090_Seq             # <--- 改个名，一眼看出是串行
#SBATCH --array=2360-2999%4                        # <--- 你的 array 设定
#SBATCH --output=/home/dongbos/Combine_optim_Borzoi_SNP/jobs/logs/%x_%A_%a.out
#SBATCH --error=/home/dongbos/Combine_optim_Borzoi_SNP/jobs/logs/%x_%A_%a.err
#SBATCH --partition=zhanglab.p
#SBATCH --nodelist=galaxy                  # <--- A6000 节点
#SBATCH --time=3-00:00:00                    # <--- 串行跑6个器官其实很快(约10-15分钟)，给2小时绰绰有余
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4                    # <--- 串行虽然只跑1个，但多给点CPU核读数据会快很多
#SBATCH --mem=36G                            # <--- 给足内存，防止 System RAM 不够
#SBATCH --nodes=1

# --- 错误处理 ---
# set -e  <--- 保持注释

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

# --- 运行代码 ---
SCRIPT_PATH="/home/dongbos/Combine_optim_Borzoi_SNP/src/train_model/MVP_multi_head_enformer_CAGE.py"

# 定义要跑的 6 个 Tissue
# TISSUES=("blood" "brain" "liver" "heart" "muscle" "Pancreas")
# TISSUES=("blood" "brain") # 先跑3个
TISSUES=("liver" "heart" "muscle" "Pancreas") 

echo "=========================================================="
echo "🚀 Starting Sequential Execution for Gene $GENE_INDEX"
echo "=========================================================="

# --- ♻️ 串行循环 (最稳模式) ---
for TISSUE in "${TISSUES[@]}"; do
    echo "----------------------------------------------------------"
    echo "   -> [$(date +%H:%M:%S)] Processing: $TISSUE ..."
    
    # ⚠️ 去掉了 '&' 符号，这会让 bash 等待 python 跑完再跑下一个
    python -u $SCRIPT_PATH --index $GENE_INDEX --k $K_VAL --tissue $TISSUE
    
    # 检查上一步的退出状态（可选，只打印不退出）
    if [ $? -eq 0 ]; then
        echo "   ✅ $TISSUE finished successfully."
    else
        echo "   ❌ $TISSUE failed!"
    fi
done

echo "=========================================================="
echo "🎉 All tissues finished for Gene Index $GENE_INDEX at: $(date)"
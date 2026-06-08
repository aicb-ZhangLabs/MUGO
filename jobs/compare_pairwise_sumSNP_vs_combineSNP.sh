#!/bin/bash
#SBATCH --job-name=Compare_SNP_Sum_vs_All    # <--- 任务名称
#SBATCH --array=0-99%6                    # <--- ⚠️ 关键: 修改这里！范围 0-2999，%20 表示同时只跑 20 个任务
#SBATCH --output=/home/dongbos/Combine_optim_Borzoi_SNP/jobs/logs/%x_%A_%a.out
#SBATCH --error=/home/dongbos/Combine_optim_Borzoi_SNP/jobs/logs/%x_%A_%a.err
#SBATCH --partition=zhanglab.p
#SBATCH --nodelist=laniakea                  # <--- 沿用你的设置
#SBATCH --time=02:00:00                      # <--- 分析任务通常比训练快，2小时通常足够 (每个基因)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G                            # <--- Borzoi 推理需要较大显存和内存，32G 比较稳
#SBATCH --nodes=1

# --- 错误处理 ---
set -e

# 避免 HDF5 文件锁问题
export HDF5_USE_FILE_LOCKING='FALSE'

# --- 加载环境 ---
echo "Activating Conda environment..."
eval "$(conda shell.bash hook)"
conda activate meta-borzoi

# --- 获取当前的 Gene Index ---
# SLURM_ARRAY_TASK_ID 会自动对应 array 中的数字
GENE_INDEX=$SLURM_ARRAY_TASK_ID

# --- 调试信息 ---
echo "Master Job ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID (Gene Index): $GENE_INDEX"
echo "Running on Node: $SLURMD_NODENAME"

# --- 脚本路径 (修改为你指定的分析脚本) ---
SCRIPT_PATH="/home/dongbos/Combine_optim_Borzoi_SNP/src/interpretability/compare_pairwise_sumSNP_vs_combineSNP.py"

# --- 运行代码 ---
echo "Starting analysis for index $GENE_INDEX..."
python -u $SCRIPT_PATH --index $GENE_INDEX

echo "Analysis for Gene Index $GENE_INDEX finished at: $(date)"
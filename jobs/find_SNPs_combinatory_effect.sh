#!/bin/bash
#SBATCH --job-name=Scan_Syn_N5             # <--- 任务名
#SBATCH --array=4-5                        # <--- 关键：0-5 对应 6 个 Tissue，同时发 6 个任务
#SBATCH --output=/home/dongbos/Combine_optim_Borzoi_SNP/jobs/logs/%x_%A_%a.out
#SBATCH --error=/home/dongbos/Combine_optim_Borzoi_SNP/jobs/logs/%x_%A_%a.err
#SBATCH --partition=zhanglab.p
#SBATCH --nodelist=laniakea                  # <--- 建议指定 GPU 节点 (galaxy/laniakea)
#SBATCH --time=10:00:00                    # 扫描 3000 个基因，6 小时绰绰有余
#SBATCH --gres=gpu:1                       # 每个任务领 1 张卡
#SBATCH --cpus-per-task=4                  # 4 核 CPU 够用了
#SBATCH --mem=32G                          # 32G 内存足够 (推理比训练省内存)
#SBATCH --nodes=1

export HDF5_USE_FILE_LOCKING='FALSE'

# --- 加载环境 ---
echo "Activating Conda environment..."
eval "$(conda shell.bash hook)"
conda activate meta-borzoi

# --- 核心逻辑：利用 Array ID 映射 Tissue ---
# 定义 Tissue 数组 (注意顺序，下标从 0 开始)
TISSUES=("blood" "brain" "liver" "heart" "muscle" "Pancreas")

# 获取当前任务对应的 Tissue
CURRENT_TISSUE=${TISSUES[$SLURM_ARRAY_TASK_ID]}

# 脚本路径 (请确保这就是你刚才修改好的那个带 cudnn fix 的脚本)
SCRIPT_PATH="/home/dongbos/Combine_optim_Borzoi_SNP/src/interpretability/find_gene_additive_redundant_synagey_topN_SNP.py"

# 参数设置
TOP_N=5

echo "=========================================================="
echo "🚀 Starting Interaction Scan Task $SLURM_ARRAY_TASK_ID"
echo "   Target Tissue: $CURRENT_TISSUE"
echo "   Node: $SLURMD_NODENAME"
echo "   GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "=========================================================="

# --- 运行 Python 脚本 ---
# 每个任务只跑这就行了，它会自动扫描该 Tissue 文件夹下的所有基因
python -u $SCRIPT_PATH --tissue $CURRENT_TISSUE --n $TOP_N

if [ $? -eq 0 ]; then
    echo "✅ Scan for $CURRENT_TISSUE finished successfully."
else
    echo "❌ Scan for $CURRENT_TISSUE failed."
fi

echo "🎉 Job finished at: $(date)"
#!/bin/bash
#SBATCH --job-name=Scan_Interaction_Full     # <--- 任务名称
#SBATCH --output=/home/dongbos/Combine_optim_Borzoi_SNP/jobs/logs/%x_%j.out
#SBATCH --error=/home/dongbos/Combine_optim_Borzoi_SNP/jobs/logs/%x_%j.err
#SBATCH --partition=zhanglab.p
#SBATCH --nodelist=voyager                  # <--- 沿用你的设置
#SBATCH --time=72:00:00                      # <--- 既然是跑全量且不急，给足3天时间，防止断掉
#SBATCH --gres=gpu:1                         # <--- 必须有 GPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G                            # <--- 32G 内存足够 Borzoi 推理
#SBATCH --nodes=1
#SBATCH --ntasks=1                           # <--- 单任务 (脚本内部自己循环)

# --- 错误处理 ---
set -e

# 避免 HDF5 文件锁问题
export HDF5_USE_FILE_LOCKING='FALSE'

# --- 加载环境 ---
echo "Activating Conda environment..."
eval "$(conda shell.bash hook)"
conda activate meta-borzoi

# --- 脚本路径 ---
# 确保这里的路径指向你刚才修改好的 scan_interaction_modes_pairwise.py
SCRIPT_PATH="/home/dongbos/Combine_optim_Borzoi_SNP/src/interpretability/find_gene_additive_redundant_synagey.py"

# --- 运行代码 ---
echo "Starting Full Interaction Scan at $(date)..."
echo "Running script: $SCRIPT_PATH"

# 关键: 
# 1. python -u 让 print 实时输出到 log，不会卡在缓存里
# 2. 不加 --limit 参数，默认跑全量 (Full Mode)
python -u $SCRIPT_PATH

echo "Analysis finished at: $(date)"
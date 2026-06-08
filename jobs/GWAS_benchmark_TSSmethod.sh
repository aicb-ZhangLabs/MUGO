#!/bin/bash
#SBATCH --job-name=GWAS_Stats           # 任务名
#SBATCH --array=0-17%6                    # 关键：6个组织 x 3个K值 = 18个任务 (索引0-17)
#SBATCH --output=/home/dongbos/Combine_optim_Borzoi_SNP/jobs/logs/gwas_stats_%a.out
#SBATCH --error=/home/dongbos/Combine_optim_Borzoi_SNP/jobs/logs/gwas_stats_%a.err
#SBATCH --partition=zhanglab.p          # 分区
#SBATCH --time=01:00:00                 # 每个任务也就跑几千个基因，1小时绰绰有余
#SBATCH --cpus-per-task=4               # 给4个核加速数据读取
#SBATCH --mem=32G                       # GWAS数据较大，给32G防崩
#SBATCH --nodes=1
#SBATCH --nodelist=galaxy 

# --- 环境设置 ---
source /home/dongbos/miniconda3/etc/profile.d/conda.sh
conda activate meta-borzoi

# --- 脚本路径 ---
SCRIPT_PATH="/home/dongbos/Combine_optim_Borzoi_SNP/src/interpretability/fig4_gwas/cal_gwas_enrichment_benchmark_distance_based.py"

# --- 定义参数列表 ---
# 6 个组织
TISSUES=("blood" "liver" "heart" "brain" "muscle" "pancreas")
# 3 个 K 值
KS=(10 20 50)

# --- 🧮 核心逻辑：把 Array ID 映射到参数组合 ---
# 计算总 K 的数量
NUM_KS=${#KS[@]}

# 计算当前 Array ID 对应的索引
# TISSUE_IDX = ID / 3 (整除)
# K_IDX      = ID % 3 (取余)
TISSUE_IDX=$((SLURM_ARRAY_TASK_ID / NUM_KS))
K_IDX=$((SLURM_ARRAY_TASK_ID % NUM_KS))

# 获取对应的参数
CURRENT_TISSUE=${TISSUES[$TISSUE_IDX]}
CURRENT_K=${KS[$K_IDX]}

echo "=========================================================="
echo "🚀 Task ID: $SLURM_ARRAY_TASK_ID"
echo "   Target Tissue: [${CURRENT_TISSUE}]"
echo "   Target K:      [${CURRENT_K}]"
echo "=========================================================="

# --- 运行 Python ---
# 注意：这里去掉了 --test，跑全量数据！
python -u $SCRIPT_PATH \
    --tissue $CURRENT_TISSUE \
    --k $CURRENT_K \
    --mode best

echo "✅ Finished Task $SLURM_ARRAY_TASK_ID"
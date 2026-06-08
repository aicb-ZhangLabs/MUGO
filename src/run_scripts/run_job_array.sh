#!/bin/bash
#SBATCH --job-name=MultiModal_Optim          # 任务名
#SBATCH --output=/home/dongbos/Combine_optim_Borzoi_SNP/jobs/logs/%x_%A_%a.out
#SBATCH --error=/home/dongbos/Combine_optim_Borzoi_SNP/jobs/logs/%x_%A_%a.err
#SBATCH --partition=zhanglab.p               # 分区
#SBATCH --time=12:00:00                      # 12小时足够跑完一个基因的3个模态
#SBATCH --gres=gpu:1                         # 每个任务领1块卡
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# ================= 配置区域 =================

# 接受外部传参：Tissue名字 (blood 或 brain)
TISSUE=$1

if [ -z "$TISSUE" ]; then
    echo "❌ Error: No tissue specified!"
    exit 1
fi

# 路径配置
BASE_DIR="/home/dongbos/Combine_optim_Borzoi_SNP"
SCRIPT_DIR="${BASE_DIR}/src/train_model"
INDEX_DIR="${BASE_DIR}/src/run_scripts/job_indices"
INDEX_FILE="${INDEX_DIR}/indices_${TISSUE}.csv"

# ================= 环境加载 =================
export HDF5_USE_FILE_LOCKING='FALSE'
echo "Activating Conda environment..."
eval "$(conda shell.bash hook)"
conda activate meta-borzoi

# ================= Index 提取逻辑 =================

# SLURM_ARRAY_TASK_ID 代表我们要处理 CSV 的第几行数据
# 因为 CSV 第一行是 Header，所以我们需要读取第 (TASK_ID + 1) 行
LINE_NUM=$((SLURM_ARRAY_TASK_ID + 1))

# 使用 sed 提取特定行
# CSV 格式: Gene,Index (例如: GAPDH,1024)
ROW=$(sed -n "${LINE_NUM}p" "$INDEX_FILE")

if [ -z "$ROW" ]; then
    echo "❌ Error: Could not read line $LINE_NUM from $INDEX_FILE"
    exit 1
fi

# 解析 Gene 和 Index (假设逗号分隔)
GENE_NAME=$(echo "$ROW" | cut -d',' -f1 | tr -d '\r')
GENE_INDEX=$(echo "$ROW" | cut -d',' -f2 | tr -d '\r')

echo "=========================================================="
echo "🚀 Job ID: $SLURM_ARRAY_JOB_ID | Task ID: $SLURM_ARRAY_TASK_ID"
echo "🧬 Target Gene: $GENE_NAME"
echo "🔢 Borzoi Index: $GENE_INDEX"
echo "🫀 Tissue: $TISSUE"
echo "🖥️ Node: $SLURMD_NODENAME"
echo "=========================================================="

# ================= 串行执行三个模态 =================
# 在同一个 GPU 上，跑完 ATAC 接着跑 ChIP 接着跑 DNAse
# 这样省去了反复排队的时间

K_VAL=10

# 1. ATAC
echo "--- [1/3] Running ATAC ---"
python -u ${SCRIPT_DIR}/multi_head_borzoi_ATAC.py \
    --tissue $TISSUE --index $GENE_INDEX --k $K_VAL

# 2. ChIP
echo "--- [2/3] Running ChIP ---"
python -u ${SCRIPT_DIR}/multi_head_borzoi_CHIP.py \
    --tissue $TISSUE --index $GENE_INDEX --k $K_VAL

# 3. DNAse
echo "--- [3/3] Running DNAse ---"
python -u ${SCRIPT_DIR}/multi_head_borzoi_DNAse.py \
    --tissue $TISSUE --index $GENE_INDEX --k $K_VAL

echo "=========================================================="
echo "🎉 All modalities finished for $GENE_NAME ($GENE_INDEX)"
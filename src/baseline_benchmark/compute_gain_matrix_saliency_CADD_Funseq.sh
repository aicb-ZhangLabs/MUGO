#!/bin/bash
#SBATCH --job-name=Benchmark_Summary         # 任务名
#SBATCH --output=/home/dongbos/Combine_optim_Borzoi_SNP/jobs/logs/%x_%j.out
#SBATCH --error=/home/dongbos/Combine_optim_Borzoi_SNP/jobs/logs/%x_%j.err
#SBATCH --partition=zhanglab.p
#SBATCH --nodelist=laniakea                  # 指定 A6000 节点
#SBATCH --time=12:00:00                      # 6小时足够跑几千个基因了
#SBATCH --gres=gpu:1                         # 🔥 必须申请 GPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G                            # 内存给足，防止 pandas 读大文件崩掉
#SBATCH --nodes=1

# --- 错误处理 ---
export HDF5_USE_FILE_LOCKING='FALSE'

# --- 加载环境 ---
echo "Activating Conda environment..."
eval "$(conda shell.bash hook)"
conda activate meta-borzoi

# --- 检查 GPU 状态 (Debug) ---
echo "Checking GPU..."
nvidia-smi
if [ $? -ne 0 ]; then
    echo "❌ Error: NVIDIA GPU not found!"
    exit 1
fi

# --- 脚本路径 ---
SCRIPT_PATH="/home/dongbos/Combine_optim_Borzoi_SNP/src/baseline_benchmark/compute_gain_matrix_saliency_CADD_Funseq.py"

# =================================================================
# 👇👇👇 [配置区域] 请通过 注释/取消注释 选择一组跑 👇👇👇
# =================================================================

# --- 1. 选择 Tissue (只能开一个) ---
# TISSUE="blood"
# TISSUE="brain"
# TISSUE="liver"
# TISSUE="heart"
# TISSUE="muscle"
# TISSUE="pancreas"
# TISSUE="lung"
TISSUE="kidney"

# --- 2. 选择 Modality (只能开一个) ---
MODALITY="RNA"
# MODALITY="ATAC"
# MODALITY="DNAse"
# MODALITY="CAGE"
# MODALITY="ChIP"

# =================================================================
# 👆👆👆 [配置结束] 👆👆👆
# =================================================================

if [ -z "$TISSUE" ] || [ -z "$MODALITY" ]; then
    echo "❌ Error: Please uncomment one TISSUE and one MODALITY."
    exit 1
fi

echo "=========================================================="
echo "🚀 Starting Benchmark Summary Task"
echo "📍 Script: $SCRIPT_PATH"
echo "🎯 Target: Tissue=[$TISSUE] | Modality=[$MODALITY]"
echo "📅 Date: $(date)"
echo "=========================================================="

# --- 运行 Python 脚本 ---
python -u $SCRIPT_PATH \
  --tissue $TISSUE \
  --modality $MODALITY

# --- 结束 ---
if [ $? -eq 0 ]; then
    echo "✅ [SUCCESS] Benchmark finished for $TISSUE - $MODALITY"
else
    echo "❌ [FAILED] Benchmark exited with errors."
    exit 1
fi
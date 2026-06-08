#!/bin/bash

# ================= ⚙️ 配置区域 =================

# Python 脚本的绝对路径 (请确认文件名正确)
PYTHON_SCRIPT="/home/dongbos/Combine_optim_Borzoi_SNP/src/interpretability/newversion_table2/benchmarking_on_geneset.py"

# 输入数据所在的目录 (用于检查文件是否存在)
SUBSET_DIR="/home/dongbos/Combine_optim_Borzoi_SNP/src/interpretability/newversion_table2/top100_highexp_gene"

# 日志目录
LOG_DIR="/home/dongbos/Combine_optim_Borzoi_SNP/src/interpretability/newversion_table2/logs"
mkdir -p $LOG_DIR

# 任务参数
# MODALITIES=("ATAC" "DNAse" "ChIP")
MODALITIES=("CAGE")
# MODALITIES=("RNA")
# 注意：这里用全小写，脚本里会处理文件名的大小写匹配
# TISSUES=("blood" "brain" "liver" "heart" "muscle" "pancreas" "lung" "kidney")
TISSUES=("blood" "brain" "liver" "heart" "muscle" "pancreas")
# TISSUES=("blood" "brain")

# 运行模式
MODE="top100"

# Conda 环境名称
CONDA_ENV="meta-borzoi"

# ================= 🚀 循环提交逻辑 =================

echo "========================================================"
echo "🚀 Batch Submitting Benchmark Jobs (Mode: $MODE)"
echo "   Subset Dir: $SUBSET_DIR"
echo "========================================================"

for MOD in "${MODALITIES[@]}"; do
    for TISSUE in "${TISSUES[@]}"; do
        
        # 1. 构建文件名检查逻辑
        # 因为 Pancreas 有时候是大写，有时候是小写，我们两个都检查一下
        
        # 情况A: 全小写 (e.g., blood)
        FILE_LOWER="${SUBSET_DIR}/top100_high_expr_cache_${MOD}_${TISSUE}.csv"
        # 情况B: 首字母大写 (e.g., Pancreas)
        TISSUE_CAP="$(tr '[:lower:]' '[:upper:]' <<< ${TISSUE:0:1})${TISSUE:1}"
        FILE_CAP="${SUBSET_DIR}/top100_high_expr_cache_${MOD}_${TISSUE_CAP}.csv"

        TARGET_FILE=""
        if [ -f "$FILE_LOWER" ]; then
            TARGET_FILE="$FILE_LOWER"
        elif [ -f "$FILE_CAP" ]; then
            TARGET_FILE="$FILE_CAP"
        fi

        # 2. 如果文件不存在，自动跳过
        if [ -z "$TARGET_FILE" ]; then
            echo "⚠️  [Skipping] Input file not found for: $MOD - $TISSUE"
            continue
        fi

        # 3. 准备 Job Name
        JOB_NAME="Bench_${MOD}_${TISSUE}"

        echo "✅ [Submitting] $MOD - $TISSUE (Found: $(basename $TARGET_FILE))"

        # 4. 生成并提交 SLURM 脚本 (HereDoc 模式)
        sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${LOG_DIR}/%x_%j.out
#SBATCH --error=${LOG_DIR}/%x_%j.err
#SBATCH --partition=zhanglab.p
#SBATCH --time=4:00:00                # Top100 跑得很快，2小时足够
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
# 注意：不指定 nodelist，让 SLURM 自动分配给空闲的节点 (Voyager/Enterprise/Defiant)

# --- 环境设置 ---
export HDF5_USE_FILE_LOCKING='FALSE'
eval "\$(conda shell.bash hook)"
conda activate ${CONDA_ENV}

echo "Running Benchmark for: Modality=${MOD}, Tissue=${TISSUE}, Mode=${MODE}"
echo "Running on Node: \${SLURMD_NODENAME}"

# --- 运行 Python ---
# 注意：我们传给 Python 的 tissue 参数保持小写，因为 Python 内部的 Map Key 是小写
# 如果 Python 脚本加载文件失败，请确保 Python 脚本里有处理 Pancreas 大小写的逻辑
python -u ${PYTHON_SCRIPT} \\
    --tissue ${TISSUE} \\
    --modality ${MOD} \\
    --mode ${MODE}

echo "Done."
EOT

    done
done

echo "========================================================"
echo "🎉 All submission attempts finished."
echo "   Check running jobs with: squeue -u dongbos"
echo "========================================================"


# base benchmarking_setofgenes.sh to run all tasks, not using sbatch. 
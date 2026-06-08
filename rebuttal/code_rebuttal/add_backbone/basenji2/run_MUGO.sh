#!/bin/bash

# ================= ⚙️ 配置区域 =================
# Python 脚本绝对路径 (请确保路径和你存放的位置一致)
PYTHON_SCRIPT="/home/dongbos/Combine_optim_Borzoi_SNP/rebuttal/code_rebuttal/add_backbone/basenji2/basenji2_MUGO_CAGE.py"

# CSV 名单所在的根目录
CSV_DIR="/home/dongbos/Combine_optim_Borzoi_SNP/src/interpretability/newversion_table2/top100_highexp_gene"

# 日志目录
LOG_DIR="/home/dongbos/Combine_optim_Borzoi_SNP/rebuttal/results_rebuttal/add_backbone_model/basenji2/logs"
mkdir -p "$LOG_DIR"

# 实验参数
TISSUES=("blood" "brain")
TOP_N=100
CONDA_ENV="meta-borzoi"

echo "========================================================"
echo "🚀 Submitting Basenji2 MUGO (Top $TOP_N Genes per Tissue)"
echo "========================================================"

for TISSUE in "${TISSUES[@]}"; do
    
    CSV_FILE="${CSV_DIR}/top100_high_expr_cache_CAGE_${TISSUE}.csv"
    
    if [ ! -f "$CSV_FILE" ]; then
        echo "⚠️ Error: CSV not found for $TISSUE -> $CSV_FILE"
        continue
    fi

    # 提取前 100 个基因名 (跳过表头)
    GENE_LIST=$(tail -n +2 "$CSV_FILE" | head -n "$TOP_N" | cut -d',' -f1 | tr '\n' ' ')

    JOB_NAME="bs2_mugo_${TISSUE}"
    echo "✅ [Submitting] Tissue: $TISSUE"

    # 使用 HereDoc 提交 SLURM 任务
    sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${LOG_DIR}/%x_%j.out
#SBATCH --error=${LOG_DIR}/%x_%j.err
#SBATCH --partition=zhanglab.p
#SBATCH --nodelist=laniakea
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks=1

export HDF5_USE_FILE_LOCKING='FALSE'
eval "\$(conda shell.bash hook)"
conda activate ${CONDA_ENV}

echo "Starting Basenji2 MUGO for $TISSUE on node \${SLURMD_NODENAME}"

# 把 bash 里的字符串变成数组
GENES=(${GENE_LIST})

for GENE in "\${GENES[@]}"; do
    echo "▶️ Running Gene: \$GENE"
    python -u ${PYTHON_SCRIPT} \
        --gene "\$GENE" \
        --tissue "${TISSUE}" \
        --k 10
done

echo "🎉 Job ${JOB_NAME} Finished!"
EOT

done

echo "========================================================"
echo "🎉 2 Jobs submitted! Check with: squeue -u $USER"
echo "========================================================"
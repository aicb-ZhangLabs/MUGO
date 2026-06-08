#!/bin/bash

# ================= ⚙️ 配置区域 =================
PYTHON_SCRIPT="/home/dongbos/Combine_optim_Borzoi_SNP/rebuttal/code_rebuttal/satuation_mutagenesis/Funseq_borzoi_RNA_satuation_mutagenesis.py"
CSV_DIR="/home/dongbos/Combine_optim_Borzoi_SNP/src/interpretability/newversion_table2/top100_highexp_gene"

# 日志目录
LOG_DIR="/home/dongbos/Combine_optim_Borzoi_SNP/rebuttal/results_rebuttal/satuation_mutagenesis/logs"
mkdir -p "$LOG_DIR"

# 实验参数
TISSUES=("blood" "brain")
# N_WINDOWS=("1000" "10000" "500000") 
N_WINDOWS=("100" "1000" "10000" "50000" "100000" "500000")
TOP_N=100
CONDA_ENV="meta-borzoi"

echo "========================================================"
echo "🚀 Submitting FunSeq2 Saturation Mutagenesis (Top $TOP_N Genes)"
echo "========================================================"

for TISSUE in "${TISSUES[@]}"; do
    CSV_FILE="${CSV_DIR}/top100_high_expr_cache_RNA_${TISSUE}.csv"
    if [ ! -f "$CSV_FILE" ]; then
        echo "⚠️ Error: CSV not found for $TISSUE -> $CSV_FILE"
        continue
    fi

    GENE_LIST=$(tail -n +2 "$CSV_FILE" | head -n "$TOP_N" | cut -d',' -f1 | tr '\n' ' ')
    JOB_NAME="fsq_sat_${TISSUE}"
    echo "✅ [Submitting] Tissue: $TISSUE"

    sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${LOG_DIR}/%x_%j.out
#SBATCH --error=${LOG_DIR}/%x_%j.err
#SBATCH --partition=zhanglab.p
#SBATCH --nodelist=laniakea
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks=1

export HDF5_USE_FILE_LOCKING='FALSE'
eval "\$(conda shell.bash hook)"
conda activate ${CONDA_ENV}

echo "Starting FunSeq2 Saturation Mutagenesis for $TISSUE"

GENES=(${GENE_LIST})

for GENE in "\${GENES[@]}"; do
    echo "▶️ Running Gene: \$GENE"
    
    for N_VAL in ${N_WINDOWS[@]}; do
        echo "   👉 Querying FunSeq2 for N=\$N_VAL..."
        python -u ${PYTHON_SCRIPT} \
            --gene "\$GENE" \
            --tissue "${TISSUE}" \
            --N \$N_VAL \
            --k 10
    done
done

echo "🎉 Job ${JOB_NAME} Finished!"
EOT

done
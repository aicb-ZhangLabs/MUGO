#!/bin/bash

# ================= ⚙️ 配置区域 =================
PYTHON_SCRIPT="/home/dongbos/Combine_optim_Borzoi_SNP/rebuttal/code_rebuttal/add_backbone/expecto/expecto_saliency.py"

CSV_DIR="/home/dongbos/Combine_optim_Borzoi_SNP/src/interpretability/newversion_table2/top100_highexp_gene"

LOG_DIR="/home/dongbos/Combine_optim_Borzoi_SNP/rebuttal/results_rebuttal/add_backbone_model/expecto/expecto_saliency_slurm_logs"
mkdir -p "$LOG_DIR"

TISSUES=("blood" "brain")
TOP_N=100
CONDA_ENV="meta-borzoi"

echo "========================================================"
echo "🚀 Submitting ExPecto Saliency (Top $TOP_N Genes)"
echo "========================================================"

for TISSUE in "${TISSUES[@]}"; do
    
    CSV_FILE="${CSV_DIR}/top100_high_expr_cache_CAGE_${TISSUE}.csv"
    
    if [ ! -f "$CSV_FILE" ]; then
        echo "⚠️ Error: CSV not found for $TISSUE -> $CSV_FILE"
        continue
    fi

    GENE_LIST=$(tail -n +2 "$CSV_FILE" | head -n "$TOP_N" | cut -d',' -f1 | tr '\n' ' ')

    JOB_NAME="exp_sal_${TISSUE}"
    echo "✅ [Submitting] Tissue: $TISSUE"

    sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${LOG_DIR}/%x_%j.out
#SBATCH --error=${LOG_DIR}/%x_%j.err
#SBATCH --partition=zhanglab.p
#SBATCH --nodelist=laniakea
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks=1

export HDF5_USE_FILE_LOCKING='FALSE'
eval "\$(conda shell.bash hook)"
conda activate ${CONDA_ENV}

echo "Starting ExPecto Saliency for $TISSUE on node \${SLURMD_NODENAME}"

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
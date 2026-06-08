#!/bin/bash

# ================= ⚙️ 配置区域 =================
PYTHON_SCRIPT="/home/dongbos/Combine_optim_Borzoi_SNP/rebuttal/code_rebuttal/satuation_mutagenesis/MUGO_borzoi_RNA_satuation_mutagenesis.py"
LOG_DIR="/home/dongbos/Combine_optim_Borzoi_SNP/rebuttal/results_rebuttal/satuation_mutagenesis/logs_mugo_crispr"
mkdir -p "$LOG_DIR"

# CRISPR validation gene set from filtered input results
GENES=("PLP2" "PRDX2" "GATA1" "NFE2" "FTL" "KLF1" "HDAC6" "FUT1" "NUCB1" "PQBP1" "HNRNPA1" "H1FX" "COPZ1" "BAX" "JUNB" "RPN1" "WDR83OS" "RAD23A" "DNASE2" "DHPS")

TISSUE="blood"
N_VAL="100000"
CONDA_PYTHON="/home/dongbos/miniconda3/envs/meta-borzoi/bin/python"

echo "========================================================"
echo "🚀 Submitting MUGO CRISPR Validation (1 Job per Gene)"
echo "========================================================"

for GENE in "${GENES[@]}"; do
    
    JOB_NAME="mugo_${GENE}_${TISSUE}"
    echo "✅ [Submitting] Gene: $GENE | Tissue: $TISSUE | N: $N_VAL"

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

echo "Running MUGO for \$GENE on node \${SLURMD_NODENAME}"

${CONDA_PYTHON} -u ${PYTHON_SCRIPT} \
    --gene "${GENE}" \
    --tissue "${TISSUE}" \
    --N ${N_VAL} \
    --k 10

echo "🎉 Job ${JOB_NAME} Finished!"
EOT

done

echo "========================================================"
echo "🎉 Submitted ${#GENES[@]} MUGO Jobs! Check with: squeue -u $USER"
echo "========================================================"

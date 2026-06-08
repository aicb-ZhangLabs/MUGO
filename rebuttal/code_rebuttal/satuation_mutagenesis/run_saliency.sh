#!/bin/bash

# ================= ⚙️ 配置区域 =================
# Python 脚本绝对路径 (已更新为 Saliency 版本的路径)
PYTHON_SCRIPT="/home/dongbos/Combine_optim_Borzoi_SNP/rebuttal/code_rebuttal/satuation_mutagenesis/Saliency_borzoi_RNA_satuation_mutagenesis.py"

# CSV 名单所在的根目录
CSV_DIR="/home/dongbos/Combine_optim_Borzoi_SNP/src/interpretability/newversion_table2/top100_highexp_gene"

# 日志目录
LOG_DIR="/home/dongbos/Combine_optim_Borzoi_SNP/rebuttal/results_rebuttal/satuation_mutagenesis/logs"
mkdir -p "$LOG_DIR"

# 实验参数
TISSUES=("blood" "brain")
# N_WINDOWS=("1000" "10000" "500000") # 🔥 同时跑三个尺度，直接出对抗攻击曲线！
N_WINDOWS=("100" "1000" "10000" "50000" "100000" "500000")
TOP_N=100
CONDA_ENV="meta-borzoi"

echo "========================================================"
echo "🚀 Submitting Saliency Saturation Mutagenesis (Top $TOP_N Genes)"
echo "========================================================"

for TISSUE in "${TISSUES[@]}"; do
    
    # ⚠️ 注意这里换成了 _RNA_ ，因为 Borzoi 是 RNA track
    CSV_FILE="${CSV_DIR}/top100_high_expr_cache_RNA_${TISSUE}.csv"
    
    if [ ! -f "$CSV_FILE" ]; then
        echo "⚠️ Error: CSV not found for $TISSUE -> $CSV_FILE"
        continue
    fi

    # 提取前 100 个基因名 (跳过表头)
    GENE_LIST=$(tail -n +2 "$CSV_FILE" | head -n "$TOP_N" | cut -d',' -f1 | tr '\n' ' ')

    # 更改了 Job Name，方便在 slurm 队列里和 MUGO 区分开
    JOB_NAME="sal_sat_${TISSUE}"
    echo "✅ [Submitting] Tissue: $TISSUE"

    # 使用 HereDoc 提交 SLURM 任务
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

echo "Starting Saliency Saturation Mutagenesis for $TISSUE on node \${SLURMD_NODENAME}"

# 把 bash 里的字符串变成数组
GENES=(${GENE_LIST})

for GENE in "\${GENES[@]}"; do
    echo "▶️ Running Gene: \$GENE"
    
    # 🔥 内层循环跑不同的 Window Size
    for N_VAL in ${N_WINDOWS[@]}; do
        echo "   👉 Optimizing Window Size N=\$N_VAL with Saliency..."
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

echo "========================================================"
echo "🎉 2 Jobs submitted! Check with: squeue -u $USER"
echo "========================================================"
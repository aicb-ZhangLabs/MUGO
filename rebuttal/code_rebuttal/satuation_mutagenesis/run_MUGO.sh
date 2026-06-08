#!/bin/bash

# ================= ⚙️ 配置区域 =================
# Python 脚本绝对路径 (已更新为 MUGO 版本的路径)
PYTHON_SCRIPT="/home/dongbos/Combine_optim_Borzoi_SNP/rebuttal/code_rebuttal/satuation_mutagenesis/MUGO_borzoi_RNA_satuation_mutagenesis.py"

# CSV 名单所在的根目录
CSV_DIR="/home/dongbos/Combine_optim_Borzoi_SNP/src/interpretability/newversion_table2/top100_highexp_gene"

# 日志目录
LOG_DIR="/home/dongbos/Combine_optim_Borzoi_SNP/rebuttal/results_rebuttal/satuation_mutagenesis/logs"
mkdir -p "$LOG_DIR"

# 实验参数
TISSUES=("blood" "brain")
# 🔥 替换了 100000 为 131072 (Basenji2 感受野)
N_WINDOWS=("100" "1000" "10000" "50000" "100000" "500000") 

TOP_N=100
CONDA_ENV="meta-borzoi"

echo "========================================================"
echo "🚀 Submitting Borzoi Saturation Mutagenesis (Tissue x N Parallel)"
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

    # 🔥 核心修改：把 N 的循环提到了 sbatch 的外面！
    for N_VAL in "${N_WINDOWS[@]}"; do
        
        # 动态生成 Job Name，比如 bz_sat_blood_N1000
        JOB_NAME="bz_mugo_${TISSUE}_N${N_VAL}"
        echo "✅ [Submitting] Tissue: $TISSUE | Window (N): $N_VAL"

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

echo "Starting Borzoi Saturation Mutagenesis for $TISSUE (N=${N_VAL}) on node \${SLURMD_NODENAME}"

# 把 bash 里的字符串变成数组
GENES=(${GENE_LIST})

for GENE in "\${GENES[@]}"; do
    echo "▶️ Running Gene: \$GENE (N=${N_VAL})"
    
    # 🔥 这里不再循环 N，而是直接使用外层传进来的 N_VAL
    python -u ${PYTHON_SCRIPT} \
        --gene "\$GENE" \
        --tissue "${TISSUE}" \
        --N ${N_VAL} \
        --k 10
done

echo "🎉 Job ${JOB_NAME} Finished!"
EOT

    done
done

echo "========================================================"
echo "🎉 12 Jobs submitted successfully! Check with: squeue -u $USER"
echo "========================================================"
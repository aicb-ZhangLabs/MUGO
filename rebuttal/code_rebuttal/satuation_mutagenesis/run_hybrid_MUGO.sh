#!/bin/bash

# ================= ⚙️ 配置区域 =================
# Python 脚本绝对路径
PYTHON_SCRIPT="/home/dongbos/Combine_optim_Borzoi_SNP/rebuttal/code_rebuttal/satuation_mutagenesis/hybrid_MUGO.py"

# CSV 名单所在的根目录
CSV_DIR="/home/dongbos/Combine_optim_Borzoi_SNP/src/interpretability/newversion_table2/top100_highexp_gene"

# 日志目录
LOG_DIR="/home/dongbos/Combine_optim_Borzoi_SNP/rebuttal/results_rebuttal/satuation_mutagenesis/logs_hybrid"
mkdir -p "$LOG_DIR"

# 实验参数
TISSUES=("blood" "brain")
N_WINDOWS=("100" "1000" "10000" "50000" "100000" "500000") 
TOP_N=100

# 🔥 核心核武器：直接写死 meta-borzoi 环境的 Python 绝对路径！彻底不要 conda activate 了
CONDA_PYTHON="/home/dongbos/miniconda3/envs/meta-borzoi/bin/python"

echo "========================================================"
echo "🚀 Submitting Hybrid MUGO (Saliency + MUGO) Saturation Mutagenesis"
echo "========================================================"

for TISSUE in "${TISSUES[@]}"; do
    
    # 定位名单 CSV 文件
    CSV_FILE="${CSV_DIR}/top100_high_expr_cache_RNA_${TISSUE}.csv"
    
    if [ ! -f "$CSV_FILE" ]; then
        echo "⚠️ Error: CSV not found for $TISSUE -> $CSV_FILE"
        continue
    fi

    # 提取前 100 个基因名 (跳过表头)
    GENE_LIST=$(tail -n +2 "$CSV_FILE" | head -n "$TOP_N" | cut -d',' -f1 | tr '\n' ' ')

    for N_VAL in "${N_WINDOWS[@]}"; do
        
        JOB_NAME="bz_hybrid_${TISSUE}_N${N_VAL}"
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

echo "Starting Hybrid MUGO for $TISSUE (N=${N_VAL}) on node \${SLURMD_NODENAME}"

# 把 bash 里的字符串变成数组
GENES=(${GENE_LIST})

for GENE in "\${GENES[@]}"; do
    echo "▶️ Running Gene: \$GENE (N=${N_VAL})"
    
    # 🔥 直接调用这个带有 torch 的绝对路径 Python 执行
    ${CONDA_PYTHON} -u ${PYTHON_SCRIPT} \
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
echo "🎉 12 Hybrid Jobs submitted! Used Absolute Python Path."
echo "========================================================"
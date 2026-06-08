#!/bin/bash

# ================= ⚙️ 配置区域 =================
# Python 脚本绝对路径
PYTHON_SCRIPT="/home/dongbos/Combine_optim_Borzoi_SNP/rebuttal/code_rebuttal/mugo_RNA_random_seed.py"

# CSV 名单所在的根目录
CSV_DIR="/home/dongbos/Combine_optim_Borzoi_SNP/src/interpretability/newversion_table2/top100_highexp_gene"

# 日志目录
LOG_DIR="/home/dongbos/Combine_optim_Borzoi_SNP/rebuttal/logs_seed_test"
mkdir -p "$LOG_DIR"

# 实验参数
SEEDS=(42 123 2026)
TISSUES=("blood" "brain")
TOP_N=100  # 🔥 直接拉满 100 个！
CONDA_ENV="meta-borzoi"

echo "========================================================"
echo "🚀 Submitting Random Seed Ablation (Top $TOP_N Genes)"
echo "========================================================"

for TISSUE in "${TISSUES[@]}"; do
    
    # 定位名单 CSV 文件
    CSV_FILE="${CSV_DIR}/top100_high_expr_cache_RNA_${TISSUE}.csv"
    
    if [ ! -f "$CSV_FILE" ]; then
        echo "⚠️ Error: CSV not found for $TISSUE -> $CSV_FILE"
        continue
    fi

    # 提取前 100 个基因名 (跳过第一行表头，以逗号分隔取第一列)
    GENE_LIST=$(tail -n +2 "$CSV_FILE" | head -n "$TOP_N" | cut -d',' -f1 | tr '\n' ' ')

    for SEED in "${SEEDS[@]}"; do
        
        JOB_NAME="Seed${SEED}_${TISSUE}"
        echo "✅ [Submitting] Tissue: $TISSUE | Seed: $SEED"

        # 使用 HereDoc 提交 SLURM 任务
        sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${LOG_DIR}/%x_%j.out
#SBATCH --error=${LOG_DIR}/%x_%j.err
#SBATCH --partition=zhanglab.p
#SBATCH --time=10:00:00               # 🔥 时间放宽到 10 小时，防止跑到 90 多个的时候被杀
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1

export HDF5_USE_FILE_LOCKING='FALSE'
eval "\$(conda shell.bash hook)"
conda activate ${CONDA_ENV}

echo "Starting Seed $SEED for $TISSUE on node \${SLURMD_NODENAME}"

# 把 bash 里的字符串变成数组
GENES=(${GENE_LIST})

for GENE in "\${GENES[@]}"; do
    echo "▶️ Running Gene: \$GENE"
    python -u ${PYTHON_SCRIPT} \
        --gene "\$GENE" \
        --seed ${SEED} \
        --tissue ${TISSUE}
done

echo "🎉 Job ${JOB_NAME} Finished!"
EOT

    done
done

echo "========================================================"
echo "🎉 6 Jobs submitted! Check with: squeue -u dongbos"
echo "========================================================"
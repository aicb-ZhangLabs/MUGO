#!/bin/bash

# ================= ⚙️ 配置区域 =================
# Python 脚本绝对路径
PYTHON_SCRIPT="/home/dongbos/Combine_optim_Borzoi_SNP/rebuttal/code_rebuttal/sensitivity_analysis/sensitivity_MUGO_tau.py"

# CSV 名单所在的根目录
CSV_DIR="/home/dongbos/Combine_optim_Borzoi_SNP/src/interpretability/newversion_table2/top100_highexp_gene"

# 日志目录
LOG_DIR="/home/dongbos/Combine_optim_Borzoi_SNP/rebuttal/code_rebuttal/sensitivity_analysis/logs"
mkdir -p "$LOG_DIR"

# 实验参数
TAUS=(1.0 5.0 10.0 20.0 50.0 100.0 500.0)  # 探索更高的温度
TISSUES=("blood") # 先只跑1个tissue
TOP_N=100  # 拉满 100 个！
CONDA_ENV="meta-borzoi"

echo "========================================================"
echo "🚀 Submitting Tau Parameter Sensitivity Ablation (Top $TOP_N Genes)"
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

    for TAU in "${TAUS[@]}"; do
        
        # 将点号替换为下划线，防止日志文件名解析出问题 (1.0 -> 1_0)
        SAFE_TAU=$(echo $TAU | tr '.' '_')
        JOB_NAME="Tau${SAFE_TAU}_${TISSUE}"
        
        echo "✅ [Submitting] Tissue: $TISSUE | Tau: $TAU"

        # 使用 HereDoc 提交 SLURM 任务
        sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${LOG_DIR}/%x_%j.out
#SBATCH --error=${LOG_DIR}/%x_%j.err
#SBATCH --partition=zhanglab.p
#SBATCH --time=10:00:00               # 🔥 时间放宽到 10 小时
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1

export HDF5_USE_FILE_LOCKING='FALSE'

# 🔥 核心修复：直接 source Conda 底层脚本，彻底解决 conda init 报错
source /home/dongbos/miniconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

echo "Starting Tau $TAU for $TISSUE on node \${SLURMD_NODENAME}"

# 把 bash 里的字符串变成数组
GENES=(${GENE_LIST})

for GENE in "\${GENES[@]}"; do
    echo "▶️ Running Gene: \$GENE"
    python -u ${PYTHON_SCRIPT} \
        --gene "\$GENE" \
        --tau ${TAU} \
        --tissue ${TISSUE}
done

echo "🎉 Job ${JOB_NAME} Finished!"
EOT

    done
done

echo "========================================================"
echo "🎉 Jobs submitted! Check with: squeue -u dongbos"
echo "========================================================"
#!/bin/bash
#SBATCH --job-name=CADD_Bench
#SBATCH --array=31-3000%50                    # <--- CADD很快，可以把并发调大，比如同时跑50个
#SBATCH --output=/home/dongbos/Combine_optim_Borzoi_SNP/jobs/logs/%x_%A_%a.out
#SBATCH --error=/home/dongbos/Combine_optim_Borzoi_SNP/jobs/logs/%x_%A_%a.err
#SBATCH --partition=zhanglab.p               # <--- 使用通用分区
#SBATCH --time=01:00:00                      # 查表非常快，1小时绰绰有余
#SBATCH --cpus-per-task=1                    # CADD查表是单线程IO密集型，1个核足够
#SBATCH --mem=4G                             # 只要加载索引，4G内存足够
#SBATCH --nodes=1
#SBATCH --exclude=laniakea                     # (可选) 建议避开GPU节点，把GPU留给Borzoi任务

# 如果文件只在 galaxy 节点，请取消下面这行的注释，并注释掉上面的 exclude
# #SBATCH --nodelist=galaxy

export HDF5_USE_FILE_LOCKING='FALSE'

echo "Activating Conda environment..."
eval "$(conda shell.bash hook)"
conda activate meta-borzoi

GENE_INDEX=$SLURM_ARRAY_TASK_ID

# 你的脚本路径
SCRIPT_PATH="/home/dongbos/Combine_optim_Borzoi_SNP/src/baseline_benchmark/CADD_benchmark.py"

# 定义要跑的 6 个 Tissue
# 注意：CADD本身的分数是通用的，但你的脚本可能需要--tissue参数来决定结果存哪个文件夹
# TISSUES=("brain" "liver" "heart" "muscle" "Pancreas")
TISSUES=("blood") # 先只在blood一个tissue上跑

echo "=========================================================="
echo "🚀 Starting CADD Benchmark for Gene Index $GENE_INDEX"
echo "   Node: $SLURMD_NODENAME"
echo "=========================================================="

# --- 循环运行所有 Tissue ---
for TISSUE in "${TISSUES[@]}"; do
    echo "----------------------------------------------------------"
    echo "   -> [$(date +%H:%M:%S)] Querying CADD for: $TISSUE ..."
    
    # CADD 很快，直接串行跑即可
    python -u $SCRIPT_PATH --index $GENE_INDEX --tissue $TISSUE
    
    if [ $? -eq 0 ]; then
        echo "   ✅ $TISSUE finished."
    else
        echo "   ❌ $TISSUE failed."
    fi
done

echo "🎉 All tissues finished for Gene Index $GENE_INDEX at: $(date)"
#!/bin/bash
#SBATCH --job-name=FunSeq_Bench
#SBATCH --array=101-3000%40                    # <--- 任务轻量级，可以设置高并发 (如 %50)
#SBATCH --output=/home/dongbos/Combine_optim_Borzoi_SNP/jobs/logs/%x_%A_%a.out
#SBATCH --error=/home/dongbos/Combine_optim_Borzoi_SNP/jobs/logs/%x_%A_%a.err
#SBATCH --partition=zhanglab.p               # <--- 使用通用分区
#SBATCH --time=02:00:00                      # 查表通常很快，2小时足够跑完6个Tissue
#SBATCH --cpus-per-task=1                    # Tabix查询是单线程的，1个核足够
#SBATCH --mem=8G                             # 给8G内存以防万一 (虽然通常4G就够)
#SBATCH --nodes=1
#SBATCH --exclude=voyager,laniakea            # (建议) 避开宝贵的GPU节点

export HDF5_USE_FILE_LOCKING='FALSE'

echo "Activating Conda environment..."
eval "$(conda shell.bash hook)"
conda activate meta-borzoi

GENE_INDEX=$SLURM_ARRAY_TASK_ID

# 脚本路径
SCRIPT_PATH="/home/dongbos/Combine_optim_Borzoi_SNP/src/baseline_benchmark/Funseq2_benchmark.py"

# 定义要跑的 Tissue
# FunSeq2 分数本身通常是通用的，但你的脚本逻辑需要 Tissue 参数来分文件夹存储
# TISSUES=("blood" "brain" "liver" "heart" "muscle" "Pancreas")
TISSUES=("blood") # 调试用

echo "=========================================================="
echo "🚀 Starting FunSeq2 Benchmark for Gene Index $GENE_INDEX"
echo "   Node: $SLURMD_NODENAME"
echo "=========================================================="

# --- 循环运行所有 Tissue ---
for TISSUE in "${TISSUES[@]}"; do
    echo "----------------------------------------------------------"
    echo "   -> [$(date +%H:%M:%S)] Querying FunSeq2 for: $TISSUE ..."
    
    # 串行执行
    python -u $SCRIPT_PATH --index $GENE_INDEX --tissue $TISSUE
    
    if [ $? -eq 0 ]; then
        echo "   ✅ $TISSUE finished."
    else
        echo "   ❌ $TISSUE failed."
    fi
done

echo "🎉 All tissues finished for Gene Index $GENE_INDEX at: $(date)"
#!/bin/bash
#SBATCH --job-name=vcf_extract          # 任务名称
#SBATCH --output=./logs/vcf_%a.out      # 标准输出日志 (%a 代表数组ID)
#SBATCH --error=./logs/vcf_%a.err       # 错误日志
#SBATCH --array=100-3000                    # ⚠️ 关键：这里设置你要跑的 index 范围 (例如 0到99)
#SBATCH --time=03:00:00                 # 每个任务的最大运行时间
#SBATCH --mem=4G                        # 每个任务需要的内存
#SBATCH --cpus-per-task=1               # 每个任务需要的CPU核数
#SBATCH --partition=zhanglab.p                # ⚠️ 修改为你服务器的队列名称 (如 gpu, compute, defq)

# 1. 激活环境 (根据你的 conda 环境名修改)
source /home/dongbos/miniconda3/etc/profile.d/conda.sh
conda activate meta-borzoi

# 2. 定义脚本路径
PROJECT_DIR="/home/dongbos/Combine_optim_Borzoi_SNP"
SCRIPT_PATH="$PROJECT_DIR/src/process_data/extract_vcf_liftoverto_hg38.py"

# 3. 打印当前运行信息 (Debug用)
echo "Running Array Job ID: $SLURM_ARRAY_TASK_ID"
echo "On Host: $(hostname)"

# 4. 运行 Python 脚本
# $SLURM_ARRAY_TASK_ID 会自动变成 --array 设置的数字 (例如 0, 1, 2...)
python $SCRIPT_PATH --gene_index $SLURM_ARRAY_TASK_ID
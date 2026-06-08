#!/bin/bash
#SBATCH --job-name=CrossVal_B_E          # <--- 默认任务名 (提交时会被覆盖)
#SBATCH --output=/home/dongbos/Combine_optim_Borzoi_SNP/results/compare_enformer_borzoi/%x_%j.out # <--- Log 存到 compare 文件夹
#SBATCH --error=/home/dongbos/Combine_optim_Borzoi_SNP/results/compare_enformer_borzoi/%x_%j.err
#SBATCH --partition=zhanglab.p
#SBATCH --nodelist=voyager              # <--- 指定节点
#SBATCH --time=12:00:00                  # <--- 6小时足够跑完推断
#SBATCH --gres=gpu:1                     # <--- 需要 GPU
#SBATCH --cpus-per-task=4                # <--- 4核处理数据够了
#SBATCH --mem=48G                        # <--- 显存和内存需求
#SBATCH --nodes=1

# --- 获取参数 ---
# 通过命令行传入 Tissue 名称: sbatch script.sh blood  # liver heart muscle pancreas brain 
TISSUE=$1

# 检查是否传入了参数
if [ -z "$TISSUE" ]; then
  echo "❌ Error: No tissue specified."
  echo "Usage: sbatch run_cross_val.sh <tissue>"
  exit 1
fi

echo "=========================================================="
echo "🚀 Job Start: Cross-Validation for Tissue: [ $TISSUE ]"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "=========================================================="

# --- 加载环境 ---
eval "$(conda shell.bash hook)"
conda activate meta-borzoi

# --- 路径配置 ---
# 🔥 请确保这里指向你刚才保存的 compute_cross_val_data.py 的路径
SCRIPT_PATH="/home/dongbos/Combine_optim_Borzoi_SNP/src/interpretability/fig3_table_tissuespecific_multimodal/compute_cross_val_data.py"
BASE_DIR="/home/dongbos/Combine_optim_Borzoi_SNP"

# --- 创建 Log 文件夹 (防止报错) ---
mkdir -p /home/dongbos/Combine_optim_Borzoi_SNP/results/compare_enformer_borzoi

# --- 运行代码 ---
# 使用 -u 实时输出日志
python -u $SCRIPT_PATH --tissue $TISSUE --base_dir $BASE_DIR

# --- 结束检查 ---
if [ $? -eq 0 ]; then
    echo "=========================================================="
    echo "✅ [ $TISSUE ] Computation Finished Successfully at: $(date)"
else
    echo "=========================================================="
    echo "❌ [ $TISSUE ] Computation Failed at: $(date)"
    exit 1
fi
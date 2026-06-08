#!/bin/bash
#SBATCH --job-name=Fig3_Calc_Matrix          # <--- 任务名
#SBATCH --output=/home/dongbos/Combine_optim_Borzoi_SNP/results/Fig3_multi_modal/%x_%j.out # <--- Log 直接存到图的文件夹
#SBATCH --error=/home/dongbos/Combine_optim_Borzoi_SNP/results/Fig3_multi_modal/%x_%j.err
#SBATCH --partition=zhanglab.p
#SBATCH --nodelist=galaxy                  # <--- 指定节点 (或者 voyager/galaxy)
#SBATCH --time=16:00:00                      # <--- 4小时足够了 (预计30分钟)
#SBATCH --gres=gpu:1                         # <--- 需要 1 张卡跑 Borzoi
#SBATCH --cpus-per-task=8                    # <--- 给 8 个核，加快 18,000 个文件的扫描速度
#SBATCH --mem=64G                            # <--- 给足内存，防止 Borzoi OOM
#SBATCH --nodes=1

# --- 错误处理 ---
# set -e

export HDF5_USE_FILE_LOCKING='FALSE'

# --- 加载环境 ---
echo "=========================================================="
echo "🚀 Activating Conda environment..."
eval "$(conda shell.bash hook)"
conda activate meta-borzoi

# --- 脚本路径 ---
# 你的计算脚本路径
SCRIPT_PATH="/home/dongbos/Combine_optim_Borzoi_SNP/src/interpretability/fig3_table_tissuespecific_multimodal/cal_fig3_datamatrix_filterout_low_exp_gene.py"

# --- 调试信息 ---
echo "Job ID: $SLURM_JOB_ID"
echo "Running on Node: $SLURMD_NODENAME"
echo "Script Path: $SCRIPT_PATH"
echo "Log Path: /home/dongbos/Combine_optim_Borzoi_SNP/results/Fig3_multi_modal/"
echo "=========================================================="

# --- 运行代码 ---
# -u 参数让 python 实时输出 log，不要缓存，这样你可以用 tail -f 实时看进度
python -u $SCRIPT_PATH --mode CAGE  # RNA, ATAC, CAGE

# --- 结束检查 ---
if [ $? -eq 0 ]; then
    echo "=========================================================="
    echo "🎉 Calculation Finished Successfully at: $(date)"
    echo "Check results in: /home/dongbos/Combine_optim_Borzoi_SNP/results/Fig3_multi_modal/"
else
    echo "=========================================================="
    echo "❌ Calculation Failed at: $(date)"
fi
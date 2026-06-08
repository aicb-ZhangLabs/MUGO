import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from scipy.stats import pearsonr, spearmanr

# ================= ⚙️ 配置 =================
BASE_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP'
DATA_DIR = f'{BASE_DIR}/results/multimodal_benchmark'
FIG_DIR = f'{BASE_DIR}/results/figures/scatter_plots' # 图片保存路径

os.makedirs(FIG_DIR, exist_ok=True)

# 列名映射 (保持和你之前的脚本一致)
COL_MUGO = 'Borzoi_Gain'
COL_SAL = 'Saliency_Gain'

def load_data(tissue, modality='ATAC'):
    """读取 Benchmark CSV"""
    # 简单的文件名映射
    file_tag = modality
    if modality == 'RNA-seq': file_tag = 'RNA'
    
    filename = f'benchmark_{file_tag}_{tissue}.csv'
    csv_path = os.path.join(DATA_DIR, filename)
    
    if not os.path.exists(csv_path):
        print(f"❌ Error: File not found at {csv_path}")
        return None
        
    df = pd.read_csv(csv_path)
    
    # 筛选所需列，取绝对值，并去除空值 (只有两者都有值的点才能画散点)
    if COL_MUGO not in df.columns or COL_SAL not in df.columns:
        print(f"❌ Error: Columns {COL_MUGO} or {COL_SAL} missing in CSV.")
        return None
        
    df_clean = df[[COL_MUGO, COL_SAL]].copy()
    
    # 🔥 关键：取绝对值 (Gain 通常看 Magnitude)
    df_clean['MUGO_Abs'] = df_clean[COL_MUGO].abs()
    df_clean['Sal_Abs'] = df_clean[COL_SAL].abs()
    
    # 去除 NaN
    df_clean = df_clean.dropna()
    
    print(f"✅ Loaded {len(df_clean)} valid genes for {tissue} ({modality}).")
    return df_clean

def plot_scatter(df, tissue, modality):
    """绘制 Scatter Plot"""
    x = df['Sal_Abs']
    y = df['MUGO_Abs']
    
    # 1. 计算统计量
    # 胜率: MUGO > Saliency 的比例
    win_count = np.sum(y > x)
    total = len(y)
    win_rate = (win_count / total) * 100 if total > 0 else 0
    
    # 相关性
    corr, _ = pearsonr(x, y)
    
    # 2. 设置画布
    plt.figure(figsize=(7, 7))
    sns.set_style("whitegrid")
    
    # 3. 绘制散点
    # 根据是否胜出上色
    colors = ['#d62728' if yi > xi else '#1f77b4' for xi, yi in zip(x, y)]
    # 红色 = MUGO赢, 蓝色 = Saliency赢
    
    plt.scatter(x, y, c=colors, alpha=0.6, s=15, edgecolor='w', linewidth=0.3)
    
    # 4. 绘制对角线
    max_val = max(x.max(), y.max()) * 1.05
    plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.7, label='y=x (Equal Gain)')
    
    # 5. 装饰图表
    plt.title(f"{modality} - {tissue.capitalize()}\nMUGO vs. Saliency Attribution", fontsize=14)
    plt.xlabel("Saliency Gain (Absolute)", fontsize=12)
    plt.ylabel("MUGO Gain (Absolute)", fontsize=12)
    
    plt.xlim(0, max_val)
    plt.ylim(0, max_val)
    
    # 6. 添加统计信息文本
    stats_text = (
        f"N = {total}\n"
        f"Pearson r = {corr:.2f}\n"
        f"MUGO Higher: {win_rate:.1f}%"
    )
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
             fontsize=11, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    # 7. 保存
    out_file = os.path.join(FIG_DIR, f'Scatter_{modality}_{tissue}.png')
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    print(f"🖼️  Plot saved to: {out_file}")
    # plt.show() # 服务器上通常不开这个

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tissue', type=str, default='blood', help='Tissue name (e.g., blood, liver)')
    parser.add_argument('--mode', type=str, default='ATAC', choices=['ATAC', 'RNA-seq', 'CAGE', 'DNAse'], help='Modality')
    args = parser.parse_args()

    print(f"🚀 Analyzing {args.mode} for {args.tissue}...")
    
    df = load_data(args.tissue, args.mode)
    
    if df is not None and not df.empty:
        plot_scatter(df, args.tissue, args.mode)
    else:
        print("⚠️ No data to plot.")

if __name__ == "__main__":
    main()
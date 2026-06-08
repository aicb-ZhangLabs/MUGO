'''
plot script for find_gene_additive_redundant_synagey_topN_SNP.py 
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

# ================= ⚙️ 配置 =================
BASE_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP'
INPUT_DIR = f'{BASE_DIR}/results/interaction_scan_multi'
OUTPUT_DIR = f'{BASE_DIR}/results/interaction_scan_multi/plots'

# 阈值设定 (和扫描脚本保持一致)
THRESHOLD = 0.10 
LOWER_BOUND = 1.0 - THRESHOLD # 0.9
UPPER_BOUND = 1.0 + THRESHOLD # 1.1

# 绘图颜色
COLOR_SYN = '#2ecc71' # Green
COLOR_ADD = '#95a5a6' # Grey
COLOR_RED = '#e74c3c' # Red
COLOR_KDE = '#34495e' # Dark Blue line

# 图片保存格式
SAVE_FORMAT = 'svg'  # 关键：设置为 svg

def main():
    parser = argparse.ArgumentParser(description="Plot Interaction Ratio Distribution")
    parser.add_argument('--tissue', type=str, default='blood', help="Target tissue (e.g. blood)")
    parser.add_argument('--n', type=int, default=5, help="Top N used (default 5)")
    args = parser.parse_args()
    
    tissue = args.tissue.lower()
    csv_filename = f"{tissue}_top{args.n}_interactions.csv"
    csv_path = os.path.join(INPUT_DIR, csv_filename)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"📖 Loading data from: {csv_path}")
    if not os.path.exists(csv_path):
        print(f"❌ File not found! Please run scan_multi_interaction.py first.")
        return

    df = pd.read_csv(csv_path)
    
    # 过滤掉 Ratio 为空或无穷大的异常值
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['Ratio'])
    
    # 统计数量
    total = len(df)
    n_syn = len(df[df['Ratio'] > UPPER_BOUND])
    n_red = len(df[df['Ratio'] < LOWER_BOUND])
    n_add = total - n_syn - n_red
    
    print(f"📊 Statistics for {tissue.capitalize()}:")
    print(f"   Total Genes: {total}")
    print(f"   Synergistic: {n_syn} ({n_syn/total:.1%})")
    print(f"   Additive:    {n_add} ({n_add/total:.1%})")
    print(f"   Redundant:   {n_red} ({n_red/total:.1%})")

    # --- 开始绘图 ---
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # 为了画图好看，我们限制 X 轴范围，聚焦在主峰附近
    # 极端值会被截断在图外，但统计数字是准确的
    plot_data = df['Ratio']
    x_min, x_max = 0.5, 1.5
    
    # 1. 绘制背景分区颜色
    plt.axvspan(x_min, LOWER_BOUND, color=COLOR_RED, alpha=0.1, label='Redundant (Saturation)')
    plt.axvspan(LOWER_BOUND, UPPER_BOUND, color=COLOR_ADD, alpha=0.1, label='Additive (Linear)')
    plt.axvspan(UPPER_BOUND, x_max, color=COLOR_SYN, alpha=0.1, label='Synergistic (Cooperative)')
    
    # 2. 绘制直方图 (Histogram)
    # 修改点：bins=150, binrange=(x_min, x_max) 强制只在可视范围内切分格子，让分布更细腻
    sns.histplot(plot_data, 
                 bins=150, 
                 binrange=(x_min, x_max),
                 kde=True, 
                 stat="count", 
                 color=COLOR_KDE, 
                 edgecolor=None, 
                 alpha=0.6,
                 line_kws={'linewidth': 2})
    
    # 3. 装饰
    plt.xlim(x_min, x_max)
    plt.axvline(1.0, color='black', linestyle='--', linewidth=1) # 基准线
    
    plt.title(f"Interaction Spectrum: {tissue.capitalize()} (Top {args.n} SNPs)", fontsize=16, pad=20, fontweight='bold')
    plt.xlabel("Interaction Ratio (Actual Gain / Sum of Single Gains)", fontsize=12)
    plt.ylabel("Number of Genes", fontsize=12)
    
    # 4. 添加统计文本框
    stats_text = (
        f"$\mathbf{{Synergistic (>1.1)}}$: {n_syn} ({n_syn/total:.1%})\n"
        f"$\mathbf{{Additive (0.9-1.1)}}$: {n_add} ({n_add/total:.1%})\n"
        f"$\mathbf{{Redundant (<0.9)}}$: {n_red} ({n_red/total:.1%})"
    )
    plt.gca().text(0.02, 0.95, stats_text, transform=plt.gca().transAxes, 
                   fontsize=11, verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    # 5. 保存
    # 修改点：使用 .svg 后缀，并指定 format='svg' 
    out_file = f"{OUTPUT_DIR}/{tissue}_interaction_spectrum.{SAVE_FORMAT}"
    plt.tight_layout()
    plt.savefig(out_file, format=SAVE_FORMAT, bbox_inches='tight')
    print(f"\n✅ Plot saved to: {out_file}")
    plt.close()

if __name__ == "__main__":
    main()
'''
Plot gene-level combinatorial effect visualization.
python plot_case_study.py --gene CAMKK1 --tissue blood
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import ast

# ================= ⚙️ 配置 =================
BASE_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP'
INPUT_DIR = f'{BASE_DIR}/results/interaction_scan_multi'
OUTPUT_DIR = f'{BASE_DIR}/results/interaction_scan_multi/example_visualizations'

# 颜色配置
COLOR_PALETTE = sns.color_palette("Blues", n_colors=10) # 单个SNP用蓝色系
COLOR_COMBO_SYN = '#2ecc71' # 协同用绿色
COLOR_COMBO_RED = '#e74c3c' # 冗余用红色
COLOR_COMBO_ADD = '#34495e' # 加性用深蓝

def main():
    parser = argparse.ArgumentParser(description="Plot gene-level combinatorial effect visualization")
    parser.add_argument('--gene', type=str, required=True, help="Gene Name (e.g. SPX)")
    parser.add_argument('--tissue', type=str, default='blood', help="Tissue (e.g. blood)")
    parser.add_argument('--n', type=int, default=5, help="Top N used (default 5)")
    args = parser.parse_args()
    
    # 1. 读取数据
    tissue = args.tissue.lower()
    csv_path = os.path.join(INPUT_DIR, f"{tissue}_top{args.n}_interactions.csv")
    
    if not os.path.exists(csv_path):
        print(f"❌ Input CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    # 查找基因
    row = df[df['Gene'] == args.gene]
    if row.empty:
        print(f"❌ Gene '{args.gene}' not found in {tissue} results.")
        return
    
    row = row.iloc[0]
    
    # 解析数据
    # Single_Gains_Detail 是字符串 "[1.2, 0.5...]"，需要转回 list
    single_gains = ast.literal_eval(row['Single_Gains_Detail'])
    combo_gain = float(row['Combo_Gain'])
    sum_gain = float(row['Single_Gains_Sum'])
    ratio = float(row['Ratio'])
    category = row['Category']
    
    print(f"🧬 Gene: {args.gene}")
    print(f"   Sum of Singles: {sum_gain:.4f}")
    print(f"   Actual Combo:   {combo_gain:.4f}")
    print(f"   Ratio:          {ratio:.4f} ({category})")
    
    # --- 绘图 ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.figure(figsize=(7, 8))
    sns.set_style("whitegrid")
    
    # X轴位置
    x_pos = [0, 1]
    bar_width = 0.5
    
    # 1. 绘制左侧 Stacked Bar (Expected Sum)
    bottom = 0
    # 对 gain 排序，把大的放下面，视觉上更稳
    sorted_singles = sorted(single_gains, reverse=True)
    
    for i, g in enumerate(sorted_singles):
        plt.bar(x_pos[0], g, bottom=bottom, width=bar_width, 
                color=COLOR_PALETTE[i % len(COLOR_PALETTE)], 
                edgecolor='white', label=f'SNP {i+1}' if i < 3 else "")
        # 在色块中间标数值
        if abs(g) > sum_gain * 0.05: # 太小的就不标了
            plt.text(x_pos[0], bottom + g/2, f"{g:.1f}", ha='center', va='center', 
                     color='white', fontsize=9, fontweight='bold')
        bottom += g
        
    # 2. 绘制右侧 Solid Bar (Actual Combo)
    if category == 'Synergistic':
        combo_color = COLOR_COMBO_SYN
    elif category == 'Redundant':
        combo_color = COLOR_COMBO_RED
    else:
        combo_color = COLOR_COMBO_ADD
        
    plt.bar(x_pos[1], combo_gain, width=bar_width, color=combo_color, 
            edgecolor='black', linewidth=1.5)
    
    plt.text(x_pos[1], combo_gain + (combo_gain*0.02), f"{combo_gain:.2f}", 
             ha='center', va='bottom', fontsize=12, fontweight='bold', color=combo_color)

    # 3. 绘制差异箭头 (Synergy Gap)
    # 在两个柱子顶部之间画箭头
    max_height = max(sum_gain, combo_gain)
    gap = combo_gain - sum_gain
    
    if abs(gap) > 0.5: # 只有差异明显才画
        arrow_x = 1.0
        arrow_y_start = sum_gain
        arrow_y_end = combo_gain
        
        props = dict(arrowstyle='-|>', color='black', lw=1.5)
        
        # 如果是 Synergistic，箭头向上
        if gap > 0:
            plt.annotate(f"+{gap:.2f}\n(Synergy)", 
                         xy=(0.5, (sum_gain + combo_gain)/2), 
                         ha='center', va='center', fontsize=11, fontweight='bold',
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=combo_color, alpha=0.9))
            
            # 连线
            plt.plot([0.25, 0.75], [sum_gain, sum_gain], 'k--', alpha=0.5) # 左顶虚线
            plt.plot([0.75, 0.75], [sum_gain, combo_gain], 'k-', alpha=0.8) # 垂直线
            
        # 如果是 Redundant，箭头向下
        else:
            plt.annotate(f"{gap:.2f}\n(Saturation)", 
                         xy=(0.5, (sum_gain + combo_gain)/2), 
                         ha='center', va='center', fontsize=11, fontweight='bold',
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=combo_color, alpha=0.9))

    # 4. 装饰
    plt.xticks(x_pos, ['Naive Linear Sum\n(Expected)', 'MVP Model Prediction\n(Actual)'], fontsize=11, fontweight='bold')
    plt.ylabel('Predicted Expression Gain', fontsize=12)
    plt.title(f"Combinatorial Logic: {args.gene}", fontsize=14, pad=20, fontweight='bold')
    
    # 动态调整 Y 轴上限，留出标题空间
    plt.ylim(0, max_height * 1.2)
    
    # 5. 保存
    out_file = f"{OUTPUT_DIR}/{tissue}_{args.gene}_interaction_visualization.png"
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    print(f"✅ Plot saved to: {out_file}")

if __name__ == "__main__":
    main()

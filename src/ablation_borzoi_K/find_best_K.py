import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ================= ⚙️ 配置区域 =================
BASE_RESULT_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP/results/ablation_borzoi_K'
INPUT_CSV = f'{BASE_RESULT_DIR}/final_ablation_table.csv'
OUTPUT_DIR = f'{BASE_RESULT_DIR}/plots'

# 我们选定的最佳 K
BEST_K = 10

# 图片保存格式
SAVE_FORMAT = 'svg'  # 关键：设置为 svg

# ===============================================

def load_and_clean_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ Input file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    
    # 🔍 调试：打印一下原始列名，看看到底叫什么
    print(f"📋 Raw Columns in CSV: {df.columns.tolist()}")
    
    # 1. 清理列名 (增强版：兼容 Mean 和 Max，兼容带不带括号)
    # 注意：replace 是 substring 替换，所以顺序很重要
    new_columns = []
    for c in df.columns:
        c = c.replace('Ensemble Heads ($K$)', 'K')
        c = c.replace('Ensemble Heads', 'K') # 备用
        
        # 兼容 Max 和 Mean
        if 'Gain' in c:
            c = 'Gain' 
        
        c = c.replace('Robustness (Gini)', 'Gini')
        c = c.replace('GWAS Enrichment (OR)', 'GWAS_OR')
        
        # 兼容 GTEx 各种写法
        if 'GTEx' in c and 'Overlap' in c:
            c = 'GTEx_Pct'
        if 'GTEx' in c and 'Recall' in c:
            c = 'GTEx_Pct'
            
        new_columns.append(c)
        
    df.columns = new_columns
    
    # 🔍 调试：打印清理后的列名
    print(f"✅ Cleaned Columns: {df.columns.tolist()}")
    
    # 2. 清理 GTEx 百分号 (如果有)
    if 'GTEx_Pct' in df.columns and df['GTEx_Pct'].dtype == object:
        df['GTEx_Pct'] = df['GTEx_Pct'].astype(str).str.replace(r'\\%', '', regex=True).str.replace('%', '').astype(float)
        
    return df

def plot_dual_axis_tradeoff(df, out_dir):
    """
    画最核心的 Trade-off 图：Gain (上升) vs GWAS (下降)
    """
    sns.set_style("white")
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # X轴处理：为了让点均匀分布，我们使用 Range 作为 X 轴，然后替换 Label
    # 这样避免 K=50 把前面 K=1,3,5 挤在一起
    x_indices = np.arange(len(df))
    k_labels = df['K'].astype(str).tolist()

    # --- 左轴: Optimization Power (Gain) ---
    color1 = '#e74c3c' # 红色
    ax1.plot(x_indices, df['Gain'], color=color1, marker='o', linewidth=3, label='Expression Gain')
    ax1.set_xlabel('Ensemble Size (K)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Max Expression Gain (Fold)', fontsize=12, fontweight='bold', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(k_labels, fontsize=11)
    ax1.grid(True, axis='x', linestyle='--', alpha=0.5)

    # --- 右轴: Biological Precision (GWAS OR) ---
    ax2 = ax1.twinx()
    color2 = '#3498db' # 蓝色
    ax2.plot(x_indices, df['GWAS_OR'], color=color2, marker='s', linewidth=3, linestyle='--', label='GWAS Enrichment')
    ax2.set_ylabel('GWAS Enrichment (Odds Ratio)', fontsize=12, fontweight='bold', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    # --- 高亮 Best K ---
    best_idx = df[df['K'] == BEST_K].index[0]
    
    # 画一条垂直竖线表示选择
    plt.axvline(x=best_idx, color='grey', linestyle=':', alpha=0.8, linewidth=1.5, zorder=0)
    
    # 添加标注文本
    # 获取对应坐标
    gain_val = df.loc[best_idx, 'Gain']
    
    ax1.annotate(f'Optimal K={BEST_K}\n(Sweet Spot)', 
                 xy=(best_idx, gain_val), xytext=(best_idx, gain_val + 5),
                 ha='center', fontsize=10, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='black'))

    plt.title('Optimization Potency vs. Biological Fidelity Trade-off', fontsize=14, pad=20)
    plt.tight_layout()
    
    # 修改：保存为 SVG 
    save_path = f"{out_dir}/k_selection_tradeoff_dual.{SAVE_FORMAT}"
    plt.savefig(save_path, format=SAVE_FORMAT, dpi=300)
    print(f"   📊 Saved Dual-Axis Plot: {save_path}")
    plt.close()

def plot_dashboard(df, out_dir):
    """
    画 2x2 的全景图，展示所有指标
    """
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    x_indices = np.arange(len(df))
    k_labels = df['K'].astype(str).tolist()
    
    metrics = [
        ('Gain', 'Max Expression Gain', '#e74c3c'), # Red
        ('Gini', 'Robustness (Gini Coeff.)', '#9b59b6'), # Purple
        ('GWAS_OR', 'GWAS Enrichment (OR)', '#3498db'), # Blue
        ('GTEx_Pct', 'GTEx Overlap (%)', '#2ecc71') # Green
    ]
    
    best_idx = df[df['K'] == BEST_K].index[0]

    for ax, (col, title, color) in zip(axes.flatten(), metrics):
        # 画线
        ax.plot(x_indices, df[col], marker='o', linewidth=2.5, color=color)
        
        # 标记 Best K
        ax.plot(best_idx, df.loc[best_idx, col], marker='o', markersize=12, 
                markerfacecolor='white', markeredgecolor='black', markeredgewidth=2)
        ax.axvline(x=best_idx, color='grey', linestyle='--', alpha=0.5)
        
        # 标数值
        for i, val in enumerate(df[col]):
            offset = val * 0.05
            ax.text(i, val + offset if i % 2 == 0 else val - offset, 
                    f"{val:.2f}", ha='center', fontsize=9, color=color, fontweight='bold')

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(x_indices)
        ax.set_xticklabels(k_labels)
        ax.set_xlabel('Ensemble Size (K)')
    
    plt.suptitle(f"Ablation Metrics Overview (Highlighting K={BEST_K})", fontsize=16, y=0.95)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 修改：保存为 SVG 
    save_path = f"{out_dir}/k_selection_dashboard.{SAVE_FORMAT}"
    plt.savefig(save_path, format=SAVE_FORMAT, dpi=300)
    print(f"   📊 Saved Dashboard Plot: {save_path}")
    plt.close()

def main():
    print("🚀 Generating K Selection Plots (SVG)...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. 加载数据
    try:
        df = load_and_clean_data(INPUT_CSV)
        print("✅ Data Loaded.")
    except Exception as e:
        print(e)
        return

    # 2. 画图
    plot_dual_axis_tradeoff(df, OUTPUT_DIR)
    plot_dashboard(df, OUTPUT_DIR)
    
    print("\n✅ All plots generated!")

if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from adjustText import adjust_text  # 如果没有安装这个库，可以注释掉相关的 label 代码，或者 pip install adjustText

# ==========================================
# 1. 配置路径
# ==========================================
BASE_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP'
RESULT_DIR = f'{BASE_DIR}/results/res_enrichment_conservative_borzoi'
CSV_PATH = f'{RESULT_DIR}/gene_enrichment_stats_K10.csv'

# 确保输出目录存在
os.makedirs(RESULT_DIR, exist_ok=True)

# 设置绘图风格
sns.set(style="whitegrid", font_scale=1.2)
plt.rcParams['font.family'] = 'sans-serif'

def load_data(csv_path):
    if not os.path.exists(csv_path):
        print(f"❌ File not found: {csv_path}")
        return None
    return pd.read_csv(csv_path)

# ==========================================
# 2. 图1: 全局富集趋势 (Grouped Bar Plot)
# ==========================================
def plot_global_trend(df):
    print("📊 Plotting Global Enrichment Trend...")
    
    # 计算全局平均值 (排除 NaN/Inf)
    tiers = ["Top10pct", "Top5pct", "Top1pct"]
    labels = ["Top 10%", "Top 5%", "Top 1%"]
    
    avg_base = []
    avg_model = []
    enrichment_folds = []
    
    for tier in tiers:
        # 计算平均 Rate
        mean_base = df[f"{tier}_Base_Rate"].mean()
        mean_model = df[f"{tier}_Model_Rate"].mean()
        
        avg_base.append(mean_base * 100)  # 转为百分比
        avg_model.append(mean_model * 100)
        
        # 计算总体 Fold (Model Mean / Base Mean) 这样比取 Fold 的平均更稳健
        fold = mean_model / mean_base if mean_base > 0 else 0
        enrichment_folds.append(fold)

    # 准备绘图数据
    x = np.arange(len(tiers))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    rects1 = ax.bar(x - width/2, avg_base, width, label='Random Baseline', color='#bdc3c7', alpha=0.8)
    rects2 = ax.bar(x + width/2, avg_model, width, label='Borzoi Top-K', color='#e74c3c', alpha=0.9)
    
    # 设置标签
    ax.set_ylabel('Percentage of High-Conservation Variants (%)')
    ax.set_title('Global Enrichment of Evolutionary Conservation\n(Borzoi Model vs. Random Baseline)', fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    # 在柱子上标注 Enrichment Fold
    def autolabel(rects, folds):
        for i, rect in enumerate(rects):
            height = rect.get_height()
            fold = folds[i]
            if height > 0:
                ax.annotate(f'{fold:.1f}x Fold',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 5),  # 垂直偏移
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=12, fontweight='bold', color='#c0392b')

    autolabel(rects2, enrichment_folds)
    
    # 调整Y轴上限，留点空间给文字
    ax.set_ylim(0, max(avg_model) * 1.2)
    
    plt.tight_layout()
    save_path = f"{RESULT_DIR}/global_enrichment_trend.png"
    plt.savefig(save_path, dpi=300)
    print(f"✅ Saved: {save_path}")
    plt.close()

# ==========================================
# 3. 图2: 基因特异性散点图 (Scatter Plot)
# ==========================================
def plot_gene_scatter(df):
    print("📊 Plotting Gene-Specific Scatter Plot...")
    
    # 过滤掉无效数据
    # 关注 Top 1% 的表现
    target_tier = "Top1pct"
    
    # 为了避免 log(0)，加一个小常数
    df_clean = df.copy()
    epsilon = 1e-4
    
    # X轴: Baseline Rate (越左边越难)
    # Y轴: Model Rate (越上面越准)
    # 颜色/大小: Enrichment Fold
    
    # 也可以画 Volcano: X=Log2(Fold), Y=Model Rate
    # 这里我们画 Enrichment vs Baseline，更能体现“大海捞针”的能力
    
    x_val = df_clean[f"{target_tier}_Base_Rate"] * 100
    y_val = df_clean[f"{target_tier}_Model_Rate"] * 100
    
    # 只要 Top 1% Model Rate > 0 的点 (只有命中的点才有展示意义)
    plot_df = df_clean[df_clean[f"{target_tier}_Model_Rate"] > 0].copy()
    
    if plot_df.empty:
        print("⚠️ No genes with Top 1% hits found. Skipping scatter plot.")
        return

    # 计算 Enrichment 用于颜色映射
    plot_df['Enrichment_Log'] = plot_df[f"{target_tier}_Enrichment"]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制散点
    scatter = ax.scatter(
        plot_df[f"{target_tier}_Base_Rate"] * 100, 
        plot_df[f"{target_tier}_Model_Rate"] * 100,
        c=plot_df[f"{target_tier}_Enrichment"], 
        cmap='viridis', 
        s=100, 
        alpha=0.8,
        edgecolors='w'
    )
    
    # 添加对角线 (y=x, 即 Random Performance)
    max_val = max(plot_df[f"{target_tier}_Model_Rate"].max() * 100, plot_df[f"{target_tier}_Base_Rate"].max() * 100)
    ax.plot([0, max_val], [0, max_val], ls='--', c='gray', alpha=0.5, label='Random Baseline (1x)')
    
    # 标注明星基因 (Top 5 Enrichment)
    top_genes = plot_df.nlargest(7, f"{target_tier}_Enrichment")
    texts = []
    for _, row in top_genes.iterrows():
        # 只有当富集倍数 > 2 时才标，避免标一堆垃圾
        if row[f"{target_tier}_Enrichment"] > 2.0:
            texts.append(ax.text(
                row[f"{target_tier}_Base_Rate"] * 100, 
                row[f"{target_tier}_Model_Rate"] * 100, 
                row['Gene'], 
                fontsize=10, fontweight='bold'
            ))
            
    # 尝试自动调整文字位置避免重叠 (需要 adjustText 库，没有就跳过)
    try:
        adjust_text(texts, arrowprops=dict(arrowstyle='-', color='k', lw=0.5))
    except:
        pass

    # 装饰
    cbar = plt.colorbar(scatter)
    cbar.set_label(f'{target_tier} Fold Enrichment')
    
    ax.set_xlabel(f'Random Baseline: % of {target_tier} variants in candidate pool')
    ax.set_ylabel(f'Model Performance: % of {target_tier} variants in Top-10')
    ax.set_title(f'Model Precision on Extreme Conservation ({target_tier})', fontsize=14)
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    save_path = f"{RESULT_DIR}/gene_scatter_top1pct.png"
    plt.savefig(save_path, dpi=300)
    print(f"✅ Saved: {save_path}")
    plt.close()

def main():
    df = load_data(CSV_PATH)
    if df is not None:
        plot_global_trend(df)
        plot_gene_scatter(df)
        print("\n🎉 All plots generated successfully!")

if __name__ == "__main__":
    main()
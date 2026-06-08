'''
总的来说一个gene多个SNPs组合可能有的是Synergistic，有的是Redundant，有的是Additive，需要画更精细的图。
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# ================= 配置 =================
BASE_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP'
# 使用清洗后的数据 (信噪比高)
INPUT_CSV = f'{BASE_DIR}/results/interaction_scan/pairwise_scan_cleaned.csv'
OUTPUT_DIR = f'{BASE_DIR}/results/interaction_scan'
os.makedirs(OUTPUT_DIR, exist_ok=True)
# =======================================

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"File not found: {INPUT_CSV}")
        return

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} pairs.")

    # ---------------------------------------------------------
    # Part 1: Global Stacked Bar Plot (宏观分布)
    # ---------------------------------------------------------
    
    # 按 Gene 分组，计算每种 Category 的比例
    gene_stats = df.groupby(['Gene', 'Category']).size().unstack(fill_value=0)
    
    # 补全可能缺失的列
    for col in ['Synergistic', 'Redundant', 'Additive']:
        if col not in gene_stats.columns:
            gene_stats[col] = 0
            
    # 计算百分比
    gene_stats['Total'] = gene_stats.sum(axis=1)
    gene_stats_pct = gene_stats.div(gene_stats['Total'], axis=0) * 100
    
    # 筛选：只展示至少有 3 个有效 Pair 的 Gene (避免数据太稀疏)
    valid_genes = gene_stats[gene_stats['Total'] >= 3].index
    plot_data = gene_stats_pct.loc[valid_genes]
    
    # 排序：先按 Synergistic 降序，再按 Redundant 升序
    plot_data = plot_data.sort_values(['Synergistic', 'Redundant'], ascending=[False, True])
    
    # 如果基因太多，按排序后的前段、中段和后段抽样展示，避免 X 轴过密
    if len(plot_data) > 60:
        subset = pd.concat([plot_data.head(25), plot_data.iloc[len(plot_data)//2-10:len(plot_data)//2+10], plot_data.tail(25)])
    else:
        subset = plot_data

    # 画图
    plt.figure(figsize=(12, 6))
    subset[['Synergistic', 'Additive', 'Redundant']].plot(
        kind='bar', stacked=True, 
        color=['#e74c3c', '#95a5a6', '#3498db'], 
        width=0.8, edgecolor='black', linewidth=0.5, ax=plt.gca()
    )
    
    plt.title('Heterogeneity of Combinatorial Logic Across Genes', fontsize=14, fontweight='bold')
    plt.ylabel('Percentage of Pairs (%)', fontsize=12)
    plt.xlabel('Genes (Sorted by Synergy)', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/gene_complexity_stacked.png", dpi=300)
    print(f"✅ Saved Stacked Plot to {OUTPUT_DIR}/gene_complexity_stacked.png")
    plt.close()

    # ---------------------------------------------------------
    # Part 2: Single Gene Heatmap (微观机制)
    # ---------------------------------------------------------
    
    # 选择同时包含 Synergistic 和 Redundant pair 的基因用于热图
    mixed_genes = gene_stats[
        (gene_stats['Synergistic'] > 0) & 
        (gene_stats['Redundant'] > 0)
    ]
    
    if not mixed_genes.empty:
        # 使用 pair 数最多的混合基因
        target_gene = mixed_genes['Total'].idxmax()
        print(f"\n🔍 Selected mixed gene for heatmap: {target_gene}")
    else:
        # 如果没有混合的，使用 Pair 数最多的基因
        target_gene = gene_stats['Total'].idxmax()
        print(f"\n🔍 No mixed genes found. Using gene with most pairs: {target_gene}")

    # 提取该基因的所有数据
    gene_df = df[df['Gene'] == target_gene].copy()
    
    # 构建 5x5 矩阵
    # Pair 格式是 "R1+R2", 我们需要解析出 1 和 2
    matrix = np.full((5, 5), np.nan) # 5x5 空矩阵
    
    for _, row in gene_df.iterrows():
        # 解析 "R1+R2" -> idx 0, 1
        r1_str, r2_str = row['Pair'].split('+')
        idx1 = int(r1_str[1:]) - 1 # 0-based
        idx2 = int(r2_str[1:]) - 1
        
        # 填入 Ratio
        matrix[idx1, idx2] = row['Ratio']
        matrix[idx2, idx1] = row['Ratio'] # 对称
        
    # 对角线填 1.0 (自己对自己是 Additive 的基准)
    np.fill_diagonal(matrix, 1.0)

    # 画热图
    plt.figure(figsize=(7, 6))
    cmap = sns.diverging_palette(240, 10, as_cmap=True, center='light') # 蓝-白-红
    
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap=cmap, center=1.0,
                xticklabels=[f'Rank {i}' for i in range(1,6)],
                yticklabels=[f'Rank {i}' for i in range(1,6)],
                square=True, linewidths=.5, cbar_kws={"label": "Interaction Ratio (Obs/Exp)"})
    
    plt.title(f'Interaction Map: {target_gene}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/heatmap_{target_gene}.png", dpi=300)
    print(f"✅ Saved Heatmap to {OUTPUT_DIR}/heatmap_{target_gene}.png")

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import argparse
from tqdm import tqdm  # ✅ [新增] 引入进度条库

# ================= ⚙️ 命令行参数配置 =================
def parse_args():
    parser = argparse.ArgumentParser(description="Generate summary plots for Borzoi optimization results.")
    parser.add_argument('--tissue', type=str, required=True, 
                        help="Target tissue name (e.g., brain, blood, liver, heart, muscle, Pancreas)")
    parser.add_argument('--k', type=int, default=10, help="Target K value (default: 10)")
    return parser.parse_args()

# =================================================

def collect_data(input_dir):
    """
    读取数据，专门提取 Best Epoch (Max Gain) 的数据
    """
    print(f"📂 [1/2] Reading from: {input_dir}")
    
    if not os.path.exists(input_dir):
        print(f"❌ Error: Input directory not found: {input_dir}")
        return pd.DataFrame()

    csv_files = glob.glob(os.path.join(input_dir, "*_optim_log.csv"))
    summary_data = []
    
    print(f"   Found {len(csv_files)} files.")

    # ✅ [修改] 使用 tqdm 包裹循环，显示进度条
    for filename in tqdm(csv_files, desc="Processing CSVs", unit="gene"):
        try:
            df = pd.read_csv(filename)
            if df.empty or 'Baseline' not in df.columns or 'Gain' not in df.columns: 
                continue

            gene_name = os.path.basename(filename).split('_optim_log')[0]
            
            # ✅ [核心修改]：找到 Gain 最大的那一行 (Best Epoch)
            best_idx = df['Gain'].idxmax()
            row_best = df.iloc[best_idx]
            
            # 使用 Best Epoch 的数据
            baseline = row_best['Baseline']
            best_gain = row_best['Gain'] # 这是历史最高 Gain
            
            # 计算百分比提升
            safe_base = max(float(baseline), 0.1)
            pct_gain = (best_gain / safe_base) * 100
            
            # ✅ [核心修改]：提取 Best Epoch 那一刻的 Rank 1-8 投票分
            # 这样反映的是达到最优解时的 SNP 权重分布
            vote_dict = {}
            for i in range(1, 9): 
                vote_dict[f'Rank{i}_Vote'] = row_best.get(f'Rank{i}_Score', 0)

            entry = {
                'Gene': gene_name,
                'Baseline': baseline,
                'Final_Gain': best_gain, # 为了兼容绘图代码，命名保持 Final_Gain，但实际存的是 Best
                'Max_Gain': best_gain,   # Best 和 Max 现在是一样的
                'Pct_Gain': pct_gain,
            }
            entry.update(vote_dict)
            summary_data.append(entry)
            
        except Exception as e:
            # 使用 tqdm.write 来打印错误，防止打断进度条显示
            tqdm.write(f"⚠️ Error processing {filename}: {e}")
            continue
            
    return pd.DataFrame(summary_data)

def plot_charts_final(df, output_dir, tissue_name):
    """
    绘制所有图表
    """
    print(f"🎨 [2/2] Generating FINAL plots in: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 全局字体设置
    sns.set(style="whitegrid", font_scale=0.9)
    
    # -------------------------------------------------------
    # 📊 图 1: MA-Style Plot 
    # -------------------------------------------------------
    plt.figure(figsize=(10, 8))
    # Hue 使用 Rank1 Vote，Size 使用 Gain
    sns.scatterplot(
        data=df, x='Baseline', y='Pct_Gain', hue='Rank1_Vote', 
        palette='viridis', size='Final_Gain', sizes=(20, 250), alpha=0.8,
        rasterized=True
    )
    plt.xscale('log')
    plt.axhline(0, color='grey', linestyle='--', alpha=0.5)
    plt.title(f'[{tissue_name}] Relative Impact: % Gain vs. Baseline (Best Epoch)', fontsize=14, fontweight='bold')
    plt.xlabel('Baseline Expression (Log Scale)')
    plt.ylabel('Percentage Increase (%)')
    
    # 标注 Top 5 Gene
    if not df.empty:
        top_genes = df.nlargest(5, 'Pct_Gain')
        for _, row in top_genes.iterrows():
            plt.text(row['Baseline'], row['Pct_Gain'], row['Gene'], fontsize=9, fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'1_{tissue_name}_MA_Plot_Percentage.svg'), format='svg', dpi=600)
    plt.close()

    # -------------------------------------------------------
    # 📊 图 2: Gain Distribution
    # -------------------------------------------------------
    plt.figure(figsize=(5, 8)) 
    sns.kdeplot(df['Pct_Gain'], fill=True, color='b', alpha=0.6)
    # 根据数据动态调整 X 轴范围，防止过于空旷
    max_pct = df['Pct_Gain'].max() if not df['Pct_Gain'].empty else 100
    plt.xlim(-5, max_pct * 1.1) 
    plt.title(f'[{tissue_name}] Distribution of Best Gains', fontsize=14, fontweight='bold')
    plt.xlabel('Percentage Increase (%)') 
    plt.ylabel('Density')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'2_{tissue_name}_Gain_Distribution.svg'), format='svg')
    plt.close()

    # =======================================================
    # 🌟 图 3a: Vote Confidence (Top 8 Ranks) - 矮胖型
    # =======================================================
    plt.figure(figsize=(3.5, 2.2))
    
    cols = [f'Rank{i}_Vote' for i in range(1, 9)]
    if not all(c in df.columns for c in cols):
        cols = [c for c in cols if c in df.columns]

    if cols: # 确保有数据才画
        vote_df = df[cols].melt(var_name='Rank', value_name='Votes')
        vote_df['Rank'] = vote_df['Rank'].apply(lambda x: x.replace('_Vote', '').replace('Rank', 'Rank '))
        
        sns.boxplot(x='Votes', y='Rank', data=vote_df, palette="Blues_r", width=0.7, orient='h', fliersize=1.5)
        
        plt.title(f'[{tissue_name}] Consensus (Best Epoch)', fontsize=10, fontweight='bold', pad=4)
        plt.xlabel('Votes', fontsize=8)
        plt.ylabel('') 
        plt.xticks(fontsize=7)
        plt.yticks(fontsize=7)
        sns.despine(left=True, bottom=False)
        
        plt.tight_layout(pad=0.5)
        plt.savefig(os.path.join(output_dir, f'3a_{tissue_name}_Vote_Confidence.svg'), format='svg')
    plt.close()

    # =======================================================
    # 🌟 图 3b: Optimization Stability (Top 10 Only)
    # =======================================================
    # 注意：因为现在只取了 Best Epoch，Final_Gain 和 Max_Gain 是一样的
    # 所以这个图变成了展示 Top 10 Genes 的 Best Gain 的棒棒糖图 (Lollipop Chart)
    plt.figure(figsize=(3.5, 3.0))
    
    if not df.empty:
        top_10 = df.nlargest(10, 'Final_Gain').sort_values('Final_Gain', ascending=True)
        
        plt.hlines(y=top_10['Gene'], xmin=0, xmax=top_10['Final_Gain'], color='grey', alpha=0.5, linewidth=1.2)
        plt.scatter(top_10['Final_Gain'], top_10['Gene'], color='red', s=40, alpha=1.0, zorder=5)
        
        plt.title(f'[{tissue_name}] Top 10 Best Gains', fontsize=10, fontweight='bold', pad=4)
        plt.xlabel('Expression Gain', fontsize=8)
        plt.ylabel('') 
        
        plt.xticks(fontsize=7)
        plt.yticks(fontsize=7)
        plt.grid(axis='x', linestyle='--', alpha=0.5)
        
        plt.tight_layout(pad=0.5)
        plt.savefig(os.path.join(output_dir, f'3b_{tissue_name}_Top10_Lollipop.svg'), format='svg')
    plt.close()

    # -------------------------------------------------------
    # 📊 图 5: Top 15 Bar Chart
    # -------------------------------------------------------
    if not df.empty:
        top_15_pct = df.nlargest(15, 'Pct_Gain')
        plt.figure(figsize=(8, 8))
        sns.barplot(x='Pct_Gain', y='Gene', data=top_15_pct, palette='magma')
        plt.title(f'[{tissue_name}] Top 15 Genes by Relative Improvement %', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'5_{tissue_name}_Top15_Percentage.svg'), format='svg')
    plt.close()

# 🚀 运行
if __name__ == "__main__":
    args = parse_args()
    
    # 路径配置
    TARGET_K = args.k
    TISSUE = args.tissue
    
    folder_prefix = TISSUE
    
    BASE_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP/results'
    CSV_RES_DIR = os.path.join(BASE_DIR, f'{folder_prefix}_K{TARGET_K}_borzoi_modeltrain_res')
    OUTPUT_DIR = os.path.join(BASE_DIR, 'final_summary_plots', f'{TISSUE}_K{TARGET_K}')

    print(f"🚀 Starting Analysis for Tissue: {TISSUE}")
    print(f"   Target Dir: {CSV_RES_DIR}")

    final_df = collect_data(CSV_RES_DIR)
    
    if not final_df.empty:
        plot_charts_final(final_df, OUTPUT_DIR, TISSUE)
        print(f"✅ All charts generated for {TISSUE}! Check: {OUTPUT_DIR}")
    else:
        print("⚠️ No data found. Please check the tissue name and directory path.")
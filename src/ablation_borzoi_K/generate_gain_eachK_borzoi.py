'''
取的best epo去做ablation 和Benchmark。 
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import argparse
from tqdm import tqdm
from scipy.stats import entropy

# ================= ⚙️ 配置区域 =================
BASE_INPUT_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP/results'
BASE_OUTPUT_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP/results/ablation_borzoi_K/borzoi_gain'

# 你的 K 值列表
K_LIST = [1, 3, 5, 10, 20, 50] 

# ===============================================

def calculate_diversity_metrics(row_best, k):
    """
    计算投票的多样性指标 (Entropy, Gini)
    """
    if k == 1:
        return 0.0, 1.0 # K=1 没啥多样性可言，Entropy=0, Gini=1 (独裁)

    # 提取所有 Head 的投票分
    votes = []
    for i in range(1, k + 1):
        v = row_best.get(f'Rank{i}_Score', 0)
        votes.append(v)
    
    votes = np.array(votes)
    total_votes = np.sum(votes)
    
    if total_votes == 0:
        return 0.0, 0.0

    # 1. Normalized Entropy (归一化熵)
    # 范围 0-1。1代表完全均匀分布(最散)，0代表完全集中(最稳)
    probs = votes / total_votes
    ent = entropy(probs) / np.log(k) # Normalize by log(K)
    
    # 2. Gini Coefficient (基尼系数)
    # 范围 0-1。1代表极度不平等(集中)，0代表完全平等(分散)
    # Gini = (2 * AUC) - 1
    # 简化算法:
    sorted_votes = np.sort(votes)
    n = len(votes)
    cumulative = np.cumsum(sorted_votes)
    gini = (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n
    
    return ent, gini

def process_single_k(k_val):
    # 1. 构建输入路径
    # 假设你的文件夹命名格式是 multihead_MVP_res_K1, multihead_MVP_res_K3 ...
    input_dir = os.path.join(BASE_INPUT_DIR, f'multihead_MVP_res_K{k_val}')
    output_dir = os.path.join(BASE_OUTPUT_DIR, f'K{k_val}')
    
    print(f"\n🚀 Processing K={k_val}")
    print(f"   📂 Input:  {input_dir}")
    print(f"   💾 Output: {output_dir}")
    
    if not os.path.exists(input_dir):
        print(f"   ⚠️ Directory not found, skipping K={k_val}")
        return pd.DataFrame()

    os.makedirs(output_dir, exist_ok=True)
    
    csv_files = glob.glob(os.path.join(input_dir, "*_optim_log.csv"))
    summary_data = []
    
    for filename in tqdm(csv_files, desc=f"Reading K={k_val}", unit="file"):
        try:
            df = pd.read_csv(filename)
            if df.empty or 'Baseline' not in df.columns or 'Gain' not in df.columns: 
                continue

            gene_name = os.path.basename(filename).split('_optim_log')[0]
            
            # ✅ 取 Best Epoch
            best_idx = df['Gain'].idxmax()
            row_best = df.iloc[best_idx]
            
            baseline = row_best['Baseline']
            best_gain = row_best['Gain']
            safe_base = max(float(baseline), 0.1)
            pct_gain = (best_gain / safe_base) * 100
            
            # ✅ 计算多样性指标
            ent, gini = calculate_diversity_metrics(row_best, k_val)

            # 提取前 5 个 Vote (用于画图)
            vote_dict = {}
            for i in range(1, 6): # 只取前5个画图够了
                vote_dict[f'Rank{i}_Vote'] = row_best.get(f'Rank{i}_Score', 0)

            entry = {
                'Gene': gene_name,
                'K': k_val,
                'Baseline': baseline,
                'Max_Gain': best_gain,
                'Pct_Gain': pct_gain,
                'Entropy': ent,   # 越低越稳
                'Gini': gini      # 越高越稳(集中)
            }
            entry.update(vote_dict)
            summary_data.append(entry)
            
        except Exception:
            continue
            
    df_res = pd.DataFrame(summary_data)
    
    if df_res.empty:
        print(f"   ⚠️ No valid data found for K={k_val}")
        return df_res

    # === 保存 Summary CSV (给 Table 1 用) ===
    summary_csv_path = os.path.join(output_dir, f'summary_metrics_K{k_val}.csv')
    df_res.to_csv(summary_csv_path, index=False)
    print(f"   ✅ Metrics saved to: {summary_csv_path}")
    
    # === 画图 ===
    plot_charts(df_res, output_dir, k_val)
    
    return df_res

def plot_charts(df, output_dir, k):
    sns.set(style="whitegrid", font_scale=0.9)
    
    # 1. Gain Distribution (Density)
    plt.figure(figsize=(5, 4))
    sns.kdeplot(df['Pct_Gain'], fill=True, color='purple', alpha=0.6)
    plt.xlim(-5, df['Pct_Gain'].max() * 1.1)
    plt.title(f'Gain Distribution (K={k})', fontsize=12, fontweight='bold')
    plt.xlabel('Percentage Gain (%)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'gain_dist_K{k}.png'), dpi=300)
    plt.close()
    
    # 2. Vote Confidence (Entropy vs. Gain) - 看看稳不稳
    if k > 1:
        plt.figure(figsize=(6, 5))
        sns.scatterplot(data=df, x='Entropy', y='Pct_Gain', alpha=0.6, color='teal')
        plt.title(f'Diversity vs. Gain (K={k})', fontsize=12)
        plt.xlabel('Vote Entropy (0=Focused, 1=Random)')
        plt.ylabel('Percentage Gain')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'entropy_vs_gain_K{k}.png'), dpi=300)
        plt.close()

def main():
    all_k_data = []
    
    for k in K_LIST:
        df_k = process_single_k(k)
        if not df_k.empty:
            # 计算全局平均指标
            avg_gain = df_k['Max_Gain'].mean()
            avg_pct = df_k['Pct_Gain'].mean()
            avg_gini = df_k['Gini'].mean()
            
            all_k_data.append({
                'K': k,
                'Mean_Raw_Gain': avg_gain,
                'Mean_Pct_Gain': avg_pct,
                'Mean_Gini': avg_gini
            })
            
    # 最后生成一个总的 Ablation 趋势表
    if all_k_data:
        final_summary = pd.DataFrame(all_k_data)
        final_path = os.path.join(BASE_OUTPUT_DIR, 'ablation_trend_summary.csv')
        final_summary.to_csv(final_path, index=False)
        print(f"\n🏆 All done! Global trend saved to: {final_path}")
        print(final_summary)

if __name__ == "__main__":
    main()
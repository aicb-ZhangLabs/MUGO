import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
import numpy as np
from matplotlib.lines import Line2D

def parse_args():
    parser = argparse.ArgumentParser(description="Compare Enformer and Borzoi performance with Caching.")
    
    # --- 路径配置 ---
    default_gene_list = "/home/dongbos/Combine_optim_Borzoi_SNP/dataset/gene_3000_borzoi_gencode_v41_hg38.csv"
    default_borzoi_dir = "/home/dongbos/Combine_optim_Borzoi_SNP/results/blood_K10_borzoi_CAGE_modeltrain_res"
    default_enformer_dir = "/home/dongbos/Combine_optim_Borzoi_SNP/results/blood_K10_enformer_modeltrain_CAGE_res"
    
    # 新的输出文件夹
    default_output_dir = "/home/dongbos/Combine_optim_Borzoi_SNP/results/compare_enforemr_borzoi"
    
    parser.add_argument("--tissue", type=str, default="blood", help="Tissue name (default: blood)")
    parser.add_argument("--gene_list", type=str, default=default_gene_list, help="Path to the gene list CSV")
    parser.add_argument("--borzoi_dir", type=str, default=default_borzoi_dir, help="Directory for Borzoi results")
    parser.add_argument("--enformer_dir", type=str, default=default_enformer_dir, help="Directory for Enformer results")
    parser.add_argument("--output_dir", type=str, default=default_output_dir, help="Directory to save CSV and Plot")
    
    return parser.parse_args()

def get_max_epoch_gain(file_path):
    """
    读取 CSV，按 Step 排序，返回最后一步（最大 Epoch）的 Gain。
    """
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            return None
        last_step = df.sort_values(by="Step").iloc[-1]
        return last_step['Gain']
    except Exception:
        return None

def main():
    args = parse_args()
    
    # 0. 准备输出目录
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")
    
    # --- Modify filenames to include tissue name ---
    cache_csv_path = os.path.join(args.output_dir, f"{args.tissue}_extracted_gains_cache.csv")
    plot_path = os.path.join(args.output_dir, f"{args.tissue}_model_comparison_scatter_fitted.png")

    # 1. 读取原始 Gene List
    print(f"Loading master gene list from: {args.gene_list}")
    try:
        master_df = pd.read_csv(args.gene_list)
    except FileNotFoundError:
        print("Error: Gene list file not found.")
        return

    # 2. 检查缓存 (Cache Logic)
    cached_data = pd.DataFrame()
    processed_genes = set()

    if os.path.exists(cache_csv_path):
        print(f"Found cached data at: {cache_csv_path}")
        cached_data = pd.read_csv(cache_csv_path)
        # 记录已经处理过的 gene_name
        if not cached_data.empty:
            processed_genes = set(cached_data['gene_name'].astype(str))
        print(f"Loaded {len(cached_data)} genes from cache.")
    else:
        print(f"No cache found for {args.tissue}. Starting fresh.")

    # 3. 筛选出需要处理的 Genes
    # 只有不在 processed_genes 里的才需要去读文件
    genes_to_process = []
    for _, row in master_df.iterrows():
        g_name = str(row['gene_name'])
        if g_name not in processed_genes:
            genes_to_process.append(g_name)
    
    print(f"Total genes: {len(master_df)}. Need to process: {len(genes_to_process)}.")

    # 4. 处理新基因 (读取文件)
    new_results = []
    
    if len(genes_to_process) > 0:
        for gene_name in tqdm(genes_to_process, desc="Reading Logs for new genes"):
            # 构造文件名
            enf_filename = f"{gene_name}_enformer_optim_log.csv"
            bor_filename = f"{gene_name}_borzoi_CAGE_optim_log.csv"
            
            enf_path = os.path.join(args.enformer_dir, enf_filename)
            bor_path = os.path.join(args.borzoi_dir, bor_filename)
            
            if os.path.exists(enf_path) and os.path.exists(bor_path):
                g_enf = get_max_epoch_gain(enf_path)
                g_bor = get_max_epoch_gain(bor_path)
                
                if g_enf is not None and g_bor is not None:
                    new_results.append({
                        'gene_name': gene_name,
                        'enformer_gain': g_enf,
                        'borzoi_gain': g_bor
                    })
    else:
        print("All genes are already cached. Skipping file reading.")

    # 5. 合并数据并更新缓存
    if new_results:
        new_df = pd.DataFrame(new_results)
        # 合并旧数据和新数据
        final_df = pd.concat([cached_data, new_df], ignore_index=True)
        # 保存回 CSV
        final_df.to_csv(cache_csv_path, index=False)
        print(f"Added {len(new_df)} new genes. Cache updated at: {cache_csv_path}")
    else:
        final_df = cached_data
        print("No new valid data found or processed.")

    # 如果没有任何数据（既没有缓存也没读到新数据），退出
    if final_df.empty:
        print("No valid data available to plot.")
        return
    
    print(f"Total data points for plotting: {len(final_df)}")

    # 6. 绘图准备
    x = final_df['enformer_gain'].values
    y = final_df['borzoi_gain'].values

    if len(x) < 2:
        print("Not enough points to calculate correlation.")
        return

    # 计算统计指标
    p_corr, _ = pearsonr(x, y)
    s_corr, _ = spearmanr(x, y)

    # 7. 绘图
    plt.figure(figsize=(8, 8))
    
    plt.scatter(x, y, alpha=0.5, s=20, edgecolors='w', linewidth=0.5, color='steelblue', label='Genes')
    
    # 拟合线
    slope, intercept = np.polyfit(x, y, 1)
    fit_line = slope * x + intercept
    plt.plot(x, fit_line, color='darkorange', linewidth=2, label=f'Fit Line (Slope={slope:.2f})')
    
    # 对角线
    all_vals = np.concatenate([x, y])
    min_val, max_val = np.min(all_vals), np.max(all_vals)
    padding = (max_val - min_val) * 0.05
    plt.plot([min_val - padding, max_val + padding], [min_val - padding, max_val + padding], 
             'k--', alpha=0.3, label='Identity (x=y)')
    
    plt.title(f"Enformer vs Borzoi Gain Comparison ({args.tissue})", fontsize=14)
    plt.xlabel("Enformer Max Epoch Gain", fontsize=12)
    plt.ylabel("Borzoi Max Epoch Gain", fontsize=12)
    
    # 图例
    handles, labels = plt.gca().get_legend_handles_labels()
    stats_handle = Line2D([], [], color='none', label=f'Pearson r = {p_corr:.3f}')
    stats_handle2 = Line2D([], [], color='none', label=f'Spearman ρ = {s_corr:.3f}')
    handles.extend([stats_handle, stats_handle2])
    
    plt.legend(handles=handles, loc='best', frameon=True, fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(plot_path, dpi=300)
    print(f"Plot saved to: {plot_path}")

if __name__ == "__main__":
    main()
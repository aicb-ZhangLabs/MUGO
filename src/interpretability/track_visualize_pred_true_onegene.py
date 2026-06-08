import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

# ================= 🔧 基础配置路径 =================
BASE_RES_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP/results'
META_CSV_PATH = '/home/dongbos/Combine_optim_Borzoi_SNP/dataset/gene_3000_borzoi_gencode_v41_hg38.csv'
GTEX_DATA_PATH = '/home/dongbos/Combine_optim_Borzoi_SNP/dataset/GTEx_Analysis_v8_eQTL/Whole_Blood.v8.signif_variant_gene_pairs.txt.gz'
# ==================================================

plt.rcParams['svg.fonttype'] = 'none'

def ensure_dir(path):
    if not os.path.exists(path): os.makedirs(path)

def get_gene_info(index, meta_path):
    df = pd.read_csv(meta_path)
    row = df.iloc[index]
    return {'gene_name': row['gene_name'], 'gene_id_clean': row['gene_ID'].split('.')[0], 'chr': str(row['chr'])}

def parse_model_snps(log_path, k_val):
    """
    动态读取 Rank 1 到 Rank K 的数据
    """
    if not os.path.exists(log_path): 
        print(f"❌ File not found: {log_path}")
        return pd.DataFrame()
    
    df = pd.read_csv(log_path)
    if df.empty: return pd.DataFrame()

    last_row = df.iloc[-1]
    snps = []
    
    # 遍历 1 到 K
    for i in range(1, k_val + 1):
        col_pos = f'Rank{i}_Pos'
        col_score = f'Rank{i}_Score'
        
        if col_pos in last_row:
            snps.append({
                'pos': int(last_row[col_pos]), 
                'score': float(last_row[col_score]), 
                'rank': i
            })
            
    return pd.DataFrame(snps)

def load_gtex_variants(gene_id_clean, gtex_path):
    # 读取 GTEx 数据
    use_cols = ['variant_id', 'gene_id', 'pval_nominal']
    chunks = []
    try:
        # 如果文件很大，分块读取
        reader = pd.read_csv(gtex_path, sep='\t', usecols=use_cols, chunksize=100000, compression='gzip')
        for chunk in reader:
            filtered = chunk[chunk['gene_id'].str.startswith(gene_id_clean)].copy()
            if not filtered.empty:
                filtered['pos'] = filtered['variant_id'].map(lambda x: int(x.split('_')[1]))
                chunks.append(filtered)
        if chunks:
            df = pd.concat(chunks)
            df['neg_log_p'] = -np.log10(df['pval_nominal'])
            return df
    except Exception as e:
        print(f"⚠️ Error loading GTEx: {e}")
        pass
    return pd.DataFrame()

def plot_adaptive_mirror(gene_info, model_df, gtex_df, save_path):
    """
    自适应视野 + 纯净背景 (Mirror Plot)
    """
    gene_name = gene_info['gene_name']
    
    # === 1. 计算视野范围 (Adaptive Zoom) ===
    important_pos = list(model_df['pos']) # 所有的 Model预测点
    
    # 找出重合点
    hits = []
    if not gtex_df.empty:
        common_pos = set(model_df['pos']).intersection(set(gtex_df['pos']))
        hits = list(common_pos)
        important_pos.extend(hits)
        
    # 计算 Min/Max
    if not important_pos: 
        print("⚠️ No valid positions to plot.")
        return
    
    x_min, x_max = min(important_pos), max(important_pos)
    span = x_max - x_min
    
    # 增加 Padding (左右各留 20% 的空间)
    padding = max(span * 0.2, 2000) # 最小留 2kb
    x_min -= padding
    x_max += padding
    
    # 筛选绘图数据
    m_df = model_df[(model_df['pos'] >= x_min) & (model_df['pos'] <= x_max)]
    g_df = gtex_df[(gtex_df['pos'] >= x_min) & (gtex_df['pos'] <= x_max)] if not gtex_df.empty else pd.DataFrame()

    # === 2. 绘图 ===
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.05})
    
    # --- Top: Model ---
    # 蓝色棒棒糖
    markerline, stemlines, baseline = ax1.stem(m_df['pos'], m_df['score'], linefmt='C0-', markerfmt='C0o', basefmt=" ")
    plt.setp(stemlines, 'linewidth', 2, 'alpha', 0.8)
    plt.setp(markerline, 'markersize', 9)
    
    ax1.set_ylabel("Model Importance", fontsize=13, fontweight='bold', color='C0')
    ax1.set_title(f"Target Identification: {gene_name}", fontsize=16, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.3)
    # y轴稍微留一点空间
    ax1.set_ylim(0, m_df['score'].max() * 1.25 if not m_df.empty else 1.0)

    # --- Bottom: GTEx (Ground Truth) ---
    if not g_df.empty:
        # 区分显著和不显著
        sig_mask = g_df['neg_log_p'] > 5 # p < 1e-5 (heuristic threshold)
        
        # 1. 不显著的背景点 (浅灰)
        ax2.scatter(g_df[~sig_mask]['pos'], g_df[~sig_mask]['neg_log_p'], 
                    s=15, color='#E0E0E0', alpha=0.5, label='Non-significant', zorder=1)
        
        # 2. 显著的背景点 (深灰)
        ax2.scatter(g_df[sig_mask]['pos'], g_df[sig_mask]['neg_log_p'], 
                    s=15, color='gray', alpha=0.7, label='GTEx Significant', zorder=2)

    ax2.set_ylabel(r"$-\log_{10}(P_{GTEx})$", fontsize=13, fontweight='bold', color='gray')
    ax2.set_xlabel(f"Genomic Position (chr{gene_info['chr']})", fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    # 倒影 Y 轴
    max_p = g_df['neg_log_p'].max() if not g_df.empty else 5
    ax2.set_ylim(max(max_p * 1.1, 8), 0) # 0 在最上面

    # === 3. 高亮 Hits (重合点) ===
    for pos in hits:
        if pos < x_min or pos > x_max: continue
        
        score_val = model_df[model_df['pos'] == pos]['score'].values[0]
        # 注意: 这里一定要取 gtex_df 原始值，不能取被截断的 g_df
        matches = gtex_df[gtex_df['pos'] == pos]
        if matches.empty: continue
        pval = matches['neg_log_p'].values[0]
        
        # 只有分数还不错的才高亮 (>0.1, 阈值可调)
        if score_val > 0.0:
            # 画红星
            ax1.scatter([pos], [score_val], color='#D62728', s=200, zorder=10, marker='*', edgecolors='black', label='Validated Hit')
            ax2.scatter([pos], [pval], color='#D62728', s=200, zorder=10, marker='*', edgecolors='black')
            
            # 画贯穿线
            ax1.axvline(x=pos, color='#D62728', linestyle='--', alpha=0.5, linewidth=1.5)
            ax2.axvline(x=pos, color='#D62728', linestyle='--', alpha=0.5, linewidth=1.5)
            
            # 标注 P 值
            ax2.annotate(f"P={10**(-pval):.1e}", xy=(pos, pval), xytext=(5, 5), textcoords='offset points',
                         color='#D62728', fontsize=9, fontweight='bold', bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, ec="none"))

    ax1.set_xlim(x_min, x_max)
    
    # Legend
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label: ax1.legend(by_label.values(), by_label.keys(), loc='upper right')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.0)
    
    print(f"💾 Saved plot to {save_path}")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, default=155, help="Index of the gene in metadata CSV")
    parser.add_argument('--k', type=int, default=10, help="K value to select input/output folder") 
    args = parser.parse_args()
    
    # === 动态构建路径 ===
    # 输入：例如 .../multihead_MVP_res_K10
    model_res_dir = os.path.join(BASE_RES_DIR, f'multihead_MVP_res_K{args.k}')
    
    # 输出：例如 .../results/plots/K10
    output_img_dir = os.path.join(BASE_RES_DIR, 'plots', f'K{args.k}')
    
    ensure_dir(output_img_dir)
    
    # === 开始处理 ===
    info = get_gene_info(args.index, META_CSV_PATH)
    print(f"🎨 [K={args.k}] Plotting for {info['gene_name']} (Index {args.index})...")
    
    # 构造 CSV 路径
    csv_path = os.path.join(model_res_dir, f"{info['gene_name']}_optim_log.csv")
    
    # 读取数据
    m_df = parse_model_snps(csv_path, args.k)
    
    if m_df.empty:
        print(f"⚠️ No model results found for {info['gene_name']} in K={args.k} folder.")
    else:
        g_df = load_gtex_variants(info['gene_id_clean'], GTEX_DATA_PATH)
        
        save_path = os.path.join(output_img_dir, f"{info['gene_name']}_clean_mirror.svg")
        
        plot_adaptive_mirror(info, m_df, g_df, save_path)
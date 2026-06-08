import os
import pandas as pd
import numpy as np
import argparse
import pyBigWig
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import gzip
import seaborn as sns

# ==========================================
# 1. 配置路径 (请根据你的环境修改)
# ==========================================
BASE_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP'
DATASET_DIR = f'{BASE_DIR}/dataset'
RESULTS_BASE_DIR = f'{BASE_DIR}/results'

META_CSV = f'{DATASET_DIR}/gene_3000_borzoi_gencode_v41_hg38.csv'
GTF_PATH = f'{DATASET_DIR}/gencode.v41.annotation.gtf.gz'  # 必须要有这个才能画基因结构
PHYLOP_BW = f'{DATASET_DIR}/PolyP_hg38/hg38.phyloP100way.bw'

# 输出目录
OUTPUT_PLOT_DIR = f'{RESULTS_BASE_DIR}/res_enrichment_conservative_borzoi/case_studies'
os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)

# 绘图风格设置
plt.style.use('seaborn-v0_8-white')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.linewidth'] = 1.5

# ==========================================
# 2. 辅助函数
# ==========================================

def get_gene_meta(index, meta_path):
    df = pd.read_csv(meta_path)
    row = df.iloc[index]
    chrom = f"chr{row['chr']}".replace('chrchr', 'chr')
    return {
        'name': row['gene_name'],
        'id': row['gene_ID'],
        'chrom': chrom,
        'tss': int(row['pos']),
        'strand': row['strand']
    }

def get_exons(gene_id, gtf_path, chrom, center_pos, window_size=20000):
    """从GTF读取Exon结构，只保留窗口内的"""
    exons = []
    start_limit = center_pos - window_size
    end_limit = center_pos + window_size
    
    try:
        with gzip.open(gtf_path, 'rt') as f:
            for line in f:
                if line.startswith('#'): continue
                parts = line.strip().split('\t')
                if len(parts) < 9: continue
                if parts[2] != 'exon': continue
                if gene_id not in parts[8]: continue
                
                # 检查是否在染色体上 (简单检查)
                if parts[0] != chrom and f"chr{parts[0]}" != chrom: continue

                e_start = int(parts[3])
                e_end = int(parts[4])
                
                # 只要在窗口内的，或者这就够了
                if e_end > start_limit and e_start < end_limit:
                    exons.append((e_start, e_end))
    except Exception as e:
        print(f"⚠️ Warning: Could not read GTF ({e}). Gene track will be empty.")
        
    return exons

def parse_top_snps(log_path, k):
    if not os.path.exists(log_path): return []
    df = pd.read_csv(log_path)
    if df.empty: return []
    last = df.iloc[-1]
    
    snps = []
    for i in range(1, k+1):
        p_col = f"Rank{i}_Pos"
        s_col = f"Rank{i}_Score"
        m_col = f"Rank{i}_RefAlt"
        if p_col in last:
            snps.append({
                'Rank': i,
                'Pos': int(last[p_col]),
                'Score': float(last[s_col]),
                'Mutation': last[m_col]
            })
    return snps

def get_phylop(chrom, snps, bw_path):
    try:
        bw = pyBigWig.open(bw_path)
        for s in snps:
            # Check range
            try:
                val = bw.values(chrom, s['Pos'], s['Pos']+1)[0]
                s['PhyloP'] = val if not np.isnan(val) else 0.0
            except:
                s['PhyloP'] = 0.0
        bw.close()
    except:
        for s in snps: s['PhyloP'] = 0.0
    return snps

# ==========================================
# 3. 核心绘图逻辑 (The Fancy Part)
# ==========================================

def plot_case_study(gene_info, snps, exons, output_path):
    print(f"🎨 Drawing case study for {gene_info['name']}...")
    
    # 定义绘图窗口 (以 TSS 为中心，或者包含所有 SNP)
    snp_pos = [s['Pos'] for s in snps]
    min_p, max_p = min(snp_pos), max(snp_pos)
    padding = 2000 # 左右留白 2kb
    xlims = (min_p - padding, max_p + padding)
    
    # 创建画布: 上面是 Lollipop，下面是 Gene Structure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
    
    # --- Colormap setup ---
    # PhyloP 越高越红，越低越蓝/灰
    cmap = plt.cm.get_cmap('RdYlBu_r') # Red=Conserved, Blue=Fast-Evolving
    norm = mcolors.Normalize(vmin=-2, vmax=6) # 设置颜色范围，Top 1% (~3.5) 会很红
    
    # === AX1: Lollipop Plot ===
    
    # 绘制基线
    ax1.axhline(0, color='black', linewidth=1)
    
    max_score = 0
    
    for s in snps:
        x = s['Pos']
        y = s['Score']
        p = s['PhyloP']
        rank = s['Rank']
        max_score = max(max_score, y)
        
        color = cmap(norm(p))
        
        # 1. 画茎 (Stem)
        ax1.plot([x, x], [0, y], color='gray', linestyle='--', linewidth=1, alpha=0.6, zorder=1)
        
        # 2. 画糖 (Head)
        # 大小随 Rank 变化 (Rank 1 最大)
        size = 150 if rank == 1 else (100 if rank <=3 else 60)
        edge_color = 'black' if p > 3.5 else 'none' # Top 1% 加黑边框
        lw = 2 if p > 3.5 else 0
        
        sc = ax1.scatter(x, y, c=[p], cmap='RdYlBu_r', norm=norm, s=size, edgecolor=edge_color, linewidth=lw, zorder=2)
        
        # 3. 标注 (只标 Top 3 或 极度保守的)
        is_top_conserved = p > 3.5
        if rank <= 3 or is_top_conserved:
            label = f"Rank {rank}\n{s['Mutation']}"
            if is_top_conserved:
                label += f"\nPhyloP={p:.1f}★"
                font_weight = 'bold'
            else:
                font_weight = 'normal'
                
            ax1.annotate(label, (x, y), xytext=(0, 10), textcoords='offset points', 
                         ha='center', fontsize=9, fontweight=font_weight,
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    # 添加 Colorbar
    cbar = plt.colorbar(sc, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label('PhyloP Conservation Score', rotation=270, labelpad=15)
    
    # 装饰 AX1
    ax1.set_ylabel('Model Importance Score', fontsize=12, fontweight='bold')
    ax1.set_title(f"Case Study: {gene_info['name']} ({gene_info['chrom']})", fontsize=16, fontweight='bold', pad=15)
    ax1.set_ylim(0, max_score * 1.2)
    ax1.grid(axis='y', linestyle=':', alpha=0.5)
    
    # 标注阈值线 (Top 1% Conservation threshold)
    # 可以在 Colorbar 上做文章，或者在图例里说明
    
    # === AX2: Gene Structure Track ===
    
    # 画一条中心线代表 Intron
    ax2.plot([xlims[0], xlims[1]], [0.5, 0.5], color='#2c3e50', linewidth=1, zorder=1)
    
    # 画 Exons
    gene_color = '#34495e'
    for start, end in exons:
        # 确保 Exon 在显示范围内
        if end < xlims[0] or start > xlims[1]: continue
        
        width = end - start
        rect = patches.Rectangle((start, 0.25), width, 0.5, linewidth=0, edgecolor=None, facecolor=gene_color, zorder=2)
        ax2.add_patch(rect)
        
    # 标注 TSS
    tss = gene_info['tss']
    if xlims[0] < tss < xlims[1]:
        ax2.plot([tss, tss], [0.25, 0.75], color='#e74c3c', linewidth=2)
        ax2.text(tss, 0.8, "TSS", ha='center', color='#e74c3c', fontsize=10, fontweight='bold')
        # 画个箭头表示方向
        direction = 1 if gene_info['strand'] == '+' else -1
        ax2.arrow(tss, 0.9, direction * 500, 0, head_width=0.1, head_length=200, fc='#e74c3c', ec='#e74c3c')

    # 装饰 AX2
    ax2.set_yticks([])
    ax2.set_xlabel(f'Genomic Position ({gene_info["chrom"]})', fontsize=12)
    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_xlim(xlims)
    
    # 加上图例说明
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='w', markeredgecolor='black', markersize=10, label='Top 1% Conserved (PhyloP > 3.5)'),
        patches.Patch(facecolor=gene_color, label='Exons'),
    ]
    ax2.legend(handles=legend_elements, loc='upper right', frameon=False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"✅ Plot saved to: {output_path}")
    plt.close()

# ==========================================
# 4. 主程序
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, required=True, help="Gene index")
    parser.add_argument('--k', type=int, default=10, help="K value used")
    args = parser.parse_args()
    
    # 1. 获取基因信息
    gene_info = get_gene_meta(args.index, META_CSV)
    
    # 2. 读取 Log 获取 SNPs
    log_file = f"{RESULTS_BASE_DIR}/multihead_MVP_res_K{args.k}/{gene_info['name']}_optim_log.csv"
    snps = parse_top_snps(log_file, args.k)
    
    if not snps:
        print(f"❌ No SNPs found in {log_file}")
        return
        
    # 3. 获取 PhyloP 分数
    snps = get_phylop(gene_info['chrom'], snps, PHYLOP_BW)
    
    # 4. 获取 Exon 结构 (用于画底部 Track)
    # 为了画得好看，我们取 SNP 分布的范围稍微扩大一点
    center_pos = int(np.mean([s['Pos'] for s in snps]))
    exons = get_exons(gene_info['id'], GTF_PATH, gene_info['chrom'], center_pos, window_size=50000)
    
    # 5. 绘图
    out_file = f"{OUTPUT_PLOT_DIR}/{gene_info['name']}_case_study.png"
    plot_case_study(gene_info, snps, exons, out_file)

if __name__ == "__main__":
    main()
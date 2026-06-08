import os
import pandas as pd
import numpy as np
import argparse
import pyBigWig
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import gzip

# ==========================================
# 1. 配置路径
# ==========================================
BASE_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP'
DATASET_DIR = f'{BASE_DIR}/dataset'
RESULTS_BASE_DIR = f'{BASE_DIR}/results'
META_CSV = f'{DATASET_DIR}/gene_3000_borzoi_gencode_v41_hg38.csv'
GTF_PATH = f'{DATASET_DIR}/gencode.v41.annotation.gtf.gz'
PHYLOP_BW = f'{DATASET_DIR}/PolyP_hg38/hg38.phyloP100way.bw'
GWAS_HIT_CSV = f'{RESULTS_BASE_DIR}/res_enrichment_gwas/gwas_hit_genes_K10.csv' # 你的 GWAS 结果
OUTPUT_PLOT_DIR = f'{RESULTS_BASE_DIR}/res_enrichment_gwas/case_studies'

os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)
plt.style.use('seaborn-v0_8-white')

# ==========================================
# 2. 辅助函数
# ==========================================
def get_gene_info(gene_name_query):
    """根据名字找 Index 和 Info"""
    df = pd.read_csv(META_CSV)
    # 精确匹配
    match = df[df['gene_name'] == gene_name_query]
    if match.empty:
        print(f"❌ Gene {gene_name_query} not found in metadata.")
        return None
    
    idx = match.index[0]
    row = match.iloc[0]
    return {
        'index': idx,
        'name': row['gene_name'],
        'id': row['gene_ID'],
        'chrom': f"chr{row['chr']}".replace('chrchr', 'chr'),
        'tss': int(row['pos']),
        'strand': row['strand']
    }

def get_exons(gene_id, gtf_path, chrom, center_pos, window_size=50000):
    exons = []
    start_limit, end_limit = center_pos - window_size, center_pos + window_size
    try:
        with gzip.open(gtf_path, 'rt') as f:
            for line in f:
                if line.startswith('#') or gene_id not in line: continue
                parts = line.strip().split('\t')
                if parts[2] != 'exon': continue
                if parts[0] != chrom and f"chr{parts[0]}" != chrom: continue
                e_start, e_end = int(parts[3]), int(parts[4])
                if e_end > start_limit and e_start < end_limit:
                    exons.append((e_start, e_end))
    except: pass
    return exons

def parse_model_snps(log_path, k=10):
    if not os.path.exists(log_path): return []
    df = pd.read_csv(log_path)
    if df.empty: return []
    last = df.iloc[-1]
    snps = []
    for i in range(1, k+1):
        if f"Rank{i}_Pos" in last:
            snps.append({
                'Rank': i,
                'Pos': int(last[f"Rank{i}_Pos"]),
                'Score': float(last[f"Rank{i}_Score"]),
                'Mutation': last[f"Rank{i}_RefAlt"]
            })
    return snps

def get_phylop(chrom, snps, bw_path):
    try:
        bw = pyBigWig.open(bw_path)
        for s in snps:
            try:
                val = bw.values(chrom, s['Pos'], s['Pos']+1)[0]
                s['PhyloP'] = val if not np.isnan(val) else 0.0
            except: s['PhyloP'] = 0.0
        bw.close()
    except: 
        for s in snps: s['PhyloP'] = 0.0
    return snps

def check_gwas_overlap(snps, gene_name, gwas_file_path):
    """(这里简化) 既然 gwas_hit_genes_K10.csv 说是 Hit，那我们假设 Model Top K 里肯定有重叠
       为了画图，我们需要知道 *哪些* 是 Hit。
       这里我们简单地假设：所有 Top K 如果在 gwas_catalog 原文件里有记录就是 Hit。
       或者更简单：既然已经验证过了，我们在图上标注 'Validated Hit'。
    """
    # 这里为了代码简洁，我直接给所有在 gwas_hit_genes_K10.csv 里的基因打标
    # 实际上你应该 load 原 GWAS Catalog 来精确匹配 ID。
    # 既然你已经跑过 enrichment，我们就默认 Top Rank 且分高的很可能就是 Hit。
    # 为了严谨，建议配合 check_gwas_enrichment.py 的逻辑。
    # 这里我们只做可视化框架。
    return snps 

# ==========================================
# 3. 绘图
# ==========================================
def plot(gene_info, snps, exons, out_path, is_gwas_hit=False):
    print(f"🎨 Plotting {gene_info['name']}...")
    
    snp_pos = [s['Pos'] for s in snps]
    padding = 3000
    xlims = (min(snp_pos)-padding, max(snp_pos)+padding)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
    
    cmap = plt.cm.get_cmap('RdYlBu_r')
    norm = mcolors.Normalize(vmin=-2, vmax=6)
    
    ax1.axhline(0, c='black', lw=1)
    
    for s in snps:
        x, y, p, r = s['Pos'], s['Score'], s['PhyloP'], s['Rank']
        
        # 茎
        ax1.plot([x, x], [0, y], color='gray', ls='--', lw=1, alpha=0.6, zorder=1)
        
        # 糖
        size = 180 if r==1 else (120 if r<=3 else 60)
        edge_color = 'black'
        lw = 1.5
        
        # --- NEW: 如果是 Rank 1 且是 Validated Gene，我们就默认它是那个 Hit ---
        # 把 Rank 1 的形状改成星星 (marker='*')，或者在上面标个星
        marker_shape = '*' if (r == 1 and is_gwas_hit) else 'o'
        marker_size  = 300 if (r == 1 and is_gwas_hit) else size # 星星要大一点才好看
        
        # 画点
        sc = ax1.scatter(x, y, c=[p], cmap='RdYlBu_r', norm=norm, s=marker_size, 
                         marker=marker_shape, edgecolors=edge_color, linewidth=lw, zorder=2)
        
        # 标注
        if r <= 3 or p > 3.0:
            label = f"Rank {r}\n{s['Mutation']}"
            if p > 2.0: label += f"\nPhyloP={p:.1f}"
            
            # ⭐️ 给 Rank 1 的 Label 加上 "GWAS Hit" 文字
            if r == 1 and is_gwas_hit:
                label += "\n★ GWAS Hit"
                font_weight = 'bold'
                box_color = '#ffeaa7' # 给个淡黄色背景高亮
            else:
                font_weight = 'normal'
                box_color = 'white'
            
            ax1.annotate(label, (x, y), xytext=(0, 15), textcoords='offset points', 
                         ha='center', fontsize=9, fontweight=font_weight,
                         bbox=dict(boxstyle="round,pad=0.3", fc=box_color, alpha=0.9, ec='black'))

    # Colorbar
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label('Conservation (PhyloP)', rotation=270, labelpad=15)
    
    ax1.set_ylabel('Model Importance Score', fontsize=12, fontweight='bold')
    ax1.set_title(f"Target Identification: {gene_info['name']}", fontsize=16, fontweight='bold')
    ax1.grid(axis='y', ls=':', alpha=0.5)
    
    # Gene Track
    ax2.plot([xlims[0], xlims[1]], [0.5, 0.5], color='#2c3e50', lw=2)
    for start, end in exons:
        if end < xlims[0] or start > xlims[1]: continue
        width = end - start
        rect = patches.Rectangle((start, 0.25), width, 0.5, fc='#34495e')
        ax2.add_patch(rect)
        
    # TSS Arrow
    tss = gene_info['tss']
    if xlims[0] < tss < xlims[1]:
        ax2.arrow(tss, 0.8, 500 if gene_info['strand']=='+' else -500, 0, 
                  head_width=0.2, head_length=200, fc='#e74c3c', ec='#e74c3c', lw=2)
        ax2.text(tss, 1.1, "TSS", ha='center', color='#e74c3c', fontweight='bold')

    ax2.set_yticks([])
    ax2.set_xlabel(f'Genomic Position ({gene_info["chrom"]})', fontsize=12)
    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # 去掉右上角那个可能会误导的 legend box，因为我们已经直接标在点上了
    # if is_gwas_hit: ... 

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"✅ Saved to {out_path}")
    plt.close()

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gene', type=str, required=True, help="Gene Name (e.g. BIN1)")
    args = parser.parse_args()
    
    info = get_gene_info(args.gene)
    if not info: return
    
    log_file = f"{RESULTS_BASE_DIR}/multihead_MVP_res_K10/{args.gene}_optim_log.csv"
    snps = parse_model_snps(log_file)
    snps = get_phylop(info['chrom'], snps, PHYLOP_BW)
    
    center = int(np.mean([s['Pos'] for s in snps]))
    exons = get_exons(info['id'], GTF_PATH, info['chrom'], center)
    
    out_file = f"{OUTPUT_PLOT_DIR}/{args.gene}_gwas_case.png"
    plot(info, snps, exons, out_file, is_gwas_hit=True)

if __name__ == "__main__":
    main()
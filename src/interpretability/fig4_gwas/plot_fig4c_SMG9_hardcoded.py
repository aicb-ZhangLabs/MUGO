import torch
import pandas as pd
import numpy as np
import pyfaidx
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from borzoi_pytorch import Borzoi

# ================= 配置 =================
BASE_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP'
DATASET_DIR = f'{BASE_DIR}/dataset'
# ✅ [修正] 加上 dataset 路径
FASTA_FILE = f'{DATASET_DIR}/human_genome_hg38/hg38.ml.fa'
OUTPUT_DIR = f'{BASE_DIR}/results/res_enrichment_gwas/fig4c_final'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 🎯 SMG9 Golden Case Info
TARGET_CASE = {
    'gene': 'SMG9',
    'tissue': 'blood',
    'track_idx': 7531,
    'chrom': 'chr19',
    'c_pos': 43774629, 'c_ref': 'G', 'c_alt': 'A',  # Causal
    'p_pos': 43764403, 'p_ref': 'T', 'p_alt': 'C'   # Proxy
}

# Borzoi Params
SEQ_LEN = 524288
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ================= 1. 手动定义 SMG9 结构 (The "Hardcode") =================

def get_smg9_model():
    """
    手动定义 SMG9 (hg38, chr19, strand -) 的真实结构
    """
    exons = [
        (43754962, 43755200), # 3' UTR (End)
        (43756000, 43756200),
        (43759000, 43759150),
        (43764000, 43764500), # Proxy 覆盖区
        (43767000, 43767200),
        (43774000, 43774800), # Causal 覆盖区
        (43780000, 43780500),
        (43790000, 43790200),
        (43797000, 43797294)  # 5' TSS (Start)
    ]
    strand = '-'
    tss = 43797294 
    return exons, strand, tss

# ================= 2. 数据处理工具 =================

def seq_to_one_hot(seq_str):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    one_hot = np.zeros((4, len(seq_str)), dtype=np.float32)
    for i, base in enumerate(seq_str):
        if base in mapping: one_hot[mapping[base], i] = 1.0
    return torch.tensor(one_hot)

def load_ref_sequence(chrom, center_pos, fasta_path):
    genome = pyfaidx.Fasta(fasta_path)
    start = center_pos - SEQ_LEN // 2
    end = center_pos + SEQ_LEN // 2
    try:
        seq_str = genome[chrom][start:end].seq.upper()
    except KeyError:
        chrom = chrom.replace('chr', '')
        seq_str = genome[chrom][start:end].seq.upper()
    
    pred_len = 6144 * 32
    crop = (SEQ_LEN - pred_len) // 2
    return seq_to_one_hot(seq_str).unsqueeze(0), start, start + crop, end - crop, seq_str

def mutate_sequence(ref_tensor, abs_pos, seq_start, alt_base):
    rel_pos = abs_pos - seq_start
    if rel_pos < 0 or rel_pos >= SEQ_LEN: return ref_tensor
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    mut_tensor = ref_tensor.clone()
    if alt_base in mapping:
        mut_tensor[0, :, rel_pos] = 0
        mut_tensor[0, mapping[alt_base], rel_pos] = 1.0
    return mut_tensor

# ================= 3. 高级绘图 =================

def draw_gene_model(ax, exons, strand, tss, plot_start, plot_end):
    # 1. Intron Line
    gene_start = exons[0][0]
    gene_end = exons[-1][1]
    draw_start = max(gene_start, plot_start)
    draw_end = min(gene_end, plot_end)
    ax.plot([draw_start, draw_end], [0.5, 0.5], color='#333333', linewidth=1.5, zorder=1)

    # 2. Exons
    for (es, ee) in exons:
        if ee < plot_start or es > plot_end: continue
        r_start = max(es, plot_start)
        r_width = min(ee, plot_end) - r_start
        rect = patches.Rectangle((r_start, 0.25), r_width, 0.5, 
                                 facecolor='#003366', edgecolor=None, zorder=2)
        ax.add_patch(rect)

    # 3. Strand Arrows
    arrow_step = 3000 
    arrow_marker = '>' if strand == '+' else '<'
    curr = max(gene_start, plot_start) + 1000
    while curr < min(gene_end, plot_end):
        in_exon = False
        for es, ee in exons:
            if es <= curr <= ee:
                in_exon = True; break
        if not in_exon:
            ax.text(curr, 0.5, arrow_marker, ha='center', va='center', 
                    fontsize=9, color='#333333', fontweight='bold', zorder=1)
        curr += arrow_step

    # 4. TSS
    if plot_start <= tss <= plot_end:
        ax.plot([tss, tss], [0.5, 0.9], color='black', linewidth=1.5)
        arrow_len = (plot_end - plot_start) * 0.03
        dx = arrow_len if strand == '+' else -arrow_len
        ax.arrow(tss, 0.9, dx, 0, head_width=0.15, head_length=abs(dx)*0.4, fc='k', ec='k')
        ax.text(tss, 1.1, "TSS", ha='center', fontsize=11, fontweight='bold')

def plot_final_fig4c(gene_name, tissue, chrom, 
                    pred_start, pred_end, 
                    exons, strand, tss,
                    c_pos, p_pos, 
                    ref_track, delta_c, delta_p, 
                    output_path):
    
    # 动态调整窗口
    plot_start = min(c_pos, p_pos) - 6000
    plot_end = max(c_pos, p_pos) + 6000
    
    def pos_to_bin(p): return (p - pred_start) // 32
    bin_start = max(0, pos_to_bin(plot_start))
    bin_end = min(len(ref_track), pos_to_bin(plot_end))
    
    track_x = np.linspace(plot_start, plot_end, bin_end - bin_start)
    y_ref = ref_track[bin_start:bin_end]
    y_dc = delta_c[bin_start:bin_end]
    y_dp = delta_p[bin_start:bin_end]
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True, 
                             gridspec_kw={'height_ratios': [0.5, 1, 1, 1], 'hspace': 0.1})
    
    # Track 0: Gene Model
    ax0 = axes[0]
    ax0.set_title(f"{gene_name} Locus ({chrom}) | {tissue.upper()} | Strand: {strand}", fontsize=14, fontweight='bold', pad=20)
    draw_gene_model(ax0, exons, strand, tss, plot_start, plot_end)
    
    # SNP Markers
    ax0.scatter([c_pos], [0.8], marker='*', s=300, color='#D62728', label='Causal SNP', zorder=10, clip_on=False)
    ax0.scatter([p_pos], [0.8], marker='v', s=150, color='gray', label='Proxy SNP', zorder=10, clip_on=False)
    ax0.set_ylim(0, 1.5)
    ax0.axis('off')
    ax0.legend(loc='upper right', frameon=False, ncol=2, bbox_to_anchor=(1, 1.1))

    # Guides
    for ax in axes[1:]:
        ax.axvline(c_pos, color='#D62728', linestyle='--', linewidth=1.2, alpha=0.6)
        ax.axvline(p_pos, color='gray', linestyle='--', linewidth=1.2, alpha=0.6)

    # Track 1: Ref
    ax1 = axes[1]
    ax1.fill_between(track_x, y_ref, color='#DDDDDD')
    ax1.plot(track_x, y_ref, color='#666666', linewidth=1)
    ax1.set_ylabel("Ref Expr.", fontsize=12)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Track 2: Causal
    ax2 = axes[2]
    ax2.plot(track_x, y_dc, color='#D62728', lw=1.8)
    ax2.fill_between(track_x, y_dc, 0, color='#D62728', alpha=0.25)
    ax2.set_ylabel("$\Delta$ Causal", fontsize=12, color='#D62728', fontweight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Track 3: Proxy
    ax3 = axes[3]
    ax3.plot(track_x, y_dp, color='gray', lw=1.8)
    ax3.fill_between(track_x, y_dp, 0, color='gray', alpha=0.25)
    ax3.set_ylabel("$\Delta$ Proxy", fontsize=12, color='gray')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    y_lim = max(np.max(np.abs(y_dc)), np.max(np.abs(y_dp))) * 1.15
    if y_lim == 0: y_lim = 0.1
    ax2.set_ylim(-y_lim, y_lim)
    ax3.set_ylim(-y_lim, y_lim)
    
    # X-axis
    ax3.ticklabel_format(style='plain', axis='x', useOffset=False)
    plt.xlabel(f"Genomic Position (hg38)", fontsize=13)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Figure Saved: {output_path}")

# ================= 主程序 =================

def main():
    print("🚀 Generating Hardcoded SMG9 Figure...")
    
    info = TARGET_CASE
    exons, strand, tss = get_smg9_model() 
    
    print("⏳ Loading Borzoi...")
    model = Borzoi.from_pretrained('johahi/borzoi-replicate-0').to(DEVICE).eval()
    
    center = (info['c_pos'] + info['p_pos']) // 2
    ref_tensor, inp_start, pred_start, pred_end, ref_str = load_ref_sequence(info['chrom'], center, FASTA_FILE)
    
    seq_c = mutate_sequence(ref_tensor, info['c_pos'], inp_start, info['c_alt'])
    seq_p = mutate_sequence(ref_tensor, info['p_pos'], inp_start, info['p_alt'])
    
    print("   Running Inference...")
    with torch.no_grad():
        ref_tensor = ref_tensor.to(DEVICE)
        seq_c = seq_c.to(DEVICE)
        seq_p = seq_p.to(DEVICE)
        
        idx = info['track_idx']
        p_ref = model(ref_tensor)[:, idx, :].cpu().numpy().flatten()
        p_c = model(seq_c)[:, idx, :].cpu().numpy().flatten()
        p_p = model(seq_p)[:, idx, :].cpu().numpy().flatten()

    out_name = f"{OUTPUT_DIR}/Figure4C_SMG9_Corrected.png"
    plot_final_fig4c(info['gene'], info['tissue'], info['chrom'],
                     pred_start, pred_end,
                     exons, strand, tss,
                     info['c_pos'], info['p_pos'],
                     p_ref, p_c - p_ref, p_p - p_ref,
                     out_name)

if __name__ == "__main__":
    main()
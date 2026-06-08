'''
do fig 5 for the paper, is 1*4 figure, and include top/mid/bottome group genes for stats of combinatorial effect and case study for each group. 
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import torch
import pyfaidx
from borzoi_pytorch import Borzoi

torch.backends.cudnn.enabled = False 

# ================= ⚙️ Global Config =================
BASE_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP'
INPUT_DIR = f'{BASE_DIR}/results/interaction_scan_multi'
PLOT_DIR = f'{BASE_DIR}/results/interaction_scan_multi/plots'

# Model & Data Paths
FASTA_PATH = f'{BASE_DIR}/dataset/human_genome_hg38/hg38.ml.fa'
# Template for log directory (supports different naming conventions)
LOG_DIR_TEMPLATE = f'{BASE_DIR}/results/{{tissue}}_K10_borzoi_modeltrain_res' 

SEQ_LEN = 524288
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Colors
COLOR_COMBO = '#e74c3c'  # Red for Combinatorial (Actual)
COLOR_ADD = '#3498db'    # Blue for Additive (Sum)
COLOR_SYNERGY = '#2ecc71' # Green
COLOR_REDUNDANT = '#e74c3c' # Red
COLOR_ADDITIVE = '#95a5a6' # Grey

TISSUE_MAP = {
    'blood': (550, 551),
    'brain': (10, 11),
    'liver': (22, 23),
    'muscle': (32, 33)
}

# ================= 🧬 Utils =================

def get_track_id(tissue, strand):
    if tissue not in TISSUE_MAP: return 550 
    return TISSUE_MAP[tissue][0] if strand == '+' else TISSUE_MAP[tissue][1]

def seq_to_one_hot(seq_str):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    one_hot = np.zeros((4, len(seq_str)), dtype=np.float32)
    for i, base in enumerate(seq_str):
        if base in mapping: one_hot[mapping[base], i] = 1.0
    return torch.tensor(one_hot)

def get_cage_gain(model, input_tensor, track_idx):
    with torch.no_grad():
        output = model(input_tensor.to(DEVICE))
    center = output.shape[-1] // 2
    # Sum center 40 bins (~1280bp)
    val = output[0, track_idx, center-20:center+20].sum().item()
    return val

def run_ablation_analysis(gene_name, chrom, pos, strand, tissue, log_path, model):
    try:
        log_df = pd.read_csv(log_path)
        best_step = log_df.loc[log_df['Gain'].idxmax()]
    except Exception as e:
        print(f"⚠️ Failed to read log for {gene_name}: {e}")
        return None

    top_snps = [] 
    for i in range(1, 6):
        p_col = f'Rank{i}_Pos'
        ra_col = f'Rank{i}_RefAlt' 
        if p_col in best_step and ra_col in best_step:
            snp_pos = int(best_step[p_col])
            ref_alt = best_step[ra_col]
            if isinstance(ref_alt, str) and '->' in ref_alt:
                alt = ref_alt.split('->')[1]
                top_snps.append({'pos': snp_pos, 'alt': alt})
    
    if len(top_snps) < 5:
        return None

    genome = pyfaidx.Fasta(FASTA_PATH)
    start = pos - SEQ_LEN // 2
    end = pos + SEQ_LEN // 2
    try:
        wt_seq_str = genome[f"chr{chrom}"][start:end].seq.upper()
    except:
        wt_seq_str = genome[str(chrom)][start:end].seq.upper()
        
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    track_idx = get_track_id(tissue, strand)
    
    wt_tensor = seq_to_one_hot(wt_seq_str).unsqueeze(0)
    baseline_val = get_cage_gain(model, wt_tensor, track_idx)
    
    # Calculate Single Gains (for Additive Curve)
    single_gains = []
    for snp in top_snps:
        rel_pos = snp['pos'] - start
        if 0 <= rel_pos < SEQ_LEN:
            mut_tensor = wt_tensor.clone()
            alt_idx = mapping.get(snp['alt'], 0)
            mut_tensor[0, :, rel_pos] = 0 
            mut_tensor[0, alt_idx, rel_pos] = 1.0 
            val = get_cage_gain(model, mut_tensor, track_idx)
            single_gains.append(val - baseline_val)
        else:
            single_gains.append(0)

    # Calculate Combinatorial Gains (Cumulative)
    combo_gains = []
    current_mut_tensor = wt_tensor.clone()
    for i, snp in enumerate(top_snps):
        rel_pos = snp['pos'] - start
        if 0 <= rel_pos < SEQ_LEN:
            alt_idx = mapping.get(snp['alt'], 0)
            current_mut_tensor[0, :, rel_pos] = 0
            current_mut_tensor[0, alt_idx, rel_pos] = 1.0
        val = get_cage_gain(model, current_mut_tensor, track_idx)
        combo_gains.append(val - baseline_val)

    steps = [0, 1, 2, 3, 4, 5]
    y_combo = [0] + combo_gains
    y_add = [0]
    running_sum = 0
    for g in single_gains:
        running_sum += g
        y_add.append(running_sum)
        
    return steps, y_combo, y_add

# ================= 📊 Main Plotting Logic =================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tissue', type=str, default='blood')
    parser.add_argument('--n', type=int, default=5)
    args = parser.parse_args()
    
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    csv_name = f"{args.tissue}_top{args.n}_interactions.csv"
    csv_path = os.path.join(INPUT_DIR, csv_name)
    if not os.path.exists(csv_path):
        print(f"❌ CSV not found: {csv_path}")
        return
        
    df = pd.read_csv(csv_path)

    # 1. Ensure Ratio column exists
    if 'Ratio' not in df.columns:
        if 'Combo_Gain' in df.columns and 'Single_Gains_Sum' in df.columns:
            df['Ratio'] = df['Combo_Gain'] / (df['Single_Gains_Sum'] + 1e-9)
        else:
            print("Cannot calculate Ratio.")
            return
            
    # 2. Clean Data
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['Ratio'])
    
    # 3. Sort by Ratio (Descending)
    df_sorted = df.sort_values(by='Ratio', ascending=False).reset_index(drop=True)
    
    print(f"Total valid genes: {len(df_sorted)}")

    # 4. Create Groups
    top_100 = df_sorted.head(100)
    mid_idx = len(df_sorted) // 2
    mid_100 = df_sorted.iloc[mid_idx-50 : mid_idx+50]
    bot_100 = df_sorted.tail(100)
    
    groups = [
        ('Top 100\n(Synergy)', top_100),
        ('Middle 100\n(Additive)', mid_100),
        ('Bottom 100\n(Redundancy)', bot_100)
    ]
    
    # 5. Select Representative Genes
    rep_genes = []
    if not top_100.empty: rep_genes.append(top_100.iloc[0]) 
    if not mid_100.empty: rep_genes.append(mid_100.iloc[len(mid_100)//2]) 
    if not bot_100.empty: rep_genes.append(bot_100.iloc[-1])
    
    # ================= 🎨 Plotting: 1x4 Compact Layout =================
    
    # Set canvas: Wide and Short (e.g., width=16, height=3.5)
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5), gridspec_kw={'width_ratios': [1.2, 1, 1, 1]})
    plt.subplots_adjust(wspace=0.3, left=0.05, right=0.98, bottom=0.2)
    
    # --- Plot 1: Combined Violin Plot (Distribution) ---
    print("🎨 Plotting Combined Violin Distribution...")
    ax_dist = axes[0]
    
    # Prepare data for seaborn violinplot
    plot_data = []
    for label, sub_df in groups:
        short_label = label.split('\n')[0] # "Top 100", "Middle 100", "Bottom 100"
        temp = sub_df[['Ratio']].copy()
        temp['Group'] = short_label
        plot_data.append(temp)
    
    if plot_data:
        combined_df = pd.concat(plot_data)
        # Filter extreme outliers for better visualization
        combined_df = combined_df[combined_df['Ratio'].between(0, 3.0)]
        
        sns.violinplot(data=combined_df, x='Group', y='Ratio', ax=ax_dist, 
                       palette=[COLOR_SYNERGY, COLOR_ADDITIVE, COLOR_REDUNDANT], 
                       alpha=0.7, inner="quartile")
        
        ax_dist.axhline(1.0, color='k', linestyle='--', alpha=0.5, linewidth=1)
        ax_dist.set_title("Synergy Distribution", fontweight='bold', fontsize=11)
        ax_dist.set_xlabel("")
        ax_dist.set_ylabel("Interaction Ratio")
        # Rotate x labels slightly if needed
        ax_dist.tick_params(axis='x', labelsize=9)

    # --- Plot 2, 3, 4: Case Studies (Line Charts) ---
    print("🚀 Loading Borzoi Model for Case Studies...")
    try:
        model = Borzoi.from_pretrained('johahi/borzoi-replicate-0').to(DEVICE).eval()
    except Exception as e:
        print(f"❌ Failed to load Borzoi: {e}")
        model = None

    # Load gene list metadata helper
    gene_list_path = f'{BASE_DIR}/dataset/gene_3000_borzoi_gencode_v41_hg38.csv'
    try:
        meta_df = pd.read_csv(gene_list_path)
        meta_dict = meta_df.set_index('gene_name').to_dict('index')
    except:
        meta_dict = {}

    if model:
        log_dir = LOG_DIR_TEMPLATE.format(tissue=args.tissue)
        
        for i, gene_row in enumerate(rep_genes):
            # i=0 (Top) -> axes[1], i=1 (Mid) -> axes[2], i=2 (Bot) -> axes[3]
            if i >= 3: break
            ax = axes[i+1]
            
            gene_name = gene_row['Gene']
            
            # Metadata retrieval
            if gene_name in meta_dict:
                chrom = meta_dict[gene_name]['chr']
                pos = int(meta_dict[gene_name]['pos'])
                strand = meta_dict[gene_name]['strand']
            else:
                try:
                    chrom = gene_row.get('Chr', gene_row.get('chr'))
                    pos = int(gene_row.get('Pos', gene_row.get('pos')))
                    strand = gene_row.get('Strand', gene_row.get('strand'))
                except:
                    continue
            
            # Find Log File
            log_path = os.path.join(log_dir, f"{gene_name}_optim_log.csv")
            if not os.path.exists(log_path):
                 log_path = os.path.join(log_dir, f"{gene_name}_borzoi_CAGE_optim_log.csv")
            
            if not os.path.exists(log_path):
                ax.text(0.5, 0.5, "Log Missing", ha='center')
                continue

            print(f"   Inference: {gene_name} (Ratio={gene_row['Ratio']:.2f})")
            res = run_ablation_analysis(gene_name, chrom, pos, strand, args.tissue, log_path, model)
            
            if res:
                steps, y_combo, y_add = res
                
                # Plot Lines
                ax.plot(steps, y_combo, 'o-', color=COLOR_COMBO, linewidth=2, markersize=5, label='Combinatorial')
                ax.plot(steps, y_add, 's--', color=COLOR_ADD, linewidth=2, markersize=5, label='Additive')
                ax.fill_between(steps, y_combo, y_add, color='gray', alpha=0.1)
                
                # Title & Labels
                group_name = groups[i][0].split('\n')[1].replace('(', '').replace(')', '') # Synergy/Additive/Redundancy
                ax.set_title(f"{gene_name} ({group_name})", fontweight='bold', fontsize=11)
                
                ax.set_xlabel('# Mutations', fontsize=9)
                ax.set_xticks(steps)
                
                # Only show Y-label for the first line chart (Plot 2) to save space
                if i == 0: 
                    ax.set_ylabel('Expression Gain', fontsize=9)
                
                # Legend inside plot 2
                if i == 0:
                    ax.legend(fontsize=8, loc='upper left', frameon=False)
                
                # Annotate Ratio
                final_ratio = y_combo[-1] / (y_add[-1] + 1e-9)
                color = 'green' if final_ratio > 1.1 else ('red' if final_ratio < 0.9 else 'gray')
                ax.text(0.50, 0.05, f"Ratio: {final_ratio:.2f}", 
                        transform=ax.transAxes, color=color, fontweight='bold', fontsize=9,
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    # Save to file with tissue name
    out_file = f"{PLOT_DIR}/{args.tissue}_dashboard_compact_1x4.pdf"
    plt.tight_layout()
    plt.savefig(out_file, format='pdf', bbox_inches='tight')
    print(f"\n✅ Compact plot saved to: {out_file}")

if __name__ == "__main__":
    main()
import torch
import pandas as pd
import numpy as np
import pyfaidx
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from borzoi_pytorch import Borzoi

# ================= Configuration =================
BASE_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP'
DATASET_DIR = f'{BASE_DIR}/dataset'
CANDIDATES_FILE = f'{BASE_DIR}/results/res_enrichment_gwas/Fig4C_Candidates_v2.csv'
OUTPUT_DIR = f'{BASE_DIR}/results/res_enrichment_gwas/fig4c_cases_v2'
os.makedirs(OUTPUT_DIR, exist_ok=True)

TISSUE_MAP = {
    'blood': 7531, 'brain': 7539, 'liver': 7563, 
    'heart': 7557, 'muscle': 7569, 'pancreas': 7577
}

# Borzoi Specific Constants
SEQ_LEN = 524288
BIN_SIZE = 32
OUTPUT_BINS = 6144  # Borzoi outputs 6144 bins
# The valid prediction window length in bp
PRED_LEN = OUTPUT_BINS * BIN_SIZE  # 196,608 bp
# The offset from the start of the Input Sequence to the start of the Prediction
# It crops (524288 - 196608) / 2 = 163,840 bp from each side
CROP_OFFSET = (SEQ_LEN - PRED_LEN) // 2 

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ================= Utility Functions =================

def seq_to_one_hot(seq_str):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    one_hot = np.zeros((4, len(seq_str)), dtype=np.float32)
    for i, base in enumerate(seq_str):
        if base in mapping:
            one_hot[mapping[base], i] = 1.0
    return torch.tensor(one_hot)

def load_ref_sequence(chrom, center_pos, fasta_path):
    genome = pyfaidx.Fasta(fasta_path)
    # Input Window (Huge)
    start = center_pos - SEQ_LEN // 2
    end = center_pos + SEQ_LEN // 2
    
    try:
        seq_str = genome[chrom][start:end].seq.upper()
    except KeyError:
        chrom = chrom.replace('chr', '')
        seq_str = genome[chrom][start:end].seq.upper()
        
    # Calculate the valid PREDICTION Window (Smaller)
    pred_start = start + CROP_OFFSET
    pred_end = end - CROP_OFFSET
    
    return seq_to_one_hot(seq_str).unsqueeze(0), start, pred_start, pred_end, seq_str

def mutate_sequence(ref_tensor, ref_seq_str, abs_pos, input_start_pos, ref_base, alt_base):
    # Indel Check
    if len(str(ref_base)) != 1 or len(str(alt_base)) != 1:
        print(f"      ⚠️ Indel detected ({ref_base}>{alt_base}). Skipping mutation.")
        return ref_tensor
        
    rel_pos = abs_pos - input_start_pos
    if rel_pos < 0 or rel_pos >= SEQ_LEN:
        return ref_tensor
    
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    mut_tensor = ref_tensor.clone()
    
    # Simple mutation (no complex ref check for debug simplicity)
    if alt_base in mapping:
        mut_tensor[0, :, rel_pos] = 0
        mut_tensor[0, mapping[alt_base], rel_pos] = 1.0
        
    return mut_tensor

# ================= Plotting Function =================

def plot_tracks(gene_name, tissue, chrom, 
               pred_start, pred_end,  # Prediction Window
               causal_pos, proxy_pos, 
               ref_track, delta_causal, delta_proxy, 
               output_path):
    
    # 1. Define Plot Window (Zoom in on SNPs + 5kb)
    plot_start = min(causal_pos, proxy_pos) - 5000
    plot_end = max(causal_pos, proxy_pos) + 5000
    
    # 2. Check if SNPs are inside the Valid Prediction Window
    if plot_start < pred_start or plot_end > pred_end:
        print(f"      ❌ ERROR: SNPs are outside Borzoi's valid prediction window!")
        print(f"         Pred Window: {pred_start}-{pred_end}")
        print(f"         Plot Window: {plot_start}-{plot_end}")
        print(f"         Try centering the sequence closer to the SNPs.")
        return

    # 3. Map Genomic Coord -> Output Bin Index
    # Coordinate relative to the start of the PREDICTION window
    def pos_to_bin(p): 
        rel_p = p - pred_start
        return rel_p // BIN_SIZE
    
    bin_start = max(0, pos_to_bin(plot_start))
    bin_end = min(len(ref_track), pos_to_bin(plot_end))
    
    # Debug Info
    print(f"      [Plot Info] Pred Window: {pred_start}-{pred_end}")
    print(f"      [Plot Info] SNPs: {causal_pos}, {proxy_pos}")
    print(f"      [Plot Info] Bin Range: {bin_start}-{bin_end} (Total Bins: {len(ref_track)})")
    
    if bin_start >= bin_end: 
        print("      ❌ Error: Bin start >= Bin end. Plot window invalid.")
        return
    
    # Slice Data
    # Calculate exact genomic positions for x-axis
    track_x = np.linspace(plot_start, plot_end, bin_end - bin_start)
    
    y_ref = ref_track[bin_start:bin_end]
    y_dc = delta_causal[bin_start:bin_end]
    y_dp = delta_proxy[bin_start:bin_end]
    
    # Plotting
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [0.4, 1, 1, 1]})
    
    # Track 0: Annotations
    ax0 = axes[0]
    ax0.set_title(f"{gene_name} ({chrom}) | {tissue.upper()}", fontsize=12, fontweight='bold')
    ax0.plot([plot_start, plot_end], [0.5, 0.5], 'k-', lw=1)
    ax0.scatter([causal_pos], [0.5], marker='*', s=250, color='#D62728', label='Causal', zorder=10)
    ax0.scatter([proxy_pos], [0.5], marker='v', s=100, color='gray', label='Proxy', zorder=10)
    ax0.axis('off')
    ax0.legend(loc='upper right', frameon=False, ncol=2)
    
    # Track 1: Ref
    ax1 = axes[1]
    ax1.fill_between(track_x, y_ref, color='lightgrey')
    ax1.set_ylabel("Ref Expr.", fontsize=10)
    
    # Track 2: Delta Causal
    ax2 = axes[2]
    ax2.plot(track_x, y_dc, color='#D62728', lw=1.5)
    ax2.fill_between(track_x, y_dc, 0, color='#D62728', alpha=0.2)
    ax2.set_ylabel("$\Delta$ Causal", fontsize=10, color='#D62728')
    
    # Track 3: Delta Proxy
    ax3 = axes[3]
    ax3.plot(track_x, y_dp, color='gray', lw=1.5)
    ax3.fill_between(track_x, y_dp, 0, color='gray', alpha=0.2)
    ax3.set_ylabel("$\Delta$ Proxy", fontsize=10, color='gray')
    
    # Limits
    y_max = max(np.max(np.abs(y_dc)), np.max(np.abs(y_dp))) 
    if y_max == 0: y_max = 0.1
    ax2.set_ylim(-y_max * 1.1, y_max * 1.1)
    ax3.set_ylim(-y_max * 1.1, y_max * 1.1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"      📸 Saved: {output_path}")

# ================= Main =================

def main():
    if not os.path.exists(CANDIDATES_FILE):
        print("❌ Candidates file not found!")
        return

    df = pd.read_csv(CANDIDATES_FILE)
    top_candidates = df.head(10)
    
    print(f"🚀 Processing top {len(top_candidates)} candidates...")
    
    print("⏳ Loading Borzoi Model...")
    model = Borzoi.from_pretrained('johahi/borzoi-replicate-0').to(DEVICE).eval()
    FASTA = f'{DATASET_DIR}/human_genome_hg38/hg38.ml.fa'

    for idx, row in top_candidates.iterrows():
        tissue = row['Tissue']
        track_idx = TISSUE_MAP.get(tissue)
        if not track_idx: continue
        
        gene = row['Gene']
        chrom = row['Chrom']
        c_pos, p_pos = int(row['Causal_Pos']), int(row['Proxy_Pos'])
        
        c_ref, c_alt = str(row['Causal_Ref']), str(row['Causal_Alt'])
        p_ref, p_alt = str(row['Proxy_Ref']), str(row['Proxy_Alt'])
        
        print(f"\n[{idx+1}] {gene}: Causal {c_ref}>{c_alt} ({c_pos}) vs Proxy {p_ref}>{p_alt} ({p_pos})")
        
        # Center the sequence on the midpoint of the two SNPs
        # This is critical to ensure SNPs are in the valid prediction window
        center = (c_pos + p_pos) // 2
        
        # Load and get coordinate mapping
        ref_tensor, input_start, pred_start, pred_end, ref_str = load_ref_sequence(chrom, center, FASTA)
        
        # Mutate
        seq_c = mutate_sequence(ref_tensor, ref_str, c_pos, input_start, c_ref, c_alt)
        seq_p = mutate_sequence(ref_tensor, ref_str, p_pos, input_start, p_ref, p_alt)
        
        with torch.no_grad():
            ref_tensor = ref_tensor.to(DEVICE)
            seq_c = seq_c.to(DEVICE)
            seq_p = seq_p.to(DEVICE)
            
            # Prediction: Shape [1, Tracks, 6144]
            # We take only the target tissue track
            p_ref = model(ref_tensor)[:, track_idx, :].cpu().numpy().flatten()
            p_c = model(seq_c)[:, track_idx, :].cpu().numpy().flatten()
            p_p = model(seq_p)[:, track_idx, :].cpu().numpy().flatten()
            
        plot_tracks(gene, tissue, chrom, 
                   pred_start, pred_end,  # Pass prediction window boundaries
                   c_pos, p_pos,
                   p_ref, p_c - p_ref, p_p - p_ref,
                   f"{OUTPUT_DIR}/{tissue}_{gene}.png")

if __name__ == "__main__":
    main()
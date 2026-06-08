import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr, gaussian_kde
import numpy as np
from matplotlib.lines import Line2D
import torch
import pyfaidx
from borzoi_pytorch import Borzoi
import json
import traceback

# ==========================================
# 0. 全局配置
# ==========================================
torch.backends.cudnn.enabled = False 

SEQ_LEN = 524288
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# TISSUE_MAP: (Plus_Strand_ID, Minus_Strand_ID)
TISSUE_MAP = {
    'blood': (550, 551),
    'brain': (10, 11),
    'liver': (22, 23),
    'heart': (18, 19),
    'muscle': (32, 33),
    'Pancreas': (542, 543)
}

# ==========================================
# 1. 工具函数
# ==========================================

def seq_to_one_hot(seq_str):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    one_hot = np.zeros((4, len(seq_str)), dtype=np.float32)
    for i, base in enumerate(seq_str):
        if base in mapping:
            one_hot[mapping[base], i] = 1.0
    return torch.tensor(one_hot)

def get_track_id(tissue, strand):
    if str(strand).strip() == '+': return TISSUE_MAP[tissue][0]
    elif str(strand).strip() == '-': return TISSUE_MAP[tissue][1]
    else: return TISSUE_MAP[tissue][0]

def calculate_expression_score_cage(model, input_seq, target_track_idx):
    with torch.no_grad():
        output = model(input_seq)
    output_len = output.shape[-1]
    center_bin = output_len // 2
    window_bins = 20
    start_bin = max(0, center_bin - window_bins)
    end_bin = min(output_len, center_bin + window_bins)
    total_expr = output[:, target_track_idx, start_bin:end_bin].sum()
    return total_expr.item()

def prepare_sequence_with_specific_snps(gene_name, chrom, tss, target_snps_pos, fasta_path, snp_csv_path):
    genome = pyfaidx.Fasta(fasta_path)
    start = tss - SEQ_LEN // 2
    end = tss + SEQ_LEN // 2
    
    try: ref_seq_str = genome[f"chr{chrom}"][start:end].seq.upper()
    except KeyError: ref_seq_str = genome[str(chrom)][start:end].seq.upper()
        
    if len(ref_seq_str) != SEQ_LEN: return None, None, "SeqLenMismatch"

    ref_tensor = seq_to_one_hot(ref_seq_str).unsqueeze(0)
    mut_tensor = ref_tensor.clone()

    if not target_snps_pos: return ref_tensor, mut_tensor, "NoSNPs"

    if not os.path.exists(snp_csv_path): return None, None, "SNPFileMissing"
    
    snp_df = pd.read_csv(snp_csv_path)
    if 'POS_hg38' in snp_df.columns: snp_df['pos'] = snp_df['POS_hg38'].astype(int)
    
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    applied_count = 0

    for target_pos in target_snps_pos:
        match = snp_df[snp_df['pos'] == target_pos]
        if match.empty: continue 
        alt_base = match.iloc[0]['ALT']
        rel_pos = target_pos - start
        if 0 <= rel_pos < SEQ_LEN:
            if alt_base in mapping:
                vec = torch.zeros(4)
                vec[mapping[alt_base]] = 1.0
                mut_tensor[0, :, rel_pos] = vec
                applied_count += 1
    
    return ref_tensor, mut_tensor, applied_count

def extract_log_info(file_path):
    try:
        df = pd.read_csv(file_path)
        if df.empty: return None, []
        best_row = df.loc[df['Gain'].idxmax()]
        max_gain = best_row['Gain']
        hits = []
        for i in range(1, 11):
            score_col = f"Rank{i}_Score"
            pos_col = f"Rank{i}_Pos"
            if score_col in best_row and pos_col in best_row:
                if best_row[score_col] > 0.5:
                    hits.append(int(best_row[pos_col]))
        return max_gain, hits
    except Exception: return None, []

# ==========================================
# 2. 核心处理逻辑
# ==========================================

def process_single_gene(row, args, model):
    gene_name = row['gene_name']
    chrom = row['chr']
    pos = int(row['pos'])
    strand = row['strand']
    
    # A. Borzoi Self Gain
    bor_path = os.path.join(args.borzoi_dir, f"{gene_name}_borzoi_CAGE_optim_log.csv")
    if not os.path.exists(bor_path): return None
    borzoi_self_gain, _ = extract_log_info(bor_path)
    if borzoi_self_gain is None: return None

    # B. Enformer Hits
    enf_path = os.path.join(args.enformer_dir, f"{gene_name}_enformer_optim_log.csv")
    if not os.path.exists(enf_path): return None
    _, enformer_hits = extract_log_info(enf_path)

    # C. Borzoi Cross Gain
    if not enformer_hits:
        borzoi_cross_gain = 0.0
    else:
        snp_csv = os.path.join(args.snp_dir, f"{gene_name}_snps_hg38.csv")
        ref_tensor, mut_tensor, status = prepare_sequence_with_specific_snps(
            gene_name, chrom, pos, enformer_hits, args.fasta, snp_csv
        )
        if status in ["SeqLenMismatch", "SNPFileMissing"]: return None
        
        track_idx = get_track_id(args.tissue, strand)
        # Inference
        base_score = calculate_expression_score_cage(model, ref_tensor.to(DEVICE), track_idx)
        mut_score = calculate_expression_score_cage(model, mut_tensor.to(DEVICE), track_idx)
        borzoi_cross_gain = mut_score - base_score

    return {
        'gene_name': gene_name,
        'borzoi_self_gain': borzoi_self_gain,
        'borzoi_cross_gain': borzoi_cross_gain,
        'enformer_snp_count': len(enformer_hits),
        'tissue': args.tissue
    }

# ==========================================
# 3. 主流程
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    BASE_DIR = "/home/dongbos/Combine_optim_Borzoi_SNP"
    parser.add_argument("--tissue", type=str, default="blood")
    parser.add_argument("--gene_list", type=str, default=f"{BASE_DIR}/dataset/gene_3000_borzoi_gencode_v41_hg38.csv")
    parser.add_argument("--borzoi_dir", type=str, default=f"{BASE_DIR}/results/blood_K10_borzoi_CAGE_modeltrain_res")
    parser.add_argument("--enformer_dir", type=str, default=f"{BASE_DIR}/results/blood_K10_enformer_modeltrain_CAGE_res")
    parser.add_argument("--output_dir", type=str, default=f"{BASE_DIR}/results/compare_enforemr_borzoi")
    parser.add_argument("--fasta", type=str, default=f"{BASE_DIR}/dataset/human_genome_hg38/hg38.ml.fa")
    parser.add_argument("--snp_dir", type=str, default=f"{BASE_DIR}/dataset/gene_snps_hg38")
    parser.add_argument("--force", action="store_true", help="Ignore cache and recompute")
    args = parser.parse_args()

    # Cache Setup
    CACHE_DIR = os.path.join(args.output_dir, f"cache_{args.tissue}")
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # 1. Load Genes
    print(f"📖 Loading gene list...")
    master_df = pd.read_csv(args.gene_list)
    
    # 2. Check Cache
    genes_to_compute = []
    cached_files = set(os.listdir(CACHE_DIR))
    
    if args.force:
        genes_to_compute = [row for _, row in master_df.iterrows()]
    else:
        for _, row in master_df.iterrows():
            gname = str(row['gene_name'])
            if f"{gname}.json" not in cached_files:
                genes_to_compute.append(row)
    
    print(f"🔍 Total Genes: {len(master_df)} | Cached: {len(master_df) - len(genes_to_compute)} | To Compute: {len(genes_to_compute)}")

    # 3. Compute
    if genes_to_compute:
        print("🚀 Loading Borzoi Model...")
        model = Borzoi.from_pretrained('johahi/borzoi-replicate-0').to(DEVICE).eval()
        
        for row in tqdm(genes_to_compute, desc="Processing"):
            try:
                res = process_single_gene(row, args, model)
                if res:
                    # Save Cache immediately
                    with open(os.path.join(CACHE_DIR, f"{res['gene_name']}.json"), 'w') as f:
                        json.dump(res, f)
            except Exception as e:
                print(f"Error {row['gene_name']}: {e}")
                continue

    # 4. Aggregate & Plot
    print("📊 Aggregating Results...")
    all_results = []
    for f in os.listdir(CACHE_DIR):
        if f.endswith(".json"):
            with open(os.path.join(CACHE_DIR, f), 'r') as json_file:
                all_results.append(json.load(json_file))
    
    if not all_results: return

    df = pd.DataFrame(all_results).dropna()
    x = df['borzoi_self_gain'].values
    y = df['borzoi_cross_gain'].values
    
    # ================= 🎨 Density Plotting =================
    
    # Calculate Density
    try:
        xy = np.vstack([x, y])
        z = gaussian_kde(xy)(xy)
        # Sort points by density (so dense points are plotted on top)
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]
    except:
        print("⚠️ KDE failed (singular matrix?), using constant color.")
        z = np.ones_like(x)

    plt.figure(figsize=(8, 8))
    
    # Scatter with Density Color
    sc = plt.scatter(x, y, c=z, s=25, cmap='Spectral_r', alpha=0.8, edgecolor='none')
    
    # Regression
    slope, intercept = np.polyfit(x, y, 1)
    plt.plot(x, slope * x + intercept, color='black', linestyle='-', linewidth=2, label=f'Fit (Slope={slope:.2f})')
    
    # Identity Line (y=x)
    all_vals = np.concatenate([x, y])
    min_val, max_val = np.min(all_vals), np.max(all_vals)
    padding = (max_val - min_val) * 0.05
    limit_range = [min_val - padding, max_val + padding]
    
    plt.plot(limit_range, limit_range, 'k--', alpha=0.5, label='Identity (y=x)')
    plt.xlim(limit_range)
    plt.ylim(limit_range)

    # Stats
    p_r, _ = pearsonr(x, y)
    s_r, _ = spearmanr(x, y)
    
    plt.title(f"Cross-Model Validation: Borzoi vs Enformer ({args.tissue})", fontsize=14, fontweight='bold')
    plt.xlabel("Borzoi Self-Optimized Gain", fontsize=12)
    plt.ylabel("Borzoi Gain using Enformer SNPs", fontsize=12)
    
    # Legend
    handles, _ = plt.gca().get_legend_handles_labels()
    handles.append(Line2D([], [], color='none', label=f'Pearson r={p_r:.3f}'))
    handles.append(Line2D([], [], color='none', label=f'Spearman ρ={s_r:.3f}'))
    plt.legend(handles=handles, loc='upper left', frameon=True)
    
    plt.grid(True, linestyle=':', alpha=0.4)
    plt.tight_layout()
    
    out_path = os.path.join(args.output_dir, f"cross_validation_density_{args.tissue}.png")
    plt.savefig(out_path, dpi=300)
    print(f"✅ Plot saved: {out_path}")

if __name__ == "__main__":
    main()
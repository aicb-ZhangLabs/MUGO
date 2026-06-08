import os
import pandas as pd
import numpy as np
import argparse
import pyBigWig
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ================= ⚙️ Configuration Area =================
BASE_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP'
DATASET_DIR = f'{BASE_DIR}/dataset'
SNP_POOL_DIR = f'{BASE_DIR}/dataset/gene_snps_hg38'
META_CSV_PATH = f'{BASE_DIR}/dataset/gene_3000_borzoi_gencode_v41_hg38.csv'

# Ensure correct path to BigWig
PHYLOP_BW = f'{DATASET_DIR}/PolyP_hg38/hg38.phyloP100way.bw'

# Output Directory
BASE_OUTPUT_DIR = f'{BASE_DIR}/results/ablation_borzoi_K/PolyP_hit'

# Ablation K List
K_LIST = [1, 3, 5, 10, 20, 50]

# Vote Threshold for filtering model outputs
VOTE_THRESHOLD = 0.5

# Genome-wide PhyloP thresholds (Pre-calculated)
THRESHOLDS = {
    "Top10pct": 0.9820,
    "Top5pct":  1.5410,
    "Top1pct":  3.5450
}
# ===============================================

def load_gene_metadata():
    """Load gene metadata to map Gene Name -> Chromosome"""
    print("📖 Loading Metadata...")
    if not os.path.exists(META_CSV_PATH):
        raise FileNotFoundError(f"Metadata not found: {META_CSV_PATH}")
    df = pd.read_csv(META_CSV_PATH)
    # Map Gene Name -> Chrom (e.g., '1', 'X')
    gene_to_chrom = dict(zip(df['gene_name'], df['chr']))
    return gene_to_chrom

def get_candidate_snps(gene_name):
    """Load all candidate SNPs for a gene (Background Pool)"""
    csv_path = f"{SNP_POOL_DIR}/{gene_name}_snps_hg38.csv"
    if not os.path.exists(csv_path): return []
    try:
        df = pd.read_csv(csv_path)
        # Compatibility for different column names
        if 'POS_hg38' in df.columns: return df['POS_hg38'].astype(int).tolist()
        elif 'pos' in df.columns: return df['pos'].astype(int).tolist()
    except: pass
    return []

def get_best_epoch_valid_snps(log_path, k):
    """
    1. Identify the max-gain epoch
    2. Filter SNPs with Score > 0.5
    3. Return deduplicated set of SNPs
    """
    if not os.path.exists(log_path): return set()
    try:
        df = pd.read_csv(log_path)
        if df.empty: return set()
        
        # Use the max-gain epoch
        if 'Gain' in df.columns:
            best_idx = df['Gain'].idxmax()
            row = df.iloc[best_idx]
        else:
            row = df.iloc[-1]

        valid_snps = []
        for i in range(1, k+1):
            col_pos = f"Rank{i}_Pos"
            col_score = f"Rank{i}_Score"
            
            if col_pos in row and col_score in row:
                try:
                    score = float(row[col_score])
                    if score > VOTE_THRESHOLD:
                        valid_snps.append(int(row[col_pos]))
                except:
                    continue
        return set(valid_snps)
    except: 
        return set()

def get_phylop_scores_batch(chrom, positions, bw_handle):
    """Batch retrieve PhyloP scores from BigWig"""
    scores = []
    # Format Chromosome (BigWig usually expects 'chr1', 'chrX')
    bw_chrom = f"chr{chrom}".replace('chrchr', 'chr')
    
    if bw_chrom not in bw_handle.chroms():
        return np.full(len(positions), np.nan)
        
    for pos in positions:
        try:
            # BigWig is 0-based half-open, SNP pos is 1-based
            # Query [pos-1, pos)
            val = bw_handle.values(bw_chrom, pos-1, pos)[0]
            scores.append(val if not np.isnan(val) else np.nan)
        except:
            scores.append(np.nan)
    return np.array(scores)

def plot_enrichment_chart(enrichment_data, output_dir, k):
    """Generate bar chart for enrichment across tiers"""
    tiers = ["Top10pct", "Top5pct", "Top1pct"]
    values = [enrichment_data.get(f"{t}_Enrichment", 0) for t in tiers]
    
    plt.figure(figsize=(6, 5))
    sns.set_style("whitegrid")
    
    # Create bar plot
    ax = sns.barplot(x=tiers, y=values, palette="Greens_r")
    
    plt.title(f"PhyloP Conservation Enrichment (K={k})", fontsize=12, pad=15)
    plt.ylabel("Enrichment (Fold vs Background)", fontsize=10)
    plt.xlabel("Conservation Tier", fontsize=10)
    plt.axhline(1.0, color='grey', linestyle='--', alpha=0.5) # Reference line
    
    # Add labels
    for i, v in enumerate(values):
        ax.text(i, v + 0.1, f"{v:.1f}x", ha='center', fontsize=10, fontweight='bold')
        
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"phylop_enrichment_K{k}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

def process_single_k(k_val, gene_to_chrom, bw):
    input_dir = f'{BASE_DIR}/results/multihead_MVP_res_K{k_val}'
    output_dir = f'{BASE_OUTPUT_DIR}/K{k_val}'
    
    print(f"\n🚀 Processing K={k_val} ...")
    if not os.path.exists(input_dir):
        print(f"⚠️  Input dir not found: {input_dir}, skipping.")
        return None

    os.makedirs(output_dir, exist_ok=True)
    
    log_files = glob.glob(f"{input_dir}/*_optim_log.csv")
    
    # Global counters for enrichment calculation
    global_stats = {tier: {'model_hits': 0, 'model_total': 0, 'bg_hits': 0, 'bg_total': 0} 
                    for tier in THRESHOLDS}
    
    all_model_scores = []

    for log_path in tqdm(log_files, desc=f"Scanning K={k_val}"):
        gene = os.path.basename(log_path).replace('_optim_log.csv', '')
        if gene not in gene_to_chrom: continue
        
        chrom = gene_to_chrom[gene]
        
        # 1. Get Model SNPs (Deduplicated + Filtered)
        model_pos_set = get_best_epoch_valid_snps(log_path, k_val)
        if not model_pos_set: continue
        model_pos_list = list(model_pos_set)
        
        # 2. Get Background SNPs
        bg_pos_list = get_candidate_snps(gene)
        if not bg_pos_list: continue
        
        # 3. Retrieve Scores
        model_scores = get_phylop_scores_batch(chrom, model_pos_list, bw)
        bg_scores = get_phylop_scores_batch(chrom, bg_pos_list, bw)
        
        # Filter NaNs
        model_scores = model_scores[~np.isnan(model_scores)]
        bg_scores = bg_scores[~np.isnan(bg_scores)]
        
        if len(model_scores) == 0 or len(bg_scores) == 0: continue
        
        all_model_scores.extend(model_scores)
        
        # 4. Calculate Hits per Tier
        for tier, thresh in THRESHOLDS.items():
            # Model Stats
            m_hits = np.sum(model_scores > thresh)
            global_stats[tier]['model_hits'] += m_hits
            global_stats[tier]['model_total'] += len(model_scores)
            
            # Background Stats
            b_hits = np.sum(bg_scores > thresh)
            global_stats[tier]['bg_hits'] += b_hits
            global_stats[tier]['bg_total'] += len(bg_scores)

    # === Summary Calculation ===
    if not all_model_scores: return None
    
    # Mean Score
    mean_score = np.mean(all_model_scores)
    
    summary_dict = {
        'K': k_val,
        'Mean_PhyloP_Score': mean_score
    }
    
    print(f"   📊 K={k_val} Stats:")
    print(f"      Mean Score: {mean_score:.3f}")
    
    for tier in THRESHOLDS:
        d = global_stats[tier]
        m_rate = d['model_hits'] / d['model_total'] if d['model_total'] > 0 else 0
        b_rate = d['bg_hits'] / d['bg_total'] if d['bg_total'] > 0 else 0
        enrich = m_rate / b_rate if b_rate > 0 else 0
        
        summary_dict[f'{tier}_Enrichment'] = enrich
        print(f"      {tier} Enrichment: {enrich:.2f}x (Bg Rate: {b_rate:.1%})")

    # ✅ [NEW] Save Intermediate Data immediately
    csv_path = os.path.join(output_dir, f"enrichment_stats_K{k_val}.csv")
    pd.DataFrame([summary_dict]).to_csv(csv_path, index=False)
    
    # ✅ [NEW] Generate Plot immediately
    plot_enrichment_chart(summary_dict, output_dir, k_val)

    return summary_dict

def main():
    # 1. Load Metadata
    gene_to_chrom = load_gene_metadata()
    
    if not os.path.exists(PHYLOP_BW):
        print(f"❌ BigWig file not found: {PHYLOP_BW}")
        return
        
    try:
        bw = pyBigWig.open(PHYLOP_BW)
    except Exception as e:
        print(f"❌ Error opening BigWig: {e}")
        return

    summary_list = []
    
    # 2. Iterate K
    try:
        for k in K_LIST:
            res = process_single_k(k, gene_to_chrom, bw)
            if res:
                summary_list.append(res)
    finally:
        bw.close() 
            
    # 3. Save Final Summary Table
    if summary_list:
        final_df = pd.DataFrame(summary_list)
        
        # Reorder columns
        cols = ['K', 'Mean_PhyloP_Score', 'Top10pct_Enrichment', 'Top5pct_Enrichment', 'Top1pct_Enrichment']
        final_df = final_df[cols]
        
        final_path = f"{BASE_OUTPUT_DIR}/ablation_phylop_summary.csv"
        final_df.to_csv(final_path, index=False)
        
        print("\n" + "="*60)
        print(f"🏆 PhyloP Ablation Summary Saved to: {final_path}")
        print(final_df.to_string(index=False))
        print("="*60)

if __name__ == "__main__":
    main()

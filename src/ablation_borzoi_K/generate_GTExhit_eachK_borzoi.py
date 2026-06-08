import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse

# ================= ⚙️ Configuration Area =================
BASE_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP'
META_CSV_PATH = f'{BASE_DIR}/dataset/gene_3000_borzoi_gencode_v41_hg38.csv'
SNP_POOL_DIR = f'{BASE_DIR}/dataset/gene_snps_hg38'
GTEX_BASE_DIR = f'{BASE_DIR}/dataset/GTEx_Analysis_v8_eQTL'

# Output Directory
BASE_OUTPUT_DIR = f'{BASE_DIR}/results/ablation_borzoi_K/GTEx_hit'

# Ablation K List
K_LIST = [1, 3, 5, 10, 20, 50]

# GTEx P-value Thresholds
THRESHOLDS = [1e-5, 1e-8, 1e-10, 1e-12, 1e-15]
# Main Threshold for Table 1 Summary
MAIN_THRESHOLD = 1e-5 

# Vote Threshold (Confidence Filter)
VOTE_THRESHOLD = 0.5

# GTEx File Map
GTEX_FILE_MAP = {
    'brain': 'Brain_Cortex.v8.signif_variant_gene_pairs.txt.gz',
    'blood': 'Whole_Blood.v8.signif_variant_gene_pairs.txt.gz',
    'liver': 'Liver.v8.signif_variant_gene_pairs.txt.gz',
    'heart': 'Heart_Left_Ventricle.v8.signif_variant_gene_pairs.txt.gz',
    'muscle': 'Muscle_Skeletal.v8.signif_variant_gene_pairs.txt.gz',
    'pancreas': 'Pancreas.v8.signif_variant_gene_pairs.txt.gz'
}
# ===============================================

def load_gene_metadata():
    """Load gene list and ID mapping"""
    print("📖 Loading Metadata...")
    if not os.path.exists(META_CSV_PATH):
        raise FileNotFoundError(f"Metadata not found: {META_CSV_PATH}")
    df = pd.read_csv(META_CSV_PATH)
    name_to_id = dict(zip(df['gene_name'], df['gene_ID'].apply(lambda x: x.split('.')[0])))
    return name_to_id

def load_gtex_dict(tissue, target_gene_ids):
    """Load GTEx data for specific tissue"""
    gtex_filename = GTEX_FILE_MAP.get(tissue)
    if not gtex_filename:
        raise ValueError(f"Unknown tissue: {tissue}")
    
    gtex_path = f'{GTEX_BASE_DIR}/{gtex_filename}'
    if not os.path.exists(gtex_path):
        raise FileNotFoundError(f"GTEx file not found: {gtex_path}")

    print(f"📖 Loading GTEx Data for {tissue.upper()}...")
    gtex_dict = {}
    target_set = set(target_gene_ids)
    
    try:
        reader = pd.read_csv(gtex_path, sep='\t', 
                             usecols=['variant_id', 'gene_id', 'pval_nominal'], 
                             chunksize=100000, compression='gzip')
        
        for chunk in tqdm(reader, desc="Parsing GTEx", unit="chunk"):
            chunk['clean_id'] = chunk['gene_id'].str.split('.').str[0]
            mask = chunk['clean_id'].isin(target_set)
            if not mask.any(): continue
            
            filtered = chunk[mask].copy()
            filtered['pos'] = filtered['variant_id'].apply(lambda x: int(x.split('_')[1]))
            
            for gid, group in filtered.groupby('clean_id'):
                if gid not in gtex_dict:
                    gtex_dict[gid] = []
                gtex_dict[gid].append(group[['pos', 'pval_nominal']])
        
        for gid in gtex_dict:
            gtex_dict[gid] = pd.concat(gtex_dict[gid])
            
        print(f"✅ Loaded GTEx data for {len(gtex_dict)} genes.")
        return gtex_dict
        
    except Exception as e:
        print(f"❌ Error loading GTEx: {e}")
        return {}

def get_best_epoch_valid_snps(log_path, k):
    """
    1. Find the max-gain epoch
    2. Filter Score > 0.5
    3. Return deduplicated set
    """
    if not os.path.exists(log_path): return set()
    try:
        df = pd.read_csv(log_path)
        if df.empty: return set()
        
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

def get_real_pool_size(gene_name):
    """Get total candidate SNPs for background rate"""
    csv_path = f"{SNP_POOL_DIR}/{gene_name}_snps_hg38.csv"
    if not os.path.exists(csv_path): return None
    try:
        with open(csv_path, 'rb') as f:
            return sum(1 for _ in f) - 1 
    except: return None

def process_single_k(k_val, tissue, gene_map, gtex_data):
    input_dir = f'{BASE_DIR}/results/multihead_MVP_res_K{k_val}'
    output_dir = f'{BASE_OUTPUT_DIR}/K{k_val}'
    
    print(f"\n🚀 Processing K={k_val} ...")
    if not os.path.exists(input_dir):
        print(f"⚠️  Input dir not found: {input_dir}, skipping.")
        return None

    os.makedirs(output_dir, exist_ok=True)
    
    log_files = glob.glob(f"{input_dir}/*_optim_log.csv")
    
    # Stats container
    stats = {t: {'model_hits': 0, 'model_total': 0, 'bg_hits': 0, 'bg_total': 0} for t in THRESHOLDS}
    
    for f in tqdm(log_files, desc=f"Scanning K={k_val}"):
        gene_name = os.path.basename(f).replace('_optim_log.csv', '')
        if gene_name not in gene_map: continue
        gene_id = gene_map[gene_name]
        
        unique_model_snps = get_best_epoch_valid_snps(f, k_val)
        if not unique_model_snps: continue
        
        pool_size = get_real_pool_size(gene_name)
        if not pool_size: continue
        
        if gene_id not in gtex_data:
            df_gtex = pd.DataFrame(columns=['pos', 'pval_nominal'])
        else:
            df_gtex = gtex_data[gene_id]
            
        for thresh in THRESHOLDS:
            true_hits_set = set(df_gtex[df_gtex['pval_nominal'] < thresh]['pos'])
            
            hits = len(unique_model_snps.intersection(true_hits_set))
            total_preds = len(unique_model_snps)
            
            stats[thresh]['model_hits'] += hits
            stats[thresh]['model_total'] += total_preds
            
            num_true = len(true_hits_set)
            stats[thresh]['bg_hits'] += num_true
            stats[thresh]['bg_total'] += pool_size

    # --- Save Detailed Statistics for this K ---
    # ✅ [NEW] Saving the breakdown data
    breakdown_list = []
    plot_x = []
    plot_y = []
    
    for t in THRESHOLDS:
        d = stats[t]
        m = d['model_hits'] / d['model_total'] if d['model_total'] > 0 else 0
        b = d['bg_hits'] / d['bg_total'] if d['bg_total'] > 0 else 0
        e = m / b if b > 0 else 0
        
        breakdown_list.append({
            'Threshold': t,
            'Model_Hits': d['model_hits'],
            'Model_Total': d['model_total'],
            'Model_Rate': m,
            'Background_Rate': b,
            'Enrichment': e
        })
        
        if e > 0:
            plot_x.append(str(t))
            plot_y.append(e)
            
    # Save CSV
    pd.DataFrame(breakdown_list).to_csv(f"{output_dir}/enrichment_stats_K{k_val}.csv", index=False)
    
    # --- Plotting ---
    if plot_y:
        plt.figure(figsize=(6, 5))
        plt.plot(plot_x, plot_y, marker='o', linewidth=2, color='#e74c3c')
        plt.title(f"GTEx Enrichment (Valid > 0.5, K={k_val})", fontsize=12)
        plt.xlabel("P-value Threshold")
        plt.ylabel("Enrichment (Fold)")
        plt.grid(True, alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/enrichment_curve_K{k_val}.png", dpi=300)
        plt.close()

    # Return Summary for MAIN Threshold
    main_row = next(item for item in breakdown_list if item["Threshold"] == MAIN_THRESHOLD)
    
    return {
        'K': k_val,
        'Overlap_Rate_pct': main_row['Model_Rate'] * 100,
        'Background_Rate_pct': main_row['Background_Rate'] * 100,
        'Enrichment_Factor': main_row['Enrichment']
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tissue', type=str, default='blood', help='Tissue to analyze (default: blood)')
    args = parser.parse_args()
    
    tissue = args.tissue.lower()
    
    gene_map = load_gene_metadata()
    
    all_genes = set()
    for k in K_LIST:
        d = f'{BASE_DIR}/results/multihead_MVP_res_K{k}'
        if os.path.exists(d):
            files = glob.glob(f"{d}/*_optim_log.csv")
            for f in files:
                g = os.path.basename(f).replace('_optim_log.csv', '')
                if g in gene_map: all_genes.add(gene_map[g])
    
    print(f"🔍 Found {len(all_genes)} unique genes.")
    gtex_data = load_gtex_dict(tissue, list(all_genes))
    
    summary_list = []
    for k in K_LIST:
        res = process_single_k(k, tissue, gene_map, gtex_data)
        if res:
            summary_list.append(res)
            
    if summary_list:
        df_final = pd.DataFrame(summary_list)
        final_path = f"{BASE_OUTPUT_DIR}/ablation_gtex_summary.csv"
        df_final.to_csv(final_path, index=False)
        print("\n" + "="*60)
        print(f"🏆 GTEx Ablation Summary (Final Verified) Saved to: {final_path}")
        print(df_final.to_string(index=False))
        print("="*60)

if __name__ == "__main__":
    main()

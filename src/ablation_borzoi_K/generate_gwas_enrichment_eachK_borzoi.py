import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from scipy import stats
from tqdm import tqdm
import argparse

# ================= ⚙️ Configuration Area =================
BASE_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP'
SNP_POOL_DIR = f'{BASE_DIR}/dataset/gene_snps_hg38'
GWAS_FILE = f'{BASE_DIR}/dataset/GWAS_Catelog_disease/gwas_catalog.zip'
META_CSV_PATH = f'{BASE_DIR}/dataset/gene_3000_borzoi_gencode_v41_hg38.csv'

# Output Directory
BASE_OUTPUT_DIR = f'{BASE_DIR}/results/ablation_borzoi_K/GWAS_disease_hit'

# Ablation K List
K_LIST = [1, 3, 5, 10, 20, 50]

# Thresholds
GWAS_P_THRESHOLD = 5e-8
VOTE_THRESHOLD = 0.5  # Only count SNPs with vote score > 0.5

# Disease Categories
CATEGORIES = {
    "Cancer": ["cancer", "carcinoma", "tumor", "neoplasm", "leukemia", "lymphoma", "melanoma"],
    "Neurological": ["alzheimer", "parkinson", "brain", "cognitive", "schizophrenia", "depression", "autism", "neuro", "dementia", "mental", "bipolar"],
    "Cardiovascular": ["heart", "cardio", "artery", "vascular", "blood pressure", "hypertension", "stroke", "atrial", "coronary"],
    "Immune/Autoimmune": ["immune", "rheumatoid", "lupus", "asthma", "allergy", "sclerosis", "crohn", "inflammatory", "psoriasis", "celiac"],
    "Metabolic": ["diabetes", "obesity", "bmi", "cholesterol", "lipid", "glucose", "insulin", "metabolic", "body mass"],
    "Global (All)": [] 
}
# ========================================================

def load_and_categorize_gwas(gwas_path):
    print(f"📖 Loading and categorizing GWAS Catalog...")
    try:
        df = pd.read_csv(gwas_path, sep='\t', compression='zip', low_memory=False, 
                 usecols=['CHR_ID', 'CHR_POS', 'P-VALUE', 'MAPPED_TRAIT'])
    except:
        print("⚠️ Zip read failed, trying standard read...")
        df = pd.read_csv(gwas_path, sep='\t', low_memory=False, 
                         usecols=['CHR_ID', 'CHR_POS', 'P-VALUE', 'MAPPED_TRAIT'])

    # Clean P-values
    df['P_VAL_FLOAT'] = pd.to_numeric(df['P-VALUE'], errors='coerce')
    df = df[df['P_VAL_FLOAT'] < GWAS_P_THRESHOLD].copy()
    
    # Clean Coordinates
    df['clean_chrom'] = df['CHR_ID'].astype(str).apply(lambda x: f"chr{x}" if not str(x).startswith('chr') else x)
    df['clean_pos'] = pd.to_numeric(df['CHR_POS'], errors='coerce').astype('Int64')
    df = df.dropna(subset=['clean_pos'])
    
    # Build Category Sets
    cat_sets = {k: set() for k in CATEGORIES.keys()}
    
    df['trait_lower'] = df['MAPPED_TRAIT'].astype(str).str.lower()
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Categorizing GWAS"):
        key = (row['clean_chrom'], row['clean_pos'])
        trait = row['trait_lower']
        
        # Add to Global
        cat_sets['Global (All)'].add(key)
        
        # Add to specific categories
        for cat, keywords in CATEGORIES.items():
            if cat == 'Global (All)': continue
            for kw in keywords:
                if kw in trait:
                    cat_sets[cat].add(key)
                    break 
                    
    print("\n📊 GWAS Hits per Category:")
    for cat, s in cat_sets.items():
        print(f"   - {cat}: {len(s)} unique SNPs")
        
    return cat_sets

def get_candidate_snps(gene_name):
    """Load all candidate SNPs for background calculation"""
    csv_path = f"{SNP_POOL_DIR}/{gene_name}_snps_hg38.csv"
    if not os.path.exists(csv_path): return []
    try:
        df = pd.read_csv(csv_path)
        if 'POS_hg38' in df.columns: return df['POS_hg38'].astype(int).tolist()
        elif 'pos' in df.columns: return df['pos'].astype(int).tolist()
    except: pass
    return []

def get_best_epoch_valid_snps(log_path, k):
    """
    Core logic:
    1. Identify the max-gain epoch
    2. Filter SNPs with Vote Score > 0.5
    3. Return a unique SET (deduplicated)
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
                    # 🛡️ Threshold Filter
                    if score > VOTE_THRESHOLD:
                        valid_snps.append(int(row[col_pos]))
                except:
                    continue
        
        # 🛡️ Deduplication
        return set(valid_snps)
    except: 
        return set()

def process_single_k(k_val, gene_chrom_map, gwas_cat_sets, tissue):
    input_dir = f'{BASE_DIR}/results/multihead_MVP_res_K{k_val}'
    output_dir = f'{BASE_OUTPUT_DIR}/K{k_val}'
    
    print(f"\n🚀 Processing K={k_val} ...")
    if not os.path.exists(input_dir):
        print(f"⚠️  Input dir not found: {input_dir}, skipping.")
        return None

    os.makedirs(output_dir, exist_ok=True)
    
    log_files = glob.glob(f"{input_dir}/*_optim_log.csv")
    
    # Statistics Counters
    # Structure: counts[Category] = {'model_hits': 0, ...}
    counts = {cat: {'model_hits': 0, 'model_total': 0, 'bg_hits': 0, 'bg_total': 0} for cat in CATEGORIES}
    
    for log_path in tqdm(log_files, desc=f"Scanning K={k_val}"):
        gene = os.path.basename(log_path).replace('_optim_log.csv', '')
        if gene not in gene_chrom_map: continue
        
        chrom = f"chr{gene_chrom_map[gene]}".replace('chrchr', 'chr')
        
        # A. Model Set (Unique & Filtered) & Background Set
        unique_model_snps = get_best_epoch_valid_snps(log_path, k_val)
        
        # Skip if no confident prediction found
        if not unique_model_snps: continue
        
        bg_pos = get_candidate_snps(gene)
        if not bg_pos: continue
        
        # Create full coordinate tuples (chrom, pos) for set operations
        model_set = set([(chrom, p) for p in unique_model_snps])
        bg_set = set([(chrom, p) for p in bg_pos])
        
        # B. Calculate Hits for each Category
        for cat, gwas_set in gwas_cat_sets.items():
            # Model Hits
            m_hits = len(model_set.intersection(gwas_set))
            counts[cat]['model_hits'] += m_hits
            counts[cat]['model_total'] += len(model_set) # Denominator is N_unique
            
            # Background Hits
            b_hits = len(bg_set.intersection(gwas_set))
            counts[cat]['bg_hits'] += b_hits
            counts[cat]['bg_total'] += len(bg_set)

    # --- Calculate OR and CI ---
    plot_data = []
    
    for cat in CATEGORIES:
        d = counts[cat]
        a, b = d['model_hits'], d['model_total'] - d['model_hits']
        c, d_val = d['bg_hits'], d['bg_total'] - d['bg_hits']
        
        if c == 0 or a == 0: 
            odds_ratio = 0
            p_value = 1.0
            ci_lower, ci_upper = 0, 0
        else:
            odds_ratio, p_value = stats.fisher_exact([[a, b], [c, d_val]])
            
            # 95% CI
            try:
                log_or = np.log(odds_ratio)
                se = np.sqrt(1/a + 1/b + 1/c + 1/d_val)
                ci_lower = np.exp(log_or - 1.96 * se)
                ci_upper = np.exp(log_or + 1.96 * se)
            except:
                ci_lower, ci_upper = odds_ratio, odds_ratio
        
        plot_data.append({
            'K': k_val,
            'Category': cat,
            'OR': odds_ratio,
            'Lower': ci_lower,
            'Upper': ci_upper,
            'P_val': p_value,
            'Hits': a,
            'Total_Preds': d['model_total'] # Logging total predictions
        })

    # Save CSV for this K
    df_res = pd.DataFrame(plot_data)
    df_res.to_csv(f"{output_dir}/enrichment_stats_K{k_val}.csv", index=False)
    
    # Plot Forest Plot
    plot_forest(df_res, output_dir, k_val, tissue)
    
    return df_res

def plot_forest(df_plot, output_dir, k, tissue):
    if df_plot.empty: return

    # Sort
    df_plot['SortKey'] = df_plot['Category'].apply(lambda x: 0 if 'Global' in x else 1)
    df_plot = df_plot.sort_values(by=['SortKey', 'OR'], ascending=[True, True])
    
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    y_pos = range(len(df_plot))
    colors = ['#e74c3c' if x >= 1 else '#95a5a6' for x in df_plot['OR']]
    
    plt.errorbar(df_plot['OR'], y_pos, 
                 xerr=[df_plot['OR'] - df_plot['Lower'], df_plot['Upper'] - df_plot['OR']], 
                 fmt='o', ecolor='#bdc3c7', capsize=5, markersize=8, 
                 mfc=None, mec=None) 
    
    for i, (x, c) in enumerate(zip(df_plot['OR'], colors)):
        plt.plot(x, y_pos[i], 'o', color=c, markersize=10)
    
    plt.axvline(x=1, color='black', linestyle='--', linewidth=1)
    
    plt.yticks(y_pos, df_plot['Category'], fontsize=12, fontweight='bold')
    plt.xlabel("Odds Ratio (Enrichment)", fontsize=12)
    plt.title(f"Trait-Specific Enrichment (Valid > 0.5, K={k})", fontsize=14, pad=20)
    
    # Annotate
    max_val = max(df_plot['Upper']) if not df_plot.empty else 1
    display_limit = min(max_val, 10.0) 
    
    for i, row in enumerate(df_plot.itertuples()):
        label = f"OR={row.OR:.2f} (p={row.P_val:.1e})"
        pos_x = min(row.Upper + 0.2, display_limit + 0.2)
        plt.text(pos_x, i, label, va='center', fontsize=9, color='#34495e')

    plt.xlim(0.0, display_limit * 1.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/forest_plot_K{k}.png", dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tissue', type=str, default='blood', help='Target tissue for analysis')
    args = parser.parse_args()
    
    tissue = args.tissue.lower()
    
    # 1. Prepare Data
    gwas_cat_sets = load_and_categorize_gwas(GWAS_FILE)
    meta_df = pd.read_csv(META_CSV_PATH)
    gene_chrom_map = dict(zip(meta_df['gene_name'], meta_df['chr']))
    
    all_k_results = []
    
    # 2. Iterate K
    for k in K_LIST:
        df_k = process_single_k(k, gene_chrom_map, gwas_cat_sets, tissue)
        if df_k is not None:
            all_k_results.append(df_k)
            
    # 3. Save Final Summary Table
    if all_k_results:
        final_df = pd.concat(all_k_results)
        final_path = f"{BASE_OUTPUT_DIR}/ablation_gwas_summary.csv"
        final_df.to_csv(final_path, index=False)
        
        print("\n" + "="*60)
        print(f"🏆 GWAS Ablation Summary (Corrected) Saved to: {final_path}")
        # Print a snapshot for 'Global (All)' category
        print("Snapshot (Global Enrichment):")
        print(final_df[final_df['Category'] == 'Global (All)'][['K', 'OR', 'P_val']].to_string(index=False))
        print("="*60)

if __name__ == "__main__":
    main()

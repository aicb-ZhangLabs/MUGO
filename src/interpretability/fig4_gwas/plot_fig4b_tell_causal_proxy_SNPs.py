import pandas as pd
import numpy as np
import os
import glob
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# ================= Configuration =================
BASE_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP'
CAUSAL_PROXY_DIR = f'{BASE_DIR}/dataset/GWAS_Catelog_disease/UKBB_causal_proxy'
RESULTS_DIR_BASE = f'{BASE_DIR}/results'

# Tissues to analyze (Brain excluded due to low causal count)
TARGET_TISSUES = ['blood', 'liver', 'muscle', 'pancreas', 'heart']

# Model Result Folders
TISSUE_FOLDERS = {
    'blood': 'blood_K10_borzoi_modeltrain_res',
    'liver': 'liver_K10_borzoi_modeltrain_res',
    'muscle': 'muscle_K10_borzoi_modeltrain_res',
    'pancreas': 'Pancreas_K10_borzoi_modeltrain_res',
    'heart': 'heart_K10_borzoi_modeltrain_res'
}

def load_model_top_k(tissue, k=10):
    """Load Model Top K SNPs (hg38 pos) for all genes in the tissue."""
    folder = TISSUE_FOLDERS.get(tissue)
    if not folder: return set()
    
    res_path = f"{RESULTS_DIR_BASE}/{folder}"
    files = glob.glob(f"{res_path}/*_optim_log.csv")
    
    model_snps = set()
    
    print(f"   Loading Model Results for {tissue} ({len(files)} genes)...")
    for f in tqdm(files, leave=False):
        try:
            # Note: Model outputs are hg38. We assume the input causal/proxy files are now hg38.
            # We store only 'Pos' here. Since we process gene-by-gene, 
            # and the causal/proxy comparison is typically local, this is acceptable.
            # (Strictly speaking, we should match (Chrom, Pos), but optim_log lacks Chrom column).
            
            df = pd.read_csv(f)
            if df.empty: continue
            
            # Get Best Epoch
            best_idx = df['Gain'].idxmax()
            row = df.iloc[best_idx]
            
            for i in range(1, k+1):
                col = f"Rank{i}_Pos"
                if col in row:
                    model_snps.add(int(row[col]))
        except:
            continue
            
    return model_snps

def main():
    plot_data = []
    
    print("🚀 Starting Causal vs Proxy Analysis (hg38)...")

    for tissue in TARGET_TISSUES:
        print(f"\nAnalyzing [{tissue.upper()}]...")
        
        # 1. Load LiftOver-corrected Causal/Proxy Truth Table
        # ✅ Updated filename to look for _hg38.csv
        cp_file = f"{CAUSAL_PROXY_DIR}/{tissue}_causal_proxy_hg38.csv"
        
        if not os.path.exists(cp_file):
            print(f"⚠️ File not found: {cp_file}")
            continue
            
        ukbb_df = pd.read_csv(cp_file)
        
        # 2. Load Model Top 10 Predictions
        model_set = load_model_top_k(tissue, k=10)
        
        if not model_set:
            print("❌ Model results empty.")
            continue
            
        # 3. Calculate Hit Rates
        stats = {'Causal': {'hits': 0, 'total': 0}, 'Proxy': {'hits': 0, 'total': 0}}
        
        for idx, row in ukbb_df.iterrows():
            snp_type = row['type'] # 'Causal' or 'Proxy'
            pos = int(row['pos'])
            
            stats[snp_type]['total'] += 1
            if pos in model_set:
                stats[snp_type]['hits'] += 1
        
        # 4. Aggregate
        for stype in ['Causal', 'Proxy']:
            n_hits = stats[stype]['hits']
            n_total = stats[stype]['total']
            rate = (n_hits / n_total * 100) if n_total > 0 else 0
            
            print(f"   - {stype}: {n_hits}/{n_total} ({rate:.2f}%)")
            
            plot_data.append({
                'Tissue': tissue.capitalize(),
                'SNP Type': stype,
                'Recovery Rate (%)': rate,
                'Count': f"{n_hits}/{n_total}"
            })

    # ================= Plotting =================
    if not plot_data:
        print("No data to plot.")
        return

    df = pd.DataFrame(plot_data)
    
    # Save raw plotting data
    df.to_csv(f"{BASE_DIR}/results/res_enrichment_gwas/Figure4B_data_hg38.csv", index=False)
    
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    # Colors: Causal (Red) vs Proxy (Grey)
    colors = {'Causal': '#D62728', 'Proxy': '#7F7F7F'}
    
    ax = sns.barplot(
        data=df,
        x='Tissue',
        y='Recovery Rate (%)',
        hue='SNP Type',
        palette=colors,
        edgecolor="black"
    )
    
    # Annotate Fold Change
    tissues = df['Tissue'].unique()
    for i, tissue in enumerate(tissues):
        sub = df[df['Tissue'] == tissue]
        try:
            causal_rate = sub[sub['SNP Type'] == 'Causal']['Recovery Rate (%)'].values[0]
            proxy_rate = sub[sub['SNP Type'] == 'Proxy']['Recovery Rate (%)'].values[0]
            
            # Label fold change only if proxy_rate > 0
            if proxy_rate > 0:
                fold = causal_rate / proxy_rate
                # Place text slightly above the taller bar
                y_pos = max(causal_rate, proxy_rate) + (df['Recovery Rate (%)'].max() * 0.05)
                ax.text(i, y_pos, f"{fold:.1f}x", 
                        ha='center', fontsize=12, fontweight='bold', color='black')
            elif causal_rate > 0:
                 # Infinite fold change case
                 y_pos = causal_rate + (df['Recovery Rate (%)'].max() * 0.05)
                 ax.text(i, y_pos, "Inf", ha='center', fontsize=12, fontweight='bold', color='black')

        except:
            pass

    plt.title('Discrimination Power: Causal vs. Proxy Variants (hg38 Corrected)', fontsize=16, fontweight='bold')
    plt.ylabel('Recovery Rate in Model Top 10 (%)', fontsize=14)
    plt.xlabel('', fontsize=12)
    plt.legend(title='Variant Type')
    
    out_fig = f"{BASE_DIR}/results/res_enrichment_gwas/Figure4B_Causal_vs_Proxy_hg38.png"
    plt.savefig(out_fig, dpi=300)
    print(f"\n✅ Figure 4B saved to: {out_fig}")

if __name__ == "__main__":
    main()
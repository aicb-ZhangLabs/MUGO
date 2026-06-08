import pandas as pd
import numpy as np
import os
import glob
import scipy.stats as stats
from tqdm import tqdm
import argparse

# ================= 命令行参数配置 =================
parser = argparse.ArgumentParser(description="Calculate GWAS Enrichment for specific tissue")
parser.add_argument('--tissue', type=str, required=True, 
                    choices=['brain', 'blood', 'liver', 'heart', 'muscle', 'pancreas', 'Pancreas'],
                    help="Select tissue type to analyze (e.g., brain, blood)")
# ✅ [新增] 模式选择参数
parser.add_argument('--mode', type=str, default='best', choices=['best', 'last'],
                    help="Choose 'best' to use the epoch with max Gain, or 'last' for final epoch. Default: best")

args = parser.parse_args()

# 统一转为小写处理
CURRENT_TISSUE = args.tissue.lower()
CURRENT_MODE = args.mode  # 'best' or 'last'

# ================= 路径配置 =================
BASE_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP'
SNP_POOL_DIR = f'{BASE_DIR}/dataset/gene_snps_hg38'       # Baseline SNP Pool
GWAS_FILE = f'{BASE_DIR}/dataset/GWAS_Catelog_disease/gwas_catalog.zip' 
META_CSV_PATH = f'{BASE_DIR}/dataset/gene_3000_borzoi_gencode_v41_hg38.csv'

# ✅ [核心配置] 组织 -> 文件夹名 映射表
TISSUE_CONFIG = {
    'brain': 'brain_K10_borzoi_modeltrain_res',
    'blood': 'blood_K10_borzoi_modeltrain_res',
    'liver': 'liver_K10_borzoi_modeltrain_res',
    'heart': 'heart_K10_borzoi_modeltrain_res',
    'muscle': 'muscle_K10_borzoi_modeltrain_res',
    'pancreas': 'Pancreas_K10_borzoi_modeltrain_res' 
}

folder_name = TISSUE_CONFIG[CURRENT_TISSUE]
RESULTS_DIR = f'{BASE_DIR}/results/{folder_name}'
# Output Dir 增加 mode 后缀区分
OUTPUT_DIR = f'{BASE_DIR}/results/res_enrichment_gwas/{CURRENT_TISSUE}_{CURRENT_MODE}'

print(f"🔧 Configuration for [{CURRENT_TISSUE.upper()}]:")
print(f"   🎯 Mode:          {CURRENT_MODE.upper()} EPOCH")
print(f"   📂 Model Results: {RESULTS_DIR}")
print(f"   💾 Output Dir:    {OUTPUT_DIR}")

os.makedirs(OUTPUT_DIR, exist_ok=True)
GWAS_P_THRESHOLD = 5e-8

# ==========================================
# 2. 核心处理函数
# ==========================================

def load_gwas_hits(gwas_path):
    print(f"📖 Loading GWAS Catalog from: {gwas_path}")
    try:
        df = pd.read_csv(gwas_path, sep='\t', low_memory=False, 
                         usecols=['CHR_ID', 'CHR_POS', 'P-VALUE', 'MAPPED_TRAIT', 'SNPS'])
    except:
        print("⚠️ Direct read failed, trying as ZIP...")
        try:
            df = pd.read_csv(gwas_path, sep='\t', compression='zip', low_memory=False,
                             usecols=['CHR_ID', 'CHR_POS', 'P-VALUE', 'MAPPED_TRAIT', 'SNPS'])
        except Exception as e:
            print(f"❌ Failed to read GWAS file. Error: {e}")
            raise e

    df['P_VAL_FLOAT'] = pd.to_numeric(df['P-VALUE'], errors='coerce')
    df_sig = df[df['P_VAL_FLOAT'] < GWAS_P_THRESHOLD].copy()
    
    df_sig['clean_chrom'] = df_sig['CHR_ID'].astype(str).apply(
        lambda x: f"chr{x}" if not str(x).startswith('chr') else x
    )
    
    df_sig['clean_pos'] = pd.to_numeric(df_sig['CHR_POS'], errors='coerce')
    df_sig = df_sig.dropna(subset=['clean_pos'])
    df_sig['clean_pos'] = df_sig['clean_pos'].astype(int)
    
    gwas_set = set(zip(df_sig['clean_chrom'], df_sig['clean_pos']))
    print(f"✅ Loaded {len(gwas_set)} unique significant GWAS hits (P < {GWAS_P_THRESHOLD})")
    return gwas_set, df_sig

def get_candidate_snps(gene_name):
    csv_path = f"{SNP_POOL_DIR}/{gene_name}_snps_hg38.csv"
    if not os.path.exists(csv_path): return []
    try:
        df = pd.read_csv(csv_path)
        if 'POS_hg38' in df.columns:
            return df['POS_hg38'].astype(int).tolist()
        elif 'pos' in df.columns:
            return df['pos'].astype(int).tolist()
    except:
        pass
    return []

def parse_model_top_k(log_path, mode, k=10):
    """
    ✅ [核心修改] 根据 mode 读取 Best 或 Last Epoch
    """
    if not os.path.exists(log_path): return []
    try:
        df = pd.read_csv(log_path)
        if df.empty: return []

        if mode == 'best':
            # 找到 Gain 最大的那一行索引
            best_idx = df['Gain'].idxmax()
            target_row = df.iloc[best_idx]
        else:
            # 默认取最后一行
            target_row = df.iloc[-1]
        
        snps = []
        for i in range(1, k+1):
            col = f"Rank{i}_Pos"
            if col in target_row:
                snps.append(int(target_row[col]))
        return snps
    except Exception as e:
        # print(f"Error parsing {log_path}: {e}")
        return []

# ==========================================
# 3. 主逻辑
# ==========================================

def main():
    gwas_hits_set, gwas_df = load_gwas_hits(GWAS_FILE)
    
    meta_df = pd.read_csv(META_CSV_PATH)
    gene_chrom_map = dict(zip(meta_df['gene_name'], meta_df['chr']))
    
    log_files = glob.glob(f"{RESULTS_DIR}/*_optim_log.csv")
    
    if not log_files:
        print(f"⚠️  No optimization logs found in {RESULTS_DIR}")
        return

    results = []
    
    stats_counter = {
        'model_hits': 0, 'model_total': 0,
        'bg_hits': 0,    'bg_total': 0
    }

    print(f"🚀 Analyzing {len(log_files)} genes for {CURRENT_TISSUE.upper()} (Mode: {CURRENT_MODE})...")
    
    for log_path in tqdm(log_files):
        gene = os.path.basename(log_path).replace('_optim_log.csv', '')
        if gene not in gene_chrom_map: continue
        
        chrom_raw = gene_chrom_map[gene]
        chrom = f"chr{chrom_raw}".replace('chrchr', 'chr')
        
        # A. Model Set (传入当前 Mode)
        model_pos = parse_model_top_k(log_path, mode=CURRENT_MODE, k=10)
        model_set = set([(chrom, p) for p in model_pos])
        
        # B. Background Set
        bg_pos_list = get_candidate_snps(gene)
        bg_set = set([(chrom, p) for p in bg_pos_list])
        
        if not bg_set: continue
        
        # C. Intersect
        model_hits = model_set.intersection(gwas_hits_set)
        bg_hits = bg_set.intersection(gwas_hits_set)
        
        # D. Update Stats
        stats_counter['model_hits'] += len(model_hits)
        stats_counter['model_total'] += len(model_set)
        stats_counter['bg_hits'] += len(bg_hits)
        stats_counter['bg_total'] += len(bg_set)
        
        # E. Record Hits
        if len(model_hits) > 0:
            traits = []
            for hit in model_hits:
                hit_rows = gwas_df[(gwas_df['clean_chrom'] == hit[0]) & (gwas_df['clean_pos'] == hit[1])]
                t = hit_rows['MAPPED_TRAIT'].dropna().unique().tolist()
                traits.extend(t)
            
            results.append({
                'Gene': gene,
                'Hits_Count': len(model_hits),
                'Traits': "; ".join(set(traits))
            })

    # 4. Final Stats
    a = stats_counter['model_hits']
    b = stats_counter['model_total'] - a
    c = stats_counter['bg_hits']
    d = stats_counter['bg_total'] - c
    
    print("\n" + "="*60)
    print(f"📊 {CURRENT_TISSUE.upper()} ({CURRENT_MODE.upper()}) GWAS ENRICHMENT RESULTS")
    print("="*60)
    
    model_rate = a / stats_counter['model_total'] * 100 if stats_counter['model_total'] else 0
    bg_rate = c / stats_counter['bg_total'] * 100 if stats_counter['bg_total'] else 0
    
    print(f"Model Hit Rate: {model_rate:.4f}% ({a}/{stats_counter['model_total']})")
    print(f"Random Rate:    {bg_rate:.4f}% ({c}/{stats_counter['bg_total']})")
    
    enrich_factor = model_rate / bg_rate if bg_rate > 0 else 0
    print(f"Enrichment:     {enrich_factor:.2f}x")

    if c > 0 and b > 0:
        odds_ratio, p_value = stats.fisher_exact([[a, b], [c, d]])
        print(f"\n🚀 Odds Ratio: {odds_ratio:.4f}")
        print(f"🎯 P-value:    {p_value:.4e}")
    else:
        print("\n⚠️ Cannot calc OR")

    if results:
        out_csv = f"{OUTPUT_DIR}/{CURRENT_TISSUE}_{CURRENT_MODE}_gwas_hits_K10.csv"
        pd.DataFrame(results).to_csv(out_csv, index=False)
        print(f"\n💾 Saved GWAS Hits List to: {out_csv}")
    else:
        print(f"\n⚠️ No GWAS hits found for {CURRENT_TISSUE} model results.")

if __name__ == "__main__":
    main()
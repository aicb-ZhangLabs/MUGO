import pandas as pd
import numpy as np
import os
import glob
import scipy.stats as stats
from tqdm import tqdm
import argparse

# ================= 命令行参数配置 =================
parser = argparse.ArgumentParser(description="Calculate GWAS Enrichment: Model vs Random vs TSS")
parser.add_argument('--tissue', type=str, required=True, 
                    choices=['brain', 'blood', 'liver', 'heart', 'muscle', 'pancreas', 'Pancreas'],
                    help="Select tissue type to analyze")
parser.add_argument('--mode', type=str, default='best', choices=['best', 'last'],
                    help="Choose 'best' (max Gain epoch) or 'last' epoch. Default: best")
parser.add_argument('--k', type=int, default=10, 
                    help="Top K SNPs to evaluate for Model and TSS (default: 10)")
parser.add_argument('--test', action='store_true', 
                    help="If set, only process first 100 genes for quick debugging")

args = parser.parse_args()

CURRENT_TISSUE = args.tissue.lower()
CURRENT_MODE = args.mode
TOP_K = args.k
IS_TEST_MODE = args.test

# ================= 路径配置 =================
BASE_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP'
SNP_POOL_DIR = f'{BASE_DIR}/dataset/gene_snps_hg38'       
GWAS_FILE = f'{BASE_DIR}/dataset/GWAS_Catelog_disease/gwas_catalog.zip' 
META_CSV_PATH = f'{BASE_DIR}/dataset/gene_3000_borzoi_gencode_v41_hg38.csv'

TISSUE_CONFIG = {
    'brain': 'brain_K10_borzoi_modeltrain_res',
    'blood': 'blood_K10_borzoi_modeltrain_res',
    'liver': 'liver_K10_borzoi_modeltrain_res',
    'heart': 'heart_K10_borzoi_modeltrain_res',
    'muscle': 'muscle_K10_borzoi_modeltrain_res',
    'pancreas': 'Pancreas_K10_borzoi_modeltrain_res' 
}

# ✅ [新增] 定义Trait关键词类别
CATEGORIES = {
    "Cancer": ["cancer", "carcinoma", "tumor", "neoplasm", "leukemia", "lymphoma", "melanoma"],
    "Neurological": ["alzheimer", "parkinson", "brain", "cognitive", "schizophrenia", "depression", "autism", "neuro", "dementia", "mental", "bipolar"],
    "Cardiovascular": ["heart", "cardio", "artery", "vascular", "blood pressure", "hypertension", "stroke", "atrial", "coronary"],
    "Immune/Autoimmune": ["immune", "rheumatoid", "lupus", "asthma", "allergy", "sclerosis", "crohn", "inflammatory", "psoriasis", "celiac", "blood", "platelet", "white blood", "red blood", "hematocrit", "hemoglobin", "cell count"], # 合并了部分Blood关键词到Immune
    "Metabolic": ["diabetes", "obesity", "bmi", "cholesterol", "lipid", "glucose", "insulin", "metabolic", "body mass", "liver", "triglyceride", "bilirubin"], # 合并了部分Liver关键词到Metabolic
    "Muscle": ["muscle", "grip strength", "lean body mass", "myasthenia", "sarcopenia"], # 新增Muscle类别
    "Global (All)": [] 
}

# ✅ [新增] 定义组织到Trait类别的映射
# 这里定义了每个Tissue应该关注哪些类别的疾病
TISSUE_TRAIT_MAPPING = {
    'brain': ['Neurological'],
    'blood': ['Immune/Autoimmune', 'Cardiovascular'], # Blood can relate to immune and cardio
    'liver': ['Metabolic'],
    'heart': ['Cardiovascular'],
    'muscle': ['Muscle', 'Metabolic'], # Muscle often related to metabolic traits too
    'pancreas': ['Metabolic', 'Cancer'] # Pancreas heavily linked to diabetes (metabolic)
}

folder_name = TISSUE_CONFIG.get(CURRENT_TISSUE, TISSUE_CONFIG.get('pancreas'))
RESULTS_DIR = f'{BASE_DIR}/results/{folder_name}'
OUTPUT_DIR = f'{BASE_DIR}/results/res_enrichment_gwas/{CURRENT_TISSUE}_{CURRENT_MODE}'

print(f"🔧 Configuration for [{CURRENT_TISSUE.upper()}]:")
print(f"   🎯 Mode:          {CURRENT_MODE.upper()} EPOCH")
print(f"   🔢 Top K:         {TOP_K}")
print(f"   🧪 Test Mode:     {'ON (Max 100 genes)' if IS_TEST_MODE else 'OFF'}")
print(f"   📂 Model Results: {RESULTS_DIR}")
print(f"   💾 Output Dir:    {OUTPUT_DIR}")

os.makedirs(OUTPUT_DIR, exist_ok=True)
GWAS_P_THRESHOLD = 5e-8

# ==========================================
# 2. 核心处理函数
# ==========================================

def load_and_filter_gwas_hits(gwas_path, tissue):
    print(f"📖 Loading GWAS Catalog...")
    try:
        df = pd.read_csv(gwas_path, sep='\t', compression='zip', low_memory=False,
                             usecols=['CHR_ID', 'CHR_POS', 'P-VALUE', 'MAPPED_TRAIT', 'SNPS'])
    except:
        df = pd.read_csv(gwas_path, sep='\t', low_memory=False, 
                         usecols=['CHR_ID', 'CHR_POS', 'P-VALUE', 'MAPPED_TRAIT', 'SNPS'])

    # 1. P-value 过滤
    df['P_VAL_FLOAT'] = pd.to_numeric(df['P-VALUE'], errors='coerce')
    df_sig = df[df['P_VAL_FLOAT'] < GWAS_P_THRESHOLD].copy()
    
    # 2. ✅ [核心修改] Tissue-Specific Trait 过滤
    # 获取该组织关注的疾病类别
    target_categories = TISSUE_TRAIT_MAPPING.get(tissue, [])
    
    # 收集所有相关的关键词
    keywords = []
    if target_categories:
        for cat in target_categories:
            if cat in CATEGORIES:
                keywords.extend(CATEGORIES[cat])
    
    if not keywords:
        print(f"⚠️ Warning: No specific trait keywords mapped for tissue '{tissue}'. Using ALL traits.")
    else:
        # 构建正则模式: 'diabetes|obesity|...'
        # 使用 re.escape 避免特殊字符导致正则错误，但这里关键词比较简单直接join也行
        pattern = '|'.join(keywords)
        original_count = len(df_sig)
        
        # 筛选 MAPPED_TRAIT 列包含关键词的行 (忽略大小写)
        # na=False 处理空值
        df_sig = df_sig[df_sig['MAPPED_TRAIT'].str.contains(pattern, case=False, na=False)]
        
        filtered_count = len(df_sig)
        print(f"🔍 Filtered GWAS Traits for [{tissue}]:")
        print(f"   Categories: {target_categories}")
        print(f"   Keywords (top 5): {keywords[:5]} ...")
        print(f"   Kept {filtered_count}/{original_count} hits ({filtered_count/original_count*100:.2f}%)")

    # 3. 坐标清洗
    df_sig['clean_chrom'] = df_sig['CHR_ID'].astype(str).apply(
        lambda x: f"chr{x}" if not str(x).startswith('chr') else x
    )
    df_sig['clean_pos'] = pd.to_numeric(df_sig['CHR_POS'], errors='coerce')
    df_sig = df_sig.dropna(subset=['clean_pos'])
    df_sig['clean_pos'] = df_sig['clean_pos'].astype(int)
    
    # 转为集合
    gwas_set = set(zip(df_sig['clean_chrom'], df_sig['clean_pos']))
    print(f"✅ Final Loaded: {len(gwas_set)} unique significant SNPs for {tissue}")
    return gwas_set, df_sig

def get_candidate_snps_df(gene_name):
    csv_path = f"{SNP_POOL_DIR}/{gene_name}_snps_hg38.csv"
    if not os.path.exists(csv_path): return None
    try:
        df = pd.read_csv(csv_path)
        if 'POS_hg38' in df.columns:
            df.rename(columns={'POS_hg38': 'pos'}, inplace=True)
        elif 'POS' in df.columns:
             df.rename(columns={'POS': 'pos'}, inplace=True)
        return df
    except:
        return None

def parse_model_top_k(log_path, mode, k=10):
    if not os.path.exists(log_path): return []
    try:
        df = pd.read_csv(log_path)
        if df.empty: return []

        if mode == 'best':
            best_idx = df['Gain'].idxmax()
            target_row = df.iloc[best_idx]
        else:
            target_row = df.iloc[-1]
        
        snps = []
        for i in range(1, k+1):
            col = f"Rank{i}_Pos"
            if col in target_row:
                snps.append(int(target_row[col]))
        return snps
    except:
        return []

def get_tss_top_k(snp_df, tss_pos, k=10):
    if snp_df is None or snp_df.empty: return []
    snp_df['dist_to_tss'] = (snp_df['pos'] - tss_pos).abs()
    top_k_df = snp_df.nsmallest(k, 'dist_to_tss')
    return top_k_df['pos'].astype(int).tolist()

# ==========================================
# 3. 主逻辑
# ==========================================

def main():
    # 1. Load GWAS (With Tissue-Specific Filtering!)
    gwas_hits_set, gwas_df = load_and_filter_gwas_hits(GWAS_FILE, CURRENT_TISSUE)
    
    # 2. Load Metadata
    print(f"📖 Loading Gene Metadata from {META_CSV_PATH}...")
    meta_df = pd.read_csv(META_CSV_PATH)
    col_map = {c.lower(): c for c in meta_df.columns}
    
    col_tss_explicit = col_map.get('tss')
    col_pos_generic  = col_map.get('pos') or col_map.get('position')
    col_start        = col_map.get('start') or col_map.get('gene_start')
    
    target_col = col_tss_explicit or col_pos_generic or col_start
    if not target_col:
        raise ValueError("Cannot find TSS column")

    gene_info_map = {}
    for _, row in meta_df.iterrows():
        gname = row['gene_name']
        chrom = row['chr']
        gene_info_map[gname] = {'chr': chrom, 'tss': int(row[target_col])}

    log_files = glob.glob(f"{RESULTS_DIR}/*_optim_log.csv")
    
    if IS_TEST_MODE:
        log_files = log_files[:100]
        print("⚠️ TEST MODE: Processing only first 100 genes.")

    stats_counter = {
        'model_hits': 0, 'model_total': 0,
        'bg_hits': 0,    'bg_total': 0,
        'tss_hits': 0,   'tss_total': 0 
    }
    
    results_detail = []

    print(f"🚀 Analyzing...")
    for log_path in tqdm(log_files):
        gene = os.path.basename(log_path).replace('_optim_log.csv', '')
        if gene not in gene_info_map: continue
        
        info = gene_info_map[gene]
        chrom = f"chr{info['chr']}".replace('chrchr', 'chr')
        tss_pos = info['tss']
        
        snp_df = get_candidate_snps_df(gene)
        if snp_df is None or snp_df.empty: continue
        
        # A. Background
        bg_pos_list = snp_df['pos'].tolist()
        bg_set = set([(chrom, p) for p in bg_pos_list])
        
        # B. Model
        model_pos = parse_model_top_k(log_path, mode=CURRENT_MODE, k=TOP_K)
        model_set = set([(chrom, p) for p in model_pos])
        
        # C. TSS
        tss_pos_list = get_tss_top_k(snp_df, tss_pos, k=TOP_K)
        tss_set = set([(chrom, p) for p in tss_pos_list])

        model_hits = model_set.intersection(gwas_hits_set)
        bg_hits = bg_set.intersection(gwas_hits_set)
        tss_hits = tss_set.intersection(gwas_hits_set)
        
        stats_counter['model_hits'] += len(model_hits)
        stats_counter['model_total'] += len(model_set)
        stats_counter['bg_hits'] += len(bg_hits)
        stats_counter['bg_total'] += len(bg_set)
        stats_counter['tss_hits'] += len(tss_hits)
        stats_counter['tss_total'] += len(tss_set)
        
        if len(model_hits) > 0:
            traits = []
            for hit in model_hits:
                hit_rows = gwas_df[(gwas_df['clean_chrom'] == hit[0]) & (gwas_df['clean_pos'] == hit[1])]
                t = hit_rows['MAPPED_TRAIT'].dropna().unique().tolist()
                traits.extend(t)
            results_detail.append({
                'Gene': gene,
                'Model_Hits': len(model_hits),
                'Traits': "; ".join(set(traits))
            })

    # Output
    def calc_rate(hits, total): return (hits / total * 100) if total > 0 else 0
    model_rate = calc_rate(stats_counter['model_hits'], stats_counter['model_total'])
    tss_rate = calc_rate(stats_counter['tss_hits'], stats_counter['tss_total'])
    bg_rate = calc_rate(stats_counter['bg_hits'], stats_counter['bg_total'])
    
    enrich_vs_random = model_rate / bg_rate if bg_rate > 0 else 0
    enrich_vs_tss = model_rate / tss_rate if tss_rate > 0 else 0

    p_value = 1.0
    if stats_counter['model_total'] > 0:
        a = stats_counter['model_hits']
        b = stats_counter['model_total'] - a
        c = stats_counter['tss_hits']
        d = stats_counter['tss_total'] - c
        odds, p_value = stats.fisher_exact([[a, b], [c, d]])

    print("\n" + "="*60)
    print(f"📊 {CURRENT_TISSUE.upper()} (Trait-Specific) GWAS BENCHMARK (Top {TOP_K})")
    print("="*60)
    print(f"1. Random: {bg_rate:.4f}%")
    print(f"2. TSS:    {tss_rate:.4f}%")
    print(f"3. Model:  {model_rate:.4f}%")
    print(f"🚀 Enrichment vs TSS: {enrich_vs_tss:.2f}x")
    
    summary_data = {
        'Tissue': CURRENT_TISSUE,
        'Top_K': TOP_K,
        'Total_Genes': len(log_files),
        'Background_Rate': bg_rate,
        'TSS_Rate': tss_rate,
        'Model_Rate': model_rate,
        'Enrichment_vs_Random': enrich_vs_random,
        'Enrichment_vs_TSS': enrich_vs_tss,
        'P_Value_Model_vs_TSS': p_value,
        'Model_Hits': stats_counter['model_hits'],
        'TSS_Hits': stats_counter['tss_hits'],
        'Background_Hits': stats_counter['bg_hits']
    }
    
    summary_csv = f"{OUTPUT_DIR}/{CURRENT_TISSUE}_{CURRENT_MODE}_K{TOP_K}_summary_stats.csv"
    pd.DataFrame([summary_data]).to_csv(summary_csv, index=False)
    print(f"💾 Saved summary stats to: {summary_csv}")

    # Save Detailed Hits
    if results_detail:
        detail_csv = f"{OUTPUT_DIR}/{CURRENT_TISSUE}_{CURRENT_MODE}_K{TOP_K}_hits_detail.csv"
        pd.DataFrame(results_detail).to_csv(detail_csv, index=False)
        print(f"💾 Saved detailed hits to: {detail_csv}")

if __name__ == "__main__":
    main()
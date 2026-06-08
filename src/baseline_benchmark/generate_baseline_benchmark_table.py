'''
use to generate final baseline benchmark table for paper. 
现在用top10的SNPs去先算gain和enrichment再生成表格，然后SNP的gain现在似乎只用了top1的SNP，之后看看要不要改一下。
'''
import pandas as pd
import numpy as np
import os
import glob
from scipy import stats
from tqdm import tqdm
import argparse

# ================= ⚙️ 配置区域 =================
BASE_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP'
BASELINE_ROOT = f'{BASE_DIR}/results/baseline_benchmark'
SNP_POOL_DIR = f'{BASE_DIR}/dataset/gene_snps_hg38'

# 数据库路径
GWAS_FILE = f'{BASE_DIR}/dataset/GWAS_Catelog_disease/gwas_catalog.zip'
GTEX_BASE_DIR = f'{BASE_DIR}/dataset/GTEx_Analysis_v8_eQTL'
META_CSV_PATH = f'{BASE_DIR}/dataset/gene_3000_borzoi_gencode_v41_hg38.csv'

# 阈值
GWAS_P_THRESHOLD = 5e-8
GTEX_P_THRESHOLD = 1e-5
TOP_K = 10  # 每个方法取前 10 个 SNP 进行评估

# Baseline 方法配置 (对应文件夹名和列名)
METHODS = {
    'CADD': {
        'path': 'CADD/raw_res', 
        'score_col': 'CADD_PHRED', 'ascending': False, 'gain_col': None
    },
    'FunSeq2': {
        'path': 'FunSeq2/raw_res', 
        'score_col': 'FunSeq_Score', 'ascending': False, 'gain_col': None
    },
    'Feature_Ablation': {
        'path': 'Feature_Ablation/raw_res', 
        'score_col': 'Impact_Score', 'ascending': False, 'gain_col': None # Impact Score 本身代表重要性
    },
    'Greedy_ISM': {
        'path': 'Greedy_ISM/raw_res', 
        'score_col': 'Gain', 'ascending': False, 'gain_col': 'Gain'
    },
    'Random_Search': {
        'path': 'Random_Search/raw_res', 
        'score_col': 'Best_Gain', 'ascending': False, 'gain_col': 'Best_Gain'
    },
    'Saliency_Map': {
        'path': 'Saliency_Map/raw_res', 
        'score_col': 'Saliency_Score', 'ascending': False, 'gain_col': None
    }
}

# GTEx 文件映射
GTEX_FILE_MAP = {
    'brain': 'Brain_Cortex.v8.signif_variant_gene_pairs.txt.gz',
    'blood': 'Whole_Blood.v8.signif_variant_gene_pairs.txt.gz',
    'liver': 'Liver.v8.signif_variant_gene_pairs.txt.gz',
    'heart': 'Heart_Left_Ventricle.v8.signif_variant_gene_pairs.txt.gz',
    'muscle': 'Muscle_Skeletal.v8.signif_variant_gene_pairs.txt.gz',
    'pancreas': 'Pancreas.v8.signif_variant_gene_pairs.txt.gz'
}

# ===============================================

def load_resources(tissue):
    print("📖 Loading Resources...")
    
    # 1. Metadata
    meta_df = pd.read_csv(META_CSV_PATH)
    gene_map = dict(zip(meta_df['gene_name'], meta_df['chr'])) # Name -> Chrom
    id_map = dict(zip(meta_df['gene_name'], meta_df['gene_ID'].apply(lambda x: x.split('.')[0]))) # Name -> ID
    
    # 2. GWAS (Global All Categories)
    print("   Loading GWAS...")
    try:
        gwas_df = pd.read_csv(GWAS_FILE, sep='\t', compression='zip', low_memory=False, 
                              usecols=['CHR_ID', 'CHR_POS', 'P-VALUE'])
    except:
        gwas_df = pd.read_csv(GWAS_FILE, sep='\t', low_memory=False, 
                              usecols=['CHR_ID', 'CHR_POS', 'P-VALUE'])
        
    gwas_df['P_VAL_FLOAT'] = pd.to_numeric(gwas_df['P-VALUE'], errors='coerce')
    gwas_df = gwas_df[gwas_df['P_VAL_FLOAT'] < GWAS_P_THRESHOLD]
    
    gwas_set = set()
    for _, row in gwas_df.iterrows():
        chrom = str(row['CHR_ID'])
        if not chrom.startswith('chr'): chrom = f"chr{chrom}"
        try:
            pos = int(row['CHR_POS'])
            gwas_set.add((chrom, pos))
        except: continue
        
    # 3. GTEx
    print(f"   Loading GTEx for {tissue}...")
    gtex_path = f'{GTEX_BASE_DIR}/{GTEX_FILE_MAP[tissue]}'
    gtex_dict = {} # GeneID -> Set of Positions
    
    reader = pd.read_csv(gtex_path, sep='\t', usecols=['variant_id', 'gene_id', 'pval_nominal'],
                         chunksize=100000, compression='gzip')
    
    for chunk in reader:
        chunk['clean_id'] = chunk['gene_id'].str.split('.').str[0]
        # 只保留显著的
        sig = chunk[chunk['pval_nominal'] < GTEX_P_THRESHOLD].copy()
        sig['pos'] = sig['variant_id'].apply(lambda x: int(x.split('_')[1]))
        
        for gid, grp in sig.groupby('clean_id'):
            if gid not in gtex_dict: gtex_dict[gid] = set()
            gtex_dict[gid].update(grp['pos'].tolist())
            
    print("✅ Resources Loaded.")
    return gene_map, id_map, gwas_set, gtex_dict

def get_real_pool_size(gene_name):
    csv_path = f"{SNP_POOL_DIR}/{gene_name}_snps_hg38.csv"
    if not os.path.exists(csv_path): return None
    try:
        with open(csv_path, 'rb') as f: return sum(1 for _ in f) - 1 
    except: return None

def get_candidate_snps(gene_name):
    csv_path = f"{SNP_POOL_DIR}/{gene_name}_snps_hg38.csv"
    if not os.path.exists(csv_path): return []
    try:
        df = pd.read_csv(csv_path)
        if 'POS_hg38' in df.columns: return df['POS_hg38'].astype(int).tolist()
        elif 'pos' in df.columns: return df['pos'].astype(int).tolist()
    except: pass
    return []

def evaluate_method(method_name, tissue, resources):
    gene_map, id_map, gwas_set, gtex_dict = resources
    config = METHODS[method_name]
    
    input_dir = f"{BASELINE_ROOT}/{config['path']}/{tissue}"
    if not os.path.exists(input_dir):
        print(f"⚠️  Directory not found for {method_name}: {input_dir}")
        return None

    print(f"🚀 Evaluating {method_name}...")
    
    # ✅ [修改] 变量名改为 counts，避免与 scipy.stats 冲突
    counts = {
        'model_gwas_hits': 0, 'model_gtex_hits': 0, 'model_total': 0,
        'bg_gwas_hits': 0, 'bg_gtex_hits': 0, 'bg_total': 0,
        'gain_sum': 0, 'gain_count': 0
    }
    
    files = glob.glob(f"{input_dir}/*.csv")
    
    for f in tqdm(files, desc=method_name):
        gene_name = os.path.basename(f).split('_')[0] 
        
        if gene_name not in gene_map: continue
        chrom = f"chr{gene_map[gene_name]}".replace('chrchr', 'chr')
        gene_id = id_map.get(gene_name)
        
        # 1. 读取并排序取 Top K
        try:
            df = pd.read_csv(f)
            df = df.sort_values(by=config['score_col'], ascending=config['ascending']).head(TOP_K)
            
            if 'Pos' in df.columns: pos_col = 'Pos'
            elif 'pos' in df.columns: pos_col = 'pos'
            else: continue
            
            top_snps = set(df[pos_col].astype(int).tolist()) # 去重
            
            if config['gain_col'] and config['gain_col'] in df.columns:
                top_gain = df.iloc[0][config['gain_col']]
                counts['gain_sum'] += top_gain  # ✅ 使用 counts
                counts['gain_count'] += 1       # ✅ 使用 counts
                
        except Exception as e:
            continue
            
        # 2. 背景 SNP
        bg_snps = get_candidate_snps(gene_name)
        if not bg_snps: continue
        bg_set = set(bg_snps)
        
        # 3. GWAS Hit Check
        for p in top_snps:
            if (chrom, p) in gwas_set: counts['model_gwas_hits'] += 1 # ✅ 使用 counts
        counts['model_total'] += len(top_snps)
        
        for p in bg_set:
            if (chrom, p) in gwas_set: counts['bg_gwas_hits'] += 1    # ✅ 使用 counts
        counts['bg_total'] += len(bg_set)
        
        # 4. GTEx Hit Check
        if gene_id in gtex_dict:
            true_eqtls = gtex_dict[gene_id]
            counts['model_gtex_hits'] += len(top_snps.intersection(true_eqtls)) # ✅ 使用 counts
            counts['bg_gtex_hits'] += len(bg_set.intersection(true_eqtls))      # ✅ 使用 counts

    # === Calculate Metrics ===
    res = {'Method': method_name}
    
    # 1. Gain
    if counts['gain_count'] > 0:
        res['Mean_Gain'] = counts['gain_sum'] / counts['gain_count'] # ✅ 使用 counts
    else:
        res['Mean_Gain'] = np.nan
        
    # 2. GWAS Enrichment (OR)
    a = counts['model_gwas_hits'] # ✅ 使用 counts
    b = counts['model_total'] - a
    c = counts['bg_gwas_hits']
    d = counts['bg_total'] - c
    
    if c > 0 and b > 0:
        # ✅ 现在 stats 指向 scipy.stats，不会报错了
        odds_ratio, _ = stats.fisher_exact([[a, b], [c, d]]) 
        res['GWAS_OR'] = odds_ratio
    else:
        res['GWAS_OR'] = 0
        
    # 3. GTEx Enrichment (Fold)
    m_rate = counts['model_gtex_hits'] / counts['model_total'] if counts['model_total'] > 0 else 0
    b_rate = counts['bg_gtex_hits'] / counts['bg_total'] if counts['bg_total'] > 0 else 0
    
    res['GTEx_Enrichment'] = m_rate / b_rate if b_rate > 0 else 0
    res['GTEx_Overlap_Pct'] = m_rate * 100
    
    return res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tissue', type=str, default='blood', help='Target tissue')
    args = parser.parse_args()
    
    # 1. 加载资源
    resources = load_resources(args.tissue)
    
    all_res = []
    
    # 2. 遍历所有方法
    for method in METHODS.keys():
        res = evaluate_method(method, args.tissue, resources)
        if res:
            all_res.append(res)
            
    # 3. 生成表格
    df = pd.DataFrame(all_res)
    
    # 格式化
    df['Mean_Gain'] = df['Mean_Gain'].map('{:.2f}'.format)
    df['GWAS_OR'] = df['GWAS_OR'].map('{:.2f}'.format)
    df['GTEx_Enrichment'] = df['GTEx_Enrichment'].map('{:.2f}x'.format)
    df['GTEx_Overlap_Pct'] = df['GTEx_Overlap_Pct'].map('{:.1f}%'.format)
    
    # 排序 (按 GWAS 还是 GTEx 看你需求，这里默认按 GWAS)
    df = df.sort_values('GWAS_OR', ascending=False)
    
    out_path = f'{BASE_DIR}/results/baseline_benchmark/final_benchmark_table_{args.tissue}.csv'
    df.to_csv(out_path, index=False)
    
    print("\n" + "="*60)
    print(f"🏆 Final Benchmark Table ({args.tissue}):")
    print(df.to_string(index=False))
    print(f"\n💾 Saved to: {out_path}")
    print("="*60)

    # 4. 生成 LaTeX
    tex_path = out_path.replace('.csv', '.tex')
    with open(tex_path, 'w') as f:
        f.write(df.to_latex(index=False))
    print(f"💾 LaTeX saved to: {tex_path}")

if __name__ == "__main__":
    main()
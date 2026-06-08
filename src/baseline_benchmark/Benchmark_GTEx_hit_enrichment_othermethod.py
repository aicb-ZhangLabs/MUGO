import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse
import json

# ================= 命令行参数配置 =================
parser = argparse.ArgumentParser(description="Calculate GTEx Enrichment Benchmark (Fixed Top-K) with Caching & Barplot")
parser.add_argument('--tissue', type=str, default='blood', 
                    choices=['brain', 'blood', 'liver', 'heart', 'muscle', 'pancreas'],
                    help="Select tissue type to analyze")
parser.add_argument('--k', type=int, default=5, help="Number of Top variants to evaluate (default: 5)")
parser.add_argument('--force', action='store_true', help="Force re-calculation (ignore cache)")
args = parser.parse_args()

CURRENT_TISSUE = args.tissue.lower()
TOP_K = args.k

# ================= 路径配置 =================
BASE_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP'
META_CSV_PATH = f'{BASE_DIR}/dataset/gene_3000_borzoi_gencode_v41_hg38.csv'
SNP_POOL_DIR = f'{BASE_DIR}/dataset/gene_snps_hg38'
GTEX_BASE_DIR = f'{BASE_DIR}/dataset/GTEx_Analysis_v8_eQTL'
RESULTS_ROOT = f'{BASE_DIR}/results'
OUTPUT_DIR = f'{BASE_DIR}/results/baseline_benchmark'
CACHE_DIR = f'{OUTPUT_DIR}/cache/{CURRENT_TISSUE}_k{TOP_K}'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# 🔥 [修改点 1] 更新 Saliency 路径 (增加 RNA 文件夹) & 包含 CADD
METHOD_DIRS = {
    'Borzoi': f'{RESULTS_ROOT}/{CURRENT_TISSUE}_K10_borzoi_modeltrain_res',
    # Saliency 路径更新：.../raw_res/RNA/{tissue}
    'Saliency': f'{RESULTS_ROOT}/baseline_benchmark/Saliency_Map/raw_res/RNA/{CURRENT_TISSUE}',
    'FunSeq': f'{RESULTS_ROOT}/baseline_benchmark/FunSeq2/raw_res/{CURRENT_TISSUE}',
    'CADD': f'{RESULTS_ROOT}/baseline_benchmark/CADD/raw_res/{CURRENT_TISSUE}'
}

TISSUE_GTEX_MAP = {
    'brain': 'Brain_Cortex.v8.signif_variant_gene_pairs.txt.gz',
    'blood': 'Whole_Blood.v8.signif_variant_gene_pairs.txt.gz',
    'liver': 'Liver.v8.signif_variant_gene_pairs.txt.gz',
    'heart': 'Heart_Left_Ventricle.v8.signif_variant_gene_pairs.txt.gz',
    'muscle': 'Muscle_Skeletal.v8.signif_variant_gene_pairs.txt.gz',
    'pancreas': 'Pancreas.v8.signif_variant_gene_pairs.txt.gz'
}
GTEX_DATA_PATH = f'{GTEX_BASE_DIR}/{TISSUE_GTEX_MAP[CURRENT_TISSUE]}'

THRESHOLDS = [1e-3, 1e-4, 1e-5, 1e-8, 1e-10, 1e-12, 1e-15, 1e-20]

# 🔥 [修改点 2] 包含所有 4 种方法
METHODS_LIST = ['Borzoi', 'Saliency', 'FunSeq', 'CADD']

# ================= 工具函数 =================

def load_gene_metadata():
    df = pd.read_csv(META_CSV_PATH)
    name_to_id = dict(zip(df['gene_name'], df['gene_ID'].apply(lambda x: x.split('.')[0])))
    return name_to_id

def get_real_pool_size(gene_name):
    csv_path = f"{SNP_POOL_DIR}/{gene_name}_snps_hg38.csv"
    if not os.path.exists(csv_path): return set()
    try:
        df = pd.read_csv(csv_path)
        return set(df['POS_hg38'].astype(int))
    except: return set()

def load_gtex_dict(target_gene_ids):
    if not os.path.exists(GTEX_DATA_PATH): return {}
    print(f"📖 Loading GTEx Data for {len(target_gene_ids)} genes...")
    gtex_dict = {}
    target_set = set(target_gene_ids)
    try:
        reader = pd.read_csv(GTEX_DATA_PATH, sep='\t', usecols=['variant_id', 'gene_id', 'pval_nominal'], chunksize=100000, compression='gzip')
        for chunk in reader:
            chunk['clean_id'] = chunk['gene_id'].str.split('.').str[0]
            mask = chunk['clean_id'].isin(target_set)
            if not mask.any(): continue
            filtered = chunk[mask].copy()
            filtered['pos'] = filtered['variant_id'].apply(lambda x: int(x.split('_')[1]))
            for gid, group in filtered.groupby('clean_id'):
                if gid not in gtex_dict: gtex_dict[gid] = []
                gtex_dict[gid].append(group[['pos', 'pval_nominal']])
        for gid in gtex_dict: gtex_dict[gid] = pd.concat(gtex_dict[gid])
        return gtex_dict
    except Exception as e:
        print(f"Error loading GTEx: {e}")
        return {}

def get_method_top_k(gene, method, k):
    """获取指定方法的 Top K SNP Pos"""
    if method == 'Borzoi':
        path = f"{METHOD_DIRS['Borzoi']}/{gene}_optim_log.csv"
        if not os.path.exists(path): return []
        try:
            df = pd.read_csv(path)
            best_row = df.loc[df['Gain'].idxmax()]
            snps = []
            for i in range(1, k+1):
                pos_col = f"Rank{i}_Pos"
                if pos_col in best_row and pd.notna(best_row[pos_col]):
                    snps.append(int(best_row[pos_col]))
            return snps
        except: return []
    else:
        # 🔥 [修改点 3] 路径映射与文件名逻辑
        path_map = {
            'Saliency': (f"{METHOD_DIRS['Saliency']}/{gene}_saliency.csv", 'Saliency_Score'),
            'FunSeq': (f"{METHOD_DIRS['FunSeq']}/{gene}_funseq.csv", 'FunSeq_Score'),
            'CADD': (f"{METHOD_DIRS['CADD']}/{gene}_cadd.csv", 'CADD_PHRED')
        }
        
        if method not in path_map: return []
        path, col = path_map[method]
        
        if not os.path.exists(path): return []
        try:
            df = pd.read_csv(path)
            df.columns = df.columns.str.strip()
            
            # 兼容列名大小写或变体
            if 'Pos' not in df.columns: 
                if 'POS_hg38' in df.columns: df.rename(columns={'POS_hg38': 'Pos'}, inplace=True)
                elif 'pos' in df.columns: df.rename(columns={'pos': 'Pos'}, inplace=True)
                else: return []
                
            if col not in df.columns: return []
            
            # 排序取 Top K (所有分数都是越高越好)
            return df.sort_values(by=col, ascending=False).head(k)['Pos'].astype(int).tolist()
        except: return []

# ================= 核心逻辑 =================

def process_gene(gene, gene_id, gtex_df, pool_snps):
    """处理单个基因"""
    gene_stats = {}
    
    selected_map = {m: get_method_top_k(gene, m, TOP_K) for m in METHODS_LIST}
    
    for thresh in THRESHOLDS:
        t_str = str(thresh)
        gene_stats[t_str] = {}
        
        if gtex_df is not None and not gtex_df.empty:
            true_hits_set = set(gtex_df[gtex_df['pval_nominal'] < thresh]['pos'])
        else:
            true_hits_set = set()
            
        n_true_in_pool = len(pool_snps.intersection(true_hits_set))
        gene_stats[t_str]['Random'] = {
            'hits': n_true_in_pool,
            'total': len(pool_snps)
        }
        
        for m in METHODS_LIST:
            candidates = selected_map[m]
            if not candidates:
                gene_stats[t_str][m] = {'hits': 0, 'total': 0}
                continue
            
            n_hits = sum(1 for p in candidates if p in true_hits_set)
            gene_stats[t_str][m] = {
                'hits': n_hits,
                'total': len(candidates)
            }
            
    return gene_stats

def main():
    # 1. 扫描所有方法的共同基因 (调试用)
    def get_genes(path, suffix):
        if not os.path.exists(path): return set()
        return {os.path.basename(f).replace(suffix, '') for f in glob.glob(f"{path}/*{suffix}")}

    # 🔥 [修改点 4] Intersection 加入 CADD
    # 注意：这里 Saliency 的路径已经更新为包含 RNA 的
    common_genes = sorted(list(
        get_genes(METHOD_DIRS['Borzoi'], '_optim_log.csv') &
        get_genes(METHOD_DIRS['Saliency'], '_saliency.csv') &
        get_genes(METHOD_DIRS['FunSeq'], '_funseq.csv') &
        get_genes(METHOD_DIRS['CADD'], '_cadd.csv')
    ))
    
    print(f"🔍 Found {len(common_genes)} common genes.")
    if len(common_genes) == 0: 
        print("⚠️  Warning: No common genes found. Checking paths:")
        for m, p in METHOD_DIRS.items():
            print(f"   {m}: {p}")
        return

    # 2. 区分已缓存和未缓存的基因
    genes_to_process = []
    if args.force:
        genes_to_process = common_genes
    else:
        for gene in common_genes:
            cache_file = f"{CACHE_DIR}/{gene}.json"
            if not os.path.exists(cache_file):
                genes_to_process.append(gene)
            else:
                # 🔥 [智能缓存检查] 检查现有的 Cache 是否包含所有新加的方法
                try:
                    with open(cache_file, 'r') as f:
                        data = json.load(f)
                        first_key = list(data.keys())[0]
                        # 检查 CADD/Saliency 是否都在
                        if not all(m in data[first_key] for m in METHODS_LIST):
                            genes_to_process.append(gene) 
                except:
                    genes_to_process.append(gene) 
    
    print(f"🚀 Cached: {len(common_genes) - len(genes_to_process)} | To Process: {len(genes_to_process)}")

    # 3. 处理基因
    if genes_to_process:
        gene_map = load_gene_metadata()
        needed_ids = [gene_map[g] for g in genes_to_process if g in gene_map]
        gtex_data = load_gtex_dict(needed_ids)
        
        for gene in tqdm(genes_to_process, desc="Processing Genes"):
            if gene not in gene_map: continue
            gene_id = gene_map[gene]
            pool_snps = get_real_pool_size(gene)
            if not pool_snps: continue 
            
            df_gtex = gtex_data.get(gene_id, None)
            stats_json = process_gene(gene, gene_id, df_gtex, pool_snps)
            
            with open(f"{CACHE_DIR}/{gene}.json", 'w') as f:
                json.dump(stats_json, f)

    # 4. 汇总
    print("📊 Aggregating results from cache...")
    agg_stats = {str(t): {m: {'hits': 0, 'total': 0} for m in METHODS_LIST + ['Random']} for t in THRESHOLDS}
    
    for gene in tqdm(common_genes, desc="Aggregating"):
        cache_file = f"{CACHE_DIR}/{gene}.json"
        if not os.path.exists(cache_file): continue
        
        with open(cache_file, 'r') as f:
            gene_stats = json.load(f)
            
        for t_str, methods_data in gene_stats.items():
            if t_str in agg_stats:
                for m, data in methods_data.items():
                    if m in agg_stats[t_str]:
                        agg_stats[t_str][m]['hits'] += data['hits']
                        agg_stats[t_str][m]['total'] += data['total']

    # 5. 准备绘图数据
    plot_data = []
    for thresh in THRESHOLDS:
        t_str = str(thresh)
        r_stat = agg_stats[t_str]['Random']
        base_rate = r_stat['hits'] / r_stat['total'] if r_stat['total'] > 0 else 0
        
        if base_rate == 0: continue 
        
        for m in METHODS_LIST:
            m_stat = agg_stats[t_str][m]
            m_rate = m_stat['hits'] / m_stat['total'] if m_stat['total'] > 0 else 0
            enrich = m_rate / base_rate
            
            plot_data.append({
                'Threshold': f"{thresh:.0e}",
                'Method': m,
                'Enrichment': enrich
            })

    # 6. 绘图
    if not plot_data:
        print("❌ No data available for plotting.")
        return

    df_plot = pd.DataFrame(plot_data)
    df_plot.to_csv(f"{OUTPUT_DIR}/enrichment_fixed_k{TOP_K}_barplot_data.csv", index=False)
    
    plt.figure(figsize=(12, 7))
    sns.set_theme(style="whitegrid")
    
    # 🔥 [修改点 5] 分配颜色
    colors = {
        'Borzoi': '#e74c3c',   # Red
        'Saliency': '#2ecc71', # Green
        'FunSeq': '#9b59b6',   # Purple
        'CADD': '#3498db'      # Blue
    }
    
    ax = sns.barplot(
        data=df_plot,
        x='Threshold',
        y='Enrichment',
        hue='Method',
        palette=colors,
        edgecolor='black',
        linewidth=1,
        alpha=0.9
    )
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', fontsize=8, padding=3)

    plt.title(f"GTEx Enrichment Benchmark (Top-{TOP_K}, {CURRENT_TISSUE.title()})", fontsize=16, fontweight='bold')
    plt.xlabel("GTEx P-value Threshold", fontsize=12)
    plt.ylabel("Enrichment Factor (vs Random)", fontsize=12)
    plt.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    save_path = f"{OUTPUT_DIR}/enrichment_fixed_k{TOP_K}_barplot.png"
    plt.savefig(save_path, dpi=300)
    print(f"✅ Barplot saved: {save_path}")

if __name__ == "__main__":
    main()
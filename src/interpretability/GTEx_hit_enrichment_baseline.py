import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse

# ================= 命令行参数配置 =================
parser = argparse.ArgumentParser(description="Calculate GTEx Enrichment for specific tissue")
parser.add_argument('--tissue', type=str, required=True, 
                    choices=['brain', 'blood', 'liver', 'heart', 'muscle', 'pancreas', 'Pancreas'],
                    help="Select tissue type to analyze (e.g., brain, blood)")
args = parser.parse_args()

# 统一转为小写处理，防止 Pancreas 大小写问题
CURRENT_TISSUE = args.tissue.lower()

# ================= 路径配置 =================
BASE_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP'
META_CSV_PATH = f'{BASE_DIR}/dataset/gene_3000_borzoi_gencode_v41_hg38.csv'
SNP_POOL_DIR = f'{BASE_DIR}/dataset/gene_snps_hg38'
GTEX_BASE_DIR = f'{BASE_DIR}/dataset/GTEx_Analysis_v8_eQTL'

# ✅ [核心配置] 组织 -> 文件夹名 & GTEx文件名 映射表
# 请根据实际 GTEx 文件名核对 'gtex_file' 的值
TISSUE_CONFIG = {
    'brain': {
        'folder': 'brain_K10_borzoi_modeltrain_res',
        'gtex_file': 'Brain_Cortex.v8.signif_variant_gene_pairs.txt.gz', # 假设是 Cortex，如果是其他部位请修改
        'color': '#3498db' # 蓝色
    },
    'blood': {
        'folder': 'blood_K10_borzoi_modeltrain_res', # 如果是原来的 multihead 文件夹，请在这里修改
        'gtex_file': 'Whole_Blood.v8.signif_variant_gene_pairs.txt.gz',
        'color': '#e74c3c' # 红色
    },
    'liver': {
        'folder': 'liver_K10_borzoi_modeltrain_res',
        'gtex_file': 'Liver.v8.signif_variant_gene_pairs.txt.gz',
        'color': '#2ecc71' # 绿色
    },
    'heart': {
        'folder': 'heart_K10_borzoi_modeltrain_res',
        'gtex_file': 'Heart_Left_Ventricle.v8.signif_variant_gene_pairs.txt.gz', # 假设是左心室
        'color': '#9b59b6' # 紫色
    },
    'muscle': {
        'folder': 'muscle_K10_borzoi_modeltrain_res',
        'gtex_file': 'Muscle_Skeletal.v8.signif_variant_gene_pairs.txt.gz',
        'color': '#e67e22' # 橙色
    },
    'pancreas': {
        'folder': 'Pancreas_K10_borzoi_modeltrain_res', # 注意：如果文件夹也是小写请改为 pancreas
        'gtex_file': 'Pancreas.v8.signif_variant_gene_pairs.txt.gz',
        'color': '#f1c40f' # 黄色
    }
}

# 获取当前配置
if CURRENT_TISSUE not in TISSUE_CONFIG:
    raise ValueError(f"Unsupported tissue: {CURRENT_TISSUE}")

config = TISSUE_CONFIG[CURRENT_TISSUE]

# ✅ [修正] 动态生成路径
RESULTS_DIR = f'{BASE_DIR}/results/{config["folder"]}'
GTEX_DATA_PATH = f'{GTEX_BASE_DIR}/{config["gtex_file"]}'

# 输出目录加上 tissue 后缀，防止覆盖
OUTPUT_DIR = f'{BASE_DIR}/results/res_enrichment_gtex/{CURRENT_TISSUE}'

print(f"🔧 Configuration for [{CURRENT_TISSUE.upper()}]:")
print(f"   📂 Model Results: {RESULTS_DIR}")
print(f"   🧬 GTEx Data:     {GTEX_DATA_PATH}")
print(f"   💾 Output Dir:    {OUTPUT_DIR}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 定义 P-value 阶梯
THRESHOLDS = [1e-5, 1e-8, 1e-10, 1e-12, 1e-15, 1e-20]

# ===========================================

def load_gene_metadata():
    """加载基因列表和 ID 映射"""
    print("📖 Loading Metadata...")
    df = pd.read_csv(META_CSV_PATH)
    name_to_id = dict(zip(df['gene_name'], df['gene_ID'].apply(lambda x: x.split('.')[0])))
    return name_to_id

def get_real_pool_size(gene_name):
    """读取 CSV 计算真实的 Candidate Pool Size"""
    csv_path = f"{SNP_POOL_DIR}/{gene_name}_snps_hg38.csv"
    if not os.path.exists(csv_path):
        return None
    try:
        with open(csv_path, 'r') as f:
            return sum(1 for _ in f) - 1
    except:
        return None

def load_gtex_dict(target_gene_ids):
    """
    一次性加载 GTEx 数据并按 Gene ID 索引。
    """
    if not os.path.exists(GTEX_DATA_PATH):
        raise FileNotFoundError(f"GTEx file not found: {GTEX_DATA_PATH}")

    print(f"📖 Loading GTEx Data for {CURRENT_TISSUE.upper()} (filtering for {len(target_gene_ids)} genes)...")
    gtex_dict = {}
    target_set = set(target_gene_ids)
    
    try:
        reader = pd.read_csv(GTEX_DATA_PATH, sep='\t', 
                             usecols=['variant_id', 'gene_id', 'pval_nominal'], 
                             chunksize=100000, compression='gzip')
        
        for chunk in tqdm(reader, desc="Parsing GTEx"):
            chunk['clean_id'] = chunk['gene_id'].str.split('.').str[0]
            
            mask = chunk['clean_id'].isin(target_set)
            if not mask.any(): continue
            
            filtered = chunk[mask].copy()
            # 解析 Pos (variant_id: chr_pos_ref_alt_b38)
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

def parse_model_top_k(log_path, k=10):
    if not os.path.exists(log_path): return []
    try:
        df = pd.read_csv(log_path)
        if df.empty: return []
        last = df.iloc[-1]
        return [int(last[f"Rank{i}_Pos"]) for i in range(1, k+1) if f"Rank{i}_Pos" in last]
    except: return []

# ================= 主逻辑 =================

def main():
    # 1. 准备基因列表
    gene_map = load_gene_metadata()
    
    # 2. 扫描已有的结果文件
    log_files = glob.glob(f"{RESULTS_DIR}/*_optim_log.csv")
    
    if not log_files:
        print(f"⚠️  No optimization logs found in {RESULTS_DIR}")
        return

    valid_genes = []
    for f in log_files:
        gname = os.path.basename(f).replace('_optim_log.csv', '')
        if gname in gene_map:
            valid_genes.append(gname)
            
    print(f"🚀 Found {len(valid_genes)} genes with optimization results in {RESULTS_DIR}")
    
    # 3. 加载 GTEx 数据
    needed_ids = [gene_map[g] for g in valid_genes]
    gtex_data = load_gtex_dict(needed_ids)
    
    # 4. 全局统计计数器
    stats = {t: {'model_hits': 0, 'model_total': 0, 'base_hits': 0, 'base_total': 0} for t in THRESHOLDS}
    
    # 5. 遍历基因计算
    for gene in tqdm(valid_genes, desc=f"Analyzing Genes ({CURRENT_TISSUE})"):
        gene_id = gene_map[gene]
        
        pool_size = get_real_pool_size(gene)
        if not pool_size or pool_size == 0: continue
        
        log_path = f"{RESULTS_DIR}/{gene}_optim_log.csv"
        top_k = parse_model_top_k(log_path, k=10)
        if not top_k: continue
        
        if gene_id not in gtex_data:
            df_gtex = pd.DataFrame(columns=['pos', 'pval_nominal'])
        else:
            df_gtex = gtex_data[gene_id]
            
        for thresh in THRESHOLDS:
            true_hits_set = set(df_gtex[df_gtex['pval_nominal'] < thresh]['pos'])
            num_true = len(true_hits_set)
            
            stats[thresh]['base_hits'] += num_true
            stats[thresh]['base_total'] += pool_size
            
            hits = sum(1 for p in top_k if p in true_hits_set)
            stats[thresh]['model_hits'] += hits
            stats[thresh]['model_total'] += len(top_k)

    # 6. 计算最终结果并绘图
    plot_x = []
    plot_y = []
    
    print("\n" + "="*60)
    print(f"📊 {CURRENT_TISSUE.upper()} GTEx ENRICHMENT RESULTS (Tiered)")
    print("="*60)
    print(f"{'P-Value <':<10} | {'Model Rate':<12} | {'Base Rate':<12} | {'Enrichment':<10}")
    
    results_list = []
    
    for thresh in THRESHOLDS:
        d = stats[thresh]
        m_rate = d['model_hits'] / d['model_total'] if d['model_total'] > 0 else 0
        b_rate = d['base_hits'] / d['base_total'] if d['base_total'] > 0 else 0
        enrich = m_rate / b_rate if b_rate > 0 else 0
        
        print(f"{thresh:<10.0e} | {m_rate:.2%}       | {b_rate:.2%}       | {enrich:.2f}x")
        
        results_list.append({
            'Threshold': thresh,
            'Model_Rate': m_rate,
            'Base_Rate': b_rate,
            'Enrichment': enrich
        })
        
        if enrich > 0:
            plot_x.append(str(thresh))
            plot_y.append(enrich)

    # 7. 保存数据
    pd.DataFrame(results_list).to_csv(f"{OUTPUT_DIR}/{CURRENT_TISSUE}_enrichment_stats.csv", index=False)
    
    # 8. 画图
    plt.figure(figsize=(8, 6))
    sns.set_style("whitegrid")
    
    # 使用 Tissue 专属颜色
    plot_color = config.get('color', '#333333')
    
    plt.plot(plot_x, plot_y, marker='o', linewidth=3, color=plot_color, markersize=10)
    plt.title(f"Model Enrichment vs {CURRENT_TISSUE.title()} GTEx Significance", fontsize=14, pad=20)
    plt.xlabel("GTEx P-value Threshold (More Stringent →)", fontsize=12)
    plt.ylabel("Enrichment Factor (Model / Random)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    for i, txt in enumerate(plot_y):
        plt.annotate(f"{txt:.1f}x", (plot_x[i], plot_y[i]), 
                     xytext=(0, 10), textcoords='offset points', 
                     ha='center', fontweight='bold', color=plot_color)

    plt.tight_layout()
    save_path = f"{OUTPUT_DIR}/{CURRENT_TISSUE}_enrichment_curve.png"
    plt.savefig(save_path, dpi=300)
    print(f"\n✅ Plot saved to: {save_path}")

if __name__ == "__main__":
    main()
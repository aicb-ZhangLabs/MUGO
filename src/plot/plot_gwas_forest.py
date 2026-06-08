import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from scipy import stats
from tqdm import tqdm
import argparse

# ================= 命令行参数配置 =================
parser = argparse.ArgumentParser(description="Generate GWAS Forest Plot for specific tissue")
parser.add_argument('--tissue', type=str, required=True, 
                    choices=['brain', 'blood', 'liver', 'heart', 'muscle', 'pancreas', 'Pancreas'],
                    help="Select tissue type to analyze (e.g., brain, blood)")
parser.add_argument('--mode', type=str, default='best', choices=['best', 'last'],
                    help="Choose 'best' to use the epoch with max Gain, or 'last' for final epoch. Default: best")

args = parser.parse_args()

# 统一转为小写处理
CURRENT_TISSUE = args.tissue.lower()
CURRENT_MODE = args.mode

# ================= 路径配置 =================
BASE_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP'
SNP_POOL_DIR = f'{BASE_DIR}/dataset/gene_snps_hg38'
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
# Output Dir 增加 mode 区分
OUTPUT_DIR = f'{BASE_DIR}/results/res_enrichment_gwas/{CURRENT_TISSUE}_{CURRENT_MODE}'

print(f"🔧 Configuration for [{CURRENT_TISSUE.upper()}]:")
print(f"   🎯 Mode:          {CURRENT_MODE.upper()} EPOCH")
print(f"   📂 Model Results: {RESULTS_DIR}")
print(f"   💾 Output Dir:    {OUTPUT_DIR}")

os.makedirs(OUTPUT_DIR, exist_ok=True)
GWAS_P_THRESHOLD = 5e-8

# === 定义疾病大类 (Keywords) ===
# 你可以根据需要扩展这个字典
CATEGORIES = {
    "Cancer": ["cancer", "carcinoma", "tumor", "neoplasm", "leukemia", "lymphoma", "melanoma"],
    "Neurological": ["alzheimer", "parkinson", "brain", "cognitive", "schizophrenia", "depression", "autism", "neuro", "dementia", "mental", "bipolar"],
    "Cardiovascular": ["heart", "cardio", "artery", "vascular", "blood pressure", "hypertension", "stroke", "atrial", "coronary"],
    "Immune/Autoimmune": ["immune", "rheumatoid", "lupus", "asthma", "allergy", "sclerosis", "crohn", "inflammatory", "psoriasis", "celiac"],
    "Metabolic": ["diabetes", "obesity", "bmi", "cholesterol", "lipid", "glucose", "insulin", "metabolic", "body mass"],
    "Global (All)": [] # 特殊类别，包含所有
}

# ==========================================
# 2. 核心函数
# ==========================================

def load_and_categorize_gwas(gwas_path):
    print(f"📖 Loading and categorizing GWAS Catalog...")
    try:
        df = pd.read_csv(gwas_path, sep='\t', low_memory=False, 
                         usecols=['CHR_ID', 'CHR_POS', 'P-VALUE', 'MAPPED_TRAIT'])
    except:
        print("⚠️ Standard read failed, trying alternative parsing...")
        df = pd.read_csv(gwas_path.split('?')[0], sep='\t', compression='zip', low_memory=False, 
                 usecols=['CHR_ID', 'CHR_POS', 'P-VALUE', 'MAPPED_TRAIT'])

    # 清洗 P-value
    df['P_VAL_FLOAT'] = pd.to_numeric(df['P-VALUE'], errors='coerce')
    df = df[df['P_VAL_FLOAT'] < GWAS_P_THRESHOLD].copy()
    
    # 清洗坐标
    df['clean_chrom'] = df['CHR_ID'].astype(str).apply(lambda x: f"chr{x}" if not str(x).startswith('chr') else x)
    df['clean_pos'] = pd.to_numeric(df['CHR_POS'], errors='coerce').astype('Int64')
    df = df.dropna(subset=['clean_pos'])
    
    # 构建 Category Sets
    cat_sets = {k: set() for k in CATEGORIES.keys()}
    
    df['trait_lower'] = df['MAPPED_TRAIT'].astype(str).str.lower()
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Categorizing GWAS"):
        key = (row['clean_chrom'], row['clean_pos'])
        trait = row['trait_lower']
        
        cat_sets['Global (All)'].add(key)
        
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
    csv_path = f"{SNP_POOL_DIR}/{gene_name}_snps_hg38.csv"
    if not os.path.exists(csv_path): return []
    try:
        df = pd.read_csv(csv_path)
        if 'POS_hg38' in df.columns: return df['POS_hg38'].astype(int).tolist()
        elif 'pos' in df.columns: return df['pos'].astype(int).tolist()
    except: pass
    return []

def parse_model_top_k(log_path, mode, k=10):
    """
    ✅ [核心修改] 支持 best/last 模式
    """
    if not os.path.exists(log_path): return []
    try:
        df = pd.read_csv(log_path)
        if df.empty: return []

        if mode == 'best':
            # 找到 Gain 最大的那一行
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
    except: 
        return []

# ==========================================
# 3. 统计与绘图
# ==========================================

def main():
    # 1. 准备数据
    gwas_cat_sets = load_and_categorize_gwas(GWAS_FILE)
    
    meta_df = pd.read_csv(META_CSV_PATH)
    gene_chrom_map = dict(zip(meta_df['gene_name'], meta_df['chr']))
    
    log_files = glob.glob(f"{RESULTS_DIR}/*_optim_log.csv")
    
    if not log_files:
        print(f"⚠️  No optimization logs found in {RESULTS_DIR}")
        return

    # 统计每个 Category 的计数器
    counts = {cat: {'model_hits': 0, 'model_total': 0, 'bg_hits': 0, 'bg_total': 0} for cat in CATEGORIES}
    
    print(f"\n🚀 Scanning {len(log_files)} genes for {CURRENT_TISSUE.upper()} (Mode: {CURRENT_MODE})...")
    
    for log_path in tqdm(log_files):
        gene = os.path.basename(log_path).replace('_optim_log.csv', '')
        if gene not in gene_chrom_map: continue
        
        chrom = f"chr{gene_chrom_map[gene]}".replace('chrchr', 'chr')
        
        # A. Model Set (传入 Mode)
        model_pos = parse_model_top_k(log_path, mode=CURRENT_MODE, k=10)
        bg_pos = get_candidate_snps(gene)
        
        if not bg_pos: continue
        
        model_set = set([(chrom, p) for p in model_pos])
        bg_set = set([(chrom, p) for p in bg_pos])
        
        # B. 针对每个 Category 计算 Hit
        for cat, gwas_set in gwas_cat_sets.items():
            # Model Hits
            m_hits = len(model_set.intersection(gwas_set))
            counts[cat]['model_hits'] += m_hits
            counts[cat]['model_total'] += len(model_set)
            
            # Background Hits
            b_hits = len(bg_set.intersection(gwas_set))
            counts[cat]['bg_hits'] += b_hits
            counts[cat]['bg_total'] += len(bg_set)

    # 2. 计算 OR 和 CI
    plot_data = []
    
    print("\n📊 Enrichment by Category:")
    print(f"{'Category':<20} | {'OR':<6} | {'P-val':<10} | {'CI_Lower':<8} | {'CI_Upper':<8}")
    
    for cat in CATEGORIES:
        d = counts[cat]
        a, b = d['model_hits'], d['model_total'] - d['model_hits']
        c, d_val = d['bg_hits'], d['bg_total'] - d['bg_hits']
        
        if c == 0 or a == 0: continue 
        
        odds_ratio, p_value = stats.fisher_exact([[a, b], [c, d_val]])
        
        # 95% CI
        try:
            log_or = np.log(odds_ratio) if odds_ratio > 0 else 0
            se = np.sqrt(1/a + 1/b + 1/c + 1/d_val)
            ci_lower = np.exp(log_or - 1.96 * se)
            ci_upper = np.exp(log_or + 1.96 * se)
        except:
            ci_lower, ci_upper = odds_ratio, odds_ratio
            
        print(f"{cat:<20} | {odds_ratio:.2f}   | {p_value:.2e}   | {ci_lower:.2f}     | {ci_upper:.2f}")
        
        plot_data.append({
            'Category': cat,
            'OR': odds_ratio,
            'Lower': ci_lower,
            'Upper': ci_upper,
            'P_val': p_value,
            'Hits': a
        })

    # 3. 画森林图
    if not plot_data:
        print("⚠️ No significant hits found in any category. Skipping plot.")
        return

    df_plot = pd.DataFrame(plot_data)
    
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
    plt.title(f"[{CURRENT_MODE.upper()}] Trait-Specific Enrichment: {CURRENT_TISSUE.title()}", fontsize=14, pad=20)
    
    # 动态调整 X 轴范围
    max_val = max(df_plot['Upper']) if not df_plot.empty else 1
    # 避免 CI 过宽导致压缩，设置上限
    display_limit = min(max_val, 20.0) 
    
    for i, row in enumerate(df_plot.itertuples()):
        label = f"OR={row.OR:.2f} (p={row.P_val:.1e})"
        # 标注位置调整
        pos_x = min(row.Upper + 0.5, display_limit + 0.5)
        plt.text(pos_x, i, label, va='center', fontsize=9, color='#34495e')

    plt.xlim(0.0, display_limit * 1.3)
    
    plt.tight_layout()
    out_path = f"{OUTPUT_DIR}/{CURRENT_TISSUE}_{CURRENT_MODE}_gwas_forest_plot.png"
    plt.savefig(out_path, dpi=300)
    
    df_plot.to_csv(f"{OUTPUT_DIR}/{CURRENT_TISSUE}_{CURRENT_MODE}_enrichment_summary.csv", index=False)
    
    print(f"\n✅ Forest plot saved to: {out_path}")

if __name__ == "__main__":
    main()
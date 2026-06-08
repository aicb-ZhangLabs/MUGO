import pandas as pd
import numpy as np
import os
import glob
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# ================= 配置 =================
BASE_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP'
CAUSAL_PROXY_DIR = f'{BASE_DIR}/dataset/GWAS_Catelog_disease/UKBB_causal_proxy'
RESULTS_DIR_BASE = f'{BASE_DIR}/results'

# 过滤掉 Heart (数据太少)
TARGET_TISSUES = ['blood', 'liver', 'muscle', 'pancreas']

# Model 结果文件夹
TISSUE_FOLDERS = {
    'blood': 'blood_K10_borzoi_modeltrain_res',
    'liver': 'liver_K10_borzoi_modeltrain_res',
    'muscle': 'muscle_K10_borzoi_modeltrain_res',
    'pancreas': 'Pancreas_K10_borzoi_modeltrain_res'
}

# ✅ 关键参数：最小分辨距离 (bp)
# 只有当 Proxy 离 Causal 远于这个距离时，我们才考核模型能不能分得清。
# Borzoi 的 bin size 通常是 32-128bp，但感受野重叠严重。建议设为 2000 或 5000。
MIN_DIST_THRESHOLD = 2000 

def load_model_top_k(tissue, k=10):
    """Load Model Top K SNPs (hg38 pos)"""
    folder = TISSUE_FOLDERS.get(tissue)
    if not folder: return set()
    
    res_path = f"{RESULTS_DIR_BASE}/{folder}"
    files = glob.glob(f"{res_path}/*_optim_log.csv")
    
    model_snps = set()
    print(f"   Loading Model Results for {tissue}...")
    for f in tqdm(files, leave=False):
        try:
            df = pd.read_csv(f)
            if df.empty: continue
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
    print(f"🚀 Starting Analysis (Filter: Proxy dist > {MIN_DIST_THRESHOLD}bp)...")

    for tissue in TARGET_TISSUES:
        print(f"\nAnalyzing [{tissue.upper()}]...")
        
        # 1. 读取 LiftOver 后的文件
        cp_file = f"{CAUSAL_PROXY_DIR}/{tissue}_causal_proxy_hg38.csv"
        if not os.path.exists(cp_file):
            print(f"⚠️ File not found: {cp_file}")
            continue
            
        ukbb_df = pd.read_csv(cp_file)
        
        # 2. ✅ 应用距离过滤器 (只针对 Proxy)
        # Causal 保留
        causals = ukbb_df[ukbb_df['type'] == 'Causal'].copy()
        
        # Proxy 过滤
        proxies_raw = ukbb_df[ukbb_df['type'] == 'Proxy'].copy()
        
        # 计算距离: Proxy Pos - Linked Causal Pos
        # 注意：如果文件中没有 linked_causal_pos_hg38 列，我们需要跳过或用 pos 近似
        if 'linked_causal_pos_hg38' in proxies_raw.columns:
            proxies_raw['dist_to_causal'] = (proxies_raw['pos'] - proxies_raw['linked_causal_pos_hg38']).abs()
            
            # 过滤
            proxies_filtered = proxies_raw[proxies_raw['dist_to_causal'] > MIN_DIST_THRESHOLD]
            
            n_dropped = len(proxies_raw) - len(proxies_filtered)
            print(f"   ✂️ Filter: Dropped {n_dropped} proxies too close (<{MIN_DIST_THRESHOLD}bp) to causal.")
        else:
            print("   ⚠️ Warning: No linkage info found. Skipping distance filter.")
            proxies_filtered = proxies_raw

        # 合并回用于统计的 DF
        analysis_df = pd.concat([causals, proxies_filtered])

        # 3. 读取模型预测
        model_set = load_model_top_k(tissue, k=10)
        
        # 4. 统计命中率
        stats = {'Causal': {'hits': 0, 'total': 0}, 'Proxy': {'hits': 0, 'total': 0}}
        
        for idx, row in analysis_df.iterrows():
            snp_type = row['type']
            pos = int(row['pos'])
            
            stats[snp_type]['total'] += 1
            if pos in model_set:
                stats[snp_type]['hits'] += 1
        
        # 5. 输出结果 & 准备画图
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

    # ================= 画图 =================
    if not plot_data: return

    df = pd.DataFrame(plot_data)
    
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    # 颜色: Causal红, Proxy灰
    colors = {'Causal': '#D62728', 'Proxy': '#999999'}
    
    ax = sns.barplot(
        data=df,
        x='Tissue',
        y='Recovery Rate (%)',
        hue='SNP Type',
        palette=colors,
        edgecolor="black",
        linewidth=1.2
    )
    
    # 标注 Fold Change
    tissues = df['Tissue'].unique()
    for i, tissue in enumerate(tissues):
        sub = df[df['Tissue'] == tissue]
        try:
            causal_rate = sub[sub['SNP Type'] == 'Causal']['Recovery Rate (%)'].values[0]
            proxy_rate = sub[sub['SNP Type'] == 'Proxy']['Recovery Rate (%)'].values[0]
            
            # 计算 Fold
            if proxy_rate > 0:
                fold = causal_rate / proxy_rate
                label = f"{fold:.1f}x"
            elif causal_rate > 0:
                label = "Inf" # Proxy是0，Causal有值，无限倍
            else:
                label = ""

            # 标注位置
            y_max = max(causal_rate, proxy_rate)
            if label:
                ax.text(i, y_max + 0.1, label, ha='center', va='bottom', fontsize=12, fontweight='bold')
        except:
            pass

    plt.title(f'Resolution Test: Causal vs. Distal Proxy (>{MIN_DIST_THRESHOLD}bp)', fontsize=15, fontweight='bold')
    plt.ylabel('Recovery Rate in Model Top 10 (%)', fontsize=12)
    plt.xlabel('')
    plt.legend(title='Variant Type')
    
    out_fig = f"{BASE_DIR}/results/res_enrichment_gwas/Figure4B_DistFilter_{MIN_DIST_THRESHOLD}bp.png"
    plt.savefig(out_fig, dpi=300)
    print(f"\n✅ Figure 4B (Filtered) saved to: {out_fig}")

if __name__ == "__main__":
    main()
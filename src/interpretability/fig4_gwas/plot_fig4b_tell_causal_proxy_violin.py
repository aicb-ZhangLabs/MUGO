import pandas as pd
import numpy as np
import os
import glob
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import fisher_exact

# ================= Configuration =================
BASE_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP'
CAUSAL_PROXY_DIR = f'{BASE_DIR}/dataset/GWAS_Catelog_disease/UKBB_causal_proxy'
RESULTS_DIR_BASE = f'{BASE_DIR}/results'

# 目标组织
TARGET_TISSUES = ['blood', 'liver', 'muscle', 'pancreas']

# 文件夹映射
TISSUE_FOLDERS = {
    'blood': 'blood_K10_borzoi_modeltrain_res',
    'liver': 'liver_K10_borzoi_modeltrain_res',
    'muscle': 'muscle_K10_borzoi_modeltrain_res',
    'pancreas': 'Pancreas_K10_borzoi_modeltrain_res'
}

# 过滤器配置 (同之前)
MIN_DIST_THRESHOLD = 2000 
BOOTSTRAP_N = 1000 # 重采样次数

def load_model_top_k(tissue, k=10):
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

def run_bootstrap(hits, total, n_boot=1000):
    """
    对 0/1 数组进行 Bootstrap 重采样，返回 n_boot 个 Rate
    """
    if total == 0: return [0] * n_boot
    
    # 构造原始数据 [1, 1, 0, 0, ...]
    data = np.array([1]*hits + [0]*(total-hits))
    
    boot_rates = []
    # 使用 numpy 快速重采样
    for _ in range(n_boot):
        sample = np.random.choice(data, size=total, replace=True)
        boot_rates.append(sample.mean() * 100) # 转百分比
    return boot_rates

def main():
    plot_data = []
    stats_summary = [] # 用来存 P 值和 Rate 方便画图标注
    
    print(f"🚀 Starting Bootstrap Violin Analysis...")

    for tissue in TARGET_TISSUES:
        print(f"\nAnalyzing [{tissue.upper()}]...")
        
        # 1. Load Data
        cp_file = f"{CAUSAL_PROXY_DIR}/{tissue}_causal_proxy_hg38.csv"
        if not os.path.exists(cp_file): continue
        ukbb_df = pd.read_csv(cp_file)
        
        # 2. Filter Proxies
        causals = ukbb_df[ukbb_df['type'] == 'Causal'].copy()
        proxies = ukbb_df[ukbb_df['type'] == 'Proxy'].copy()
        
        if 'linked_causal_pos_hg38' in proxies.columns:
            proxies['dist'] = (proxies['pos'] - proxies['linked_causal_pos_hg38']).abs()
            proxies = proxies[proxies['dist'] > MIN_DIST_THRESHOLD]
        
        # 3. Model Hits
        model_set = load_model_top_k(tissue, k=10)
        
        # Count Hits
        c_total = len(causals)
        c_hits = sum(1 for p in causals['pos'] if int(p) in model_set)
        
        p_total = len(proxies)
        p_hits = sum(1 for p in proxies['pos'] if int(p) in model_set)
        
        print(f"   Causal: {c_hits}/{c_total} | Proxy: {p_hits}/{p_total}")

        # 4. Bootstrap Distribution (为了画 Violin)
        c_boot = run_bootstrap(c_hits, c_total, BOOTSTRAP_N)
        p_boot = run_bootstrap(p_hits, p_total, BOOTSTRAP_N)
        
        # 添加到画图数据
        for val in c_boot:
            plot_data.append({'Tissue': tissue.capitalize(), 'Type': 'Causal', 'BootRate': val})
        for val in p_boot:
            plot_data.append({'Tissue': tissue.capitalize(), 'Type': 'Proxy', 'BootRate': val})
            
        # 5. Fisher Exact Test (为了标 P-value)
        # Contingency Table: [[Causal_Hit, Causal_Miss], [Proxy_Hit, Proxy_Miss]]
        table = [[c_hits, c_total - c_hits], [p_hits, p_total - p_hits]]
        odds, p_val = fisher_exact(table, alternative='greater') # 单尾检验: Causal > Proxy?
        
        stats_summary.append({
            'Tissue': tissue.capitalize(),
            'P_Val': p_val,
            'Causal_Rate': (c_hits/c_total*100) if c_total else 0,
            'Proxy_Rate': (p_hits/p_total*100) if p_total else 0
        })

    # ================= Plotting =================
    if not plot_data: return

    df = pd.DataFrame(plot_data)
    
    plt.figure(figsize=(11, 7))
    sns.set_theme(style="whitegrid")
    
    # 颜色
    colors = {'Causal': '#D62728', 'Proxy': '#999999'} # 红 vs 灰
    
    # 画 Violin
    # inner='quartile' 会显示分布的中位数和四分位线，看起来信息量很大
    ax = sns.violinplot(data=df, x='Tissue', y='BootRate', hue='Type', 
                        split=True, # 左右分开画，对比更强烈
                        inner='quartile',
                        palette=colors,
                        linewidth=1.2)
    
    # 标注 P-value 和 Rate
    # 获取 y 轴范围来确定标注位置
    y_max = df['BootRate'].max()
    y_min = df['BootRate'].min()
    y_range = y_max - y_min
    
    for i, stat in enumerate(stats_summary):
        tissue = stat['Tissue']
        p_val = stat['P_Val']
        c_rate = stat['Causal_Rate']
        p_rate = stat['Proxy_Rate']
        
        # 1. 标注 P-value (在上方)
        # 将 P 值转为科学计数法或星号
        p_text = f"P={p_val:.1e}" if p_val < 0.001 else f"P={p_val:.3f}"
        if p_val < 0.05: p_text += " *"
        if p_val < 0.01: p_text += "*"
        
        # 找该 Tissue 在图中的最高点
        this_tissue_max = df[df['Tissue']==tissue]['BootRate'].max()
        ax.text(i, this_tissue_max + y_range*0.05, p_text, 
                ha='center', va='bottom', fontsize=11, fontweight='bold', color='black')
        
        # 2. 标注具体 Rate (可选，标在 Violin 肚子里或下方)
        # 我们标在底部 x轴上方一点
        ax.text(i-0.2, y_min - y_range*0.05, f"{c_rate:.1f}%", color='#D62728', ha='center', fontsize=10, fontweight='bold')
        ax.text(i+0.2, y_min - y_range*0.05, f"{p_rate:.1f}%", color='#666666', ha='center', fontsize=10, fontweight='bold')

    plt.title('Bootstrap Analysis: Causal vs. Distal Proxy Discrimination', fontsize=16, fontweight='bold')
    plt.ylabel('Bootstrapped Recovery Rate (%)', fontsize=14)
    plt.xlabel('')
    
    # 调整 Legend
    plt.legend(title='Variant Type', loc='upper right')
    
    # 保存
    out_fig = f"{BASE_DIR}/results/res_enrichment_gwas/Figure4B_Violin_Bootstrap.png"
    plt.savefig(out_fig, dpi=300, bbox_inches='tight')
    print(f"\n✅ Violin Plot saved to: {out_fig}")

if __name__ == "__main__":
    main()
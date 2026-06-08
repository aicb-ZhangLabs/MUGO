'''
2*3, enformer + borzoi 3000个gene的gain的plot，CAGE的enformer+borzoi 6个tissue，证明跨modality一致性的图。
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from scipy.stats import pearsonr, spearmanr, gaussian_kde
from tqdm import tqdm

# ================= ⚙️ 配置区域 =================

BASE_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP'
DATA_ROOT = f'{BASE_DIR}/results'

# 结果保存路径 (你的要求)
OUTPUT_DIR = f'{BASE_DIR}/results/Fig3_multi_modal/comparison_plots'
CACHE_DIR = f'{OUTPUT_DIR}/cache'

# 确保目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# 基因列表路径
GENE_LIST_CSV = f'{BASE_DIR}/dataset/gene_3000_borzoi_gencode_v41_hg38.csv'

# 组织列表 (注意文件夹命名的大小写，Pancreas 首字母大写)
TISSUES = ['blood', 'brain', 'liver', 'heart', 'muscle', 'Pancreas']

# 绘图字体设置
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'sans-serif',
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'pdf.fonttype': 42, # 确保 PDF 字体嵌入，方便后期编辑
    'ps.fonttype': 42
})

# ================= 🛠️ 数据读取函数 =================

def extract_max_gain(log_file):
    """
    读取 CSV 中的 Max Gain。
    如果有任何读取错误或文件不存在，返回 None。
    """
    try:
        if not os.path.exists(log_file):
            return None
        df = pd.read_csv(log_file)
        if df.empty or 'Gain' not in df.columns:
            return None
        return df['Gain'].max()
    except Exception:
        return None

def load_data_for_tissue(tissue, gene_df, force_reload=False):
    """
    读取指定 Tissue 的数据。
    逻辑：检查 Cache -> 有则读 -> 无则遍历 3000 文件并生成 Cache。
    """
    # 缓存文件名
    cache_file = os.path.join(CACHE_DIR, f"comparison_data_{tissue}.csv")
    
    # 1. 如果缓存存在，直接读取
    if os.path.exists(cache_file) and not force_reload:
        print(f"   📖 Loading cache for {tissue}...")
        return pd.read_csv(cache_file)
    
    print(f"   ⚙️  Parsing raw logs for {tissue} (This may take a minute)...")
    
    # 2. 定义原始数据文件夹路径
    bor_dir = os.path.join(DATA_ROOT, f"{tissue}_K10_borzoi_CAGE_modeltrain_res")
    enf_dir = os.path.join(DATA_ROOT, f"{tissue}_K10_enformer_modeltrain_CAGE_res")
    
    data_records = []
    
    # 3. 遍历所有基因
    for _, row in tqdm(gene_df.iterrows(), total=len(gene_df), desc=f"   Scanning {tissue}", leave=False):
        gene = row['gene_name']
        
        # 构造文件名
        bor_path = os.path.join(bor_dir, f"{gene}_borzoi_CAGE_optim_log.csv")
        enf_path = os.path.join(enf_dir, f"{gene}_enformer_optim_log.csv")
        
        bor_gain = extract_max_gain(bor_path)
        enf_gain = extract_max_gain(enf_path)
        
        # 只有两个模型都有结果才记录
        if bor_gain is not None and enf_gain is not None:
            data_records.append({
                'Gene': gene,
                'Borzoi_Gain': bor_gain,
                'Enformer_Gain': enf_gain,
                'Tissue': tissue
            })
            
    # 4. 保存缓存
    if not data_records:
        print(f"   ⚠️ Warning: No valid matching data found for {tissue}!")
        return pd.DataFrame()
        
    df = pd.DataFrame(data_records)
    df.to_csv(cache_file, index=False)
    print(f"   💾 Cache saved to {cache_file} ({len(df)} genes)")
    
    return df

# ================= 🎨 绘图核心函数 =================

def plot_comparison_grid():
    print(f"🚀 Starting Plotting Script...")
    print(f"📂 Results will be saved to: {OUTPUT_DIR}")
    
    # 读取基因列表
    if not os.path.exists(GENE_LIST_CSV):
        print(f"❌ Error: Gene list not found at {GENE_LIST_CSV}")
        return
    gene_df = pd.read_csv(GENE_LIST_CSV)
    
    # 创建 2行3列 的画布
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, tissue in enumerate(TISSUES):
        ax = axes[idx]
        
        # 1. 加载数据
        df = load_data_for_tissue(tissue, gene_df)
        
        if df.empty:
            ax.text(0.5, 0.5, "No Data Found", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(tissue)
            continue
            
        x = df['Enformer_Gain'].values
        y = df['Borzoi_Gain'].values
        
        # 2. 计算密度 (KDE) 用于着色
        try:
            xy = np.vstack([x, y])
            z = gaussian_kde(xy)(xy)
            # 排序：让密度高的点画在上面
            sort_idx = z.argsort()
            x, y, z = x[sort_idx], y[sort_idx], z[sort_idx]
        except:
            z = np.ones_like(x) # 失败则用单色
            
        # 3. 绘制散点图
        sc = ax.scatter(x, y, c=z, s=20, cmap='Spectral_r', alpha=0.7, edgecolor='none')
        
        # 4. 添加对角线 (Identity Line y=x)
        min_lim = min(x.min(), y.min())
        max_lim = max(x.max(), y.max())
        padding = (max_lim - min_lim) * 0.05
        lims = [min_lim - padding, max_lim + padding]
        
        ax.plot(lims, lims, 'k--', alpha=0.5, lw=1.5, label='y=x')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        
        # 5. 计算相关性
        r, _ = pearsonr(x, y)
        rho, _ = spearmanr(x, y)
        
        # 6. 设置标题和标签
        display_tissue = tissue if tissue != 'Pancreas' else 'Pancreas'
        ax.set_title(f"{display_tissue} (N={len(df)})", fontsize=14, fontweight='bold')
        
        # 只在边缘添加轴标签
        if idx >= 3: # 底部一行
            ax.set_xlabel("Enformer Optimization Gain", fontsize=11)
        if idx % 3 == 0: # 左侧一列
            ax.set_ylabel("Borzoi Optimization Gain", fontsize=11)
            
        # 7. 在图上标注统计信息
        stats_text = f"Pearson r = {r:.3f}\nSpearman ρ = {rho:.3f}"
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                fontsize=11, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#cccccc'))
        
        # 8. 简单的线性回归线 (红色)
        m, b = np.polyfit(x, y, 1)
        ax.plot(x, m*x + b, color='red', alpha=0.8, lw=2, label='Fit')

    plt.tight_layout()
    
    # 保存图片
    out_png = os.path.join(OUTPUT_DIR, "Figure_Borzoi_vs_Enformer_Scatter_Grid.png")
    out_svg = os.path.join(OUTPUT_DIR, "Figure_Borzoi_vs_Enformer_Scatter_Grid.svg")
    out_pdf = os.path.join(OUTPUT_DIR, "Figure_Borzoi_vs_Enformer_Scatter_Grid.pdf") # 新增 PDF 路径
    
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_svg, format='svg')
    plt.savefig(out_pdf, format='pdf') # 新增保存 PDF
    
    print(f"✅ Plot saved to: {out_png}")
    print(f"✅ PDF saved to: {out_pdf}")
    # plt.show() # 如果在服务器上跑，可以注释掉这行

if __name__ == "__main__":
    plot_comparison_grid()
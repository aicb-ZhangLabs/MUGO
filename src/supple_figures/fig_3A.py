'''
enformer CAGE 6个tissue的gain，证明不同modal也可以跑。2*3图 (Density Plot Style)
'''

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import gaussian_kde

# ================= ⚙️ 配置区域 =================

# 根目录
BASE_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP/results'
# 缓存和图片保存目录 (当前目录)
OUTPUT_DIR = os.getcwd()

# 任务定义: Enformer CAGE 6个 Tissue
TASKS = [
    ('CAGE', ['blood', 'brain', 'liver', 'heart', 'muscle', 'pancreas'])
]

# 绘图设置: 半张 A4 大小
FIG_SIZE = (14, 8) 
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'sans-serif',
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'pdf.fonttype': 42 # 确保 PDF 文字可编辑
})

# ================= 🛠️ 核心函数 =================

def get_folder_path(tissue):
    """
    构建 Enformer CAGE 的文件夹路径
    格式: {tissue}_K10_enformer_modeltrain_CAGE_res
    修正: Pancreas 文件夹首字母大写
    """
    # 特殊处理: Pancreas 首字母大写
    if tissue == 'pancreas':
        folder_tissue_name = 'Pancreas'
    else:
        folder_tissue_name = tissue
        
    dir_name = f"{folder_tissue_name}_K10_enformer_modeltrain_CAGE_res"
    return os.path.join(BASE_DIR, dir_name)

def process_tissue_data(tissue):
    """读取单个组织的所有基因 CSV，提取 Max Gain"""
    # 这里的 cache 文件名加个 enformer 前缀
    cache_file = os.path.join(OUTPUT_DIR, f"cache_enformer_CAGE_{tissue}.csv")
    
    # 1. 检查缓存
    if os.path.exists(cache_file):
        return pd.read_csv(cache_file)
    
    # 2. 扫描文件
    folder_path = get_folder_path(tissue)
    if not os.path.exists(folder_path):
        print(f"⚠️ Warning: Directory not found: {folder_path}")
        return pd.DataFrame()
    
    csv_files = glob.glob(os.path.join(folder_path, "*_optim_log.csv"))
    if not csv_files:
        print(f"⚠️ Warning: No CSV files found in {folder_path}")
        return pd.DataFrame()

    results = []
    
    # 3. 逐个读取
    print(f"Processing Enformer CAGE - {tissue} ({len(csv_files)} files)...")
    for f in tqdm(csv_files, leave=False):
        try:
            df = pd.read_csv(f)
            if df.empty: continue
            
            # 找到 Gain 最大的行
            best_row = df.loc[df['Gain'].idxmax()]
            
            baseline = best_row['Baseline']
            gain = best_row['Gain']
            
            # 计算相对增益
            if baseline == 0:
                rel_gain = 0
            else:
                rel_gain = gain / baseline

            results.append({
                'Gene': os.path.basename(f).split('_')[0],
                'Baseline': baseline,
                'Gain': gain,
                'Relative_Gain': rel_gain
            })
        except Exception as e:
            continue

    # 4. 保存缓存
    res_df = pd.DataFrame(results)
    res_df.to_csv(cache_file, index=False)
    return res_df

# ================= 🎨 绘图主程序 =================

def plot_enformer_cage_density():
    # 准备画布: 2行 3列
    fig, axes = plt.subplots(2, 3, figsize=FIG_SIZE, constrained_layout=True)
    axes_flat = axes.flatten()
    
    plot_idx = 0
    tissues = TASKS[0][1] # 获取 tissue 列表
    
    for tissue in tissues:
        if plot_idx >= 6: break
        
        ax = axes_flat[plot_idx]
        
        # 获取数据
        df = process_tissue_data(tissue)
        
        if not df.empty:
            x = df['Baseline'].values
            y = df['Relative_Gain'].values
            
            # 数据清洗：X轴是Log尺度，所以要去掉 <=0 的点
            valid_mask = (x > 0.1) & np.isfinite(x) & np.isfinite(y)
            x_plot = x[valid_mask]
            y_plot = y[valid_mask]
            
            # === 🔥 Density Plot 核心逻辑 ===
            try:
                # 为了让密度图在 Log 坐标轴下看起来自然，我们在计算密度时对 X 取 Log
                x_log = np.log10(x_plot)
                xy = np.vstack([x_log, y_plot])
                
                # 计算高斯核密度
                z = gaussian_kde(xy)(xy)
                
                # 排序：让密度高的点(红色)画在最上面，避免被遮挡
                sort_idx = z.argsort()
                x_plot, y_plot, z = x_plot[sort_idx], y_plot[sort_idx], z[sort_idx]
                
                # 绘制散点
                sc = ax.scatter(x_plot, y_plot, c=z, s=15, cmap='Spectral_r', alpha=0.8, edgecolor=None)
            except Exception as e:
                # 如果点太少或计算失败，回退到单色
                print(f"Density calculation failed for {tissue}: {e}")
                ax.scatter(x_plot, y_plot, c='#d95f02', s=15, alpha=0.6)
            
            # 装饰图表
            ax.axhline(0, color='#333333', linestyle='--', linewidth=1.0, alpha=0.6)
            ax.set_xscale('log')
            
            # 显示均值信息 (右上角)
            mean_gain = df['Relative_Gain'].mean()
            stats_text = f"Mean Gain: {mean_gain:.2%}"
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
                    fontsize=10, fontweight='bold', ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='#cccccc'))
        else:
            ax.text(0.5, 0.5, "Data Not Found", ha='center', va='center')

        # 标题和标签
        ax.set_title(f"Enformer CAGE - {tissue.capitalize()}", fontsize=12, pad=10)
        
        # 坐标轴标签处理
        if plot_idx % 3 == 0:
            ax.set_ylabel("Relative Gain")
        else:
            ax.set_ylabel("")
            
        if plot_idx >= 3:
            ax.set_xlabel("Wildtype Baseline (Log Scale)")
        else:
            ax.set_xlabel("")
            
        plot_idx += 1

    # 保存
    save_png = os.path.join(OUTPUT_DIR, "Enformer_CAGE_Density_Plot.png")
    save_pdf = os.path.join(OUTPUT_DIR, "Enformer_CAGE_Density_Plot.pdf")
    
    plt.savefig(save_png, dpi=300)
    plt.savefig(save_pdf, format='pdf')
    
    print(f"\n✅ Figure saved to: {save_png}")
    print(f"✅ PDF saved to: {save_pdf}")

if __name__ == "__main__":
    plot_enformer_cage_density()
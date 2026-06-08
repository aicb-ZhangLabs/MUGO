'''
Plot 4x3 grid (Total 12 plots) for RNA and CAGE modalities.
Layout: A4 Portrait style.
'''

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ================= ⚙️ 配置区域 =================

# 根目录 (请确认此路径是否需要修改)
BASE_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP/results'
# 图片保存目录
OUTPUT_DIR = os.getcwd()

# 目标组织列表
# TARGET_TISSUES = ['blood', 'brain', 'liver', 'heart', 'muscle', 'pancreas']
TARGET_TISSUES = ['lung', 'kidney']

# 任务定义: (Modality, List of Tissues)
TASKS = [
    ('RNA', TARGET_TISSUES),
    ('CAGE', TARGET_TISSUES)
]

# 文件夹命名规则映射
FOLDER_SUFFIX_MAP = {
    'RNA': '',       # RNA 文件夹无后缀
    'CAGE': 'CAGE'   # CAGE 文件夹带 CAGE 后缀
}

# 绘图设置: (宽, 高) - 设置为 12x16 英寸，接近 A4 比例，保证清晰度
FIG_SIZE = (12, 16) 

# ================= 🛠️ 核心函数 =================

def get_folder_path(modality, tissue):
    """根据模态和组织构建文件夹路径"""
    
    # 1. 处理 Pancreas 首字母大写的问题 (文件夹通常是 Pancreas)
    if tissue.lower() == 'pancreas':
        folder_tissue = 'Pancreas'
    else:
        folder_tissue = tissue

    # 2. 获取后缀
    suffix = FOLDER_SUFFIX_MAP.get(modality, modality)
    
    # 构建路径: {Tissue}_K10_borzoi_{Suffix}_modeltrain_res
    if suffix:
        dir_name = f"{folder_tissue}_K10_borzoi_{suffix}_modeltrain_res"
    else:
        dir_name = f"{folder_tissue}_K10_borzoi_modeltrain_res"
        
    return os.path.join(BASE_DIR, dir_name)

# def process_tissue_data(modality, tissue):
#     """读取单个组织的所有基因 CSV，提取 Max Gain"""
#     # 缓存文件名
#     cache_file = os.path.join(OUTPUT_DIR, f"cache_{modality}_{tissue}.csv")
    
#     # 1. 检查缓存 (如果之前跑过，直接读缓存快很多)
#     if os.path.exists(cache_file):
#         return pd.read_csv(cache_file)
    
#     # 2. 扫描文件
#     folder_path = get_folder_path(modality, tissue)
#     if not os.path.exists(folder_path):
#         print(f"⚠️ Warning: Directory not found: {folder_path}")
#         return pd.DataFrame()
    
#     csv_files = glob.glob(os.path.join(folder_path, "*_optim_log.csv"))
#     if not csv_files:
#         print(f"⚠️ Warning: No CSV files found in {folder_path}")
#         return pd.DataFrame()

#     results = []
    
#     # 3. 逐个读取
#     # print(f"Processing {modality} - {tissue}...")
#     for f in tqdm(csv_files, desc=f"{modality}-{tissue}", leave=False):
#         try:
#             df = pd.read_csv(f)
#             if df.empty: continue
            
#             # 找到 Gain 最大的行
#             best_row = df.loc[df['Gain'].idxmax()]
            
#             baseline = best_row['Baseline']
#             gain = best_row['Gain']
            
#             # 计算相对增益
#             if baseline == 0:
#                 rel_gain = 0
#             else:
#                 rel_gain = gain / baseline

#             results.append({
#                 'Gene': os.path.basename(f).split('_')[0],
#                 'Baseline': baseline,
#                 'Gain': gain,
#                 'Relative_Gain': rel_gain
#             })
#         except Exception as e:
#             continue

#     # 4. 保存缓存
#     if results:
#         res_df = pd.DataFrame(results)
#         res_df.to_csv(cache_file, index=False)
#         return res_df
#     else:
#         return pd.DataFrame()



import tarfile
import io

def process_tissue_data(modality, tissue):
    """直接从 .tar.gz 压缩包中读取单个组织的所有基因 CSV"""
    cache_file = os.path.join(OUTPUT_DIR, f"cache_{modality}_{tissue}.csv")
    
    # 1. 检查缓存
    if os.path.exists(cache_file):
        return pd.read_csv(cache_file)
    
    # 2. 构建 .tar.gz 路径
    folder_path = get_folder_path(modality, tissue)
    tar_path = f"{folder_path}.tar.gz"  # 直接加上后缀
    
    if not os.path.exists(tar_path):
        print(f"⚠️ Warning: Tar archive not found: {tar_path}")
        return pd.DataFrame()
    
    results = []
    
    # 3. 打开压缩包并在内存中读取
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            # 筛选出所有 _optim_log.csv 结尾的文件
            csv_members = [m for m in tar.getmembers() if m.name.endswith("_optim_log.csv")]
            
            if not csv_members:
                print(f"⚠️ Warning: No target CSVs found inside {tar_path}")
                return pd.DataFrame()
                
            for member in tqdm(csv_members, desc=f"{modality}-{tissue} (Tar)", leave=False):
                # 提取文件流
                f = tar.extractfile(member)
                if f is not None:
                    try:
                        # Pandas 可以直接读文件流
                        df = pd.read_csv(f)
                        if df.empty: continue
                        
                        best_row = df.loc[df['Gain'].idxmax()]
                        baseline = best_row['Baseline']
                        gain = best_row['Gain']
                        rel_gain = gain / baseline if baseline != 0 else 0

                        # member.name 可能是 "blood_K10.../TP53_optim_log.csv"
                        # 获取纯基因名
                        filename = os.path.basename(member.name)
                        gene_name = filename.split('_')[0]

                        results.append({
                            'Gene': gene_name,
                            'Baseline': baseline,
                            'Gain': gain,
                            'Relative_Gain': rel_gain
                        })
                    except Exception as e:
                        continue
    except Exception as e:
        print(f"❌ Error reading tar file {tar_path}: {e}")
        return pd.DataFrame()

    # 4. 保存缓存
    if results:
        res_df = pd.DataFrame(results)
        res_df.to_csv(cache_file, index=False)
        return res_df
    else:
        return pd.DataFrame()

# ================= 🎨 绘图主程序 =================

def plot_all():
    # 准备画布: 4行 3列
    fig, axes = plt.subplots(4, 3, figsize=FIG_SIZE, constrained_layout=True)
    axes_flat = axes.flatten()
    
    plot_idx = 0
    total_plots = 4 * 3
    
    # 遍历任务 (RNA 先，CAGE 后)
    for modality, tissues in TASKS:
        for tissue in tissues:
            if plot_idx >= total_plots: break
            
            ax = axes_flat[plot_idx]
            
            # 获取数据
            df = process_tissue_data(modality, tissue)
            
            if not df.empty:
                # 绘制散点图
                sns.scatterplot(
                    data=df, 
                    x='Baseline', 
                    y='Relative_Gain', 
                    ax=ax, 
                    s=15, 
                    color='#2b83ba' if modality == 'RNA' else '#d7191c', # RNA蓝，CAGE红，区分一下
                    alpha=0.6, 
                    edgecolor='w',
                    linewidth=0.2
                )
                
                # 辅助线 (0 增益线)
                ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
                
                # 设置 Log X 轴 (Baseline 跨度通常很大)
                ax.set_xscale('log')
                
                # 在图上标注均值
                mean_gain = df['Relative_Gain'].mean()
                ax.text(0.05, 0.92, f"Mean Gain: {mean_gain:.2%}", 
                        transform=ax.transAxes, fontsize=10, fontweight='bold',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

            else:
                ax.text(0.5, 0.5, "Data Not Found", ha='center', va='center')
                print(f"❌ Data missing for {modality} - {tissue}")

            # 标题设置
            ax.set_title(f"{modality} - {tissue.capitalize()}", fontsize=12, fontweight='bold')
            
            # 坐标轴标签逻辑 (只在最左侧和最下侧显示，避免混乱)
            # 行号: plot_idx // 3, 列号: plot_idx % 3
            row = plot_idx // 3
            col = plot_idx % 3
            
            if col == 0:
                ax.set_ylabel("Relative Gain", fontsize=10)
            else:
                ax.set_ylabel("")
                
            if row == 3: # 最后一行
                ax.set_xlabel("Wildtype Baseline (log scale)", fontsize=10)
            else:
                ax.set_xlabel("")
                
            plot_idx += 1

    # 保存
    save_name = "Optimization_RNA_CAGE_4x3_A4.png"
    save_path = os.path.join(OUTPUT_DIR, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Figure saved to: {save_path}")
    
    # PDF 版本 (矢量图适合论文)
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')

if __name__ == "__main__":
    plot_all()

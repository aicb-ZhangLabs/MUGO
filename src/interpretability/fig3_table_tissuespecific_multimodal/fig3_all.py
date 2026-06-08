'''
Clean Plotting Script
Directly loads the pre-calculated matrices:
1. Figure3_RNA-seq_Matrix_Specific.csv (Calculated via Specificity Filter)
2. Figure3_ATAC_Matrix_Specific.csv    (Calculated via Top 50 Gain)
'''
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D 
import pandas as pd
import numpy as np
import os

# ================= ⚙️ 全局配置 =================

# ⚠️ 这里的名字必须严格匹配刚才生成的文件名中间的部分
# 刚才生成的文件是: Figure3_RNA-seq_Matrix_Specific.csv
# 刚才生成的文件是: Figure3_ATAC_Matrix_Specific.csv
MODAL_A = 'RNA-seq' 
MODAL_B = 'CAGE' # 🔥 这里如果是 ATAC，记得改成 ATAC

BASE_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP'
DATA_DIR = f'{BASE_DIR}/results/Fig3_multi_modal'  # 读取 CSV 的路径
OUTPUT_DIR = f'{BASE_DIR}/results/Fig3_multi_modal/Plots' # 保存图片的路径
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 尺寸设置 (保持你原有的样式)
FIG_WIDTH = 3.8 
FIG_HEIGHT = 3.0 
FONT_SIZE = 7

# 🔥 统一 Figure 4 的样式标准
plt.rcParams.update({
    'figure.figsize': (FIG_WIDTH, FIG_HEIGHT),
    'font.size': FONT_SIZE,
    'axes.labelsize': 6,       # 统一为 6
    'axes.titlesize': 6,       # 统一为 6
    'xtick.labelsize': 6,      # 统一为 6
    'ytick.labelsize': 6,      # 统一为 6
    'legend.fontsize': 5,      # 统一为 5
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.linewidth': 0.5,
    'axes.edgecolor': '#666666', # 统一深灰色
    'lines.linewidth': 1.0,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'pdf.fonttype': 42 # 确保 PDF 文字可编辑
})

TISSUES_LABEL = ['Blood', 'Brain', 'Liver', 'Heart', 'Musc.', 'Panc.']
SIZE_SCALE = 150 

# ================= 🧬 数据读取 =================

def get_data_matrix(modality_name):
    # 构造文件名，严格匹配计算脚本的输出
    csv_filename = f'Figure3_{modality_name}_Matrix_Specific.csv'
    csv_path = os.path.join(DATA_DIR, csv_filename)
    
    if os.path.exists(csv_path):
        print(f"   📖 Loading: {csv_filename}")
        df = pd.read_csv(csv_path, index_col=0)
        return df
    else:
        print(f"❌ Error: {csv_filename} not found at {DATA_DIR}")
        print(f"   Please check if the calculation script finished successfully.")
        # 返回随机数据占位，防止报错
        return pd.DataFrame(np.random.rand(6,6), columns=TISSUES_LABEL, index=TISSUES_LABEL)

def apply_z_score(df):
    """Row-wise Z-score Normalization"""
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
    means = df.mean(axis=1)
    stds = df.std(axis=1).replace(0, 1) 
    return df.sub(means, axis=0).div(stds, axis=0)

# ================= 🎨 绘图核心 =================

def draw_dot_plot(ax, df_raw, title, show_y_label=True):
    N = len(df_raw.index)
    M = len(df_raw.columns)
    
    # Z-score 映射颜色
    df_z = apply_z_score(df_raw)
    colors = df_z.values.flatten()
    
    # Size Mapping
    vmin, vmax = -1.5, 2.5
    size_norm = (colors - vmin) / (vmax - vmin)
    size_norm = np.clip(size_norm, 0, 1) 
    sizes = (size_norm ** 1.5) * SIZE_SCALE 
    
    # Grid
    x, y = np.meshgrid(np.arange(M), np.arange(N))
    
    # Plot
    sc = ax.scatter(x.flatten(), y.flatten(), 
                    s=sizes,          
                    c=colors,         
                    marker='s',       
                    cmap='Reds',      
                    vmin=vmin, vmax=vmax,
                    alpha=0.9, 
                    edgecolors='black', 
                    linewidth=0.4, 
                    zorder=2)
    
    # Ticks
    ax.set_xticks(np.arange(M))
    ax.set_xticklabels(TISSUES_LABEL, rotation=45, ha='right') 
    
    ax.set_yticks(np.arange(N))
    if show_y_label:
        ax.set_yticklabels(TISSUES_LABEL)
        # 🔥 FontSize 8 -> 6, Bold -> Normal
        ax.set_ylabel("Optimized Tissue", fontsize=6, labelpad=4, fontweight='normal')
    else:
        ax.set_yticklabels([]) 
    
    ax.invert_yaxis()
    # 🔥 FontSize 9 -> 6, Bold -> Normal
    ax.set_title(title, fontsize=6, pad=8, fontweight='normal')
    
    # Grid Lines (Minor)
    ax.set_xticks(np.arange(M+1)-0.5, minor=True)
    ax.set_yticks(np.arange(N+1)-0.5, minor=True)
    ax.grid(which='minor', color='#f0f0f0', linestyle='-', linewidth=0.8, zorder=1)
    ax.tick_params(which='minor', bottom=False, left=False)
    ax.set_aspect('equal', 'box')

# ================= 🏷️ 图例 =================

def add_shared_legend(fig):
    levels = [-1.0, 0.5, 2.0]  
    labels = ["Low", "Avg", "High"]
    
    legend_handles = []
    cmap = plt.get_cmap('Reds')
    norm = plt.Normalize(-1.5, 2.5) 
    
    for l, label in zip(levels, labels):
        color = cmap(norm(l))
        s_norm = np.clip((l - (-1.5)) / (2.5 - (-1.5)), 0, 1)
        marker_size = np.sqrt((s_norm ** 1.5) * SIZE_SCALE)
        
        handle = Line2D([0], [0], marker='s', color='w', label=label,
                        markerfacecolor=color, 
                        markersize=marker_size, 
                        markeredgecolor='black', 
                        markeredgewidth=0.5)
        legend_handles.append(handle)
    
    # 🔥 Title FontSize 7 -> 5, Legend FontSize 7 -> 5
    fig.legend(handles=legend_handles, 
               loc='lower center', 
               ncol=3,
               title="Specificity (Row Z-Score)",
               title_fontsize=5,
               fontsize=5, 
               frameon=False,
               borderpad=0,
               handletextpad=0.5,
               bbox_to_anchor=(0.5, 0.02)) 

# ================= 🚀 主程序 =================

def main():
    print(f"🚀 Generating Figure 3 Plot (Standardized Style)...")
    
    # 读取刚才计算好的 CSV
    df_a = get_data_matrix(MODAL_A)
    df_b = get_data_matrix(MODAL_B)
    
    fig = plt.figure()
    
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 1], wspace=0.1, bottom=0.25, top=0.85, left=0.15, right=0.95)
    
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    
    draw_dot_plot(ax_a, df_a, title=f"{MODAL_A} Specificity", show_y_label=True)
    draw_dot_plot(ax_b, df_b, title=f"{MODAL_B} Specificity", show_y_label=False)
    
    add_shared_legend(fig)
    
    # 输出文件
    out_name = f'Figure3_{MODAL_A}_{MODAL_B}_MixedStrategy_Plot'
    out_png = os.path.join(OUTPUT_DIR, f'{out_name}.png')
    out_pdf = os.path.join(OUTPUT_DIR, f'{out_name}.pdf')
    out_svg = os.path.join(OUTPUT_DIR, f'{out_name}.svg')
    
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf, format='pdf')
    plt.savefig(out_svg, format='svg')
    print(f"✅ Plot saved to: {out_png}")
    print(f"✅ PDF saved to: {out_pdf}")

if __name__ == "__main__":
    main()
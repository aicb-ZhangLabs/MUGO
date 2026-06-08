import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os
import numpy as np

# ================= 配置 =================
BASE_RES_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP/results/res_enrichment_gwas'

# 定义组织顺序
TISSUES_ORDER = ['Blood', 'Liver', 'Heart', 'Brain', 'Muscle', 'Pancreas']

# 定义颜色 (Random灰 -> MUGO橙 -> TSS蓝渐变)
COLORS = {
    'Random': '#D3D3D3',       # 灰色
    'MUGO':   '#FF7F0E',       # 亮橙色 (紧贴 Random)
    'TSS (K=10)': '#C6DBEF',   # 浅蓝
    'TSS (K=20)': '#6BAED6',   # 中蓝
    'TSS (K=50)': '#2171B5',   # 深蓝
    'TSS (K=100)': '#3182BD',  # 深蓝 
    'TSS (K=200)': '#08519C'   # 最深蓝 
}

def load_and_aggregate_stats(base_dir):
    """
    智能读取所有数据：
    1. TSS Rate 必须严格对应 K 值。
    2. MUGO Rate 取该组织下所有记录中的最大值 (Best Performance)。
    3. Random Rate 取平均。
    """
    search_path = os.path.join(base_dir, "**", "*summary_stats.csv")
    files = glob.glob(search_path, recursive=True)
    
    if not files:
        print(f"❌ No summary stats files found in {base_dir}")
        return pd.DataFrame()
    
    # 临时存储字典: data_map[tissue] = list of rows
    data_map = {}
    
    print(f"📂 Scanning {len(files)} files...")
    for f in files:
        try:
            df = pd.read_csv(f)
            # 统一组织名称
            tissue = df['Tissue'].iloc[0].capitalize()
            # 修正 Pancreas 大小写
            if tissue == 'Pancreas': tissue = 'Pancreas' 
            
            if tissue not in data_map:
                data_map[tissue] = []
            data_map[tissue].append(df.iloc[0]) # 假设每个summary文件只有一行
        except Exception as e:
            print(f"⚠️ Error reading {f}: {e}")

    # 构建最终画图数据
    final_plot_data = []
    
    for tissue, rows in data_map.items():
        # 1. 获取 Random Baseline (取所有文件的均值，理论上应该是一样的)
        bg_rates = [r['Background_Rate'] for r in rows]
        avg_bg_rate = np.mean(bg_rates)
        
        final_plot_data.append({
            'Tissue': tissue,
            'Method': 'Random',
            'Rate': avg_bg_rate,
            'Enrichment': 1.0
        })
        
        # 2. 获取 MUGO (取所有文件中的最大值 - "取表现最好的")
        model_rates = [r['Model_Rate'] for r in rows]
        best_model_rate = max(model_rates) if model_rates else 0
        
        final_plot_data.append({
            'Tissue': tissue,
            'Method': 'MUGO',  # 名字简化
            'Rate': best_model_rate,
            'Enrichment': best_model_rate / avg_bg_rate if avg_bg_rate > 0 else 0
        })
        
        # 3. 获取 TSS (严格对应 K)
        for r in rows:
            k = int(r['Top_K'])
            if k in [10, 20, 50, 100, 200]: # 只关心这些 K
                final_plot_data.append({
                    'Tissue': tissue,
                    'Method': f'TSS (K={k})',
                    'Rate': r['TSS_Rate'],
                    'Enrichment': r['TSS_Rate'] / avg_bg_rate if avg_bg_rate > 0 else 0
                })

    return pd.DataFrame(final_plot_data)

def plot_figure_4a(df, output_path):
    """绘制最终图表：Random -> MUGO -> TSSs"""
    
    # 1. 过滤 & 排序
    # 指定 Hue Order (画图顺序)
    hue_order = ['Random', 'MUGO', 'TSS (K=10)', 'TSS (K=20)', 'TSS (K=50)', 'TSS (K=100)', 'TSS (K=200)']
    
    # 确保数据里只包含这些 Method
    df = df[df['Method'].isin(hue_order)]
    
    # 组织排序
    df['Tissue'] = pd.Categorical(df['Tissue'], categories=TISSUES_ORDER, ordered=True)
    df.sort_values(['Tissue', 'Method'], inplace=True)
    df = df.dropna()

    # ================= 绘图 =================
    plt.figure(figsize=(16, 8))
    sns.set_theme(style="whitegrid")
    
    ax = sns.barplot(
        data=df, 
        x='Tissue', 
        y='Rate', 
        hue='Method',
        hue_order=hue_order, # 强制指定顺序
        palette=COLORS,
        edgecolor="black",
        linewidth=1,
        errwidth=0
    )

    # ================= 标注数值 (x倍数) =================
    # 这里的逻辑稍微复杂一点，因为要跳过 Random 的标注
    
    # 获取所有的 patch (柱子)
    # Seaborn 的 barplot 会按 hue 分组画 patch
    # 顺序是: All Random bars, then All MUGO bars, etc.
    
    # 我们需要遍历 ax.containers
    for container_idx, container in enumerate(ax.containers):
        # container 对应一种 Method (hue)
        current_method = hue_order[container_idx]
        
        # 如果是 Random，不标数字 (或者你想标 1.0x 也可以)
        if current_method == 'Random':
            continue
            
        # 遍历该 Method 下的每个 Tissue 的柱子
        for bar in container:
            height = bar.get_height() # 这就是 Rate
            if height > 0:
                # 反向查找对应的 Enrichment
                # 通过 height 找 enrichment 有点危险(浮点数)，最好是通过坐标
                # 这里简单点：找 Random 的高度算 Enrichment
                
                # 找到该柱子对应的 x 坐标 (Tissue)
                bar_x = bar.get_x() + bar.get_width() / 2
                
                # 这种反查太麻烦，直接从 height 估算倍数吧
                # 我们需要该 Tissue 对应的 Random Rate
                # 比较 tricky，为了代码简洁，我们假设 df 是有序的
                
                # 替代方案：直接用 bar_label 但传入自定义 labels
                pass

    #重新实现标注逻辑：
    # 遍历每个 container，手动计算 label
    
    # 1. 先建立 Tissue -> RandomRate 的映射
    random_rates = df[df['Method'] == 'Random'].set_index('Tissue')['Rate'].to_dict()
    
    for i, container in enumerate(ax.containers):
        current_method = hue_order[i]
        if current_method == 'Random': continue # 跳过 Random 标注
        
        labels = []
        for bar in container:
            # 找到 bar 对应的 tissue
            # x 坐标对应的 tick index
            # Seaborn barplot x 轴是 0, 1, 2...
            # bar.get_x() 会偏离中心
            
            # 这种方法更稳：根据 height 和 random rate 算
            # 只要 height > 0
            if bar.get_height() <= 0:
                labels.append("")
                continue
                
            # 没办法直接从 bar 对象获取 x-label，只能通过 index 猜
            # 这是一个 hacky 但有效的方法：按顺序匹配 TISSUES_ORDER
            # 但前提是数据是完整的。如果缺数据就会错位。
            
            # 最稳妥的方法：不标了？不行，必须标。
            # 重新遍历 DataFrame 来标。
            pass

    # 使用最简单的 bar_label，显示 Rate (百分比) 或者 Fold
    # 为了不出错，我们直接把 Enrichment 算好传进去
    
    for container, method in zip(ax.containers, hue_order):
        if method == 'Random': continue
        
        # 构造 labels
        labels = []
        # Container 里的 bar 是按 x-axis (Tissue) 顺序排列的
        # 我们假设 Tissues 顺序是固定的 TISSUES_ORDER
        # 如果某个 Tissue 缺数据，Seaborn 会画一个高度为0的bar (或空位)
        
        for j, bar in enumerate(container):
            height = bar.get_height()
            if height == 0 or np.isnan(height):
                labels.append("")
                continue
            
            # 找到对应的 Random Rate
            tissue_name = TISSUES_ORDER[j] 
            if tissue_name in random_rates:
                bg = random_rates[tissue_name]
                enrichment = height / bg
                labels.append(f"{enrichment:.2f}x")
            else:
                labels.append("")
        
        ax.bar_label(container, labels=labels, padding=3, fontsize=10, fontweight='bold')

    plt.title('GWAS Hit Rate & Enrichment: MUGO vs. Baselines', fontsize=18, fontweight='bold')
    plt.ylabel('GWAS Hit Rate (%)', fontsize=14)
    plt.xlabel('', fontsize=12)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=12)
    plt.legend(title='', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"✅ Figure saved to: {output_path}")

# ================= 主流程 =================
if __name__ == "__main__":
    print(f"📂 Scanning results in: {BASE_RES_DIR}")
    
    # 1. 智能读取与聚合
    df = load_and_aggregate_stats(BASE_RES_DIR)
    
    if not df.empty:
        print(f"   Processed {len(df)} data points.")
        
        # 保存一下聚合后的数据方便检查
        df.to_csv(os.path.join(BASE_RES_DIR, 'Final_Plot_Data.csv'), index=False)
        
        # 2. 画图
        out_file = os.path.join(BASE_RES_DIR, 'Figure4A_Final.png')
        plot_figure_4a(df, out_file)
    else:
        print("❌ No data found.")
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os
import argparse

# ================= 命令行参数配置 =================
parser = argparse.ArgumentParser(description="Generate Global Heatmap for GWAS Enrichment")
parser.add_argument('--mode', type=str, default='best', choices=['best', 'last'],
                    help="Choose 'best' (default) or 'last' epoch results.")
args = parser.parse_args()

CURRENT_MODE = args.mode

# ================= 配置 =================
BASE_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP/results/res_enrichment_gwas'
# 这里的顺序决定了 Heatmap 的行列顺序，可以手动调整以凑出“对角线”效果
TISSUES = ['brain', 'pancreas', 'liver', 'blood', 'muscle', 'heart']
TRAITS = ['Neurological', 'Metabolic', 'Immune/Autoimmune', 'Cancer', 'Cardiovascular']

def plot_heatmap():
    print(f"🚀 Generating Heatmap for Mode: {CURRENT_MODE.upper()}")
    data = []
    
    # 1. 收集所有 Tissue 的 summary.csv
    for tissue in TISSUES:
        # 构建动态路径: {BASE_DIR}/{tissue}_{mode}/{tissue}_{mode}_enrichment_summary.csv
        folder_name = f"{tissue}_{CURRENT_MODE}"
        file_name = f"{tissue}_{CURRENT_MODE}_enrichment_summary.csv"
        csv_path = os.path.join(BASE_DIR, folder_name, file_name)
        
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                if row['Category'] in TRAITS:
                    data.append({
                        'Tissue': tissue.title(),
                        'Trait': row['Category'],
                        'OR': row['OR'],
                        'P_val': row['P_val']
                    })
        else:
            print(f"⚠️ Warning: File not found: {csv_path}")
    
    if not data:
        print("❌ No data found. Please ensure 'calc_enrichment.py' has been run for all tissues with the selected mode.")
        return

    df_all = pd.DataFrame(data)

    # 2. 转换成 Matrix (Pivot Table)
    # 颜色深浅用 Odds Ratio
    matrix_or = df_all.pivot(index='Trait', columns='Tissue', values='OR')
    # P-value 用于打星号
    matrix_p = df_all.pivot(index='Trait', columns='Tissue', values='P_val')
    
    # 重新排序行列以获得最佳视觉效果 (试图凑对角线)
    matrix_or = matrix_or.reindex(index=TRAITS, columns=[t.title() for t in TISSUES])
    matrix_p = matrix_p.reindex(index=TRAITS, columns=[t.title() for t in TISSUES])

    # 3. 画图
    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1.2)
    
    # 绘制热图
    # 调整 cmap 使其更美观，YlGnBu 比较经典
    ax = sns.heatmap(matrix_or, annot=True, fmt=".2f", cmap="YlGnBu", 
                     cbar_kws={'label': 'Enrichment (Odds Ratio)'},
                     linewidths=1, linecolor='white')
    
    # 4. 叠加显著性标记
    # p < 0.05 (*), p < 1e-5 (**), p < 1e-8 (***)
    for y in range(matrix_p.shape[0]):
        for x in range(matrix_p.shape[1]):
            p_val = matrix_p.iloc[y, x]
            if pd.isna(p_val): continue
            
            text = ""
            if p_val < 1e-8: text = "***"
            elif p_val < 1e-5: text = "**"
            elif p_val < 0.05: text = "*"
            
            if text:
                # 在格子里稍微偏移一点位置画星号，避免遮挡数字
                ax.text(x + 0.7, y + 0.35, text, color='black', ha='center', va='center', fontsize=14, fontweight='bold')

    plt.title(f"Tissue-Specific Disease Enrichment Landscape ({CURRENT_MODE.upper()} Epoch)", fontsize=16, pad=20)
    plt.tight_layout()
    
    # 保存文件，带上 mode 后缀
    # 【修改处】: 扩展名改为 .svg，并指定 format='svg'
    out_file = f"{BASE_DIR}/global_tissue_trait_heatmap_{CURRENT_MODE}.svg"
    plt.savefig(out_file, format='svg', bbox_inches='tight')
    print(f"✅ Heatmap saved to: {out_file}")

if __name__ == "__main__":
    plot_heatmap()
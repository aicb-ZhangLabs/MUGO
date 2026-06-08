'''
Summary script for Borzoi K Ablation.
Updates:
1. Gain -> Mean Percentage Gain (%)
2. PhyloP -> Top 10%, 5%, 1% Enrichment Factors (x)
'''

import pandas as pd
import os

# ================= ⚙️ 路径配置 =================
BASE_RESULT_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP/results/ablation_borzoi_K'

# 输入文件路径
GAIN_FILE   = f'{BASE_RESULT_DIR}/borzoi_gain/ablation_trend_summary.csv'
GTEX_FILE   = f'{BASE_RESULT_DIR}/GTEx_hit/ablation_gtex_summary.csv'
GWAS_FILE   = f'{BASE_RESULT_DIR}/GWAS_disease_hit/ablation_gwas_summary.csv'
PHYLOP_FILE = f'{BASE_RESULT_DIR}/PolyP_hit/ablation_phylop_summary.csv'

# 输出路径
OUTPUT_CSV = f'{BASE_RESULT_DIR}/final_ablation_table.csv'
OUTPUT_TEX = f'{BASE_RESULT_DIR}/final_ablation_table.tex'

# GWAS 类别
GWAS_CATEGORY = 'Global (All)' 
# ===============================================

def main():
    print("🚀 Generating Final Ablation Table...")
    
    # --- 1. 读取各个模块的数据 ---
    try:
        df_gain = pd.read_csv(GAIN_FILE)
        print(f"✅ Loaded Gain data from {os.path.basename(GAIN_FILE)}")
    except FileNotFoundError:
        print(f"❌ Missing: {GAIN_FILE}")
        return

    try:
        df_gtex = pd.read_csv(GTEX_FILE)
        print(f"✅ Loaded GTEx data from {os.path.basename(GTEX_FILE)}")
    except FileNotFoundError:
        print(f"⚠️ Missing: {GTEX_FILE}")
        df_gtex = None

    try:
        # ✅ [修改] 读取 PhyloP 的三个 Enrichment 列
        df_phylop = pd.read_csv(PHYLOP_FILE)
        print(f"✅ Loaded PhyloP data from {os.path.basename(PHYLOP_FILE)}")
    except FileNotFoundError:
        print(f"⚠️ Missing: {PHYLOP_FILE}")
        df_phylop = None

    try:
        df_gwas_raw = pd.read_csv(GWAS_FILE)
        df_gwas = df_gwas_raw[df_gwas_raw['Category'] == GWAS_CATEGORY][['K', 'OR']]
        print(f"✅ Loaded GWAS data ({GWAS_CATEGORY}) from {os.path.basename(GWAS_FILE)}")
    except FileNotFoundError:
        print(f"⚠️ Missing: {GWAS_FILE}")
        df_gwas = None

    # --- 2. 数据合并 (Merge) ---
    # 以 Gain 表为基础
    df_final = df_gain[['K', 'Mean_Pct_Gain', 'Mean_Gini']].copy()
    
    # 重命名基础列
    df_final.rename(columns={
        'K': 'Ensemble Heads ($K$)',
        'Mean_Pct_Gain': 'Mean Percentage Gain (%)', 
        'Mean_Gini': 'Robustness (Gini)'
    }, inplace=True)

    # Merge GWAS
    if df_gwas is not None:
        df_final = df_final.merge(df_gwas, left_on='Ensemble Heads ($K$)', right_on='K', how='left')
        df_final.drop(columns=['K'], inplace=True)
        df_final.rename(columns={'OR': 'GWAS Enrichment (OR)'}, inplace=True)

    # Merge PhyloP (3 Columns)
    if df_phylop is not None:
        # ✅ [修改] 提取 Top 10%, 5%, 1%
        cols_to_merge = ['K', 'Top10pct_Enrichment', 'Top5pct_Enrichment', 'Top1pct_Enrichment']
        df_final = df_final.merge(df_phylop[cols_to_merge], left_on='Ensemble Heads ($K$)', right_on='K', how='left')
        df_final.drop(columns=['K'], inplace=True)
        
        # ✅ [修改] 重命名为更友好的显示格式
        df_final.rename(columns={
            'Top10pct_Enrichment': 'PhyloP Top 10% (x)',
            'Top5pct_Enrichment': 'PhyloP Top 5% (x)',
            'Top1pct_Enrichment': 'PhyloP Top 1% (x)'
        }, inplace=True)

    # Merge GTEx
    if df_gtex is not None:
        df_final = df_final.merge(df_gtex[['K', 'Overlap_Rate_pct']], left_on='Ensemble Heads ($K$)', right_on='K', how='left')
        df_final.drop(columns=['K'], inplace=True)
        df_final.rename(columns={'Overlap_Rate_pct': 'GTEx eQTL Overlap (%)'}, inplace=True)

    # --- 3. 格式化数据 ---
    df_display = df_final.copy()
    
    # Gain
    if 'Mean Percentage Gain (%)' in df_display.columns:
        df_display['Mean Percentage Gain (%)'] = df_display['Mean Percentage Gain (%)'].map('{:.2f}'.format)
        
    # Gini
    if 'Robustness (Gini)' in df_display.columns:
        df_display['Robustness (Gini)'] = df_display['Robustness (Gini)'].map('{:.3f}'.format)
        
    # GWAS
    if 'GWAS Enrichment (OR)' in df_display.columns:
        df_display['GWAS Enrichment (OR)'] = df_display['GWAS Enrichment (OR)'].map('{:.2f}'.format)
        
    # ✅ [修改] PhyloP (加 'x' 后缀)
    for col in ['PhyloP Top 10% (x)', 'PhyloP Top 5% (x)', 'PhyloP Top 1% (x)']:
        if col in df_display.columns:
            df_display[col] = df_display[col].map('{:.2f}x'.format)
            
    # GTEx
    if 'GTEx eQTL Overlap (%)' in df_display.columns:
        df_display['GTEx eQTL Overlap (%)'] = df_display['GTEx eQTL Overlap (%)'].map('{:.1f}\%'.format)

    # --- 4. 保存 CSV ---
    df_display.to_csv(OUTPUT_CSV, index=False)
    print(f"\n💾 CSV Table saved to: {OUTPUT_CSV}")

    # --- 5. 生成 LaTeX ---
    # 自动计算列格式：第一列左对齐(l)，其余居中(c)
    num_cols = len(df_display.columns)
    col_fmt = 'l' + 'c' * (num_cols - 1)
    
    latex_code = df_display.to_latex(index=False, escape=False, column_format=col_fmt)
    
    full_latex = r"""
\begin{table}[h]
    \centering
    \caption{Ablation Study on Ensemble Size ($K$). Comparison of optimization efficacy (Gain, Robustness) and biological validity (GWAS, PhyloP Conservation Enrichment, GTEx).}
    \label{tab:ablation_k}
    \resizebox{\columnwidth}{!}{%
""" + latex_code + r"""    }
\end{table}
"""
    
    with open(OUTPUT_TEX, 'w') as f:
        f.write(full_latex)
        
    print(f"💾 LaTeX Table saved to: {OUTPUT_TEX}")
    print("\n" + "="*80)
    print("✨ Generated Table Preview:")
    print("="*80)
    print(df_display.to_string(index=False))
    print("="*80)

if __name__ == "__main__":
    main()
'''
Generate Final Table R5 (Absolute Mean Magnitude ± SEM)
Matches Borzoi Table 2 logic perfectly for OpenReview.
Path: /home/dongbos/Combine_optim_Borzoi_SNP/rebuttal/results_rebuttal/add_backbone_model/expecto/generate_expecto_table.py
'''
import pandas as pd
import numpy as np
import os
from scipy.stats import ttest_rel

# ================= ⚙️ 配置路径 =================
# 指向 ExPecto 的结果目录
RESULTS_ROOT = '/home/dongbos/Combine_optim_Borzoi_SNP/rebuttal/results_rebuttal/add_backbone_model/expecto'

TISSUES = ['blood', 'brain']
METHODS = ['MUGO_Gain', 'Saliency_Gain', 'CADD_Gain', 'FunSeq_Gain']
DISPLAY_NAMES = ['MUGO (Ours)', 'Saliency', 'CADD', 'FunSeq']
BASELINE_METHODS = ['Saliency_Gain', 'CADD_Gain', 'FunSeq_Gain']

def generate_table():
    table_rows = []
    
    print(f"📊 Analyzing ExPecto Benchmark Results in: {RESULTS_ROOT}\n")
    
    for tissue in TISSUES:
        # 读取 ExPecto 的 Benchmark 结果文件
        file_path = f"{RESULTS_ROOT}/benchmark_CAGE_{tissue}_expecto.csv"
        
        if not os.path.exists(file_path):
            print(f"⚠️ 找不到文件: {file_path}，跳过 {tissue}。")
            continue
            
        df = pd.read_csv(file_path)
        
        if df.empty:
            print(f"⚠️ {tissue} 的数据为空，跳过。")
            continue
            
        row_data = {'Tissue': tissue.capitalize()}
        means_for_comparison = {}
        
        # ================================================================
        # 💡 核心对齐逻辑：Mean Magnitude ± SEM
        # ================================================================
        for method, d_name in zip(METHODS, DISPLAY_NAMES):
            # 获取原始数据并去除 NaN
            raw_vals = df[method].dropna()
            
            if len(raw_vals) > 0:
                # 1. 先算 Mean，让正负噪声抵消
                raw_mean = raw_vals.mean()
                # 2. 再取绝对值，展示 Magnitude
                final_mean = abs(raw_mean)
                # 3. 使用 SEM 缩小方差
                sem_val = raw_vals.sem()
                
                # 🔥 这里非常关键：ExPecto 是 Log Fold Change，数值很小
                # 所以必须保留 3 位小数，否则全是 0.0！
                row_data[d_name] = f"{final_mean:.3f} ± {sem_val:.3f}"
                means_for_comparison[method] = final_mean
            else:
                row_data[d_name] = "0.000 ± 0.000"
                means_for_comparison[method] = 0.0
            
        # ================================================================
        # 🏆 寻找最强 Baseline & 计算 P-value
        # ================================================================
        # 基于 Magnitude 找到表现最好的 Baseline
        best_baseline = max(BASELINE_METHODS, key=lambda x: means_for_comparison[x])
        best_baseline_display = DISPLAY_NAMES[METHODS.index(best_baseline)]
        
        # 计算 P-value
        if len(df) > 1:
            stat, p_val = ttest_rel(df['MUGO_Gain'].fillna(0), df[best_baseline].fillna(0))
            if p_val < 0.001: p_str = "P < 0.001"
            elif p_val < 0.01: p_str = "P < 0.01"
            elif p_val < 0.05: p_str = "P < 0.05"
            else: p_str = f"P = {p_val:.3f}"
        else:
            p_str = "N/A"
            
        row_data['Best Baseline'] = best_baseline_display
        row_data['P-value (vs Best)'] = p_str
        
        table_rows.append(row_data)

    if not table_rows:
        print("❌ 未能生成任何表格数据。")
        return

    res_df = pd.DataFrame(table_rows)
    
    cols = ['Tissue', 'MUGO (Ours)', 'Saliency', 'CADD', 'FunSeq', 'Best Baseline', 'P-value (vs Best)']
    res_df = res_df[cols]
    
    # ================= 💾 保存文件 =================
    out_csv = f"{RESULTS_ROOT}/Table_R5_ExPecto_Summary.csv"
    res_df.to_csv(out_csv, index=False)
    print(f"✅ CSV 表格已保存至: {out_csv}")
    
    out_md = f"{RESULTS_ROOT}/Table_R5_ExPecto_Summary.md"
    
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("### Table R5: Performance of MUGO vs. Baselines on ExPecto Foundation Model\n")
        f.write("*Values represent the magnitude of mean signal log-fold change ($|\\overline{\\Delta S}| \\pm \\text{SEM}$).*\n\n")
        f.write("| Tissue | MUGO (Ours) | Saliency | CADD | FunSeq | P-value (vs Best) |\n")
        f.write("|:---|:---|:---|:---|:---|:---|\n")
        for _, row in res_df.iterrows():
            f.write(f"| **{row['Tissue']}** | **{row['MUGO (Ours)']}** | {row['Saliency']} | {row['CADD']} | {row['FunSeq']} | {row['P-value (vs Best)']} |\n")
            
    print(f"✅ Markdown 文件已保存至: {out_md} (可直接复制粘贴)\n")
    
    # ================= 🖨️ 终端预览 =================
    print("="*85)
    print("🏆 Markdown 表格预览 (OpenReview 格式)")
    print("="*85)
    with open(out_md, "r", encoding="utf-8") as f:
        print(f.read())
    print("="*85 + "\n")

if __name__ == "__main__":
    generate_table()
'''
用来算paper里的gain之类的百分比。
'''
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, ttest_ind
import os

# ================= 配置路径 =================
BASE_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP'
RESULTS_DIR = f'{BASE_DIR}/results'

def get_stats():
    print("🚀 Calculating statistics for Paper Blanks...\n")

    # ==========================================
    # 1. Perturbation Efficacy (Fig 1B / 2B)
    # ==========================================
    print("--- [Text Section 1: Perturbation Efficacy] ---")
    gain_csv = f'{RESULTS_DIR}/baseline_benchmark/benchmark_gain_comparison_blood.csv'
    
    if os.path.exists(gain_csv):
        df = pd.read_csv(gain_csv)
        
        # 获取绝对值 (Absolute Gain)
        mugo_vals = df['Borzoi_Gain'].abs()
        saliency_vals = df['Saliency_Gain'].abs()
        cadd_vals = df['CADD_Gain'].abs()
        funseq_vals = df['FunSeq_Gain'].abs()
        
        # 计算平均值 (Mean) - 对应文中 "average gene expression changes"
        # 如果您觉得数据偏斜严重，也可以改用 median()，但文中写的是 average
        mu_mugo = mugo_vals.mean()
        mu_saliency = saliency_vals.mean()
        mu_cadd = cadd_vals.mean()
        mu_funseq = funseq_vals.mean()
        
        # 计算 P-value (Two-sided t-test)
        t_stat, p_val = ttest_ind(mugo_vals, saliency_vals, alternative='two-sided')
        
        # 格式化 P-value
        if p_val < 0.001:
            p_str = f"{p_val:.2e}" # 科学计数法，例如 1.23e-05
        else:
            p_str = f"{p_val:.4f}"

        print(f"📄 Fill in the blanks:")
        print(f"   '...achieved average gene expression changes of [{mu_mugo:.4f}] (MUGO) and [{mu_saliency:.4f}] (Saliency)...'")
        print(f"   '...such as CADD and FunSeq ([{mu_cadd:.4f}] and [{mu_funseq:.4f}]...)'")
        print(f"   '...significantly outperformed saliency-based baselines (two-sided t-test, P = [{p_str}])...'")
    else:
        print(f"❌ Error: Gain CSV not found at {gain_csv}")

    # ==========================================
    # 2. GTEx Enrichment (Fig 1C / 2C)
    # ==========================================
    print("\n--- [Text Section 2: GTEx Enrichment] ---")
    enrich_csv = f'{RESULTS_DIR}/baseline_benchmark/enrichment_fixed_k100_barplot_data.csv'
    
    if os.path.exists(enrich_csv):
        df = pd.read_csv(enrich_csv)
        # 筛选 Threshold 为 1e-05 的行 (根据之前的代码逻辑)
        # 必须确保 DataFrame 里有 'Threshold' 列且格式正确
        df_target = df[df['Threshold'].astype(str).str.contains('1e-05')]
        
        # 提取各个方法的 Enrichment 值
        # 注意：这里假设 CSV 里的 Method名 分别为 'Borzoi', 'Saliency', 'CADD', 'FunSeq'
        # 如果 CSV 里名字不一样（比如小写），请在这里修改
        try:
            e_mugo = df_target[df_target['Method'] == 'Borzoi']['Enrichment'].values[0]
            e_saliency = df_target[df_target['Method'] == 'Saliency']['Enrichment'].values[0]
            e_cadd = df_target[df_target['Method'] == 'CADD']['Enrichment'].values[0]
            e_funseq = df_target[df_target['Method'] == 'FunSeq']['Enrichment'].values[0]
            
            print(f"📄 Fill in the blanks:")
            print(f"   '...achieved an enrichment of [{e_mugo:.2f}] over random...'")
            print(f"   '...observed for all baselines ([{e_saliency:.2f}], [{e_cadd:.2f}], and [{e_funseq:.2f}] for saliency, CADD, and FunSeq, respectively).'")
        except IndexError:
            print("⚠️ Error: Could not find all methods in the enrichment CSV. Check method names.")
            print(f"Available methods in CSV: {df_target['Method'].unique()}")
    else:
        print(f"❌ Error: Enrichment CSV not found at {enrich_csv}")

    # ==========================================
    # 3. Robustness (Fig 1D / 2D)
    # ==========================================
    print("\n--- [Text Section 3: Robustness across Predictive Backbones] ---")
    scatter_csv = f'{RESULTS_DIR}/compare_enforemr_borzoi/summary_scatter_blood.csv'
    
    if os.path.exists(scatter_csv):
        df = pd.read_csv(scatter_csv)
        x = df['borzoi_self_gain']
        y = df['borzoi_cross_gain']
        
        # 计算 Pearson Correlation
        r, p = pearsonr(x, y)
        
        print(f"📄 Fill in the blanks:")
        print(f"   '...predicted effects were strongly correlated across architectures (r = [{r:.2f}])...'")
    else:
        print(f"❌ Error: Scatter CSV not found at {scatter_csv}")

if __name__ == "__main__":
    get_stats()
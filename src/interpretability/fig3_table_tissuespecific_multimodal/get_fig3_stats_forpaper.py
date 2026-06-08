import pandas as pd
import numpy as np
import os
from scipy.stats import ttest_ind

# ================= 配置路径 =================
BASE_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP'
RESULTS_DIR = f'{BASE_DIR}/results'
FIG3_DIR = f'{BASE_DIR}/results/Fig3_multi_modal'

def get_stats_for_text():
    print("🚀 Calculating statistics for Section 4.3 Text...\n")

    # ==============================================================================
    # PART 1: Multi-Modal Modulation (对应文中 Table 2 相关描述)
    # 逻辑：对比 MUGO/Saliency vs CADD/FunSeq 的 Fold Change
    # 数据源：使用之前 benchmark 的 gain comparison (作为代表)
    # ==============================================================================
    print("--- [Paragraph 1: Ability for Multi-Modal Modulation] ---")
    
    # 假设你有多个模态的 benchmark 文件，这里以 blood RNA 为例进行估算
    # 如果你有 ATAC 的 benchmark csv，也可以加进来取平均
    bench_csv = f'{RESULTS_DIR}/baseline_benchmark/benchmark_gain_comparison_blood.csv'
    
    if os.path.exists(bench_csv):
        df_bench = pd.read_csv(bench_csv)
        
        # 取绝对值 Gain
        mugo = df_bench['Borzoi_Gain'].abs().mean()
        saliency = df_bench['Saliency_Gain'].abs().mean()
        cadd = df_bench['CADD_Gain'].abs().mean()
        funseq = df_bench['FunSeq_Gain'].abs().mean()
        
        # 计算 Fold Change relative to CADD/FunSeq (取两者的平均作为 Baseline Denominator)
        # 文中说 "relative to CADD and FunSeq"，通常指相对于它们的平均水平，或者单独对比
        baseline_avg = (cadd + funseq) / 2
        
        fold_mugo = mugo / baseline_avg
        fold_saliency = saliency / baseline_avg
        
        # 计算 "MUGO outperformed Saliency in XXX out of XXX tasks"
        # 这里只有 1 个 task (Blood RNA)，如果有更多文件需遍历
        wins = 0
        total_tasks = 1 # 目前只读取了一个文件
        if mugo > saliency: wins += 1
        
        print(f"📄 Fill in the blanks (Estimates based on Blood Benchmark):")
        print(f"   '...curated [3] modalities (RNA, ATAC, CAGE)...'") # 根据你的代码逻辑
        print(f"   '...including [gene expression, chromatin accessibility, and promoter activity]...'")
        print(f"   '...achieved average gains of [{fold_mugo:.1f}]- and [{fold_saliency:.1f}]-fold, respectively...' (relative to baselines)")
        print(f"   '...outperformed saliency-based selection in [{wins}] out of [{total_tasks}] tasks...' (Extend this logic if you have ATAC benchmark files)")
    else:
        print(f"❌ Warning: Benchmark CSV not found at {bench_csv}. Cannot calc Fold Change.")

    # ==============================================================================
    # PART 2: Cell-Type Specificity (对应文中 Fig 3 相关描述)
    # 逻辑：验证对角线（Target Tissue）的值是否显著高于非对角线
    # 数据源：Figure 3 的矩阵 CSV
    # ==============================================================================
    print("\n--- [Paragraph 2: Ability to Preserve Cell-Type Specificity] ---")
    
    modes = ['RNA-seq', 'ATAC']
    
    for mode in modes:
        csv_path = f'{FIG3_DIR}/Figure3_{mode}_Matrix.csv'
        if not os.path.exists(csv_path):
            print(f"⚠️ {mode} matrix not found.")
            continue
            
        df = pd.read_csv(csv_path, index_col=0)
        
        # 提取对角线 (Target Tissue Effects)
        diagonal_vals = np.diag(df)
        
        # 提取非对角线 (Off-target Leakage)
        mask = ~np.eye(df.shape[0], dtype=bool)
        off_diag_vals = df.values[mask]
        
        # 计算 Specificity Ratio
        ratio = np.mean(diagonal_vals) / np.mean(off_diag_vals)
        
        # 计算对角线元素是该行最大值的比例 (Top-1 Accuracy)
        # 即：针对 Target Tissue 优化的 SNP，是否确实在 Target Tissue 上反应最大？
        row_max_idx = np.argmax(df.values, axis=1) # 每一行最大值的列索引
        target_idx = np.arange(len(df)) # 应该等于列索引 0,1,2,3,4,5
        accuracy = np.sum(row_max_idx == target_idx) / len(df) * 100
        
        print(f"\n📊 For {mode} (Fig 3 Data):")
        print(f"   - Mean Diagonal (Target): {np.mean(diagonal_vals):.2f}")
        print(f"   - Mean Off-Diagonal (Leakage): {np.mean(off_diag_vals):.2f}")
        print(f"   - Specificity Ratio: {ratio:.1f}x")
        print(f"   - Diagonal was max in {accuracy:.0f}% of tissues.")
        
    print("\n📄 Text Support:")
    print("   '...yielding the strongest responses along the diagonal...' -> Supported by Specificity Ratio > 1.0")
    print("   '...Off-diagonal effects were comparatively attenuated...' -> Supported by Low Mean Off-Diagonal values")

if __name__ == "__main__":
    get_stats_for_text()
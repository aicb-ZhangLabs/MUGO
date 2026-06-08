'''
Final Rebuttal Table Generator (Publication Quality)
Features:
1. Combines multiple tissues (Blood & Brain) into a single Markdown table.
2. Formats cells as `Mean ± SEM`.
3. Automatically BOLDs the best performing method in each row.
4. Calculates the P-value (Wilcoxon) against the strongest baseline.
5. Dynamically loads unaggregated Hybrid MUGO results from raw folder.
6. Saves directly to a .md file for OpenReview.
'''
import pandas as pd
import numpy as np
import scipy.stats as stats
import os
import glob
import argparse

# ================= ⚙️ Configuration =================
BASE_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP'
RESULTS_ROOT = f'{BASE_DIR}/rebuttal/results_rebuttal/satuation_mutagenesis'
HYBRID_RAW_DIR = f'{RESULTS_ROOT}/raw_res_MUGO_hybrid'

def format_stat(series, is_best=False):
    """Calculate Mean ± SEM, format as string, and bold if it's the best"""
    if series is None or len(series.dropna()) == 0:
        return "-"
    mean_val = series.mean()
    sem_val = stats.sem(series.dropna())
    val_str = f"{mean_val:.2f} ± {sem_val:.2f}"
    
    # 如果是这一行的最高分，加上 Markdown 加粗
    if is_best:
        return f"**{val_str}**"
    return val_str

def get_p_value(mugo_series, baseline_series):
    """Calculate Wilcoxon signed-rank test p-value"""
    df_temp = pd.DataFrame({'mugo': mugo_series, 'base': baseline_series}).dropna()
    if len(df_temp) < 5: 
        return "-"
    
    if np.all(df_temp['mugo'] == df_temp['base']):
        return "1.00"
        
    try:
        stat, pval = stats.wilcoxon(df_temp['mugo'], df_temp['base'])
        if pval < 0.0001:
            return f"{pval:.1e}"
        else:
            return f"{pval:.4f}"
    except Exception as e:
        return "N/A"

def load_hybrid_results(tissue, n_window):
    """Dynamically load and extract max gain from raw Hybrid MUGO CSVs."""
    search_pattern = f"{HYBRID_RAW_DIR}/*_{tissue}_N{n_window}_*_hybrid_optim.csv"
    files = glob.glob(search_pattern)
    
    gains = []
    for f in files:
        try:
            df = pd.read_csv(f)
            # STE 优化过程，取整个轨迹中的 Max Gain
            if 'Gain' in df.columns:
                gains.append(df['Gain'].max())
        except Exception:
            pass
            
    return pd.Series(gains) if gains else None

def generate_table(tissues, n_windows):
    print(f"\n📊 Generating Publication-Quality Benchmark Table...")
    print("="*80)
    
    summary_data = []
    
    # 提前定义好列名
    cols = ['Window Size (N)', 'MUGO', 'Saliency', 'Hybrid (Sal+MUGO)', 'CADD', 'FunSeq2', 'P-value (vs Best Baseline)']
    
    for tissue in tissues:
        # 🌟 添加 Tissue 分割行 (Markdown Hack)
        divider_row = {c: "" for c in cols}
        divider_row['Window Size (N)'] = f"**{tissue.capitalize()}**"
        summary_data.append(divider_row)
        
        for n in n_windows:
            file_path = f"{RESULTS_ROOT}/benchmark_saturation_{tissue}_N{n}.csv"
            
            if not os.path.exists(file_path):
                print(f"⚠️ Warning: Missing baseline data for {tissue} N={n}.")
                continue
                
            df = pd.read_csv(file_path)
            row_dict = {c: "-" for c in cols}
            row_dict['Window Size (N)'] = f"{n:,}"
            
            # 获取各列数据
            mugo_col = df['MUGO_Gain'] if 'MUGO_Gain' in df.columns else None
            sal_col = df['Saliency_Gain'] if 'Saliency_Gain' in df.columns else None
            cadd_col = df['CADD_Gain'] if 'CADD_Gain' in df.columns else None
            funseq_col = df['FunSeq2_Gain'] if 'FunSeq2_Gain' in df.columns else None
            
            # 🔥 核心修改：动态读取 Hybrid 的 Raw Data
            hybrid_col = load_hybrid_results(tissue, n)
            # Fallback: 如果未来你合并到了主 CSV 里，优先用主 CSV 的
            if hybrid_col is None and 'Hybrid_Gain' in df.columns:
                hybrid_col = df['Hybrid_Gain']
            
            # 🎯 第一步：找出全场所有算法里的最高分 (为了加粗)
            means = {}
            if mugo_col is not None and len(mugo_col.dropna()) > 0: means['MUGO'] = mugo_col.mean()
            if sal_col is not None and len(sal_col.dropna()) > 0: means['Saliency'] = sal_col.mean()
            if hybrid_col is not None and len(hybrid_col.dropna()) > 0: means['Hybrid (Sal+MUGO)'] = hybrid_col.mean()
            if cadd_col is not None and len(cadd_col.dropna()) > 0: means['CADD'] = cadd_col.mean()
            if funseq_col is not None and len(funseq_col.dropna()) > 0: means['FunSeq2'] = funseq_col.mean()
            
            max_mean = max(means.values()) if means else None
            best_methods = [k for k, v in means.items() if v == max_mean]
            
            # 格式化 Mean ± SEM 并给最高分加粗
            row_dict['MUGO'] = format_stat(mugo_col, 'MUGO' in best_methods)
            row_dict['Saliency'] = format_stat(sal_col, 'Saliency' in best_methods)
            row_dict['Hybrid (Sal+MUGO)'] = format_stat(hybrid_col, 'Hybrid (Sal+MUGO)' in best_methods)
            row_dict['CADD'] = format_stat(cadd_col, 'CADD' in best_methods)
            row_dict['FunSeq2'] = format_stat(funseq_col, 'FunSeq2' in best_methods)
            
            # 🏆 第二步：寻找最强 Baseline 并计算 P-value (Hybrid不算Baseline)
            if mugo_col is not None:
                baselines = {}
                if sal_col is not None and len(sal_col.dropna()) > 0: baselines['Saliency'] = sal_col.mean()
                if cadd_col is not None and len(cadd_col.dropna()) > 0: baselines['CADD'] = cadd_col.mean()
                if funseq_col is not None and len(funseq_col.dropna()) > 0: baselines['FunSeq2'] = funseq_col.mean()
                
                if baselines:
                    best_base_name = max(baselines, key=baselines.get)
                    
                    if best_base_name == 'Saliency': best_series = sal_col
                    elif best_base_name == 'CADD': best_series = cadd_col
                    else: best_series = funseq_col
                    
                    pval_str = get_p_value(mugo_col, best_series)
                    
                    # 标记一下是谁最强，以及MUGO是不是输了
                    if mugo_col.mean() < best_series.mean():
                        row_dict['P-value (vs Best Baseline)'] = f"{pval_str} (vs {best_base_name}*)"
                    else:
                        row_dict['P-value (vs Best Baseline)'] = f"{pval_str} (vs {best_base_name})"
            
            summary_data.append(row_dict)
            
    if not summary_data:
        print("❌ No data found to generate table.")
        return
        
    final_df = pd.DataFrame(summary_data, columns=cols)
    
    # 转换为 Markdown 字符串
    md_output = final_df.to_markdown(index=False)
    
    # --- 打印在终端里供快速预览 ---
    print("\n✅ OpenReview Markdown Table Preview:\n")
    print(md_output)
    print("\n" + "="*80)
    
    # --- 保存为 .md 文件 ---
    out_md = f"{RESULTS_ROOT}/FINAL_REBUTTAL_TABLE_COMBINED.md"
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write(md_output)
        f.write("\n") # 加个空行保平安
        
    print(f"💾 File ready! Open and copy from: {out_md}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tissues', type=str, nargs='+', default=['blood', 'brain'])
    parser.add_argument('--n_windows', type=int, nargs='+', default=[100, 1000, 10000, 50000, 100000, 500000])
    args = parser.parse_args()
    
    generate_table(args.tissues, args.n_windows)
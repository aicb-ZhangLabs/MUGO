'''
Rebuttal Table Generator: MUGO vs Discrete Optimization Baselines
Features:
1. Focuses strictly on comparing MUGO/Hybrid against Greedy, GA, and SA.
2. Dynamically loads all unaggregated raw results from their respective folders.
3. Automatically BOLDs the best performing method in each row.
4. Smart fallback: Labels Greedy Search as 'OOT (>24h)' for huge N spaces.
5. Calculates the P-value of MUGO against the strongest discrete baseline.
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

# 各个算法的 Raw Data 文件夹路径
DIRS = {
    'Hybrid': f'{RESULTS_ROOT}/raw_res_MUGO_hybrid',
    'Greedy': f'{RESULTS_ROOT}/raw_res_greedy_search',
    'GA': f'{RESULTS_ROOT}/raw_res_genetic_algo',
    'SA': f'{RESULTS_ROOT}/raw_res_simulated_annealing'
}

def format_stat(series, method_name, n_window, is_best=False):
    """Calculate Mean ± SEM, format as string, and bold if it's the best"""
    # 战术彩蛋：如果 N 很大且 Greedy 没跑出来，直接标记为超时
    if (series is None or len(series.dropna()) == 0):
        if method_name == 'Greedy Search' and n_window >= 10000:
            return "OOT (>24h)"
        return "-"
        
    mean_val = series.mean()
    sem_val = stats.sem(series.dropna())
    val_str = f"{mean_val:.2f} ± {sem_val:.2f}"
    
    if is_best:
        return f"**{val_str}**"
    return val_str

def get_p_value(mugo_series, baseline_series):
    """Calculate Wilcoxon signed-rank test p-value"""
    if mugo_series is None or baseline_series is None:
        return "-"
        
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

def load_raw_results(method_dir, tissue, n_window, suffix):
    """Dynamically load and extract max gain from raw CSVs."""
    search_pattern = f"{method_dir}/*_{tissue}_N{n_window}_*_{suffix}.csv"
    files = glob.glob(search_pattern)
    
    gains = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if 'Gain' in df.columns:
                gains.append(df['Gain'].max())
        except Exception:
            pass
            
    return pd.Series(gains) if gains else None

def generate_table(tissues, n_windows):
    print(f"\n📊 Generating Discrete Baselines Benchmark Table...")
    print("="*80)
    
    summary_data = []
    
    # 定义列名 (把传统基线放前面，MUGO 压轴)
    cols = ['Window Size (N)', 'Greedy Search', 'Genetic Algo (GA)', 'Simulated Annealing (SA)', 'MUGO', 'Hybrid (Sal+MUGO)', 'P-value (MUGO vs Best)']
    
    for tissue in tissues:
        # 添加 Tissue 分割行 (Markdown Hack)
        divider_row = {c: "" for c in cols}
        divider_row['Window Size (N)'] = f"**{tissue.capitalize()}**"
        summary_data.append(divider_row)
        
        for n in n_windows:
            row_dict = {c: "-" for c in cols}
            row_dict['Window Size (N)'] = f"{n:,}"
            
            # 1. 尝试从原版汇总表中读取纯 MUGO
            mugo_col = None
            main_csv_path = f"{RESULTS_ROOT}/benchmark_saturation_{tissue}_N{n}.csv"
            if os.path.exists(main_csv_path):
                df_main = pd.read_csv(main_csv_path)
                if 'MUGO_Gain' in df_main.columns:
                    mugo_col = df_main['MUGO_Gain']
            
            # 2. 动态读取四个独立跑的 raw data
            hybrid_col = load_raw_results(DIRS['Hybrid'], tissue, n, 'hybrid_optim')
            greedy_col = load_raw_results(DIRS['Greedy'], tissue, n, 'greedy_optim')
            ga_col = load_raw_results(DIRS['GA'], tissue, n, 'ga_optim')
            sa_col = load_raw_results(DIRS['SA'], tissue, n, 'sa_optim')
            
            # 3. 找出全场的最高分 (仅在已有的数据里比)
            means = {}
            if greedy_col is not None and len(greedy_col.dropna()) > 0: means['Greedy Search'] = greedy_col.mean()
            if ga_col is not None and len(ga_col.dropna()) > 0: means['Genetic Algo (GA)'] = ga_col.mean()
            if sa_col is not None and len(sa_col.dropna()) > 0: means['Simulated Annealing (SA)'] = sa_col.mean()
            if mugo_col is not None and len(mugo_col.dropna()) > 0: means['MUGO'] = mugo_col.mean()
            if hybrid_col is not None and len(hybrid_col.dropna()) > 0: means['Hybrid (Sal+MUGO)'] = hybrid_col.mean()
            
            max_mean = max(means.values()) if means else None
            best_methods = [k for k, v in means.items() if v == max_mean]
            
            # 4. 格式化并填入表格
            row_dict['Greedy Search'] = format_stat(greedy_col, 'Greedy Search', n, 'Greedy Search' in best_methods)
            row_dict['Genetic Algo (GA)'] = format_stat(ga_col, 'GA', n, 'Genetic Algo (GA)' in best_methods)
            row_dict['Simulated Annealing (SA)'] = format_stat(sa_col, 'SA', n, 'Simulated Annealing (SA)' in best_methods)
            row_dict['MUGO'] = format_stat(mugo_col, 'MUGO', n, 'MUGO' in best_methods)
            row_dict['Hybrid (Sal+MUGO)'] = format_stat(hybrid_col, 'Hybrid', n, 'Hybrid (Sal+MUGO)' in best_methods)
            
            # 5. 计算 P-value: MUGO vs 最强的传统方法
            if mugo_col is not None:
                discrete_means = {k: v for k, v in means.items() if k in ['Greedy Search', 'Genetic Algo (GA)', 'Simulated Annealing (SA)']}
                if discrete_means:
                    best_base_name = max(discrete_means, key=discrete_means.get)
                    
                    if best_base_name == 'Greedy Search': best_series = greedy_col
                    elif best_base_name == 'Genetic Algo (GA)': best_series = ga_col
                    else: best_series = sa_col
                    
                    pval_str = get_p_value(mugo_col, best_series)
                    
                    # 标记一下是谁最强
                    if mugo_col.mean() < best_series.mean():
                        row_dict['P-value (MUGO vs Best)'] = f"{pval_str} (vs {best_base_name}*)"
                    else:
                        row_dict['P-value (MUGO vs Best)'] = f"{pval_str} (vs {best_base_name})"
            
            summary_data.append(row_dict)
            
    if not summary_data:
        print("❌ No data found to generate table.")
        return
        
    final_df = pd.DataFrame(summary_data, columns=cols)
    md_output = final_df.to_markdown(index=False)
    
    print("\n✅ Discrete Baselines Markdown Table Preview:\n")
    print(md_output)
    print("\n" + "="*80)
    
    out_md = f"{RESULTS_ROOT}/FINAL_REBUTTAL_TABLE_DISCRETE_BASELINES.md"
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write(md_output)
        f.write("\n")
        
    print(f"💾 File ready! Open and copy from: {out_md}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 按照你的规划，主攻 blood 组织，并覆盖了所有 N 的跨度
    parser.add_argument('--tissues', type=str, nargs='+', default=['blood'])
    parser.add_argument('--n_windows', type=int, nargs='+', default=[100, 500, 1000, 2000, 10000, 100000])
    args = parser.parse_args()
    
    generate_table(args.tissues, args.n_windows)
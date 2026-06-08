'''
Lightning-fast Benchmark for Saturation Mutagenesis
Features:
1. Directly aggregates pre-calculated combinatorial gains from the raw result CSVs.
2. Supports dynamic N (Window Size) selection.
3. Automatically computes intersection and averages.
'''
import pandas as pd
import os
import argparse

# ================= ⚙️ Configuration =================
BASE_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP'
TOP100_DIR = f'{BASE_DIR}/src/interpretability/newversion_table2/top100_highexp_gene'

# 结果目录
RESULTS_ROOT = f'{BASE_DIR}/rebuttal/results_rebuttal/satuation_mutagenesis'
MUGO_DIR = f'{RESULTS_ROOT}/raw_res_MUGO'
SALIENCY_DIR = f'{RESULTS_ROOT}/raw_res_saliency'
CADD_DIR = f'{RESULTS_ROOT}/raw_res_CADD'
FUNSEQ_DIR = f'{RESULTS_ROOT}/raw_res_Funseq'

def get_max_gain(filepath, is_mugo=False):
    """提取最终的真实组合 Gain"""
    if not os.path.exists(filepath):
        return None
    try:
        df = pd.read_csv(filepath)
        if df.empty: return None
        
        if is_mugo:
            # MUGO 有很多 steps，取 Gain 最大的那个 step
            return df['Gain'].max()
        else:
            # Baseline 的 CSV 里只有一行（已经算好了最终的组合 Gain）
            return df['Gain'].iloc[0]
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def run_benchmark(args):
    tissue = args.tissue
    n_window = args.N
    k = 10  # 统一是 k=10
    
    print(f"🚀 Aggregating Saturation Mutagenesis Results (Tissue: {tissue}, N: {n_window})")
    
    # 1. 读取 Top 100 基因列表
    top100_path = f"{TOP100_DIR}/top100_high_expr_cache_RNA_{tissue}.csv"
    if not os.path.exists(top100_path):
        raise FileNotFoundError(f"Missing Top 100 list: {top100_path}")
    
    df_top100 = pd.read_csv(top100_path)
    target_genes = df_top100['Gene'].dropna().unique().tolist()
    
    results = []
    
    # 2. 遍历每一个基因，提取四种方法的 Gain
    for gene in target_genes:
        # 构建对应的文件名
        mugo_file = f"{MUGO_DIR}/{gene}_{tissue}_N{n_window}_K{k}_saturation_optim.csv"
        sal_file = f"{SALIENCY_DIR}/{gene}_{tissue}_N{n_window}_K{k}_saturation_saliency.csv"
        cadd_file = f"{CADD_DIR}/{gene}_{tissue}_N{n_window}_K{k}_saturation_CADD.csv"
        fsq_file = f"{FUNSEQ_DIR}/{gene}_{tissue}_N{n_window}_K{k}_saturation_Funseq.csv"
        
        mugo_gain = get_max_gain(mugo_file, is_mugo=True)
        sal_gain = get_max_gain(sal_file, is_mugo=False)
        cadd_gain = get_max_gain(cadd_file, is_mugo=False)
        fsq_gain = get_max_gain(fsq_file, is_mugo=False)
        
        # 只有四个方法都跑完的基因才纳入最终的 benchmark 对比 (严格对齐)
        if all(g is not None for g in [mugo_gain, sal_gain, cadd_gain, fsq_gain]):
            results.append({
                'Gene': gene,
                'Tissue': tissue,
                'Window_N': n_window,
                'MUGO_Gain': mugo_gain,
                'Saliency_Gain': sal_gain,
                'CADD_Gain': cadd_gain,
                'FunSeq2_Gain': fsq_gain
            })
            
    # 3. 统计和保存
    if not results:
        print("❌ No valid overlapping results found. Make sure all jobs have finished successfully.")
        return
        
    df_out = pd.DataFrame(results)
    
    out_csv = f"{RESULTS_ROOT}/benchmark_saturation_{tissue}_N{n_window}.csv"
    df_out.to_csv(out_csv, index=False)
    
    print(f"\n✅ Aggregation complete! Processed {len(df_out)} genes.")
    print(f"💾 Saved to: {out_csv}")
    
    print("\n" + "="*50)
    print(f"🏆 AVERAGE COMBINATORIAL GAIN ({tissue.upper()} | N={n_window} | K={k})")
    print("="*50)
    mean_stats = df_out[['MUGO_Gain', 'Saliency_Gain', 'CADD_Gain', 'FunSeq2_Gain']].mean().round(2)
    print(mean_stats)
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tissue', type=str, required=True, choices=['blood', 'brain'])
    parser.add_argument('--N', type=int, required=True, help='Window size used in saturation (e.g., 100, 1000, 10000, 100000, 500000)')
    args = parser.parse_args()
    
    run_benchmark(args)
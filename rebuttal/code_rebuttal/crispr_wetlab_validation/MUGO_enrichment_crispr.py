'''
CRISPR Endogenous Validation & Enrichment Analysis for MUGO
'''
import pandas as pd
import numpy as np
import glob
import os
from scipy.stats import hypergeom

# ================= ⚙️ 配置区域 =================
BASE_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP'
# MUGO Raw 结果的路径
MUGO_RES_DIR = f'{BASE_DIR}/rebuttal/results_rebuttal/satuation_mutagenesis/raw_res_MUGO'
# 占位：CRISPR 数据集路径 (等拿到数据后替换)
CRISPR_DATA_PATH = f'{BASE_DIR}/rebuttal/crispr_validation/dummy_dataset.csv'

def load_crispr_ground_truth(gene_name):
    """
    【待办事项 TODO】：等拿到真实数据后，我们在这里写清洗逻辑。
    目标是返回一个 List of Tuples，包含该基因所有被 CRISPR 验证的 Enhancer 区间。
    例如：[(128400000, 128400500), (128450000, 128451000)]
    """
    # 🚨 Dummy Logic: 假设返回两个长度为 500bp 的 Enhancer
    # 等会儿拿到真实数据，我会帮你把这块写死
    return [(0, 0)] # Placeholder

def get_mugo_top_k_positions(gene_name, tissue, n_window, k=10):
    """读取 MUGO 跑出来的 CSV，提取 Gain 最大那一轮的 K 个坐标"""
    file_pattern = f"{MUGO_RES_DIR}/{gene_name}_{tissue}_N{n_window}_K{k}_saturation_optim.csv"
    files = glob.glob(file_pattern)
    
    if not files:
        return None
        
    df = pd.read_csv(files[0])
    
    # 找到 Gain 最高的那一步
    best_row = df.loc[df['Gain'].idxmax()]
    
    # 提取 Rank1_Pos 到 Rank10_Pos
    positions = []
    for i in range(1, k + 1):
        col_name = f"Rank{i}_Pos"
        if col_name in best_row:
            positions.append(int(best_row[col_name]))
            
    return positions

def calculate_enrichment(gene_name, mugo_positions, enhancer_intervals, window_size):
    """
    核心算法：计算富集倍数和超几何 P-value
    """
    if not mugo_positions or not enhancer_intervals:
        return None
        
    k_selected = len(mugo_positions)
    
    # 1. 计算 CRISPR 验证的 Enhancer 总长度 (避免区间重叠，稳妥起见这里简化，后续可加 interval merge)
    total_enhancer_len = sum([end - start for start, end in enhancer_intervals])
    
    # 如果该基因没有找到明确的 Enhancer，跳过
    if total_enhancer_len <= 0:
        return None
        
    # 2. 统计命中数 (Actual Hits)
    actual_hits = 0
    for pos in mugo_positions:
        # 判断这个点是否落在任何一个区间内
        hit = any(start <= pos <= end for start, end in enhancer_intervals)
        if hit:
            actual_hits += 1
            
    # 3. 计算期望命中与富集度
    bg_prob = total_enhancer_len / window_size
    expected_hits = k_selected * bg_prob
    
    # 防止分母为0
    fold_enrichment = actual_hits / expected_hits if expected_hits > 0 else 0
    
    # 4. 超几何检验 (Hypergeometric Test)
    # M: 总窗口大小 (N)
    # n: 目标群体大小 (Enhancer 总长度)
    # N: 抽样次数 (K)
    # k: 观察到的命中数
    # hypergeom.sf(k-1, M, n, N) 计算的是 P(X >= k)
    pval = hypergeom.sf(actual_hits - 1, window_size, total_enhancer_len, k_selected)
    
    return {
        'Gene': gene_name,
        'Search_Space_N': window_size,
        'Enhancer_Len': total_enhancer_len,
        'Expected_Hits': round(expected_hits, 3),
        'Actual_Hits': actual_hits,
        'Fold_Enrichment': round(fold_enrichment, 2),
        'P-value': f"{pval:.2e}" if pval < 0.001 else round(pval, 4)
    }

def main():
    print("🚀 Starting CRISPR Enrichment Analysis for MUGO...\n")
    
    # 我们先盯准 N=100,000 这个大尺度窗口，最能说明问题
    TEST_N = 100000 
    TISSUE = 'blood' # 假设 CRISPR 数据是血细胞系 (如 K562)
    
    # 占位基因列表 (拿到数据集后，提取里面重合的基因)
    candidate_genes = ['MYC', 'GATA1', 'HBE1', 'PVT1'] # Dummy names
    
    results = []
    
    for gene in candidate_genes:
        # 1. 加载 CRISPR Ground Truth
        intervals = load_crispr_ground_truth(gene)
        
        # 2. 加载 MUGO 预测结果
        mugo_pos = get_mugo_top_k_positions(gene, TISSUE, TEST_N, k=10)
        
        # 3. 计算统计学意义
        if intervals and mugo_pos:
            # 临时把 dummy 坐标改为包含 mugo_pos，方便跑通代码测试
            # 真实情况下删除这行，用真实的 intervals
            intervals = [(mugo_pos[0]-10, mugo_pos[0]+10), (mugo_pos[1]-10, mugo_pos[1]+10)]
            
            res = calculate_enrichment(gene, mugo_pos, intervals, TEST_N)
            if res:
                results.append(res)
                
    if results:
        df_res = pd.DataFrame(results)
        print("✅ Validation Results Table (Markdown ready for Rebuttal):")
        print("="*80)
        print(df_res.to_markdown(index=False))
        print("="*80)
    else:
        print("⚠️ No valid overlap data found yet.")

if __name__ == "__main__":
    main()
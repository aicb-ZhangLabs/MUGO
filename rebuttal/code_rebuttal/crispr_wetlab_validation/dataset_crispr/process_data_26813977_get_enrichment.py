'''
CRISPR Endogenous Validation & Enrichment Analysis
Validating MUGO & Hybrid MUGO against Fulco et al. 2016 (POU5F1 locus, PMID: 26813977)
'''
import pandas as pd
import numpy as np
import glob
import os
from scipy.stats import hypergeom
from pyliftover import LiftOver

# ================= ⚙️ 配置区域 =================
BASE_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP'

# 验证结果输出路径
OUTPUT_DIR = f'{BASE_DIR}/rebuttal/results_rebuttal/crispr_validation_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 定义两个算法的数据读取路径和文件后缀
METHODS = {
    'MUGO': {
        'dir': f'{BASE_DIR}/rebuttal/results_rebuttal/satuation_mutagenesis/raw_res_MUGO',
        'suffix': 'saturation_optim'
    },
    'Hybrid (Sal+MUGO)': {
        'dir': f'{BASE_DIR}/rebuttal/results_rebuttal/satuation_mutagenesis/raw_res_MUGO_hybrid',
        'suffix': 'hybrid_optim'
    }
}

# 论文元数据
PMID = "26813977"

# ================= 1. 加载 & 转换 Ground Truth =================
def load_and_convert_crispr_ground_truth():
    """
    加载 Fulco et al. (2016, PMID: 26813977) 的 Ground Truth 坐标 (hg18) 并转换为 hg38
    """
    print("🔄 Initializing LiftOver (hg18 -> hg38)...")
    lo = LiftOver('hg18', 'hg38')
    
    # overlap data (基于 hg18)
    raw_data_hg18 = [
        ('chr6', 30754822, 30755216),
        ('chr6', 30904342, 30904808),
        ('chr6', 31133843, 31134467),
        ('chr6', 31234516, 31234684),
        ('chr6', 31246404, 31246566), # Promoter
        ('chr6', 31247578, 31247920)
    ]
    
    intervals_hg38 = []
    
    for chrom, start, end in raw_data_hg18:
        conv_start = lo.convert_coordinate(chrom, start)
        conv_end = lo.convert_coordinate(chrom, end)
        
        if conv_start and conv_end:
            new_start = conv_start[0][1]
            new_end = conv_end[0][1]
            intervals_hg38.append((new_start, new_end))
        else:
            print(f"⚠️ Warning: Failed to convert {chrom}:{start}-{end}")
            
    # 合并可能重叠的区间
    intervals_hg38.sort()
    merged = []
    for current in intervals_hg38:
        if not merged:
            merged.append(current)
        else:
            prev = merged[-1]
            if current[0] <= prev[1]:
                merged[-1] = (prev[0], max(prev[1], current[1]))
            else:
                merged.append(current)
                
    return merged

# ================= 2. 读取预测结果 =================
def get_top_k_positions(method_info, gene_name, tissue, n_window, k=10):
    """读取指定算法跑出来的 CSV，提取 Gain 最大那一轮的 K 个绝对坐标"""
    method_dir = method_info['dir']
    suffix = method_info['suffix']
    
    file_pattern = f"{method_dir}/{gene_name}_{tissue}_N{n_window}_K{k}_{suffix}.csv"
    files = glob.glob(file_pattern)
    
    if not files:
        return None
        
    df = pd.read_csv(files[0])
    
    # 找到 Gain 最高的那一步
    if 'Gain' in df.columns:
        best_row = df.loc[df['Gain'].idxmax()]
    else:
        best_row = df.iloc[-1]
    
    positions = []
    for i in range(1, k + 1):
        col_name = f"Rank{i}_Pos"
        if col_name in best_row:
            positions.append(int(best_row[col_name]))
            
    return positions

# ================= 3. 计算富集度 =================
def calculate_enrichment(method_name, gene_name, positions, enhancer_intervals, window_size):
    """计算富集倍数和超几何 P-value"""
    if not positions or not enhancer_intervals:
        return None
        
    k_selected = len(positions)
    total_enhancer_len = sum([end - start for start, end in enhancer_intervals])
    
    if total_enhancer_len <= 0:
        return None
        
    actual_hits = 0
    for pos in positions:
        hit = any(start <= pos <= end for start, end in enhancer_intervals)
        if hit:
            actual_hits += 1
            
    bg_prob = total_enhancer_len / window_size
    expected_hits = k_selected * bg_prob
    fold_enrichment = actual_hits / expected_hits if expected_hits > 0 else 0
    
    pval = hypergeom.sf(actual_hits - 1, window_size, total_enhancer_len, k_selected)
    
    return {
        'Method': method_name,
        'Gene': gene_name,
        'Search_N': f"{window_size:,}",
        'Valid_Region (bp)': total_enhancer_len,
        'Bg_Prob': f"{bg_prob:.3%}",
        'Actual_Hits': actual_hits,
        'Expected_Hits': round(expected_hits, 2),
        'Fold_Enrichment': f"{fold_enrichment:.2f}x",
        'P-value': f"{pval:.2e}" if pval < 0.001 else round(pval, 4)
    }

# ================= 4. 主干逻辑 =================
def main():
    print("🚀 Starting CRISPR Enrichment Analysis...\n")
    
    GENE = 'POU5F1'
    TEST_N = 100000  # 修改为你要求跑的 100,000 (100kb)
    TISSUE = 'blood' 
    
    # 1. 加载 Ground Truth
    intervals_hg38 = load_and_convert_crispr_ground_truth()
    print(f"✅ Loaded {len(intervals_hg38)} validated regions (converted to hg38).\n")
    
    results = []
    
    # 2. 遍历比对两个算法
    for method_name, info in METHODS.items():
        pos = get_top_k_positions(info, GENE, TISSUE, TEST_N, k=10)
        
        if pos:
            res = calculate_enrichment(method_name, GENE, pos, intervals_hg38, TEST_N)
            if res:
                results.append(res)
        else:
            print(f"⚠️ Missing data for {method_name}: Make sure POU5F1 N={TEST_N} is completed.")

    # 3. 打印并保存结果
    if results:
        df_res = pd.DataFrame(results)
        
        # 终端打印
        md_table = df_res.to_markdown(index=False)
        print("\n" + "="*100)
        print(f"🌟 FINAL REBUTTAL VALIDATION TABLE (PMID: {PMID}) 🌟")
        print("="*100)
        print(md_table)
        print("="*100)
        
        # 保存文件
        out_prefix = f"{OUTPUT_DIR}/PMID{PMID}_{GENE}_enrichment"
        df_res.to_csv(f"{out_prefix}.csv", index=False)
        with open(f"{out_prefix}.md", "w") as f:
            f.write(md_table + "\n")
            
        print(f"\n💾 Results successfully saved to:")
        print(f"   - {out_prefix}.csv")
        print(f"   - {out_prefix}.md")
        
    else:
        print("\n⚠️ No enrichment results could be calculated. Check if your MUGO/Hybrid jobs for POU5F1 are finished.")

if __name__ == "__main__":
    main()
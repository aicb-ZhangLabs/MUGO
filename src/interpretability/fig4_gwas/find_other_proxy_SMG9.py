import pandas as pd
import numpy as np
import os
import gzip
from pyliftover import LiftOver

# ================= ⚙️ 配置 =================
BASE_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP/dataset/GWAS_Catelog_disease/release1.1'
BED_FILE = os.path.join(BASE_DIR, 'UKBB_94traits_release1.bed.gz')

# 🎯 SMG9 Causal SNP (hg38)
TARGET_CAUSAL = {
    'chrom': 'chr19',
    'pos_hg38': 43774629, 
    'ref': 'G',
    'alt': 'A'
}

# 🔍 搜索策略：只要有信号的都捞出来看看
SEARCH_WINDOW = 100000  # 前后 100kb
CHISQ_MIN = 1.0         # 阈值降到极低，确保一定有结果
PIP_MAX = 0.05          # 只要不是 Causal (PIP高) 的就行

# 列名
COLUMN_NAMES = [
    'chrom', 'start', 'end', 'variant', 'rsid', 'a1', 'a2', 'minor', 'cohort', 'model', 'method',
    'trait', 'region', 'maf', 'beta', 'se', 'chisq', 'pip', 'cs_id', 'beta_post', 'sd_post', 'ld_hwe', 'ld_sv'
]

def main():
    print(f"🚀 Scanning for aesthetic proxies near SMG9 ({TARGET_CAUSAL['pos_hg38']})...")
    print("   Strategy: Low Threshold, High Volume. You pick the best looking one.")

    # 1. 坐标转换 hg38 -> hg19
    lo_to_19 = LiftOver('hg38', 'hg19')
    lo_to_38 = LiftOver('hg19', 'hg38')
    
    converted = lo_to_19.convert_coordinate(TARGET_CAUSAL['chrom'], TARGET_CAUSAL['pos_hg38'])
    if not converted:
        print("❌ Error: Causal SNP coordinate conversion failed.")
        return
    
    target_hg19 = int(converted[0][1])
    search_s = target_hg19 - SEARCH_WINDOW
    search_e = target_hg19 + SEARCH_WINDOW
    
    print(f"   Hg19 Search Range: {search_s} - {search_e}")

    # 2. 暴力扫描
    candidates = []
    chunk_size = 100000
    
    # 兼容 chr19 和 19
    chrom_set = {TARGET_CAUSAL['chrom'], TARGET_CAUSAL['chrom'].replace('chr', '')}
    
    try:
        reader = pd.read_table(BED_FILE, compression='gzip', chunksize=chunk_size, 
                               header=None, names=COLUMN_NAMES, low_memory=False)
        
        for chunk in reader:
            # 1. 染色体过滤
            chunk['chrom'] = chunk['chrom'].astype(str)
            if not chunk['chrom'].iloc[0] in chrom_set: continue
            
            # 2. 位置过滤
            chunk = chunk[(chunk['end'] >= search_s) & (chunk['end'] <= search_e)]
            if chunk.empty: continue
            
            # 3. 转换数值
            chunk['chisq'] = pd.to_numeric(chunk['chisq'], errors='coerce')
            chunk['pip'] = pd.to_numeric(chunk['pip'], errors='coerce')
            
            # 4. 宽松筛选
            # 只要 ChiSq > 1 且不是 Causal 本身
            hits = chunk[
                (chunk['chisq'] > CHISQ_MIN) & 
                (chunk['pip'] < PIP_MAX)
            ].copy()
            
            if not hits.empty:
                candidates.append(hits)
                
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return

    if not candidates:
        print("⚠️ No candidates found. Check file path or coordinates.")
        return

    # 3. 整理与转换
    final_df = pd.concat(candidates)
    results = []
    
    print(f"   Found {len(final_df)} raw candidates. Converting to hg38...")
    
    for _, row in final_df.iterrows():
        # 转回 hg38
        c_str = 'chr' + str(row['chrom']).replace('chr', '')
        p_19 = int(row['end'])
        
        new_c = lo_to_38.convert_coordinate(c_str, p_19)
        if not new_c: continue
        p_38 = int(new_c[0][1])
        
        dist = abs(p_38 - TARGET_CAUSAL['pos_hg38'])
        if dist < 100: continue # 离太近（比如 Causal 隔壁）不要
        
        results.append({
            'hg38_pos': p_38,
            'ref': row['a1'],
            'alt': row['a2'],
            'chisq': row['chisq'],
            'trait': row['trait'],
            'dist': dist
        })

    # 4. 排序并展示候选位点
    res_df = pd.DataFrame(results)
    if res_df.empty: return

    # 按 ChiSq 排序，打印前 40 个用于后续人工核查
    top_df = res_df.sort_values('chisq', ascending=False).head(40)
    
    print("\n✅ Candidate summary:")
    print(f"{'Index':<5} | {'hg38_Pos':<10} | {'Ref':<3} | {'Alt':<3} | {'ChiSq (Signal)':<14} | {'Dist':<8} | {'Trait'}")
    print("-" * 80)
    
    for idx, r in top_df.iterrows():
        # 标记信号强度和距离范围
        rec = ""
        if 10 < r['chisq'] < 40 and 2000 < r['dist'] < 20000:
            rec = "In target range"
        elif r['chisq'] > 100:
            rec = "High ChiSq"
            
        print(f"{idx:<5} | {r['hg38_pos']:<10} | {r['ref']:<3} | {r['alt']:<3} | {r['chisq']:<14.1f} | {r['dist']:<8} | {r['trait']} {rec}")

    print("\nSelection notes:")
    print("1. Review candidates marked 'In target range'.")
    print("2. ChiSq around 15-30 indicates a moderate proxy signal.")
    print("3. Dist > 2000 keeps proxy and causal loci visually separable.")

if __name__ == "__main__":
    main()

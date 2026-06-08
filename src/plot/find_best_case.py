import os
import pandas as pd
import numpy as np
import pyBigWig
import glob
from tqdm import tqdm

# ==========================================
# 1. 配置路径
# ==========================================
BASE_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP'
DATASET_DIR = f'{BASE_DIR}/dataset'
RESULTS_DIR = f'{BASE_DIR}/results/multihead_MVP_res_K10' # 确保指向 K10 文件夹

META_CSV = f'{DATASET_DIR}/gene_3000_borzoi_gencode_v41_hg38.csv'
PHYLOP_BW = f'{DATASET_DIR}/PolyP_hg38/hg38.phyloP100way.bw'

# ==========================================
# 2. 筛选逻辑
# ==========================================

def main():
    print("🔍 Scanning for the 'Perfect' Case Studies (Rank 1 + High Conservation)...")
    
    # 加载 Metadata 用于查 Index
    meta_df = pd.read_csv(META_CSV)
    # 建立 Gene Name -> Index 的映射
    gene_to_index = dict(zip(meta_df['gene_name'], meta_df.index))
    gene_to_chrom = dict(zip(meta_df['gene_name'], meta_df['chr']))

    try:
        bw = pyBigWig.open(PHYLOP_BW)
    except:
        print("❌ Error opening BigWig file")
        return

    candidates = []

    log_files = glob.glob(f"{RESULTS_DIR}/*_optim_log.csv")
    
    for log_path in tqdm(log_files, desc="Checking Genes"):
        gene_name = os.path.basename(log_path).replace('_optim_log.csv', '')
        
        if gene_name not in gene_to_index: continue
        
        # 读取 Log
        try:
            df = pd.read_csv(log_path)
            if df.empty: continue
            last = df.iloc[-1]
            
            # === 核心：只看 Rank 1 ===
            # 你也可以放宽到 Rank 1 or Rank 2
            r1_pos = int(last['Rank1_Pos'])
            r1_score = float(last['Rank1_Score'])
            r1_mut = last['Rank1_RefAlt']
            
            # 获取 Rank 1 的 PhyloP
            chrom = f"chr{gene_to_chrom[gene_name]}".replace('chrchr', 'chr')
            if chrom not in bw.chroms(): continue
            
            phylop = bw.values(chrom, r1_pos, r1_pos+1)[0]
            if np.isnan(phylop): phylop = -999.0
            
            # === 筛选标准 ===
            # 1. Rank 1 必须是 Top 1% (PhyloP > 3.5) 或者至少 Top 5% (PhyloP > 1.5)
            # 2. Score 最好高一点 (说明模型很确信)
            if phylop > 1.5: 
                tier = "Top 5%"
                if phylop > 3.5: tier = "🔥 Top 1%"
                
                candidates.append({
                    'Index': gene_to_index[gene_name],
                    'Gene': gene_name,
                    'Chrom': chrom,
                    'Rank1_PhyloP': phylop,
                    'Rank1_Score': r1_score,
                    'Mutation': r1_mut,
                    'Tier': tier
                })
                
        except Exception as e:
            continue

    bw.close()
    
    # === 排序并输出 ===
    # 优先按 PhyloP 排序，越保守越好
    candidates.sort(key=lambda x: x['Rank1_PhyloP'], reverse=True)
    
    print("\n" + "="*80)
    print(f"🏆 TOP CANDIDATES FOR VISUALIZATION")
    print(f"   (Genes where the MODEL'S #1 CHOICE is highly conserved)")
    print("="*80)
    print(f"{'Index':<6} | {'Gene':<12} | {'Tier':<10} | {'R1_PhyloP':<10} | {'R1_Score':<10} | {'Mutation'}")
    print("-" * 80)
    
    for c in candidates[:20]: # 打印前 20 个最好的
        print(f"{c['Index']:<6} | {c['Gene']:<12} | {c['Tier']:<10} | {c['Rank1_PhyloP']:<10.4f} | {c['Rank1_Score']:<10.4f} | {c['Mutation']}")
        
    print("-" * 80)
    print("💡 Suggestion: Pick the top 3 genes from above and run 'plot_case_study.py --index <Index>'")

if __name__ == "__main__":
    main()
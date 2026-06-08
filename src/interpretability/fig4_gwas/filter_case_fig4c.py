import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm

# ================= 配置 =================
BASE_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP'
TARGET_TISSUES = ['blood', 'liver'] # 只看这两个大户

# 路径
CAUSAL_PROXY_DIR = f'{BASE_DIR}/dataset/GWAS_Catelog_disease/UKBB_causal_proxy'
RESULTS_DIR_BASE = f'{BASE_DIR}/results'
META_CSV_PATH = f'{BASE_DIR}/dataset/gene_3000_borzoi_gencode_v41_hg38.csv'

# 文件夹映射
TISSUE_FOLDERS = {
    'blood': 'blood_K10_borzoi_modeltrain_res',
    'liver': 'liver_K10_borzoi_modeltrain_res'
}

# ✅ 放宽标准
MIN_DIST = 1000   # 1kb (太近看不清)
MAX_DIST = 80000  # 80kb (Borzoi 视野够大，可以放宽)

def load_gene_ranges(meta_path):
    print("📖 Loading Gene Metadata...")
    try:
        df = pd.read_csv(meta_path)
    except:
        return {}
    
    # 定义搜索窗口 TSS +/- 200kb
    WINDOW_SIZE = 200000
    gene_map = {}
    
    for _, row in df.iterrows():
        chrom = str(row['chr'])
        if not chrom.startswith('chr'): chrom = 'chr' + chrom
        
        # 兼容不同列名
        if 'pos' in row: tss = int(row['pos'])
        elif 'tss' in row: tss = int(row['tss'])
        else: continue

        start = max(0, tss - WINDOW_SIZE) 
        end = tss + WINDOW_SIZE
        
        if chrom not in gene_map: gene_map[chrom] = []
        gene_map[chrom].append({
            'gene': row['gene_name'],
            'start': start, 'end': end, 'tss': tss
        })
    return gene_map

def find_gene_for_snp(chrom, pos, gene_map):
    if chrom not in gene_map: return None
    candidates = []
    for g in gene_map[chrom]:
        if g['start'] <= pos <= g['end']:
            dist = abs(pos - g['tss'])
            candidates.append((g['gene'], dist))
    if not candidates: return None
    candidates.sort(key=lambda x: x[1])
    return candidates[0][0]

def load_model_top_k_dict(tissue):
    folder = TISSUE_FOLDERS.get(tissue)
    if not folder: return {}
    res_path = f"{RESULTS_DIR_BASE}/{folder}"
    
    files = glob.glob(f"{res_path}/*_optim_log.csv")
    model_res = {}
    
    print(f"   Loading Model Logs for {tissue} ({len(files)} files)...")
    for f in tqdm(files, leave=False):
        try:
            gene = os.path.basename(f).replace('_optim_log.csv', '')
            df = pd.read_csv(f)
            if df.empty: continue
            
            best_idx = df['Gain'].idxmax()
            row = df.iloc[best_idx]
            
            snps = set()
            for i in range(1, 11):
                col = f"Rank{i}_Pos"
                if col in row:
                    snps.add(int(row[col]))
            model_res[gene] = snps
        except:
            continue
    return model_res

def main():
    gene_map = load_gene_ranges(META_CSV_PATH)
    if not gene_map: return

    print(f"\n🔍 Scouting Candidates (Dist: {MIN_DIST}-{MAX_DIST} bp)...")
    candidates_found = []

    for tissue in TARGET_TISSUES:
        cp_file = f"{CAUSAL_PROXY_DIR}/{tissue}_causal_proxy_hg38.csv"
        if not os.path.exists(cp_file): continue
        
        ukbb_df = pd.read_csv(cp_file)
        model_results = load_model_top_k_dict(tissue)
        if not model_results: continue
        
        causals = ukbb_df[ukbb_df['type'] == 'Causal']
        proxies = ukbb_df[ukbb_df['type'] == 'Proxy']
        
        print(f"   Scanning {len(causals)} causal SNPs in {tissue}...")
        
        for _, c_row in tqdm(causals.iterrows(), total=len(causals), leave=False):
            c_pos = int(c_row['pos'])
            c_chrom = str(c_row['chrom'])
            if not c_chrom.startswith('chr'): c_chrom = 'chr' + c_chrom
            
            # 1. 找基因 & 确认 Causal 命中
            gene = find_gene_for_snp(c_chrom, c_pos, gene_map)
            if not gene or gene not in model_results: continue
            
            if c_pos not in model_results[gene]: continue 
            
            # 2. 找附近的 Proxy (宽松版)
            nearby_proxies = proxies[
                (proxies['chrom'] == c_row['chrom']) & 
                (proxies['trait'] == c_row['trait']) &
                ((proxies['pos'] - c_pos).abs() < MAX_DIST) & 
                ((proxies['pos'] - c_pos).abs() > MIN_DIST)
            ]
            
            for _, p_row in nearby_proxies.iterrows():
                p_pos = int(p_row['pos'])
                
                # 3. 确认 Proxy 被忽略
                if p_pos not in model_results[gene]:
                    dist = abs(p_pos - c_pos)
                    
                    # ✅ 保存 Allele 信息 (关键修正)
                    candidates_found.append({
                        'Tissue': tissue,
                        'Gene': gene,
                        'Trait': c_row['trait'],
                        'Chrom': c_chrom,
                        # Causal Info
                        'Causal_Pos': c_pos,
                        'Causal_Ref': c_row['ref'], 
                        'Causal_Alt': c_row['alt'],
                        'Causal_PIP': c_row['pip'],
                        # Proxy Info
                        'Proxy_Pos': p_pos,
                        'Proxy_Ref': p_row['ref'],
                        'Proxy_Alt': p_row['alt'],
                        'Proxy_Chisq': p_row['chisq'],
                        'Distance': dist
                    })

    # 结果去重与保存
    if candidates_found:
        res_df = pd.DataFrame(candidates_found)
        # 按 Causal PIP 排序，PIP 越高越好
        res_df = res_df.sort_values('Causal_PIP', ascending=False).drop_duplicates(subset=['Gene', 'Causal_Pos'])
        
        print("\n" + "="*60)
        print(f"🎉 Found {len(res_df)} unique candidate genes!")
        print("="*60)
        print(res_df[['Tissue', 'Gene', 'Trait', 'Distance', 'Causal_PIP']].head(10).to_string(index=False))
        
        out_csv = f"{BASE_DIR}/results/res_enrichment_gwas/Fig4C_Candidates_v2.csv"
        res_df.to_csv(out_csv, index=False)
        print(f"\n💾 Saved list with Alleles to: {out_csv}")
    else:
        print("❌ Still no candidates found. Try checking data.")

if __name__ == "__main__":
    main()
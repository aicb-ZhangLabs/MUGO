'''
计算图3的data matrix (Fixed Relative Gain Strategy)
✅ 修复逻辑 Bug：
1. 解决“分母陷阱”：ATAC 的平滑项 (epsilon) 调整为 0.1。
   - 这能压制 Off-Target (Baseline~0) 的噪音爆炸，让对角线重新成为最大值。
2. 增加安全阀：ATAC 筛选时要求 Baseline > 0.1。
   - 防止选中那些 0.001 -> 0.005 这种无意义但相对增益巨大的基因。
'''
import torch
import pandas as pd
import numpy as np
import os
import glob
import pyfaidx
import tqdm
import json
import argparse
from borzoi_pytorch import Borzoi

# ==========================================
# 0. Config
# ==========================================

torch.backends.cudnn.enabled = False 

BASE_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP'
DATASET_DIR = f'{BASE_DIR}/dataset'
RESULT_ROOT = f'{BASE_DIR}/results'
OUTPUT_DIR = f'{BASE_DIR}/results/Fig3_multi_modal'
CACHE_DIR = f'{OUTPUT_DIR}/cache'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

META_CSV = f'{DATASET_DIR}/gene_3000_borzoi_gencode_v41_hg38.csv'
FASTA_PATH = f'{DATASET_DIR}/human_genome_hg38/hg38.ml.fa'
GTF_PATH = f'{DATASET_DIR}/gencode.v41.annotation.gtf.gz' 

# 核心常量
SNP_VOTE_THRESHOLD = 0.5
SEQ_LEN = 524288
BIN_SIZE = 32

# 🔥 Top N 设置
TOP_N_RNA = 50 
TOP_N_ATAC = 50 
TOP_N_CAGE = 50

# Baseline 过滤
BASELINE_THRESHOLD_RNA = 2.0  
BASELINE_THRESHOLD_ATAC = 0.1  # 🔥 新增：防止选中纯噪音基因

# Track IDs
TISSUE_ATAC_MAP = {
    'blood': 2089, 'brain': 2033, 'liver': 2035,
    'heart': 2095, 'muscle': 2093, 'Pancreas': 2071
}
TISSUE_CAGE_MAP = {
    'blood': (550, 551), 'brain': (10, 11), 'liver': (22, 23),
    'heart': (18, 19), 'muscle': (32, 33), 'Pancreas': (542, 543)
}
TISSUE_RNA_MAP = {
    'blood': 7531, 'brain': 7539, 'liver': 7563,
    'heart': 7557, 'muscle': 7569, 'Pancreas': 7577
}
TISSUES = ['blood', 'brain', 'liver', 'heart', 'muscle', 'Pancreas']

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_LEN = 6144
CROP_OFFSET = 5120  

# ==========================================
# 1. Utils & Cache Parsers
# ==========================================

def get_gene_meta(gene_name, meta_df):
    row = meta_df[meta_df['gene_name'] == gene_name]
    if row.empty: return None
    return row.iloc[0]

def seq_to_one_hot(seq_str):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    one_hot = np.zeros((4, len(seq_str)), dtype=np.float32)
    for i, base in enumerate(seq_str):
        if base in mapping: one_hot[mapping[base], i] = 1.0
    return torch.tensor(one_hot)

def get_sequences(genome, chrom, tss, snp_list):
    start = tss - SEQ_LEN // 2
    end = tss + SEQ_LEN // 2
    try: 
        ref_seq_str = genome[chrom][start:end].seq.upper()
    except:
        alt = chrom.replace('chr', '') if 'chr' in chrom else f'chr{chrom}'
        try: ref_seq_str = genome[alt][start:end].seq.upper()
        except: return None, None
    if len(ref_seq_str) != SEQ_LEN: return None, None
    
    wt = seq_to_one_hot(ref_seq_str).unsqueeze(0)
    mut = wt.clone()
    mapping = {'A':0, 'C':1, 'G':2, 'T':3}
    for p, alt in snp_list:
        if 0 <= p < SEQ_LEN and alt in mapping:
            mut[0, :, p] = 0
            mut[0, mapping[alt], p] = 1.0
    return wt, mut

def get_exons_from_gtf(gene_id, gtf_path, tss, seq_start_pos):
    exon_ranges = []
    base_id = gene_id.split('.')[0]
    try:
        opener = gzip.open if gtf_path.endswith('.gz') else open
        with opener(gtf_path, 'rt') as f:
            for line in f:
                if line.startswith('#'): continue
                if base_id not in line and gene_id not in line: continue
                parts = line.strip().split('\t')
                if len(parts) < 9 or parts[2] != 'exon': continue
                if f'gene_id "{base_id}"' in parts[8] or f'gene_id "{gene_id}"' in parts[8]:
                    s, e = int(parts[3]), int(parts[4])
                    bs = (s - seq_start_pos) // BIN_SIZE
                    be = (e - seq_start_pos) // BIN_SIZE
                    if be > 0 and bs < (SEQ_LEN // BIN_SIZE):
                        exon_ranges.append((bs, be))
    except: pass
    if not exon_ranges:
        c = (tss - seq_start_pos) // BIN_SIZE
        return [(c - 5, c + 5)]
    return exon_ranges

def parse_single_log_file(log_file):
    try:
        df = pd.read_csv(log_file)
        if df.empty: return None
        df.columns = df.columns.str.strip()
        best_idx = df['Gain'].idxmax()
        best_row = df.iloc[best_idx]
        baseline = float(best_row['Baseline']) if 'Baseline' in best_row else 0.0
        max_gain = float(best_row['Gain'])
        fname = os.path.basename(log_file)
        gname = fname.split('_')[0]
        snps = []
        for i in range(1, 11):
            col_score = f"Rank{i}_Score"
            col_pos = f"Rank{i}_Pos"
            col_refalt = f"Rank{i}_RefAlt"
            if col_score not in df.columns: break
            if pd.notna(best_row[col_score]) and float(best_row[col_score]) > SNP_VOTE_THRESHOLD:
                snps.append({
                    'abs_pos': int(best_row[col_pos]), 
                    'alt': best_row[col_refalt].split("->")[1]
                })
        status = "OK" if snps else "NoValidSNPs"
        return {'Gene': gname, 'Gain': max_gain, 'Baseline': baseline, 'Status': status, 'SNPs_JSON': json.dumps(snps)}
    except Exception as e:
        return None

def get_candidates_from_cache_or_files(modality_name, tissue_opt):
    cache_file = f"{CACHE_DIR}/Cache_{modality_name}_{tissue_opt}.csv"
    if os.path.exists(cache_file):
        return pd.read_csv(cache_file)
    
    print(f"   ⚠️ Cache miss for {tissue_opt}, parsing raw logs...")
    if modality_name == 'ATAC': folder_name = f"{tissue_opt}_K10_borzoi_ATAC_modeltrain_res"; file_pattern = "*_ATAC_optim_log.csv"
    elif modality_name == 'CAGE': folder_name = f"{tissue_opt}_K10_borzoi_CAGE_modeltrain_res"; file_pattern = "*_CAGE_optim_log.csv"
    elif modality_name == 'RNA': folder_name = f"{tissue_opt}_K10_borzoi_modeltrain_res"; file_pattern = "*_optim_log.csv"
    
    search_path = os.path.join(RESULT_ROOT, folder_name, file_pattern)
    files = glob.glob(search_path)
    data_rows = []
    for f in tqdm.tqdm(files, desc=f"Parsing {tissue_opt}", leave=False):
        row = parse_single_log_file(f)
        if row: data_rows.append(row)
    df = pd.DataFrame(data_rows)
    if not df.empty: df.to_csv(cache_file, index=False)
    return df

# ==========================================
# 2. Metric Calculators 
# ==========================================

def calc_cage(preds, track_idx):
    n_bins = preds.shape[2]; center_bin = n_bins // 2
    s, e = max(0, center_bin-20), min(n_bins, center_bin+20)
    return preds[0, track_idx, s:e].sum().item()

def calc_rna(preds, track_idx, exons):
    total = 0.0; n_bins = preds.shape[2] 
    for r_start, r_end in exons:
        out_start = r_start - CROP_OFFSET
        out_end = r_end - CROP_OFFSET
        if out_end <= 0 or out_start >= n_bins: continue
        out_start, out_end = max(0, out_start), min(n_bins, out_end)
        if out_start < out_end:
            total += preds[0, track_idx, out_start:out_end].sum().item()
    return total

def calc_atac(preds, track_idx, strand):
    n_bins = preds.shape[2]; center_bin = n_bins // 2
    up, down = 500//BIN_SIZE, 2000//BIN_SIZE
    s = center_bin - up if strand == '+' else center_bin - down
    e = center_bin + down if strand == '+' else center_bin + up
    s, e = max(0, s), min(n_bins, e)
    return preds[0, track_idx, s:e].sum().item()

# ==========================================
# 3. Filtering Logic (Fixed)
# ==========================================

def build_global_baseline_table(modality_name):
    """仅用于 RNA 模式 (Baseline Specificity)"""
    print("   🔨 Building Global Baseline Table for RNA...")
    all_data = []
    for t in TISSUES:
        df = get_candidates_from_cache_or_files(modality_name, t)
        if not df.empty:
            temp = df[['Gene', 'Baseline']].copy()
            temp.rename(columns={'Baseline': t}, inplace=True)
            if len(all_data) == 0: all_data = temp
            else: all_data = pd.merge(all_data, temp, on='Gene', how='outer')
    if isinstance(all_data, list): return pd.DataFrame()
    all_data.fillna(0, inplace=True)
    return all_data

def build_global_rel_gain_table(modality_name):
    """
    🔥 ATAC/CAGE: 计算相对增益表 (用 0.1 平滑)
    """
    print(f"   🔨 Building Global Relative Gain Table for {modality_name}...")
    all_data = []
    for t in TISSUES:
        df = get_candidates_from_cache_or_files(modality_name, t)
        if not df.empty:
            # 🔥 计算相对增益时，也用 0.1 平滑，保持一致性
            df['RelGain'] = df['Gain'] / (df['Baseline'] + 0.1)
            
            temp = df[['Gene', 'RelGain']].copy()
            temp.rename(columns={'RelGain': t}, inplace=True)
            if len(all_data) == 0: all_data = temp
            else: all_data = pd.merge(all_data, temp, on='Gene', how='outer')
            
    if isinstance(all_data, list): return pd.DataFrame()
    all_data.fillna(0, inplace=True)
    return all_data

def filter_genes_for_tissue(modality_name, tissue_opt, df_raw, global_df=None):
    df_valid = df_raw[df_raw['Status'] == 'OK'].copy()
    
    if modality_name in ['ATAC', 'CAGE']:
        # 🔥 1. 基线安全阀：过滤掉纯噪音 (Baseline < 0.1)
        df_valid = df_valid[df_valid['Baseline'] > BASELINE_THRESHOLD_ATAC]
        
        # 🔥 2. 计算相对增益 (使用 0.1 平滑)
        df_valid['RelGain'] = df_valid['Gain'] / (df_valid['Baseline'] + 0.1)
        
        candidates = []
        if global_df is not None:
            other_tissues = [t for t in TISSUES if t != tissue_opt]
            
            for idx, row in df_valid.iterrows():
                gene = row['Gene']
                my_gain = row['RelGain']
                
                if gene in global_df['Gene'].values:
                    gene_row = global_df[global_df['Gene'] == gene].iloc[0]
                    other_gains = [gene_row[t] for t in other_tissues]
                    max_other = max(other_gains) if other_gains else 0
                else:
                    max_other = 0 
                
                # Filter: > 1.2 倍
                if my_gain > 1.2 * max_other:
                    candidates.append(row)
        else:
            candidates = df_valid.to_dict('records')
            
        df_passed = pd.DataFrame(candidates)
        if df_passed.empty: return df_passed
        
        # 3. 排序
        df_passed.sort_values(by='RelGain', ascending=False, inplace=True)
        top_n = TOP_N_ATAC if modality_name == 'ATAC' else TOP_N_CAGE
        return df_passed.head(top_n)
    
    else: 
        # 🧊 RNA: 保持原样
        if global_df is None: return pd.DataFrame()
        df_merged = pd.merge(df_valid, global_df, on='Gene', how='left') 
        candidates = []
        for idx, row in df_merged.iterrows():
            if row['Baseline'] < BASELINE_THRESHOLD_RNA: continue
            tissue_vals = [row[t] for t in TISSUES if t in row]
            if not tissue_vals: continue
            if row[tissue_opt] >= max(tissue_vals) * 0.8: 
                candidates.append(row)
        df_passed = pd.DataFrame(candidates)
        df_passed.sort_values(by='Gain', ascending=False, inplace=True) 
        return df_passed.head(TOP_N_RNA)

# ==========================================
# 4. Main Analysis Loop
# ==========================================

def run_analysis_for_modality(modality_name, track_map, genome, model, meta_df):
    print(f"\n🔵 Starting Analysis for Modality: {modality_name}")
    
    global_df = None
    if modality_name == 'RNA':
        global_df = build_global_baseline_table(modality_name)
    elif modality_name in ['ATAC', 'CAGE']:
        global_df = build_global_rel_gain_table(modality_name)

    matrix_df = pd.DataFrame(index=TISSUES, columns=TISSUES, dtype=float)
    
    for tissue_opt in TISSUES:
        df_raw = get_candidates_from_cache_or_files(modality_name, tissue_opt)
        if df_raw.empty: continue
            
        top_genes_df = filter_genes_for_tissue(modality_name, tissue_opt, df_raw, global_df)
        
        if top_genes_df.empty:
            print(f"      ⚠️ No genes passed filter for {tissue_opt}")
            matrix_df.loc[tissue_opt, :] = 0.0
            continue
        
        print(f"      ✅ {tissue_opt}: {len(top_genes_df)} genes selected.")
            
        tissue_gains = {t: [] for t in TISSUES}
        
        for idx, row in tqdm.tqdm(top_genes_df.iterrows(), total=len(top_genes_df), desc=f"   Inferencing {tissue_opt}", leave=False):
            gene_name = row['Gene']
            snps = json.loads(row['SNPs_JSON']) 
            
            meta = get_gene_meta(gene_name, meta_df)
            if meta is None: continue
            chrom, tss, strand, gene_id = f"chr{meta['chr']}", int(meta['pos']), meta['strand'], meta['gene_ID']
            seq_start_pos = tss - SEQ_LEN // 2
            
            rel_snps = [(s['abs_pos'] - seq_start_pos, s['alt']) for s in snps]
            wt_seq, mut_seq = get_sequences(genome, chrom, tss, rel_snps)
            if wt_seq is None: continue
            
            with torch.no_grad():
                pred_wt = model(wt_seq.to(DEVICE))
                pred_mut = model(mut_seq.to(DEVICE))
            
            exon_ranges = []
            if modality_name == 'RNA':
                exon_ranges = get_exons_from_gtf(gene_id, GTF_PATH, tss, seq_start_pos)
            
            for t_eval in TISSUES:
                val_wt, val_mut = 0, 0
                if modality_name == 'ATAC':
                    track_id = TISSUE_ATAC_MAP[t_eval]
                    val_wt = calc_atac(pred_wt, track_id, strand)
                    val_mut = calc_atac(pred_mut, track_id, strand)
                elif modality_name == 'CAGE':
                    track_pair = TISSUE_CAGE_MAP[t_eval]
                    tid = track_pair[0] if strand == '+' else track_pair[1]
                    val_wt = calc_cage(pred_wt, tid)
                    val_mut = calc_cage(pred_mut, tid)
                elif modality_name == 'RNA':
                    track_id = TISSUE_RNA_MAP[t_eval]
                    val_wt = calc_rna(pred_wt, track_id, exon_ranges)
                    val_mut = calc_rna(pred_mut, track_id, exon_ranges)
                
                # 🔥🔥🔥 计算 Gain (使用 0.1 平滑，压制 Off-Target 噪音) 🔥🔥🔥
                if modality_name in ['ATAC', 'CAGE']:
                    epsilon = 0.1 
                else:
                    epsilon = 1.0 
                
                if val_wt >= 0:
                    gain_pct = (val_mut - val_wt) / (val_wt + epsilon) * 100
                else:
                    gain_pct = 0 
                tissue_gains[t_eval].append(gain_pct)
        
        for t in TISSUES:
            if len(tissue_gains[t]) > 0:
                matrix_df.loc[tissue_opt, t] = np.mean(tissue_gains[t])
            else:
                matrix_df.loc[tissue_opt, t] = 0.0
            
    return matrix_df

# ==========================================
# 5. Entry Point
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, 
                        choices=['all', 'ATAC', 'CAGE', 'RNA'], 
                        help='Choose modality to calculate')
    args = parser.parse_args()

    print(f"🚀 Starting Matrix Calculation (Fix: RelGain + 0.1 Smoothing)")
    
    genome = pyfaidx.Fasta(FASTA_PATH)
    meta_df = pd.read_csv(META_CSV)
    print("🔌 Loading Borzoi Model...")
    model = Borzoi.from_pretrained('johahi/borzoi-replicate-0').to(DEVICE).eval()
    
    modes_to_run = [args.mode] if args.mode != 'all' else ['ATAC', 'CAGE', 'RNA']
    
    if 'ATAC' in modes_to_run:
        mat = run_analysis_for_modality('ATAC', TISSUE_ATAC_MAP, genome, model, meta_df)
        mat.to_csv(f'{OUTPUT_DIR}/Figure3_ATAC_Matrix_Specific.csv') 
        print(f"\n✅ ATAC Matrix Saved.")
        print(mat)
        
    if 'CAGE' in modes_to_run:
        mat = run_analysis_for_modality('CAGE', TISSUE_CAGE_MAP, genome, model, meta_df)
        mat.to_csv(f'{OUTPUT_DIR}/Figure3_CAGE_Matrix_Specific.csv')
        print(f"\n✅ CAGE Matrix Saved.")
        print(mat)
        
    if 'RNA' in modes_to_run:
        mat = run_analysis_for_modality('RNA', TISSUE_RNA_MAP, genome, model, meta_df)
        mat.to_csv(f'{OUTPUT_DIR}/Figure3_RNA-seq_Matrix_Specific.csv')
        print(f"\n✅ RNA Matrix Saved.")
        print(mat)

    print("🎉 All done!")

if __name__ == "__main__":
    main()
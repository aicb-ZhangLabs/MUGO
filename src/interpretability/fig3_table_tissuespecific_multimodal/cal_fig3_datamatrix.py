'''
计算图3的data matrix，现在是用绝对的gain最大选出SNP（直接查表），然后算相对增益的时候是相对值，可能有些问题，看看加一些别的filter比如wt表达量要>1啥的。
这个没有cache目前，考虑之后加一个
'''
import torch
import pandas as pd
import numpy as np
import os
import glob
import pyfaidx
import tqdm
import gzip
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

os.makedirs(OUTPUT_DIR, exist_ok=True)

META_CSV = f'{DATASET_DIR}/gene_3000_borzoi_gencode_v41_hg38.csv'
FASTA_PATH = f'{DATASET_DIR}/human_genome_hg38/hg38.ml.fa'
GTF_PATH = f'{DATASET_DIR}/gencode.v41.annotation.gtf.gz' 

# 核心常量 (复用训练代码逻辑)
SNP_VOTE_THRESHOLD = 0.5
SEQ_LEN = 524288
BIN_SIZE = 32
TOP_N_GENES = 100 

# Borzoi 输出相关
OUTPUT_LEN = 6144
CROP_OFFSET = 5120  # (16384 - 6144) / 2

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
TISSUES = list(TISSUE_ATAC_MAP.keys())

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 1. Utils (复用逻辑)
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
    try: ref_seq_str = genome[chrom][start:end].seq.upper()
    except:
        alt = chrom.replace('chr', '') if 'chr' in chrom else f'chr{chrom}'
        try: ref_seq_str = genome[alt][start:end].seq.upper()
        except: return None, None
    if len(ref_seq_str) != SEQ_LEN: return None, None
    wt = seq_to_one_hot(ref_seq_str).unsqueeze(0)
    mut = wt.clone(); mapping = {'A':0, 'C':1, 'G':2, 'T':3}
    for p, alt in snp_list:
        if 0<=p<SEQ_LEN and alt in mapping:
            mut[0,:,p]=0; mut[0,mapping[alt],p]=1.0
    return wt, mut

def get_exons_from_gtf(gene_id, gtf_path, tss, seq_start_pos):
    """
    复用训练代码中的 Exon 解析逻辑
    返回的是相对于 Input Sequence Start 的 bin index
    """
    exon_ranges = []; base_id = gene_id.split('.')[0]
    try:
        opener = gzip.open if gtf_path.endswith('.gz') else open
        with opener(gtf_path, 'rt') as f:
            for line in f:
                if line.startswith('#') or base_id not in line: continue
                parts = line.strip().split('\t')
                if len(parts)<9 or parts[2]!='exon': continue
                # 兼容带版本号和不带版本号的 gene_id
                if f'gene_id "{base_id}"' in parts[8] or f'gene_id "{gene_id}"' in parts[8]:
                    s,e = int(parts[3]), int(parts[4])
                    bs, be = (s-seq_start_pos)//BIN_SIZE, (e-seq_start_pos)//BIN_SIZE
                    if be>0 and bs<(SEQ_LEN//BIN_SIZE): exon_ranges.append((bs, be))
    except: pass
    if not exon_ranges:
        c = (tss-seq_start_pos)//BIN_SIZE
        return [(c-5, c+5)]
    return exon_ranges

def parse_log_file(log_file):
    try:
        df = pd.read_csv(log_file)
        if df.empty: return None
        # 取 Gain 最大的那一步 (Best Epoch)
        best_idx = df['Gain'].idxmax(); max_gain = df.loc[best_idx, 'Gain']
        best_row = df.iloc[best_idx]; fname = os.path.basename(log_file)
        gname = fname.split('_')[0]; snps = []
        for i in range(1, 11):
            if f"Rank{i}_Score" not in df.columns: break
            # 只有当 Score 大于阈值时才选入
            if float(best_row[f"Rank{i}_Score"]) > SNP_VOTE_THRESHOLD:
                snps.append({
                    'abs_pos': int(best_row[f"Rank{i}_Pos"]), 
                    'alt': best_row[f"Rank{i}_RefAlt"].split("->")[1]
                })
        return {'gene': gname, 'gain': max_gain, 'snps': snps}
    except: return None

# ==========================================
# 2. Metric Calculators (Fixing Bugs Here)
# ==========================================

# Borzoi Output Shape is (Batch, Tracks, Seq_Len) e.g. (1, 7611, 6144)

def calc_cage(preds, track_idx):
    # preds shape: [Batch, Tracks, Length]
    n_bins = preds.shape[2] 
    center_bin = n_bins // 2
    
    # 逻辑：中心 ±20 bins
    # 修正索引顺序: [0, track_idx, slice]
    s, e = max(0, center_bin-20), min(n_bins, center_bin+20)
    return preds[0, track_idx, s:e].sum().item()

def calc_rna(preds, track_idx, exons):
    # preds shape: [Batch, Tracks, Length]
    total = 0.0
    n_bins = preds.shape[2] # 6144
    
    for r_start, r_end in exons:
        # ⚡️ 核心修正：减去 CROP_OFFSET (复用训练代码逻辑)
        out_start = r_start - CROP_OFFSET
        out_end = r_end - CROP_OFFSET
        
        # 边界检查
        if out_end <= 0 or out_start >= n_bins: continue
        out_start = max(0, out_start)
        out_end = min(n_bins, out_end)
        
        if out_start < out_end:
            # ⚡️ 核心修正：索引顺序 [0, track, length]
            total += preds[0, track_idx, out_start:out_end].sum().item()
            
    return total

def calc_atac(preds, track_idx, strand):
    # preds shape: [Batch, Tracks, Length]
    n_bins = preds.shape[2]
    center_bin = n_bins // 2
    
    # 逻辑：非对称窗口
    up, down = 500//BIN_SIZE, 2000//BIN_SIZE
    s = center_bin - up if strand == '+' else center_bin - down
    e = center_bin + down if strand == '+' else center_bin + up
    
    # 修正索引顺序
    s, e = max(0, s), min(n_bins, e)
    return preds[0, track_idx, s:e].sum().item()

# ==========================================
# 3. Main (Modality-Specific)
# ==========================================

def run_analysis_for_modality(modality_name, track_map, genome, model, meta_df):
    print(f"\n🔵 Starting Analysis for Modality: {modality_name}...")
    matrix = pd.DataFrame(index=TISSUES, columns=TISSUES, dtype=float)
    
    for tissue_opt in TISSUES:
        # 1. 动态确定文件夹路径
        if modality_name == 'ATAC':
            folder_name = f"{tissue_opt}_K10_borzoi_ATAC_modeltrain_res"
            file_pattern = "*_ATAC_optim_log.csv"
        elif modality_name == 'CAGE':
            folder_name = f"{tissue_opt}_K10_borzoi_CAGE_modeltrain_res"
            file_pattern = "*_CAGE_optim_log.csv"
        elif modality_name == 'RNA':
            folder_name = f"{tissue_opt}_K10_borzoi_modeltrain_res"
            file_pattern = "*_optim_log.csv"
            
        search_path = os.path.join(RESULT_ROOT, folder_name, file_pattern)
        files = glob.glob(search_path)
        
        if not files:
            print(f"   ⚠️ No logs found for {tissue_opt} ({modality_name}). Skipping.")
            continue

        # 2. 筛选 Top Genes
        candidates = []
        for f in files: 
            res = parse_log_file(f)
            if res and res['gain'] > 0 and len(res['snps']) > 0: 
                candidates.append(res)
        
        candidates.sort(key=lambda x: x['gain'], reverse=True)
        top_genes = candidates[:TOP_N_GENES]
        print(f"   [{tissue_opt}] Found {len(top_genes)} top genes.")

        # 3. 推理计算 Gain
        gains = {t: [] for t in TISSUES}

        for gene_data in tqdm.tqdm(top_genes, desc=f"   Eval {tissue_opt}"):
            meta = get_gene_meta(gene_data['gene'], meta_df)
            if meta is None: continue
            
            chrom, tss, strand, gene_id = f"chr{meta['chr']}", int(meta['pos']), meta['strand'], meta['gene_ID']
            
            # Prepare Seq
            seq_start_pos = tss - SEQ_LEN // 2
            rel_snps = [(s['abs_pos'] - seq_start_pos, s['alt']) for s in gene_data['snps']]
            wt_seq, mut_seq = get_sequences(genome, chrom, tss, rel_snps)
            if wt_seq is None: continue
            
            with torch.no_grad():
                # Output shape: [1, Tracks, SeqLen]
                pred_wt = model(wt_seq.to(DEVICE))
                pred_mut = model(mut_seq.to(DEVICE))
            
            # RNA 需要 Exon
            exon_ranges = []
            if modality_name == 'RNA':
                exon_ranges = get_exons_from_gtf(gene_id, GTF_PATH, tss, seq_start_pos)

            # 在所有 Tissue 上评估 (填列)
            for t_eval in TISSUES:
                track_id = track_map[t_eval]
                v_wt, v_mut = 0, 0
                
                if modality_name == 'ATAC':
                    v_wt = calc_atac(pred_wt, track_id, strand)
                    v_mut = calc_atac(pred_mut, track_id, strand)
                elif modality_name == 'CAGE':
                    tid = track_id[0] if strand == '+' else track_id[1]
                    v_wt = calc_cage(pred_wt, tid)
                    v_mut = calc_cage(pred_mut, tid)
                elif modality_name == 'RNA':
                    v_wt = calc_rna(pred_wt, track_id, exon_ranges)
                    v_mut = calc_rna(pred_mut, track_id, exon_ranges)
                
                gain_pct = (v_mut - v_wt) / (v_wt + 1) * 100 # need add larger psudonumber.
                gains[t_eval].append(gain_pct)

        # 4. 汇总均值 (填行)
        for t in TISSUES:
            matrix.loc[tissue_opt, t] = np.mean(gains[t])
            
    return matrix

def main():
    parser = argparse.ArgumentParser()
    # ⚡️ 新增 mode 参数，支持只跑单模态
    parser.add_argument('--mode', type=str, required=True, 
                        choices=['all', 'ATAC', 'CAGE', 'RNA'], 
                        help='Choose modality to calculate: ATAC, CAGE, RNA, or all')
    args = parser.parse_args()

    print(f"🚀 Starting Modality-Specific Calculation (Mode: {args.mode})...")
    
    genome = pyfaidx.Fasta(FASTA_PATH)
    meta_df = pd.read_csv(META_CSV)
    print("🔌 Loading Borzoi Model...")
    model = Borzoi.from_pretrained('johahi/borzoi-replicate-0').to(DEVICE).eval()
    
    # 决定跑哪些
    modes_to_run = [args.mode] if args.mode != 'all' else ['ATAC', 'CAGE', 'RNA']
    
    if 'ATAC' in modes_to_run:
        mat = run_analysis_for_modality('ATAC', TISSUE_ATAC_MAP, genome, model, meta_df)
        mat.to_csv(f'{OUTPUT_DIR}/Figure3_ATAC_Matrix.csv')
        print("✅ ATAC Matrix Saved.")
        
    if 'CAGE' in modes_to_run:
        mat = run_analysis_for_modality('CAGE', TISSUE_CAGE_MAP, genome, model, meta_df)
        mat.to_csv(f'{OUTPUT_DIR}/Figure3_CAGE_Matrix.csv')
        print("✅ CAGE Matrix Saved.")
        
    if 'RNA' in modes_to_run:
        mat = run_analysis_for_modality('RNA', TISSUE_RNA_MAP, genome, model, meta_df)
        mat.to_csv(f'{OUTPUT_DIR}/Figure3_RNA-seq_Matrix.csv')
        print("✅ RNA Matrix Saved.")

    print("🎉 All done!")

if __name__ == "__main__":
    main()
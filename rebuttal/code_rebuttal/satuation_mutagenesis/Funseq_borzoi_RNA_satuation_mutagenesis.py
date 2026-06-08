'''
In Silico Combinatorial Saturation Mutagenesis - FunSeq2 Baseline
Features:
1. Queries FunSeq2 scores for every possible alternative allele in the N-bp window.
2. Selects Top-K variants based on absolute FunSeq2 scores.
3. Evaluates the combined Top-K sequence through Borzoi to find the true combinatorial gain.
'''
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pyfaidx
import pysam
import os
import gzip
import argparse
from borzoi_pytorch import Borzoi

torch.backends.cudnn.enabled = False 

# ==========================================
# 0. 核心配置
# ==========================================
TISSUE_MAP = {
    'blood': 7531,  
    'brain': 7539,  
    'liver': 7563,  
    'heart': 7557,  
    'muscle': 7569,  
    'Pancreas': 7577,  
    'kidney': 7560,  
    'lung': 7566,  
}

FUNSEQ_FILE = "/home/dongbos/Combine_optim_Borzoi_SNP/dataset/Funseq2_data/hg38.funseq2.1.6.liftover.bed.bgz"

# ==========================================
# 1. Data Utils
# ==========================================
SEQ_LEN = 524288 

def seq_to_one_hot(seq_str):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    one_hot = np.zeros((4, len(seq_str)), dtype=np.float32)
    for i, base in enumerate(seq_str):
        if base in mapping:
            one_hot[mapping[base], i] = 1.0
    return torch.tensor(one_hot)

def prepare_tensors_saturation(fasta_path, chrom, center_pos, n_window, seq_len=SEQ_LEN):
    genome = pyfaidx.Fasta(fasta_path)
    start = center_pos - seq_len // 2
    end = center_pos + seq_len // 2
    ref_seq_str = genome[chrom][start:end].seq.upper()
    
    if len(ref_seq_str) != seq_len:
        raise ValueError(f"Sequence length mismatch: {len(ref_seq_str)} vs {seq_len}")
        
    ref_tensor = seq_to_one_hot(ref_seq_str).unsqueeze(0) 
    alt_tensor = ref_tensor.clone()
    
    snp_indices_list = []
    snp_meta_list = [] 
    
    comp_map = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'A'}
    base_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    
    rel_center = seq_len // 2
    win_start = rel_center - n_window // 2
    win_end = rel_center + n_window // 2
    
    for rel_pos in range(win_start, win_end):
        ref_base = ref_seq_str[rel_pos]
        alt_base = comp_map.get(ref_base, 'A')
        
        alt_tensor[0, :, rel_pos] = 0.0
        if alt_base in base_idx:
            alt_tensor[0, base_idx[alt_base], rel_pos] = 1.0
            
        snp_indices_list.append(rel_pos)
        snp_meta_list.append({
            'abs_pos': start + rel_pos, 
            'ref': ref_base, 
            'alt': alt_base
        })
        
    print(f"🔥 Saturated {len(snp_indices_list)} positions in {n_window}bp window centered at TSS.")
    return ref_tensor, alt_tensor, torch.tensor(snp_indices_list).long(), start, snp_meta_list

def get_exons_from_gtf(gene_name, gene_id, gtf_path, tss, seq_start_pos):
    exon_ranges = []
    POOL_SIZE = 32 
    try:
        with gzip.open(gtf_path, 'rt') as f:
            for line in f:
                if line.startswith('#'): continue
                if gene_id not in line: continue
                parts = line.strip().split('\t')
                if len(parts) < 9: continue
                if parts[2] == 'exon' and f'gene_id "{gene_id}' in parts[8]: 
                    s, e = int(parts[3]), int(parts[4])
                    b_start, b_end = (s - seq_start_pos) // POOL_SIZE, (e - seq_start_pos) // POOL_SIZE
                    if b_end > 0: exon_ranges.append((b_start, b_end))
    except: pass
    if not exon_ranges:
        center_bin = (tss - seq_start_pos) // POOL_SIZE
        return [(center_bin - 5, center_bin + 5)]
    return exon_ranges

def get_gene_meta_by_name(gene_name, meta_csv_path):
    print(f"📖 Reading Metadata from: {meta_csv_path}")
    df = pd.read_csv(meta_csv_path)
    matched = df[df['gene_name'] == gene_name]
    if len(matched) == 0:
        raise ValueError(f"Gene '{gene_name}' not found in {meta_csv_path}")
    row = matched.iloc[0]
    chrom = f"chr{row['chr']}" 
    tss = int(row['pos'])
    return chrom, tss, row['strand'], row['gene_ID'], row['gene_name']

def calculate_expression_score(model, input_seq, exon_regions, target_track_idx):
    output = model(input_seq)
    OUTPUT_LEN, CROP_OFFSET = 6144, 5120
    total_expr = 0
    for r_start, r_end in exon_regions:
        out_start = r_start - CROP_OFFSET
        out_end = r_end - CROP_OFFSET
        if out_end <= 0 or out_start >= OUTPUT_LEN: continue
        out_start, out_end = max(0, out_start), min(OUTPUT_LEN, out_end)
        if out_start < out_end:
            total_expr += output[:, target_track_idx, out_start:out_end].sum()
    return total_expr

# ==========================================
# 2. FunSeq2 Query Routine
# ==========================================
def run_funseq_baseline(gene_name_arg, k_arg, tissue_arg, track_idx_arg, n_window_arg):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if track_idx_arg is not None:
        target_idx = track_idx_arg
    elif tissue_arg in TISSUE_MAP:
        target_idx = TISSUE_MAP[tissue_arg]
    else:
        raise ValueError(f"Unknown tissue '{tissue_arg}'")

    BASE_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP'
    DATASET_DIR = f'{BASE_DIR}/dataset'
    RESULT_DIR = f'{BASE_DIR}/rebuttal/results_rebuttal/satuation_mutagenesis/raw_res_Funseq'
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    META_CSV = f'{DATASET_DIR}/gene_3000_borzoi_gencode_v41_hg38.csv'
    GTF_PATH = f'{DATASET_DIR}/gencode.v41.annotation.gtf.gz'
    FASTA_PATH = f'{DATASET_DIR}/human_genome_hg38/hg38.ml.fa'
    
    chrom, tss, strand, gene_id, gene_name = get_gene_meta_by_name(gene_name_arg, META_CSV)
    
    ref_seq, alt_seq, snp_positions, seq_start_pos, snp_meta_list = prepare_tensors_saturation(
        FASTA_PATH, chrom, center_pos=tss, n_window=n_window_arg
    )

    print(f"🚀 Querying FunSeq2 for {n_window_arg} positions...")
    
    # 建立映射: pos -> list_index
    target_snps = {info['abs_pos']: idx for idx, info in enumerate(snp_meta_list)}
    snp_scores = {idx: 0.0 for idx in range(len(snp_meta_list))}
    
    if not os.path.exists(FUNSEQ_FILE):
        raise FileNotFoundError(f"FunSeq file not found: {FUNSEQ_FILE}")
        
    fs = pysam.TabixFile(FUNSEQ_FILE)
    
    min_pos = snp_meta_list[0]['abs_pos']
    max_pos = snp_meta_list[-1]['abs_pos']
    
    try:
        # 一次性提取目标区间的记录，大幅提升查询速度
        records = fs.fetch(chrom, min_pos - 1, max_pos + 1)
        
        for line in records:
            parts = line.split('\t')
            if len(parts) <= 6: continue
                
            # FunSeq Bed format: chrom, start (0-based), end (1-based)
            r_start = int(parts[1])
            r_end = int(parts[2])
            
            raw_score_str = parts[6]
            score_part = raw_score_str.split(';')[0]
            
            if score_part in [".", ""]:
                score = 0.0
            else:
                try:
                    score = float(score_part)
                except ValueError:
                    score = 0.0
            
            # FunSeq2 往往给出区间的分数，我们把区间内的目标点都赋上分数
            for p in range(r_start + 1, r_end + 1):
                if p in target_snps:
                    idx = target_snps[p]
                    # 有时候一个点可能会命中多个注释，取最高分
                    if score > snp_scores[idx]:
                        snp_scores[idx] = score
                        
    except ValueError:
        print(f"⚠️ Region {chrom}:{min_pos}-{max_pos} not found in FunSeq2.")
    
    # 转成 list 并按分数降序排列提取 Top K
    score_list = [(idx, score) for idx, score in snp_scores.items()]
    score_list.sort(key=lambda x: x[1], reverse=True)
    
    current_k = min(k_arg, len(score_list))
    top_k_snps = score_list[:current_k]
    
    print(f"✅ Extracted Top-{current_k} FunSeq2 variants. Loading Borzoi for Evaluation...")

    # === Borzoi 验证环节 ===
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ref_seq = ref_seq.to(DEVICE)
    alt_seq = alt_seq.to(DEVICE)
    snp_positions = snp_positions.to(DEVICE)
    exon_regions = get_exons_from_gtf(gene_name, gene_id, GTF_PATH, tss, seq_start_pos)
    
    model_borzoi = Borzoi.from_pretrained('johahi/borzoi-replicate-0').to(DEVICE).eval()
    for p in model_borzoi.parameters(): p.requires_grad = False

    with torch.no_grad():
        baseline_expr = calculate_expression_score(model_borzoi, ref_seq, exon_regions, target_idx).item()
        
        # 构造同时包含 Top-K 突变的新序列
        mut_seq = ref_seq.clone()
        for rank, (idx, score) in enumerate(top_k_snps):
            rel_pos = snp_positions[idx].item()
            mut_seq[0, :, rel_pos] = alt_seq[0, :, rel_pos]
            
        mut_expr = calculate_expression_score(model_borzoi, mut_seq, exon_regions, target_idx).item()
        
    final_gain = mut_expr - baseline_expr
    print(f"📉 Baseline Expression ({tissue_arg}): {baseline_expr:.4f}")
    print(f"🎯 FunSeq2 Top-{current_k} Final Gain: {final_gain:+.4f}")
    
    # === 记录并保存结果 ===
    row_data = {
        "Gain": final_gain,
        "Baseline": baseline_expr,
        "Tissue": tissue_arg,
        "TrackIdx": target_idx,
        "Window_N": n_window_arg
    }
    
    for rank, (idx, score) in enumerate(top_k_snps):
        snp_info = snp_meta_list[idx]
        row_data[f"Rank{rank+1}_Pos"] = snp_info['abs_pos']
        row_data[f"Rank{rank+1}_RefAlt"] = f"{snp_info['ref']}->{snp_info['alt']}"
        row_data[f"Rank{rank+1}_Score"] = score

    csv_filename = f"{RESULT_DIR}/{gene_name}_{tissue_arg}_N{n_window_arg}_K{k_arg}_saturation_Funseq.csv"
    pd.DataFrame([row_data]).to_csv(csv_filename, index=False)
    print(f"✅ Finished {gene_name}. Saved to {csv_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gene', type=str, required=True, help='Name of the gene (e.g., ELP1)')
    parser.add_argument('--k', type=int, default=10, help='Number of mutations to select')
    parser.add_argument('--tissue', type=str, default='blood')
    parser.add_argument('--manual_track_id', type=int, default=None)
    parser.add_argument('--N', type=int, default=1000, 
                        help='Size of the saturation mutagenesis window centered at TSS')
    
    args = parser.parse_args()
    
    run_funseq_baseline(gene_name_arg=args.gene, 
                        k_arg=args.k, 
                        tissue_arg=args.tissue,
                        track_idx_arg=args.manual_track_id,
                        n_window_arg=args.N)
'''
In Silico Combinatorial Saturation Mutagenesis - Saliency Baseline
Features:
1. Calculates input gradients for the N-bp window.
2. Selects Top-K variants based on gradient differences (Grad[Alt] - Grad[Ref]).
3. Evaluates the combined Top-K sequence to find the true combinatorial gain.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import pyfaidx
import os
import gzip
import argparse
from borzoi_pytorch import Borzoi

torch.backends.cudnn.enabled = False 

# ==========================================
# 0. 核心配置 (Track ID 映射)
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
# 2. Main Saliency Routine
# ==========================================
def run_saliency(gene_name_arg, k_arg, tissue_arg, track_idx_arg, n_window_arg):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if track_idx_arg is not None:
        target_idx = track_idx_arg
    elif tissue_arg in TISSUE_MAP:
        target_idx = TISSUE_MAP[tissue_arg]
    else:
        raise ValueError(f"Unknown tissue '{tissue_arg}'")

    BASE_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP'
    DATASET_DIR = f'{BASE_DIR}/dataset'
    
    # 按照你的要求，结果存入这个 Saliency 专属文件夹
    RESULT_DIR = f'{BASE_DIR}/rebuttal/results_rebuttal/satuation_mutagenesis/raw_res_saliency'
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    META_CSV = f'{DATASET_DIR}/gene_3000_borzoi_gencode_v41_hg38.csv'
    GTF_PATH = f'{DATASET_DIR}/gencode.v41.annotation.gtf.gz'
    FASTA_PATH = f'{DATASET_DIR}/human_genome_hg38/hg38.ml.fa'
    
    chrom, tss, strand, gene_id, gene_name = get_gene_meta_by_name(gene_name_arg, META_CSV)
    
    ref_seq, alt_seq, snp_positions, seq_start_pos, snp_meta_list = prepare_tensors_saturation(
        FASTA_PATH, chrom, center_pos=tss, n_window=n_window_arg
    )

    ref_seq = ref_seq.to(DEVICE)
    alt_seq = alt_seq.to(DEVICE)
    snp_positions = snp_positions.to(DEVICE)
    exon_regions = get_exons_from_gtf(gene_name, gene_id, GTF_PATH, tss, seq_start_pos)
    
    print("Loading Borzoi...")
    model_borzoi = Borzoi.from_pretrained('johahi/borzoi-replicate-0').to(DEVICE).eval()
    for p in model_borzoi.parameters(): p.requires_grad = False

    print(f"🚀 Running Saliency Baseline (N={n_window_arg}, K={k_arg}) for {gene_name}...")

    # === Saliency 计算开始 ===
    ref_seq.requires_grad = True
    expr = calculate_expression_score(model_borzoi, ref_seq, exon_regions, target_idx)
    baseline_expr = expr.item()
    print(f"📉 Baseline Expression ({tissue_arg}): {baseline_expr:.4f}")

    # 求导获取输入梯度
    model_borzoi.zero_grad()
    expr.backward()
    
    grads = ref_seq.grad[0] # Shape: (4, 524288)
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    
    snp_scores = []
    # 遍历 N-window 内的每一个位点计算 Saliency Score
    for idx, snp_info in enumerate(snp_meta_list):
        rel_pos = snp_positions[idx].item()
        ref_base = snp_info['ref']
        alt_base = snp_info['alt']
        
        if ref_base in mapping and alt_base in mapping:
            ref_idx = mapping[ref_base]
            alt_idx = mapping[alt_base]
            # 计算梯度带来的增益 (Alt 梯度 - Ref 梯度)
            score = grads[alt_idx, rel_pos] - grads[ref_idx, rel_pos]
            snp_scores.append((idx, score.item()))
        else:
            snp_scores.append((idx, -9999.0))
            
    # 按梯度增益降序排列并选取 Top K
    snp_scores.sort(key=lambda x: x[1], reverse=True)
    current_k = min(k_arg, len(snp_scores))
    top_k_snps = snp_scores[:current_k]
    
    # 构造同时包含 Top-K 突变的新序列
    mut_seq = ref_seq.clone().detach()
    mut_seq.requires_grad = False
    
    for rank, (idx, score) in enumerate(top_k_snps):
        rel_pos = snp_positions[idx].item()
        alt_tensor_slice = alt_seq[0, :, rel_pos]
        mut_seq[0, :, rel_pos] = alt_tensor_slice
        
    # 前向传播算这 10 个突变组合在一起的真实得分
    with torch.no_grad():
        mut_expr = calculate_expression_score(model_borzoi, mut_seq, exon_regions, target_idx).item()
        
    final_gain = mut_expr - baseline_expr
    print(f"🎯 Saliency Top-{current_k} Final Gain: {final_gain:+.4f}")
    
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

    csv_filename = f"{RESULT_DIR}/{gene_name}_{tissue_arg}_N{n_window_arg}_K{k_arg}_saturation_saliency.csv"
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
    
    run_saliency(gene_name_arg=args.gene, 
                 k_arg=args.k, 
                 tissue_arg=args.tissue,
                 track_idx_arg=args.manual_track_id,
                 n_window_arg=args.N)
'''
4. Feature Ablation (Masking)： 类似于 ISM，但是把位点 Mask 成 0 或者 N，而不是突变成别的碱基。 
python feature_ablation.py --index 0 --tissue brain
'''

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pyfaidx
import os
import gzip
import argparse
from tqdm import tqdm
from borzoi_pytorch import Borzoi

# 禁用 cuDNN 以防万一 (和你之前的 fix 一样)
torch.backends.cudnn.enabled = False 

# ================= 核心配置 =================
TISSUE_MAP = {
    'blood': 7531, 'brain': 7539, 'liver': 7563,
    'heart': 7557, 'muscle': 7569, 'Pancreas': 7577,    
}
SEQ_LEN = 524288 

# ================= Data Utils =================
def seq_to_one_hot(seq_str):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    one_hot = np.zeros((4, len(seq_str)), dtype=np.float32)
    for i, base in enumerate(seq_str):
        if base in mapping: one_hot[mapping[base], i] = 1.0
    return torch.tensor(one_hot)

def prepare_tensors(csv_path, fasta_path, chrom, center_pos, seq_len=SEQ_LEN):
    genome = pyfaidx.Fasta(fasta_path)
    start = center_pos - seq_len // 2
    end = center_pos + seq_len // 2
    ref_seq_str = genome[chrom][start:end].seq.upper()
    
    ref_tensor = seq_to_one_hot(ref_seq_str).unsqueeze(0) 
    
    if not os.path.exists(csv_path): raise FileNotFoundError(f"SNP file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if 'POS_hg38' in df.columns: df.rename(columns={'POS_hg38': 'pos'}, inplace=True)
    if 'ALT' in df.columns: df.rename(columns={'ALT': 'alt'}, inplace=True)
    df['pos'] = df['pos'].astype(int)
    
    snp_indices_list = []
    snp_meta_list = [] 
    
    for idx, row in df.iterrows():
        abs_pos = int(row['pos'])
        rel_pos = abs_pos - start
        if 0 <= rel_pos < seq_len:
            snp_indices_list.append(rel_pos)
            snp_meta_list.append({
                'abs_pos': abs_pos, 
                'ref': row['REF'] if 'REF' in row else 'N', 
                'alt': row['alt']
            })
    return ref_tensor, snp_indices_list, start, snp_meta_list

def get_exons_from_gtf(gene_name, gene_id, gtf_path, tss, seq_start_pos):
    exon_ranges = []
    POOL_SIZE = 32 
    try:
        with gzip.open(gtf_path, 'rt') as f:
            for line in f:
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

def get_gene_meta_by_index(target_index, meta_csv_path):
    df = pd.read_csv(meta_csv_path)
    row = df.iloc[target_index]
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

# ================= Ablation Logic =================

def run_ablation(gene_index_arg, tissue_arg):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if tissue_arg in TISSUE_MAP: target_idx = TISSUE_MAP[tissue_arg]
    else: raise ValueError(f"Unknown tissue '{tissue_arg}'")

    # 路径配置
    BASE_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP'
    DATASET_DIR = f'{BASE_DIR}/dataset'
    BENCHMARK_DIR = f'{BASE_DIR}/results/baseline_benchmark/Feature_Ablation'
    SAVE_DIR = f'{BENCHMARK_DIR}/raw_res/{tissue_arg}'
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Metadata
    META_CSV = f'{DATASET_DIR}/gene_3000_borzoi_gencode_v41_hg38.csv'
    GTF_PATH = f'{DATASET_DIR}/gencode.v41.annotation.gtf.gz'
    FASTA_PATH = f'{DATASET_DIR}/human_genome_hg38/hg38.ml.fa'
    
    chrom, tss, strand, gene_id, gene_name = get_gene_meta_by_index(gene_index_arg, META_CSV)
    SNP_CSV = f'{DATASET_DIR}/gene_snps_hg38/{gene_name}_snps_hg38.csv'
    
    try:
        ref_seq, snp_indices, seq_start_pos, snp_meta_list = prepare_tensors(
            SNP_CSV, FASTA_PATH, chrom, center_pos=tss
        )
    except FileNotFoundError: return

    ref_seq = ref_seq.to(DEVICE)
    exon_regions = get_exons_from_gtf(gene_name, gene_id, GTF_PATH, tss, seq_start_pos)
    
    print("Loading Borzoi...")
    model_borzoi = Borzoi.from_pretrained('johahi/borzoi-replicate-0').to(DEVICE).eval()
    
    # 计算 Baseline (Clean Ref)
    with torch.no_grad():
        baseline_expr = calculate_expression_score(model_borzoi, ref_seq, exon_regions, target_idx).item()
    print(f"📉 Baseline Expression: {baseline_expr:.4f}")
    
    results = []
    print(f"🎭 Running Feature Ablation (Masking) on {len(snp_indices)} SNPs...")
    
    for i in tqdm(range(len(snp_indices))):
        snp_info = snp_meta_list[i]
        rel_pos = snp_indices[i]
        
        # 构造 Ablated Input
        input_seq = ref_seq.clone()
        
        # ⚠️ 核心操作：Masking
        # 将该位置的 One-hot 向量全部设为 0 (相当于 N)
        input_seq[0, :, rel_pos] = 0.0 
        
        # 推理
        with torch.no_grad():
            masked_expr = calculate_expression_score(model_borzoi, input_seq, exon_regions, target_idx).item()
            
        # Metric: Impact = |Change|
        # 通常我们看 Impact (影响力) 或者 Drop (损失)
        # 这里记录 Change，负数表示 Mask 后表达下降（说明原位点很重要）
        change = masked_expr - baseline_expr
        impact = abs(change) # 绝对值越大越重要
        
        results.append({
            'Gene': gene_name,
            'Tissue': tissue_arg,
            'Pos': snp_info['abs_pos'],
            'Ref': snp_info['ref'],
            'Alt': snp_info['alt'], # 虽然没用到 Alt，存下来方便以后核对
            'Baseline_Expr': baseline_expr,
            'Masked_Expr': masked_expr,
            'Change': change, 
            'Impact_Score': impact # 排序依据
        })
        
    # 保存
    df_res = pd.DataFrame(results)
    # 按 Impact 降序排列 (影响力最大的排前面)
    df_res = df_res.sort_values(by='Impact_Score', ascending=False)
    
    csv_path = f"{SAVE_DIR}/{gene_name}_ablation.csv"
    df_res.to_csv(csv_path, index=False)
    
    print(f"✅ Ablation Done. Top 1 Impact: {df_res.iloc[0]['Impact_Score']:.4f}")
    print(f"💾 Saved to: {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, required=True)
    parser.add_argument('--tissue', type=str, default='blood')
    args = parser.parse_args()
    
    run_ablation(gene_index_arg=args.index, 
                 tissue_arg=args.tissue)



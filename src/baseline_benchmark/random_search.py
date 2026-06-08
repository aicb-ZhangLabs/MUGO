'''
2. Random Search (Baseline of Baselines)： 从 Candidate Pool 里随机抽取 K 个 SNP，重复 N 次（比如1000次），取分最高的那个组合。（enrichment 1x）  
python run_random_baseline.py --index 0 --tissue brain --k 10 --trials 200
'''

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pyfaidx
import os
import gzip
import argparse
import random
from tqdm import tqdm
from borzoi_pytorch import Borzoi

torch.backends.cudnn.enabled = False 

# ==========================================
# 0. 核心配置 (保持一致)
# ==========================================

TISSUE_MAP = {
    'blood': 7531, 'brain': 7539, 'liver': 7563,
    'heart': 7557, 'muscle': 7569, 'Pancreas': 7577,    
}

# ==========================================
# 1. Data Utils (完全复用)
# ==========================================
SEQ_LEN = 524288 

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
    alt_tensor = ref_tensor.clone()
    
    if not os.path.exists(csv_path): raise FileNotFoundError(f"SNP file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if 'POS_hg38' in df.columns: df.rename(columns={'POS_hg38': 'pos'}, inplace=True)
    if 'ALT' in df.columns: df.rename(columns={'ALT': 'alt'}, inplace=True)
    df['pos'] = df['pos'].astype(int)
    
    snp_indices_list = []
    snp_meta_list = [] 
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    for idx, row in df.iterrows():
        abs_pos = int(row['pos'])
        rel_pos = abs_pos - start
        if 0 <= rel_pos < seq_len:
            alt_base = row['alt']
            ref_base = row['REF'] if 'REF' in row else 'N'
            if alt_base in mapping:
                vec = torch.zeros(4)
                vec[mapping[alt_base]] = 1.0
                alt_tensor[0, :, rel_pos] = vec
                snp_indices_list.append(rel_pos)
                snp_meta_list.append({'abs_pos': abs_pos, 'ref': ref_base, 'alt': alt_base})
    return ref_tensor, alt_tensor, snp_indices_list, start, snp_meta_list

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

# ==========================================
# 2. Random Search Logic
# ==========================================

def run_random_search(gene_index_arg, tissue_arg, k_arg=10, num_trials=200):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if tissue_arg in TISSUE_MAP: target_idx = TISSUE_MAP[tissue_arg]
    else: raise ValueError(f"Unknown tissue '{tissue_arg}'")

    # 路径配置
    BASE_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP'
    DATASET_DIR = f'{BASE_DIR}/dataset'
    BENCHMARK_DIR = f'{BASE_DIR}/results/baseline_benchmark/Random_Search'
    SAVE_DIR = f'{BENCHMARK_DIR}/raw_res/{tissue_arg}'
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 加载数据
    META_CSV = f'{DATASET_DIR}/gene_3000_borzoi_gencode_v41_hg38.csv'
    GTF_PATH = f'{DATASET_DIR}/gencode.v41.annotation.gtf.gz'
    FASTA_PATH = f'{DATASET_DIR}/human_genome_hg38/hg38.ml.fa'
    
    chrom, tss, strand, gene_id, gene_name = get_gene_meta_by_index(gene_index_arg, META_CSV)
    SNP_CSV = f'{DATASET_DIR}/gene_snps_hg38/{gene_name}_snps_hg38.csv'
    
    try:
        ref_seq, alt_seq, snp_indices_list, seq_start_pos, snp_meta_list = prepare_tensors(
            SNP_CSV, FASTA_PATH, chrom, center_pos=tss
        )
    except FileNotFoundError: return

    ref_seq = ref_seq.to(DEVICE)
    alt_seq = alt_seq.to(DEVICE)
    exon_regions = get_exons_from_gtf(gene_name, gene_id, GTF_PATH, tss, seq_start_pos)
    
    # 加载模型
    print("Loading Borzoi...")
    model_borzoi = Borzoi.from_pretrained('johahi/borzoi-replicate-0').to(DEVICE).eval()
    
    with torch.no_grad():
        baseline_expr = calculate_expression_score(model_borzoi, ref_seq, exon_regions, target_idx).item()
    print(f"📉 Baseline: {baseline_expr:.4f}")
    
    # === RANDOM SEARCH LOOP ===
    best_gain = -9999
    best_combo = []
    
    print(f"🎲 Running {num_trials} Random Trials (K={k_arg})...")
    
    # 确保 SNP 数量够选
    pool_size = len(snp_indices_list)
    if pool_size < k_arg:
        print("⚠️ Not enough SNPs to sample K, selecting all.")
        k_arg = pool_size

    # 用于保存所有 Trial 的结果，看分布（可选）
    all_trials = []

    for _ in tqdm(range(num_trials)):
        # 1. 随机采样 K 个 Index
        sampled_indices_idx = random.sample(range(pool_size), k_arg)
        
        # 2. 构造 Input: 将这些位置设为 Alt
        input_seq = ref_seq.clone()
        for idx in sampled_indices_idx:
            rel_pos = snp_indices_list[idx]
            input_seq[0, :, rel_pos] = alt_seq[0, :, rel_pos]
            
        # 3. 推理
        with torch.no_grad():
            mutant_expr = calculate_expression_score(model_borzoi, input_seq, exon_regions, target_idx).item()
        
        gain = mutant_expr - baseline_expr
        
        # 记录
        trial_record = {
            'Trial_ID': _,
            'Gain': gain,
            'Indices': sampled_indices_idx # 存的是 index 列表
        }
        all_trials.append(trial_record)
        
        if gain > best_gain:
            best_gain = gain
            best_combo = sampled_indices_idx

    # 1. 保存 Best Combo (加了 N{num_trials} 到文件名)
    final_snps = []
    for i, idx in enumerate(best_combo):
        info = snp_meta_list[idx]
        final_snps.append({
            'Rank': i+1, 
            'Pos': info['abs_pos'],
            'Ref': info['ref'],
            'Alt': info['alt'],
            'Best_Gain': best_gain,
            'Total_Trials': num_trials # 把总次数也记在表里
        })
        
    df_best = pd.DataFrame(final_snps)
    # 文件名增加 _N{num_trials}
    csv_best_path = f"{SAVE_DIR}/{gene_name}_random_best_K{k_arg}_N{num_trials}.csv"
    df_best.to_csv(csv_best_path, index=False)
    print(f"✅ Saved Best Combo to: {csv_best_path}")

    # 2. (新增) 保存所有 Trials 的 Gain 分布
    # 这对画图证明你的方法显著优于随机非常有帮助
    df_dist = pd.DataFrame(all_trials) # all_trials 包含 Trial_ID, Gain, Indices
    # 只存 Gain 就够了，为了省空间可以不存 Indices
    csv_dist_path = f"{SAVE_DIR}/{gene_name}_random_distribution_K{k_arg}_N{num_trials}.csv"
    df_dist[['Trial_ID', 'Gain']].to_csv(csv_dist_path, index=False)
    print(f"📊 Saved Gain Distribution to: {csv_dist_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, required=True)
    parser.add_argument('--tissue', type=str, default='blood')
    parser.add_argument('--k', type=int, default=10, help="Number of SNPs to select")
    parser.add_argument('--trials', type=int, default=200, help="Number of random attempts")
    
    args = parser.parse_args()
    
    run_random_search(gene_index_arg=args.index, 
                      tissue_arg=args.tissue,
                      k_arg=args.k,
                      num_trials=args.trials)
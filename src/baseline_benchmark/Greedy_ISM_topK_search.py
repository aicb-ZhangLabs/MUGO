'''
1. Greedy ISM (Top-K Single Scan) :  暴力扫描序列里的每一个SNP（比如2000个），计算每个SNP单独突变后的 Borzoi Score 变化。然后简单地把分数最高的 Top K 个挑出来。 additive assumption。 
# 跑第 0 个基因，在 brain 上
python Greedy_ISM_topK_search.py --index 0 --tissue brain
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
from tqdm import tqdm
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
}

# ==========================================
# 1. Data Utils (直接复用你的 Training 脚本)
# ==========================================

SEQ_LEN = 524288 

def seq_to_one_hot(seq_str):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    one_hot = np.zeros((4, len(seq_str)), dtype=np.float32)
    for i, base in enumerate(seq_str):
        if base in mapping:
            one_hot[mapping[base], i] = 1.0
    return torch.tensor(one_hot)

def prepare_tensors(csv_path, fasta_path, chrom, center_pos, seq_len=SEQ_LEN):
    # print(f"Loading Genome from {fasta_path}...")
    genome = pyfaidx.Fasta(fasta_path)
    start = center_pos - seq_len // 2
    end = center_pos + seq_len // 2
    ref_seq_str = genome[chrom][start:end].seq.upper()
    if len(ref_seq_str) != seq_len:
        raise ValueError(f"Sequence length mismatch: {len(ref_seq_str)} vs {seq_len}")
    
    # Base Reference Tensor
    ref_tensor = seq_to_one_hot(ref_seq_str).unsqueeze(0) 
    
    # Alt Tensor (包含所有突变，用于提取单个突变的碱基信息)
    alt_tensor = ref_tensor.clone()
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"SNP file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    # 兼容列名
    if 'POS_hg38' in df.columns:
        df.rename(columns={'POS_hg38': 'pos'}, inplace=True)
    if 'ALT' in df.columns:
        df.rename(columns={'ALT': 'alt'}, inplace=True)
        
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
                alt_tensor[0, :, rel_pos] = vec # 把这个位置变成 Alt
                
                snp_indices_list.append(rel_pos)
                snp_meta_list.append({'abs_pos': abs_pos, 'ref': ref_base, 'alt': alt_base})
                
    print(f"Found {len(snp_indices_list)} SNPs in window.")
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

def get_gene_meta_by_index(target_index, meta_csv_path):
    print(f"📖 Reading Metadata from: {meta_csv_path}")
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
            # target_track_idx 这里的维度需要注意，如果是 batch>1 需要修改，但这里我们 batch=1
            total_expr += output[:, target_track_idx, out_start:out_end].sum()
    return total_expr

# ==========================================
# 2. Main Greedy Scan Logic
# ==========================================

def run_greedy_scan(gene_index_arg, tissue_arg):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 确定 Track Index
    if tissue_arg in TISSUE_MAP:
        target_idx = TISSUE_MAP[tissue_arg]
        print(f"🔍 Auto-detected Track ID for '{tissue_arg}': {target_idx}")
    else:
        raise ValueError(f"Unknown tissue '{tissue_arg}'. Available: {list(TISSUE_MAP.keys())}")

    # 2. 文件夹结构配置
    BASE_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP'
    DATASET_DIR = f'{BASE_DIR}/dataset'
    
    # [Output Structure]
    # /home/dongbos/Combine_optim_Borzoi_SNP/results/baseline_benchmark/Greedy_ISM/raw_res/{tissue}/
    BENCHMARK_DIR = f'{BASE_DIR}/results/baseline_benchmark/Greedy_ISM'
    SAVE_DIR = f'{BENCHMARK_DIR}/raw_res/{tissue_arg}'
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"📂 Greedy results will be saved to: {SAVE_DIR}")
    
    # 3. 加载 Metadata
    META_CSV = f'{DATASET_DIR}/gene_3000_borzoi_gencode_v41_hg38.csv'
    GTF_PATH = f'{DATASET_DIR}/gencode.v41.annotation.gtf.gz'
    FASTA_PATH = f'{DATASET_DIR}/human_genome_hg38/hg38.ml.fa'
    
    chrom, tss, strand, gene_id, gene_name = get_gene_meta_by_index(gene_index_arg, META_CSV)
    SNP_CSV = f'{DATASET_DIR}/gene_snps_hg38/{gene_name}_snps_hg38.csv'
    
    # 4. 准备 Tensor
    try:
        # ref_seq: 全是 ref 的序列
        # alt_seq: 全是 alt 的序列 (我们这里只用它来提取单个 alt base)
        ref_seq, alt_seq, snp_positions, seq_start_pos, snp_meta_list = prepare_tensors(
            SNP_CSV, FASTA_PATH, chrom, center_pos=tss
        )
    except FileNotFoundError:
        print(f"⚠️ SNP file not found for {gene_name}, skipping.")
        return

    ref_seq = ref_seq.to(DEVICE)
    alt_seq = alt_seq.to(DEVICE)
    # snp_positions 是 index 的列表
    
    exon_regions = get_exons_from_gtf(gene_name, gene_id, GTF_PATH, tss, seq_start_pos)
    
    # 5. 加载模型
    print("Loading Borzoi...")
    model_borzoi = Borzoi.from_pretrained('johahi/borzoi-replicate-0').to(DEVICE).eval()
    
    # 6. 计算 Baseline (Ref Seq)
    print("Calculating Baseline Expression...")
    with torch.no_grad():
        baseline_expr = calculate_expression_score(model_borzoi, ref_seq, exon_regions, target_idx).item()
    print(f"📉 Baseline Expression ({tissue_arg}): {baseline_expr:.4f}")
    
    # 7. 开始 Greedy Scan
    # 逻辑：遍历每个 SNP，构造一个 input 只有该 SNP 突变，计算 Gain
    
    results = []
    
    print(f"🚀 Scanning {len(snp_meta_list)} SNPs for {gene_name}...")
    
    # 使用 tqdm 显示进度条
    for i in tqdm(range(len(snp_meta_list))):
        snp_info = snp_meta_list[i]
        rel_pos = snp_positions[i] # 相对位置
        
        # 构造 Input: 复制 Ref，然后在特定位置替换成 Alt
        # 这样保证每次只有一个 SNP 变化
        input_seq = ref_seq.clone()
        
        # 核心操作：把 Alt Tensor 在这个位置的值赋给 Input
        # alt_tensor[0, :, rel_pos] 已经是 One-hot 的 Alt Base
        input_seq[0, :, rel_pos] = alt_seq[0, :, rel_pos]
        
        # 推理
        with torch.no_grad():
            mutant_expr = calculate_expression_score(model_borzoi, input_seq, exon_regions, target_idx).item()
            
        gain = mutant_expr - baseline_expr
        
        results.append({
            'Gene': gene_name,
            'Tissue': tissue_arg,
            'Pos': snp_info['abs_pos'],
            'Ref': snp_info['ref'],
            'Alt': snp_info['alt'],
            'Baseline_Expr': baseline_expr,
            'Mutant_Expr': mutant_expr,
            'Gain': gain
        })
        
    # 8. 保存全量结果
    df_res = pd.DataFrame(results)
    
    # 按照 Gain 降序排列 (方便之后看)
    df_res = df_res.sort_values(by='Gain', ascending=False)
    
    csv_path = f"{SAVE_DIR}/{gene_name}_greedy_scan.csv"
    df_res.to_csv(csv_path, index=False)
    
    print(f"✅ Finished Greedy Scan for {gene_name}. Saved to: {csv_path}")
    print(f"   Top 1 Gain: {df_res.iloc[0]['Gain']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, required=True, help='Row index of the gene')
    parser.add_argument('--tissue', type=str, default='blood', 
                        help='Tissue name (e.g., blood, brain, liver).')
    
    args = parser.parse_args()
    
    run_greedy_scan(gene_index_arg=args.index, 
                    tissue_arg=args.tissue)
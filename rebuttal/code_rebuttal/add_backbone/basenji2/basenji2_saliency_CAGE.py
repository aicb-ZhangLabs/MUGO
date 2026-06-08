'''
Basenji2 Saliency (Gradient) Generator
Path: /home/dongbos/Combine_optim_Borzoi_SNP/rebuttal/results_rebuttal/add_backbone_model/basenji2/CAGE_saliency_raw_results
'''
import torch
import pandas as pd
import numpy as np
import pyfaidx
import os
import gzip
import argparse
from basenji2_pytorch import Basenji2, basenji2_params, basenji2_weights

torch.backends.cudnn.enabled = False 

# ==========================================
# 0. 核心配置
# ==========================================
TISSUE_MAP = {
    'blood': 4950,   
    'brain': 4680,   
    'liver': 4686,   
    'heart': 4684,   
    'muscle': 4691,  
    'Pancreas': 4946 
}

SEQ_LEN = 131072 # Basenji2 感受野
POOL_SIZE = 128  # Basenji2 分辨率

# ==========================================
# 1. Data Utils
# ==========================================
def seq_to_one_hot(seq_str):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    one_hot = np.zeros((4, len(seq_str)), dtype=np.float32)
    for i, base in enumerate(seq_str):
        if base in mapping:
            one_hot[mapping[base], i] = 1.0
    return torch.tensor(one_hot)

def prepare_tensors(csv_path, fasta_path, chrom, center_pos, seq_len=SEQ_LEN):
    genome = pyfaidx.Fasta(fasta_path)
    start = center_pos - seq_len // 2
    end = center_pos + seq_len // 2
    ref_seq_str = genome[chrom][start:end].seq.upper()
    
    if len(ref_seq_str) != seq_len:
        raise ValueError(f"Sequence length mismatch: {len(ref_seq_str)} vs {seq_len}")
        
    ref_tensor = seq_to_one_hot(ref_seq_str).unsqueeze(0) 
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"SNP file not found: {csv_path}")
        
    df = pd.read_csv(csv_path)
    df.rename(columns={'POS_hg38': 'pos', 'ALT': 'alt'}, inplace=True)
    df['pos'] = df['pos'].astype(int)
    
    snp_meta_list = [] 
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    
    for idx, row in df.iterrows():
        abs_pos = int(row['pos'])
        rel_pos = abs_pos - start
        if 0 <= rel_pos < seq_len:
            alt_base = row['alt']
            ref_base = row['REF'] if 'REF' in row else 'N'
            if alt_base in mapping and ref_base in mapping:
                snp_meta_list.append({
                    'abs_pos': abs_pos, 
                    'rel_pos': rel_pos,
                    'ref': ref_base, 
                    'alt': alt_base
                })
                
    print(f"Found {len(snp_meta_list)} valid SNPs in {seq_len}bp window.")
    return ref_tensor, start, snp_meta_list

def get_exons_from_gtf(gene_name, gene_id, gtf_path, tss, seq_start_pos):
    exon_ranges = []
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
        return [(center_bin - 1, center_bin + 2)] 
    return exon_ranges

def get_gene_meta_by_name(gene_name, meta_csv_path):
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
    if output.shape[-1] != 5313 and output.shape[1] == 5313:
        output = output.transpose(1, 2)
        
    OUTPUT_LEN = output.shape[1] 
    total_expr = 0
    for r_start, r_end in exon_regions:
        out_start, out_end = max(0, int(r_start)), min(OUTPUT_LEN, int(r_end))
        if out_start < out_end:
            total_expr += output[0, out_start:out_end, target_track_idx].sum()
    return total_expr

# ==========================================
# 2. Main Routine (Saliency)
# ==========================================
def run_saliency(gene_name_arg, tissue_arg):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if tissue_arg in TISSUE_MAP:
        target_idx = TISSUE_MAP[tissue_arg] 
    else:
        raise ValueError(f"Unknown tissue '{tissue_arg}'")

    BASE_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP'
    DATASET_DIR = f'{BASE_DIR}/dataset'
    
    # 强制保存路径为指定的 Saliency 目录
    RESULT_DIR = f'{BASE_DIR}/rebuttal/results_rebuttal/add_backbone_model/basenji2/CAGE_saliency_raw_results'
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    META_CSV = f'{DATASET_DIR}/gene_3000_borzoi_gencode_v41_hg38.csv'
    GTF_PATH = f'{DATASET_DIR}/gencode.v41.annotation.gtf.gz'
    FASTA_PATH = f'{DATASET_DIR}/human_genome_hg38/hg38.ml.fa'
    
    chrom, tss, strand, gene_id, gene_name = get_gene_meta_by_name(gene_name_arg, META_CSV)
    SNP_CSV = f'{DATASET_DIR}/gene_snps_hg38/{gene_name}_snps_hg38.csv'
    
    try:
        ref_seq, seq_start_pos, snp_meta_list = prepare_tensors(
            SNP_CSV, FASTA_PATH, chrom, center_pos=tss
        )
    except FileNotFoundError:
        print(f"⚠️ SNP file not found for {gene_name}, skipping.")
        return

    # ⚠️ 极其关键：允许输入序列计算梯度
    ref_seq = ref_seq.to(DEVICE)
    ref_seq.requires_grad = True 
    
    exon_regions = get_exons_from_gtf(gene_name, gene_id, GTF_PATH, tss, seq_start_pos)
    
    print("Loading Basenji2 Model & Weights...")
    model_basenji = Basenji2(basenji2_params["model"]).to(DEVICE).eval()
    model_basenji.load_state_dict(torch.load(basenji2_weights()))
    
    # Backbone 不参与优化，无需梯度
    for p in model_basenji.parameters(): p.requires_grad = False

    print(f"🚀 Computing Saliency for {gene_name} in {tissue_arg}...")
    
    # 1. 前向传播
    expr = calculate_expression_score(model_basenji, ref_seq, exon_regions, target_idx)
    
    # 2. 反向传播求导
    expr.backward()
    
    # 获取输入序列的梯度字典，形状为 (4, L)
    # 代表在对应位置、对应碱基上增加权重，对表达量预测值的线性影响
    grads = ref_seq.grad[0] 
    
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    results = []
    
    # 3. 提取所有候选 SNP 的 Saliency Score
    for snp in snp_meta_list:
        rel_pos = snp['rel_pos']
        ref_base = snp['ref']
        alt_base = snp['alt']
        
        grad_ref = grads[mapping[ref_base], rel_pos].item()
        grad_alt = grads[mapping[alt_base], rel_pos].item()
        
        # 核心逻辑：突变成 Alt 带来的预期收益
        saliency_score = grad_alt - grad_ref
        
        results.append({
            'Pos': snp['abs_pos'],
            'Ref': ref_base,
            'Alt': alt_base,
            'Grad_Ref': grad_ref,
            'Grad_Alt': grad_alt,
            'Saliency_Score': saliency_score
        })
        
    df_res = pd.DataFrame(results)
    # 按 Saliency 分数从高到低排序，最前面的就是最推荐突变的
    df_res = df_res.sort_values(by='Saliency_Score', ascending=False)
    
    # 保存结果，兼容旧 Benchmark 脚本的列名要求
    csv_filename = f"{RESULT_DIR}/{gene_name}_saliency.csv"
    df_res.to_csv(csv_filename, index=False)
    print(f"✅ Saliency calculated. Saved {len(df_res)} SNPs to {csv_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gene', type=str, required=True, help='Name of the gene')
    parser.add_argument('--tissue', type=str, default='blood', help='Tissue name')
    
    args = parser.parse_args()
    run_saliency(gene_name_arg=args.gene, tissue_arg=args.tissue)
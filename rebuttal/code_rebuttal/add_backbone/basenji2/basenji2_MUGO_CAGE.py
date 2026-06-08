'''
https://github.com/d-laub/basenji2-pytorch using this repo 
paper: Cross-species regulatory sequence activity prediction, PLOS 2022, basenji2. 
'''

'''
is for Basenji2 RNA-seq (CAGE) track. 
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
from basenji2_pytorch import Basenji2, basenji2_params, basenji2_weights

torch.backends.cudnn.enabled = False 

# ==========================================
# 0. 核心配置 (Basenji2 Track ID)
# ==========================================
TISSUE_MAP = {
    'blood': 4950,   
    'brain': 4680,   
    'liver': 4686,   
    'heart': 4684,   
    'muscle': 4691,  
    'Pancreas': 4946 
}

# ==========================================
# 1. Multi-Head Selector
# ==========================================
class MultiHeadSelector(nn.Module):
    def __init__(self, num_snps, snp_positions, k=10):
        super().__init__()
        self.register_buffer('snp_positions', snp_positions)
        self.k = k
        self.logits = nn.Parameter(torch.randn(k, num_snps) * 0.01)

    def forward(self, ref_seq, alt_seq, tau=1.0):
        gumbels = -torch.empty_like(self.logits).exponential_().log()
        gumbel_logits = (self.logits + gumbels) / tau
        soft_masks_k = F.softmax(gumbel_logits, dim=-1)
        index = soft_masks_k.max(dim=-1, keepdim=True)[1]
        hard_masks_k = torch.zeros_like(soft_masks_k).scatter_(-1, index, 1.0)
        masks_k = (hard_masks_k - soft_masks_k).detach() + soft_masks_k
        combined_mask = masks_k.sum(dim=0)
        final_mask = torch.clamp(combined_mask, 0.0, 1.0)
        full_seq_mask = torch.zeros(ref_seq.shape[-1], device=ref_seq.device)
        full_seq_mask[self.snp_positions] = final_mask
        full_seq_mask = full_seq_mask.view(1, 1, -1)
        input_seq = ref_seq * (1 - full_seq_mask) + alt_seq * full_seq_mask
        return input_seq, final_mask, soft_masks_k

# ==========================================
# 2. Data Utils
# ==========================================
SEQ_LEN = 131072 # Basenji2 感受野
POOL_SIZE = 128  # Basenji2 分辨率

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
    alt_tensor = ref_tensor.clone()
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"SNP file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    df.rename(columns={'POS_hg38': 'pos', 'ALT': 'alt'}, inplace=True)
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
    print(f"Found {len(snp_indices_list)} SNPs in window.")
    return ref_tensor, alt_tensor, torch.tensor(snp_indices_list).long(), start, snp_meta_list

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
# 3. Main Routine
# ==========================================
def train(gene_name_arg, k_arg, tissue_arg, track_idx_arg):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    LR, STEPS, INIT_TAU, MIN_TAU = 0.05, 200, 5.0, 0.1
    
    if track_idx_arg is not None:
        target_idx = track_idx_arg 
        print(f"🔧 Using Manually Override Track ID: {target_idx}")
    elif tissue_arg in TISSUE_MAP:
        target_idx = TISSUE_MAP[tissue_arg] 
        print(f"🔍 Auto-detected Track ID for '{tissue_arg}': {target_idx}")
    else:
        raise ValueError(f"Unknown tissue '{tissue_arg}'")

    BASE_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP'
    DATASET_DIR = f'{BASE_DIR}/dataset'
    
    # 强制使用你指定的 Rebuttal 结果保存路径
    RESULT_DIR = f'{BASE_DIR}/rebuttal/results_rebuttal/add_backbone_model/basenji2/CAGE_raw_results'
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    print(f"📂 Results will be saved to: {RESULT_DIR}")
    
    META_CSV = f'{DATASET_DIR}/gene_3000_borzoi_gencode_v41_hg38.csv'
    GTF_PATH = f'{DATASET_DIR}/gencode.v41.annotation.gtf.gz'
    FASTA_PATH = f'{DATASET_DIR}/human_genome_hg38/hg38.ml.fa'
    
    # 用 gene_name 查 metadata
    chrom, tss, strand, gene_id, gene_name = get_gene_meta_by_name(gene_name_arg, META_CSV)
    SNP_CSV = f'{DATASET_DIR}/gene_snps_hg38/{gene_name}_snps_hg38.csv'
    
    try:
        ref_seq, alt_seq, snp_positions, seq_start_pos, snp_meta_list = prepare_tensors(
            SNP_CSV, FASTA_PATH, chrom, center_pos=tss
        )
    except FileNotFoundError:
        print(f"⚠️ SNP file not found for {gene_name}, skipping.")
        return

    ref_seq = ref_seq.to(DEVICE)
    alt_seq = alt_seq.to(DEVICE)
    snp_positions = snp_positions.to(DEVICE)
    exon_regions = get_exons_from_gtf(gene_name, gene_id, GTF_PATH, tss, seq_start_pos)
    
    print("Loading Basenji2 Model & Weights...")
    model_basenji = Basenji2(basenji2_params["model"]).to(DEVICE).eval()
    model_basenji.load_state_dict(torch.load(basenji2_weights()))
    
    for p in model_basenji.parameters(): p.requires_grad = False

    with torch.no_grad():
        baseline_expr = calculate_expression_score(model_basenji, ref_seq, exon_regions, target_idx)
    print(f"📉 Baseline Expression ({tissue_arg}): {baseline_expr.item():.4f}")
    
    selector = MultiHeadSelector(len(snp_positions), snp_positions, k=k_arg).to(DEVICE)
    optimizer = torch.optim.Adam(selector.parameters(), lr=LR)

    print(f"🚀 Optimizing {gene_name} for {tissue_arg} using Basenji2...")
    history_log = []

    for step in range(STEPS):
        tau = INIT_TAU * ((MIN_TAU / INIT_TAU) ** (step / STEPS))
        optimizer.zero_grad()
        input_seq, final_mask, soft_masks_k = selector(ref_seq, alt_seq, tau=tau)
        expr = calculate_expression_score(model_basenji, input_seq, exon_regions, target_idx)
        loss = -expr
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            total_votes = soft_masks_k.sum(dim=0) 
            top_scores, top_indices = torch.topk(total_votes, k_arg)
            
            row_data = {
                "Step": step, "Loss": loss.item(), "Gain": expr.item() - baseline_expr.item(),
                "Baseline": baseline_expr.item(), "Tau": tau,
                "Tissue": tissue_arg, "TrackIdx": target_idx
            }
            for i in range(k_arg):
                idx = top_indices[i].item()
                snp_info = snp_meta_list[idx]
                row_data[f"Rank{i+1}_Pos"] = snp_info['abs_pos']
                row_data[f"Rank{i+1}_RefAlt"] = f"{snp_info['ref']}->{snp_info['alt']}"
                row_data[f"Rank{i+1}_Score"] = top_scores[i].item()
            history_log.append(row_data)

        # 改成了每 10 个 epoch 打印一次
        if step % 10 == 0:
            print(f"Step {step:3d} | Gain: {expr.item() - baseline_expr.item():+.2f}")

    # 文件名加上了组织和K值，防止覆盖
    csv_filename = f"{RESULT_DIR}/{gene_name}_{tissue_arg}_K{k_arg}_basenji2_optim_log.csv"
    pd.DataFrame(history_log).to_csv(csv_filename, index=False)
    print(f"✅ Finished {gene_name}. Saved to {csv_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 彻底移除了 index 参数，只保留 gene 参数
    parser.add_argument('--gene', type=str, required=True, help='Name of the gene (e.g., ELP1)')
    parser.add_argument('--k', type=int, default=10, help='Number of heads')
    parser.add_argument('--tissue', type=str, default='blood', help='Tissue name (e.g., blood, brain, liver)')
    parser.add_argument('--manual_track_id', type=int, default=None, help='Override track ID manually')
    
    args = parser.parse_args()
    
    train(gene_name_arg=args.gene, 
          k_arg=args.k, 
          tissue_arg=args.tissue,
          track_idx_arg=args.manual_track_id)
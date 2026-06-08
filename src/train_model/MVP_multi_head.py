'''
is for borzoi RNA-seq track. 
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

'''
V7 Update: 
1. Simplified Arguments: Only --tissue is needed. Track ID is looked up automatically.
2. Organized Output: Results are saved in 'results/{tissue}/'.
when run: python MVP_multi_head.py --tissue blood, --index GENE_INDEX, and K default is 10.  
'''

# ==========================================
# 0. 核心配置 (在这里修改 Track ID)
# ==========================================

# ⚠️⚠️⚠️ 请在这里填入你查到的 Borzoi Track Index ⚠️⚠️⚠️
# 你只需要改这里，后面都不用动
TISSUE_MAP = {
    'blood': 7531,  # GTEx Whole Blood RNA-seq 
    'brain': 7539,  # GTEx brain track.  
    'liver': 7563,  # GTEx liver track.   
    'heart': 7557,  # GTEx heart track.   
    'muscle': 7569,  # GTEx muscle track.   
    'Pancreas': 7577,  # GTEx pancreas track.    
    'kidney': 7560,  # GTEx kidney track.    
    'lung': 7566,  # GTEx lung track.    
}

# ==========================================
# 1. Multi-Head Selector (No Changes)
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
# 2. Data Utils (No Changes)
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
    # print(f"Loading Genome from {fasta_path}...") # 减少刷屏
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
            total_expr += output[:, target_track_idx, out_start:out_end].sum()
    return total_expr

# ==========================================
# 3. Main Routine
# ==========================================

def train(gene_index_arg, k_arg, tissue_arg, track_idx_arg):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    LR, STEPS, INIT_TAU, MIN_TAU = 0.05, 200, 5.0, 0.1
    
    # 1. 确定 Track Index
    if track_idx_arg is not None:
        target_idx = track_idx_arg # 如果手动指定了 ID，优先用手动的
        print(f"🔧 Using Manually Override Track ID: {target_idx}")
    elif tissue_arg in TISSUE_MAP:
        target_idx = TISSUE_MAP[tissue_arg] # 否则查表
        print(f"🔍 Auto-detected Track ID for '{tissue_arg}': {target_idx}")
    else:
        raise ValueError(f"Unknown tissue '{tissue_arg}' and no track_idx provided. Available: {list(TISSUE_MAP.keys())}")

    # 2. 自动建立分层文件夹
    # 路径变成: results/blood/
    # 如果你想保留K值区分，也可以用 results/blood_K10/，这里按你要求用 tissue 分层
    BASE_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP'
    DATASET_DIR = f'{BASE_DIR}/dataset'
    
    # [修改] 干净的文件夹结构
    RESULT_DIR = f'{BASE_DIR}/results/{tissue_arg}_K{k_arg}_borzoi_modeltrain_res'
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    print(f"📂 Results will be saved to: {RESULT_DIR}")
    
    # 3. 加载数据
    META_CSV = f'{DATASET_DIR}/gene_3000_borzoi_gencode_v41_hg38.csv'
    GTF_PATH = f'{DATASET_DIR}/gencode.v41.annotation.gtf.gz'
    FASTA_PATH = f'{DATASET_DIR}/human_genome_hg38/hg38.ml.fa'
    
    chrom, tss, strand, gene_id, gene_name = get_gene_meta_by_index(gene_index_arg, META_CSV)
    SNP_CSV = f'{DATASET_DIR}/gene_snps_hg38/{gene_name}_snps_hg38.csv'
    
    # 4. 准备 Tensor
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
    
    # 5. 模型初始化
    print("Loading Borzoi...")
    model_borzoi = Borzoi.from_pretrained('johahi/borzoi-replicate-0').to(DEVICE).eval()
    for p in model_borzoi.parameters(): p.requires_grad = False

    # Baseline
    with torch.no_grad():
        baseline_expr = calculate_expression_score(model_borzoi, ref_seq, exon_regions, target_idx)
    print(f"📉 Baseline Expression ({tissue_arg}): {baseline_expr.item():.4f}")
    
    # 6. 优化循环
    selector = MultiHeadSelector(len(snp_positions), snp_positions, k=k_arg).to(DEVICE)
    optimizer = torch.optim.Adam(selector.parameters(), lr=LR)

    print(f"🚀 Optimizing {gene_name} for {tissue_arg}...")
    history_log = []

    for step in range(STEPS):
        tau = INIT_TAU * ((MIN_TAU / INIT_TAU) ** (step / STEPS))
        
        optimizer.zero_grad()
        input_seq, final_mask, soft_masks_k = selector(ref_seq, alt_seq, tau=tau)
        expr = calculate_expression_score(model_borzoi, input_seq, exon_regions, target_idx)
        loss = -expr
        loss.backward()
        optimizer.step()
        
        # Logging
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

        if step % 50 == 0:
            print(f"Step {step:3d} | Gain: {expr.item() - baseline_expr.item():+.2f}")

    # 7. 保存结果
    csv_filename = f"{RESULT_DIR}/{gene_name}_optim_log.csv"
    pd.DataFrame(history_log).to_csv(csv_filename, index=False)
    print(f"✅ Finished {gene_name}. Saved to {csv_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, required=True, help='Row index of the gene')
    parser.add_argument('--k', type=int, default=10, help='Number of heads')
    
    # 简化后的参数：只需要传 tissue
    parser.add_argument('--tissue', type=str, default='blood', 
                        help='Tissue name (e.g., blood, brain, liver). Configured in script.')
    
    # 这是一个隐藏的高级参数，万一你想手动指定Track ID，也可以覆盖
    parser.add_argument('--manual_track_id', type=int, default=None, 
                        help='Override track ID manually')
    
    args = parser.parse_args()
    
    train(gene_index_arg=args.index, 
          k_arg=args.k, 
          tissue_arg=args.tissue,
          track_idx_arg=args.manual_track_id)
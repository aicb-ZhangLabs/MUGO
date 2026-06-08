'''
Using Enformer. https://github.com/lucidrains/enformer-pytorch. using this enformer torch repo to train the model. Enformer用的也是hg38，
enformer只有CAGE sum TSS track想想怎么说，borzoi也用类似的track来对比, enformer 也是用的hg38，不用换了。
enformer 是TSS周围10个bin，每个bin 128bp，sum as gene expression，然后borzoi 32bp所以sum TSS周围40个bin作为 gene expression。 
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
from enformer_pytorch import from_pretrained

torch.backends.cudnn.enabled = False 

'''
Enformer Version Update (CAGE Specialized):
1. Model: Enformer (EleutherAI/enformer-official-rough).
2. Logic: Optimization targets TSS-centered bins (CAGE signal), not Exon sums.
'''

# ==========================================
# 0. 核心配置
# ==========================================

TISSUE_MAP = {
    # [Blood] 对应 4950: CAGE:blood, adult, pool1
    # 这是全血样本，最接近 GTEx Whole Blood
    'blood': 4950,   
    
    # [Brain] 对应 4680: CAGE:brain, adult, pool1
    # 最通用的成人脑组织混合样本
    'brain': 4680,   
    
    # [Liver] 对应 4686: CAGE:liver, adult, pool1
    # 标准成人肝脏样本
    'liver': 4686,   
    
    # [Heart] 对应 4684: CAGE:heart, adult, pool1
    # 标准成人心脏样本
    'heart': 4684,   
    
    # [Muscle] 对应 4691: CAGE:skeletal muscle, adult, pool1
    # 注意：生物学上"Muscle"通常默认指骨骼肌 (Skeletal)，而不是平滑肌 (Smooth, 4945)
    'muscle': 4691,  
    
    # [Pancreas] 对应 4946: CAGE:pancreas, adult
    # 列表中没有 pancreas pool1，这是最标准的成人胰腺样本
    'Pancreas': 4946 
}
# ==========================================
# 1. Multi-Head Selector (无变化)
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
SEQ_LEN = 196608 
OUTPUT_BINS = 896
BIN_SIZE = 128
PADDING_OFFSET = 40960 

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

def get_gene_meta_by_index(target_index, meta_csv_path):
    print(f"📖 Reading Metadata from: {meta_csv_path}")
    df = pd.read_csv(meta_csv_path)
    row = df.iloc[target_index]
    chrom = f"chr{row['chr']}" 
    tss = int(row['pos'])
    return chrom, tss, row['strand'], row['gene_ID'], row['gene_name']

# [修改 A] 新增：只计算 TSS 窗口的 Bins
def get_tss_bin_range(tss_abs_pos, seq_start_pos, window_bins=2):
    """
    window_bins=2 意味着取中心 bin 左右各 2 个，共 5 个 bin (5 * 128bp = 640bp)
    如果想宽一点（如你说的10个bin），可以设 window_bins=5
    """
    pred_start_abs = seq_start_pos + PADDING_OFFSET
    
    # 计算 TSS 落在第几个 Bin
    center_bin = (tss_abs_pos - pred_start_abs) // BIN_SIZE
    
    # 边界检查
    start_bin = max(0, center_bin - window_bins)
    end_bin = min(OUTPUT_BINS, center_bin + window_bins + 1)
    
    print(f"🎯 TSS Target: Bin {start_bin} to {end_bin} (Center: {center_bin})")
    return start_bin, end_bin

# [修改 B] 计算 Score 时只取 TSS window sum
def calculate_expression_score_cage(model, input_seq, tss_bin_start, tss_bin_end, target_track_idx):
    input_permuted = input_seq.permute(0, 2, 1) 
    output = model(input_permuted) 
    pred = output['human'] # (1, 896, 5313)
    
    # 只 Sum TSS 附近的信号
    total_expr = pred[:, tss_bin_start:tss_bin_end, target_track_idx].sum()
    return total_expr

# ==========================================
# 3. Main Routine
# ==========================================
def train(gene_index_arg, k_arg, tissue_arg, track_idx_arg):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    LR, STEPS, INIT_TAU, MIN_TAU = 0.05, 200, 5.0, 0.1
    
    if track_idx_arg is not None:
        target_idx = track_idx_arg 
        print(f"🔧 Using Manually Override Track ID: {target_idx}")
    elif tissue_arg in TISSUE_MAP:
        target_idx = TISSUE_MAP[tissue_arg] 
        print(f"🔍 Auto-detected Track ID for '{tissue_arg}': {target_idx}")
    else:
        raise ValueError(f"Unknown tissue '{tissue_arg}'.")

    BASE_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP'
    DATASET_DIR = f'{BASE_DIR}/dataset'
    RESULT_DIR = f'{BASE_DIR}/results/{tissue_arg}_K{k_arg}_enformer_modeltrain_CAGE_res'
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    print(f"📂 Results will be saved to: {RESULT_DIR}")
    
    META_CSV = f'{DATASET_DIR}/gene_3000_borzoi_gencode_v41_hg38.csv'
    FASTA_PATH = f'{DATASET_DIR}/human_genome_hg38/hg38.ml.fa'
    
    chrom, tss, strand, gene_id, gene_name = get_gene_meta_by_index(gene_index_arg, META_CSV)
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
    
    # [关键修改] 获取 TSS 窗口 (这里设为左右各 5 个 bin，共 11 个 bin，覆盖 ~1.4kb)
    tss_bin_start, tss_bin_end = get_tss_bin_range(tss, seq_start_pos, window_bins=5)
    
    print("Loading Enformer...")
    model_enformer = from_pretrained('EleutherAI/enformer-official-rough').to(DEVICE).eval()
    for p in model_enformer.parameters(): p.requires_grad = False

    with torch.no_grad():
        # [关键修改] 调用新的 CAGE score 计算函数
        baseline_expr = calculate_expression_score_cage(model_enformer, ref_seq, tss_bin_start, tss_bin_end, target_idx)
    print(f"📉 Baseline CAGE Signal ({tissue_arg}): {baseline_expr.item():.4f}")
    
    selector = MultiHeadSelector(len(snp_positions), snp_positions, k=k_arg).to(DEVICE)
    optimizer = torch.optim.Adam(selector.parameters(), lr=LR)

    print(f"🚀 Optimizing {gene_name} for {tissue_arg} (Enformer CAGE)...")
    history_log = []

    for step in range(STEPS):
        tau = INIT_TAU * ((MIN_TAU / INIT_TAU) ** (step / STEPS))
        
        optimizer.zero_grad()
        input_seq, final_mask, soft_masks_k = selector(ref_seq, alt_seq, tau=tau)
        
        # [关键修改] 优化目标也改为 CAGE score
        expr = calculate_expression_score_cage(model_enformer, input_seq, tss_bin_start, tss_bin_end, target_idx)
        loss = -expr
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            total_votes = soft_masks_k.sum(dim=0) 
            top_scores, top_indices = torch.topk(total_votes, k_arg)
            
            row_data = {
                "Step": step, "Loss": loss.item(), "Gain": expr.item() - baseline_expr.item(),
                "Baseline": baseline_expr.item(), "Tau": tau,
                "Tissue": tissue_arg, "TrackIdx": target_idx,
                "Model": "Enformer_CAGE"
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

    csv_filename = f"{RESULT_DIR}/{gene_name}_enformer_optim_log.csv"
    pd.DataFrame(history_log).to_csv(csv_filename, index=False)
    print(f"✅ Finished {gene_name}. Saved to {csv_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, required=True, help='Row index of the gene')
    parser.add_argument('--k', type=int, default=10, help='Number of heads')
    parser.add_argument('--tissue', type=str, default='blood', help='Tissue name')
    parser.add_argument('--manual_track_id', type=int, default=None, help='Override track ID')
    args = parser.parse_args()
    train(args.index, args.k, args.tissue, args.manual_track_id)
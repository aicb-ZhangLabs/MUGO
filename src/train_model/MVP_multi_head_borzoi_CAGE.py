'''
Borzoi CAGE track. 
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
Borzoi CAGE Version:
1. Logic: Optimization targets TSS-centered bins (CAGE signal).
2. Window: Sum of center 40 bins (approx 1280bp), matching Enformer's coverage.
'''

# ==========================================
# 0. 核心配置 (Borzoi CAGE Track ID)
# ==========================================

# ⚠️⚠️⚠️ 请务必确认这些 ID！这是 Borzoi (Basenji) 的 CAGE ID ⚠️⚠️⚠️
# 这里的 ID 只是示例（或者是 RNA-seq 的），请查找 Borzoi 的 targets.txt 替换为 CAGE track index
# TISSUE_MAP: Store (Plus_Strand_ID, Minus_Strand_ID)
TISSUE_MAP = {
    'blood': (550, 551), # brozoi have +- strand. for gene on plus strand, use Plus_Strand_ID, and minus strand, use Minus_Strand_ID. 
    'brain': (10, 11),
    'liver': (22, 23),
    'heart': (18, 19),
    'muscle': (32, 33),
    'Pancreas': (542, 543)
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

SEQ_LEN = 524288 

def seq_to_one_hot(seq_str):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    one_hot = np.zeros((4, len(seq_str)), dtype=np.float32)
    for i, base in enumerate(seq_str):
        if base in mapping:
            one_hot[mapping[base], i] = 1.0
    return torch.tensor(one_hot)

def prepare_tensors(csv_path, fasta_path, chrom, center_pos, seq_len=SEQ_LEN):
    genome = pyfaidx.Fasta(fasta_path)
    # 输入序列以 TSS 为中心
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

# [修改 A] 新的表达量计算函数：针对 CAGE 信号
def calculate_expression_score_cage(model, input_seq, target_track_idx):
    output = model(input_seq) # Borzoi output shape usually: (1, Num_Tracks, Output_Len)
    
    # 动态获取输出长度
    output_len = output.shape[-1]
    
    # 因为 Input 是以 TSS 为中心 (prepare_tensors 里 center_pos=tss)
    # 所以 TSS 就在 Output Tensor 的正中间
    center_bin = output_len // 2
    
    # [关键] 取 TSS 周围 40 个 bin (左右各 20)
    # Borzoi bin size = 32bp, 40 bins ≈ 1280bp
    window_bins = 20
    start_bin = max(0, center_bin - window_bins)
    end_bin = min(output_len, center_bin + window_bins)
    
    # 求和
    total_expr = output[:, target_track_idx, start_bin:end_bin].sum()
    return total_expr

# ==========================================
# 3. Main Routine
# ==========================================

def train(gene_index_arg, k_arg, tissue_arg, track_idx_arg):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    LR, STEPS, INIT_TAU, MIN_TAU = 0.05, 200, 5.0, 0.1
    
    # 1. 确定 Track Index
    if track_idx_arg is not None:
        target_idx = track_idx_arg 
        print(f"🔧 Using Manually Override Track ID: {target_idx}")
    elif tissue_arg in TISSUE_MAP:
        target_idx = TISSUE_MAP[tissue_arg]
        print(f"🔍 Auto-detected Track ID for '{tissue_arg}': {target_idx}")
    else:
        raise ValueError(f"Unknown tissue '{tissue_arg}' and no track_idx provided.")

    BASE_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP'
    DATASET_DIR = f'{BASE_DIR}/dataset'
    
    # [修改] 文件夹命名加 _CAGE 区分
    RESULT_DIR = f'{BASE_DIR}/results/{tissue_arg}_K{k_arg}_borzoi_CAGE_modeltrain_res'
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    print(f"📂 Results will be saved to: {RESULT_DIR}")
    
    # 3. 加载数据
    META_CSV = f'{DATASET_DIR}/gene_3000_borzoi_gencode_v41_hg38.csv'
    FASTA_PATH = f'{DATASET_DIR}/human_genome_hg38/hg38.ml.fa'
    
    chrom, tss, strand, gene_id, gene_name = get_gene_meta_by_index(gene_index_arg, META_CSV)
    # 建议修改为 (更安全)
    if str(strand).strip() == '+':
        target_idx = TISSUE_MAP[tissue_arg][0]
    elif str(strand).strip() == '-':
        target_idx = TISSUE_MAP[tissue_arg][1]
    else:
        # 万一出现 '.' 或其他符号，默认用 + 链，并打印警告
        print(f"⚠️ Warning: Unknown strand '{strand}', defaulting to Plus (+)")
        target_idx = TISSUE_MAP[tissue_arg][0]

    print(f"🧬 Gene Strand: {strand} | Using Track ID: {target_idx}")

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
    
    # 5. 模型初始化
    print("Loading Borzoi...")
    model_borzoi = Borzoi.from_pretrained('johahi/borzoi-replicate-0').to(DEVICE).eval()
    for p in model_borzoi.parameters(): p.requires_grad = False

    # Baseline (调用新的 CAGE 计算函数)
    with torch.no_grad():
        baseline_expr = calculate_expression_score_cage(model_borzoi, ref_seq, target_idx)
    print(f"📉 Baseline CAGE Signal ({tissue_arg}): {baseline_expr.item():.4f}")
    
    # 6. 优化循环
    selector = MultiHeadSelector(len(snp_positions), snp_positions, k=k_arg).to(DEVICE)
    optimizer = torch.optim.Adam(selector.parameters(), lr=LR)

    print(f"🚀 Optimizing {gene_name} for {tissue_arg} (CAGE)...")
    history_log = []

    for step in range(STEPS):
        tau = INIT_TAU * ((MIN_TAU / INIT_TAU) ** (step / STEPS))
        
        optimizer.zero_grad()
        input_seq, final_mask, soft_masks_k = selector(ref_seq, alt_seq, tau=tau)
        
        # 优化目标改为 CAGE score
        expr = calculate_expression_score_cage(model_borzoi, input_seq, target_idx)
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
                "Tissue": tissue_arg, "TrackIdx": target_idx,
                "Model": "Borzoi_CAGE"
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
    csv_filename = f"{RESULT_DIR}/{gene_name}_borzoi_CAGE_optim_log.csv"
    pd.DataFrame(history_log).to_csv(csv_filename, index=False)
    print(f"✅ Finished {gene_name}. Saved to {csv_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, required=True, help='Row index of the gene')
    parser.add_argument('--k', type=int, default=10, help='Number of heads')
    parser.add_argument('--tissue', type=str, default='blood', 
                        help='Tissue name (e.g., blood, brain, liver). Configured in script.')
    parser.add_argument('--manual_track_id', type=int, default=None, 
                        help='Override track ID manually')
    
    args = parser.parse_args()
    
    train(gene_index_arg=args.index, 
          k_arg=args.k, 
          tissue_arg=args.tissue,
          track_idx_arg=args.manual_track_id)
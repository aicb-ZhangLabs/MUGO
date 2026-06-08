'''
is for borzoi DNAse-seq track optimization.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import pyfaidx
import os
import argparse
from borzoi_pytorch import Borzoi

torch.backends.cudnn.enabled = False 

'''
V1 DNAse Update: 
1. Target: DNAse-seq signal maximization (Chromatin Accessibility).
2. Window: Center +/- 1000bp (DHS Peak friendly).
3. Logic: Maximizes the total accessibility (Sum) in the window.
'''

# ==========================================
# 0. 核心配置 (在这里修改 DNAse Track ID)
# ==========================================

# ⚠️⚠️⚠️ 请在这里填入你查到的 Borzoi DNAse-seq Track Index ⚠️⚠️⚠️
# 提示: DNAse Track ID 通常与 ATAC 不同，请查阅 metadata!
TISSUE_DNASE_MAP = {
    'blood': 1524,  # DNASE:CD4-positive, alpha-beta T cell male adult (37 years)
    'brain': 1277,  # DNASE:frontal cortex male adult (27 years) and male adult (35 years)
    'liver': 1303,  # DNASE:HepG2
    'heart': 1474,  # DNASE:heart left ventricle female adult (53 years)
    'muscle': 1320,  # DNASE:psoas muscle male adult (27 years) and male adult (35 years)
    'Pancreas': 1533,  # DNASE:pancreas male adult (34 years)
}

# ==========================================
# 1. Multi-Head Selector (复用原逻辑)
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
# 2. Data Utils (复用简化版)
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

# ==========================================
# 3. New Loss Function: DNAse Window Score
# ==========================================

def calculate_dnase_score(model, input_seq, strand, target_track_idx):
    """
    计算 TSS/Center 附近的 DNAse 信号总和。
    窗口: Center +/- 1000bp (总共 2kb)。
    """
    # 1. 获取输出 [Batch, Tracks, Length]
    output = model(input_seq)
    
    # ✅ [修复] 正确解包维度
    batch_size, n_tracks, n_bins = output.shape
    
    center_bin = n_bins // 2
    bin_size = 32  # Borzoi 分辨率
    
    # [关键参数] 窗口大小: 1kb 半径
    radius_bp = 1000 
    radius_bins = radius_bp // bin_size # ~31 bins
    
    # 确定切片范围 (以 Center 为锚点)
    start_bin = center_bin - radius_bins
    end_bin = center_bin + radius_bins
        
    # 边界保护
    start_bin = max(0, start_bin)
    end_bin = min(n_bins, end_bin)
    
    # ✅ [修复] 正确切片: [Batch, Track, Length]
    # 之前是 output[:, start:end, track]，这是错的
    target_signal = output[:, target_track_idx, start_bin:end_bin]
    
    # Sum up the signal (Total Accessibility)
    total_dnase = target_signal.sum()
    
    return total_dnase

# ==========================================
# 4. Main Routine
# ==========================================

def train(gene_index_arg, k_arg, tissue_arg, track_idx_arg):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    LR, STEPS, INIT_TAU, MIN_TAU = 0.05, 200, 5.0, 0.1
    
    # 1. 确定 Track Index
    if track_idx_arg is not None:
        target_idx = track_idx_arg 
        print(f"🔧 Using Manually Override DNAse Track ID: {target_idx}")
    elif tissue_arg in TISSUE_DNASE_MAP:
        target_idx = TISSUE_DNASE_MAP[tissue_arg]
        print(f"🔍 Auto-detected DNAse Track ID for '{tissue_arg}': {target_idx}")
    else:
        raise ValueError(f"Unknown tissue '{tissue_arg}' and no track_idx provided. Please update TISSUE_DNASE_MAP.")

    # 2. 文件夹路径
    BASE_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP'
    DATASET_DIR = f'{BASE_DIR}/dataset'
    
    # 结果存放在 _DNAse_modeltrain_res 文件夹
    RESULT_DIR = f'{BASE_DIR}/results/{tissue_arg}_K{k_arg}_borzoi_DNAse_modeltrain_res'
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    print(f"📂 Results will be saved to: {RESULT_DIR}")
    
    # 3. 加载数据
    META_CSV = f'{DATASET_DIR}/gene_3000_borzoi_gencode_v41_hg38.csv'
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
    
    # 5. 模型初始化
    print("Loading Borzoi...")
    model_borzoi = Borzoi.from_pretrained('johahi/borzoi-replicate-0').to(DEVICE).eval()
    for p in model_borzoi.parameters(): p.requires_grad = False

    # Baseline Calculation (DNAse)
    with torch.no_grad():
        baseline_dnase = calculate_dnase_score(model_borzoi, ref_seq, strand, target_idx)
    print(f"📉 Baseline DNAse ({tissue_arg}): {baseline_dnase.item():.4f}")
    
    # 6. 优化循环
    selector = MultiHeadSelector(len(snp_positions), snp_positions, k=k_arg).to(DEVICE)
    optimizer = torch.optim.Adam(selector.parameters(), lr=LR)

    print(f"🚀 Optimizing {gene_name} DNAse for {tissue_arg}...")
    history_log = []

    for step in range(STEPS):
        tau = INIT_TAU * ((MIN_TAU / INIT_TAU) ** (step / STEPS))
        
        optimizer.zero_grad()
        input_seq, final_mask, soft_masks_k = selector(ref_seq, alt_seq, tau=tau)
        
        # 计算 DNAse Loss
        dnase_signal = calculate_dnase_score(model_borzoi, input_seq, strand, target_idx)
        loss = -dnase_signal # Maximize DNAse
        
        loss.backward()
        optimizer.step()
        
        # Logging
        with torch.no_grad():
            total_votes = soft_masks_k.sum(dim=0) 
            top_scores, top_indices = torch.topk(total_votes, k_arg)
            
            row_data = {
                "Step": step, "Loss": loss.item(), "Gain": dnase_signal.item() - baseline_dnase.item(),
                "Baseline": baseline_dnase.item(), "Tau": tau,
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
            print(f"Step {step:3d} | DNAse Gain: {dnase_signal.item() - baseline_dnase.item():+.2f}")

    # 7. 保存结果
    csv_filename = f"{RESULT_DIR}/{gene_name}_DNAse_optim_log.csv"
    pd.DataFrame(history_log).to_csv(csv_filename, index=False)
    print(f"✅ Finished {gene_name}. Saved to {csv_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, required=True, help='Row index of the gene')
    parser.add_argument('--k', type=int, default=10, help='Number of heads')
    parser.add_argument('--tissue', type=str, default='blood', 
                        help='Tissue name. Make sure to update TISSUE_DNASE_MAP.')
    parser.add_argument('--manual_track_id', type=int, default=None, 
                        help='Override track ID manually')
    
    args = parser.parse_args()
    
    train(gene_index_arg=args.index, 
          k_arg=args.k, 
          tissue_arg=args.tissue,
          track_idx_arg=args.manual_track_id)
'''
ExPecto Saliency (Input Gradient) Baseline Script
Path: /home/dongbos/Combine_optim_Borzoi_SNP/rebuttal/code_rebuttal/add_backbone/expecto/expecto_CAGE_saliency.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import pyfaidx
import os
import argparse

torch.backends.cudnn.enabled = False 

# ==========================================
# 0. 核心配置 (ExPecto GTEx Track 映射)
# ==========================================
TISSUE_MAP = {
    'blood': 'Whole_Blood',
    'brain': 'Brain_Cortex', 
    'liver': 'Liver',
    'heart': 'Heart_Left_Ventricle',
    'muscle': 'Muscle_Skeletal',
    'pancreas': 'Pancreas'
}

# ==========================================
# 1. 核心网络架构 (真实的 Beluga 底座 - 完美对齐权重)
# ==========================================
class Beluga(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(4, 320, 8)
        self.conv2 = nn.Conv1d(320, 320, 8)
        self.conv3 = nn.Conv1d(320, 480, 8)
        self.conv4 = nn.Conv1d(480, 480, 8)
        self.conv5 = nn.Conv1d(480, 640, 8) 
        self.conv6 = nn.Conv1d(640, 640, 8) 
        
        self.linear1 = None
        self.linear2 = None

    def load_weights(self, pth_path):
        state = torch.load(pth_path, map_location='cpu')
        if 'state_dict' in state:
            state = state['state_dict']
            
        lin1_w = state['model.1.2.1.weight']
        lin1_b = state['model.1.2.1.bias']
        lin2_w = state['model.1.4.1.weight']
        lin2_b = state['model.1.4.1.bias']
        
        self.linear1 = nn.Linear(lin1_w.shape[1], lin1_w.shape[0])
        self.linear2 = nn.Linear(lin2_w.shape[1], lin2_w.shape[0])
        
        new_state = {
            'conv1.weight': state['model.0.0.weight'].squeeze(2), 'conv1.bias': state['model.0.0.bias'],
            'conv2.weight': state['model.0.2.weight'].squeeze(2), 'conv2.bias': state['model.0.2.bias'],
            'conv3.weight': state['model.0.6.weight'].squeeze(2), 'conv3.bias': state['model.0.6.bias'],
            'conv4.weight': state['model.0.8.weight'].squeeze(2), 'conv4.bias': state['model.0.8.bias'],
            'conv5.weight': state['model.0.12.weight'].squeeze(2), 'conv5.bias': state['model.0.12.bias'],
            'conv6.weight': state['model.0.14.weight'].squeeze(2), 'conv6.bias': state['model.0.14.bias'],
            'linear1.weight': lin1_w, 'linear1.bias': lin1_b,
            'linear2.weight': lin2_w, 'linear2.bias': lin2_b,
        }
        self.load_state_dict(new_state)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, 4)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool1d(x, 4)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        return x

# ==========================================
# 2. ExPecto 组装机 
# ==========================================
class ExPectoModel(nn.Module):
    def __init__(self, beluga_pth, expecto_weights_pt, strand='+'):
        super().__init__()
        
        self.beluga = Beluga()
        self.beluga.load_weights(beluga_pth)
        for param in self.beluga.parameters():
            param.requires_grad = False
        self.beluga.eval()
            
        expecto_data = torch.load(expecto_weights_pt)
        num_tissues = expecto_data['weight'].shape[0] 
        self.tissue_names = expecto_data['tissues']
        
        self.linear = nn.Linear(20020, num_tissues)
        self.linear.weight.data = expecto_data['weight']
        self.linear.bias.data = expecto_data['bias']
        self.linear.weight.requires_grad = False
        self.linear.bias.requires_grad = False
        
        self.shifts = [0, -200, -400, -600, -800, 200, 400, 600, 800]
        spatial_weights = self._compute_spatial_weights(dist=0, strand=strand)
        self.register_buffer('spatial_weights', spatial_weights)

    def _compute_spatial_weights(self, dist, strand):
        sign = 1 if strand == '+' else -1
        dist_signed = dist * sign
        weights_all_shifts = []
        for shift in self.shifts:
            d = (dist_signed + shift * sign) / 200.0
            w = np.zeros(10)
            if d <= 0:
                d_abs = np.floor(np.abs(d))
                w[:5] = np.exp(np.array([-0.01, -0.02, -0.05, -0.10, -0.20]) * d_abs)
            if d >= 0:
                d_abs = np.floor(np.abs(d))
                w[5:] = np.exp(np.array([-0.01, -0.02, -0.05, -0.10, -0.20]) * d_abs)
            weights_all_shifts.append(w)
        return torch.tensor(np.array(weights_all_shifts), dtype=torch.float32)

    def forward(self, x):
        batch_size = x.size(0)
        center = x.size(2) // 2 
        
        features_all_shifts = []
        for i, shift in enumerate(self.shifts):
            start = center - 1000 + shift
            end = center + 1000 + shift
            window = x[:, :, start:end]
            
            feat = self.beluga(window) 
            spatial_w = self.spatial_weights[i].view(1, 1, 10) 
            feat_expanded = feat.unsqueeze(2) * spatial_w      
            features_all_shifts.append(feat_expanded.view(batch_size, 20020))
            
        total_features = sum(features_all_shifts)
        expr_preds = self.linear(total_features)  
        return expr_preds

# ==========================================
# 3. Data Utils
# ==========================================
SEQ_LEN = 4000 

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
                
    print(f"Found {len(snp_indices_list)} SNPs in {seq_len}bp window.")
    return ref_tensor, alt_tensor, torch.tensor(snp_indices_list).long(), snp_meta_list

def get_gene_meta_by_name(gene_name, meta_csv_path):
    df = pd.read_csv(meta_csv_path)
    matched = df[df['gene_name'] == gene_name]
    if len(matched) == 0:
        raise ValueError(f"Gene '{gene_name}' not found in {meta_csv_path}")
    row = matched.iloc[0]
    return f"chr{row['chr']}", int(row['pos']), row['strand']

# ==========================================
# 4. Saliency Main Routine
# ==========================================
def train(gene_name_arg, k_arg, tissue_arg):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    BASE_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP'
    DATASET_DIR = f'{BASE_DIR}/dataset'
    EXPECTO_DIR = f'{BASE_DIR}/rebuttal/code_rebuttal/add_backbone/expecto'
    
    # 修正了路径名称
    RESULT_DIR = f'{BASE_DIR}/rebuttal/results_rebuttal/add_backbone_model/expecto/saliency_raw_results'
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    META_CSV = f'{DATASET_DIR}/gene_3000_borzoi_gencode_v41_hg38.csv'
    FASTA_PATH = f'{DATASET_DIR}/human_genome_hg38/hg38.ml.fa'
    SNP_CSV = f'{DATASET_DIR}/gene_snps_hg38/{gene_name_arg}_snps_hg38.csv'
    
    chrom, tss, strand = get_gene_meta_by_name(gene_name_arg, META_CSV)
    try:
        ref_seq, alt_seq, snp_positions, snp_meta_list = prepare_tensors(
            SNP_CSV, FASTA_PATH, chrom, center_pos=tss
        )
    except FileNotFoundError:
        print(f"⚠️ SNP file not found for {gene_name_arg}, skipping.")
        return

    if len(snp_positions) == 0:
        print(f"⚠️ No valid SNPs found in 4000bp window for {gene_name_arg}, skipping.")
        return

    ref_seq = ref_seq.to(DEVICE)
    alt_seq = alt_seq.to(DEVICE)
    snp_positions = snp_positions.to(DEVICE)
    
    print("Loading ExPecto Model & Weights...")
    beluga_pth = f'{EXPECTO_DIR}/deepsea.beluga.pth'
    weights_pt = f'{EXPECTO_DIR}/expecto_linear_weights.pt'
        
    model_expecto = ExPectoModel(beluga_pth, weights_pt, strand=strand).to(DEVICE).eval()
    
    gtex_tissue_name = TISSUE_MAP.get(tissue_arg, tissue_arg)
    target_idx = model_expecto.tissue_names.index(gtex_tissue_name)
    
    print(f"🚀 Running Saliency Baseline for {gene_name_arg} in {gtex_tissue_name}...")
    
    # === Saliency 计算开始 ===
    ref_seq.requires_grad = True
    expr = model_expecto(ref_seq)[0, target_idx]
    baseline_expr = expr.item()
    print(f"📉 Baseline Expression: {baseline_expr:.4f}")

    model_expecto.zero_grad()
    expr.backward()
    
    grads = ref_seq.grad[0] # Shape: (4, 4000)
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    
    snp_scores = []
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
            
    # 按分数降序排列并选取 Top K
    snp_scores.sort(key=lambda x: x[1], reverse=True)
    current_k = min(k_arg, len(snp_scores))
    top_k_snps = snp_scores[:current_k]
    
    # 构造突变后的序列
    mut_seq = ref_seq.clone().detach()
    mut_seq.requires_grad = False
    
    for rank, (idx, score) in enumerate(top_k_snps):
        rel_pos = snp_positions[idx].item()
        alt_tensor_slice = alt_seq[0, :, rel_pos]
        mut_seq[0, :, rel_pos] = alt_tensor_slice
        
    # 前向传播算最终得分
    with torch.no_grad():
        mut_expr = model_expecto(mut_seq)[0, target_idx].item()
        
    gain = mut_expr - baseline_expr
    print(f"🎯 Saliency Top {current_k} Final Gain: {gain:+.4f}")
    
    # === 记录结果 ===
    row_data = {
        "Gain": gain,
        "Baseline": baseline_expr,
        "Tissue": tissue_arg,
        "TrackIdx": target_idx
    }
    
    for rank, (idx, score) in enumerate(top_k_snps):
        snp_info = snp_meta_list[idx]
        row_data[f"Rank{rank+1}_Pos"] = snp_info['abs_pos']
        row_data[f"Rank{rank+1}_RefAlt"] = f"{snp_info['ref']}->{snp_info['alt']}"
        row_data[f"Rank{rank+1}_Score"] = score

    csv_filename = f"{RESULT_DIR}/{gene_name_arg}_{tissue_arg}_K{k_arg}_expecto_saliency_log.csv"
    pd.DataFrame([row_data]).to_csv(csv_filename, index=False)
    print(f"✅ Finished {gene_name_arg}. Saved to {csv_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gene', type=str, required=True)  # ELP1 
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--tissue', type=str, default='blood')
    
    args = parser.parse_args()
    train(gene_name_arg=args.gene, k_arg=args.k, tissue_arg=args.tissue)
'''
3. Saliency Map (Gradient-based): calculate gradients of the target output with respect to the input and select Top K SNP positions by gradient magnitude.
python saliency_map_gradient_based.py --index 0 --tissue brain --modality RNA
'''
'''
Multi-modal Saliency Map (Gradient-based)
Calculates gradients of the target track signal (RNA/ATAC/CAGE/DNAse/ChIP) 
with respect to the input sequence at SNP positions.
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

torch.backends.cudnn.enabled = False 

# ================= Track Configurations =================

TISSUE_RNA_MAP = {
    'blood': 7531, 'brain': 7539, 'liver': 7563, 
    'heart': 7557, 'muscle': 7569, 'pancreas': 7577,
    'lung': 7566, 
    'kidney': 7560,
}
TISSUE_ATAC_MAP = {
    'blood': 2089, 'brain': 2033, 'liver': 2035,
    'heart': 2095, 'muscle': 2093, 'pancreas': 2071,
}
TISSUE_CAGE_MAP = {
    'blood': (550, 551), 'brain': (10, 11), 'liver': (22, 23),
    'heart': (18, 19), 'muscle': (32, 33), 'pancreas': (542, 543)
}
TISSUE_DNASE_MAP = {
    'blood': 1524, 'brain': 1277, 'liver': 1303,
    'heart': 1474, 'muscle': 1320, 'pancreas': 1533,
}
TISSUE_CHIP_MAP = {
    'blood': 2186, 'brain': 2992, 'liver': 3633,
    'heart': 3394, 'muscle': 3907, 'pancreas': 4771,
}

SEQ_LEN = 524288 
BIN_SIZE = 32

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
    
    snp_meta_list = [] 
    snp_indices_list = []
    
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
            
    print(f"Found {len(snp_indices_list)} candidate SNPs.")
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

# ================= Modality-Specific Loss Logic =================

def calculate_loss_rna(output, exon_regions, target_track_idx):
    OUTPUT_LEN, CROP_OFFSET = 6144, 5120
    total_expr = 0
    for r_start, r_end in exon_regions:
        out_start = r_start - CROP_OFFSET
        out_end = r_end - CROP_OFFSET
        if out_end <= 0 or out_start >= OUTPUT_LEN: continue
        out_start, out_end = max(0, out_start), min(OUTPUT_LEN, out_end)
        if out_start < out_end:
            total_expr = total_expr + output[:, target_track_idx, out_start:out_end].sum()
    return total_expr

def calculate_loss_atac(output, strand, target_track_idx):
    # TSS -500 to +2000 bp
    n_bins = output.shape[-1]
    center_bin = n_bins // 2
    up = 500 // BIN_SIZE
    down = 2000 // BIN_SIZE
    
    if strand == '+':
        s = center_bin - up
        e = center_bin + down
    else:
        s = center_bin - down
        e = center_bin + up
    s, e = max(0, s), min(n_bins, e)
    return output[:, target_track_idx, s:e].sum()

def calculate_loss_cage(output, strand, track_pair):
    # TSS +/- 20 bins, Strand Specific
    n_bins = output.shape[-1]
    center_bin = n_bins // 2
    track_idx = track_pair[0] if strand == '+' else track_pair[1]
    s, e = max(0, center_bin - 20), min(n_bins, center_bin + 20)
    return output[:, track_idx, s:e].sum()

def calculate_loss_dnase(output, target_track_idx):
    # Center +/- 1000 bp
    n_bins = output.shape[-1]
    center_bin = n_bins // 2
    radius = 1000 // BIN_SIZE
    s, e = max(0, center_bin - radius), min(n_bins, center_bin + radius)
    return output[:, target_track_idx, s:e].sum()

def calculate_loss_chip(output, target_track_idx):
    # Center +/- 1000 bp
    n_bins = output.shape[-1]
    center_bin = n_bins // 2
    radius = 1000 // BIN_SIZE
    s, e = max(0, center_bin - radius), min(n_bins, center_bin + radius)
    return output[:, target_track_idx, s:e].sum()

# ================= Main Saliency Logic =================

def run_saliency_map(gene_index_arg, tissue_arg, modality_arg):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Determine Track and Loss Function
    track_info = None
    loss_fn = None
    
    if modality_arg == 'RNA':
        if tissue_arg not in TISSUE_RNA_MAP: raise ValueError(f"No RNA track for {tissue_arg}")
        track_info = TISSUE_RNA_MAP[tissue_arg]
        loss_fn = calculate_loss_rna
    elif modality_arg == 'ATAC':
        if tissue_arg not in TISSUE_ATAC_MAP: raise ValueError(f"No ATAC track for {tissue_arg}")
        track_info = TISSUE_ATAC_MAP[tissue_arg]
        loss_fn = calculate_loss_atac
    elif modality_arg == 'CAGE':
        if tissue_arg not in TISSUE_CAGE_MAP: raise ValueError(f"No CAGE track for {tissue_arg}")
        track_info = TISSUE_CAGE_MAP[tissue_arg]
        loss_fn = calculate_loss_cage
    elif modality_arg == 'DNAse':
        if tissue_arg not in TISSUE_DNASE_MAP: raise ValueError(f"No DNAse track for {tissue_arg}")
        track_info = TISSUE_DNASE_MAP[tissue_arg]
        loss_fn = calculate_loss_dnase
    elif modality_arg == 'ChIP':
        if tissue_arg not in TISSUE_CHIP_MAP: raise ValueError(f"No ChIP track for {tissue_arg}")
        track_info = TISSUE_CHIP_MAP[tissue_arg]
        loss_fn = calculate_loss_chip
    else:
        raise ValueError(f"Unknown modality: {modality_arg}")

    # 2. Setup Paths
    BASE_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP'
    DATASET_DIR = f'{BASE_DIR}/dataset'
    # Updated Save Path: .../Saliency_Map/raw_res/{MODALITY}/{TISSUE}
    SAVE_DIR = f'{BASE_DIR}/results/baseline_benchmark/Saliency_Map/raw_res/{modality_arg}/{tissue_arg}'
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    META_CSV = f'{DATASET_DIR}/gene_3000_borzoi_gencode_v41_hg38.csv'
    GTF_PATH = f'{DATASET_DIR}/gencode.v41.annotation.gtf.gz'
    FASTA_PATH = f'{DATASET_DIR}/human_genome_hg38/hg38.ml.fa'
    
    chrom, tss, strand, gene_id, gene_name = get_gene_meta_by_index(gene_index_arg, META_CSV)
    SNP_CSV = f'{DATASET_DIR}/gene_snps_hg38/{gene_name}_snps_hg38.csv'
    
    try:
        ref_tensor, snp_indices, seq_start_pos, snp_meta_list = prepare_tensors(
            SNP_CSV, FASTA_PATH, chrom, center_pos=tss
        )
    except FileNotFoundError: return

    # 3. Model & Gradient Setup
    ref_tensor = ref_tensor.to(DEVICE)
    ref_tensor.requires_grad = True 
    
    exon_regions = []
    if modality_arg == 'RNA':
        exon_regions = get_exons_from_gtf(gene_name, gene_id, GTF_PATH, tss, seq_start_pos)
    
    print("Loading Borzoi...")
    model_borzoi = Borzoi.from_pretrained('johahi/borzoi-replicate-0').to(DEVICE).eval()
    
    print(f"⚡ Calculating {modality_arg} Saliency Map for {gene_name}...")
    
    # 4. Forward Pass
    output = model_borzoi(ref_tensor)
    
    # 5. Compute Loss (Signal Strength)
    if modality_arg == 'RNA':
        target_val = loss_fn(output, exon_regions, track_info)
    elif modality_arg == 'ATAC' or modality_arg == 'DNAse' or modality_arg == 'ChIP':
        # These 3 use similar args: output, [strand for ATAC], track_idx
        # But wait, my loss functions have slightly different signatures above.
        # Let's standardize calling based on modality.
        if modality_arg == 'ATAC':
            target_val = loss_fn(output, strand, track_info)
        else: # DNAse, ChIP
            target_val = loss_fn(output, track_info)
    elif modality_arg == 'CAGE':
        target_val = loss_fn(output, strand, track_info)
    
    # 6. Backward Pass
    model_borzoi.zero_grad()
    target_val.backward()
    
    # 7. Extract Gradients
    input_grads = ref_tensor.grad.detach().cpu().squeeze(0) # [4, L]
    saliency_scores_seq = torch.max(torch.abs(input_grads), dim=0)[0].numpy() # [L]
    
    # 8. Extract SNP Scores
    results = []
    for i, rel_pos in enumerate(snp_indices):
        info = snp_meta_list[i]
        score = saliency_scores_seq[rel_pos]
        
        results.append({
            'Gene': gene_name,
            'Tissue': tissue_arg,
            'Modality': modality_arg,
            'Pos': info['abs_pos'],
            'Ref': info['ref'],
            'Alt': info['alt'],
            'Saliency_Score': score
        })
        
    # 9. Save
    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values(by='Saliency_Score', ascending=False)
    
    csv_path = f"{SAVE_DIR}/{gene_name}_saliency.csv"
    df_res.to_csv(csv_path, index=False)
    
    print(f"✅ Saliency Map Done. Top 1 Score: {df_res.iloc[0]['Saliency_Score']:.6f}")
    print(f"💾 Saved to: {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, required=True)
    parser.add_argument('--tissue', type=str, default='blood')
    parser.add_argument('--modality', type=str, default='RNA', 
                        choices=['RNA', 'ATAC', 'CAGE', 'DNAse', 'ChIP'])
    args = parser.parse_args()
    
    run_saliency_map(gene_index_arg=args.index, 
                     tissue_arg=args.tissue,
                     modality_arg=args.modality)

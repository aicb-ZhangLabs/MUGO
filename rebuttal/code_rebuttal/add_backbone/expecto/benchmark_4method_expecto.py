'''
ExPecto Benchmark for MUGO vs Saliency, CADD, FunSeq
Features:
1. Strict Receptive Field Filtering: ExPecto uses a tight 4000bp window.
2. Fair Comparison: Exactly Top 10 SNPs for all methods.
3. GTF-Free: ExPecto directly predicts log-fold expression, no exon mapping needed!
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import pyfaidx
import os
import argparse
from tqdm import tqdm

torch.backends.cudnn.enabled = False 

# ================= ⚙️ Configuration =================
BASE_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP'
DATASET_DIR = f'{BASE_DIR}/dataset'

# Rebuttal Specific Paths
EXPECTO_DIR = f'{BASE_DIR}/rebuttal/code_rebuttal/add_backbone/expecto'
RESULTS_ROOT = f'{BASE_DIR}/rebuttal/results_rebuttal/add_backbone_model/expecto'
MUGO_DIR = f'{RESULTS_ROOT}/MUGO_raw_results'
SALIENCY_DIR = f'{RESULTS_ROOT}/saliency_raw_results'

# Original Baselines Paths (Always use blood as they are model-agnostic)
CADD_DIR = f"{BASE_DIR}/results/baseline_benchmark/CADD/raw_res/blood"
FUNSEQ_DIR = f"{BASE_DIR}/results/baseline_benchmark/FunSeq2/raw_res/blood"

# Top 100 Genes List
TOP100_DIR = f'{BASE_DIR}/src/interpretability/newversion_table2/top100_highexp_gene'

# Track IDs Map for ExPecto
TISSUE_MAP = {
    'blood': 'Whole_Blood',   
    'brain': 'Brain_Cortex',   
}

SEQ_LEN = 4000   # ExPecto's strict receptive field
TARGET_N = 10    # Evaluate Top 10 SNPs
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ================= 🧬 ExPecto Architecture =================
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
        if 'state_dict' in state: state = state['state_dict']
        lin1_w, lin1_b = state['model.1.2.1.weight'], state['model.1.2.1.bias']
        lin2_w, lin2_b = state['model.1.4.1.weight'], state['model.1.4.1.bias']
        self.linear1 = nn.Linear(lin1_w.shape[1], lin1_w.shape[0])
        self.linear2 = nn.Linear(lin2_w.shape[1], lin2_w.shape[0])
        
        self.load_state_dict({
            'conv1.weight': state['model.0.0.weight'].squeeze(2), 'conv1.bias': state['model.0.0.bias'],
            'conv2.weight': state['model.0.2.weight'].squeeze(2), 'conv2.bias': state['model.0.2.bias'],
            'conv3.weight': state['model.0.6.weight'].squeeze(2), 'conv3.bias': state['model.0.6.bias'],
            'conv4.weight': state['model.0.8.weight'].squeeze(2), 'conv4.bias': state['model.0.8.bias'],
            'conv5.weight': state['model.0.12.weight'].squeeze(2), 'conv5.bias': state['model.0.12.bias'],
            'conv6.weight': state['model.0.14.weight'].squeeze(2), 'conv6.bias': state['model.0.14.bias'],
            'linear1.weight': lin1_w, 'linear1.bias': lin1_b,
            'linear2.weight': lin2_w, 'linear2.bias': lin2_b,
        })

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
        return torch.sigmoid(self.linear2(x))

class ExPectoModel(nn.Module):
    def __init__(self, beluga_pth, expecto_weights_pt):
        super().__init__()
        self.beluga = Beluga()
        self.beluga.load_weights(beluga_pth)
        for param in self.beluga.parameters(): param.requires_grad = False
        self.beluga.eval()
            
        expecto_data = torch.load(expecto_weights_pt)
        self.tissue_names = expecto_data['tissues']
        self.linear = nn.Linear(20020, expecto_data['weight'].shape[0])
        self.linear.weight.data = expecto_data['weight']
        self.linear.bias.data = expecto_data['bias']
        self.linear.weight.requires_grad = False
        self.linear.bias.requires_grad = False
        
        self.shifts = [0, -200, -400, -600, -800, 200, 400, 600, 800]
        # Buffer will be dynamically updated per gene
        self.register_buffer('spatial_weights', torch.zeros(9, 10))

    def set_strand(self, strand):
        """Dynamically update spatial weights depending on the gene's strand"""
        sign = 1 if strand == '+' else -1
        weights_all_shifts = []
        for shift in self.shifts:
            d = (shift * sign) / 200.0  # dist is 0 since centered on TSS
            w = np.zeros(10)
            if d <= 0:
                w[:5] = np.exp(np.array([-0.01, -0.02, -0.05, -0.10, -0.20]) * np.floor(np.abs(d)))
            if d >= 0:
                w[5:] = np.exp(np.array([-0.01, -0.02, -0.05, -0.10, -0.20]) * np.floor(np.abs(d)))
            weights_all_shifts.append(w)
        self.spatial_weights = torch.tensor(np.array(weights_all_shifts), dtype=torch.float32).to(DEVICE)

    def forward(self, x):
        batch_size = x.size(0)
        center = x.size(2) // 2 
        features_all_shifts = []
        for i, shift in enumerate(self.shifts):
            window = x[:, :, center - 1000 + shift : center + 1000 + shift]
            feat = self.beluga(window) 
            spatial_w = self.spatial_weights[i].view(1, 1, 10) 
            feat_expanded = feat.unsqueeze(2) * spatial_w      
            features_all_shifts.append(feat_expanded.view(batch_size, 20020))
        return self.linear(sum(features_all_shifts))

# ================= 🧮 Data Utils & Parsers =================
def seq_to_one_hot(seq_str):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    one_hot = np.zeros((4, len(seq_str)), dtype=np.float32)
    for i, base in enumerate(seq_str):
        if base in mapping: one_hot[mapping[base], i] = 1.0
    return torch.tensor(one_hot)

def get_gene_meta(gene_name, meta_csv_path):
    df = pd.read_csv(meta_csv_path)
    df['gene_name'] = df['gene_name'].astype(str).str.strip()
    row = df[df['gene_name'] == str(gene_name).strip()]
    if row.empty: return None, None, None
    row = row.iloc[0]
    return f"chr{row['chr']}", int(row['pos']), row['strand']

def construct_mutant_tensor(genome, chrom, start, end, snps):
    try: seq = genome[chrom][start:end].seq.upper()
    except KeyError: 
        if chrom.startswith('chr'): seq = genome[chrom[3:]][start:end].seq.upper()
        else: seq = genome[f'chr{chrom}'][start:end].seq.upper()
        
    tensor = seq_to_one_hot(seq).unsqueeze(0)
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    for snp in snps:
        rel_pos = int(snp['pos']) - start
        if 0 <= rel_pos < SEQ_LEN:
            alt = snp['alt']
            if alt in mapping:
                tensor[0, :, rel_pos] = 0 
                tensor[0, mapping[alt], rel_pos] = 1.0 
    return tensor

def get_expecto_snps(path, n):
    """Parses both MUGO and Saliency output files."""
    if not os.path.exists(path): return []
    df = pd.read_csv(path)
    
    # MUGO saves history, Saliency saves one row; use the max-gain row when available.
    if 'Gain' in df.columns and len(df) > 1:
        best_row = df.loc[df['Gain'].idxmax()]
    else:
        best_row = df.iloc[0]
        
    snps = []
    for i in range(1, n + 1):
        pos_col = f"Rank{i}_Pos"
        refalt_col = f"Rank{i}_RefAlt"
        if pos_col in best_row and pd.notna(best_row[pos_col]):
            ref_alt = best_row[refalt_col]
            alt_base = ref_alt.split('->')[1] if '->' in ref_alt else ref_alt
            snps.append({'pos': int(best_row[pos_col]), 'alt': alt_base})
    return snps

def get_filtered_baseline_snps(path, score_col, n, tss, seq_len):
    if not os.path.exists(path): return []
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    req_cols = ['Pos', 'Alt', score_col]
    if not all(c in df.columns for c in req_cols): return []
    
    # 🚨 Tight ExPecto Filtering (4000bp window)
    half_len = seq_len // 2
    df = df[(df['Pos'] >= tss - half_len) & (df['Pos'] < tss + half_len)]
    
    df = df.sort_values(by=score_col, ascending=False)
    return [{'pos': row['Pos'], 'alt': row['Alt']} for _, row in df.head(n).iterrows()]

# ================= 🚀 Main Benchmark Logic =================
def run_benchmark(args):
    tissue = args.tissue
    if tissue not in TISSUE_MAP:
        raise ValueError(f"Unknown tissue '{tissue}'")
    gtex_tissue = TISSUE_MAP[tissue]
    
    print(f"🚀 Starting ExPecto Benchmark for Tissue: {tissue} -> {gtex_tissue}")
    
    # 1. Load Top 100 Gene List
    top100_path = f"{TOP100_DIR}/top100_high_expr_cache_CAGE_{tissue}.csv"
    if not os.path.exists(top100_path):
        raise FileNotFoundError(f"Missing Top 100 list: {top100_path}")
    target_genes = set(pd.read_csv(top100_path)['Gene'].values)
    
    # 2. Find Available Intersections
    def get_genes(d, suffix):
        if not os.path.exists(d): return set()
        return {f.replace(suffix, '') for f in os.listdir(d) if f.endswith(suffix)}

    set_mugo = get_genes(MUGO_DIR, f"_{tissue}_K10_expecto_optim_log.csv")
    set_sal = get_genes(SALIENCY_DIR, f"_{tissue}_K10_expecto_saliency_log.csv")
    set_cadd = get_genes(CADD_DIR, "_cadd.csv")
    set_funseq = get_genes(FUNSEQ_DIR, "_funseq.csv")
    
    common_genes = sorted(list(target_genes & set_mugo & set_sal & set_cadd & set_funseq))
    
    print(f"📊 Intersection Stats: Top100 Target ({len(target_genes)}) | MUGO ({len(set_mugo)}) | Sal ({len(set_sal)}) -> Final Benchmarking: {len(common_genes)} genes")
    if len(common_genes) == 0:
        print("❌ No overlapping genes to process.")
        return

    # 3. Load Model and Genome
    print("Loading Genome and ExPecto Model...")
    fasta_path = f"{DATASET_DIR}/human_genome_hg38/hg38.ml.fa"
    genome = pyfaidx.Fasta(fasta_path)
    meta_csv = f"{DATASET_DIR}/gene_3000_borzoi_gencode_v41_hg38.csv"
    
    model = ExPectoModel(
        beluga_pth=f'{EXPECTO_DIR}/deepsea.beluga.pth',
        expecto_weights_pt=f'{EXPECTO_DIR}/expecto_linear_weights.pt'
    ).to(DEVICE).eval()
    
    track_idx = model.tissue_names.index(gtex_tissue)

    # 4. Benchmarking Loop
    results = []
    
    for gene in tqdm(common_genes, desc=f"Evaluating {gtex_tissue}"):
        chrom, tss, strand = get_gene_meta(gene, meta_csv)
        if not chrom: continue
        
        seq_start, seq_end = tss - SEQ_LEN // 2, tss + SEQ_LEN // 2
        model.set_strand(strand) # 🔥 Ensure spatial weights match the gene's strand
        
        # Load SNPs
        mugo_snps = get_expecto_snps(f"{MUGO_DIR}/{gene}_{tissue}_K10_expecto_optim_log.csv", TARGET_N)
        sal_snps = get_expecto_snps(f"{SALIENCY_DIR}/{gene}_{tissue}_K10_expecto_saliency_log.csv", TARGET_N)
        cadd_snps = get_filtered_baseline_snps(f"{CADD_DIR}/{gene}_cadd.csv", 'CADD_PHRED', TARGET_N, tss, SEQ_LEN)
        funseq_snps = get_filtered_baseline_snps(f"{FUNSEQ_DIR}/{gene}_funseq.csv", 'FunSeq_Score', TARGET_N, tss, SEQ_LEN)
        
        # Base WT Expression
        wt_tensor = construct_mutant_tensor(genome, chrom, seq_start, seq_end, [])
        with torch.no_grad():
            base_expr = model(wt_tensor.to(DEVICE))[0, track_idx].item()
        
        def calc_gain(snps):
            if not snps: return 0.0
            mut_tensor = construct_mutant_tensor(genome, chrom, seq_start, seq_end, snps)
            with torch.no_grad():
                mut_expr = model(mut_tensor.to(DEVICE))[0, track_idx].item()
            return mut_expr - base_expr

        results.append({
            'Gene': gene,
            'Tissue': tissue,
            'WT_Expression': base_expr,
            'MUGO_Gain': calc_gain(mugo_snps),
            'Saliency_Gain': calc_gain(sal_snps),
            'CADD_Gain': calc_gain(cadd_snps),
            'FunSeq_Gain': calc_gain(funseq_snps)
        })

    # 5. Output
    if not results: return
    df_out = pd.DataFrame(results)
    
    out_csv = f"{RESULTS_ROOT}/benchmark_CAGE_{tissue}_expecto.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"\n✅ All done! Results saved to: {out_csv}")
    
    print("\n" + "="*50)
    print(f"🏆 AVERAGE SIGNAL GAIN ({tissue.upper()} - ExPecto)")
    print("="*50)
    print(df_out[['MUGO_Gain', 'Saliency_Gain', 'CADD_Gain', 'FunSeq_Gain']].mean().round(4))
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tissue', type=str, required=True, choices=['blood', 'brain'])
    args = parser.parse_args()
    run_benchmark(args)

'''
Basenji2 Benchmark for MUGO vs Saliency, CADD, FunSeq
Features:
1. Strict Receptive Field Filtering: CADD and FunSeq SNPs are filtered strictly within Basenji2's 131kb window.
2. Fair Comparison: Exactly Top 10 SNPs for all methods.
3. Automated Intersection: Finds the overlap between Top 100 genes and all available results.
'''
import torch
import pandas as pd
import numpy as np
import pyfaidx
import os
import gzip
import argparse
from tqdm import tqdm
from basenji2_pytorch import Basenji2, basenji2_params, basenji2_weights

torch.backends.cudnn.enabled = False 

# ================= ⚙️ Configuration =================
BASE_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP'
DATASET_DIR = f'{BASE_DIR}/dataset'

# Rebuttal Specific Paths
RESULTS_ROOT = f'{BASE_DIR}/rebuttal/results_rebuttal/add_backbone_model/basenji2'
MUGO_DIR = f'{RESULTS_ROOT}/CAGE_raw_results'
SALIENCY_DIR = f'{RESULTS_ROOT}/CAGE_saliency_raw_results'

# Original Baselines Paths (Always use blood as they are model-agnostic)
CADD_DIR = f"{BASE_DIR}/results/baseline_benchmark/CADD/raw_res/blood"
FUNSEQ_DIR = f"{BASE_DIR}/results/baseline_benchmark/FunSeq2/raw_res/blood"

# Top 100 Genes List
TOP100_DIR = f'{BASE_DIR}/src/interpretability/newversion_table2/top100_highexp_gene'

# Track IDs Map for Basenji2
TISSUE_MAP = {
    'blood': 4950,   
    'brain': 4680,   
}

SEQ_LEN = 131072 # Basenji2 Receptive Field
POOL_SIZE = 128
TARGET_N = 10    # Evaluate Top 10 SNPs
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ================= 🧬 Utils =================
def seq_to_one_hot(seq_str):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    one_hot = np.zeros((4, len(seq_str)), dtype=np.float32)
    for i, base in enumerate(seq_str):
        if base in mapping: one_hot[mapping[base], i] = 1.0
    return torch.tensor(one_hot)

def get_gene_meta(gene_name, meta_csv_path):
    df = pd.read_csv(meta_csv_path)
    df['gene_name'] = df['gene_name'].astype(str).str.strip()
    gene_name = str(gene_name).strip()
    row = df[df['gene_name'] == gene_name]
    if row.empty: return None, None, None, None
    row = row.iloc[0]
    return f"chr{row['chr']}", int(row['pos']), row['strand'], row['gene_ID']

def get_exons_from_gtf(gene_id, gtf_path, tss, seq_start_pos):
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

def calculate_expression_score(model, input_seq, exon_regions, target_track_idx):
    with torch.no_grad():
        output = model(input_seq.to(DEVICE))
    if output.shape[-1] != 5313 and output.shape[1] == 5313:
        output = output.transpose(1, 2)
    OUTPUT_LEN = output.shape[1] 
    total_expr = 0
    for r_start, r_end in exon_regions:
        out_start, out_end = max(0, int(r_start)), min(OUTPUT_LEN, int(r_end))
        if out_start < out_end:
            total_expr += output[0, out_start:out_end, target_track_idx].sum().item()
    return total_expr

# ================= 🧮 Data Loaders =================

def get_mugo_snps(path, n):
    if not os.path.exists(path): return []
    df = pd.read_csv(path)
    max_idx = df['Gain'].idxmax()
    best_row = df.loc[max_idx]
    snps = []
    for i in range(1, n + 1):
        pos_col = f"Rank{i}_Pos"
        refalt_col = f"Rank{i}_RefAlt"
        if pos_col in best_row and pd.notna(best_row[pos_col]):
            ref_alt = best_row[refalt_col]
            alt_base = ref_alt.split('->')[1] if '->' in ref_alt else ref_alt
            snps.append({'pos': int(best_row[pos_col]), 'alt': alt_base})
    return snps

def get_saliency_snps(path, n):
    if not os.path.exists(path): return []
    df = pd.read_csv(path)
    # 假设你的Saliency跑完已经是按分数降序排好的
    return [{'pos': row['Pos'], 'alt': row['Alt']} for _, row in df.head(n).iterrows()]

# 🔥 核心修正: 带有感受野过滤的 CADD/FunSeq 读取器
def get_filtered_baseline_snps(path, score_col, n, tss, seq_len):
    if not os.path.exists(path): return []
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    req_cols = ['Pos', 'Alt', score_col]
    if not all(c in df.columns for c in req_cols): return []
    
    # 过滤落在 Basenji2 131kb 感受野以外的变异
    half_len = seq_len // 2
    df = df[(df['Pos'] >= tss - half_len) & (df['Pos'] < tss + half_len)]
    
    df = df.sort_values(by=score_col, ascending=False)
    return [{'pos': row['Pos'], 'alt': row['Alt']} for _, row in df.head(n).iterrows()]

# ================= 🚀 Main Logic =================

def run_benchmark(args):
    tissue = args.tissue
    if tissue not in TISSUE_MAP:
        raise ValueError(f"Unknown tissue '{tissue}'")
    track_idx = TISSUE_MAP[tissue]
    
    print(f"🚀 Starting Basenji2 Benchmark for Tissue: {tissue} (Track: {track_idx})")
    
    # 1. Load Top 100 Gene List
    top100_path = f"{TOP100_DIR}/top100_high_expr_cache_CAGE_{tissue}.csv"
    if not os.path.exists(top100_path):
        raise FileNotFoundError(f"Missing Top 100 list: {top100_path}")
    df_top100 = pd.read_csv(top100_path)
    target_genes = set(df_top100['Gene'].values)
    
    # 2. Find Files and Overlaps
    def get_genes_from_dir(d, suffix):
        if not os.path.exists(d): return set()
        return {f.replace(suffix, '') for f in os.listdir(d) if f.endswith(suffix)}

    mugo_suffix = f"_{tissue}_K10_basenji2_optim_log.csv"
    sal_suffix = "_saliency.csv"
    
    set_mugo = get_genes_from_dir(MUGO_DIR, mugo_suffix)
    set_sal = get_genes_from_dir(SALIENCY_DIR, sal_suffix)
    set_cadd = get_genes_from_dir(CADD_DIR, "_cadd.csv")
    set_funseq = get_genes_from_dir(FUNSEQ_DIR, "_funseq.csv")
    
    common_genes = sorted(list(target_genes & set_mugo & set_sal & set_cadd & set_funseq))
    
    print(f"📊 Intersection Stats:")
    print(f"  - Top 100 target: {len(target_genes)}")
    print(f"  - MUGO available: {len(set_mugo)}")
    print(f"  - Saliency available: {len(set_sal)}")
    print(f"  - Final Overlap to benchmark: {len(common_genes)} genes")
    
    if len(common_genes) == 0:
        print("❌ No overlapping genes to process. Exiting.")
        return

    # 3. Load Model and Genome
    print("Loading Genome and Basenji2 Model...")
    fasta_path = f"{DATASET_DIR}/human_genome_hg38/hg38.ml.fa"
    genome = pyfaidx.Fasta(fasta_path)
    
    meta_csv = f"{DATASET_DIR}/gene_3000_borzoi_gencode_v41_hg38.csv"
    gtf_path = f"{DATASET_DIR}/gencode.v41.annotation.gtf.gz"
    
    model = Basenji2(basenji2_params["model"]).to(DEVICE).eval()
    model.load_state_dict(torch.load(basenji2_weights()))
    for p in model.parameters(): p.requires_grad = False
    
    # 4. Benchmarking Loop
    results = []
    
    for gene in tqdm(common_genes, desc=f"Evaluating {tissue}"):
        chrom, tss, strand, gene_id = get_gene_meta(gene, meta_csv)
        if not chrom: continue
        seq_start, seq_end = tss - SEQ_LEN // 2, tss + SEQ_LEN // 2
        
        # Load SNPs
        mugo_snps = get_mugo_snps(f"{MUGO_DIR}/{gene}{mugo_suffix}", TARGET_N)
        sal_snps = get_saliency_snps(f"{SALIENCY_DIR}/{gene}_saliency.csv", TARGET_N)
        cadd_snps = get_filtered_baseline_snps(f"{CADD_DIR}/{gene}_cadd.csv", 'CADD_PHRED', TARGET_N, tss, SEQ_LEN)
        funseq_snps = get_filtered_baseline_snps(f"{FUNSEQ_DIR}/{gene}_funseq.csv", 'FunSeq_Score', TARGET_N, tss, SEQ_LEN)
        
        # Calculate Base Expression (WT)
        exons = get_exons_from_gtf(gene_id, gtf_path, tss, seq_start)
        wt_tensor = construct_mutant_tensor(genome, chrom, seq_start, seq_end, [])
        base_expr = calculate_expression_score(model, wt_tensor, exons, track_idx)
        
        def calc_gain(snps):
            if not snps: return 0.0
            mut_tensor = construct_mutant_tensor(genome, chrom, seq_start, seq_end, snps)
            mut_expr = calculate_expression_score(model, mut_tensor, exons, track_idx)
            return mut_expr - base_expr

        res_dict = {
            'Gene': gene,
            'Tissue': tissue,
            'WT_Expression': base_expr,
            'MUGO_Gain': calc_gain(mugo_snps),
            'Saliency_Gain': calc_gain(sal_snps),
            'CADD_Gain': calc_gain(cadd_snps),
            'FunSeq_Gain': calc_gain(funseq_snps)
        }
        results.append(res_dict)

    # 5. Save and Summarize
    if not results: return
    df_out = pd.DataFrame(results)
    
    out_csv = f"{RESULTS_ROOT}/benchmark_CAGE_{tissue}_basenji2.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"\n✅ All done! Results saved to: {out_csv}")
    
    print("\n" + "="*40)
    print(f"🏆 AVERAGE SIGNAL GAIN ({tissue.upper()} - Basenji2)")
    print("="*40)
    print(df_out[['MUGO_Gain', 'Saliency_Gain', 'CADD_Gain', 'FunSeq_Gain']].mean().round(2))
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tissue', type=str, required=True, choices=['blood', 'brain'])
    args = parser.parse_args()
    
    run_benchmark(args)
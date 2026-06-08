'''
Benchmark Borzoi, Saliency, CADD, and FunSeq for signal gain across 5 modalities.
Supports running on specific gene subsets (e.g., Top 100 High Expression).

Features:
1. Modality-specific Track IDs and Scoring Functions.
2. 🔥 MODIFIED: Fixed Top-10 SNPs for ALL methods (Fair Comparison).
   - Borzoi: Takes Rank 1-10 from optimization log regardless of probability.
   - Baselines: Takes Top 10 highest scored SNPs.
3. Always loads CADD and FunSeq results from 'blood' directory.
4. ✅ FIXES: Corrected filename suffixes for CAGE and ChIP.
'''
import torch
import pandas as pd
import numpy as np
import pyfaidx
import os
import gzip
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from borzoi_pytorch import Borzoi
import tarfile # 确保顶部有这个 import
import traceback

torch.backends.cudnn.enabled = False 

# ================= ⚙️ Configuration =================
BASE_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP'
DATASET_DIR = f'{BASE_DIR}/dataset'
RESULTS_ROOT = f'{BASE_DIR}/results'

# Output Root for new version tables
TABLE2_ROOT = f'{BASE_DIR}/src/interpretability/newversion_table2'

# Specific path for Top 100 High Expression Genes
TOP100_SOURCE_DIR = f'{TABLE2_ROOT}/top100_highexp_gene'

# --- Track IDs Maps ---
TISSUE_RNA_MAP = {
    'blood': 7531, 'brain': 7539, 'liver': 7563, 
    'heart': 7557, 'muscle': 7569, 'pancreas': 7577,  'kidney': 7561, 
    'lung': 7565, 
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

# --- Baseline Paths ---
CADD_BASE_DIR = f"{RESULTS_ROOT}/baseline_benchmark/CADD/raw_res"
FUNSEQ_BASE_DIR = f"{RESULTS_ROOT}/baseline_benchmark/FunSeq2/raw_res"
SALIENCY_BASE_DIR = f"{RESULTS_ROOT}/baseline_benchmark/Saliency_Map/raw_res"

SEQ_LEN = 524288
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
    if row.empty: return None
    row = row.iloc[0]
    return f"chr{row['chr']}", int(row['pos']), row['strand'], row['gene_ID']

def get_exons_from_gtf(gene_id, gtf_path, tss, seq_start_pos):
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

# ================= 🧮 Modality Scoring Functions =================

def calculate_rna_score(model, input_tensor, exon_regions, track_idx):
    with torch.no_grad():
        output = model(input_tensor.to(DEVICE))
    OUTPUT_LEN, CROP_OFFSET = 6144, 5120
    total_expr = 0.0
    for r_start, r_end in exon_regions:
        out_start = r_start - CROP_OFFSET
        out_end = r_end - CROP_OFFSET
        if out_end <= 0 or out_start >= OUTPUT_LEN: continue
        out_start, out_end = max(0, out_start), min(OUTPUT_LEN, out_end)
        if out_start < out_end:
            # RNA: [Batch, Track, Length]
            total_expr += output[0, track_idx, out_start:out_end].sum().item()
    return total_expr

def calculate_atac_score(model, input_tensor, strand, track_idx):
    with torch.no_grad():
        output = model(input_tensor.to(DEVICE)) 
    # ATAC: [Batch, Track, Length] - FIXED
    n_bins = output.shape[-1]
    center_bin = n_bins // 2
    BIN_SIZE = 32
    up = 500 // BIN_SIZE
    down = 2000 // BIN_SIZE
    if strand == '+':
        s, e = center_bin - up, center_bin + down
    else:
        s, e = center_bin - down, center_bin + up
    s, e = max(0, s), min(n_bins, e)
    return output[0, track_idx, s:e].sum().item()

def calculate_cage_score(model, input_tensor, track_pair, strand):
    with torch.no_grad():
        output = model(input_tensor.to(DEVICE))
    # CAGE: [Batch, Track, Length]
    n_bins = output.shape[-1]
    center_bin = n_bins // 2
    track_idx = track_pair[0] if strand == '+' else track_pair[1]
    s, e = max(0, center_bin - 20), min(n_bins, center_bin + 20)
    return output[0, track_idx, s:e].sum().item()

def calculate_dnase_score(model, input_tensor, track_idx):
    with torch.no_grad():
        output = model(input_tensor.to(DEVICE))
    # DNAse: [Batch, Track, Length] - FIXED
    n_bins = output.shape[-1]
    center_bin = n_bins // 2
    radius_bins = 1000 // 32
    s, e = max(0, center_bin - radius_bins), min(n_bins, center_bin + radius_bins)
    return output[0, track_idx, s:e].sum().item()

def calculate_chip_score(model, input_tensor, track_idx):
    with torch.no_grad():
        output = model(input_tensor.to(DEVICE))
    # ChIP: [Batch, Track, Length] - FIXED
    n_bins = output.shape[-1]
    center_bin = n_bins // 2
    radius_bins = 1000 // 32
    s, e = max(0, center_bin - radius_bins), min(n_bins, center_bin + radius_bins)
    return output[0, track_idx, s:e].sum().item()

# ================= 📊 Plotting =================

def plot_results(df, tissue, modality, output_dir):
    plot_df = df.melt(
        id_vars=['Gene'], 
        value_vars=['Borzoi_Gain', 'Saliency_Gain', 'CADD_Gain', 'FunSeq_Gain'],
        var_name='Method', 
        value_name='Signal Gain'
    )
    plot_df['Method'] = plot_df['Method'].str.replace('_Gain', '')
    
    plt.figure(figsize=(3, 4))
    sns.set_theme(style="ticks")
    
    palette = {
        'Borzoi': '#e74c3c', 'Saliency': '#2ecc71',
        'CADD': '#3498db', 'FunSeq': '#9b59b6'
    }
    
    ax = sns.barplot(
        data=plot_df, x='Method', y='Signal Gain', hue='Method',
        palette=palette, capsize=0.1, width=0.6, errwidth=1.5,
        errorbar=('ci', 95), legend=False
    )
    
    for p in ax.patches:
        if p.get_height() != 0:
            ax.annotate(f'{p.get_height():.1f}', 
                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha='center', va='bottom', fontsize=8)
    
    plt.title(f'{tissue.capitalize()} - {modality}', fontsize=10, fontweight='bold')
    plt.xlabel('')
    plt.ylabel('Signal Gain', fontsize=10)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    sns.despine()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/benchmark_{modality}_{tissue}.png", dpi=300)
    plt.close()

# ================= 🚀 Logic =================

def run_benchmark(args):
    global DEVICE
    tissue = args.tissue
    modality = args.modality
    mode = args.mode
    
    # 🔥 FIXED TARGET N for Fair Comparison
    TARGET_N = 10 
    
    # 1. Config Paths based on Modality
    if modality == 'RNA':
        if tissue not in TISSUE_RNA_MAP: raise ValueError(f"No RNA track for {tissue}")
        track_info = TISSUE_RNA_MAP[tissue]
        borzoi_res_folder = f"{tissue}_K10_borzoi_modeltrain_res"
        borzoi_suffix = "_optim_log.csv"
    elif modality == 'ATAC':
        if tissue not in TISSUE_ATAC_MAP: raise ValueError(f"No ATAC track for {tissue}")
        track_info = TISSUE_ATAC_MAP[tissue]
        borzoi_res_folder = f"{tissue}_K10_borzoi_ATAC_modeltrain_res"
        borzoi_suffix = "_ATAC_optim_log.csv"
    elif modality == 'CAGE':
        if tissue not in TISSUE_CAGE_MAP: raise ValueError(f"No CAGE track for {tissue}")
        track_info = TISSUE_CAGE_MAP[tissue]
        borzoi_res_folder = f"{tissue}_K10_borzoi_CAGE_modeltrain_res"
        
        # ✅ [FIX 1] Corrected CAGE suffix
        borzoi_suffix = "_borzoi_CAGE_optim_log.csv"
        
    elif modality == 'DNAse':
        if tissue not in TISSUE_DNASE_MAP: raise ValueError(f"No DNAse track for {tissue}")
        track_info = TISSUE_DNASE_MAP[tissue]
        borzoi_res_folder = f"{tissue}_K10_borzoi_DNAse_modeltrain_res" 
        borzoi_suffix = "_DNAse_optim_log.csv"
    elif modality == 'ChIP':
        if tissue not in TISSUE_CHIP_MAP: raise ValueError(f"No ChIP track for {tissue}")
        track_info = TISSUE_CHIP_MAP[tissue]
        borzoi_res_folder = f"{tissue}_K10_borzoi_CHIP_modeltrain_res" 
        
        # ✅ [FIX 2] Corrected ChIP suffix (Upper case CHIP)
        borzoi_suffix = "_CHIP_optim_log.csv"
        
    else:
        raise ValueError("Unknown modality")

    # 2. Config Output Directory based on Mode
    output_dir = f"{TABLE2_ROOT}/{mode}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Cache directory inside the mode folder
    cache_dir = f"{output_dir}/cache/{modality}/{tissue}"
    os.makedirs(cache_dir, exist_ok=True)
    
    # Source Paths
    borzoi_dir = f"{RESULTS_ROOT}/{borzoi_res_folder}"
    cadd_dir = f"{CADD_BASE_DIR}/blood"
    funseq_dir = f"{FUNSEQ_BASE_DIR}/blood"
    saliency_dir = f"{SALIENCY_BASE_DIR}/{modality}/{tissue}"
    
    meta_csv = f"{DATASET_DIR}/gene_3000_borzoi_gencode_v41_hg38.csv"
    gtf_path = f"{DATASET_DIR}/gencode.v41.annotation.gtf.gz"
    fasta_path = f"{DATASET_DIR}/human_genome_hg38/hg38.ml.fa"
    
    # 3. Gene Subset Logic
    subset_genes = None
    if mode == 'top100':
        subset_filename = f"top100_high_expr_cache_{modality}_{tissue}.csv"
        subset_file_path = f"{TOP100_SOURCE_DIR}/{subset_filename}"
        
        if os.path.exists(subset_file_path):
            print(f"📖 Loading gene subset from: {subset_file_path}")
            df_subset = pd.read_csv(subset_file_path)
            if 'Gene' in df_subset.columns:
                subset_genes = set(df_subset['Gene'].values)
            else:
                print(f"⚠️ Warning: 'Gene' column not found in subset file. Running on all genes.")
        else:
            print(f"❌ Error: Subset file not found at {subset_file_path}")
            return

    # # 4. Find Common Genes (Intersect with Folders)
    # def get_genes(d, suffix):
    #     if not os.path.exists(d): return set()
    #     return {f.replace(suffix, '') for f in os.listdir(d) if f.endswith(suffix)}
    
    # 4. Find Common Genes (Intersect with Folders)
    def get_genes(d, suffix):
        tar_path = f"{d}.tar.gz"
        if os.path.exists(tar_path):
            print(f"📦 Found Tar Archive: {tar_path}")
            with tarfile.open(tar_path, "r:gz") as tar:
                # 从 tar 内部遍历提取文件名
                return {os.path.basename(m.name).replace(suffix, '') for m in tar.getmembers() if m.name.endswith(suffix)}
        elif os.path.exists(d):
            return {f.replace(suffix, '') for f in os.listdir(d) if f.endswith(suffix)}
        return set()

    # ✅ [DEBUG] Print info to verify paths
    print(f"\n🔍 Debugging Paths for {tissue} - {modality}:")
    set_borzoi = get_genes(borzoi_dir, borzoi_suffix)
    set_saliency = get_genes(saliency_dir, "_saliency.csv")
    set_cadd = get_genes(cadd_dir, "_cadd.csv")
    set_funseq = get_genes(funseq_dir, "_funseq.csv")

    print(f"   -> Borzoi files found: {len(set_borzoi)} (Suffix: {borzoi_suffix})")
    print(f"   -> Saliency files found: {len(set_saliency)}")
    print(f"   -> CADD files found: {len(set_cadd)}")
    print(f"   -> FunSeq files found: {len(set_funseq)}")

    if subset_genes is not None:
         print(f"   -> Target Subset size: {len(subset_genes)}")

    folder_genes = set_borzoi & set_saliency & set_cadd & set_funseq

    if subset_genes is not None:
        common_genes = sorted(list(folder_genes & subset_genes))
        print(f"[{mode}] Filtered genes from subset: {len(common_genes)} (out of {len(subset_genes)} in list)")
    else:
        common_genes = sorted(list(folder_genes))
        print(f"[{mode}] Running on all available genes: {len(common_genes)}")
    
    if len(common_genes) == 0: 
        print("❌ No genes to process.")
        return

    # 5. Load Model
    uncached = [g for g in common_genes if not os.path.exists(f"{cache_dir}/{g}.csv")]
    genome, model = None, None
    
    if len(uncached) > 0:
        print(f"🚀 Computing {len(uncached)} genes...")
        genome = pyfaidx.Fasta(fasta_path)
        model = Borzoi.from_pretrained('johahi/borzoi-replicate-0').to(DEVICE).eval()
    else:
        print("✅ All cached.")

    # import tarfile # 确保引入了 tarfile
    
    # 🌟 [极其关键的性能优化]：在循环外部提前打开 tar 并建立索引字典
    borzoi_tar_path = f"{borzoi_dir}.tar.gz"
    borzoi_tar = None
    borzoi_member_dict = {}
    if os.path.exists(borzoi_tar_path):
        borzoi_tar = tarfile.open(borzoi_tar_path, "r:gz")
        for m in borzoi_tar.getmembers():
            if m.name.endswith(borzoi_suffix):
                # 建立映射: 纯文件名 -> tar包内的成员对象
                borzoi_member_dict[os.path.basename(m.name)] = m

    results = []
    
    for gene in tqdm(common_genes, desc="Benchmarking"):
        cache_file = f"{cache_dir}/{gene}.csv"
        
        if os.path.exists(cache_file):
            try:
                results.append(pd.read_csv(cache_file).iloc[0].to_dict())
                continue
            except: pass 
        
        if model is None: 
            genome = pyfaidx.Fasta(fasta_path)
            model = Borzoi.from_pretrained('johahi/borzoi-replicate-0').to(DEVICE).eval()

        try:
            # A. Get Borzoi SNPs (Fixed Top 10) - 支持从 Tar 直接读取内存流
            target_borzoi_name = f"{gene}{borzoi_suffix}"
            borzoi_snps = [] 
            
            if borzoi_tar is not None:
                if target_borzoi_name not in borzoi_member_dict: continue
                member = borzoi_member_dict[target_borzoi_name]
                f = borzoi_tar.extractfile(member)
                df_bor = pd.read_csv(f)
            else:
                borzoi_file = f"{borzoi_dir}/{target_borzoi_name}"
                if not os.path.exists(borzoi_file): continue
                df_bor = pd.read_csv(borzoi_file)
                
            max_idx = df_bor['Gain'].idxmax()
            best_row = df_bor.loc[max_idx]
            
            for i in range(1, TARGET_N + 1): # Rank 1 to 10
                score_col = f"Rank{i}_Score"
                pos_col = f"Rank{i}_Pos"
                refalt_col = f"Rank{i}_RefAlt"
                if score_col in best_row and pd.notna(best_row[score_col]):
                    ref_alt = best_row[refalt_col]
                    alt_base = ref_alt.split('->')[1] if '->' in ref_alt else ref_alt
                    borzoi_snps.append({'pos': int(best_row[pos_col]), 'alt': alt_base})
                else: break
            
            n_found = len(borzoi_snps)
            if n_found == 0: continue 

            # B. Get Baselines Top-N (Fixed 10)
            chrom, tss, strand, gene_id = get_gene_meta(gene, meta_csv)
            if not chrom or not gene_id: continue
            
            seq_start, seq_end = tss - SEQ_LEN // 2, tss + SEQ_LEN // 2
            
            def get_top_baseline(path, score_col, n):
                if not os.path.exists(path): return []
                df = pd.read_csv(path)
                df.columns = df.columns.str.strip()
                req_cols = ['Pos', 'Alt', score_col]
                if not all(c in df.columns for c in req_cols): return []
                df = df.sort_values(by=score_col, ascending=False)
                return [{'pos': row['Pos'], 'alt': row['Alt']} for _, row in df.head(n).iterrows()]

            snps_sal = get_top_baseline(f"{saliency_dir}/{gene}_saliency.csv", 'Saliency_Score', TARGET_N)
            snps_cadd = get_top_baseline(f"{cadd_dir}/{gene}_cadd.csv", 'CADD_PHRED', TARGET_N)
            snps_fun = get_top_baseline(f"{funseq_dir}/{gene}_funseq.csv", 'FunSeq_Score', TARGET_N)
            
            # C. Calculate Gain
            # (从这里往下保留你原来的代码不变)
            
            # C. Calculate Gain
            wt_tensor = construct_mutant_tensor(genome, chrom, seq_start, seq_end, [])
            
            def get_score(input_t):
                if modality == 'RNA':
                    exons = get_exons_from_gtf(gene_id, gtf_path, tss, seq_start)
                    return calculate_rna_score(model, input_t, exons, track_info)
                elif modality == 'ATAC':
                    return calculate_atac_score(model, input_t, strand, track_info)
                elif modality == 'CAGE':
                    return calculate_cage_score(model, input_t, track_info, strand)
                elif modality == 'DNAse':
                    return calculate_dnase_score(model, input_t, track_info)
                elif modality == 'ChIP':
                    return calculate_chip_score(model, input_t, track_info)
                return 0.0

            base_val = get_score(wt_tensor)
            
            def calc_gain(snps):
                if not snps: return 0.0
                mut_t = construct_mutant_tensor(genome, chrom, seq_start, seq_end, snps)
                return get_score(mut_t) - base_val

            res_dict = {
                'Gene': gene,
                'Tissue': tissue,
                'Modality': modality,
                'N_Valid_SNPs': TARGET_N, # Marking as fixed budget
                'Borzoi_Gain': calc_gain(borzoi_snps),
                'Saliency_Gain': calc_gain(snps_sal),
                'CADD_Gain': calc_gain(snps_cadd),
                'FunSeq_Gain': calc_gain(snps_fun)
            }
            results.append(res_dict)
            pd.DataFrame([res_dict]).to_csv(cache_file, index=False)
            
        except Exception as e:
            print(f"Error {gene}: {e}")
            continue

    if not results: return
    out_df = pd.DataFrame(results)
    out_path = f"{output_dir}/benchmark_{modality}_{tissue}.csv"
    out_df.to_csv(out_path, index=False)
    print(f"✅ Saved results to: {out_path}")
    
    # Plotting
    plot_results(out_df, tissue, modality, output_dir)
    print("\n--- Mean Gains ---")
    print(out_df[['Borzoi_Gain', 'Saliency_Gain', 'CADD_Gain', 'FunSeq_Gain']].mean())
    if borzoi_tar is not None:
        borzoi_tar.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tissue', type=str, required=True)
    parser.add_argument('--modality', type=str, required=True, 
                        choices=['RNA', 'ATAC', 'CAGE', 'DNAse', 'ChIP'])
    parser.add_argument('--mode', type=str, default='top100', choices=['all', 'top100'],
                        help="Select gene subset: 'all' or 'top100' (High Expression)")
    args = parser.parse_args()
    run_benchmark(args)
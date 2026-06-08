'''
Benchmark borzoi, ISM, gradient methods, and feature ablation for gene expression gain.
Feature:
1. Keeps original threshold logic (>0.9).
2. Supports resume/cache.
3. Tall and thin plot (3x4).
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
import traceback

torch.backends.cudnn.enabled = False 

# ================= ⚙️ Configuration =================
BASE_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP'
DATASET_DIR = f'{BASE_DIR}/dataset'
RESULTS_ROOT = f'{BASE_DIR}/results'
OUTPUT_DIR = f'{BASE_DIR}/results/baseline_benchmark'

TISSUE_MAP = {
    'blood': 7531, 'brain': 7539, 'liver': 7563, 
    'heart': 7557, 'muscle': 7569, 'pancreas': 7577,
}

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
    if row.empty: raise ValueError(f"Metadata not found for gene: {gene_name}")
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

def calculate_expression(model, input_tensor, exon_regions, track_idx):
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
            total_expr += output[0, track_idx, out_start:out_end].sum().item()
    return total_expr

def construct_mutant_tensor(genome, chrom, start, end, snps):
    try: seq = genome[chrom][start:end].seq.upper()
    except KeyError: 
        if chrom.startswith('chr'): seq = genome[chrom[3:]][start:end].seq.upper()
        else: seq = genome[f'chr{chrom}'][start:end].seq.upper()
    tensor = seq_to_one_hot(seq).unsqueeze(0)
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    for snp in snps:
        rel_pos = snp['pos'] - start
        if 0 <= rel_pos < SEQ_LEN:
            alt = snp['alt']
            if alt in mapping:
                tensor[0, :, rel_pos] = 0 
                tensor[0, mapping[alt], rel_pos] = 1.0 
    return tensor

def plot_results(df, tissue):
    # Data Prep
    plot_df = df.melt(
        id_vars=['Gene'], 
        value_vars=['Borzoi_Gain', 'ISM_Gain', 'Ablation_Gain', 'Saliency_Gain'],
        var_name='Method', 
        value_name='Expression Gain'
    )
    plot_df['Method'] = plot_df['Method'].str.replace('_Gain', '')
    
    # 【修改点】瘦高型图表 (3x4 inches)
    plt.figure(figsize=(3, 4))
    sns.set_theme(style="ticks")
    
    palette = {'Borzoi': '#e74c3c', 'ISM': '#3498db', 'Ablation': '#95a5a6', 'Saliency': '#2ecc71'}
    
    ax = sns.barplot(
        data=plot_df, 
        x='Method', 
        y='Expression Gain', 
        hue='Method',
        palette=palette, 
        capsize=0.1,
        width=0.6,  # 让柱子变细
        errwidth=1.5,
        errorbar=('ci', 95),
        legend=False
    )
    
    # 标注数值
    for p in ax.patches:
        if p.get_height() != 0:
            ax.annotate(f'{p.get_height():.1f}', 
                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha='center', va='bottom', fontsize=9, xytext=(0, 2), 
                       textcoords='offset points')
    
    plt.title(f'{tissue.capitalize()}', fontsize=12, fontweight='bold')
    plt.xlabel('')
    plt.ylabel('Expr. Gain' if tissue == 'blood' else '', fontsize=10)
    
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=9)
    sns.despine() # 去掉边框
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/benchmark_comparison_{tissue}_tall.png", dpi=300)
    plt.close()

# ================= 🚀 Logic =================

def run_benchmark(args):
    global DEVICE
    tissue = args.tissue
    if tissue not in TISSUE_MAP: raise ValueError(f"Unknown tissue: {tissue}")
    track_idx = TISSUE_MAP[tissue]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 【新增】设置缓存目录
    cache_dir = f"{OUTPUT_DIR}/cache/{tissue}"
    os.makedirs(cache_dir, exist_ok=True)
    
    borzoi_dir = f"{RESULTS_ROOT}/{tissue}_K10_borzoi_modeltrain_res"
    ism_dir = f"{RESULTS_ROOT}/baseline_benchmark/Greedy_ISM/raw_res/{tissue}"
    ablation_dir = f"{RESULTS_ROOT}/baseline_benchmark/Feature_Ablation/raw_res/{tissue}"
    saliency_dir = f"{RESULTS_ROOT}/baseline_benchmark/Saliency_Map/raw_res/{tissue}"
    meta_csv = f"{DATASET_DIR}/gene_3000_borzoi_gencode_v41_hg38.csv"
    gtf_path = f"{DATASET_DIR}/gencode.v41.annotation.gtf.gz"
    fasta_path = f"{DATASET_DIR}/human_genome_hg38/hg38.ml.fa"
    
    def get_genes(d, suffix):
        if not os.path.exists(d): return set()
        return {f.replace(suffix, '') for f in os.listdir(d) if f.endswith(suffix)}

    common_genes = sorted(list(
        get_genes(borzoi_dir, "_optim_log.csv") & 
        get_genes(ism_dir, "_greedy_scan.csv") & 
        get_genes(saliency_dir, "_saliency.csv") & 
        get_genes(ablation_dir, "_ablation.csv")
    ))
    print(f"🔍 Found {len(common_genes)} common genes.")
    if len(common_genes) == 0: return

    # 检查缓存，决定是否加载模型
    uncached_genes = [g for g in common_genes if not os.path.exists(f"{cache_dir}/{g}.csv")]
    
    genome = None
    model = None
    
    if len(uncached_genes) > 0:
        print(f"🚀 {len(uncached_genes)} genes need computation. Loading Genome and Model...")
        genome = pyfaidx.Fasta(fasta_path)
        model = Borzoi.from_pretrained('johahi/borzoi-replicate-0').to(DEVICE).eval()
        try: model(torch.zeros(1, 4, SEQ_LEN).to(DEVICE)); print("✅ GPU check passed.")
        except: DEVICE = torch.device('cpu'); model = model.to(DEVICE); print("⚠️ Switched to CPU.")
    else:
        print("✅ All genes cached. Skipping model load.")

    results = []
    
    for gene in tqdm(common_genes, desc="Benchmarking"):
        # 【修改点 1】检查缓存
        cache_file = f"{cache_dir}/{gene}.csv"
        if os.path.exists(cache_file):
            try:
                cached_df = pd.read_csv(cache_file)
                results.append(cached_df.iloc[0].to_dict())
                continue
            except:
                print(f"⚠️ Cache corrupted for {gene}, re-running.")
        
        if model is None: 
             print("❌ Error: Model not loaded but cache missing.")
             break

        try:
            # 1. 读取 Borzoi Log
            borzoi_file = f"{borzoi_dir}/{gene}_optim_log.csv"
            if not os.path.exists(borzoi_file): continue
            
            df_bor = pd.read_csv(borzoi_file)
            max_gain_idx = df_bor['Gain'].idxmax()
            best_row = df_bor.loc[max_gain_idx]
            
            borzoi_snps = [] 
            for i in range(1, 100):
                score_col = f"Rank{i}_Score"
                pos_col = f"Rank{i}_Pos"
                refalt_col = f"Rank{i}_RefAlt"
                
                if score_col in best_row and pd.notna(best_row[score_col]):
                    # 【重要】保留你原始逻辑中的阈值判断
                    if best_row[score_col] > 0.9: 
                        ref_alt = best_row[refalt_col]
                        if isinstance(ref_alt, str) and '->' in ref_alt:
                            alt_base = ref_alt.split('->')[1]
                            borzoi_snps.append({
                                'pos': int(best_row[pos_col]),
                                'alt': alt_base
                            })
                else:
                    break
            
            n_valid = len(borzoi_snps) 
            if n_valid == 0: continue 

            # 2. Baselines (只选 Top n_valid 个)
            chrom, tss, strand, gene_id = get_gene_meta(gene, meta_csv)
            seq_start = tss - SEQ_LEN // 2
            seq_end = tss + SEQ_LEN // 2
            
            def get_top(path, score_col, n, asc=False):
                df = pd.read_csv(path)
                df.columns = df.columns.str.strip()
                if 'Pos' not in df.columns: return []
                df = df.sort_values(by=score_col, ascending=asc)
                return [{'pos': row['Pos'], 'alt': row['Alt']} for _, row in df.head(n).iterrows()]

            snps_ism = get_top(f"{ism_dir}/{gene}_greedy_scan.csv", 'Gain', n_valid, False)
            snps_abl = get_top(f"{ablation_dir}/{gene}_ablation.csv", 'Impact_Score', n_valid, False)
            snps_sal = get_top(f"{saliency_dir}/{gene}_saliency.csv", 'Saliency_Score', n_valid, False)
            
            # 3. Inference
            wt_tensor = construct_mutant_tensor(genome, chrom, seq_start, seq_end, [])
            exon_regions = get_exons_from_gtf(gene_id, gtf_path, tss, seq_start)
            base_expr = calculate_expression(model, wt_tensor, exon_regions, track_idx)
            
            def calc_gain(snps):
                if not snps: return 0.0
                mut_tensor = construct_mutant_tensor(genome, chrom, seq_start, seq_end, snps)
                return calculate_expression(model, mut_tensor, exon_regions, track_idx) - base_expr

            res_dict = {
                'Gene': gene,
                'N_Valid_SNPs': n_valid,
                'Borzoi_Gain': calc_gain(borzoi_snps),
                'ISM_Gain': calc_gain(snps_ism),
                'Ablation_Gain': calc_gain(snps_abl),
                'Saliency_Gain': calc_gain(snps_sal)
            }
            results.append(res_dict)
            
            # 【修改点 2】写入缓存
            pd.DataFrame([res_dict]).to_csv(cache_file, index=False)
            
        except Exception as e:
            print(f"\n❌ Error {gene}: {e}")
            traceback.print_exc()
            continue

    if not results: return
    out_df = pd.DataFrame(results)
    out_path = f"{OUTPUT_DIR}/benchmark_comparison_{tissue}.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\n✅ Results saved to: {out_path}")
    
    plot_results(out_df, tissue)
    print("\n=== Summary Stats ===")
    print(out_df[['Borzoi_Gain', 'ISM_Gain', 'Ablation_Gain', 'Saliency_Gain']].mean())
    print(f"Average Valid SNPs per gene: {out_df['N_Valid_SNPs'].mean():.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tissue', type=str, default='blood')
    args = parser.parse_args()
    run_benchmark(args)
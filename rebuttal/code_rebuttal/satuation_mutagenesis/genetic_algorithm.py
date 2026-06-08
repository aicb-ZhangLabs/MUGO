'''
In Silico Combinatorial Saturation Mutagenesis for Borzoi (Genetic Algorithm Baseline)
Methodology: 
1. Initialize a population of random mutation combinations (mask of size K).
2. Evaluate fitness (gain) for each individual.
3. Select the top performing combinations to breed.
4. Apply crossover (union of parent mutations) and mutation (random swap).
5. Repeat for G generations to find the best synergistic combination.
'''
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pyfaidx
import os
import gzip
import argparse
import random
from borzoi_pytorch import Borzoi

torch.backends.cudnn.enabled = False 

# ==========================================
# 0. 核心配置 (Track ID 映射)
# ==========================================
TISSUE_MAP = {
    'blood': 7531,  
    'brain': 7539,  
    'liver': 7563,  
    'heart': 7557,  
    'muscle': 7569,  
    'Pancreas': 7577,  
    'kidney': 7560,  
    'lung': 7566,  
}

# ==========================================
# 1. Data Utils
# ==========================================
SEQ_LEN = 524288 

def seq_to_one_hot(seq_str):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    one_hot = np.zeros((4, len(seq_str)), dtype=np.float32)
    for i, base in enumerate(seq_str):
        if base in mapping:
            one_hot[mapping[base], i] = 1.0
    return torch.tensor(one_hot)

def prepare_tensors_saturation(fasta_path, chrom, center_pos, n_window, seq_len=SEQ_LEN):
    genome = pyfaidx.Fasta(fasta_path)
    start = center_pos - seq_len // 2
    end = center_pos + seq_len // 2
    ref_seq_str = genome[chrom][start:end].seq.upper()
    
    if len(ref_seq_str) != seq_len:
        raise ValueError(f"Sequence length mismatch: {len(ref_seq_str)} vs {seq_len}")
        
    ref_tensor = seq_to_one_hot(ref_seq_str).unsqueeze(0) 
    alt_tensor = ref_tensor.clone()
    
    snp_indices_list = []
    snp_meta_list = [] 
    
    comp_map = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'A'}
    base_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    
    rel_center = seq_len // 2
    win_start = rel_center - n_window // 2
    win_end = rel_center + n_window // 2
    
    for rel_pos in range(win_start, win_end):
        ref_base = ref_seq_str[rel_pos]
        alt_base = comp_map.get(ref_base, 'A')
        
        alt_tensor[0, :, rel_pos] = 0.0
        if alt_base in base_idx:
            alt_tensor[0, base_idx[alt_base], rel_pos] = 1.0
            
        snp_indices_list.append(rel_pos)
        snp_meta_list.append({
            'abs_pos': start + rel_pos, 
            'ref': ref_base, 
            'alt': alt_base
        })
        
    print(f"🔥 Saturated {len(snp_indices_list)} positions in {n_window}bp window centered at TSS.")
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

def get_gene_meta_by_name(gene_name, meta_csv_path):
    print(f"📖 Reading Metadata from: {meta_csv_path}")
    df = pd.read_csv(meta_csv_path)
    matched = df[df['gene_name'] == gene_name]
    if len(matched) == 0:
        raise ValueError(f"Gene '{gene_name}' not found in {meta_csv_path}")
    row = matched.iloc[0]
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
# 2. Main Routine (Genetic Algorithm)
# ==========================================
def train(gene_name_arg, k_arg, tissue_arg, track_idx_arg, n_window_arg, pop_size, generations):
    # 锁定随机种子
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if track_idx_arg is not None:
        target_idx = track_idx_arg
    elif tissue_arg in TISSUE_MAP:
        target_idx = TISSUE_MAP[tissue_arg]
    else:
        raise ValueError(f"Unknown tissue '{tissue_arg}'")

    BASE_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP'
    DATASET_DIR = f'{BASE_DIR}/dataset'
    
    # 修改输出路径至 genetic_algo 文件夹
    RESULT_DIR = f'{BASE_DIR}/rebuttal/results_rebuttal/satuation_mutagenesis/raw_res_genetic_algo'
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    META_CSV = f'{DATASET_DIR}/gene_3000_borzoi_gencode_v41_hg38.csv'
    GTF_PATH = f'{DATASET_DIR}/gencode.v41.annotation.gtf.gz'
    FASTA_PATH = f'{DATASET_DIR}/human_genome_hg38/hg38.ml.fa'
    
    chrom, tss, strand, gene_id, gene_name = get_gene_meta_by_name(gene_name_arg, META_CSV)
    
    ref_seq, alt_seq, snp_positions, seq_start_pos, snp_meta_list = prepare_tensors_saturation(
        FASTA_PATH, chrom, center_pos=tss, n_window=n_window_arg
    )

    ref_seq = ref_seq.to(DEVICE)
    alt_seq = alt_seq.to(DEVICE)
    snp_positions = snp_positions.to(DEVICE)
    exon_regions = get_exons_from_gtf(gene_name, gene_id, GTF_PATH, tss, seq_start_pos)
    
    print("Loading Borzoi...")
    model_borzoi = Borzoi.from_pretrained('johahi/borzoi-replicate-0').to(DEVICE).eval()
    for p in model_borzoi.parameters(): p.requires_grad = False

    with torch.no_grad():
        baseline_expr = calculate_expression_score(model_borzoi, ref_seq, exon_regions, target_idx)
    print(f"📉 Baseline Expression ({tissue_arg}): {baseline_expr.item():.4f}")
    
    print(f"🚀 Starting Genetic Algorithm for {gene_name} (N={n_window_arg}, K={k_arg}, Pop={pop_size}, Gen={generations})...")
    
    actual_k = min(k_arg, len(snp_positions))
    MUTATION_RATE = 0.2  # 20% 概率进行点突变变异
    
    # 步骤 1：初始化种群 (每个个体是一个包含 K 个下标的 Tensor)
    population = []
    for _ in range(pop_size):
        idx = torch.randperm(len(snp_positions), device=DEVICE)[:actual_k]
        population.append(idx)
        
    best_gain = -float('inf')
    best_individual = None
    
    with torch.no_grad():
        for gen in range(generations):
            fitnesses = []
            
            # 步骤 2：评估适应度
            for ind in population:
                combined_seq = ref_seq.clone()
                for pos_idx in ind:
                    pos = snp_positions[pos_idx]
                    combined_seq[0, :, pos] = alt_seq[0, :, pos]
                    
                expr = calculate_expression_score(model_borzoi, combined_seq, exon_regions, target_idx)
                gain = expr.item() - baseline_expr.item()
                fitnesses.append(gain)
                
            # 追踪当前最优解
            max_idx = np.argmax(fitnesses)
            if fitnesses[max_idx] > best_gain:
                best_gain = fitnesses[max_idx]
                best_individual = population[max_idx].clone()
                
            print(f"Generation [{gen+1}/{generations}] | Incumbent Gain: {best_gain:+.4f} | Cur Gen Max: {fitnesses[max_idx]:+.4f}")
            
            # 步骤 3：选择 (Selection) - 保留 Top 50% 最优个体
            sorted_indices = np.argsort(fitnesses)[::-1]
            survivors = [population[i] for i in sorted_indices[:pop_size // 2]]
            
            # 步骤 4：交叉与变异 (Crossover & Mutation) 生成新一代
            new_population = list(survivors)
            while len(new_population) < pop_size:
                p1, p2 = random.sample(survivors, 2)
                
                # Crossover: 提取父母的并集，然后从中随机选择 K 个
                union_idx = torch.unique(torch.cat((p1, p2)))
                if len(union_idx) >= actual_k:
                    child = union_idx[torch.randperm(len(union_idx), device=DEVICE)[:actual_k]]
                else:
                    child = p1.clone()
                    
                # Mutation: 随机丢弃 1 个突变，重新随机引入 1 个新的突变
                if random.random() < MUTATION_RATE and actual_k > 0:
                    drop_idx = random.randint(0, actual_k - 1)
                    
                    # 找出当前子代没用到的其他突变位点
                    mask = torch.ones(len(snp_positions), dtype=torch.bool, device=DEVICE)
                    mask[child] = False
                    avail = torch.nonzero(mask, as_tuple=True)[0]
                    
                    if len(avail) > 0:
                        add_idx = avail[torch.randint(0, len(avail), (1,))[0]]
                        child[drop_idx] = add_idx
                        
                new_population.append(child)
                
            population = new_population

    # 步骤 5：评估与保存最强个体
    print(f"🎯 Final Best Combined Gain (Genetic Algorithm): {best_gain:+.4f}")
    
    row_data = {
        "Method": "Genetic_Algorithm",
        "Window_N": n_window_arg,
        "Gain": best_gain,
        "Baseline": baseline_expr.item(),
        "Tissue": tissue_arg,
        "TrackIdx": target_idx
    }
    
    for i in range(actual_k):
        orig_idx = best_individual[i].item()
        snp_info = snp_meta_list[orig_idx]
        row_data[f"Rank{i+1}_Pos"] = snp_info['abs_pos']
        row_data[f"Rank{i+1}_RefAlt"] = f"{snp_info['ref']}->{snp_info['alt']}"
        # GA不具有明确的边缘分数，因此以 0.0 填充保证格式一致
        row_data[f"Rank{i+1}_Score"] = 0.0 

    csv_filename = f"{RESULT_DIR}/{gene_name}_{tissue_arg}_N{n_window_arg}_K{k_arg}_ga_optim.csv"
    pd.DataFrame([row_data]).to_csv(csv_filename, index=False)
    print(f"✅ Finished {gene_name}. Saved to {csv_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gene', type=str, required=True, help='Name of the gene (e.g., ELP1)')
    parser.add_argument('--k', type=int, default=10, help='Number of mutations to select')
    parser.add_argument('--tissue', type=str, default='blood')
    parser.add_argument('--manual_track_id', type=int, default=None)
    parser.add_argument('--N', type=int, default=100, 
                        help='Size of the saturation mutagenesis window centered at TSS')
    # 增加 GA 专属参数
    parser.add_argument('--pop_size', type=int, default=20, help='Population size for GA')
    parser.add_argument('--generations', type=int, default=20, help='Number of generations for GA')
    
    args = parser.parse_args()
    
    train(gene_name_arg=args.gene, 
          k_arg=args.k, 
          tissue_arg=args.tissue,
          track_idx_arg=args.manual_track_id,
          n_window_arg=args.N,
          pop_size=args.pop_size,
          generations=args.generations)

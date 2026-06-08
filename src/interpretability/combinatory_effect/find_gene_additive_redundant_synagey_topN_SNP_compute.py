import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pyfaidx
import os
import gzip
import glob
import argparse
import ast
from borzoi_pytorch import Borzoi
from tqdm import tqdm

# ✅ 禁用 cuDNN 以防报错，保留 GPU 加速
torch.backends.cudnn.enabled = False 

# ================= ⚙️ 配置区域 =================
BASE_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP'
DATASET_DIR = f'{BASE_DIR}/dataset'

# 对应训练脚本里的 TISSUE_MAP (Track ID)
TISSUE_TRACK_MAP = {
    'blood': 7531,  
    'brain': 7539,  
    'liver': 7563,  
    'heart': 7557,  
    'muscle': 7569,  
    'pancreas': 7577, # 注意大小写兼容
    'Pancreas': 7577
}

# 对应训练脚本里的文件夹命名
TISSUE_FOLDER_MAP = {
    'blood': 'blood_K10_borzoi_modeltrain_res',
    'brain': 'brain_K10_borzoi_modeltrain_res',
    'liver': 'liver_K10_borzoi_modeltrain_res',
    'heart': 'heart_K10_borzoi_modeltrain_res',
    'muscle': 'muscle_K10_borzoi_modeltrain_res',
    'pancreas': 'Pancreas_K10_borzoi_modeltrain_res',
    'Pancreas': 'Pancreas_K10_borzoi_modeltrain_res'
}

META_CSV = f'{DATASET_DIR}/gene_3000_borzoi_gencode_v41_hg38.csv'
GTF_PATH = f'{DATASET_DIR}/gencode.v41.annotation.gtf.gz'
FASTA_PATH = f'{DATASET_DIR}/human_genome_hg38/hg38.ml.fa'

OUTPUT_DIR = f'{BASE_DIR}/results/interaction_scan_multi'
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEQ_LEN = 524288
POOL_SIZE = 32
THRESHOLD = 0.10 

# ==========================================
# 2. 核心函数 (1:1 复刻 MVP_multi_head.py)
# ==========================================

def seq_to_one_hot(seq_str):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    one_hot = np.zeros((4, len(seq_str)), dtype=np.float32)
    for i, base in enumerate(seq_str):
        if base in mapping:
            one_hot[mapping[base], i] = 1.0
    return torch.tensor(one_hot)

# 复刻 prepare_tensors 的前半部分逻辑 (只取 Ref)
def get_ref_tensor(genome, chrom, center_pos, seq_len=SEQ_LEN):
    start = center_pos - seq_len // 2
    end = center_pos + seq_len // 2
    try:
        # pyfaidx 输出可能是大小写混合，统一 upper
        ref_seq_str = genome[chrom][start:end].seq.upper()
    except KeyError:
        return None, None
        
    if len(ref_seq_str) != seq_len:
        # 边界情况处理
        return None, None
        
    ref_tensor = seq_to_one_hot(ref_seq_str).unsqueeze(0) 
    return ref_tensor, start

# 复刻 MVP_multi_head.py 的 Exon 获取逻辑
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
        # MVP script uses +/- 5 here, previous script used +/- 2. Using MVP's logic.
        return [(center_bin - 5, center_bin + 5)]
    return exon_ranges

# 复刻 MVP_multi_head.py 的 Expression 计算逻辑 (硬编码 Offset)
def calculate_expression_score(model, input_seq, exon_regions, target_track_idx):
    output = model(input_seq)
    # MVP script hardcodes these:
    OUTPUT_LEN, CROP_OFFSET = 6144, 5120
    
    total_expr = torch.tensor(0.0, device=input_seq.device)
    for r_start, r_end in exon_regions:
        out_start = r_start - CROP_OFFSET
        out_end = r_end - CROP_OFFSET
        if out_end <= 0 or out_start >= OUTPUT_LEN: continue
        out_start, out_end = max(0, out_start), min(OUTPUT_LEN, out_end)
        if out_start < out_end:
            total_expr += output[:, target_track_idx, out_start:out_end].sum()
    return total_expr

# 辅助函数：应用突变 (逻辑保持一致：设置整列)
def apply_mutations(ref_tensor, snp_list):
    """
    snp_list: list of dict {'rel_pos': int, 'alt': str}
    """
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    mut_tensor = ref_tensor.clone()
    
    for s in snp_list:
        alt_base = s['alt']
        rel_pos = s['rel_pos']
        if alt_base in mapping:
            # MVP logic: 
            # vec = torch.zeros(4); vec[mapping[alt]] = 1.0; alt_tensor[..., rel_pos] = vec
            # 这里等价实现：
            mut_tensor[0, :, rel_pos] = 0.0 # 先清空
            mut_tensor[0, mapping[alt_base], rel_pos] = 1.0 # 再赋值
            
    return mut_tensor

# ==========================================
# 3. 主程序
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tissue', type=str, default='blood', help="Tissue name (e.g. blood)")
    parser.add_argument('--n', type=int, default=5, help="Top N SNPs to analyze")
    args = parser.parse_args()

    # 1. Check Inputs
    if args.tissue not in TISSUE_TRACK_MAP:
        raise ValueError(f"Unknown tissue: {args.tissue}")
    
    track_id = TISSUE_TRACK_MAP[args.tissue]
    folder_name = TISSUE_FOLDER_MAP[args.tissue]
    input_dir = f"{BASE_DIR}/results/{folder_name}"
    
    csv_out_path = os.path.join(OUTPUT_DIR, f"{args.tissue}_top{args.n}_interactions.csv")
    
    print(f"🚀 Starting Scan | Tissue: {args.tissue} (Track {track_id}) | Top {args.n}")
    print(f"📂 Input: {input_dir}")
    print(f"💾 Output: {csv_out_path}")

    # 2. Init CSV
    header = ["Gene", "N_SNPs", "Single_Gains_Sum", "Combo_Gain", "Ratio", "Category", "Single_Gains_Detail"]
    if not os.path.exists(csv_out_path):
        pd.DataFrame(columns=header).to_csv(csv_out_path, index=False)

    # 3. Load Resources
    print("📖 Loading Metadata...")
    meta_df = pd.read_csv(META_CSV)
    gene_map = {row['gene_name']: row for _, row in meta_df.iterrows()}
    
    print("🧬 Loading Genome...")
    genome = pyfaidx.Fasta(FASTA_PATH)
    
    print("🤖 Loading Borzoi...")
    model = Borzoi.from_pretrained('johahi/borzoi-replicate-0').to(DEVICE).eval()
    
    # 4. Processing
    log_files = glob.glob(f"{input_dir}/*_optim_log.csv")
    stats_counter = {'Synergistic': 0, 'Additive': 0, 'Redundant': 0, 'Skipped': 0}
    
    for log_path in tqdm(log_files, desc="Genes"):
        try:
            gene_name = os.path.basename(log_path).replace('_optim_log.csv', '')
            
            # --- Metadata ---
            if gene_name not in gene_map:
                stats_counter['Skipped'] += 1
                continue
            row_info = gene_map[gene_name]
            gene_id = row_info['gene_ID']
            chrom = f"chr{row_info['chr']}".replace('chrchr', 'chr')
            tss = int(row_info['pos'])
            
            # --- Parse Log ---
            df_log = pd.read_csv(log_path)
            if df_log.empty: continue
            last_row = df_log.iloc[-1]
            
            snps = []
            for r in range(1, args.n + 1):
                col_pos = f'Rank{r}_Pos'
                col_mut = f'Rank{r}_RefAlt'
                if col_pos in last_row and col_mut in last_row:
                    try:
                        pos = int(last_row[col_pos])
                        ref_alt = last_row[col_mut]
                        # Log format: "T->C"
                        alt = ref_alt.split('->')[1].strip()
                        snps.append({'pos': pos, 'alt': alt})
                    except: continue
            
            # Deduplicate by position
            unique_snps = {s['pos']: s for s in snps}.values()
            valid_snps = list(unique_snps)
            
            if len(valid_snps) < 2:
                stats_counter['Skipped'] += 1
                continue 
            
            # --- Prepare Tensors (Reuse Logic) ---
            ref_tensor, start_pos = get_ref_tensor(genome, chrom, tss)
            if ref_tensor is None: 
                # tqdm.write(f"⚠️ Ref Seq failed for {gene_name}")
                continue
            ref_tensor = ref_tensor.to(DEVICE)
            
            # Calc Relative Pos & Filter
            final_snps = []
            for s in valid_snps:
                rel = s['pos'] - start_pos
                if 0 <= rel < SEQ_LEN:
                    s['rel_pos'] = rel
                    final_snps.append(s)
            
            if len(final_snps) < 2: continue
            
            # --- Exons & WT ---
            exon_regions = get_exons_from_gtf(gene_name, gene_id, GTF_PATH, tss, start_pos)
            
            with torch.no_grad():
                wt_expr = calculate_expression_score(model, ref_tensor, exon_regions, track_id).item()
            
            # --- Single Gains ---
            single_gain_sum = 0.0
            single_gain_list = []
            
            for s in final_snps:
                mut_seq = apply_mutations(ref_tensor, [s])
                with torch.no_grad():
                    expr = calculate_expression_score(model, mut_seq, exon_regions, track_id).item()
                gain = expr - wt_expr
                single_gain_sum += gain
                single_gain_list.append(round(gain, 4))
                
            # --- Combo Gain ---
            combo_seq = apply_mutations(ref_tensor, final_snps)
            with torch.no_grad():
                combo_expr = calculate_expression_score(model, combo_seq, exon_regions, track_id).item()
            combo_gain = combo_expr - wt_expr
            
            # --- Categorize ---
            if abs(single_gain_sum) < 1e-4:
                ratio = 1.0
            else:
                ratio = combo_gain / single_gain_sum
                
            category = "Additive"
            if ratio > (1.0 + THRESHOLD): category = "Synergistic"
            elif ratio < (1.0 - THRESHOLD): category = "Redundant"
            
            stats_counter[category] += 1
            
            # --- Save ---
            row_data = {
                "Gene": gene_name,
                "N_SNPs": len(final_snps),
                "Single_Gains_Sum": round(single_gain_sum, 4),
                "Combo_Gain": round(combo_gain, 4),
                "Ratio": round(ratio, 4),
                "Category": category,
                "Single_Gains_Detail": str(single_gain_list)
            }
            pd.DataFrame([row_data]).to_csv(csv_out_path, mode='a', header=False, index=False)
            
            # Debug Print (Optional, show meaningful results)
            # if abs(single_gain_sum) > 0.1:
            #    tqdm.write(f"✅ {gene_name}: {category} (Sum={single_gain_sum:.2f}, Combo={combo_gain:.2f})")
            
        except Exception as e:
            # tqdm.write(f"❌ Error {gene_name}: {e}")
            continue

    print("\n" + "="*50)
    print(f"📊 Summary ({args.tissue}):")
    total = sum(stats_counter.values()) - stats_counter['Skipped']
    if total > 0:
        print(f"   🔥 Synergistic: {stats_counter['Synergistic']} ({stats_counter['Synergistic']/total:.1%})")
        print(f"   😐 Additive:    {stats_counter['Additive']}    ({stats_counter['Additive']/total:.1%})")
        print(f"   📉 Redundant:   {stats_counter['Redundant']}   ({stats_counter['Redundant']/total:.1%})")
    print(f"   Saved to: {csv_out_path}")
    print("="*50)

if __name__ == "__main__":
    main()
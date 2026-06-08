import pandas as pd
import numpy as np
import os
import argparse
import torch
import pyfaidx
from borzoi_pytorch import Borzoi
import json
import traceback
from tqdm import tqdm

# ==========================================
# 0. 全局配置 & 组织映射
# ==========================================
torch.backends.cudnn.enabled = False 
SEQ_LEN = 524288
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# TISSUE_MAP: (Plus_Strand_ID, Minus_Strand_ID)
# 注意: Pancreas 首字母大写，为了兼容之前代码里的 Key
TISSUE_TRACK_MAP = {
    'blood': (550, 551),
    'brain': (10, 11),
    'liver': (22, 23),
    'heart': (18, 19),
    'muscle': (32, 33),
    'Pancreas': (542, 543) 
}

# ==========================================
# 1. 工具函数
# ==========================================

def get_track_id(tissue, strand):
    # 处理大小写兼容性
    key = tissue
    if tissue == 'pancreas': key = 'Pancreas' # 强制转为 map 里的 key
    
    if key not in TISSUE_TRACK_MAP:
        raise ValueError(f"Unknown tissue: {tissue}")
        
    if str(strand).strip() == '+': return TISSUE_TRACK_MAP[key][0]
    elif str(strand).strip() == '-': return TISSUE_TRACK_MAP[key][1]
    else: return TISSUE_TRACK_MAP[key][0]

def seq_to_one_hot(seq_str):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    one_hot = np.zeros((4, len(seq_str)), dtype=np.float32)
    for i, base in enumerate(seq_str):
        if base in mapping:
            one_hot[mapping[base], i] = 1.0
    return torch.tensor(one_hot)

def calculate_expression_score_cage(model, input_seq, target_track_idx):
    with torch.no_grad():
        output = model(input_seq)
    output_len = output.shape[-1]
    center_bin = output_len // 2
    window_bins = 20 # +/- 20 bins
    start_bin = max(0, center_bin - window_bins)
    end_bin = min(output_len, center_bin + window_bins)
    # CAGE 是 unstranded 输出，Borzoi 需要选对 track
    total_expr = output[:, target_track_idx, start_bin:end_bin].sum()
    return total_expr.item()

def prepare_sequence_with_specific_snps(chrom, tss, target_snps_pos, genome, snp_df):
    """
    构造 Ref 和 Mut 序列张量
    """
    start = tss - SEQ_LEN // 2
    end = tss + SEQ_LEN // 2
    
    # 获取 Ref Seq
    try: 
        ref_seq_str = genome[f"chr{chrom}"][start:end].seq.upper()
    except KeyError: 
        # 尝试不带 chr 前缀
        try: ref_seq_str = genome[str(chrom)][start:end].seq.upper()
        except: return None, None, "ChromNotFound"
        
    if len(ref_seq_str) != SEQ_LEN: return None, None, "SeqLenMismatch"

    ref_tensor = seq_to_one_hot(ref_seq_str).unsqueeze(0)
    mut_tensor = ref_tensor.clone()

    if not target_snps_pos: 
        return ref_tensor, mut_tensor, "NoSNPs"

    # 查表获取 ALT base
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    applied_count = 0

    # 优化: 批量筛选 SNP，避免循环查询 DataFrame
    relevant_snps = snp_df[snp_df['pos'].isin(target_snps_pos)]
    
    for _, row in relevant_snps.iterrows():
        p = int(row['pos'])
        alt = row['ALT']
        rel_pos = p - start
        
        if 0 <= rel_pos < SEQ_LEN:
            if alt in mapping:
                vec = torch.zeros(4)
                vec[mapping[alt]] = 1.0
                mut_tensor[0, :, rel_pos] = vec
                applied_count += 1
    
    return ref_tensor, mut_tensor, applied_count

def extract_log_info(file_path):
    """
    从 optim_log.csv 提取 Max Gain 和 Top SNPs (Rank1-10 > 0.5)
    """
    try:
        df = pd.read_csv(file_path)
        if df.empty: return None, []
        
        # 找 Gain 最大的那一行
        best_idx = df['Gain'].idxmax()
        best_row = df.loc[best_idx]
        max_gain = best_row['Gain']
        
        hits = []
        for i in range(1, 11):
            score_col = f"Rank{i}_Score"
            pos_col = f"Rank{i}_Pos"
            
            if score_col in df.columns and pos_col in df.columns:
                score = best_row[score_col]
                if pd.notna(score) and score > 0.5:
                    hits.append(int(best_row[pos_col]))
        return max_gain, hits
    except Exception: 
        return None, []

# ==========================================
# 2. 单基因处理逻辑
# ==========================================

def process_gene(row, args, model, genome, snp_root_dir):
    gene_name = row['gene_name']
    chrom = row['chr']
    pos = int(row['pos'])
    strand = row['strand']
    
    # 1. 获取 Borzoi 自己的结果 (Self Gain)
    # 路径构建: {tissue}_K10_borzoi_CAGE_modeltrain_res
    # 文件名: {gene}_borzoi_CAGE_optim_log.csv
    bor_file = os.path.join(args.borzoi_dir, f"{gene_name}_borzoi_CAGE_optim_log.csv")
    if not os.path.exists(bor_file): return None # 没跑过 Borzoi
    
    borzoi_self_gain, _ = extract_log_info(bor_file)
    if borzoi_self_gain is None: return None

    # 2. 获取 Enformer 的结果 (提取 SNPs)
    # 路径构建: {tissue}_K10_enformer_modeltrain_CAGE_res
    # 文件名: {gene}_enformer_optim_log.csv
    enf_file = os.path.join(args.enformer_dir, f"{gene_name}_enformer_optim_log.csv")
    if not os.path.exists(enf_file): return None # 没跑过 Enformer
    
    _, enformer_hits = extract_log_info(enf_file)

    # 3. Borzoi Cross Inference
    # 用 Enformer 找到的 hits，在 Borzoi 上算 Gain
    borzoi_cross_gain = 0.0
    
    if len(enformer_hits) > 0:
        # 加载该基因的所有 SNP 列表
        snp_csv_path = os.path.join(snp_root_dir, f"{gene_name}_snps_hg38.csv")
        if not os.path.exists(snp_csv_path): return None
        
        # 读取 SNP 表 (加速: 只读一次)
        snp_df = pd.read_csv(snp_csv_path)
        if 'POS_hg38' in snp_df.columns: snp_df['pos'] = snp_df['POS_hg38'].astype(int)
        
        ref_t, mut_t, count = prepare_sequence_with_specific_snps(
            chrom, pos, enformer_hits, genome, snp_df
        )
        
        if isinstance(count, str): return None # Error status
        
        if count > 0:
            track_idx = get_track_id(args.tissue, strand)
            # Inference
            base_score = calculate_expression_score_cage(model, ref_t.to(DEVICE), track_idx)
            mut_score = calculate_expression_score_cage(model, mut_t.to(DEVICE), track_idx)
            borzoi_cross_gain = mut_score - base_score

    return {
        'Gene': gene_name,
        'Tissue': args.tissue,
        'Borzoi_Self_Gain': borzoi_self_gain,
        'Borzoi_Cross_Gain': borzoi_cross_gain, # gain using Enformer SNPs
        'Enformer_SNP_Count': len(enformer_hits)
    }

# ==========================================
# 3. 主程序
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Compute Borzoi Gain using Enformer SNPs")
    
    # 必须参数: Tissue
    parser.add_argument("--tissue", type=str, required=True, 
                        choices=['blood', 'brain', 'liver', 'heart', 'muscle', 'Pancreas', 'pancreas'],
                        help="Target tissue to process")
    
    # 路径配置 (默认值基于你的目录结构)
    BASE_DIR = "/home/dongbos/Combine_optim_Borzoi_SNP"
    parser.add_argument("--base_dir", type=str, default=BASE_DIR)
    
    args = parser.parse_args()
    
    # 统一 Tissue 名称 (处理 Pancreas 大小写)
    if args.tissue == 'pancreas': args.tissue = 'Pancreas'
    tissue_lower = args.tissue.lower() # 用于文件夹路径 (通常文件夹全是小写?)
    # 如果文件夹里 Pancreas 是大写的，请保留原样。这里假设文件夹是 {tissue}_K10...
    # 根据你之前的 log，文件夹似乎是 `blood_K10...` (小写)
    
    # 动态构建路径
    # Borzoi 结果: {tissue}_K10_borzoi_CAGE_modeltrain_res
    args.borzoi_dir = os.path.join(args.base_dir, "results", f"{args.tissue}_K10_borzoi_CAGE_modeltrain_res")
    
    # Enformer 结果: {tissue}_K10_enformer_modeltrain_CAGE_res
    args.enformer_dir = os.path.join(args.base_dir, "results", f"{args.tissue}_K10_enformer_modeltrain_CAGE_res")
    
    # 其他固定路径
    gene_list_path = os.path.join(args.base_dir, "dataset", "gene_3000_borzoi_gencode_v41_hg38.csv")
    fasta_path = os.path.join(args.base_dir, "dataset", "human_genome_hg38", "hg38.ml.fa")
    snp_dir = os.path.join(args.base_dir, "dataset", "gene_snps_hg38")
    output_root = os.path.join(args.base_dir, "results", "compare_enformer_borzoi")
    
    # Cache 目录
    cache_dir = os.path.join(output_root, f"cache_{args.tissue}")
    os.makedirs(cache_dir, exist_ok=True)
    
    print(f"🚀 Starting Compute for Tissue: {args.tissue}")
    print(f"   Borzoi Dir:   {args.borzoi_dir}")
    print(f"   Enformer Dir: {args.enformer_dir}")
    print(f"   Cache Dir:    {cache_dir}")

    # 1. 加载资源
    print("🔌 Loading Genome & Model...")
    genome = pyfaidx.Fasta(fasta_path)
    model = Borzoi.from_pretrained('johahi/borzoi-replicate-0').to(DEVICE).eval()
    
    gene_df = pd.read_csv(gene_list_path)
    
    # 2. 检查缓存，确定要跑哪些
    cached_files = set(os.listdir(cache_dir))
    todo_list = []
    
    for _, row in gene_df.iterrows():
        gname = row['gene_name']
        if f"{gname}.json" not in cached_files:
            todo_list.append(row)
            
    print(f"📊 Progress: {len(gene_df)-len(todo_list)} cached, {len(todo_list)} to compute.")
    
    # 3. 循环处理
    success_count = 0
    
    for row in tqdm(todo_list, desc=f"Computing {args.tissue}"):
        try:
            res = process_gene(row, args, model, genome, snp_dir)
            if res:
                # 写入缓存
                with open(os.path.join(cache_dir, f"{row['gene_name']}.json"), 'w') as f:
                    json.dump(res, f)
                success_count += 1
        except Exception as e:
            # print(f"Error {row['gene_name']}: {e}") # 调试时打开
            continue

    print(f"✅ Computation finished. New processed: {success_count}")

    # 4. 汇总生成 CSV (方便画图)
    print("📦 Aggregating into CSV...")
    all_data = []
    for fname in os.listdir(cache_dir):
        if fname.endswith(".json"):
            try:
                with open(os.path.join(cache_dir, fname), 'r') as f:
                    all_data.append(json.load(f))
            except: pass
            
    if all_data:
        df_out = pd.DataFrame(all_data)
        csv_path = os.path.join(output_root, f"cross_validation_data_{args.tissue}.csv")
        df_out.to_csv(csv_path, index=False)
        print(f"🎉 Summary Saved: {csv_path} (Rows: {len(df_out)})")
    else:
        print("⚠️ No data found to aggregate.")

if __name__ == "__main__":
    main()
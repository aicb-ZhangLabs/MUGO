import os
import pandas as pd
import numpy as np
import argparse
import pyBigWig
import glob
from tqdm import tqdm

# ==========================================
# 1. 配置路径与阈值
# ==========================================
BASE_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP'
DATASET_DIR = f'{BASE_DIR}/dataset'
RESULTS_DIR = f'{BASE_DIR}/results/multihead_MVP_res_K10' # 你的结果目录
OUTPUT_DIR = f'{BASE_DIR}/results/res_enrichment_conservative_borzoi'

PHYLOP_BW = f'{DATASET_DIR}/PolyP_hg38/hg38.phyloP100way.bw'
SNP_POOL_DIR = f'{DATASET_DIR}/gene_snps_hg38'

# 之前计算出的全基因组阈值
THRESHOLDS = {
    "Top10pct": 0.9820,
    "Top5pct":  1.5410,
    "Top1pct":  3.5450
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. 核心函数
# ==========================================

def get_phylop_scores_batch(chrom, positions, bw_handle):
    """批量获取 PhyloP 分数，处理异常值"""
    scores = []
    # 检查染色体名称格式 (BigWig 通常是 chr1, chr2...)
    if chrom not in bw_handle.chroms():
        # 尝试修正 (e.g., 1 -> chr1)
        alt_chrom = f"chr{chrom}"
        if alt_chrom not in bw_handle.chroms():
            return [np.nan] * len(positions)
        chrom = alt_chrom
        
    for pos in positions:
        try:
            # BigWig 是 0-based, half-open。
            # 假设你的 CSV pos 是 1-based (Gencode标准)，这里通常取 values(chrom, pos-1, pos)
            # 或者如果 Borzoi 是 0-based，取 values(chrom, pos, pos+1)
            # 这里沿用之前的逻辑，假设 pos 对齐到了 BigWig
            val = bw_handle.values(chrom, pos, pos + 1)[0]
            scores.append(val if not np.isnan(val) else np.nan)
        except:
            scores.append(np.nan)
    return np.array(scores)

def calculate_hit_rates(scores):
    """计算三个 Tier 的命中率"""
    valid_scores = scores[~np.isnan(scores)]
    n = len(valid_scores)
    
    if n == 0:
        return {k: 0.0 for k in THRESHOLDS}, 0
        
    stats = {}
    for name, thresh in THRESHOLDS.items():
        hits = np.sum(valid_scores > thresh)
        stats[name] = hits / n # Hit Rate
        stats[f"{name}_Count"] = hits
    
    return stats, n

def load_baseline_snps(gene_name):
    """读取该基因所有候选 SNP (Baseline Pool)"""
    csv_path = f"{SNP_POOL_DIR}/{gene_name}_snps_hg38.csv"
    if not os.path.exists(csv_path):
        return None, None
    
    df = pd.read_csv(csv_path)
    # 假设列名: POS_hg38 (or pos), CHROM (or chr)
    # 根据你的 prepare_tensors 函数逻辑
    if 'POS_hg38' in df.columns:
        pos_col = 'POS_hg38'
    elif 'pos' in df.columns:
        pos_col = 'pos'
    else:
        return None, None
        
    # 获取 Chrom (通常文件名或文件内有)
    # 这里我们稍后从 Metadata 或 文件内容获取
    # 简单起见，返回 Series
    return df[pos_col].values.astype(int), df

def parse_model_top_k(log_path, k=10):
    """解析优化日志，获取最后一步选出的 Top K SNP"""
    if not os.path.exists(log_path): return None
    try:
        df = pd.read_csv(log_path)
        if df.empty: return None
        last_row = df.iloc[-1]
        
        positions = []
        for i in range(1, k + 1):
            col_name = f"Rank{i}_Pos"
            if col_name in last_row:
                positions.append(int(last_row[col_name]))
        return positions
    except:
        return None

def get_chrom_from_filename_or_db(gene_name):
    # 这里为了简便，我们假设你有一个 metadata csv 可以查，
    # 或者我们直接去读 baseline csv 里的第一行 (如果里面有 chr 列)
    csv_path = f"{SNP_POOL_DIR}/{gene_name}_snps_hg38.csv"
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path, nrows=1)
            if 'chr' in df.columns: return str(df['chr'].iloc[0])
            if 'CHROM' in df.columns: return str(df['CHROM'].iloc[0])
        except:
            pass
    # Fallback: 如果之前的 meta_csv 还在，可以用那个查
    # 这里简单处理：如果没有 chrom 列，可能需要你传入 meta data
    # 暂时返回 None，在主循环处理
    return None

# ==========================================
# 3. 主逻辑
# ==========================================

def main():
    # 1. 获取所有已经跑出结果的基因
    log_files = glob.glob(f"{RESULTS_DIR}/*_optim_log.csv")
    gene_names = [os.path.basename(f).replace('_optim_log.csv', '') for f in log_files]
    
    print(f"📂 Found {len(gene_names)} genes with optimization results.")
    
    # 打开 BigWig (只打开一次以提高速度)
    try:
        bw = pyBigWig.open(PHYLOP_BW)
    except Exception as e:
        print(f"❌ Error opening BigWig: {e}")
        return

    results_data = []

    # 2. 还需要一个 Metadata Map 来确切知道 Chromosome
    # 加载你的 Gene Metadata CSV (为了准确的 chr)
    META_CSV = f'{DATASET_DIR}/gene_3000_borzoi_gencode_v41_hg38.csv'
    meta_df = pd.read_csv(META_CSV)
    # 建立 Gene -> Chrom 映射
    gene_to_chrom = dict(zip(meta_df['gene_name'], meta_df['chr']))

    for gene in tqdm(gene_names, desc="Processing Genes"):
        # A. 获取 Chrom
        if gene not in gene_to_chrom:
            continue # 如果不在 metadata 里，跳过
        
        chrom = f"chr{gene_to_chrom[gene]}"
        if 'chrchr' in chrom: chrom = chrom.replace('chrchr', 'chr')

        # B. 获取 Baseline SNPs
        base_pos, _ = load_baseline_snps(gene)
        if base_pos is None or len(base_pos) == 0:
            continue

        # C. 获取 Model Top K SNPs
        log_path = f"{RESULTS_DIR}/{gene}_optim_log.csv"
        top_pos = parse_model_top_k(log_path, k=10) # 假设 K=10
        if not top_pos:
            continue

        # D. 获取 PhyloP 分数
        base_scores = get_phylop_scores_batch(chrom, base_pos, bw)
        top_scores = get_phylop_scores_batch(chrom, top_pos, bw)

        # E. 计算 Hit Rates
        base_stats, base_n = calculate_hit_rates(base_scores)
        top_stats, top_n = calculate_hit_rates(top_scores)
        
        if base_n == 0 or top_n == 0:
            continue

        # F. 计算 Enrichment & 汇总
        row = {
            "Gene": gene,
            "Chrom": chrom,
            "Baseline_N": base_n,
            "TopK_N": top_n,
        }

        # 遍历三个 Tier (1%, 5%, 10%)
        for tier in ["Top10pct", "Top5pct", "Top1pct"]:
            b_rate = base_stats[tier]
            t_rate = top_stats[tier]
            
            # Enrichment = Model / Baseline
            # 处理除以 0 的情况
            if b_rate > 0:
                enrich = t_rate / b_rate
            else:
                enrich = 0.0 if t_rate == 0 else np.inf # 如果 Baseline 是 0 但 Model 中了，就是无穷大
            
            row[f"{tier}_Base_Rate"] = b_rate
            row[f"{tier}_Model_Rate"] = t_rate
            row[f"{tier}_Enrichment"] = enrich
            row[f"{tier}_Model_Hits"] = top_stats[f"{tier}_Count"] # 记录具体中了几个
        
        results_data.append(row)

    bw.close()

    # 3. 保存结果
    res_df = pd.DataFrame(results_data)
    
    # 格式化一下列顺序
    cols = ["Gene", "Chrom", "Baseline_N", "TopK_N", 
            "Top10pct_Model_Rate", "Top10pct_Base_Rate", "Top10pct_Enrichment",
            "Top5pct_Model_Rate", "Top5pct_Base_Rate", "Top5pct_Enrichment",
            "Top1pct_Model_Rate", "Top1pct_Base_Rate", "Top1pct_Enrichment"]
    
    # 确保只包含存在的列
    cols = [c for c in cols if c in res_df.columns]
    res_df = res_df[cols]

    save_path = f"{OUTPUT_DIR}/gene_enrichment_stats_K10.csv"
    res_df.to_csv(save_path, index=False)
    
    print("\n" + "="*60)
    print(f"✅ Analysis Complete! Saved to: {save_path}")
    print("="*60)
    
    # 4. 打印 Summaries (平均 Enrichment)
    print("📊 Global Averages (Mean Enrichment across genes):")
    for tier in ["Top10pct", "Top5pct", "Top1pct"]:
        # 排除 inf 和 nan
        valid_enrich = res_df[f"{tier}_Enrichment"].replace([np.inf, -np.inf], np.nan).dropna()
        if not valid_enrich.empty:
            print(f"   • {tier} Enrichment: {valid_enrich.mean():.2f}x")
        else:
            print(f"   • {tier} Enrichment: N/A")

if __name__ == "__main__":
    main()
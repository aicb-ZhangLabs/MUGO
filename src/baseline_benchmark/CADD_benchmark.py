'''
5. CADD Score： 现成的、基于规则或旧模型的致病性打分工具。把你 Candidate Pool 里的 2000 个 SNP 的 rsID 拿去查一下这些分数，选出topK，证明有些我找到的高分SNP被这些method漏掉了。 
python CADD_benchmark.py --index 0 --tissue brain
'''
import pandas as pd
import pysam
import os
import argparse
from tqdm import tqdm

# ================= 配置 =================
# 你刚才下载的文件路径
CADD_FILE = "/home/dongbos/Combine_optim_Borzoi_SNP/dataset/CADD_hg38/whole_genome_SNVs.tsv.gz"

# 你的项目路径
BASE_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP'
SNP_DIR = f'{BASE_DIR}/dataset/gene_snps_hg38'
OUTPUT_DIR = f'{BASE_DIR}/results/baseline_benchmark/CADD/raw_res'

# =======================================

def get_cadd_scores_batch(cadd_path, chrom, start, end, target_snps):
    """
    使用 Tabix 批量读取一个区间内的 CADD 分数，并匹配 Ref/Alt
    target_snps: dict {(pos, alt): snp_info}
    """
    results = []
    
    # 打开 CADD 文件 (Tabix 索引)
    if not os.path.exists(cadd_path):
        raise FileNotFoundError(f"❌ CADD file not found at: {cadd_path}\nPlease download it first.")
        
    cadd = pysam.TabixFile(cadd_path)
    
    # CADD 里的染色体通常没有 'chr' 前缀 (1, 2, ... X, Y)
    # 但你的 Metadata 可能有，需要处理一下
    query_chrom = chrom.replace("chr", "")
    
    try:
        # fetch(chrom, start, end) 获取该区间所有可能的突变记录
        # 这是一个生成器，非常省内存
        records = cadd.fetch(query_chrom, start, end)
    except ValueError:
        print(f"⚠️ Region {chrom}:{start}-{end} not found in CADD file.")
        return []

    # 遍历区间内的所有记录，看是否命中我们的 SNP
    for line in records:
        # CADD 格式: Chrom Pos Ref Alt RawScore PHRED
        parts = line.split('\t')
        pos = int(parts[1])
        ref = parts[2]
        alt = parts[3]
        
        # 检查是否是我们关注的 SNP
        # target_snps 的 key 是 (pos, alt)
        if (pos, alt) in target_snps:
            phred = float(parts[5]) # 第 6 列是 PHRED Score
            
            # 找到匹配！
            snp_data = target_snps[(pos, alt)]
            results.append({
                'Gene': snp_data['gene'],
                'Pos': pos,
                'Ref': ref,
                'Alt': alt,
                'CADD_PHRED': phred
            })
            
    return results

def run_cadd_benchmark(gene_index_arg, tissue_arg):
    # 1. 读取 Metadata (为了知道 Gene 的位置)
    META_CSV = f'{BASE_DIR}/dataset/gene_3000_borzoi_gencode_v41_hg38.csv'
    meta_df = pd.read_csv(META_CSV)
    row = meta_df.iloc[gene_index_arg]
    gene_name = row['gene_name']
    chrom = f"chr{row['chr']}"
    tss = int(row['pos'])
    
    # 2. 读取该基因的 Candidate SNP CSV
    snp_csv_path = f"{SNP_DIR}/{gene_name}_snps_hg38.csv"
    if not os.path.exists(snp_csv_path):
        print(f"Skipping {gene_name}, SNP file not found.")
        return

    snp_df = pd.read_csv(snp_csv_path)
    # 统一列名
    if 'POS_hg38' in snp_df.columns: snp_df.rename(columns={'POS_hg38': 'pos'}, inplace=True)
    if 'ALT' in snp_df.columns: snp_df.rename(columns={'ALT': 'alt'}, inplace=True)
    
    # 3. 构建查询字典 (加速匹配)
    target_snps = {}
    min_pos = snp_df['pos'].min()
    max_pos = snp_df['pos'].max()
    
    for _, s_row in snp_df.iterrows():
        target_snps[(int(s_row['pos']), s_row['alt'])] = {'gene': gene_name}

    print(f"🚀 Querying CADD for {gene_name} ({len(target_snps)} SNPs)...")
    print(f"   Region: {chrom}:{min_pos}-{max_pos}")

    # 4. 执行查询
    scores = get_cadd_scores_batch(CADD_FILE, chrom, min_pos, max_pos, target_snps)
    
    # 5. 保存结果
    save_dir = f"{OUTPUT_DIR}/{tissue_arg}"
    os.makedirs(save_dir, exist_ok=True)
    
    res_df = pd.DataFrame(scores)
    # 排序：PHRED 分数越高越有害
    res_df = res_df.sort_values(by='CADD_PHRED', ascending=False)
    
    out_path = f"{save_dir}/{gene_name}_cadd.csv"
    res_df.to_csv(out_path, index=False)
    
    print(f"✅ Found {len(res_df)}/{len(target_snps)} scores. Saved to: {out_path}")
    if not res_df.empty:
        print(f"   Top 1 CADD Score: {res_df.iloc[0]['CADD_PHRED']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, required=True)
    parser.add_argument('--tissue', type=str, default='blood') # CADD 其实跟 tissue 无关，但为了保持目录结构一致
    args = parser.parse_args()
    
    run_cadd_benchmark(args.index, args.tissue)
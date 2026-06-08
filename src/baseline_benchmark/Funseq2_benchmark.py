import pysam
import pandas as pd
import os
import argparse
from tqdm import tqdm

# ================= ⚙️ 核心配置 =================
# 请确保路径正确
FUNSEQ_FILE = "/home/dongbos/Combine_optim_Borzoi_SNP/dataset/Funseq2_data/hg38.funseq2.1.6.liftover.bed.bgz"

BASE_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP'
SNP_DIR = f'{BASE_DIR}/dataset/gene_snps_hg38'
OUTPUT_DIR = f'{BASE_DIR}/results/baseline_benchmark/FunSeq2/raw_res'
META_CSV = f'{BASE_DIR}/dataset/gene_3000_borzoi_gencode_v41_hg38.csv'
# ===============================================

def run_funseq_query(gene_index, tissue):
    # --- 1. 读取 Metadata ---
    print(f"📖 Loading metadata from index {gene_index}...")
    if not os.path.exists(META_CSV):
        raise FileNotFoundError(f"Metadata file not found: {META_CSV}")
        
    meta_df = pd.read_csv(META_CSV)
    if gene_index >= len(meta_df):
        raise IndexError(f"Index {gene_index} out of bounds for metadata (size {len(meta_df)})")
        
    row = meta_df.iloc[gene_index]
    gene_name = row['gene_name']
    # 你的 debug 脚本显示索引里是 'chr1' 格式，所以这里加 chr 前缀是对的
    chrom = f"chr{row['chr']}" 
    
    # --- 2. 读取 SNP 文件 ---
    snp_csv_path = f"{SNP_DIR}/{gene_name}_snps_hg38.csv"
    if not os.path.exists(snp_csv_path):
        print(f"⚠️ Skipping {gene_name}, SNP file not found at {snp_csv_path}")
        return

    snp_df = pd.read_csv(snp_csv_path)
    # 兼容列名
    if 'POS_hg38' in snp_df.columns: snp_df.rename(columns={'POS_hg38': 'pos'}, inplace=True)
    if 'ALT' in snp_df.columns: snp_df.rename(columns={'ALT': 'alt'}, inplace=True)

    print(f"🚀 Querying FunSeq2 (hg38) for {gene_name} ({len(snp_df)} SNPs)...")

    # --- 3. 打开 FunSeq 文件 (Tabix) ---
    if not os.path.exists(FUNSEQ_FILE):
        raise FileNotFoundError(f"FunSeq file not found: {FUNSEQ_FILE}")
        
    fs = pysam.TabixFile(FUNSEQ_FILE)
    
    results = []
    non_zero_count = 0
    
    # --- 4. 遍历查询 ---
    for idx, r_snp in tqdm(snp_df.iterrows(), total=len(snp_df)):
        pos = int(r_snp['pos'])
        # FunSeq Bed 是 0-based start, 1-based end
        start, end = pos - 1, pos
        
        score = 0.0
        
        try:
            # 直接使用带 chr 的染色体名
            records = fs.fetch(chrom, start, end)
            
            for line in records:
                parts = line.split('\t')
                
                # 🛑 关键修复：你的 debug 结果显示分数在第 6 列 (index 6)
                # 格式如: ".;No;..." 或 "0.155;No;..."
                if len(parts) <= 6: 
                    continue 
                
                raw_score_str = parts[6] 
                
                # 提取分号前的部分
                score_part = raw_score_str.split(';')[0]
                
                # 处理 "." (无分/0分)
                if score_part == "." or score_part == "":
                    score = 0.0
                else:
                    try:
                        score = float(score_part)
                    except ValueError:
                        score = 0.0
                
                if score != 0:
                    non_zero_count += 1
                break # 找到即止
                
        except ValueError:
            pass 
        except KeyError:
            # 如果染色体名不对 (比如 chrM vs M)，可以尝试去掉 chr
            try:
                records = fs.fetch(chrom.replace("chr", ""), start, end)
                # ... (重复上面的解析逻辑，略)
            except:
                pass
        except Exception:
            pass 

        results.append({
            'Gene': gene_name,
            'Pos': pos,
            'Ref': r_snp['REF'] if 'REF' in r_snp else 'N',
            'Alt': r_snp['alt'],
            'FunSeq_Score': score
        })
        
    # --- 5. 保存结果 ---
    save_dir = f"{OUTPUT_DIR}/{tissue}"
    os.makedirs(save_dir, exist_ok=True)
    
    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values(by='FunSeq_Score', ascending=False)
    
    out_path = f"{save_dir}/{gene_name}_funseq.csv"
    res_df.to_csv(out_path, index=False)
    
    print(f"✅ Finished. Found {non_zero_count} variants with non-zero scores.")
    print(f"💾 Saved to: {out_path}")
    if not res_df.empty:
        print(f"   Top Score: {res_df.iloc[0]['FunSeq_Score']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, required=True, help="Gene index in metadata csv")
    parser.add_argument('--tissue', type=str, default='brain', help="Tissue name for output folder")
    args = parser.parse_args()
    
    run_funseq_query(args.index, args.tissue)
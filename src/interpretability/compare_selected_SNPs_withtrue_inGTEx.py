import pandas as pd
import numpy as np
import os
import argparse
import sys

# ================= 配置路径 (请根据实际情况修改) =================

# 1. 基因 Meta 文件路径
META_CSV_PATH = '/home/dongbos/Combine_optim_Borzoi_SNP/dataset/gene_3000_borzoi_gencode_v41_hg38.csv'

# 2. 模型输出 Log 的文件夹
MODEL_RES_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP/results/multihead_MVP_res'

# 3. 结果保存路径
OUTPUT_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP/results/GTEx_Comparison'

# 4. [关键] GTEx 数据源路径 (指向你刚解压的 Whole Blood 显著文件)
GTEX_DATA_PATH = '/home/dongbos/Combine_optim_Borzoi_SNP/dataset/GTEx_Analysis_v8_eQTL/Whole_Blood.v8.signif_variant_gene_pairs.txt.gz'

# ===============================================================

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_gene_info(index, meta_path):
    """读取 Meta 表，获取特定 Index 的基因信息"""
    # print(f"📖 Reading Meta: {meta_path}")
    df = pd.read_csv(meta_path)
    
    if index < 0 or index >= len(df):
        raise ValueError(f"Index {index} out of bounds (0-{len(df)-1})")
        
    row = df.iloc[index]
    # 清洗 gene_id (去除版本号，例如 ENSG000001.4 -> ENSG000001)
    gene_id_clean = row['gene_ID'].split('.')[0]
    
    return {
        'gene_name': row['gene_name'],
        'gene_id_full': row['gene_ID'],
        'gene_id_clean': gene_id_clean,
        'chr': str(row['chr'])
    }

def parse_model_snps(log_path, score_threshold=0.5):
    """解析优化日志的最后一行，提取 High Score SNPs"""
    if not os.path.exists(log_path):
        print(f"❌ Log file not found: {log_path}")
        return []

    try:
        df = pd.read_csv(log_path)
        if df.empty:
            return []
            
        last_row = df.iloc[-1]
        extracted_snps = []
        
        for i in range(1, 11):
            col_pos = f'Rank{i}_Pos'
            col_score = f'Rank{i}_Score'
            col_mut = f'Rank{i}_RefAlt'
            
            if col_pos in last_row and col_score in last_row:
                score = float(last_row[col_score])
                pos = int(last_row[col_pos])
                mut = str(last_row[col_mut])
                
                if score > score_threshold:
                    extracted_snps.append({
                        'pos': pos,
                        'mutation': mut,
                        'model_score': score,
                        'rank': i
                    })
    except Exception as e:
        print(f"⚠️ Error parsing log: {e}")
        return []

    return extracted_snps

def load_gtex_signif_data(gene_id_clean, gtex_path):
    """
    专门针对 GTEx v8 *.signif_variant_gene_pairs.txt.gz 进行解析
    格式: variant_id (chr_pos_ref_alt_b38), gene_id, pval_nominal, ...
    """
    print(f"🔍 Searching GTEx Significant Pairs for {gene_id_clean}...")
    
    try:
        chunks = []
        chunk_size = 200000 # 适当的块大小
        
        # 定义需要读取的列，节省内存
        # variant_id 包含位置信息, gene_id 用于匹配, pval_nominal 是P值
        use_cols = ['variant_id', 'gene_id', 'pval_nominal']
        
        # 迭代读取压缩文件
        reader = pd.read_csv(
            gtex_path, 
            sep='\t', 
            usecols=use_cols, 
            chunksize=chunk_size, 
            compression='gzip',
            dtype={'pval_nominal': 'float32', 'variant_id': 'str', 'gene_id': 'str'}
        )
        
        found_any = False
        
        for chunk in reader:
            # 快速筛选：检查 gene_id 列是否包含我们的 gene_id_clean
            # GTEx gene_id 格式通常是 ENSGxxxxx.x
            # 使用 startswith 匹配（比 apply split 快）
            mask = chunk['gene_id'].str.startswith(gene_id_clean)
            filtered = chunk[mask].copy()
            
            if not filtered.empty:
                found_any = True
                
                # 解析 variant_id 提取 Position
                # 格式: chr1_14677_G_A_b38 -> 提取 14677
                # 逻辑: split('_')[1]
                filtered['variant_pos'] = filtered['variant_id'].map(lambda x: int(x.split('_')[1]))
                
                chunks.append(filtered)
        
        if chunks:
            gene_gtex = pd.concat(chunks)
            print(f"   -> Found {len(gene_gtex)} significant eQTLs in GTEx.")
            return gene_gtex
        else:
            print("   -> No significant eQTLs found in GTEx for this gene.")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"⚠️ Error loading GTEx data: {e}")
        return pd.DataFrame()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, required=True, help='Row index in the meta CSV')
    args = parser.parse_args()

    ensure_dir(OUTPUT_DIR)

    # 1. 获取基因信息
    gene_info = get_gene_info(args.index, META_CSV_PATH)
    gene_name = gene_info['gene_name']
    gene_id_clean = gene_info['gene_id_clean']
    
    save_path = os.path.join(OUTPUT_DIR, f"{gene_name}_gtex_comparison.csv")
    
    # (可选) 如果已经跑过，可以跳过
    # if os.path.exists(save_path):
    #     print(f"Skipping {gene_name}, already exists.")
    #     return

    print(f"🚀 Processing: {gene_name} (ID: {gene_id_clean})")

    # 2. 解析模型预测的 SNPs
    log_file = os.path.join(MODEL_RES_DIR, f"{gene_name}_optim_log.csv")
    model_snps = parse_model_snps(log_file, score_threshold=0.5)

    if not model_snps:
        print(f"💤 No SNPs with score > 0.5 found for {gene_name}.")
        return

    print(f"✅ Found {len(model_snps)} high-confidence model SNPs.")

    # 3. 加载 GTEx 验证数据
    if os.path.exists(GTEX_DATA_PATH):
        gtex_df = load_gtex_signif_data(gene_id_clean, GTEX_DATA_PATH)
    else:
        print(f"❌ GTEx file not found at: {GTEX_DATA_PATH}")
        sys.exit(1)

    # 4. 建立 GTEx 查找表
    # Key: Position (int), Value: P-value
    gtex_map = {}
    if not gtex_df.empty:
        # 如果同一个位置有多个变异(不同等位基因)，取 P 值最小的
        gtex_df = gtex_df.sort_values('pval_nominal')
        gtex_map = dict(zip(gtex_df['variant_pos'], gtex_df['pval_nominal']))

    # 5. 对比分析
    results = []
    
    for snp in model_snps:
        pos = snp['pos']
        score = snp['model_score']
        
        is_hit = False
        gtex_p = np.nan
        note = "Model_Only" # 默认：仅模型预测，GTEx中未达显著

        if pos in gtex_map:
            is_hit = True
            gtex_p = gtex_map[pos]
            # 因为我们读取的是 significant_pairs 文件，只要存在就是显著的
            note = "Significant_Overlap (GTEx Hit)"
        
        results.append({
            'Gene': gene_name,
            'Gene_ID': gene_id_clean,
            'SNP_Pos': pos,
            'Mutation': snp['mutation'],
            'Model_Rank': snp['rank'],
            'Model_Score': score,
            'GTEx_Hit': is_hit,      # True/False
            'GTEx_Pval': gtex_p,     # 具体 P 值
            'Note': note
        })

    # 6. 保存结果
    res_df = pd.DataFrame(results)
    res_df.to_csv(save_path, index=False)
    
    print("-" * 50)
    # 打印简报，只展示 Hit 的或者前几个
    print(res_df[['SNP_Pos', 'Model_Score', 'GTEx_Hit', 'GTEx_Pval']].head(10))
    print(f"💾 Saved comparison to: {save_path}")

if __name__ == "__main__":
    main()
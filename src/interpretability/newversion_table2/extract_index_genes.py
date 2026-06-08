import pandas as pd
import os

# ================= ⚙️ 配置区域 =================

# 1. Master Metadata (必须是训练脚本读取的同一个文件)
# 这里包含所有 3000 个基因的信息，Index 就是它的行号
MASTER_META_PATH = '/home/dongbos/Combine_optim_Borzoi_SNP/dataset/gene_3000_borzoi_gencode_v41_hg38.csv'

# 2. 你的 Top 100 Gene List 所在的目录
# (我们用 RNA 的列表作为锚点，因为它肯定存在且是最新的)
TARGET_LIST_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP/src/interpretability/newversion_table2/top100_highexp_gene'

# 3. 输出目录 (生成的 Index 列表存这里)
OUTPUT_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP/src/run_scripts/job_indices'

# 4. 指定要处理的 Tissue
TISSUES = ['blood', 'brain']

# ================= 🚀 执行逻辑 =================

def generate_indices():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # 1. 加载 Master Metadata
    print(f"📖 Loading Master Metadata from: {MASTER_META_PATH}")
    df_master = pd.read_csv(MASTER_META_PATH)
    
    # 创建映射字典: Gene Name -> Index (Row Number)
    # 注意: strip() 去除可能存在的空格
    df_master['gene_name'] = df_master['gene_name'].astype(str).str.strip()
    gene_to_index = {name: idx for idx, name in enumerate(df_master['gene_name'])}
    
    print(f"✅ Master Metadata loaded. Total genes: {len(gene_to_index)}")

    # 2. 遍历 Tissue 生成 Index List
    for tissue in TISSUES:
        # 读取该 Tissue 的 Top 100 基因列表
        # 使用 RNA 文件作为 Source (因为你之前已经把所有模态对齐到 RNA 了)
        target_file = os.path.join(TARGET_LIST_DIR, f"top100_high_expr_cache_RNA_{tissue}.csv")
        
        if not os.path.exists(target_file):
            print(f"⚠️ Warning: File not found for {tissue}: {target_file}")
            continue
            
        print(f"\nProcessing {tissue}...")
        df_target = pd.read_csv(target_file)
        
        job_list = []
        found_count = 0
        
        for gene in df_target['Gene']:
            gene = str(gene).strip()
            if gene in gene_to_index:
                # 找到 Index 了！
                idx = gene_to_index[gene]
                job_list.append({'Gene': gene, 'Index': idx})
                found_count += 1
            else:
                print(f"  ❌ Gene not found in Master Metadata: {gene}")
        
        # 保存结果
        if job_list:
            out_df = pd.DataFrame(job_list)
            out_filename = os.path.join(OUTPUT_DIR, f"indices_{tissue}.csv")
            out_df.to_csv(out_filename, index=False)
            print(f"  🎉 Saved {len(out_df)} indices to: {out_filename}")
        else:
            print(f"  ⚠️ No valid genes found for {tissue}!")

if __name__ == "__main__":
    generate_indices()
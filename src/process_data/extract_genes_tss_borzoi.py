'''
previous 3000 genes are get from huang ng 23, which are from hg19 and enformer coordinate. 
we use borzoi annotation file gencode v41 to get TSS postion of those 3000 genes which is based on hg38, and save in similar format as previous in csv file. 
'''


import pandas as pd
import gzip
import os

# ================= 配置路径 =================
GTF_FILE = '/home/dongbos/Combine_optim_Borzoi_SNP/dataset/gencode.v41.annotation.gtf.gz'
INPUT_CSV = '/home/dongbos/Combine_optim_Borzoi_SNP/dataset/gene_3000_hg19_enformer.csv'

# 输出文件路径 (保存在同一目录)
OUTPUT_DIR = os.path.dirname(INPUT_CSV)
OUTPUT_CSV = os.path.join(OUTPUT_DIR, 'gene_3000_borzoi_gencode_v41.csv')

def get_gene_id_base(gene_id_str):
    """去除版本号，例如 ENSG00000000457.14 -> ENSG00000000457"""
    return gene_id_str.split('.')[0]

def parse_gtf_and_get_tss(gtf_path, target_gene_ids):
    """
    读取 GTF 文件，提取目标基因的 hg38 TSS 信息。
    返回一个字典: {gene_id_base: {'chr': x, 'pos': y, 'strand': z, 'gene_name': name}}
    """
    print(f"正在读取 GTF 文件: {gtf_path} ...")
    gene_info_map = {}
    
    # 转换为集合以加快查找速度
    target_ids_set = set(target_gene_ids)
    
    with gzip.open(gtf_path, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            
            parts = line.strip().split('\t')
            
            # 我们只关心 gene 类型的行 (通常定义了整个基因的范围)
            if parts[2] != 'gene':
                continue
            
            attributes = parts[8]
            
            # 解析 gene_id
            # 属性格式通常是: gene_id "ENSG00000223972.5"; gene_type "transcribed_unprocessed_pseudogene"; ...
            # 简单的字符串查找提取 gene_id
            try:
                # 提取 gene_id "..."
                start_idx = attributes.find('gene_id "') + 9
                end_idx = attributes.find('"', start_idx)
                full_gene_id = attributes[start_idx:end_idx]
                gene_id_base = get_gene_id_base(full_gene_id)
                
                # 如果这个基因不在我们的 3000 个目标里，跳过
                if gene_id_base not in target_ids_set:
                    continue

                # 提取 gene_name (可选，用于验证)
                name_start = attributes.find('gene_name "') + 11
                name_end = attributes.find('"', name_start)
                gene_name = attributes[name_start:name_end]

                chrom = parts[0]
                # 去除 'chr' 前缀以匹配你的示例格式 (chr1 -> 1)
                if chrom.startswith('chr'):
                    chrom = chrom.replace('chr', '')
                
                strand = parts[6]
                start_pos = int(parts[3])
                end_pos = int(parts[4])
                
                # === 计算 TSS (Borzoi/Enformer 逻辑) ===
                # 正链: TSS 是起始位置
                # 负链: TSS 是终止位置 (因为是 5' 端)
                if strand == '+':
                    tss = start_pos
                else:
                    tss = end_pos
                
                gene_info_map[gene_id_base] = {
                    'chr': chrom,
                    'pos': tss,
                    'strand': strand,
                    'gene_name': gene_name # GTF里的新名字
                }
                
            except ValueError:
                continue

    return gene_info_map

def main():
    # 1. 读取原始 CSV
    print(f"正在读取原始 CSV: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    
    # 确保 gene_ID 列存在，并且去除版本号用于匹配
    # 假设 CSV 里的 gene_ID 可能是带版本或不带版本的，我们统一处理
    df['gene_id_clean'] = df['gene_ID'].astype(str).apply(get_gene_id_base)
    
    target_ids = df['gene_id_clean'].tolist()
    
    # 2. 从 GTF 获取新信息
    mapping_data = parse_gtf_and_get_tss(GTF_FILE, target_ids)
    
    print(f"GTF 读取完毕。找到 {len(mapping_data)} / {len(df)} 个基因的 v41 坐标。")
    
    # 3. 更新 DataFrame
    # 我们创建一个列表来存储新数据，以保持行顺序不变
    new_rows = []
    
    not_found_count = 0
    
    for index, row in df.iterrows():
        gid = row['gene_id_clean']
        original_index_gene = row['index_gene'] # 保留原来的 index
        
        if gid in mapping_data:
            info = mapping_data[gid]
            new_rows.append({
                'gene_ID': gid, # 使用去除了版本的 ID，或者你可以选择保留 row['gene_ID']
                'chr': info['chr'],
                'pos': info['pos'],      # 这是 v41 的 TSS
                'gene_name': info['gene_name'], # 使用 v41 的名字（防止名字变更）
                'strand': info['strand'],
                'index_gene': original_index_gene
            })
        else:
            # 如果 GTF 里没找到 (可能基因ID被废弃了)，我们可以选择跳过或者保留旧值并报错
            # 这里选择打印警告
            print(f"警告: 基因 {gid} ({row['gene_name']}) 在 GENCODE v41 中未找到。")
            not_found_count += 1
            # 这种情况下，可以选择不写入，或者写入原来的值。
            # 这里为了数据完整性，暂不写入该行，或者你可以取消下面的注释保留旧行：
            # new_rows.append({'gene_ID': gid, 'chr': 'NOT_FOUND', ...}) 
    
    # 4. 创建新的 DataFrame
    new_df = pd.DataFrame(new_rows, columns=['gene_ID', 'chr', 'pos', 'gene_name', 'strand', 'index_gene'])
    
    # 5. 保存
    new_df.to_csv(OUTPUT_CSV, index=False)
    print(f"处理完成！")
    print(f"新文件已保存至: {OUTPUT_CSV}")
    if not_found_count > 0:
        print(f"注意: 有 {not_found_count} 个基因未在 GENCODE v41 中找到，已从新文件中排除。")

if __name__ == "__main__":
    main()
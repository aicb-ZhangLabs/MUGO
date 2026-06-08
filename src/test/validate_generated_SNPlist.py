'''
we extract Geuvadis VCF file SNPs, and use liftover to convert the coordinates to GRCh38. this script is to validate those SNPs by reading from hg38 ref and compare with its ref. 
'''

import os
import pandas as pd
import argparse
import sys
import pysam # 用于快速读取 FASTA

# ==================== 路径配置 ====================
GENES_FILE = '/home/dongbos/Combine_optim_Borzoi_SNP/dataset/gene_3000_borzoi_gencode_v41_hg38.csv'
SNP_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP/dataset/gene_snps_hg38'
HG38_FASTA = '/home/dongbos/Combine_optim_Borzoi_SNP/dataset/human_genome_hg38/hg38.ml.fa'
# =================================================

def main():
    parser = argparse.ArgumentParser(description='Validate lifted SNPs against hg38 reference genome')
    parser.add_argument('--gene_index', type=int, required=True, help='Row index in gene csv (1-based)')
    args = parser.parse_args()

    # 1. 检查 FASTA 索引
    # pysam 需要 .fai 索引文件。如果没有，它会自动尝试创建，但如果目录不可写会报错。
    if not os.path.exists(HG38_FASTA + ".fai"):
        print(f"Warning: Fasta index not found for {HG38_FASTA}. pysam will try to build it...")

    # 2. 读取基因信息以获取文件名
    print(f"Reading gene file: {GENES_FILE}")
    gene_df = pd.read_csv(GENES_FILE)
    
    row_idx = args.gene_index - 1 
    if row_idx < 0 or row_idx >= len(gene_df):
        print(f"Error: gene_index {args.gene_index} is out of range.")
        sys.exit(1)

    gene_info = gene_df.iloc[row_idx]
    gene_name = gene_info['gene_name']
    print(f"Validating Gene: {gene_name}")

    # 3. 读取 SNP 文件
    snp_file = os.path.join(SNP_DIR, f"{gene_name}_snps_hg38.csv")
    if not os.path.exists(snp_file):
        print(f"Error: SNP file not found: {snp_file}")
        return

    df = pd.read_csv(snp_file)
    if df.empty:
        print("SNP file is empty.")
        return

    print(f"Loaded {len(df)} SNPs. Checking against hg38 reference...")

    # 4. 打开 Reference Genome
    try:
        fasta = pysam.FastaFile(HG38_FASTA)
    except ValueError as e:
        print(f"Error opening FASTA: {e}")
        print("Tip: Ensure you have samtools installed and run: samtools faidx your_genome.fa")
        return

    match_count = 0
    mismatch_count = 0
    mismatch_details = []

    # 5. 逐行验证
    for idx, row in df.iterrows():
        # 获取 hg38 坐标 (注意：CSV里通常是 1-based)
        pos_1based = int(row['POS_hg38'])
        chrom = str(row['CHROM'])
        
        # 确保染色体名称格式匹配 (hg38 通常是 chr1, chr2...)
        if not chrom.startswith('chr'):
            chrom_query = f"chr{chrom}"
        else:
            chrom_query = chrom

        ref_vcf = row['REF'].upper() # VCF 里的 Ref
        
        try:
            # pysam fetch 是 0-based，且左闭右开
            # 1-based pos 100 -> 0-based index 99
            ref_hg38 = fasta.fetch(chrom_query, pos_1based - 1, pos_1based).upper()
        except KeyError:
            print(f"Error: Chromosome {chrom_query} not found in Fasta.")
            continue
        except IndexError:
            print(f"Error: Position {pos_1based} out of bounds for {chrom_query}.")
            continue

        # 比较
        if ref_hg38 == ref_vcf:
            match_count += 1
        else:
            mismatch_count += 1
            # 记录错误信息以便查看
            if mismatch_count <= 5: # 只打印前5个错误
                print(f"Mismatch at {chrom_query}:{pos_1based} | VCF says: {ref_vcf} | hg38 says: {ref_hg38}")
            
            # (可选) 如果你想保存不匹配的行，可以在这里处理

    # 6. 输出统计结果
    total = match_count + mismatch_count
    accuracy = (match_count / total * 100) if total > 0 else 0

    print("-" * 40)
    print(f"Validation Results for {gene_name}")
    print("-" * 40)
    print(f"Total SNPs checked : {total}")
    print(f"Matches (OK)       : {match_count}")
    print(f"Mismatches (Fail)  : {mismatch_count}")
    print(f"Accuracy           : {accuracy:.2f}%")
    print("-" * 40)

    if accuracy < 99.0:
        print("⚠️ Warning: Accuracy is below 99%. LiftOver might have issues or Reference versions differ.")
    else:
        print("✅ Validation Passed: High concordance.")

if __name__ == "__main__":
    main()
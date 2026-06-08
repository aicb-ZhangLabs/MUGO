'''
borzoi using GENCODE v41 (hg38), and Geuvadis is using hg19, should trans VCF from hg19 to hg38 first then just use borzoi GENCODE v41 annotation
'''
import os
import pandas as pd
import argparse
import sys
from io import StringIO
import subprocess
from pyliftover import LiftOver

# ==================== 路径配置 ====================
GENES_FILE = '/home/dongbos/Combine_optim_Borzoi_SNP/dataset/gene_3000_borzoi_gencode_v41_hg38.csv'
OUTPUT_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP/dataset/gene_snps_hg38'
VCF_TEMPLATE = '/home/dongbos/Combine_optim_Borzoi_SNP/dataset/Geuvadis_vcf/GEUVADIS.chr{chr}.PH1PH2_465.IMPFRQFILT_BIALLELIC_PH.annotv2.genotypes.vcf.gz'

# Chain 文件路径
CHAIN_HG38_TO_HG19 = '/home/dongbos/liftover_chains/hg38ToHg19.over.chain.gz'
CHAIN_HG19_TO_HG38 = '/home/dongbos/liftover_chains/hg19ToHg38.over.chain.gz'

SEQUENCE_LENGTH = 524288
# =================================================

def run_bcftools(cmd):
    """运行 bcftools"""
    try:
        output = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
        return output.decode('utf-8')
    except subprocess.CalledProcessError as e:
        print(f"Error running bcftools: {cmd}")
        print(e.output.decode('utf-8'))
        return None

def main():
    parser = argparse.ArgumentParser(description='Extract hg19 SNPs and liftOver to hg38')
    parser.add_argument('--gene_index', type=int, required=True, help='Row index in gene csv (1-based)')
    args = parser.parse_args()

    # 0. 初始化
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    print("Loading Chain files...")
    lo_38_to_19 = LiftOver(CHAIN_HG38_TO_HG19)
    lo_19_to_38 = LiftOver(CHAIN_HG19_TO_HG38)

    # 1. 读取基因信息
    print(f"Reading gene file: {GENES_FILE}")
    gene_df = pd.read_csv(GENES_FILE)
    
    row_idx = args.gene_index - 1 
    if row_idx < 0 or row_idx >= len(gene_df):
        print(f"Error: gene_index {args.gene_index} is out of range.")
        sys.exit(1)

    gene_info = gene_df.iloc[row_idx]
    gene_name = gene_info['gene_name']
    chrom = str(gene_info['chr'])
    tss = int(gene_info['pos'])

    print(f"Processing Gene: {gene_name} | Chr: {chrom} | TSS (hg38): {tss}")

    # 2. 定义 hg38 窗口
    start_hg38 = tss - SEQUENCE_LENGTH // 2
    end_hg38 = tss + SEQUENCE_LENGTH // 2 - 1
    
    # 3. LiftOver Region (hg38 -> hg19)
    res_start = lo_38_to_19.convert_coordinate(f'chr{chrom}', start_hg38)
    res_end = lo_38_to_19.convert_coordinate(f'chr{chrom}', end_hg38)

    if not res_start or not res_end:
        print("Error: Could not map region boundaries to hg19.")
        return
    
    chrom_hg19 = res_start[0][0].replace('chr', '') 
    s_hg19 = res_start[0][1]
    e_hg19 = res_end[0][1]

    query_start = min(s_hg19, e_hg19)
    query_end = max(s_hg19, e_hg19)
    
    query_start -= 2000
    query_end += 2000

    print(f"Mapped Query Region (hg19): chr{chrom_hg19}:{query_start}-{query_end}")

    # 4. 查询 VCF (hg19)
    vcf_path = VCF_TEMPLATE.format(chr=chrom_hg19)
    if not os.path.exists(vcf_path):
        print(f"Error: VCF file not found: {vcf_path}")
        return

    # [修改点 1]: 增加 %INFO/AF 提取频率
    query_cmd = f"bcftools query -r {chrom_hg19}:{query_start}-{query_end} -f '%CHROM\t%POS\t%REF\t%ALT\t%INFO/AF\n' -i 'TYPE=\"snp\" && N_ALT=1' {vcf_path}"
    
    raw_snps = run_bcftools(query_cmd)
    if not raw_snps:
        print("No SNPs found in this region.")
        return

    # [修改点 2]: 读取 AF 并过滤 MAF > 0.05
    # 增加 'AF' 列名
    snps_hg19_df = pd.read_csv(StringIO(raw_snps), sep='\t', names=['CHROM', 'POS', 'REF', 'ALT', 'AF'])
    
    # 转换为数字 (处理可能的异常值)
    snps_hg19_df['AF'] = pd.to_numeric(snps_hg19_df['AF'], errors='coerce')
    snps_hg19_df = snps_hg19_df.dropna(subset=['AF']) # 删掉没有频率的行

    # 计算 MAF (Minor Allele Frequency)
    # 如果 AF > 0.5，说明 ALT 是主等位基因，MAF = 1 - AF
    snps_hg19_df['MAF'] = snps_hg19_df['AF'].apply(lambda x: x if x <= 0.5 else 1 - x)

    # 执行过滤: MAF > 0.05
    print(f"Total SNPs extracted: {len(snps_hg19_df)}")
    snps_hg19_df = snps_hg19_df[snps_hg19_df['MAF'] > 0.05].copy()
    print(f"SNPs after MAF > 0.05 filtering: {len(snps_hg19_df)}")

    if snps_hg19_df.empty:
        print("No SNPs left after filtering.")
        return

    print("Converting to hg38...")

    # 5. LiftOver SNPs (hg19 -> hg38)
    hg38_positions = []
    valid_indices = []

    for idx, row in snps_hg19_df.iterrows():
        # VCF 1-based -> 0-based
        res = lo_19_to_38.convert_coordinate(f"chr{row['CHROM']}", row['POS'] - 1)
        
        if res:
            # 0-based -> 1-based
            new_pos = res[0][1] + 1
            hg38_positions.append(new_pos)
            valid_indices.append(idx)
    
    final_df = snps_hg19_df.loc[valid_indices].copy()
    final_df['POS_hg38'] = hg38_positions
    final_df['CHROM'] = chrom 

    # 6. 最终过滤 (确保在 524288 窗口内)
    final_df = final_df[
        (final_df['POS_hg38'] >= start_hg38) & 
        (final_df['POS_hg38'] <= end_hg38)
    ].copy()

    final_df['dist_to_tss'] = final_df['POS_hg38'] - tss
    
    # 保存 (可以选择保留 MAF 列)
    output_file = os.path.join(OUTPUT_DIR, f"{gene_name}_snps_hg38.csv")
    final_df.to_csv(output_file, index=False)
    
    print(f"Success! Saved {len(final_df)} SNPs to {output_file}")

if __name__ == "__main__":
    main()
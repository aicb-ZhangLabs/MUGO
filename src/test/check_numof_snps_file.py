import os
import glob
import pandas as pd
import random

# ==================== 配置 ====================
# 你的输出目录
TARGET_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP/dataset/gene_snps_hg38'
# 抽样数量
SAMPLE_SIZE = 100
# =============================================

def check_generated_files():
    # 1. 获取所有生成的 csv 文件
    print(f"Searching for files in: {TARGET_DIR} ...")
    all_files = glob.glob(os.path.join(TARGET_DIR, "*_snps_hg38.csv"))
    total_count = len(all_files)
    
    print(f"Total SNP files found: {total_count}")

    if total_count == 0:
        print("No files found. Please check your output directory path.")
        return

    # 2. 随机抽样
    num_to_check = min(SAMPLE_SIZE, total_count)
    sampled_files = random.sample(all_files, num_to_check)

    print(f"\nChecking {num_to_check} random files...\n")
    
    # 表头格式化
    header = f"{'Status':<10} | {'Rows':<6} | {'MAF_Check':<10} | {'Filename'}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    # 统计计数器
    stats = {
        "valid": 0,    # 有数据
        "empty": 0,    # 空文件(只有表头或完全空)
        "error": 0     # 读取报错
    }

    # 3. 遍历检查
    for file_path in sampled_files:
        file_name = os.path.basename(file_path)
        status = "UNKNOWN"
        row_count = 0
        maf_status = "N/A"

        try:
            # 检查文件大小是否为 0
            if os.path.getsize(file_path) == 0:
                status = "EMPTY_0B"
                stats["empty"] += 1
            else:
                # 读取 CSV
                df = pd.read_csv(file_path)
                row_count = len(df)

                if df.empty:
                    status = "EMPTY_DF" # 有表头但没数据
                    stats["empty"] += 1
                else:
                    status = "OK"
                    stats["valid"] += 1
                    
                    # 简单检查一下是否有 MAF 列 (验证之前的过滤逻辑)
                    if 'MAF' in df.columns:
                        # 检查是否有 MAF <= 0.05 的漏网之鱼
                        min_maf = df['MAF'].min()
                        if min_maf <= 0.05:
                            maf_status = f"WARN({min_maf:.2f})"
                        else:
                            maf_status = "Pass"
                    else:
                        maf_status = "Missing"

        except Exception as e:
            status = "ERROR"
            stats["error"] += 1
            # print(f"Debug: {e}") # 如果需要调试去掉注释

        # 打印单行结果 (OK 用绿色显示，Empty 用红色显示如果支持终端颜色，这里只用文本)
        print(f"{status:<10} | {row_count:<6} | {maf_status:<10} | {file_name}")

    # 4. 最终汇总
    print("-" * len(header))
    print(f"\nSummary for {num_to_check} sampled files:")
    print(f"  [OK]    Valid files (Non-empty): {stats['valid']}")
    print(f"  [FAIL]  Empty files            : {stats['empty']}")
    print(f"  [FAIL]  Read Errors            : {stats['error']}")

    # 简单的判定
    if stats['valid'] == num_to_check:
        print("\n✅ All sampled files look good!")
    elif stats['valid'] > 0:
        print(f"\n⚠️  Some files are empty ({stats['empty']}). This might be normal if some genes have no SNPs in the region.")
    else:
        print("\n❌ All sampled files are empty or broken. Check your extraction script.")

if __name__ == "__main__":
    check_generated_files()
'''
rank genes based on borzoi wildtype expression (RNA ONLY). 
This creates the "Anchor" list for other modalities.
'''

import pandas as pd
import os
import glob
from tqdm import tqdm

# ================= ⚙️ 配置区域 =================

# 缓存文件所在的输入目录 (之前的 cache 目录)
INPUT_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP/src/supple_figures'

# 结果保存目录
OUTPUT_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP/src/interpretability/newversion_table2/top100_highexp_gene'

# 筛选配置
TOP_K = 100
# 按 Baseline (野生型表达量) 排序
SORT_COLUMN = 'Baseline'  

# ================= 🚀 执行逻辑 =================

def main():
    # 1. 创建输出目录
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"📁 Created directory: {OUTPUT_DIR}")

    # 2. 扫描 cache 文件
    # 🔥 修改点：只扫描 RNA 的文件 (cache_RNA_*.csv)
    search_path = os.path.join(INPUT_DIR, 'cache_RNA_*.csv')
    files = glob.glob(search_path)
    
    if not files:
        print(f"❌ No RNA cache files found in {INPUT_DIR}")
        print("   Please check if your RNA cache files are named like 'cache_RNA_tissue.csv'")
        return

    print(f"🔍 Found {len(files)} RNA files. Extracting Top {TOP_K} by {SORT_COLUMN}...")

    count = 0
    for f in tqdm(files):
        try:
            basename = os.path.basename(f)
            
            # 🔥 双重检查：确保只处理 RNA
            if 'RNA' not in basename:
                continue

            # 读取数据
            df = pd.read_csv(f)
            
            if df.empty:
                continue
            
            # 检查列是否存在
            if SORT_COLUMN not in df.columns:
                print(f"⚠️ Warning: Column '{SORT_COLUMN}' not found in {basename}, skipping.")
                continue

            # 3. 排序并提取 Top K
            # ascending=False 表示降序 (从大到小)，即选 Baseline 最高的
            df_sorted = df.sort_values(by=SORT_COLUMN, ascending=False)
            df_top = df_sorted.head(TOP_K)

            # 4. 构造输出文件名
            # 结果将是: top100_high_expr_cache_RNA_blood.csv
            out_filename = f"top{TOP_K}_high_expr_{basename}"
            out_path = os.path.join(OUTPUT_DIR, out_filename)

            # 5. 保存
            df_top.to_csv(out_path, index=False)
            count += 1
            
        except Exception as e:
            print(f"❌ Error processing {f}: {e}")

    print(f"\n✅ Done! Saved {count} RNA gene lists to: {OUTPUT_DIR}")
    print("👉 Next Step: Run 'distribute_genes.py' to generate ATAC/ChIP lists from these RNA lists.")

if __name__ == "__main__":
    main()
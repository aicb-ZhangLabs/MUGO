import pandas as pd
import os
import glob

# ================= ⚙️ 配置区域 =================

BASE_PATH = '/home/dongbos/Combine_optim_Borzoi_SNP/src/interpretability/newversion_table2/top100_highexp_gene'
TARGET_MODALITIES = ['ATAC', 'ChIP', 'DNAse', 'CAGE']

# ================= 🚀 执行逻辑 =================

def distribute_genes():
    # 找 RNA 文件
    rna_pattern = os.path.join(BASE_PATH, "top100_high_expr_cache_RNA_*.csv")
    rna_files = glob.glob(rna_pattern)
    
    if not rna_files:
        print(f"❌ 没找到 RNA 文件！")
        return

    print(f"🔍 找到 {len(rna_files)} 个 RNA 源文件，开始分发...")

    for rna_file in rna_files:
        basename = os.path.basename(rna_file)
        
        # ⚠️⚠️⚠️ 修复点：长度是 27，不是 25 ⚠️⚠️⚠️
        # 前缀 "top100_high_expr_cache_RNA_" 长度为 27
        tissue = basename[27:-4] 
        
        print(f"\nProcessing Tissue: {tissue} ...")
        
        try:
            df = pd.read_csv(rna_file)
            if 'Gene' not in df.columns: continue
            
            gene_list_df = df[['Gene']].copy()
            
            for mod in TARGET_MODALITIES:
                # 这样生成的文件名才是干净的：..._ATAC_pancreas.csv
                target_filename = f"top100_high_expr_cache_{mod}_{tissue}.csv"
                target_path = os.path.join(BASE_PATH, target_filename)
                
                gene_list_df.to_csv(target_path, index=False)
                print(f"  ✅ Generated: {target_filename}")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")

    print("\n🎉 修复完成！文件名现在正常了。")

if __name__ == "__main__":
    distribute_genes()
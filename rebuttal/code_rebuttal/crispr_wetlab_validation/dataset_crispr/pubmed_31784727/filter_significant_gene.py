import pandas as pd
import os

# ================= ⚙️ 配置路径 =================
# 替换为你的实际相对/绝对路径
CSV_PATH = "/home/dongbos/Combine_optim_Borzoi_SNP/rebuttal/code_rebuttal/crispr_wetlab_validation/dataset_crispr/pubmed_31784727/Fulco_2019_Table3a.csv"

def analyze_significant_genes():
    if not os.path.exists(CSV_PATH):
        print(f"❌ 找不到文件: {CSV_PATH}")
        return

    print("📊 正在加载 Fulco 2019 (Table 3a) 数据集...")
    df = pd.read_csv(CSV_PATH)

    # 1. 清理列名（防止 Excel 导出时带有隐藏空格）
    df.columns = df.columns.str.strip()

    # 2. 核心过滤：只保留 Significant == TRUE 的行
    # 这里用 astype(str) 并转大写，是为了兼容 Excel 导出的布尔值 (True) 和字符串 ("TRUE")
    df_sig = df[df['Significant'].astype(str).str.strip().str.upper() == 'TRUE']

    # 3. 统计每个 Gene 拥有多少个 Significant 的 Enhancer
    gene_counts = df_sig['Gene'].value_counts()
    
    total_significant_genes = len(gene_counts)
    total_significant_enhancers = len(df_sig)

    print("="*60)
    print("🎯 CRISPR Ground Truth 统计结果")
    print("="*60)
    print(f"✅ 共有 {total_significant_genes} 个基因拥有至少 1 个显著的 Enhancer。")
    print(f"✅ 整个表里共有 {total_significant_enhancers} 个显著的 Enhancer-Gene 对。")
    print("-" * 60)
    
    print("🔥 强烈建议用以下排名前列的基因跑 MUGO 验证 (靶点越多，P-value 越容易显著):")
    print("基因名称\t\t显著 Enhancer 数量")
    print("-" * 60)
    
    # 打印排名前 20 的基因
    for gene, count in gene_counts.head(20).items():
        print(f"{gene:<15}\t{count}")
    
    print("="*60)

if __name__ == "__main__":
    analyze_significant_genes()
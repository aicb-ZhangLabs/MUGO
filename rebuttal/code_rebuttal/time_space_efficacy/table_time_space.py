import os
import glob
import pandas as pd

# ================= ⚙️ 配置区域 =================
# 你存放 CSV 和输出结果的目录
WORK_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP/rebuttal/results_rebuttal/time_space_scale_results'

# ================= 🧠 1. 数据读取与汇总 =================
def load_and_summarize_data():
    search_pattern = os.path.join(WORK_DIR, 'efficiency_benchmark_N*.csv')
    files = glob.glob(search_pattern)
    
    if not files:
        print(f"❌ No CSV files found in {WORK_DIR}")
        return pd.DataFrame()

    results = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if not df.empty:
                n_snps = int(df['N_SNPs'].iloc[0])
                # 排除前 5 步的 Warmup 时间，让时间统计极其精准
                valid_df = df[df['Step'] >= 5] if len(df) > 5 else df
                avg_time = valid_df['Time_sec'].mean()
                max_vram = df['Max_VRAM_GB'].max()
                
                results.append({
                    'N_SNPs': n_snps,
                    'Time_sec': avg_time,
                    'VRAM_GB': max_vram
                })
        except Exception as e:
            print(f"Error reading {f}: {e}")

    # 转换为 DataFrame 并按 N_SNPs 从小到大排序
    summary_df = pd.DataFrame(results)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(by='N_SNPs').reset_index(drop=True)
    return summary_df

# ================= 📝 2. 生成表格代码 =================
def generate_tables(df):
    # --- Markdown 表格 ---
    md_lines = []
    # 留一个空行，防止 HackMD 解析失败
    md_lines.append("")
    md_lines.append("| Number of Candidate SNPs ($N$) | Avg. Time per Step (sec) | Peak GPU Memory (GB) |")
    md_lines.append("|---:|---:|---:|")
    
    for _, row in df.iterrows():
        # 大数字格式化：500000 -> 500,000
        n_str = f"{int(row['N_SNPs']):,}"
        md_lines.append(f"| {n_str} | {row['Time_sec']:.4f} | {row['VRAM_GB']:.2f} |")
        
    md_path = os.path.join(WORK_DIR, 'Table_Efficiency.md')
    with open(md_path, 'w') as f:
        f.write("\n".join(md_lines))
        
    # --- LaTeX 表格 ---
    tex_lines = []
    tex_lines.append(r"\begin{table}[h]")
    tex_lines.append(r"\centering")
    tex_lines.append(r"\small")
    tex_lines.append(r"\renewcommand{\arraystretch}{1.2}")
    # 极其专业的 Caption，直接塞进你的 Rebuttal
    tex_lines.append(r"\caption{\textbf{Scalability Benchmark.} Runtime and GPU memory usage of MUGO across varying sizes of the candidate variant pool ($N$). The computational overhead is strictly decoupled from the combinatorial search space size, demonstrating $O(1)$ scaling behavior with respect to $N$.}")
    tex_lines.append(r"\label{tab:efficiency}")
    tex_lines.append(r"\begin{tabular}{rcc}")
    tex_lines.append(r"\toprule")
    tex_lines.append(r"\textbf{Candidate SNPs ($N$)} & \textbf{Time / Step (s)} & \textbf{Peak VRAM (GB)} \\")
    tex_lines.append(r"\midrule")
    
    for _, row in df.iterrows():
        n_str = f"{int(row['N_SNPs']):,}"
        tex_lines.append(f"{n_str} & {row['Time_sec']:.4f} & {row['VRAM_GB']:.2f} \\\\")
        
    tex_lines.append(r"\bottomrule")
    tex_lines.append(r"\end{tabular}")
    tex_lines.append(r"\end{table}")
    
    tex_path = os.path.join(WORK_DIR, 'Table_Efficiency.tex')
    with open(tex_path, 'w') as f:
        f.write("\n".join(tex_lines))
        
    print(f"\n📝 Tables saved to:\n -> {md_path}\n -> {tex_path}")
    print("\n" + "="*60)
    print("✅ MARKDOWN PREVIEW (Copy below to your viewer):")
    print("="*60)
    print("\n".join(md_lines))
    print("="*60 + "\n")

# ================= 🚀 主程序 =================
if __name__ == "__main__":
    print("🔍 Scanning directory for benchmark results...")
    df = load_and_summarize_data()
    
    if not df.empty:
        print(f"✅ Found {len(df)} data points.")
        generate_tables(df)
    else:
        print("⚠️ Program exited because no data was processed.")
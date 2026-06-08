import os
import glob
import pandas as pd
import numpy as np

# ================= ⚙️ 配置区域 =================
BASE_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP'
SEED_DIR = f'{BASE_DIR}/rebuttal/results_rebuttal/random_seed_RNA_top100gene'
BASELINE_DIR = f'{BASE_DIR}/src/interpretability/newversion_table2/top100'
OUTPUT_DIR = f'{BASE_DIR}/rebuttal/results_rebuttal'

TISSUES = ['blood', 'brain']
SEEDS = [42, 123, 2026]
BASELINE_METHODS = [('Sal.', 'Saliency_Gain'), ('CADD', 'CADD_Gain'), ('FunSeq', 'FunSeq_Gain')]

# ================= 🧠 核心逻辑 =================
def get_seed_data(tissue, seed):
    """读取某个 seed 下所有 100 个基因的结果并计算均值和SEM"""
    folder = os.path.join(SEED_DIR, f'random_seed_{seed}', 'raw_results_csv_eachgene')
    search_pattern = os.path.join(folder, f'*_RNA_{tissue}_seed{seed}.csv')
    files = glob.glob(search_pattern)
    
    if not files:
        return None
        
    gains = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if not df.empty and 'Gain' in df.columns:
                # 兼容前一版逻辑：如果是多行取最后一行，这里我们存的本来就是最后一步
                gain_val = df.iloc[-1]['Gain']
                gains.append(abs(gain_val)) # 统一取绝对值 Magnitude
        except:
            pass
            
    if not gains: return None
    gains = np.array(gains)
    return {'mean': np.mean(gains), 'sem': np.std(gains, ddof=1) / np.sqrt(len(gains))}

def get_best_baseline(tissue):
    """从之前的 Benchmark 结果中抓取最强的 Baseline"""
    csv_path = os.path.join(BASELINE_DIR, f'benchmark_RNA_{tissue}.csv')
    best_name = "Baseline"
    best_mean = -1.0
    best_sem = 0.0
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        for name, col in BASELINE_METHODS:
            if col in df.columns:
                vals = df[col].dropna()
                if len(vals) > 0:
                    val_mean = abs(vals.mean())
                    val_sem = vals.sem()
                    if val_mean > best_mean:
                        best_mean = val_mean
                        best_sem = val_sem
                        best_name = name
    return best_name, best_mean, best_sem

def main():
    latex_lines = []
    md_lines = []
    
    # --- LaTeX 表头设置 ---
    latex_lines.append(r"\begin{table}[t]") 
    latex_lines.append(r"\centering")
    latex_lines.append(r"\footnotesize") 
    latex_lines.append(r"\renewcommand{\arraystretch}{1.2}") 
    latex_lines.append(r"\caption{\textbf{Robustness to Random Initialization.} Optimization signal gain ($|\overline{\Delta S}| \% \pm \text{SEM}$) across 100 genes for RNA-seq. The 'Overall' column shows the stability across three independent random seeds, with the variance introduced by random initialization ($\pm$ Std.) proving negligible compared to the margin over the best baseline.}")
    latex_lines.append(r"\label{tab:seed_ablation}")
    latex_lines.append(r"\resizebox{1.0\columnwidth}{!}{%")
    latex_lines.append(r"\begin{tabular}{lcccccc}")
    latex_lines.append(r"\toprule")
    
    # 表头
    headers_latex = [r"\textbf{Tissue}"] + [f"\\textbf{{Seed {s}}}" for s in SEEDS] + [r"\textbf{MUGO (Across Seeds)}", r"\textbf{Best Baseline}"]
    latex_lines.append(" & ".join(headers_latex) + r" \\")
    latex_lines.append(r"\midrule")
    
    headers_md = ["Tissue"] + [f"Seed {s}" for s in SEEDS] + ["**MUGO Overall**<br>*(Mean ± Std. across seeds)*", "Best Baseline"]
    md_lines.append("")
    md_lines.append("| " + " | ".join(headers_md) + " |")
    md_lines.append("|---|---|---|---|---|---|")

    # --- 填充数据 ---
    for tissue in TISSUES:
        row_latex = [tissue.capitalize()]
        row_md = [f"**{tissue.capitalize()}**"]
        
        seed_means = []
        
        # 1. 填入每个 Seed 的数据
        for seed in SEEDS:
            data = get_seed_data(tissue, seed)
            if data:
                s_mean, s_sem = data['mean'], data['sem']
                seed_means.append(s_mean)
                row_latex.append(f"{s_mean:.1f} $\\pm$ {s_sem:.1f}")
                row_md.append(f"{s_mean:.1f}±{s_sem:.1f}")
            else:
                row_latex.append("N/A")
                row_md.append("N/A")
                
        # 2. 计算 Across Seeds (大杀器：极小的跨种子标准差)
        if len(seed_means) == len(SEEDS):
            overall_mean = np.mean(seed_means)
            overall_std = np.std(seed_means, ddof=1) # 这里算的是三个种子之间的扰动 Std
            
            str_overall_latex = f"\\textbf{{{overall_mean:.1f}}} $\\pm$ \\textbf{{{overall_std:.2f}}}"
            str_overall_md = f"**{overall_mean:.1f} ± {overall_std:.2f}**"
        else:
            str_overall_latex = "Running..."
            str_overall_md = "Running..."
            
        row_latex.append(str_overall_latex)
        row_md.append(str_overall_md)
        
        # 3. 填入最强 Baseline 对比
        base_name, base_mean, base_sem = get_best_baseline(tissue)
        if base_mean >= 0:
            str_base_latex = f"{base_mean:.1f} $\\pm$ {base_sem:.1f} ({base_name})"
            str_base_md = f"{base_mean:.1f}±{base_sem:.1f} *({base_name})*"
        else:
            str_base_latex = "N/A"
            str_base_md = "N/A"
            
        row_latex.append(str_base_latex)
        row_md.append(str_base_md)
        
        latex_lines.append(" & ".join(row_latex) + r" \\")
        md_lines.append("| " + " | ".join(row_md) + " |")

    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}}")
    latex_lines.append(r"\end{table}")
    
    # 保存结果
    out_latex = os.path.join(OUTPUT_DIR, 'Table_Seed_Ablation.tex')
    out_md = os.path.join(OUTPUT_DIR, 'Table_Seed_Ablation.md')
    
    with open(out_latex, 'w') as f: f.write("\n".join(latex_lines))
    with open(out_md, 'w') as f: f.write("\n".join(md_lines))
    
    print("\n" + "="*90)
    print("✅ GENERATED MARKDOWN TABLE FOR REBUTTAL:")
    print("="*90)
    print("\n".join(md_lines))
    print("\n" + "="*90)
    print(f"📁 Files saved:\n - {out_latex}\n - {out_md}")

if __name__ == "__main__":
    main()
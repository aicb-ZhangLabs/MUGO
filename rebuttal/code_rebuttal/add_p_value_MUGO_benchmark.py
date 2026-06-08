'''
add p value to the MUGO benchmark table against Saliency, CADD, and FunSeq. 
MUGO pair with top performance baseline for each modality and tissue. 
'''
import pandas as pd
import numpy as np
import os
from scipy.stats import wilcoxon

# ================= ⚙️ 配置区域 =================

BASE_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP'
DATA_DIR = f'{BASE_DIR}/src/interpretability/newversion_table2/top100'
OUTPUT_DIR = f'{BASE_DIR}/rebuttal/results_rebuttal'
os.makedirs(OUTPUT_DIR, exist_ok=True)

METHODS_CONFIG = [
    ('MUGO',   'Borzoi_Gain'),
    ('Sal.',   'Saliency_Gain'),
    ('CADD',   'CADD_Gain'),
    ('FunSeq', 'FunSeq_Gain')
]

MODALITIES_CONFIG = {
    'RNA-seq': ['blood', 'brain', 'liver', 'heart', 'muscle', 'pancreas', 'lung', 'kidney'],
    'CAGE':    ['blood', 'brain', 'liver', 'heart', 'muscle', 'pancreas'],
    'ATAC':    ['blood', 'brain', 'liver', 'heart', 'muscle', 'pancreas'],
    'DNAse':   ['blood', 'brain', 'liver'], 
    'ChIP':    ['blood', 'brain', 'liver']
}

FILE_TAG_MAP = {
    'RNA-seq': 'RNA', 'CAGE': 'CAGE', 'ATAC': 'ATAC', 'DNAse': 'DNAse', 'ChIP': 'ChIP'
}

# ================= 🧠 核心逻辑 =================

def get_stats_corrected(modality, tissue):
    stats = {'found': False}
    file_tag = FILE_TAG_MAP.get(modality, modality)
    filename = f'benchmark_{file_tag}_{tissue}.csv'
    csv_path = os.path.join(DATA_DIR, filename)
    
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if df.empty: return stats
            stats['found'] = True
            
            raw_data = {}
            for disp_name, col_name in METHODS_CONFIG:
                if col_name in df.columns:
                    vals = df[col_name].dropna()
                    raw_data[disp_name] = vals.values
                    if len(vals) > 0:
                        stats[disp_name] = {'mean': abs(vals.mean()), 'sem': vals.sem()}
                    else:
                        stats[disp_name] = {'mean': 0.0, 'sem': 0.0}
                else:
                    raw_data[disp_name] = np.array([])
                    stats[disp_name] = {'mean': 0.0, 'sem': 0.0}
            
            # --- 🌟 P-value 计算 (MUGO vs 最强 Baseline) ---
            stats['MUGO']['pval_latex'] = ""
            stats['MUGO']['pval_md'] = ""
            mugo_vals = raw_data.get('MUGO', np.array([]))
            
            if len(mugo_vals) > 0:
                best_base_name = None
                best_base_mean = -1
                for b_name in ['Sal.', 'CADD', 'FunSeq']:
                    b_mean = stats.get(b_name, {}).get('mean', -1)
                    if b_mean > best_base_mean:
                        best_base_mean = b_mean
                        best_base_name = b_name
                
                best_base_vals = raw_data.get(best_base_name, np.array([]))
                
                if best_base_name and len(best_base_vals) == len(mugo_vals):
                    diff = mugo_vals - best_base_vals
                    if np.any(diff != 0): 
                        res = wilcoxon(mugo_vals, best_base_vals, alternative='greater')
                        pval = res.pvalue
                        
                        # 智能格式化 p-value
                        if pval < 0.001:
                            pval_str = f"{pval:.1e}".replace('e-0', 'e-')
                            stars = "***"
                        elif pval < 0.01:
                            pval_str = f"{pval:.3f}"
                            stars = "**"
                        elif pval < 0.05:
                            pval_str = f"{pval:.3f}"
                            stars = "*"
                        else:
                            pval_str = f"{pval:.2f}"
                            stars = ""
                        
                        # LaTeX 用数学模式的上标，Markdown 换成 HTML 的 <sup>
                        if stars:
                            stats['MUGO']['pval_latex'] = f"$^{{{stars}}}}$ ($p={pval_str}$)"
                            stats['MUGO']['pval_md'] = f" <sup>{stars}</sup>(p={pval_str})"
                        else:
                            stats['MUGO']['pval_latex'] = f" ($p={pval_str}$)"
                            stats['MUGO']['pval_md'] = f" (p={pval_str})"
                        
        except Exception as e:
            print(f"❌ Error reading {filename}: {e}")
            stats['found'] = False
            
    return stats

def generate_tables():
    latex_lines = []
    md_lines = []
    
    # --- LaTeX Header ---
    latex_lines.append(r"\begin{table}[t]") 
    latex_lines.append(r"\centering")
    latex_lines.append(r"\scriptsize") 
    latex_lines.append(r"\setlength{\tabcolsep}{3pt}")
    latex_lines.append(r"\renewcommand{\arraystretch}{1.2}") 
    latex_lines.append(r"\caption{\textbf{Multi-modal Benchmark.} Magnitude of Mean Gain ($|\overline{\Delta S}| \% \pm \text{SEM}$). \textbf{Bold} indicates best performance. Statistical significance comparing MUGO against the best baseline is denoted by * ($p < 0.05$), ** ($p < 0.01$), and *** ($p < 0.001$) using the Wilcoxon signed-rank test. Exact $p$-values are provided in parentheses.}")
    latex_lines.append(r"\label{tab:multimodal_benchmark_sem}")
    latex_lines.append(r"\resizebox{1.0\columnwidth}{!}{%")
    latex_lines.append(r"\begin{tabular}{llcccc}")
    latex_lines.append(r"\toprule")
    
    # --- Headers ---
    methods_latex = [f"\\textbf{{{m[0]}}}" for m in METHODS_CONFIG]
    latex_lines.append(" & ".join([r"\textbf{Modality}", r"\textbf{Tissue}"] + methods_latex) + r" \\")
    latex_lines.append(r"\midrule")
    
    methods_md = [m[0] for m in METHODS_CONFIG]
    
    # HackMD 必须要个空行，在生成的时候加上
    md_lines.append("")
    md_lines.append("| Modality | Tissue | " + " | ".join(methods_md) + " |")
    md_lines.append("|---|---|---:|---:|---:|---:|")

    mod_keys = list(MODALITIES_CONFIG.keys())

    for idx, (mod, tissues) in enumerate(MODALITIES_CONFIG.items()):
        first_row = True
        has_data = False
        
        for tissue in tissues:
            data = get_stats_corrected(mod, tissue)
            if not data.get('found'): continue
            has_data = True
            
            # Labels
            mod_latex = f"\\textbf{{{mod}}}" if first_row else ""
            mod_md = f"**{mod}**" if first_row else ""
            tissue_str = tissue.capitalize()
            if tissue_str == 'Pancreas': tissue_str = 'Panc.'
            if tissue_str == 'Muscle': tissue_str = 'Musc.'
            
            row_latex = [mod_latex, tissue_str]
            row_md = [mod_md, tissue_str]
            
            # Find best mean for bolding
            means = [data.get(m[0], {}).get('mean', -1) for m in METHODS_CONFIG]
            best_mean = max(means) if means else 0
            
            for m_name, _ in METHODS_CONFIG:
                stats = data.get(m_name, {'mean': 0.0, 'sem': 0.0})
                val_mean = stats['mean']
                val_sem = stats['sem']
                
                sig_latex = stats.get('pval_latex', '') if m_name == 'MUGO' else ''
                sig_md = stats.get('pval_md', '') if m_name == 'MUGO' else ''
                
                # LaTeX Formatting
                str_latex = f"{val_mean:.1f} $\\pm$ {val_sem:.1f}"
                if val_mean == best_mean and val_mean > 0:
                    str_latex = f"\\textbf{{{val_mean:.1f}}}{sig_latex} $\\pm$ {val_sem:.1f}"
                elif m_name == 'MUGO' and sig_latex:
                    str_latex = f"{val_mean:.1f}{sig_latex} $\\pm$ {val_sem:.1f}"
                row_latex.append(str_latex)
                
                # Markdown Formatting (优化防崩排版)
                str_md = f"{val_mean:.1f}±{val_sem:.1f}"
                if val_mean == best_mean and val_mean > 0:
                    str_md = f"**{val_mean:.1f}**{sig_md} ±{val_sem:.1f}"
                elif m_name == 'MUGO' and sig_md:
                    str_md = f"{val_mean:.1f}{sig_md} ±{val_sem:.1f}"
                row_md.append(str_md)
                
            latex_lines.append(" & ".join(row_latex) + r" \\")
            md_lines.append("| " + " | ".join(row_md) + " |")
            
            if first_row: first_row = False
            
        if idx < len(mod_keys) - 1 and has_data:
            latex_lines.append(r"\midrule")

    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}}")
    latex_lines.append(r"\end{table}")
    
    return "\n".join(latex_lines), "\n".join(md_lines)

def main():
    latex_code, md_code = generate_tables()
    
    out_latex = os.path.join(OUTPUT_DIR, 'Table2_Benchmark_Rebuttal.tex')
    with open(out_latex, 'w') as f: f.write(latex_code)
    
    out_md = os.path.join(OUTPUT_DIR, 'Table2_Benchmark_Rebuttal.md')
    with open(out_md, 'w') as f: f.write(md_code)
    
    print("\n" + "="*80)
    print("✅ GENERATED MARKDOWN TABLE (Copy below into your viewer):")
    print("="*80)
    print(md_code)
    print("\n" + "="*80)
    print(f"📁 Files saved:\n - {out_latex}\n - {out_md}")

if __name__ == "__main__":
    main()
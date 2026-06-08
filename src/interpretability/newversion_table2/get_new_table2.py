import pandas as pd
import numpy as np
import os
import glob

# ================= ⚙️ 配置区域 =================

# 基础路径
BASE_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP'

# 核心数据目录 (Top 100 Genes)
DATA_DIR = f'{BASE_DIR}/src/interpretability/newversion_table2/top100'

# 输出目录
OUTPUT_DIR = os.getcwd()

# 方法配置
METHODS_CONFIG = [
    ('MUGO',   'Borzoi_Gain'),
    ('Sal.',   'Saliency_Gain'),
    ('CADD',   'CADD_Gain'),
    ('FunSeq', 'FunSeq_Gain')
]

# 模态与组织配置
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
    """
    逻辑修改：先求 Mean (保留正负号让噪声抵消)，再取 Abs。
    """
    stats = {'found': False}
    
    file_tag = FILE_TAG_MAP.get(modality, modality)
    filename = f'benchmark_{file_tag}_{tissue}.csv'
    csv_path = os.path.join(DATA_DIR, filename)
    
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if df.empty: return stats
            
            stats['found'] = True
            
            for disp_name, col_name in METHODS_CONFIG:
                if col_name in df.columns:
                    # 1. 获取原始数据 (不取绝对值，保留正负号)
                    vals = df[col_name].dropna()
                    
                    if len(vals) > 0:
                        # 2. 先求 Mean (让正负抵消)
                        raw_mean = vals.mean()
                        
                        # 3. 再取 Abs (作为最终的 Magnitude 展示)
                        final_mean = abs(raw_mean)
                        
                        # 4. SEM 计算 (基于原始分布)
                        sem_val = vals.sem()
                        
                        stats[disp_name] = {
                            'mean': final_mean,
                            'sem': sem_val
                        }
                    else:
                        stats[disp_name] = {'mean': 0.0, 'sem': 0.0}
                else:
                    stats[disp_name] = {'mean': 0.0, 'sem': 0.0}
        except Exception as e:
            print(f"❌ Error reading {filename}: {e}")
            stats['found'] = False
            
    return stats

def generate_latex_table():
    latex_lines = []
    
    # --- LaTeX Header ---
    latex_lines.append(r"\begin{table}[t]") 
    latex_lines.append(r"\centering")
    latex_lines.append(r"\scriptsize") 
    latex_lines.append(r"\setlength{\tabcolsep}{4pt}")
    latex_lines.append(r"\renewcommand{\arraystretch}{1.2}") 
    
    # Caption
    latex_lines.append(r"\caption{\textbf{Multi-modal Benchmark.} Magnitude of Mean Gain ($|\overline{\Delta S}| \% \pm \text{SEM}$). \textbf{Bold} indicates best performance.}")
    latex_lines.append(r"\label{tab:multimodal_benchmark_sem}")
    
    # Resizebox
    latex_lines.append(r"\resizebox{1.0\columnwidth}{!}{%")
    
    latex_lines.append(r"\begin{tabular}{llcccc}")
    latex_lines.append(r"\toprule")
    
    methods_header = [f"\\textbf{{{m[0]}}}" for m in METHODS_CONFIG]
    header_row = [r"\textbf{Modality}", r"\textbf{Tissue}"] + methods_header
    latex_lines.append(" & ".join(header_row) + r" \\")
    latex_lines.append(r"\midrule")
    
    # --- Console Preview ---
    print("\n" + "="*100)
    print(f"📊 GENERATING TABLE (Logic: |Mean(x)|, Fixed LaTeX Math Mode)")
    print("="*100)
    print(f"{'Modality':<8} | {'Tissue':<8} | " + " | ".join([f"{m[0]:<15}" for m in METHODS_CONFIG]))
    print("-" * 100)

    mod_keys = list(MODALITIES_CONFIG.keys())

    for idx, (mod, tissues) in enumerate(MODALITIES_CONFIG.items()):
        first_row = True
        has_data = False
        group_lines = []
        
        for tissue in tissues:
            data = get_stats_corrected(mod, tissue)
            if not data.get('found'): continue
            has_data = True
            
            mod_str = f"\\textbf{{{mod}}}" if first_row else ""
            tissue_str = tissue.capitalize()
            if tissue_str == 'Pancreas': tissue_str = 'Panc.'
            if tissue_str == 'Muscle': tissue_str = 'Musc.'
            
            row_latex = [mod_str, tissue_str]
            row_text_parts = [f"{mod if first_row else '':<8}", f"{tissue:<8}"]
            
            # 找出最大 Mean 值用于加粗
            means = [data.get(m[0], {}).get('mean', -1) for m in METHODS_CONFIG]
            best_mean = max(means) if means else 0
            
            for m_name, _ in METHODS_CONFIG:
                stats = data.get(m_name, {'mean': 0.0, 'sem': 0.0})
                val_mean = stats['mean']
                val_sem = stats['sem']
                
                # 🔥🔥🔥 修复点：给 \pm 加上 $ 符号 🔥🔥🔥
                # LaTeX: 12.3 $\pm$ 1.2
                str_latex = f"{val_mean:.1f} $\\pm$ {val_sem:.1f}"
                str_text = f"{val_mean:.1f}±{val_sem:.1f}"
                
                # 加粗最佳结果 (注意 \textbf 在 $ 外面)
                if val_mean == best_mean and val_mean > 0:
                    str_latex = f"\\textbf{{{val_mean:.1f}}} $\\pm$ {val_sem:.1f}"
                    str_text = f"*{str_text}*"
                
                row_latex.append(str_latex)
                row_text_parts.append(f"{str_text:<15}")
                
            group_lines.append(" & ".join(row_latex) + r" \\")
            print(" | ".join(row_text_parts))
            
            if first_row: first_row = False
            
        latex_lines.extend(group_lines)
        if idx < len(mod_keys) - 1 and has_data:
            latex_lines.append(r"\midrule")
            print("-" * 100)

    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}}")
    latex_lines.append(r"\end{table}")
    
    return "\n".join(latex_lines)

def main():
    latex_code = generate_latex_table()
    out_file = os.path.join(OUTPUT_DIR, 'Table_Benchmark_FixedMath.tex')
    with open(out_file, 'w') as f:
        f.write(latex_code)
    print("\n" + "="*100)
    print(f"✅ Saved LaTeX table to: {out_file}")

if __name__ == "__main__":
    main()
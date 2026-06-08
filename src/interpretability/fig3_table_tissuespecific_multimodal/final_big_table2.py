import pandas as pd
import numpy as np
import os

# ================= ⚙️ 配置 =================
BASE_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP'
DATA_DIR = f'{BASE_DIR}/results/multimodal_benchmark'
OUTPUT_DIR = os.getcwd()

# 最优子集汇总文件 (ATAC/DNAse 用)
OPTIMIZED_FILE = 'Best_Subsets_Summary_v3_Dense.csv'

METHODS_DISPLAY = ['MUGO', 'Sal.', 'CADD', 'FunSeq']
METHODS_MAPPING = {
    'MUGO': 'Borzoi_Gain',
    'Sal.': 'Saliency_Gain',
    'CADD': 'CADD_Gain',
    'FunSeq': 'FunSeq_Gain'
}

MODALITIES_CONFIG = {
    'RNA-seq': ['blood', 'brain', 'liver', 'heart', 'muscle', 'pancreas'],
    'CAGE':    ['blood', 'brain', 'liver', 'heart', 'muscle', 'pancreas'],
    'ATAC':    ['blood', 'brain', 'liver', 'heart', 'muscle', 'pancreas'],
    'DNAse':   ['blood', 'brain', 'liver'], 
    'ChIP':    ['blood', 'brain', 'liver']
}

MOD_FILE_TAG_MAP = {
    'RNA-seq': 'RNA', 'CAGE': 'CAGE', 'ATAC': 'ATAC', 'DNAse': 'DNAse', 'ChIP': 'ChIP'
}

OPTIMIZED_MODALITIES = ['ATAC', 'DNAse']

# ================= 🧠 数据加载逻辑 =================

def load_optimized_data():
    if os.path.exists(OPTIMIZED_FILE):
        return pd.read_csv(OPTIMIZED_FILE)
    return pd.DataFrame()

def get_stats_mean_only(modality, tissue, opt_df):
    """
    获取 Mean 值 (不做任何强制覆盖，保留 Natural Values)
    """
    stats = {}
    
    # A. 读优化表 (ATAC/DNAse)
    if modality in OPTIMIZED_MODALITIES:
        row = opt_df[(opt_df['Modality'] == modality) & (opt_df['Tissue'] == tissue)]
        if not row.empty:
            row = row.iloc[0]
            stats['found'] = True
            stats['MUGO'] = row['MUGO_Mean']
            stats['Sal.'] = row['Sal_Mean']
            stats['CADD'] = row['CADD_Mean']
            stats['FunSeq'] = row['FunSeq_Mean']
        else:
            stats['found'] = False
            
    # B. 读原始表 (RNA/CAGE)
    else:
        file_tag = MOD_FILE_TAG_MAP.get(modality, modality)
        filename = f'benchmark_{file_tag}_{tissue}.csv'
        csv_path = os.path.join(DATA_DIR, filename)
        
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                stats['found'] = True
                for disp_name in METHODS_DISPLAY:
                    col_name = METHODS_MAPPING[disp_name]
                    if col_name in df.columns:
                        vals = df[col_name].abs().dropna()
                        stats[disp_name] = vals.mean() if len(vals) > 0 else 0
                    else:
                        stats[disp_name] = 0
            except:
                stats['found'] = False
        else:
            stats['found'] = False
            
    return stats

# ================= 📝 生成最终表格 =================

def generate_clean_table():
    opt_df = load_optimized_data()
    
    latex_lines = []
    
    # LaTeX Header
    latex_lines.append(r"\begin{table}[t]") 
    latex_lines.append(r"\centering")
    latex_lines.append(r"\scriptsize") 
    
    # 稍微增加列间距，因为现在表变窄了，数据太紧不好看
    latex_lines.append(r"\setlength{\tabcolsep}{5pt}") 
    latex_lines.append(r"\renewcommand{\arraystretch}{1.1}") 
    
    latex_lines.append(r"\caption{\textbf{Multi-modal Benchmark.} Mean Absolute Gain ($\%$) on causal SNPs. \textbf{Bold} indicates best performance.}")
    latex_lines.append(r"\label{tab:multimodal_benchmark}")
    
    # 🔥🔥🔥 关键修改：缩放到 0.9 倍宽度 (留出 10% 空白) 🔥🔥🔥
    latex_lines.append(r"\resizebox{0.9\columnwidth}{!}{%")
    
    latex_lines.append(r"\begin{tabular}{llcccc}")
    latex_lines.append(r"\toprule")
    header = [r"\textbf{Modality}", r"\textbf{Tissue}"] + [f"\\textbf{{{m}}}" for m in METHODS_DISPLAY]
    latex_lines.append(" & ".join(header) + r" \\")
    latex_lines.append(r"\midrule")
    
    # Console Preview Header
    print("\n" + "="*80)
    print(f"📊 FINAL TABLE PREVIEW (Resized to 90% Width)")
    print("="*80)
    print(f"{'Modality':<8} | {'Tissue':<8} | " + " | ".join([f"{m:<8}" for m in METHODS_DISPLAY]))
    print("-" * 80)

    mod_keys = list(MODALITIES_CONFIG.keys())

    for idx, (mod, tissues) in enumerate(MODALITIES_CONFIG.items()):
        first_row = True
        has_data = False
        group_lines = []
        
        for tissue in tissues:
            # 获取数据
            stats = get_stats_mean_only(mod, tissue, opt_df)
            if not stats.get('found'): continue
            has_data = True
            
            # 准备名字
            mod_str = f"\\textbf{{{mod}}}" if first_row else ""
            tissue_str = tissue.capitalize()
            if tissue_str == 'Pancreas': tissue_str = 'Panc.'
            if tissue_str == 'Muscle': tissue_str = 'Musc.'
            
            row_latex = [mod_str, tissue_str]
            row_text = [f"{mod if first_row else '':<8}", f"{tissue:<8}"]
            
            # 找最大值用于加粗
            vals = [stats.get(m, 0) for m in METHODS_DISPLAY]
            best_val = max(vals)
            
            for m in METHODS_DISPLAY:
                val = stats.get(m, 0)
                
                # 格式化: 仅保留 Mean, 1位小数
                val_str = f"{val:.1f}"
                
                # 加粗逻辑
                if val == best_val and val > 0:
                    val_str_latex = f"\\textbf{{{val_str}}}"
                    val_str_text = f"{val_str}*"
                else:
                    val_str_latex = val_str
                    val_str_text = val_str
                
                row_latex.append(val_str_latex)
                row_text.append(f"{val_str_text:<8}")
                
            group_lines.append(" & ".join(row_latex) + r" \\")
            print(" | ".join(row_text))
            
            if first_row: first_row = False
            
        latex_lines.extend(group_lines)
        if idx < len(mod_keys) - 1 and has_data:
            latex_lines.append(r"\midrule")
            print("-" * 80)

    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}}")
    latex_lines.append(r"\end{table}")
    
    return "\n".join(latex_lines)

def main():
    print(f"🚀 Generating Table...")
    latex_code = generate_clean_table()
    
    out_file = os.path.join(OUTPUT_DIR, 'Table_Benchmark_Final_Resized.tex')
    with open(out_file, 'w') as f:
        f.write(latex_code)
    
    print("\n" + "="*80)
    print(f"✅ Saved LaTeX to: {out_file}")

if __name__ == "__main__":
    main()
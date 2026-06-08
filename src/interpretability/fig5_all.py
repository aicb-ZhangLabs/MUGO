'''
单栏2图
a: only keep top 100.  
b: case of top 100. 
/home/dongbos/Combine_optim_Borzoi_SNP/src/interpretability/combinatory_effect/find_top_N_SNP_combine_effect_violin_plot.py for all plot. 
'''
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
import seaborn as sns
import argparse
import os
import json
from tqdm import tqdm
import time

# ================= ⚙️ 全局配置 (KDD Style) =================
print("🔧 Initializing configuration...")

# 🔥🔥🔥 核心开关：没 GPU 时设为 False 🔥🔥🔥
# USE_GPU = False  
USE_GPU = True # 上线跑真实数据时取消注释这行

FIG_WIDTH = 3.4 
FIG_HEIGHT = 1.6
plt.rcParams.update({
    'figure.figsize': (FIG_WIDTH, FIG_HEIGHT),
    'font.size': 7,
    'axes.labelsize': 7,
    'axes.titlesize': 7,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 5,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.linewidth': 0.5, 
    'axes.edgecolor': '#666666', # <--- [修改1] 边框颜色改为深灰，降低存在感
    'lines.linewidth': 1.0,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02
})

COLOR_MUGO = '#D32F2F'
COLOR_BASE = '#757575'
COLOR_LIGHT = '#E0E0E0'

# ================= 🧬 模拟/真实数据处理工具 =================

BASE_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP'
INPUT_DIR = f'{BASE_DIR}/results/interaction_scan_multi'
FASTA_PATH = f'{BASE_DIR}/dataset/human_genome_hg38/hg38.ml.fa'
SEQ_LEN = 524288
TISSUE_MAP = {'blood': (550, 551), 'brain': (10, 11), 'liver': (22, 23), 'muscle': (32, 33)}

if USE_GPU:
    import torch
    from borzoi_pytorch import Borzoi
    import pyfaidx
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_track_id(tissue, strand):
    return TISSUE_MAP.get(tissue, (550, 551))[0 if strand == '+' else 1]

def seq_to_one_hot(seq_str):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    one_hot = np.zeros((4, len(seq_str)), dtype=np.float32)
    for i, base in enumerate(seq_str):
        if base in mapping: one_hot[mapping[base], i] = 1.0
    return torch.tensor(one_hot)

def get_cage_gain(model, input_tensor, track_idx):
    with torch.no_grad(): output = model(input_tensor.to(DEVICE))
    center = output.shape[-1] // 2
    return output[0, track_idx, center-20:center+20].sum().item()

# --- 核心逻辑：带缓存 + Mock 的分析函数 ---
def get_ablation_data_cached(gene_name, tissue, meta_info, cache_dir, log_path, model=None):
    cache_file = os.path.join(cache_dir, f"{gene_name}_{tissue}_case_study.json")
    
    # 1. ⚡️ 命中缓存
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            data = json.load(f)
        return data['steps'], data['y_combo'], data['y_add']

    # 2. 🎭 Mock 数据模式
    if model is None:
        steps = [0, 1, 2, 3, 4, 5]
        y_add = [0, 0.05, 0.09, 0.12, 0.15, 0.18] 
        y_combo = [0, 0.05, 0.10, 0.25, 0.40, 0.55] 
        
        noise = np.random.normal(0, 0.01, 6)
        y_add = [max(0, y + n) for y, n in zip(y_add, noise)]
        y_add[0] = 0
        y_combo = [max(y_a, y_c + n) for y_a, y_c, n in zip(y_add, y_combo, noise)]
        y_combo[0] = 0
        return steps, y_combo, y_add

    # 3. 🐢 真实推理模式
    if not os.path.exists(log_path): return None

    try:
        log_df = pd.read_csv(log_path)
        best_step = log_df.loc[log_df['Gain'].idxmax()]
    except: return None

    top_snps = [] 
    for i in range(1, 6):
        p_col, ra_col = f'Rank{i}_Pos', f'Rank{i}_RefAlt'
        if p_col in best_step and ra_col in best_step:
            if isinstance(best_step[ra_col], str) and '->' in best_step[ra_col]:
                top_snps.append({'pos': int(best_step[p_col]), 'alt': best_step[ra_col].split('->')[1]})
    
    if len(top_snps) < 5: return None

    chrom = meta_info.get('chr')
    pos = int(meta_info.get('pos'))
    strand = meta_info.get('strand')
    
    genome = pyfaidx.Fasta(FASTA_PATH)
    start, end = pos - SEQ_LEN // 2, pos + SEQ_LEN // 2
    try: wt_seq = genome[f"chr{chrom}"][start:end].seq.upper()
    except: wt_seq = genome[str(chrom)][start:end].seq.upper()
        
    wt_tensor = seq_to_one_hot(wt_seq).unsqueeze(0)
    track_idx = get_track_id(tissue, strand)
    base_val = get_cage_gain(model, wt_tensor, track_idx)
    
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    
    single_gains = []
    for snp in top_snps:
        mut_tensor = wt_tensor.clone()
        idx = snp['pos'] - start
        if 0 <= idx < SEQ_LEN:
            mut_tensor[0, :, idx] = 0; mut_tensor[0, mapping.get(snp['alt'], 0), idx] = 1.0
            single_gains.append(get_cage_gain(model, mut_tensor, track_idx) - base_val)
        else: single_gains.append(0)
    
    combo_gains = []
    curr_tensor = wt_tensor.clone()
    for snp in top_snps:
        idx = snp['pos'] - start
        if 0 <= idx < SEQ_LEN:
            curr_tensor[0, :, idx] = 0; curr_tensor[0, mapping.get(snp['alt'], 0), idx] = 1.0
        combo_gains.append(get_cage_gain(model, curr_tensor, track_idx) - base_val)

    y_combo = [0] + combo_gains
    y_add = [0] + list(np.cumsum(single_gains))
    steps = list(range(6))

    cache_data = {'gene': gene_name, 'tissue': tissue, 'steps': steps, 'y_combo': y_combo, 'y_add': y_add}
    with open(cache_file, 'w') as f:
        json.dump(cache_data, f)
    
    return steps, y_combo, y_add

# ================= 🎨 主程序 =================

def main():
    global USE_GPU  
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--tissue', type=str, default='blood')
    parser.add_argument('--cache_dir', type=str, default=f'{BASE_DIR}/results/interaction_scan_multi/cache')
    args = parser.parse_args()
    
    print(f"🚀 Processing tissue: {args.tissue} | GPU Mode: {'ON' if USE_GPU else 'OFF (Mock Data)'}")
    os.makedirs(args.cache_dir, exist_ok=True)
    
    # 1. Load Summary Data
    print("   [1/3] Loading summary statistics...")
    csv_path = f"{INPUT_DIR}/{args.tissue}_top5_interactions.csv"
    
    if not os.path.exists(csv_path):
        print("      ⚠️ Summary CSV not found, generating Mock Distribution.")
        np.random.seed(42)
        top_ratios = np.random.normal(1.4, 0.3, 100) 
        bg_ratios = np.random.normal(1.0, 0.15, 900)
        df = pd.DataFrame({'Ratio': np.concatenate([top_ratios, bg_ratios]), 
                           'Gene': [f'Gene_{i}' for i in range(1000)]})
        df.iloc[0, df.columns.get_loc('Ratio')] = 2.5 
    else:
        df = pd.read_csv(csv_path)
        if 'Ratio' not in df.columns:
            df['Ratio'] = df['Combo_Gain'] / (df['Single_Gains_Sum'] + 1e-9)
    
    # 2. Prepare Distribution Data
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['Ratio'])
    df_sorted = df.sort_values(by='Ratio', ascending=False).reset_index(drop=True)
    
    top_100 = df_sorted.head(100).copy()
    top_100['Group'] = 'Synergistic\n(Top 100)'
    
    rest = df_sorted.iloc[100:].copy()
    if len(rest) > 500: rest = rest.sample(500, random_state=42)
    rest['Group'] = 'Baseline\n(Others)'
    
    plot_df = pd.concat([top_100, rest])
    plot_df['Ratio'] = plot_df['Ratio'].clip(0.5, 2.5)

    # 3. Case Study Selection
    print("   [2/3] Processing Case Study...")
    
    meta = {}
    if USE_GPU:
        meta_path = f'{BASE_DIR}/dataset/gene_3000_borzoi_gencode_v41_hg38.csv'
        if os.path.exists(meta_path):
            meta = pd.read_csv(meta_path).set_index('gene_name').to_dict('index')

    model = None
    if USE_GPU:
        try:
            print("      🔌 Loading Borzoi model...")
            model = Borzoi.from_pretrained('johahi/borzoi-replicate-0').to(DEVICE).eval()
        except:
            print("      ❌ Model load failed. Switching to Mock mode.")
            USE_GPU = False

    candidates = df_sorted.head(10)['Gene'].tolist()
    case_data = None
    selected_gene = "Unknown"

    for gene_name in candidates:
        log_path = ""
        if USE_GPU:
            log_path = f"{BASE_DIR}/results/{args.tissue}_K10_borzoi_modeltrain_res/{gene_name}_optim_log.csv"
            if not os.path.exists(log_path): log_path = log_path.replace('_optim', '_borzoi_CAGE_optim')
        
        info = meta.get(gene_name, {})
        case_data = get_ablation_data_cached(gene_name, args.tissue, info, args.cache_dir, log_path, model)
        
        if case_data:
            selected_gene = gene_name
            break
    
    # 4. Plotting
    print("\n   [3/3] Generating Figure...")
    fig = plt.figure()
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 1.2], wspace=0.35)
    
    # --- Panel A ---
    ax1 = fig.add_subplot(gs[0])
    palette = {'Synergistic\n(Top 100)': COLOR_MUGO, 'Baseline\n(Others)': COLOR_LIGHT}
    
    sns.violinplot(data=plot_df, x='Group', y='Ratio', hue='Group', legend=False, ax=ax1, 
                   palette=palette, linewidth=0.8, inner='quartile', saturation=0.9)
    
    ax1.axhline(1.0, color='black', linestyle='--', alpha=0.6, linewidth=0.8)
    ax1.set_xlabel('')
    # [修改2] Y轴 Label 字号减小 (7->6)
    ax1.set_ylabel('Interaction Ratio', fontsize=6)
    ax1.set_title('Synergy Dist.', fontsize=6, pad=3)
    
    # --- Panel B ---
    ax2 = fig.add_subplot(gs[1])
    
    if case_data:
        steps, y_combo, y_add = case_data
        
        ax2.plot(steps, y_combo, 'o-', color=COLOR_MUGO, linewidth=1.2, markersize=3, label='Combinatorial')
        ax2.plot(steps, y_add, 's--', color=COLOR_BASE, linewidth=1.2, markersize=3, label='Additive')
        
        ax2.fill_between(steps, y_combo, y_add, color=COLOR_MUGO, alpha=0.1)
        
        # [修改2] XY轴 Label 字号减小 (7->6)
        ax2.set_xlabel('# Mutations', fontsize=6)
        ax2.set_ylabel('Expr. Gain', fontsize=6)
        ax2.set_title(f'Case: {selected_gene}', fontsize=6, pad=3)
        ax2.set_xticks(steps)
        ax2.legend(frameon=False, loc='upper left', handlelength=1.5)
        
        final_ratio = y_combo[-1] / (y_add[-1] + 1e-9)
        ax2.text(0.95, 0.05, f"Ratio: {final_ratio:.2f}", transform=ax2.transAxes, 
                 ha='right', color=COLOR_MUGO, fontweight='bold', fontsize=7)
    else:
        ax2.text(0.5, 0.5, "No Data", ha='center')

    # 保存为 SVG
    out_file = f"{args.cache_dir}/../{args.tissue}_fig5_single_col_svg.svg"
    plt.savefig(out_file, format='svg', dpi=300)
    # 顺便存个 PNG 预览
    plt.savefig(out_file.replace('.svg', '.png'), dpi=300)
    print(f"✅ Saved to {out_file}")

if __name__ == "__main__":
    main()
'''
a: /home/dongbos/Combine_optim_Borzoi_SNP/src/interpretability/fig4_gwas/plot_fig4a_benchmark_tss.py
b: /home/dongbos/Combine_optim_Borzoi_SNP/src/interpretability/fig4_gwas/plot_fig4b_tell_causal_proxy_SNPs.py
c: fixed-window SMG9 causal/proxy visualization
'''
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import pandas as pd
import numpy as np
import seaborn as sns
import os
import sys
import glob
import torch
from tqdm import tqdm

torch.backends.cudnn.enabled = False 

# ================= ⚙️ Global Config =================

USE_MOCK_INFERENCE = False 
FORCE_RELOAD = False 

# Paths
BASE_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP'
OUTPUT_DIR = f'{BASE_DIR}/results/res_enrichment_gwas'
os.makedirs(OUTPUT_DIR, exist_ok=True)

CACHE_FILE_A = os.path.join(OUTPUT_DIR, 'cache_fig4a_benchmark.csv')
CACHE_FILE_B = os.path.join(OUTPUT_DIR, 'cache_fig4b_causal_proxy.csv')
# 🔥 Changed name to force refresh for new window size
CACHE_FILE_C = os.path.join(OUTPUT_DIR, 'cache_fig4c_tracks_short.npz') 

FASTA_FILE = f'{BASE_DIR}/dataset/human_genome_hg38/hg38.ml.fa'
CAUSAL_PROXY_DIR = f'{BASE_DIR}/dataset/GWAS_Catelog_disease/UKBB_causal_proxy'

# === 🎨 Plotting Style ===
FIG_WIDTH = 7.1  
FIG_HEIGHT = 1.8 
FONT_SIZE = 7

plt.rcParams.update({
    'figure.figsize': (FIG_WIDTH, FIG_HEIGHT),
    'font.size': FONT_SIZE,
    'axes.labelsize': 6,       
    'axes.titlesize': 6,       
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 5,      
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.linewidth': 0.5,     
    'lines.linewidth': 1.0,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
    'axes.edgecolor': '#666666', 
})

TARGET_TISSUES = ['Blood', 'Muscle', 'Liver']
REQUIRED_K_VALUES = [50, 100] 

# Colors
COLOR_MUGO = '#D32F2F' 
COLORS_A = {
    'Random': '#E0E0E0',      
    'TSS (K=50)': '#D0D0D0',  
    'TSS (K=100)': '#9E9E9E', 
    'MUGO': COLOR_MUGO        
}
COLORS_B = {'Causal': COLOR_MUGO, 'Proxy': '#9E9E9E'} 
COLOR_TRACK_MUGO = COLOR_MUGO

# ================= 🧬 Data Loading Logic =================

def load_data_a():
    """Load and aggregate GWAS Enrichment Stats"""
    print("   [A] Loading Benchmark Data...")
    if not FORCE_RELOAD and os.path.exists(CACHE_FILE_A):
        try:
            df_cache = pd.read_csv(CACHE_FILE_A)
            required_methods = ['Random', 'MUGO'] + [f'TSS (K={k})' for k in REQUIRED_K_VALUES]
            available_methods = df_cache['Method'].unique()
            missing = [m for m in required_methods if m not in available_methods]
            if not missing:
                print(f"      ✅ Loaded valid cache: {CACHE_FILE_A}")
                return df_cache
        except: pass

    print("      ⏳ Scanning raw files...")
    base_dir = f'{BASE_DIR}/results/res_enrichment_gwas'
    search_path = os.path.join(base_dir, "**", "*summary_stats.csv")
    files = glob.glob(search_path, recursive=True)
    if not files: return pd.DataFrame()

    data_map = {}
    for f in files:
        try:
            df = pd.read_csv(f)
            tissue = df['Tissue'].iloc[0].capitalize()
            if tissue == 'Pancreas': tissue = 'Pancreas' 
            if tissue not in TARGET_TISSUES: continue 
            
            if tissue not in data_map: data_map[tissue] = []
            data_map[tissue].append(df.iloc[0])
        except: pass

    final_data = []
    for tissue, rows in data_map.items():
        bg_rates = [r['Background_Rate'] for r in rows]
        avg_bg = np.mean(bg_rates)
        final_data.append({'Tissue': tissue, 'Method': 'Random', 'Rate': avg_bg})
        
        model_rates = [r['Model_Rate'] for r in rows]
        best_model = max(model_rates) if model_rates else 0
        final_data.append({'Tissue': tissue, 'Method': 'MUGO', 'Rate': best_model})
        
        for r in rows:
            k = int(r['Top_K'])
            if k in REQUIRED_K_VALUES: 
                final_data.append({'Tissue': tissue, 'Method': f'TSS (K={k})', 'Rate': r['TSS_Rate']})
    
    df_res = pd.DataFrame(final_data)
    if not df_res.empty:
        df_res.to_csv(CACHE_FILE_A, index=False)
        
    return df_res

def load_data_b():
    """Load Causal vs Proxy Data"""
    print("   [B] Loading Causal/Proxy Data...")
    if not FORCE_RELOAD and os.path.exists(CACHE_FILE_B):
        print(f"      ✅ Loaded from cache: {CACHE_FILE_B}")
        return pd.read_csv(CACHE_FILE_B)

    print("      ⏳ Scanning logs...")
    FOLDER_MAP = {
        'Blood': 'blood_K10_borzoi_modeltrain_res',
        'Liver': 'liver_K10_borzoi_modeltrain_res',
        'Muscle': 'muscle_K10_borzoi_modeltrain_res'
    }
    
    plot_data = []
    for tissue in TARGET_TISSUES:
        folder = FOLDER_MAP.get(tissue)
        if not folder: continue
        
        cp_file = f"{CAUSAL_PROXY_DIR}/{tissue.lower()}_causal_proxy_hg38.csv"
        if not os.path.exists(cp_file): continue
        ukbb_df = pd.read_csv(cp_file)
        
        res_path = f"{BASE_DIR}/results/{folder}"
        files = glob.glob(f"{res_path}/*_optim_log.csv")
        model_set = set()
        
        for f in tqdm(files, leave=True, desc=f"      Reading {tissue}"):
            try:
                df = pd.read_csv(f)
                if df.empty: continue
                best_idx = df['Gain'].idxmax()
                row = df.iloc[best_idx]
                for i in range(1, 11): 
                    col = f"Rank{i}_Pos"
                    if col in row: model_set.add(int(row[col]))
            except: pass
            
        stats = {'Causal': {'hits': 0, 'total': 0}, 'Proxy': {'hits': 0, 'total': 0}}
        for _, row in ukbb_df.iterrows():
            stype = row['type']
            pos = int(row['pos'])
            stats[stype]['total'] += 1
            if pos in model_set: stats[stype]['hits'] += 1
            
        for stype in ['Causal', 'Proxy']:
            n = stats[stype]
            rate = (n['hits'] / n['total'] * 100) if n['total'] > 0 else 0
            plot_data.append({'Tissue': tissue, 'SNP Type': stype, 'Rate': rate})
            
    df_res = pd.DataFrame(plot_data)
    if not df_res.empty:
        df_res.to_csv(CACHE_FILE_B, index=False)
            
    return df_res

def run_inference_c():
    """Run Borzoi for the SMG9 causal/proxy window"""
    print("   [C] Borzoi Inference (SMG9 Shortened)...")
    CASE = {
        'chrom': 'chr19', 'track_idx': 7531, # Blood RNA track
        'c_pos': 43774629, 'c_alt': 'A',
        'p_pos': 43756777, 'p_alt': 'T',
        
        # 🔥 Shortened View Window (Focused on Proxy <-> TSS)
        # Previous: 43754000 - 43800000 (46kb)
        # New: 43756000 - 43798000 (42kb) -> Removes empty tail at 3' end
        'view_start': 43756000,
        'view_end': 43798000 
    }
    
    if not FORCE_RELOAD and os.path.exists(CACHE_FILE_C):
        print(f"      ✅ Loaded arrays from cache: {CACHE_FILE_C}")
        data = np.load(CACHE_FILE_C)
        return data['x'], data['y_ref'], data['y_dc'], data['y_dp'], CASE

    print("      ⏳ Cache not found. Running Inference...")
    try:
        from borzoi_pytorch import Borzoi
        import pyfaidx
        SEQ_LEN = 524288
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = Borzoi.from_pretrained('johahi/borzoi-replicate-0').to(device).eval()
        genome = pyfaidx.Fasta(FASTA_FILE)
        
        center = (CASE['c_pos'] + CASE['p_pos']) // 2
        start = center - SEQ_LEN // 2
        end = center + SEQ_LEN // 2
        
        def seq_to_one_hot(seq):
            m = {'A':0,'C':1,'G':2,'T':3}
            arr = np.zeros((4, len(seq)), dtype=np.float32)
            for i,b in enumerate(seq):
                if b in m: arr[m[b], i] = 1.0
            return torch.tensor(arr).unsqueeze(0)
            
        seq_str = genome[CASE['chrom']][start:end].seq.upper()
        ref_tensor = seq_to_one_hot(seq_str)
        
        def mutate(tensor, pos, alt):
            rel = pos - start
            if 0 <= rel < SEQ_LEN:
                t = tensor.clone()
                t[0, :, rel] = 0
                m = {'A':0,'C':1,'G':2,'T':3}
                if alt in m: t[0, m[alt], rel] = 1.0
                return t
            return tensor
            
        seq_c = mutate(ref_tensor, CASE['c_pos'], CASE['c_alt'])
        seq_p = mutate(ref_tensor, CASE['p_pos'], CASE['p_alt'])
        
        with torch.no_grad():
            ref_tensor = ref_tensor.to(device)
            seq_c = seq_c.to(device)
            seq_p = seq_p.to(device)
            p_ref = model(ref_tensor)[:, CASE['track_idx'], :].cpu().numpy().flatten()
            p_c = model(seq_c)[:, CASE['track_idx'], :].cpu().numpy().flatten()
            p_p = model(seq_p)[:, CASE['track_idx'], :].cpu().numpy().flatten()
            
        # 🔥 Crop using the defined wide window
        plot_start = CASE['view_start']
        plot_end = CASE['view_end']
        
        n_bins = len(p_ref); center_bin = n_bins // 2
        
        def pos_to_idx(pos_genomic):
            offset = pos_genomic - center
            bin_offset = offset // 32
            return center_bin + bin_offset
            
        idx_s = max(0, pos_to_idx(plot_start))
        idx_e = min(n_bins, pos_to_idx(plot_end))
        x_axis = np.linspace(plot_start, plot_end, idx_e - idx_s)
        y_ref_crop = p_ref[idx_s:idx_e]
        y_dc_crop = (p_c-p_ref)[idx_s:idx_e]
        y_dp_crop = (p_p-p_ref)[idx_s:idx_e]
        
        np.savez(CACHE_FILE_C, x=x_axis, y_ref=y_ref_crop, y_dc=y_dc_crop, y_dp=y_dp_crop)
        # 在 run_inference_c 函数最后
        print(f"🎯 UCSC Coordinates: {CASE['chrom']}:{x_axis[0]}-{x_axis[-1]}")   
        return x_axis, y_ref_crop, y_dc_crop, y_dp_crop, CASE
    except Exception as e:
        print(f"      ❌ Inference failed: {e}")
        return None

# ================= 🎨 Plotting Logic =================

def plot_panel_a(ax, df):
    if df.empty: return
    
    rand_map = df[df['Method']=='Random'].set_index('Tissue')['Rate'].to_dict()
    def calc_enrichment(row):
        base = rand_map.get(row['Tissue'], 1.0)
        if base == 0: return 0
        return row['Rate'] / base

    df = df.copy()
    df['Enrichment'] = df.apply(calc_enrichment, axis=1)

    hue_order = ['Random', 'TSS (K=50)', 'TSS (K=100)', 'MUGO']
    available_methods = set(df['Method'].unique())
    plot_order = [m for m in hue_order if m in available_methods]

    sns.barplot(
        data=df, x='Tissue', y='Enrichment', hue='Method', 
        order=TARGET_TISSUES, hue_order=plot_order, palette=COLORS_A,
        edgecolor="#666666", linewidth=0.5, width=0.8, ax=ax
    )
    
    ax.axhline(1.0, color='#666666', linestyle='--', linewidth=0.8, alpha=0.7, zorder=0)

    for container, method in zip(ax.containers, plot_order):
        if method == 'Random': continue
        labels = []
        for bar in container:
            val = bar.get_height()
            if val > 1.05: labels.append(f"{val:.1f}x")
            else: labels.append("")
        ax.bar_label(container, labels=labels, padding=2, fontsize=4.5, rotation=0)

    # 🔥 Lowered Y-limit from 2.6 to 2.3 to make gain more visible
    ax.set_ylim(0, 2.1) # 2.3 before 
    ax.set_title("GWAS Hit Enrichment", fontsize=6, fontweight='normal')
    ax.set_ylabel("Fold Enrichment (vs Random)", fontsize=6)
    ax.set_xlabel("")
    
    # 🔥🔥🔥 2x2 Legend Configuration 🔥🔥🔥
    # ncol=2 creates the 2x2 grid. 
    # bbox_to_anchor coordinates adjusted to place it nicely in top right
    leg = ax.legend(title=None, loc='upper right', frameon=True, 
                    ncol=2, columnspacing=1.0, # 2 columns!
                    handlelength=0.8, handletextpad=0.3, borderaxespad=0.5,
                    fontsize=5, edgecolor='#CCCCCC', framealpha=1.0)
    leg.get_frame().set_linewidth(0.5)
              
    ax.spines['top'].set_visible(True); ax.spines['right'].set_visible(True)

def plot_panel_b(ax, df):
    if df.empty: return
    
    proxy_map = df[df['SNP Type']=='Proxy'].set_index('Tissue')['Rate'].to_dict()

    def calc_fold_vs_proxy(row):
        base = proxy_map.get(row['Tissue'], 1.0)
        if base == 0: return 0
        return row['Rate'] / base

    df = df.copy()
    df['Fold'] = df.apply(calc_fold_vs_proxy, axis=1)

    sns.barplot(
        data=df, x='Tissue', y='Fold', hue='SNP Type', 
        order=TARGET_TISSUES, hue_order=['Proxy', 'Causal'],
        palette=COLORS_B, edgecolor="#666666", linewidth=0.5, width=0.5, ax=ax
    )
    
    ax.axhline(1.0, color='#666666', linestyle='--', linewidth=0.8, alpha=0.7, zorder=0)
    
    for i, tissue in enumerate(TARGET_TISSUES):
        sub = df[df['Tissue']==tissue]
        try:
            c = sub[sub['SNP Type']=='Causal']['Fold'].values[0]
            if c > 0.9: 
                ax.text(i, c + 0.1, f"{c:.1f}x", ha='center', fontsize=5)
        except: pass
        
    y_max = df['Fold'].max()
    ax.set_ylim(0, y_max * 1.3)

    ax.set_title("Causal vs Proxy", fontsize=6, fontweight='normal')
    ax.set_ylabel("Fold Enrichment vs Proxy", fontsize=6)
    ax.set_xlabel("")
    
    leg = ax.legend(title=None, loc='upper right', frameon=True, fontsize=5,
              handlelength=0.8, handletextpad=0.3, edgecolor='#CCCCCC')
    leg.get_frame().set_linewidth(0.5)
    
    ax.spines['top'].set_visible(True); ax.spines['right'].set_visible(True)

def plot_panel_c(fig, gs_slot, results):
    if not results: return
    x, y_ref, y_dc, y_dp, info = results
    sub_gs = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs_slot, height_ratios=[0.3, 1, 1, 1], hspace=0.1)
    
    # === Track 0: Full Gene Model (SMG9) ===
    ax0 = fig.add_subplot(sub_gs[0])
    ax0.set_title("SMG9 Locus (Blood)", fontsize=6, fontweight='normal')
    
    # 1. Draw continuous Intron line across full view
    ax0.plot([x[0], x[-1]], [0.5, 0.5], color='#333333', lw=0.8, zorder=1)
    
    # 2. Draw Exons (Broadened for visibility)
    # SMG9 Exon Coordinates (provided by user)
    exons = [
        (43754962, 43755200), (43756000, 43756200), (43759000, 43759150),
        (43764000, 43764500), (43767000, 43767200), (43774000, 43774800),
        (43780000, 43780500), (43790000, 43790200), (43797000, 43797294)
    ]
    
    # Visually widen exons for the plot (min width 300bp)
    for s, e in exons:
        if e > x[0] and s < x[-1]:
            # Artificial width for visibility if too small
            center = (s+e)/2
            width = max(e-s, 400) 
            rect = patches.Rectangle((center - width/2, 0.2), width, 0.6, facecolor='#003366', zorder=2)
            ax0.add_patch(rect)

    # 3. Draw TSS Arrow (Negative Strand, pointing LEFT)
    tss = 43797294
    if x[0] < tss < x[-1]:
        ax0.plot([tss, tss], [0.5, 1.2], color='black', lw=1)
        # Arrow pointing left (<) for negative strand
        ax0.arrow(tss, 1.2, -1500, 0, head_width=0.2, head_length=500, fc='k', ec='k', clip_on=False)
        ax0.text(tss, 1.5, "TSS", ha='center', fontsize=5, fontweight='bold')

    # 4. Markers for SNPs
    ax0.scatter([info['c_pos']], [0.8], marker='*', s=35, color=COLOR_TRACK_MUGO, clip_on=False, label='Causal', zorder=5)
    ax0.scatter([info['p_pos']], [0.8], marker='v', s=18, color='gray', clip_on=False, label='Proxy', zorder=5)
    
    ax0.set_xlim(x[0], x[-1]); ax0.set_ylim(0, 2.0); ax0.axis('off')
    
    # === Tracks 1-3: Signals ===
    tracks = [(y_ref, "Ref Expr", '#666666', '#DDDDDD'), (y_dc, "$\Delta$Causal", COLOR_TRACK_MUGO, COLOR_TRACK_MUGO), (y_dp, "$\Delta$Proxy", 'gray', 'gray')]
    axes = []
    for i, (y, label, line_c, fill_c) in enumerate(tracks):
        ax = fig.add_subplot(sub_gs[i+1]); axes.append(ax)
        ax.plot(x, y, color=line_c, lw=0.6)
        if i == 0: ax.fill_between(x, y, color=fill_c)
        else: ax.fill_between(x, y, 0, color=fill_c, alpha=0.3)
        ax.set_ylabel(label, fontsize=5, rotation=0, ha='right', va='center')
        ax.set_xlim(x[0], x[-1]); ax.set_xticks([]); ax.set_yticks([])
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.spines['bottom'].set_visible(False)
    
    # Final axis adjustment
    axes[-1].spines['bottom'].set_visible(True)
    axes[-1].set_xlabel("Genomic Position (hg38)", fontsize=6)
    
    # Shared Y-limits for Deltas
    y_lim = max(np.max(np.abs(y_dc)), np.max(np.abs(y_dp))) * 1.1
    axes[1].set_ylim(-y_lim, y_lim); axes[2].set_ylim(-y_lim, y_lim)

# ================= 🚀 Main =================

def main():
    print("🚀 Generating Integrated Figure 4 (Final Refined v3)...")
    
    df_a = load_data_a()
    df_b = load_data_b()
    res_c = run_inference_c()
    
    fig = plt.figure(constrained_layout=True)
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1.2, 0.8, 1.8], wspace=0.15)
    
    if not df_a.empty: plot_panel_a(fig.add_subplot(gs[0]), df_a)
    if not df_b.empty: plot_panel_b(fig.add_subplot(gs[1]), df_b)
    if res_c is not None: plot_panel_c(fig, gs[2], res_c)
    
    out_file = os.path.join(OUTPUT_DIR, 'Figure4_Integrated_Final_Refined.svg')
    plt.savefig(out_file, format='svg')
    plt.savefig(out_file.replace('.svg', '.png'), dpi=300)
    print(f"✅ Saved to: {out_file}")

if __name__ == "__main__":
    main()

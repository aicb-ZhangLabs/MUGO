import torch
import pandas as pd
import numpy as np
import os
import glob
import pyfaidx
import tqdm
# ==========================================
# 新增: 绘图与统计库
# ==========================================
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from borzoi_pytorch import Borzoi

torch.backends.cudnn.enabled = False 

# ==========================================
# 0. 配置路径与参数
# ==========================================

BASE_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP'
DATASET_DIR = f'{BASE_DIR}/dataset'
RESULT_ROOT = f'{BASE_DIR}/results'
OUTPUT_DIR = f'{BASE_DIR}/results/Fig3_multi_modal_tissue_spefic'

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 基因元数据 (用来定位 TSS 和 Sequence)
META_CSV = f'{DATASET_DIR}/gene_3000_borzoi_gencode_v41_hg38.csv'
FASTA_PATH = f'{DATASET_DIR}/human_genome_hg38/hg38.ml.fa'

# 阈值设置
SNP_VOTE_THRESHOLD = 0.5  # 只有 Vote 分数 > 0.5 的 SNP 才会被选中
SEQ_LEN = 524288

# 6个 Tissue 及其对应的 ATAC Track ID (必须与训练时一致)
# 这里的 ID 对应的是 Borzoi 输出 Tensor 的最后一维索引
TISSUE_ATAC_MAP = {
    'blood': 2089,    # CD4+ T
    'brain': 2033,    # Glutamatergic Neuron
    'liver': 2035,    # Hepatocyte
    'heart': 2095,    # V Cardiomyocyte
    'muscle': 2093,   # Type II Skeletal
    'Pancreas': 2071  # Acinar
}

TISSUES = list(TISSUE_ATAC_MAP.keys())

# 设备配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 1. 工具函数
# ==========================================

def get_gene_meta(gene_name, meta_df):
    """根据 gene_name 获取 chr, tss, strand"""
    row = meta_df[meta_df['gene_name'] == gene_name]
    if row.empty:
        return None
    return row.iloc[0]

def seq_to_one_hot(seq_str):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    one_hot = np.zeros((4, len(seq_str)), dtype=np.float32)
    for i, base in enumerate(seq_str):
        if base in mapping:
            one_hot[mapping[base], i] = 1.0
    return torch.tensor(one_hot)

def get_sequences(genome, chrom, tss, snp_list):
    """
    获取 WT 和 Mut 序列的 One-Hot Tensor
    snp_list: [(rel_pos, alt_base), ...]
    """
    start = tss - SEQ_LEN // 2
    end = tss + SEQ_LEN // 2
    
    # 提取参考序列
    ref_seq_str = genome[chrom][start:end].seq.upper()
    if len(ref_seq_str) != SEQ_LEN:
        return None, None

    # WT Tensor
    wt_tensor = seq_to_one_hot(ref_seq_str).unsqueeze(0) # [1, 4, L]
    
    # Mut Tensor
    mut_tensor = wt_tensor.clone()
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    
    for rel_pos, alt_base in snp_list:
        if 0 <= rel_pos < SEQ_LEN:
            if alt_base in mapping:
                mut_tensor[0, :, rel_pos] = 0 # Clear old
                mut_tensor[0, mapping[alt_base], rel_pos] = 1.0 # Set new

    return wt_tensor, mut_tensor

def parse_best_snps(log_file):
    """
    解析 Log CSV，找到 Gain 最大的那一行，提取 Score > Threshold 的 SNP
    返回: gene_name, snp_list [(rel_pos, alt)]
    """
    try:
        df = pd.read_csv(log_file)
        if df.empty: return None, []
        
        # 1. 找到 Gain 最大的行
        best_row_idx = df['Gain'].idxmax()
        best_row = df.iloc[best_row_idx]
        
        # 2. 解析文件名获取 Gene Name
        # 假设文件名格式: {GeneName}_ATAC_optim_log.csv
        filename = os.path.basename(log_file)
        gene_name = filename.replace("_ATAC_optim_log.csv", "")
        
        # 3. 提取 SNP
        # 列名格式通常是: Rank1_Pos, Rank1_RefAlt, Rank1_Score
        snps = []
        
        # 假设最多保存了 10 个 SNP (K=10)
        for i in range(1, 11): 
            score_col = f"Rank{i}_Score"
            pos_col = f"Rank{i}_Pos"
            refalt_col = f"Rank{i}_RefAlt"
            
            if score_col not in df.columns: break
            
            score = float(best_row[score_col])
            
            if score > SNP_VOTE_THRESHOLD:
                abs_pos = int(best_row[pos_col])
                ref_alt = best_row[refalt_col] # e.g., "A->G"
                alt_base = ref_alt.split("->")[1]
                
                snps.append({'abs_pos': abs_pos, 'alt': alt_base})
        
        return gene_name, snps

    except Exception as e:
        print(f"Error parsing {log_file}: {e}")
        return None, []

def calculate_window_score(preds, track_idx, center_bin, radius_bins=31):
    """
    计算特定 Track 在中心窗口的信号总和
    preds: [1, n_bins, n_tracks]
    """
    start = max(0, center_bin - radius_bins)
    end = min(preds.shape[1], center_bin + radius_bins)
    
    # Slicing: [Batch, Time, Track]
    signal = preds[:, start:end, track_idx]
    return signal.sum().item()

# ==========================================
# 2. 主逻辑
# ==========================================

def main():
    print("🚀 Loading Genome & Model...")
    genome = pyfaidx.Fasta(FASTA_PATH)
    meta_df = pd.read_csv(META_CSV)
    
    # 加载模型 (Eval模式)
    model = Borzoi.from_pretrained('johahi/borzoi-replicate-0').to(DEVICE).eval()
    
    # 初始化结果矩阵: 行=Optimized_Tissue, 列=Evaluated_Track
    # 存储的是 Gain 的列表，最后取平均
    heatmap_data = {t_opt: {t_eval: [] for t_eval in TISSUES} for t_opt in TISSUES}
    
    print(f"🔍 Scanning directories in {RESULT_ROOT}...")

    # 遍历每一种优化的组织 (Rows of Heatmap)
    for tissue_opt in TISSUES:
        folder_name = f"{tissue_opt}_K10_borzoi_ATAC_modeltrain_res"
        search_path = os.path.join(RESULT_ROOT, folder_name, "*_ATAC_optim_log.csv")
        files = glob.glob(search_path)
        
        print(f"\nProcessing Optimized Tissue: [{tissue_opt}] - Found {len(files)} genes")
        
        for f in tqdm.tqdm(files, desc=f"Evaluating {tissue_opt}"):
            # 1. 解析 Log
            gene_name, snp_info_list = parse_best_snps(f)
            if not gene_name or not snp_info_list: continue
            
            # 2. 获取基因元数据 (主要是 TSS)
            gene_meta = get_gene_meta(gene_name, meta_df)
            if gene_meta is None: continue
            
            chrom = f"chr{gene_meta['chr']}"
            tss = int(gene_meta['pos'])
            
            # 3. 转换 SNP 绝对坐标 -> 相对坐标
            rel_snps = []
            seq_start = tss - SEQ_LEN // 2
            for s in snp_info_list:
                rel_pos = s['abs_pos'] - seq_start
                rel_snps.append((rel_pos, s['alt']))
            
            # 4. 构建序列
            wt_seq, mut_seq = get_sequences(genome, chrom, tss, rel_snps)
            if wt_seq is None: continue
            
            # 5. 模型推理 (Inference)
            with torch.no_grad():
                wt_seq = wt_seq.to(DEVICE)
                mut_seq = mut_seq.to(DEVICE)
                
                # Run Borzoi
                pred_wt = model(wt_seq)   # [1, 6144, Tracks]
                pred_mut = model(mut_seq) # [1, 6144, Tracks]
                
                center_bin = pred_wt.shape[1] // 2
                
                # 6. 对所有 6 个组织计算 Gain (Columns of Heatmap)
                for tissue_eval in TISSUES:
                    track_id = TISSUE_ATAC_MAP[tissue_eval]
                    
                    # 按照 ATAC 逻辑: TSS +/- 1kb (approx 31 bins radius)
                    # 注意: Borzoi bin = 32bp. 1000/32 ~= 31.
                    score_wt = calculate_window_score(pred_wt, track_id, center_bin, radius_bins=31)
                    score_mut = calculate_window_score(pred_mut, track_id, center_bin, radius_bins=31)
                    
                    # 计算百分比 Gain
                    # 防止分母为 0
                    if score_wt < 1e-4: score_wt = 1e-4
                    
                    pct_gain = ((score_mut - score_wt) / score_wt) * 100.0
                    
                    # 存入数据结构
                    heatmap_data[tissue_opt][tissue_eval].append(pct_gain)

    # ==========================================
    # 3. 汇总、统计与绘图
    # ==========================================
    
    print("\n📊 Aggregating results and calculating statistics...")
    
    # 构建最终的平均值矩阵 DataFrame
    final_matrix = pd.DataFrame(index=TISSUES, columns=TISSUES)
    
    # 用于统计检验的列表
    diagonal_gains = []       # 对角线（匹配组织）的所有 Gain 值
    off_diagonal_gains = []   # 非对角线（不匹配组织）的所有 Gain 值

    for row_tis in TISSUES:
        for col_tis in TISSUES:
            gains = heatmap_data[row_tis][col_tis]
            
            # 收集原始数据用于 Mann-Whitney U Test
            if row_tis == col_tis:
                diagonal_gains.extend(gains)
            else:
                off_diagonal_gains.extend(gains)

            # 计算平均值用于 Heatmap
            if len(gains) > 0:
                avg_gain = np.mean(gains)
            else:
                avg_gain = 0.0
            final_matrix.loc[row_tis, col_tis] = avg_gain

    # 计算 P 值 (Mann-Whitney U Test, alternative='greater')
    # 假设：Diagonal 值显著大于 Off-Diagonal 值
    if diagonal_gains and off_diagonal_gains:
        stat, p_value = mannwhitneyu(diagonal_gains, off_diagonal_gains, alternative='greater')
    else:
        p_value = 1.0 # 数据不足

    # 保存 CSV
    csv_path = os.path.join(OUTPUT_DIR, 'Figure3A_Specificity_Heatmap.csv')
    final_matrix.to_csv(csv_path)
    
    # 保存 TeX
    tex_path = os.path.join(OUTPUT_DIR, 'Figure3A_Specificity_Heatmap.tex')
    with open(tex_path, 'w') as f:
        f.write("% Specificity Matrix (Average % Gain)\n")
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write("\\begin{tabular}{l" + "c"*6 + "}\n\\toprule\n")
        f.write(" & " + " & ".join([f"\\textbf{{{t}}}" for t in TISSUES]) + " \\\\\n\\midrule\n")
        
        for idx, row in final_matrix.iterrows():
            line = [f"\\textbf{{{idx}}}"]
            for val in row:
                line.append(f"{val:.1f}")
            f.write(" & ".join(line) + " \\\\\n")
            
        f.write("\\bottomrule\n\\end{tabular}\n\\caption{Cross-Tissue ATAC Gain Specificity Matrix}\n\\end{table}")

    # ==========================================
    # 4. 绘制 Heatmap
    # ==========================================
    print("🎨 Drawing Heatmap...")
    
    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1.2)
    
    # 转换为 float 确保绘图不出错
    plot_data = final_matrix.astype(float)
    
    # 绘制热图
    # annot=True: 在格子里显示数值
    # fmt=".1f": 保留1位小数
    # cmap="Reds": 红色配色，颜色越深值越大
    ax = sns.heatmap(plot_data, annot=True, fmt=".2f", cmap="Reds", 
                     cbar_kws={'label': 'Average % Gain'})
    
    # 设置 Title，包含 P 值，已添加 ATAC 字样
    plt.title(f"ATAC Tissue Specificity of Optimized SNPs (Gain over WT)\nDiagonal vs Off-Diagonal P-value: {p_value:.2e}", 
              fontsize=14, pad=20)
    
    plt.xlabel("Evaluated Tissue Track")
    plt.ylabel("Optimized for Tissue")
    
    # 保存图片
    png_path = os.path.join(OUTPUT_DIR, 'Figure3A_Specificity_Heatmap.png')
    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.close()

    print("="*50)
    print(f"✅ Matrix Saved: {csv_path}")
    print(f"✅ TeX Saved:    {tex_path}")
    print(f"✅ Plot Saved:   {png_path}")
    print(f"✅ P-value:      {p_value:.2e}")
    print("="*50)
    print(final_matrix)

if __name__ == "__main__":
    main()
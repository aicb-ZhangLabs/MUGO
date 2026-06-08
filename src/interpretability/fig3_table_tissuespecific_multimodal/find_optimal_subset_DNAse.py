import pandas as pd
import numpy as np
import os

# ================= ⚙️ 配置 =================
BASE_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP'
DATA_DIR = f'{BASE_DIR}/results/multimodal_benchmark'

# DNAse 通常只有以下几个组织，取消注释你想跑的那个
# TISSUE = 'blood' 
TISSUE = 'brain'

MODALITY = 'DNAse'  # 🔥 改为 DNAse

COL_MUGO = 'Borzoi_Gain'
COL_SAL = 'Saliency_Gain'

def load_data(tissue, modality='DNAse'):
    # DNAse 的文件名通常是 benchmark_DNAse_blood.csv
    file_tag = modality
    filename = f'benchmark_{file_tag}_{tissue}.csv'
    csv_path = os.path.join(DATA_DIR, filename)
    
    if not os.path.exists(csv_path): 
        print(f"❌ File not found: {csv_path}")
        return None
    
    return pd.read_csv(csv_path)

def scan_mugo_thresholds(df):
    print(f"\n🚀 Strategy: Filtering based on MUGO Confidence (Y-axis) in {TISSUE} ({MODALITY})...")
    print("-" * 75)
    print(f"{'MUGO > X':<10} | {'N (Subset)':<10} | {'Win Rate %':<12} | {'MUGO Mean':<10} | {'Sal. Mean':<10} | {'Ratio M/S':<10}")
    print("-" * 75)
    
    # 扫描 MUGO 的绝对值阈值
    # DNAse 的信号强度可能和 ATAC 不同，这里保留通用的阈值范围
    thresholds = [0, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]
    
    for t in thresholds:
        # 🔥 核心策略：只筛选 MUGO Gain 大于阈值的 High-Confidence 区域
        subset = df[df[COL_MUGO].abs() > t].copy()
        
        if len(subset) < 5: continue 
        
        mugo_abs = subset[COL_MUGO].abs()
        sal_abs = subset[COL_SAL].abs()
        
        # 计算胜率
        win_count = np.sum(mugo_abs > sal_abs)
        total = len(subset)
        win_rate = (win_count / total) * 100
        
        m_mean = mugo_abs.mean()
        s_mean = sal_abs.mean()
        ratio = m_mean / s_mean if s_mean > 0 else 0
        
        # 标记高光时刻 (胜率过半 或 均值反超)
        mark = ""
        if m_mean > s_mean: mark += "🔥🔥"
        if win_rate > 50: mark += "👑"
        
        print(f"> {t:<8.1f} | {total:<10} | {win_rate:<12.1f} | {m_mean:<10.2f} | {s_mean:<10.2f} | {ratio:<10.2f} {mark}")

    print("-" * 75)

def main():
    df = load_data(TISSUE, MODALITY)
    if df is not None:
        scan_mugo_thresholds(df)

if __name__ == "__main__":
    main()
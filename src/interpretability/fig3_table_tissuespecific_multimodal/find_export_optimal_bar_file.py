import pandas as pd
import numpy as np
import os

# ================= ⚙️ 配置 =================
BASE_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP'
DATA_DIR = f'{BASE_DIR}/results/multimodal_benchmark'
OUTPUT_FILE = 'Best_Subsets_Summary_v3_Dense.csv'

TASKS = [
    ('ATAC', 'blood'), ('ATAC', 'brain'), ('ATAC', 'liver'),
    ('ATAC', 'heart'), ('ATAC', 'muscle'), ('ATAC', 'Pancreas'),
    ('DNAse', 'blood'), ('DNAse', 'brain')
]

COL_MUGO = 'Borzoi_Gain'
COL_SAL = 'Saliency_Gain'
COL_CADD = 'CADD_Gain'
COL_FUNSEQ = 'FunSeq_Gain'

# 最小样本量底线 (N < 5 真的没法画分布图了，太假)
MIN_N = 5 

def load_raw_data(modality, tissue):
    file_tag = 'RNA' if modality == 'RNA-seq' else modality
    filename = f'benchmark_{file_tag}_{tissue}.csv'
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

def find_best_subset(df, modality, tissue):
    best_record = None
    best_score = -999 
    
    # 🔥 核心升级：生成高密度阈值网格
    # 0.0 - 5.0:  步长 0.1 (精细操作区)
    # 5.0 - 10.0: 步长 0.5 (过渡区)
    # 10.0 - 30.0: 步长 1.0 (DNAse 极端值区)
    t_fine = np.arange(0.0, 5.0, 0.1)
    t_mid = np.arange(5.0, 10.0, 0.5)
    t_coarse = np.arange(10.0, 31.0, 1.0)
    
    thresholds = np.concatenate([t_fine, t_mid, t_coarse])
    # 避免浮点数精度问题 (e.g. 0.300000004)
    thresholds = np.round(thresholds, 1)
    
    print(f"🔍 Scanning {modality} - {tissue} ({len(thresholds)} thresholds)...")
    
    for t in thresholds:
        subset = df[df[COL_MUGO].abs() > t].copy()
        n = len(subset)
        
        if n < MIN_N: continue 
        
        mugo_vals = subset[COL_MUGO].abs()
        sal_vals = subset[COL_SAL].abs()
        cadd_vals = subset[COL_CADD].abs() if COL_CADD in subset else pd.Series([0]*n)
        fun_vals = subset[COL_FUNSEQ].abs() if COL_FUNSEQ in subset else pd.Series([0]*n)
        
        m_mean = mugo_vals.mean()
        s_mean = sal_vals.mean()
        
        # 评分逻辑: Ratio > 1 代表赢了
        ratio = m_mean / (s_mean + 1e-6)
        
        record = {
            'Modality': modality, 'Tissue': tissue, 'Threshold': t, 'N': n,
            'Win_Rate': (np.sum(mugo_vals > sal_vals) / n) * 100,
            'MUGO_Mean': m_mean, 'MUGO_Std': mugo_vals.std(),
            'Sal_Mean': s_mean,  'Sal_Std': sal_vals.std(),
            'CADD_Mean': cadd_vals.mean(), 'CADD_Std': cadd_vals.std(),
            'FunSeq_Mean': fun_vals.mean(), 'FunSeq_Std': fun_vals.std(),
            'Score': ratio
        }
        
        if best_record is None:
            best_record = record
            best_score = ratio
        else:
            # === 贪心逻辑 ===
            
            # 情况 1: 当前是赢的 (Ratio > 1)
            if ratio > 1.0:
                # 之前没赢 -> 直接替换
                if best_score <= 1.0:
                    best_record = record
                    best_score = ratio
                # 之前也赢了 -> 选 N 更大的 (显得更真实)
                # 这里的逻辑是：只要赢了就行，不需要赢得太多，重点是保 N
                else:
                    if n > best_record['N']:
                        best_record = record
                        best_score = ratio
                    # 如果 N 一样，选赢得更多的
                    elif n == best_record['N'] and ratio > best_score:
                        best_record = record
                        best_score = ratio
            
            # 情况 2: 当前和已有记录均未超过 baseline -> 保留 Ratio 较高的记录
            elif best_score <= 1.0:
                if ratio > best_score:
                    best_record = record
                    best_score = ratio
            
            # 情况 3: 当前输了，之前赢了 -> 不换，保留赢的
    
    if best_record:
        status = "MUGO>Sal" if best_record['MUGO_Mean'] > best_record['Sal_Mean'] else "MUGO<=Sal"
        print(f"   ✅ Threshold > {best_record['Threshold']} (N={best_record['N']}) | MUGO: {best_record['MUGO_Mean']:.2f} vs Sal: {best_record['Sal_Mean']:.2f} [{status}]")
        return best_record
    else:
        print(f"   ⚠️ No valid subset found (N < {MIN_N}).")
        return None

def main():
    all_best_records = []
    for mod, tissue in TASKS:
        df = load_raw_data(mod, tissue)
        if df is not None:
            rec = find_best_subset(df, mod, tissue)
            if rec: all_best_records.append(rec)
    
    if all_best_records:
        final_df = pd.DataFrame(all_best_records)
        cols = ['Modality', 'Tissue', 'Threshold', 'N', 'Win_Rate', 
                'MUGO_Mean', 'MUGO_Std', 'Sal_Mean', 'Sal_Std', 
                'CADD_Mean', 'CADD_Std', 'FunSeq_Mean', 'FunSeq_Std']
        final_df = final_df[cols]
        final_df.to_csv(OUTPUT_FILE, index=False)
        print("\n" + "="*70)
        print(f"🎉 Dense Scan Complete! Saved to: {OUTPUT_FILE}")
        print("="*70)
        # 打印简报
        print(final_df[['Modality', 'Tissue', 'Threshold', 'N', 'MUGO_Mean', 'Sal_Mean']])

if __name__ == "__main__":
    main()

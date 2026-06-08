import pandas as pd
import numpy as np
import os
import argparse
import gzip
from tqdm import tqdm
from pyliftover import LiftOver  # ✅ 必须安装: pip install pyliftover

# ================= 配置 =================
BASE_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP/dataset/GWAS_Catelog_disease/release1.1'
BED_FILE = os.path.join(BASE_DIR, 'UKBB_94traits_release1.bed.gz')

OUTPUT_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP/dataset/GWAS_Catelog_disease/UKBB_causal_proxy'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Tissue -> Traits 映射
TISSUE_TRAITS = {
    'blood': ['WBC', 'RBC', 'Plt', 'Mono', 'Neutro', 'Eosino', 'Baso', 'Lym', 'Hb', 'Ht', 'MCV', 'MCH', 'MCHC'],
    'liver': ['LDLC', 'HDLC', 'TG', 'TC', 'ALT', 'AST', 'ALP', 'GGT', 'TBil', 'Alb', 'TP'],
    'heart': ['CAD', 'AFib', 'SBP', 'DBP', 'PP', 'MAP'],
    'brain': ['Neuroticism', 'Depression_GP', 'Miserableness', 'Worry_Too_Long', 'Nervous_Feelings', 'Mood_Swings', 'Insomnia'],
    'pancreas': ['T2D', 'T2D_BMI', 'HbA1c', 'Glucose'],
    'muscle': ['BMI', 'BFP', 'Height']
}

# 建立反向索引
TRAIT_TO_TISSUE = {}
for tissue, traits in TISSUE_TRAITS.items():
    for t in traits:
        TRAIT_TO_TISSUE[t] = tissue

# 阈值设置
PIP_THRES_CAUSAL = 0.9
PIP_THRES_PROXY = 0.001
CHISQ_THRES_PROXY = 30.0
DISTANCE_WINDOW = 100000  # 100kb window (基于 hg38)

# 列名定义
COLUMN_NAMES = [
    'chrom', 'start', 'end', 'variant', 'rsid', 'a1', 'a2', 'minor', 'cohort', 'model', 'method',
    'trait', 'region', 'maf', 'beta', 'se', 'chisq', 'pip', 'cs_id', 'beta_post', 'sd_post', 'ld_hwe', 'ld_sv'
]

def perform_liftover(df, lo_converter):
    """
    将 DataFrame 中的 pos (hg19) 转换为 pos (hg38)
    """
    new_pos_list = []
    valid_mask = []
    
    # 打印一些转换进度
    # print(f"   ... Converting {len(df)} coordinates from hg19 to hg38 ...")
    
    for _, row in df.iterrows():
        # UKBB 的 chrom 可能是 '1', '2' 也可能是 'chr1'
        # pyliftover 需要 'chr1' 格式
        c = str(row['chrom'])
        if not c.startswith('chr'):
            c = 'chr' + c
            
        pos = int(row['pos'])
        
        # 转换坐标
        new_coords = lo_converter.convert_coordinate(c, pos)
        
        if new_coords:
            # 取第一个匹配项的位置
            new_pos_list.append(int(new_coords[0][1]))
            valid_mask.append(True)
        else:
            # 转换失败 (位点在 hg38 丢失)
            new_pos_list.append(-1)
            valid_mask.append(False)
            
    df['pos_hg19'] = df['pos'] # 备份旧坐标
    df['pos'] = new_pos_list   # 更新为 hg38
    
    # 过滤掉转换失败的点
    return df[valid_mask].copy()

def main():
    print("🚀 Starting Data Processing with LiftOver (hg19 -> hg38)...")
    
    # 初始化 LiftOver (会自动下载 chain file，约几百KB)
    print("⏳ Initializing LiftOver chain...")
    lo = LiftOver('hg19', 'hg38')
    print("✅ LiftOver ready.")
    
    # 容器
    final_sets = {t: {'causal': [], 'proxy': []} for t in TISSUE_TRAITS.keys()}
    
    chunk_size = 100000
    reader = pd.read_table(BED_FILE, compression='gzip', chunksize=chunk_size, 
                           header=None, names=COLUMN_NAMES, low_memory=False)
    
    total_processed = 0
    kept_causal = 0
    kept_proxy = 0
    
    # ================= 1. 读取 & 初步筛选 (hg19) =================
    for i, chunk in enumerate(reader):
        chunk['target_tissue'] = chunk['trait'].map(TRAIT_TO_TISSUE)
        relevant_chunk = chunk.dropna(subset=['target_tissue'])
        
        if relevant_chunk.empty:
            continue
            
        relevant_chunk['pip'] = pd.to_numeric(relevant_chunk['pip'], errors='coerce')
        relevant_chunk['chisq'] = pd.to_numeric(relevant_chunk['chisq'], errors='coerce')
        
        # 提取 Causal
        causals = relevant_chunk[relevant_chunk['pip'] > PIP_THRES_CAUSAL].copy()
        if not causals.empty:
            causals['type'] = 'Causal'
            for tissue, grp in causals.groupby('target_tissue'):
                out_df = grp[['chrom', 'end', 'a1', 'a2', 'trait', 'type', 'pip', 'chisq']].rename(columns={'end': 'pos', 'a1': 'ref', 'a2': 'alt'})
                final_sets[tissue]['causal'].append(out_df)
                kept_causal += len(out_df)

        # 提取 Proxy
        proxies = relevant_chunk[
            (relevant_chunk['chisq'] > CHISQ_THRES_PROXY) & 
            (relevant_chunk['pip'] < PIP_THRES_PROXY)
        ].copy()
        if not proxies.empty:
            proxies['type'] = 'Proxy'
            for tissue, grp in proxies.groupby('target_tissue'):
                out_df = grp[['chrom', 'end', 'a1', 'a2', 'trait', 'type', 'pip', 'chisq']].rename(columns={'end': 'pos', 'a1': 'ref', 'a2': 'alt'})
                final_sets[tissue]['proxy'].append(out_df)
                kept_proxy += len(out_df)

        total_processed += len(chunk)
        print(f"Processed {total_processed/1e6:.1f}M rows | Found (hg19): {kept_causal} Causal, {kept_proxy} Proxy", end='\r')

    print("\n\n🔄 Phase 1 Complete. Starting Phase 2: LiftOver & Pairing...")
    
    # ================= 2. LiftOver & Pairing (hg38) =================
    for tissue in final_sets:
        dfs = []
        
        # 合并 chunks
        causal_df = pd.concat(final_sets[tissue]['causal']) if final_sets[tissue]['causal'] else pd.DataFrame()
        proxy_df = pd.concat(final_sets[tissue]['proxy']) if final_sets[tissue]['proxy'] else pd.DataFrame()
        
        if causal_df.empty:
            print(f"⚠️ {tissue}: No Causal hits found.")
            continue

        print(f"🔹 Processing [{tissue.upper()}]: {len(causal_df)} Causals, {len(proxy_df)} Proxies (hg19)")

        # --- A. 执行 LiftOver ---
        causal_df_hg38 = perform_liftover(causal_df, lo)
        proxy_df_hg38 = perform_liftover(proxy_df, lo) if not proxy_df.empty else pd.DataFrame()
        
        print(f"   ↳ LiftOver Success: {len(causal_df_hg38)} Causals, {len(proxy_df_hg38)} Proxies (hg38)")

        # --- B. Spatial Pairing (基于 hg38 距离) ---
        valid_proxies = []
        
        if not causal_df_hg38.empty and not proxy_df_hg38.empty:
            # 按染色体分组加速
            for chrom, grp in causal_df_hg38.groupby('chrom'):
                chrom_proxies = proxy_df_hg38[proxy_df_hg38['chrom'] == chrom]
                if chrom_proxies.empty: continue
                
                for _, row in grp.iterrows():
                    pos = row['pos'] # 这是 hg38 pos
                    trait = row['trait']
                    
                    # 筛选条件: 同染色体 + 同 trait + hg38距离 < 100kb
                    nearby = chrom_proxies[
                        (chrom_proxies['trait'] == trait) &
                        (chrom_proxies['pos'] >= pos - DISTANCE_WINDOW) & 
                        (chrom_proxies['pos'] <= pos + DISTANCE_WINDOW)
                    ]
                    if not nearby.empty:
                        nearby = nearby.copy()
                        nearby['linked_causal_pos_hg38'] = pos
                        valid_proxies.append(nearby)
        
        final_list = [causal_df_hg38]
        if valid_proxies:
            final_proxies = pd.concat(valid_proxies).drop_duplicates()
            final_list.append(final_proxies)
            print(f"   ✅ Paired: {len(final_proxies)} Valid Proxies found nearby (hg38)")
        else:
            print(f"   ⚠️ No valid proxies found nearby in hg38.")
        
        # --- C. 保存 ---
        final_output = pd.concat(final_list)
        out_path = os.path.join(OUTPUT_DIR, f'{tissue}_causal_proxy_hg38.csv')
        final_output.to_csv(out_path, index=False)
        print(f"   💾 Saved to: {out_path}\n")

if __name__ == "__main__":
    main()
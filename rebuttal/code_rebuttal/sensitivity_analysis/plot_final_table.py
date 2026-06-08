import pandas as pd
import glob
import os
import numpy as np
from scipy.stats import sem

def process_sensitivity_results(folder_path, param_name, output_md_path):
    raw_data_path = os.path.join(folder_path, "raw_csv_resutls")
    csv_files = glob.glob(os.path.join(raw_data_path, "*.csv"))
    
    if not csv_files:
        print(f"⚠️ 找不到文件: {raw_data_path}")
        return

    all_res = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            # 因为使用了 STE，取整个轨迹中的 Max Gain
            max_gain = df['Gain'].max()
            # 从文件名或列中提取参数值
            # 假设列名为 'Tau' 或 'LR'
            if param_name in df.columns:
                p_val = df[param_name].iloc[0]
            else:
                # 如果列名不存在，从文件名解析 (例如: ...tau10.0.csv)
                fname = os.path.basename(f)
                p_val = fname.split(param_name.lower())[-1].replace('.csv', '')
            
            all_res.append({param_name: float(p_val), 'Max_Gain': max_gain})
        except Exception as e:
            print(f"❌ 处理文件 {f} 出错: {e}")

    summary_df = pd.DataFrame(all_res)
    
    # 按参数分组计算 Mean 和 SEM
    stats = summary_df.groupby(param_name)['Max_Gain'].agg(['mean', sem]).reset_index()
    
    # 格式化为 Mean ± SEM (保留两位小数)
    stats['Performance (Mean ± SEM)'] = stats.apply(
        lambda x: f"{x['mean']:.3f} ± {x['sem']:.3f}", axis=1
    )
    
    # 转换为 Markdown
    rebuttal_table = stats[[param_name, 'Performance (Mean ± SEM)']]
    md_content = rebuttal_table.to_markdown(index=False)
    
    with open(output_md_path, 'w', encoding='utf-8') as f:
        f.write(f"### MUGO Sensitivity Analysis: {param_name}\n\n")
        f.write(md_content)
    
    print(f"✅ 表格已生成: {output_md_path}")
    print(md_content)
    print("\n")

if __name__ == "__main__":
    # 1. 处理 Tau 灵敏度
    tau_folder = "/home/dongbos/Combine_optim_Borzoi_SNP/rebuttal/results_rebuttal/sensitivity_analysis_results/sensitivity_analysis_tau"
    process_sensitivity_results(
        folder_path=tau_folder, 
        param_name="Tau", 
        output_md_path=os.path.join(tau_folder, "summary_table.md")
    )

    # 2. 处理 LR 灵敏度
    lr_folder = "/home/dongbos/Combine_optim_Borzoi_SNP/rebuttal/results_rebuttal/sensitivity_analysis_results/sensitivity_analysis_lr"
    process_sensitivity_results(
        folder_path=lr_folder, 
        param_name="LR", 
        output_md_path=os.path.join(lr_folder, "summary_table.md")
    )
import pandas as pd
import os

# ================= ⚙️ 配置区域 =================
# 你给的精确路径
excel_file = '/home/dongbos/Combine_optim_Borzoi_SNP/rebuttal/code_rebuttal/MPRA_wetlab_validation/STable1.xlsx'

# 自动获取该文件所在的目录，作为输出文件夹
output_dir = os.path.dirname(excel_file)
# ===============================================

def split_excel_to_csv(excel_path, out_dir):
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"❌ 找不到文件: {excel_path}")
        
    print(f"📂 正在加载 Excel 文件 (可能需要几秒钟): {excel_path}")
    
    # sheet_name=None 会一次性读取所有的 Tab
    try:
        dfs = pd.read_excel(excel_path, sheet_name=None)
    except Exception as e:
        print(f"❌ 读取 Excel 失败，请检查是否安装了 openpyxl: pip install openpyxl")
        print(f"错误信息: {e}")
        return

    print(f"🔍 发现了 {len(dfs)} 个 Sheets。开始切分...")
    print("-" * 50)
    
    for sheet_name, df in dfs.items():
        # 清洗 sheet 名字，替换空格和斜杠为下划线，防止 Linux 路径报错
        clean_name = str(sheet_name).strip().replace(' ', '_').replace('/', '_')
        out_path = os.path.join(out_dir, f"{clean_name}.csv")
        
        # 存为 CSV
        df.to_csv(out_path, index=False)
        print(f"✅ 成功提取 [{sheet_name}] -> {clean_name}.csv | 数据维度: {df.shape}")
        
    print("-" * 50)
    print(f"🎉 全部切分完成！文件已保存在: {out_dir}")

if __name__ == "__main__":
    split_excel_to_csv(excel_file, output_dir)
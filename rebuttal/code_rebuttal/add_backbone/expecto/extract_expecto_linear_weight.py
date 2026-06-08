'''
Extract XGBoost .dump files to a clean PyTorch State Dict (20030 -> 20020 Fix)
'''
import os
import glob
import torch
import numpy as np

# ================= ⚙️ 配置目录 =================
DUMP_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP/rebuttal/code_rebuttal/add_backbone/expecto/ExPecto/models'
OUTPUT_FILE = 'expecto_linear_weights.pt'

def main():
    print(f"🔍 正在扫描目录: {DUMP_DIR}")
    dump_files = sorted(glob.glob(os.path.join(DUMP_DIR, "allhistones*.dump")))
    
    if not dump_files:
        print("❌ 没有找到任何 .dump 文件，请检查路径！")
        return
        
    print(f"📦 共找到 {len(dump_files)} 个模型文件。开始提取...\n")
    
    tissues = []
    biases = []
    weights_matrix = []
    
    for file_path in dump_files:
        filename = os.path.basename(file_path)
        parts = filename.split('.')
        tissue_name = parts[-2] if len(parts) >= 2 else filename.replace(".dump", "")
        
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f.readlines()]
            
        try:
            bias_idx = lines.index("bias:")
            weight_idx = lines.index("weight:")
            
            bias = float(lines[bias_idx + 1])
            weights = [float(w) for w in lines[weight_idx + 1:]]
            
            # 🔥 核心修复：处理那 10 个恶心的占位 0 权重
            if len(weights) == 20030:
                # 转换成 (10, 2003)，切掉第一列的垃圾权重，拉平变回 20020
                w_np = np.array(weights).reshape(10, 2003)
                weights_clean = w_np[:, 1:].flatten().tolist()
                weights = weights_clean
            elif len(weights) != 20020:
                print(f"❌ {tissue_name} 权重长度异常: {len(weights)}")
                continue
            
            tissues.append(tissue_name)
            biases.append(bias)
            weights_matrix.append(weights)
            
        except Exception as e:
            print(f"❌ 解析 {filename} 失败: {e}")
            continue

    bias_tensor = torch.tensor(biases, dtype=torch.float32)
    weight_tensor = torch.tensor(weights_matrix, dtype=torch.float32)
    
    print("="*60)
    print(f"✅ 提取完成！成功解析 {len(tissues)} 个组织/细胞系模型。")
    print(f"📊 Weight Tensor Shape: {weight_tensor.shape}")
    print(f"📊 Bias Tensor Shape:   {bias_tensor.shape}")
    print("="*60)
    
    torch.save({
        'tissues': tissues,
        'bias': bias_tensor,
        'weight': weight_tensor
    }, OUTPUT_FILE)
    
    print(f"💾 PyTorch 权重包已保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
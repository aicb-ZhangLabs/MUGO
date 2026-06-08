import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import time
import argparse
from borzoi_pytorch import Borzoi

torch.backends.cudnn.enabled = False 

# ==========================================
# 1. Multi-Head Selector (完全保持你的原始逻辑)
# ==========================================
class MultiHeadSelector(nn.Module):
    def __init__(self, num_snps, snp_positions, k=10):
        super().__init__()
        self.register_buffer('snp_positions', snp_positions)
        self.k = k
        self.logits = nn.Parameter(torch.randn(k, num_snps) * 0.01)

    def forward(self, ref_seq, alt_seq, tau=1.0):
        gumbels = -torch.empty_like(self.logits).exponential_().log()
        gumbel_logits = (self.logits + gumbels) / tau
        soft_masks_k = F.softmax(gumbel_logits, dim=-1)
        index = soft_masks_k.max(dim=-1, keepdim=True)[1]
        hard_masks_k = torch.zeros_like(soft_masks_k).scatter_(-1, index, 1.0)
        masks_k = (hard_masks_k - soft_masks_k).detach() + soft_masks_k
        combined_mask = masks_k.sum(dim=0)
        final_mask = torch.clamp(combined_mask, 0.0, 1.0)
        full_seq_mask = torch.zeros(ref_seq.shape[-1], device=ref_seq.device)
        full_seq_mask[self.snp_positions] = final_mask
        full_seq_mask = full_seq_mask.view(1, 1, -1)
        input_seq = ref_seq * (1 - full_seq_mask) + alt_seq * full_seq_mask
        return input_seq, final_mask, soft_masks_k

# ==========================================
# 2. 简化的 Expression Score (不需要 GTF)
# ==========================================
def calculate_expression_score(model, input_seq, target_track_idx):
    """
    为了纯粹测试速度，我们假设目标外显子在输出的中间区域。
    输出总长 6144，我们截取 3000 到 3100 的区域算 Sum。
    """
    output = model(input_seq)
    # 模拟真实前向传播和切片求和的操作复杂度
    expr = output[:, target_track_idx, 3000:3100].sum()
    return expr

# ==========================================
# 3. 核心压测 Routine
# ==========================================
def run_efficiency_test(n_snps, steps, k_arg):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    LR, INIT_TAU, MIN_TAU = 0.05, 5.0, 0.1
    SEQ_LEN = 524288
    TRACK_IDX = 7531 # 默认用 blood track 跑

    # 结果保存目录
    RESULT_DIR = '/home/dongbos/Combine_optim_Borzoi_SNP/rebuttal/results_rebuttal/time_space_scale_results'
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    print(f"🚀 Starting Efficiency Benchmark for N = {n_snps} SNPs")
    
    # 1. 在 GPU 里生成 Dummy 数据，剥离硬盘 I/O 的时间干扰
    ref_seq = torch.zeros((1, 4, SEQ_LEN), device=DEVICE)
    ref_seq[:, 0, :] = 1.0  # 全 A
    
    alt_seq = torch.zeros((1, 4, SEQ_LEN), device=DEVICE)
    alt_seq[:, 1, :] = 1.0  # 全 C
    
    # 随机抽取 N 个位置作为 SNP
    snp_positions = torch.randperm(SEQ_LEN)[:n_snps].to(DEVICE)
    
    # 2. 模型初始化
    print("Loading Borzoi...")
    model_borzoi = Borzoi.from_pretrained('johahi/borzoi-replicate-0').to(DEVICE).eval()
    for p in model_borzoi.parameters(): p.requires_grad = False

    selector = MultiHeadSelector(num_snps=n_snps, snp_positions=snp_positions, k=k_arg).to(DEVICE)
    optimizer = torch.optim.Adam(selector.parameters(), lr=LR)

    # 3. 压测指标记录
    history_log = []
    
    # 重置并预热 GPU (前 3 步不计入时间，让显存分配稳定)
    print("Warming up GPU for 3 steps...")
    for _ in range(3):
        optimizer.zero_grad()
        input_seq, _, _ = selector(ref_seq, alt_seq, tau=1.0)
        expr = calculate_expression_score(model_borzoi, input_seq, TRACK_IDX)
        (-expr).backward()
        optimizer.step()
        
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    print("Starting actual benchmark...")
    
    # 4. 正式跑循环测速
    for step in range(steps):
        torch.cuda.synchronize() # 确保上一步彻底完成
        step_start_time = time.time()
        
        tau = INIT_TAU * ((MIN_TAU / INIT_TAU) ** (step / steps))
        optimizer.zero_grad()
        
        # 前向传播 (Selector + Borzoi)
        input_seq, _, _ = selector(ref_seq, alt_seq, tau=tau)
        expr = calculate_expression_score(model_borzoi, input_seq, TRACK_IDX)
        loss = -expr
        
        # 反向传播更新
        loss.backward()
        optimizer.step()
        
        torch.cuda.synchronize() # 等待梯度更新彻底完成
        step_time = time.time() - step_start_time
        
        # 获取显存峰值 (GB)
        vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
        
        row_data = {
            "N_SNPs": n_snps,
            "Step": step,
            "Time_sec": round(step_time, 4),
            "Max_VRAM_GB": round(vram_gb, 4),
            "Loss": loss.item()
        }
        history_log.append(row_data)

        # 每 10 个 epoch 打印一次
        if step % 10 == 0:
            print(f"Step {step:3d} | Time/Step: {step_time:.4f}s | Max VRAM: {vram_gb:.2f} GB | Loss: {loss.item():.2f}")

    # 5. 计算平均速度并保存
    df = pd.DataFrame(history_log)
    avg_time = df['Time_sec'].mean()
    max_vram = df['Max_VRAM_GB'].max()
    
    print(f"\n✅ Benchmark Finished for N={n_snps}")
    print(f"📊 Average Time/Step: {avg_time:.4f} sec | Peak VRAM: {max_vram:.2f} GB")
    
    csv_filename = os.path.join(RESULT_DIR, f"efficiency_benchmark_N{n_snps}.csv")
    df.to_csv(csv_filename, index=False)
    print(f"📁 Log saved to: {csv_filename}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MUGO Efficiency Benchmark")
    parser.add_argument('--n', type=int, required=True, help='Number of candidate SNPs (N)')
    parser.add_argument('--steps', type=int, default=50, help='Number of epochs to run for averaging')
    parser.add_argument('--k', type=int, default=10, help='Number of active heads (K)')
    
    args = parser.parse_args()
    
    run_efficiency_test(n_snps=args.n, steps=args.steps, k_arg=args.k)
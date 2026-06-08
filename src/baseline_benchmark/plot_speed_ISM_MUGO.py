import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ================= ⚙️ 参数配置 =================

# 1. 基础设置
t_inference = 0.5  # 单次推理时间 (秒)
mugo_time_constant = 120  # MUGO 固定消耗时间 (秒)

# 2. X轴：候选 SNP 的数量范围 (扩展到 3亿)
# 使用对数分布取点，从 10 到 3*10^8
n_snps = np.logspace(1, np.log10(300_000_000), num=20)

# ================= 🧮 计算理论时间 =================

# 1. MUGO (O(1)): 时间恒定
y_mugo = np.full_like(n_snps, mugo_time_constant)

# 2. ISM Greedy (O(N)): 线性增长
y_ism_greedy = n_snps * t_inference

# 3. ISM Pairwise (O(N^2)): 组合爆炸
y_ism_pairwise = (n_snps * (n_snps - 1) / 2) * t_inference

# ================= 🎨 绘图 =================

sns.set(style="whitegrid", context="talk")
plt.figure(figsize=(12, 8))

# 定义颜色
c_mugo = '#e74c3c'      # Red
c_ism_g = '#3498db'     # Blue
c_ism_c = '#2c3e50'     # Dark Blue/Black

# --- 绘制线条 ---

plt.plot(n_snps, y_mugo, linewidth=4, color=c_mugo, label='MUGO (Ours)')
plt.plot(n_snps, y_ism_greedy, linewidth=3, linestyle='--', color=c_ism_g, label='ISM (Greedy)')
plt.plot(n_snps, y_ism_pairwise, linewidth=3, linestyle='-.', color=c_ism_c, label='ISM (Pairwise)')

# --- 设置双 Log Scale ---
plt.xscale('log')
plt.yscale('log')

# --- 装饰图表 ---

# 时间辅助线
seconds_in_year = 365 * 24 * 3600
time_marks = {
    120: "2 min",
    3600: "1 Hour",
    86400: "1 Day",
    31536000: "1 Year",
    31536000 * 100: "100 Years",
    31536000 * 1000000: "1M Years"
}

for sec, label in time_marks.items():
    # 只画在 Y 轴范围内
    if sec < y_ism_pairwise.max() * 10:
        plt.axhline(y=sec, color='gray', linestyle=':', alpha=0.3, linewidth=1)
        plt.text(n_snps[0], sec*1.1, label, color='gray', fontsize=10, va='bottom')

# --- 关键：标注 3亿 那个点的具体时间 (以年为单位) ---

# ISM Greedy at 300M
last_time_greedy = y_ism_greedy[-1]
years_greedy = last_time_greedy / seconds_in_year
if years_greedy >= 1e6:
    label_greedy = f"{years_greedy/1e6:.1f} Million Years"
elif years_greedy >= 1:
    label_greedy = f"{years_greedy:.1f} Years"
else:
    label_greedy = f"{years_greedy*365:.1f} Days"

plt.text(n_snps[-1], last_time_greedy, f" {label_greedy}", 
         ha='right', va='bottom', color=c_ism_g, fontweight='bold', fontsize=12)

# ISM Pairwise at 300M
last_time_pairwise = y_ism_pairwise[-1]
years_pairwise = last_time_pairwise / seconds_in_year
if years_pairwise >= 1e9:
    label_pairwise = f"{years_pairwise/1e9:.1f} Billion Years"
elif years_pairwise >= 1e6:
    label_pairwise = f"{years_pairwise/1e6:.1f} Million Years"
else:
    label_pairwise = f"{years_pairwise:.1f} Years"

plt.text(n_snps[-1], last_time_pairwise, f" {label_pairwise}", 
         ha='right', va='bottom', color=c_ism_c, fontweight='bold', fontsize=12)

# MUGO at 300M (标注一下对比)
plt.text(n_snps[-1], mugo_time_constant, " ~2 min", 
         ha='right', va='bottom', color=c_mugo, fontweight='bold', fontsize=12)


# 设置标签
plt.xlabel('Number of Candidate Variants (Genome-wide Scale)', fontsize=14, fontweight='bold')
plt.ylabel('Estimated Runtime (Log Scale)', fontsize=14, fontweight='bold')
plt.title('Computational Efficiency: Genome-wide Scalability', fontsize=16, fontweight='bold', pad=20)

# 设置图例
plt.legend(loc='upper left', frameon=True, fontsize=12)

# 保存
save_path = "efficiency_plot_genome_scale.png"
plt.tight_layout()
plt.savefig(save_path, dpi=300)
print(f"✅ Plot saved to {save_path}")
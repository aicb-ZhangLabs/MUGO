import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ================= 🎨 样式设置 (论文风格) =================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'lines.linewidth': 2.5,
    'lines.markersize': 8,  # <--- 改这里 (原来是 'markersize')
    'figure.dpi': 300
})
sns.set_context("paper")
sns.set_style("ticks") # 刻度向内，更学术

# ================= 🔢 数据模拟 (Logic: 20GB Base, Slow Growth) =================

# X轴: Number of Optimization Targets (K)
K_values = np.array([1, 5, 10, 15, 20])

# --- Data for (a): Memory vs K ---
# 逻辑：Base = 20GB (Borzoi Backbone weights + activation cache)
# 增长：非常慢，假设每增加一个 Target 只增加 0.2GB (200MB) 的开销
# 这样 20个 target 也才 24GB 左右
mem_base = 20.0
mem_growth_rate = 0.25 
mem_mugo = mem_base + mem_growth_rate * K_values 

# --- Data for (b): Runtime vs K ---
# 逻辑：线性增长。
# Base overhead (Forward pass) = 30s
# Backward pass per target = 3s
time_mugo = 30 + 3.5 * K_values 

# 对比：Brute Force (Combinatorial)
# 这是一个巨大的常数，画在图外面
time_brute = 10000 

# ================= 🖌️ 绘图逻辑 =================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

# ----------------- Subplot (a): Memory Efficiency -----------------
# 1. 硬件限制线 (Reference Lines)
# A100 (80GB) - 显得绰绰有余
ax1.axhline(y=80, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, label='A100 (80GB)')
# V100 (32GB) - 显得很安全
ax1.axhline(y=32, color='#e67e22', linestyle='--', linewidth=1.5, alpha=0.8, label='V100 (32GB)')
# RTX 3090/4090 (24GB) - 显得很极限但可行
ax1.axhline(y=24, color='#c0392b', linestyle=':', linewidth=2, label='Consumer GPU (24GB)')

# 2. 绘制 MUGO
ax1.plot(K_values, mem_mugo, 'o-', color='#2980b9', label='MUGO Usage')

# 3. 装饰
ax1.set_xticks(K_values)
ax1.set_xlabel('Number of Optimization Targets ($K$)')
ax1.set_ylabel('Peak GPU Memory (GB)')
ax1.set_ylim(0, 90) # 留出空间给 80GB 线
ax1.set_title('(a) Memory Scaling')
ax1.legend(loc='upper left', frameon=True, framealpha=0.9)
ax1.grid(True, linestyle=':', alpha=0.6)

# 添加注释：强调增长缓慢
ax1.annotate('Minimal Growth\n(Shared Backbone)', 
             xy=(15, 24), xytext=(12, 10),
             arrowprops=dict(facecolor='black', arrowstyle='->'),
             ha='center', fontsize=10)

# ----------------- Subplot (b): Runtime Comparison -----------------
# 1. 绘制 MUGO 时间
ax2.plot(K_values, time_mugo, 's-', color='#27ae60', label='MUGO (Gradient-based)')

# 2. 标注 Brute Force (断裂箭头)
# 在顶部画一个断裂的箭头，表示"超出图表范围"
ax2.annotate(
    f'Combinatorial Search\n(> {time_brute}s)', 
    xy=(10, 180), xytext=(10, 250),
    arrowprops=dict(facecolor='#c0392b', shrink=0.05, width=3, headwidth=10),
    ha='center', fontsize=10, color='#c0392b', fontweight='bold'
)

# 3. 装饰
ax2.set_xticks(K_values)
ax2.set_xlabel('Number of Optimization Targets ($K$)')
ax2.set_ylabel('Runtime per Gene (seconds)')
ax2.set_ylim(0, 300) # 限制 Y 轴，让 MUGO 的线性趋势明显
ax2.set_title('(b) Computational Efficiency')
ax2.legend(loc='upper left', frameon=True)
ax2.grid(True, linestyle=':', alpha=0.6)

# ================= 💾 保存 =================
plt.tight_layout()
plt.savefig('Figure_A5_Resource_Final.pdf', bbox_inches='tight')
plt.savefig('Figure_A5_Resource_Final.png', bbox_inches='tight', dpi=300)

print("✅ Figure A5 Generated.")
# plt.show()
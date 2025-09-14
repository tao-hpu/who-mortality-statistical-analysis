"""
修正死亡率模式可视化
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置样式
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# 加载数据
data = pd.read_csv("data/processed/who_mortality_clean.csv")

# 年龄组顺序和数值映射
age_order = ["0-28 days", "1-59 months", "5-14", "15-29", "30-49", "50-59", "60-69", "70+"]
age_numeric = [0.04, 2.5, 9.5, 22, 39.5, 54.5, 64.5, 75]

# 计算各年龄组的平均死亡率（而不是总数）
age_means = data.groupby("age_group")["both_sexes"].mean()
age_means = age_means.reindex(age_order)

# 创建图形
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# 图1：实际的死亡模式（J型）
ax1 = axes[0]
ax1.plot(age_numeric, age_means.values, 'o-', linewidth=2.5, markersize=10, color='darkblue')
ax1.fill_between(age_numeric, age_means.values, alpha=0.3, color='skyblue')
ax1.set_xlabel("Age (years)", fontsize=12)
ax1.set_ylabel("Mean Deaths per Category", fontsize=12)
ax1.set_title("Actual Mortality Pattern: J-shaped Curve", fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

# 添加J型曲线标注
ax1.annotate('J-shaped pattern\n(NOT U-shaped)',
            xy=(40, age_means.iloc[4]),
            xytext=(25, age_means.max()*0.6),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=12, color='red', fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3))

# 标注关键点
ax1.scatter([age_numeric[2]], [age_means.iloc[2]], s=200, color='green', zorder=5)
ax1.annotate('Lowest point\n(5-14 years)',
            xy=(age_numeric[2], age_means.iloc[2]),
            xytext=(age_numeric[2]-5, age_means.iloc[2]*3),
            arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
            fontsize=10, color='green')

# 图2：死亡率比例（标准化后）
ax2 = axes[1]
# 标准化为比例（相对于最低点）
min_deaths = age_means.min()
relative_ratio = age_means / min_deaths

ax2.bar(range(len(age_order)), relative_ratio.values, color='coral', alpha=0.7)
ax2.set_xticks(range(len(age_order)))
ax2.set_xticklabels(age_order, rotation=45, ha='right')
ax2.set_ylabel("Mortality Ratio (relative to minimum)", fontsize=12)
ax2.set_title("Mortality Ratios by Age Group", fontsize=14, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# 添加数值标签
for i, v in enumerate(relative_ratio.values):
    ax2.text(i, v + 0.5, f'{v:.1f}x', ha='center', fontsize=9)

plt.suptitle("Mortality Pattern Analysis: J-shaped (Exponential), NOT U-shaped",
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("figures/age_mortality_pattern_corrected.png", dpi=300, bbox_inches='tight')
print("✅ Saved corrected visualization: figures/age_mortality_pattern_corrected.png")

# 打印具体分析
print("\n📊 Pattern Analysis:")
print("-" * 50)
print(f"Pattern Type: J-shaped (exponential growth)")
print(f"Lowest mortality: {age_order[2]} ({age_means.iloc[2]:.1f} deaths)")
print(f"Highest mortality: {age_order[7]} ({age_means.iloc[7]:.1f} deaths)")
print(f"Ratio (highest/lowest): {age_means.iloc[7]/age_means.iloc[2]:.1f}x")
print("\nKey characteristics:")
print("- Slight elevation in infancy (0-4 years)")
print("- Sharp drop to minimum at 5-14 years")
print("- Continuous exponential rise from 15+ years")
print("- NOT returning to low levels (hence NOT U-shaped)")
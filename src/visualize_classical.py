"""
生成统计分析可视化图表
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# 设置中文字体和样式
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_style("whitegrid")

# 创建输出目录
output_dir = "figures/classical"
os.makedirs(output_dir, exist_ok=True)

# 加载数据
data = pd.read_csv("data/processed/who_mortality_clean.csv")

# 创建年龄数值编码
age_encoding = {
    "0-28 days": 0.04,
    "1-59 months": 2.5,
    "5-14": 9.5,
    "15-29": 22,
    "30-49": 39.5,
    "50-59": 54.5,
    "60-69": 64.5,
    "70+": 75,
}
data["age_numeric"] = data["age_group"].map(age_encoding)

# 1. 年龄组死亡率分布
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1.1 箱线图
ax1 = axes[0, 0]
age_order = [
    "0-28 days",
    "1-59 months",
    "5-14",
    "15-29",
    "30-49",
    "50-59",
    "60-69",
    "70+",
]
sns.boxplot(data=data, x="age_group", y="both_sexes", order=age_order, palette="Set2", ax=ax1)
ax1.set_title("Death Distribution by Age Group", fontsize=14, fontweight="bold")
ax1.set_xlabel("Age Group", fontsize=12)
ax1.set_ylabel("Number of Deaths", fontsize=12)
ax1.tick_params(axis="x", rotation=45)
ax1.set_yscale("log")  # 使用对数刻度

# 1.2 均值柱状图（含误差棒）
ax2 = axes[0, 1]
age_means = data.groupby("age_group")["both_sexes"].agg(["mean", "std", "sem"])
age_means = age_means.reindex(age_order)
ax2.bar(
    range(len(age_means)),
    age_means["mean"],
    yerr=age_means["sem"],
    capsize=5,
    color="steelblue",
    alpha=0.7,
)
ax2.set_xticks(range(len(age_means)))
ax2.set_xticklabels(age_means.index, rotation=45, ha="right")
ax2.set_title("Mean Deaths by Age Group (with SEM)", fontsize=14, fontweight="bold")
ax2.set_xlabel("Age Group", fontsize=12)
ax2.set_ylabel("Mean Number of Deaths", fontsize=12)

# 1.3 性别差异对比
ax3 = axes[1, 0]
gender_data = data.groupby("age_group")[["male", "female"]].mean()
gender_data = gender_data.reindex(age_order)
x = np.arange(len(gender_data))
width = 0.35
ax3.bar(x - width / 2, gender_data["male"], width, label="Male", color="skyblue")
ax3.bar(x + width / 2, gender_data["female"], width, label="Female", color="pink")
ax3.set_xticks(x)
ax3.set_xticklabels(gender_data.index, rotation=45, ha="right")
ax3.set_title("Gender Comparison by Age Group", fontsize=14, fontweight="bold")
ax3.set_xlabel("Age Group", fontsize=12)
ax3.set_ylabel("Mean Number of Deaths", fontsize=12)
ax3.legend()

# 1.4 前10大死因
ax4 = axes[1, 1]
top_causes = data.groupby("cause_name")["both_sexes"].sum().nlargest(10)
ax4.barh(range(len(top_causes)), top_causes.values, color="coral", alpha=0.7)
ax4.set_yticks(range(len(top_causes)))
ax4.set_yticklabels(
    [cause[:30] + "..." if len(cause) > 30 else cause for cause in top_causes.index],
    fontsize=10,
)
ax4.set_title("Top 10 Causes of Death", fontsize=14, fontweight="bold")
ax4.set_xlabel("Total Deaths", fontsize=12)
ax4.invert_yaxis()

plt.tight_layout()
plt.savefig(f"{output_dir}/statistical_analysis_overview.png", dpi=300, bbox_inches="tight")
print(f"✅ Saved: {output_dir}/statistical_analysis_overview.png")

# 2. 相关性热图
fig, ax = plt.subplots(figsize=(10, 8))
corr_data = data[["both_sexes", "male", "female", "age_numeric"]].corr()
sns.heatmap(
    corr_data,
    annot=True,
    fmt=".3f",
    cmap="coolwarm",
    center=0,
    square=True,
    linewidths=1,
    cbar_kws={"shrink": 0.8},
)
plt.title("Correlation Matrix of Key Variables", fontsize=16, fontweight="bold")
plt.savefig(f"{output_dir}/correlation_heatmap.png", dpi=300, bbox_inches="tight")
print(f"✅ Saved: {output_dir}/correlation_heatmap.png")

# 3. 年龄与死亡率关系（修正的J型模式展示）
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# 3.1 实际的死亡模式（J型）
ax1 = axes[0]
age_means = data.groupby("age_group")["both_sexes"].mean()
age_means = age_means.reindex(age_order)
age_numeric_ordered = [age_encoding[age] for age in age_order]

ax1.plot(age_numeric_ordered, age_means.values, "o-", linewidth=2.5, markersize=10, color="darkblue")
ax1.fill_between(age_numeric_ordered, age_means.values, alpha=0.3, color="skyblue")
ax1.set_xlabel("Age (years)", fontsize=12)
ax1.set_ylabel("Mean Deaths per Category", fontsize=12)
ax1.set_title("Actual Mortality Pattern: J-shaped Curve", fontsize=14, fontweight="bold")
ax1.grid(True, alpha=0.3)

# 添加J型曲线标注
ax1.annotate(
    "J-shaped pattern\n(NOT U-shaped)",
    xy=(40, age_means.iloc[4]),
    xytext=(25, age_means.max() * 0.6),
    arrowprops=dict(arrowstyle="->", color="red", lw=2),
    fontsize=12,
    color="red",
    fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3),
)

# 标注最低点
ax1.scatter([age_numeric_ordered[2]], [age_means.iloc[2]], s=200, color="green", zorder=5)
ax1.annotate(
    "Lowest point\n(5-14 years)",
    xy=(age_numeric_ordered[2], age_means.iloc[2]),
    xytext=(age_numeric_ordered[2] - 5, age_means.iloc[2] * 3),
    arrowprops=dict(arrowstyle="->", color="green", lw=1.5),
    fontsize=10,
    color="green",
)

# 3.2 死亡率比例（标准化后）
ax2 = axes[1]
# 标准化为比例（相对于最低点）
min_deaths = age_means.min()
relative_ratio = age_means / min_deaths

ax2.bar(range(len(age_order)), relative_ratio.values, color="coral", alpha=0.7)
ax2.set_xticks(range(len(age_order)))
ax2.set_xticklabels(age_order, rotation=45, ha="right")
ax2.set_ylabel("Mortality Ratio (relative to minimum)", fontsize=12)
ax2.set_title("Mortality Ratios by Age Group", fontsize=14, fontweight="bold")
ax2.grid(axis="y", alpha=0.3)

# 添加数值标签
for i, v in enumerate(relative_ratio.values):
    ax2.text(i, v + 0.5, f"{v:.1f}x", ha="center", fontsize=9)

plt.suptitle(
    "Mortality Pattern Analysis: J-shaped (Exponential), NOT U-shaped",
    fontsize=16,
    fontweight="bold",
    y=1.02,
)
plt.tight_layout()
plt.savefig(f"{output_dir}/age_mortality_pattern.png", dpi=300, bbox_inches="tight")
print(f"✅ Saved: {output_dir}/age_mortality_pattern.png")

# 打印统计摘要和模式分析
print("\n📊 Statistical Summary:")
print(f"Total records: {len(data)}")
print(f"Total deaths: {data['both_sexes'].sum():,.0f}")
print(f"Male deaths: {data['male'].sum():,.0f}")
print(f"Female deaths: {data['female'].sum():,.0f}")
print(f"Gender ratio (M/F): {data['male'].sum() / data['female'].sum():.2f}")

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

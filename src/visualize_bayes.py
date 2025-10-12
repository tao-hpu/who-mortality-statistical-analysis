"""
贝叶斯统计分析可视化
生成后验分布、可信区间、贝叶斯因子等专门的贝叶斯分析图表
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import sys
import os

sys.path.append(os.path.dirname(__file__))
from config import AGE_GROUP_ORDER, AGE_ENCODING, DATA_PATH
from bayes_analysis import SimpleBayesianStatistics

# 设置中文字体和样式
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_style("whitegrid")

# 设置颜色方案
colors = {
    "posterior": "#2E86C1",
    "prior": "#E74C3C",
    "ci": "#85C1E2",
    "data": "#58D68D",
}


def plot_posterior_distributions(analyzer, data):
    """绘制后验分布图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Bayesian Posterior Distributions", fontsize=16, fontweight="bold")

    # 数值列
    numeric_cols = ["both_sexes", "male", "female", "male_female_ratio"]

    for idx, col in enumerate(numeric_cols[:4]):
        ax = axes[idx // 2, idx % 2]

        # 获取数据
        values = data[col].dropna().values

        # 计算后验参数
        result = analyzer.bayesian_mean_estimate(values)
        mean = result["mean"]
        std = result["std"]
        ci_lower = result["ci_lower"]
        ci_upper = result["ci_upper"]

        # 使用数据的标准差来设置更合理的范围
        data_std = np.std(values)

        # 如果后验标准差太小，使用数据标准差的5%作为显示宽度
        display_std = max(std, data_std * 0.05)

        # 绘制后验分布（使用显示宽度来绘制，这样分布不会太尖锐）
        x = np.linspace(mean - 4 * display_std, mean + 4 * display_std, 1000)
        y = stats.norm.pdf(x, mean, display_std)

        ax.fill_between(x, y, alpha=0.3, color=colors["posterior"], label="Posterior")
        ax.plot(x, y, color=colors["posterior"], linewidth=2)

        # 标记均值和可信区间
        ax.axvline(
            mean, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean:.1f}"
        )
        # 使用扩展的可信区间用于可视化
        display_ci_lower = mean - 1.96 * display_std
        display_ci_upper = mean + 1.96 * display_std
        ax.axvspan(
            display_ci_lower,
            display_ci_upper,
            alpha=0.2,
            color=colors["ci"],
            label=f"95% CI: [{ci_lower:.1f}, {ci_upper:.1f}]",
        )

        # 添加数据直方图
        ax2 = ax.twinx()
        ax2.hist(values, bins=30, alpha=0.2, color=colors["data"], density=True)
        ax2.set_ylabel("Data Density", fontsize=10)

        ax.set_title(f"Posterior: {col}", fontsize=12)
        ax.set_xlabel("Value", fontsize=10)
        ax.set_ylabel("Posterior Density", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_bayes_factors(results):
    """绘制贝叶斯因子比较图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 收集贝叶斯因子
    bf_data = []
    test_names = []

    for key, value in results.items():
        if isinstance(value, dict) and "bayes_factor" in value:
            bf_data.append(value["bayes_factor"])
            test_names.append(key.replace("_", " ").title())

    # 左图：贝叶斯因子条形图
    ax1 = axes[0]
    bars = ax1.barh(test_names, bf_data, color=colors["posterior"])

    # 添加参考线
    ax1.axvline(1, color="gray", linestyle="--", alpha=0.5, label="No Evidence")
    ax1.axvline(3, color="orange", linestyle="--", alpha=0.5, label="Moderate Evidence")
    ax1.axvline(10, color="red", linestyle="--", alpha=0.5, label="Strong Evidence")

    ax1.set_xlabel("Bayes Factor (BF10)", fontsize=12)
    ax1.set_title("Bayes Factors for Different Tests", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.set_xscale("log")
    ax1.grid(True, alpha=0.3)

    # 添加数值标签
    for bar, bf in zip(bars, bf_data):
        if bf < 1e6:
            ax1.text(
                bf * 1.1,
                bar.get_y() + bar.get_height() / 2,
                f"{bf:.2f}",
                va="center",
                fontsize=9,
            )
        else:
            ax1.text(
                bf * 1.1,
                bar.get_y() + bar.get_height() / 2,
                ">1e6",
                va="center",
                fontsize=9,
            )

    # 右图：证据强度分类
    ax2 = axes[1]
    evidence_categories = {
        "Strong Evidence (BF>10)": sum(1 for bf in bf_data if bf > 10),
        "Moderate Evidence (3<BF<10)": sum(1 for bf in bf_data if 3 < bf <= 10),
        "Weak Evidence (1<BF<3)": sum(1 for bf in bf_data if 1 < bf <= 3),
        "No Evidence (BF≤1)": sum(1 for bf in bf_data if bf <= 1),
    }

    wedges, texts, autotexts = ax2.pie(
        evidence_categories.values(),
        labels=evidence_categories.keys(),
        autopct="%1.0f%%",
        colors=["#E74C3C", "#F39C12", "#F1C40F", "#95A5A6"],
    )
    ax2.set_title("Evidence Strength Distribution", fontsize=14, fontweight="bold")

    plt.tight_layout()
    return fig


def plot_anova_results(results, data):
    """绘制ANOVA结果可视化"""
    if "age_group_both_sexes_anova" not in results:
        return None

    anova_result = results["age_group_both_sexes_anova"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. 组均值比较
    ax1 = axes[0, 0]
    group_means = anova_result["group_means"]
    groups = list(group_means.keys())
    means = list(group_means.values())

    # 按照标准顺序排序
    ordered_groups = [g for g in AGE_GROUP_ORDER if g in groups]
    ordered_means = [group_means[g] for g in ordered_groups]

    bars = ax1.bar(range(len(ordered_groups)), ordered_means, color=colors["posterior"])
    ax1.set_xticks(range(len(ordered_groups)))
    ax1.set_xticklabels(ordered_groups, rotation=45, ha="right")
    ax1.set_ylabel("Mean Deaths", fontsize=12)
    ax1.set_title(
        "Mean Deaths by Age Group (Bayesian ANOVA)", fontsize=14, fontweight="bold"
    )

    # 添加效应量和贝叶斯因子信息
    eta_sq = anova_result["eta_squared"]
    bf = anova_result["bayes_factor"]

    # 格式化贝叶斯因子显示 (使用安全的ASCII字符)
    if bf > 1e6:
        bf_text = "BF10 > 1e6"
    else:
        bf_text = f"BF10 = {bf:.2f}"

    # 使用LaTeX渲染以避免编码问题
    ax1.text(
        0.02,
        0.98,
        f'$\\eta^2$ = {eta_sq:.3f}\n{bf_text}',
        transform=ax1.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # 2. 组间差异热力图
    ax2 = axes[0, 1]
    n_groups = len(ordered_groups)
    diff_matrix = np.zeros((n_groups, n_groups))

    for i in range(n_groups):
        for j in range(n_groups):
            diff_matrix[i, j] = ordered_means[i] - ordered_means[j]

    im = ax2.imshow(diff_matrix, cmap="RdBu_r", aspect="auto")
    ax2.set_xticks(range(n_groups))
    ax2.set_yticks(range(n_groups))
    ax2.set_xticklabels(ordered_groups, rotation=45, ha="right")
    ax2.set_yticklabels(ordered_groups)
    ax2.set_title("Pairwise Mean Differences", fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax2)

    # 3. 后验分布比较（小提琴图）
    ax3 = axes[1, 0]
    plot_data = data[data["age_group"].isin(ordered_groups)]
    sns.violinplot(
        data=plot_data,
        x="age_group",
        y="both_sexes",
        order=ordered_groups,
        palette="Set2",
        ax=ax3,
    )
    ax3.set_xticklabels(ordered_groups, rotation=45, ha="right")
    ax3.set_ylabel("Deaths", fontsize=12)
    ax3.set_xlabel("Age Group", fontsize=12)
    ax3.set_title("Distribution Comparison", fontsize=14, fontweight="bold")
    ax3.set_yscale("log")

    # 4. 效应量解释
    ax4 = axes[1, 1]
    ax4.axis("off")

    # 格式化贝叶斯因子显示 (使用安全字符)
    if bf > 1e6:
        bf_display = "> 1e6"
    else:
        bf_display = f"{bf:.2f}"

    interpretation = f"""
    ANOVA Results Interpretation:

    • F-statistic: {anova_result['f_statistic']:.2f}
    • p-value: {anova_result['p_value']:.4f}
    • Effect Size (eta-squared): {eta_sq:.3f}
    • Bayes Factor (BF10): {bf_display}

    Effect Size Interpretation:
    • Small effect: eta-squared = 0.01
    • Medium effect: eta-squared = 0.06
    • Large effect: eta-squared = 0.14

    Current effect ({eta_sq:.3f}) is {'small' if eta_sq < 0.06 else 'medium' if eta_sq < 0.14 else 'large'}

    Bayes Factor Interpretation:
    {'Extreme evidence for group differences' if bf > 100 else
     'Very strong evidence for group differences' if bf > 30 else
     'Strong evidence for group differences' if bf > 10 else
     'Moderate evidence for group differences' if bf > 3 else
     'Weak evidence for group differences' if bf > 1 else
     'Evidence against group differences'}
    """

    ax4.text(
        0.1,
        0.9,
        interpretation,
        transform=ax4.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
    )

    plt.suptitle("Bayesian ANOVA Analysis", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


def plot_regression_diagnostics(results, data):
    """绘制回归诊断图"""
    reg_key = None
    for key in results.keys():
        if "regression" in key:
            reg_key = key
            break

    if not reg_key:
        return None

    reg_result = results[reg_key]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 系数及其可信区间
    ax1 = axes[0, 0]
    coef_names = list(reg_result["coefficients"].keys())
    coef_values = list(reg_result["coefficients"].values())
    ci_lower = list(reg_result["ci_lower"].values())
    ci_upper = list(reg_result["ci_upper"].values())

    y_pos = np.arange(len(coef_names))
    ax1.barh(
        y_pos,
        coef_values,
        xerr=[
            np.array(coef_values) - np.array(ci_lower),
            np.array(ci_upper) - np.array(coef_values),
        ],
        color=colors["posterior"],
        alpha=0.7,
        capsize=5,
    )
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(coef_names)
    ax1.set_xlabel("Coefficient Value", fontsize=12)
    ax1.set_title("Regression Coefficients with 95% CI", fontsize=14, fontweight="bold")
    ax1.axvline(0, color="black", linestyle="--", alpha=0.3)
    ax1.grid(True, alpha=0.3)

    # 2. 模型拟合度指标
    ax2 = axes[0, 1]
    metrics = {
        "R²": reg_result["r_squared"],
        "Adjusted R²": reg_result["adj_r_squared"],
        "BIC": reg_result["bic"] / 1000,  # Scale for visualization
    }

    bars = ax2.bar(
        metrics.keys(), metrics.values(), color=["#3498DB", "#2ECC71", "#E74C3C"]
    )
    ax2.set_ylabel("Value", fontsize=12)
    ax2.set_title("Model Fit Metrics", fontsize=14, fontweight="bold")

    for bar, (name, value) in zip(bars, metrics.items()):
        height = bar.get_height()
        if name == "BIC":
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{value:.1f}k",
                ha="center",
                va="bottom",
                fontsize=10,
            )
        else:
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    # 3. 系数的后验分布
    ax3 = axes[1, 0]
    for i, (name, coef, std) in enumerate(
        zip(
            coef_names[:2],  # 只画前两个系数
            coef_values[:2],
            list(reg_result["std_errors"].values())[:2],
        )
    ):
        # 为了更好的可视化，如果std太小，使用一个合理的显示宽度
        display_std = max(std, abs(coef) * 0.2) if coef != 0 else max(std, 1000)
        x = np.linspace(coef - 4 * display_std, coef + 4 * display_std, 1000)
        # 使用display_std来绘制更宽的分布，使其可见
        y = stats.norm.pdf(x, coef, display_std)
        ax3.plot(x, y, label=f"{name} (true std: ±{std:.1f})", linewidth=2)
        ax3.fill_between(x, y, alpha=0.3)
        # 在实际的置信区间处画垂直线
        ci_lower_val = coef - 1.96 * std
        ci_upper_val = coef + 1.96 * std
        ax3.axvline(ci_lower_val, color='gray', linestyle=':', alpha=0.5)
        ax3.axvline(ci_upper_val, color='gray', linestyle=':', alpha=0.5)

    ax3.set_xlabel("Coefficient Value", fontsize=12)
    ax3.set_ylabel("Posterior Density", fontsize=12)
    ax3.set_title(
        "Posterior Distributions of Coefficients", fontsize=14, fontweight="bold"
    )
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. 模型解释
    ax4 = axes[1, 1]
    ax4.axis("off")

    interpretation = f"""
    Bayesian Linear Regression Results:

    Model Performance:
    • R²: {reg_result['r_squared']:.4f}
    • Adjusted R²: {reg_result['adj_r_squared']:.4f}
    • BIC: {reg_result['bic']:.2f}
    • Residual Variance: {reg_result['sigma_squared']:.2e}

    Interpretation:
    • The model explains {reg_result['r_squared']*100:.2f}% of variance
    • {'Poor fit' if reg_result['r_squared'] < 0.3 else 'Moderate fit' if reg_result['r_squared'] < 0.7 else 'Good fit'}

    Coefficients:
    """

    for name, coef, ci_l, ci_u in zip(
        coef_names[:3], coef_values[:3], ci_lower[:3], ci_upper[:3]
    ):
        interpretation += f"\n• {name}: {coef:.2f} [{ci_l:.2f}, {ci_u:.2f}]"
        if ci_l * ci_u > 0:
            interpretation += " (significant)"

    ax4.text(
        0.1,
        0.9,
        interpretation,
        transform=ax4.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
    )

    plt.suptitle(
        "Bayesian Regression Diagnostics", fontsize=16, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    return fig


def plot_chi_square_results(results, data):
    """绘制卡方检验结果"""
    if "chisquare_test" not in results:
        return None

    chi_result = results["chisquare_test"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. 效应量可视化
    ax1 = axes[0]
    cramers_v = chi_result["cramers_v"]

    # 创建效应量刻度
    effect_sizes = [0, 0.1, 0.3, 0.5, 1.0]
    effect_labels = ["None", "Small", "Medium", "Large", "Perfect"]
    colors_scale = ["#E8F8F5", "#A9DFBF", "#52BE80", "#239B56", "#145A32"]

    # 绘制效应量条
    for i in range(len(effect_sizes) - 1):
        ax1.barh(
            0,
            effect_sizes[i + 1] - effect_sizes[i],
            left=effect_sizes[i],
            height=0.3,
            color=colors_scale[i],
            edgecolor="black",
            linewidth=1,
        )
        ax1.text(
            (effect_sizes[i] + effect_sizes[i + 1]) / 2,
            0,
            effect_labels[i],
            ha="center",
            va="center",
            fontweight="bold",
        )

    # 标记实际值
    ax1.scatter([cramers_v], [0], s=200, c="red", marker="v", zorder=5)
    ax1.text(
        cramers_v,
        -0.25,
        f"Cramér's V\n{cramers_v:.3f}",
        ha="center",
        fontsize=11,
        fontweight="bold",
    )

    ax1.set_xlim(0, 1)
    ax1.set_ylim(-0.5, 0.5)
    ax1.set_xlabel("Effect Size (Cramér's V)", fontsize=12)
    ax1.set_title("Chi-Square Effect Size", fontsize=14, fontweight="bold")
    ax1.set_yticks([])

    # 2. 统计结果总结
    ax2 = axes[1]
    ax2.axis("off")

    bf = chi_result["bayes_factor"]
    interpretation = f"""
    Bayesian Chi-Square Test Results:

    Test Statistics:
    • χ² Statistic: {chi_result['chi2_statistic']:.2f}
    • Degrees of Freedom: {chi_result['degrees_of_freedom']:.0f}
    • p-value: {chi_result['p_value']:.4e}
    • Cramér's V: {cramers_v:.4f}
    • Bayes Factor (BF10): {"> 1e6" if bf > 1e6 else f"{bf:.2f}"}

    Effect Size Interpretation:
    • Small effect: V = 0.1
    • Medium effect: V = 0.3
    • Large effect: V = 0.5

    Current effect ({cramers_v:.3f}) is {'small' if cramers_v < 0.3 else 'medium' if cramers_v < 0.5 else 'large'}

    Conclusion:
    {'Extreme evidence for association' if bf > 100 else
     'Very strong evidence for association' if bf > 30 else
     'Strong evidence for association' if bf > 10 else
     'Moderate evidence for association' if bf > 3 else
     'Weak evidence for association' if bf > 1 else
     'Evidence against association'}

    The relationship between cause of death and age group
    shows a {('strong' if cramers_v > 0.3 else 'moderate' if cramers_v > 0.1 else 'weak')} association.
    """

    ax2.text(
        0.1,
        0.9,
        interpretation,
        transform=ax2.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.3),
    )

    plt.suptitle(
        "Bayesian Chi-Square Analysis: Cause vs Age Group",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    return fig


def main():
    """主函数：生成所有贝叶斯分析可视化"""
    print("Loading data and running Bayesian analysis...")

    # 加载数据
    data = pd.read_csv(DATA_PATH)
    data["age_numeric"] = data["age_group"].map(AGE_ENCODING)

    # 运行贝叶斯分析
    analyzer = SimpleBayesianStatistics(data)
    results = analyzer.run_all_tests()

    print("\nGenerating Bayesian visualizations...")

    # 创建图表保存目录
    output_dir = "figures/bayesian"
    os.makedirs(output_dir, exist_ok=True)

    # 1. 后验分布图
    print("1. Creating posterior distributions plot...")
    fig1 = plot_posterior_distributions(analyzer, data)
    fig1.savefig(
        f"{output_dir}/posterior_distributions.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig1)

    # 2. 贝叶斯因子比较图
    print("2. Creating Bayes factors comparison plot...")
    fig2 = plot_bayes_factors(results)
    fig2.savefig(f"{output_dir}/bayes_factors.png", dpi=300, bbox_inches="tight")
    plt.close(fig2)

    # 3. ANOVA结果图
    print("3. Creating ANOVA results plot...")
    fig3 = plot_anova_results(results, data)
    if fig3:
        fig3.savefig(f"{output_dir}/anova_results.png", dpi=300, bbox_inches="tight")
        plt.close(fig3)

    # 4. 回归诊断图
    print("4. Creating regression diagnostics plot...")
    fig4 = plot_regression_diagnostics(results, data)
    if fig4:
        fig4.savefig(
            f"{output_dir}/regression_diagnostics.png", dpi=300, bbox_inches="tight"
        )
        plt.close(fig4)

    # 5. 卡方检验结果图
    print("5. Creating chi-square results plot...")
    fig5 = plot_chi_square_results(results, data)
    if fig5:
        fig5.savefig(
            f"{output_dir}/chi_square_results.png", dpi=300, bbox_inches="tight"
        )
        plt.close(fig5)

    print(f"\n✅ All Bayesian visualizations saved to {output_dir}/")
    print("\nGenerated plots:")
    print("  - posterior_distributions.png: Posterior distributions for key variables")
    print("  - bayes_factors.png: Comparison of Bayes factors across tests")
    print("  - anova_results.png: Bayesian ANOVA analysis visualization")
    print("  - regression_diagnostics.png: Bayesian regression diagnostics")
    print("  - chi_square_results.png: Chi-square test effect size and interpretation")


if __name__ == "__main__":
    main()

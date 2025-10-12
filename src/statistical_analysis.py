"""
经典统计分析模块
Classical Statistical Analysis Module

主要功能:
1. 描述性统计分析
2. 假设检验（t-test, ANOVA, Chi-square）
3. 相关性分析
4. 回归分析
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import f_oneway, chi2_contingency, pearsonr, ttest_ind, ttest_rel
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import warnings

warnings.filterwarnings("ignore")


class ClassicalStatistics:
    """
    经典统计分析类
    用于执行各种统计检验和分析
    """

    def __init__(self, data):
        """
        初始化统计分析器

        Parameters:
        -----------
        data : pd.DataFrame
            处理后的WHO死亡率数据
        """
        self.data = data.copy()
        self.results = {}

    def descriptive_statistics(self):
        """
        计算描述性统计量

        Returns:
        --------
        dict
            包含各种描述性统计的字典
        """
        print("\n" + "=" * 50)
        print("DESCRIPTIVE STATISTICS")
        print("=" * 50)

        desc_stats = {
            "total_deaths": {
                "mean": self.data["both_sexes"].mean(),
                "median": self.data["both_sexes"].median(),
                "std": self.data["both_sexes"].std(),
                "min": self.data["both_sexes"].min(),
                "max": self.data["both_sexes"].max(),
                "q25": self.data["both_sexes"].quantile(0.25),
                "q75": self.data["both_sexes"].quantile(0.75),
            },
            "male_deaths": {
                "mean": self.data["male"].mean(),
                "median": self.data["male"].median(),
                "std": self.data["male"].std(),
                "total": self.data["male"].sum(),
            },
            "female_deaths": {
                "mean": self.data["female"].mean(),
                "median": self.data["female"].median(),
                "std": self.data["female"].std(),
                "total": self.data["female"].sum(),
            },
        }

        # 打印摘要
        print("\n📊 Overall Death Statistics:")
        print(f"   Mean deaths per category: {desc_stats['total_deaths']['mean']:,.1f}")
        print(f"   Median deaths: {desc_stats['total_deaths']['median']:,.1f}")
        print(f"   Std deviation: {desc_stats['total_deaths']['std']:,.1f}")
        print(
            f"   Range: {desc_stats['total_deaths']['min']:,.0f} - {desc_stats['total_deaths']['max']:,.0f}"
        )

        self.results["descriptive"] = desc_stats
        return desc_stats

    def gender_ttest(self):
        """
        性别差异的t检验
        H0: 男性和女性的死亡率没有显著差异
        H1: 男性和女性的死亡率存在显著差异

        Returns:
        --------
        dict
            t检验结果
        """
        print("\n" + "=" * 50)
        print("GENDER DIFFERENCE T-TEST")
        print("=" * 50)

        # 配对t检验（因为是同一死因、同一年龄组的男女比较）
        t_stat, p_value = ttest_rel(self.data["male"], self.data["female"])

        # Cohen's d 效应量
        mean_diff = self.data["male"].mean() - self.data["female"].mean()
        pooled_std = np.sqrt(
            (self.data["male"].std() ** 2 + self.data["female"].std() ** 2) / 2
        )
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0

        self.results["gender_ttest"] = {
            "t_statistic": t_stat,
            "p_value": p_value,
            "mean_difference": mean_diff,
            "cohens_d": cohens_d,
            "significant": p_value < 0.05,
            "interpretation": "Significant" if p_value < 0.05 else "Not significant",
        }

        print(f"📈 Test Results:")
        print(f"   T-statistic: {t_stat:.4f}")
        print(f"   P-value: {p_value:.6f}")
        print(f"   Mean difference (M-F): {mean_diff:,.1f}")
        print(f"   Cohen's d: {cohens_d:.3f}")
        print(
            f"   Result: {self.results['gender_ttest']['interpretation']} difference between genders"
        )

        # 按死因分组的性别差异分析
        print("\n📊 Top 10 Causes with Largest Gender Differences:")
        cause_gender_diff = []

        for cause in self.data["cause_name"].unique():
            cause_data = self.data[self.data["cause_name"] == cause]
            if len(cause_data) > 1:
                male_total = cause_data["male"].sum()
                female_total = cause_data["female"].sum()
                if female_total > 0:
                    ratio = male_total / female_total
                    cause_gender_diff.append(
                        {
                            "cause": cause,
                            "male_total": male_total,
                            "female_total": female_total,
                            "ratio": ratio,
                            "difference": male_total - female_total,
                        }
                    )

        # 排序并显示前10个
        cause_gender_df = pd.DataFrame(cause_gender_diff)
        if not cause_gender_df.empty:
            cause_gender_df = cause_gender_df.sort_values("ratio", ascending=False)
            self.results["gender_by_cause"] = cause_gender_df

            for i, row in cause_gender_df.head(10).iterrows():
                print(f"   {row['cause'][:40]}: Ratio={row['ratio']:.2f}")

        return self.results["gender_ttest"]

    def age_group_anova(self):
        """
        不同年龄组死亡模式的单因素方差分析
        H0: 所有年龄组的平均死亡率相同
        H1: 至少有一个年龄组的平均死亡率不同

        Returns:
        --------
        dict
            ANOVA分析结果
        """
        print("\n" + "=" * 50)
        print("AGE GROUP ONE-WAY ANOVA")
        print("=" * 50)

        # 准备各年龄组的数据
        age_groups = self.data["age_group"].unique()
        age_data = [
            self.data[self.data["age_group"] == ag]["both_sexes"].values
            for ag in age_groups
        ]

        # ANOVA检验
        f_stat, p_value = f_oneway(*age_data)

        # 计算效应量 (eta-squared)
        grand_mean = self.data["both_sexes"].mean()
        ss_between = sum(
            [len(group) * (group.mean() - grand_mean) ** 2 for group in age_data]
        )
        ss_total = sum([(x - grand_mean) ** 2 for x in self.data["both_sexes"]])
        eta_squared = ss_between / ss_total if ss_total > 0 else 0

        self.results["age_anova"] = {
            "f_statistic": f_stat,
            "p_value": p_value,
            "eta_squared": eta_squared,
            "significant": p_value < 0.05,
            "interpretation": "Significant" if p_value < 0.05 else "Not significant",
        }

        print(f"📈 Test Results:")
        print(f"   F-statistic: {f_stat:.4f}")
        print(f"   P-value: {p_value:.6f}")
        print(f"   Eta-squared: {eta_squared:.4f}")
        print(
            f"   Result: {self.results['age_anova']['interpretation']} difference between age groups"
        )

        # 显示各年龄组的平均死亡数
        print("\n📊 Mean Deaths by Age Group:")
        age_means = self.data.groupby("age_group")["both_sexes"].agg(
            ["mean", "std", "count"]
        )
        age_means = age_means.sort_values("mean", ascending=False)
        for age, row in age_means.iterrows():
            print(f"   {age:15} Mean={row['mean']:>10,.1f}  Std={row['std']:>10,.1f}")

        # 如果ANOVA显著，进行事后检验
        if p_value < 0.05:
            print("\n📊 Post-hoc Analysis (Tukey HSD):")
            print("   Performing pairwise comparisons...")

            # 准备数据进行Tukey HSD
            tukey_data = pd.DataFrame(
                {"deaths": self.data["both_sexes"], "age_group": self.data["age_group"]}
            )

            tukey_result = pairwise_tukeyhsd(
                tukey_data["deaths"], tukey_data["age_group"], alpha=0.05
            )

            self.results["tukey_hsd"] = tukey_result

            # 显示显著的配对比较
            tukey_df = pd.DataFrame(
                data=tukey_result.summary().data[1:],
                columns=tukey_result.summary().data[0],
            )
            significant_pairs = tukey_df[tukey_df["reject"] == True]

            if not significant_pairs.empty:
                print("   Significant pairwise differences found:")
                for _, row in significant_pairs.head(5).iterrows():
                    print(
                        f"   {row['group1']} vs {row['group2']}: p={float(row['p-adj']):.4f}"
                    )

        return self.results["age_anova"]

    def correlation_analysis(self):
        """
        相关性分析
        分析年龄、性别与死亡率之间的相关性

        Returns:
        --------
        dict
            相关性分析结果
        """
        print("\n" + "=" * 50)
        print("CORRELATION ANALYSIS")
        print("=" * 50)

        # 为年龄组创建数值编码
        age_encoding = {
            "0-28 days": 0.04,  # ~0.04 years
            "1-59 months": 2.5,  # ~2.5 years
            "5-14": 9.5,
            "15-29": 22,
            "30-49": 39.5,
            "50-59": 54.5,
            "60-69": 64.5,
            "70+": 75,
        }

        self.data["age_numeric"] = self.data["age_group"].map(age_encoding)

        correlations = {}

        # 1. 年龄与死亡数的相关性
        r_age_deaths, p_age_deaths = pearsonr(
            self.data["age_numeric"].dropna(),
            self.data["both_sexes"].loc[self.data["age_numeric"].notna()],
        )
        correlations["age_vs_deaths"] = {
            "r": r_age_deaths,
            "p": p_age_deaths,
            "significant": p_age_deaths < 0.05,
        }

        # 2. 男性与女性死亡数的相关性
        r_male_female, p_male_female = pearsonr(self.data["male"], self.data["female"])
        correlations["male_vs_female"] = {
            "r": r_male_female,
            "p": p_male_female,
            "significant": p_male_female < 0.05,
        }

        # 3. 计算相关矩阵
        corr_matrix = self.data[["both_sexes", "male", "female", "age_numeric"]].corr()
        correlations["correlation_matrix"] = corr_matrix

        self.results["correlations"] = correlations

        # 打印结果
        print(f"📈 Correlation Results:")
        print(f"\n1. Age vs Deaths:")
        print(f"   Pearson r = {r_age_deaths:.4f}")
        print(f"   P-value = {p_age_deaths:.6f}")
        print(
            f"   Interpretation: {'Significant' if p_age_deaths < 0.05 else 'Not significant'} correlation"
        )

        print(f"\n2. Male vs Female Deaths:")
        print(f"   Pearson r = {r_male_female:.4f}")
        print(f"   P-value = {p_male_female:.6f}")
        print(
            f"   Interpretation: {'Significant' if p_male_female < 0.05 else 'Not significant'} correlation"
        )

        print(f"\n3. Correlation Matrix:")
        print(corr_matrix.round(3))

        return correlations

    def chi_square_test(self):
        """
        卡方独立性检验
        H0: 死因分布与年龄组独立
        H1: 死因分布与年龄组相关

        Returns:
        --------
        dict
            卡方检验结果
        """
        print("\n" + "=" * 50)
        print("CHI-SQUARE TEST OF INDEPENDENCE")
        print("=" * 50)

        # 创建列联表（死因 x 年龄组）
        contingency_table = pd.crosstab(
            self.data["cause_name"],
            self.data["age_group"],
            values=self.data["both_sexes"],
            aggfunc="sum",
        ).fillna(0)

        # 只保留死亡数较多的前20个死因（避免稀疏矩阵）
        top_causes = (
            self.data.groupby("cause_name")["both_sexes"].sum().nlargest(20).index
        )
        contingency_table = contingency_table.loc[top_causes]

        # 执行卡方检验
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)

        # 计算Cramér's V（效应量）
        n = contingency_table.sum().sum()
        min_dim = min(contingency_table.shape[0] - 1, contingency_table.shape[1] - 1)
        cramers_v = np.sqrt(chi2 / (n * min_dim)) if n > 0 and min_dim > 0 else 0

        self.results["chi_square"] = {
            "chi2_statistic": chi2,
            "p_value": p_value,
            "degrees_of_freedom": dof,
            "cramers_v": cramers_v,
            "significant": p_value < 0.05,
            "interpretation": "Dependent" if p_value < 0.05 else "Independent",
        }

        print(f"📈 Test Results:")
        print(f"   Chi-square statistic: {chi2:.4f}")
        print(f"   P-value: {p_value:.6f}")
        print(f"   Degrees of freedom: {dof}")
        print(f"   Cramér's V: {cramers_v:.4f}")
        print(
            f"   Result: Cause of death and age group are {self.results['chi_square']['interpretation']}"
        )

        # 显示对卡方贡献最大的组合
        if p_value < 0.05:
            # 计算标准化残差
            residuals = (contingency_table - expected) / np.sqrt(expected)

            print("\n📊 Top Contributing Cells (Standardized Residuals > 2):")
            # 找出贡献最大的单元格
            for cause in residuals.index:
                for age in residuals.columns:
                    if abs(residuals.loc[cause, age]) > 2:
                        print(
                            f"   {cause[:30]} x {age}: {residuals.loc[cause, age]:.2f}"
                        )

        return self.results["chi_square"]

    def run_all_tests(self):
        """
        运行所有统计检验

        Returns:
        --------
        dict
            所有检验结果的汇总
        """
        print("\n" + "=" * 60)
        print("  RUNNING ALL STATISTICAL TESTS")
        print("=" * 60)

        # 1. 描述性统计
        self.descriptive_statistics()

        # 2. 性别差异t检验
        self.gender_ttest()

        # 3. 年龄组ANOVA
        self.age_group_anova()

        # 4. 相关性分析
        self.correlation_analysis()

        # 5. 卡方检验
        self.chi_square_test()

        # 创建结果摘要
        print("\n" + "=" * 60)
        print("  STATISTICAL ANALYSIS SUMMARY")
        print("=" * 60)

        summary = pd.DataFrame(
            [
                {
                    "Test": "Gender T-Test",
                    "Statistic": self.results["gender_ttest"]["t_statistic"],
                    "P-Value": self.results["gender_ttest"]["p_value"],
                    "Effect Size": f"d={self.results['gender_ttest']['cohens_d']:.3f}",
                    "Result": self.results["gender_ttest"]["interpretation"],
                },
                {
                    "Test": "Age Group ANOVA",
                    "Statistic": self.results["age_anova"]["f_statistic"],
                    "P-Value": self.results["age_anova"]["p_value"],
                    "Effect Size": f"η²={self.results['age_anova']['eta_squared']:.3f}",
                    "Result": self.results["age_anova"]["interpretation"],
                },
                {
                    "Test": "Chi-Square Independence",
                    "Statistic": self.results["chi_square"]["chi2_statistic"],
                    "P-Value": self.results["chi_square"]["p_value"],
                    "Effect Size": f"V={self.results['chi_square']['cramers_v']:.3f}",
                    "Result": self.results["chi_square"]["interpretation"],
                },
            ]
        )

        print("\n📊 Statistical Test Results Summary:")
        print(summary.to_string(index=False))

        # 解释效应量
        print("\n📈 Effect Size Interpretation:")
        print("   Cohen's d: 0.2=small, 0.5=medium, 0.8=large")
        print("   Eta-squared: 0.01=small, 0.06=medium, 0.14=large")
        print("   Cramér's V: 0.1=small, 0.3=medium, 0.5=large")

        return self.results


# 主程序
if __name__ == "__main__":
    """
    直接运行此脚本进行统计分析
    """
    print("=" * 60)
    print("  WHO Mortality Statistical Analysis")
    print("=" * 60)

    try:
        # 加载处理后的数据
        data_path = "data/processed/who_mortality_clean.csv"

        if not pd.io.common.file_exists(data_path):
            print(f"❌ Error: Processed data not found at {data_path}")
            print("   Please run data_processing.py first.")
            exit(1)

        print(f"📂 Loading data from {data_path}...")
        data = pd.read_csv(data_path)
        print(f"✅ Loaded {len(data)} records")

        # 初始化统计分析器
        analyzer = ClassicalStatistics(data)

        # 运行所有检验
        results = analyzer.run_all_tests()

        print("\n" + "=" * 60)
        print("  ANALYSIS COMPLETE")
        print("=" * 60)
        print("\n✅ All statistical tests completed successfully!")
        print("   Results have been stored in the analyzer.results dictionary")

    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback

        traceback.print_exc()

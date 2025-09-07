"""
经典统计分析模块
执行t检验、ANOVA、相关性分析等
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import warnings

warnings.filterwarnings("ignore")


class ClassicalStatistics:
    """经典统计分析类"""

    def __init__(self, data):
        self.data = data
        self.results = {}

    def gender_ttest(self):
        """
        性别差异的配对t检验
        H0: 男性和女性的死亡率没有显著差异
        """
        print("=== Gender Difference T-Test ===")

        # 整体t检验
        t_stat, p_value = stats.ttest_rel(self.data["male"], self.data["female"])

        self.results["gender_ttest"] = {
            "t_statistic": t_stat,
            "p_value": p_value,
            "significant": p_value < 0.05,
        }

        print(f"T-statistic: {t_stat:.4f}")
        print(f"P-value: {p_value:.4f}")
        print(
            f"Result: {'Significant' if p_value < 0.05 else 'Not significant'} difference between genders"
        )

        # 按死因分组的t检验
        cause_tests = []
        for cause in self.data["cause_name"].unique():
            cause_data = self.data[self.data["cause_name"] == cause]
            if len(cause_data) > 1:
                t, p = stats.ttest_rel(cause_data["male"], cause_data["female"])
                cause_tests.append(
                    {
                        "cause": cause,
                        "t_statistic": t,
                        "p_value": p,
                        "significant": p < 0.05,
                    }
                )

        self.results["gender_by_cause"] = pd.DataFrame(cause_tests)

        return self.results["gender_ttest"]

    def age_group_anova(self):
        """
        不同年龄组死亡模式的ANOVA分析
        H0: 所有年龄组的平均死亡率相同
        """
        print("\n=== Age Group ANOVA ===")

        # 准备数据
        age_groups = self.data["age_group"].unique()
        age_data = [
            self.data[self.data["age_group"] == ag]["both_sexes"].values
            for ag in age_groups
        ]

        # ANOVA检验
        f_stat, p_value = stats.f_oneway(*age_data)

        self.results["age_anova"] = {
            "f_statistic": f_stat,
            "p_value": p_value,
            "significant": p_value < 0.05,
        }

        print(f"F-statistic: {f_stat:.4f}")
        print(f"P-value: {p_value:.4f}")
        print(
            f"Result: {'Significant' if p_value < 0.05 else 'Not significant'} difference between age groups"
        )

        # 事后检验（Tukey HSD）
        if p_value < 0.05:
            print("\nPerforming Tukey HSD post-hoc test...")
            data_for_tukey = pd.DataFrame(
                {"deaths": self.data["both_sexes"], "age_group": self.data["age_group"]}
            )

            tukey_result = pairwise_tukeyhsd(
                data_for_tukey["deaths"], data_for_tukey["age_group"], alpha=0.05
            )

            self.results["tukey_hsd"] = tukey_result
            print(tukey_result)

        return self.results["age_anova"]

    def correlation_analysis(self):
        """
        相关性分析
        """
        print("\n=== Correlation Analysis ===")

        # 年龄编码（用于相关性计算）
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

        # 计算相关系数
        correlations = {
            "age_vs_deaths": stats.pearsonr(
                self.data["age_numeric"], self.data["both_sexes"]
            ),
            "male_vs_female": stats.pearsonr(self.data["male"], self.data["female"]),
        }

        self.results["correlations"] = correlations

        for name, (r, p) in correlations.items():
            print(f"{name}: r={r:.4f}, p={p:.4f}")

        return correlations

    def chi_square_test(self):
        """
        卡方检验：死因分布是否与年龄组独立
        """
        print("\n=== Chi-Square Test of Independence ===")

        # 创建列联表
        contingency_table = pd.crosstab(
            self.data["cause_name"],
            self.data["age_group"],
            values=self.data["both_sexes"],
            aggfunc="sum",
        ).fillna(0)

        # 执行卡方检验
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

        self.results["chi_square"] = {
            "chi2_statistic": chi2,
            "p_value": p_value,
            "degrees_of_freedom": dof,
            "significant": p_value < 0.05,
        }

        print(f"Chi-square statistic: {chi2:.4f}")
        print(f"P-value: {p_value:.4f}")
        print(f"Degrees of freedom: {dof}")
        print(
            f"Result: {'Dependent' if p_value < 0.05 else 'Independent'} relationship"
        )

        return self.results["chi_square"]

    def run_all_tests(self):
        """
        运行所有统计检验
        """
        print("=" * 50)
        print("RUNNING ALL STATISTICAL TESTS")
        print("=" * 50)

        self.gender_ttest()
        self.age_group_anova()
        self.correlation_analysis()
        self.chi_square_test()

        return self.results


# 主程序
if __name__ == "__main__":
    # 加载处理后的数据
    data = pd.read_csv("data/processed/who_mortality_clean.csv")

    # 初始化统计分析
    stats_analyzer = ClassicalStatistics(data)

    # 运行所有检验
    results = stats_analyzer.run_all_tests()

    # 保存结果
    print("\n" + "=" * 50)
    print("ANALYSIS COMPLETE")
    print("=" * 50)

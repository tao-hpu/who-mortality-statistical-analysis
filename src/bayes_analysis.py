"""
简化版贝叶斯统计分析模块
避免复杂依赖，使用更稳定的方法
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sys
import os
sys.path.append(os.path.dirname(__file__))
from config import AGE_GROUP_ORDER, AGE_ENCODING, DATA_PATH

warnings.filterwarnings("ignore")


class SimpleBayesianStatistics:
    """
    简化的贝叶斯统计分析类
    使用共轭先验和解析解避免MCMC采样
    """

    def __init__(self, data):
        """初始化贝叶斯统计分析器"""
        self.data = data.copy()
        self.results = {}

    def bayesian_mean_estimate(self, values, prior_strength=1.0):
        """
        使用共轭先验的贝叶斯均值估计
        假设正态分布的数据，使用正态-逆伽马共轭先验
        """
        n = len(values)
        sample_mean = np.mean(values)
        sample_var = np.var(values, ddof=1)

        # 先验参数
        prior_mean = sample_mean  # 无信息先验
        prior_precision = prior_strength / sample_var

        # 后验参数
        post_precision = prior_precision + n
        post_mean = (prior_precision * prior_mean + n * sample_mean) / post_precision
        post_var = 1 / post_precision + sample_var / n

        # 95% 可信区间
        ci_lower = post_mean - 1.96 * np.sqrt(post_var)
        ci_upper = post_mean + 1.96 * np.sqrt(post_var)

        return {
            'mean': post_mean,
            'std': np.sqrt(post_var),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }

    def descriptive_statistics(self):
        """执行贝叶斯描述性统计分析"""
        print("\n--- 贝叶斯描述性统计分析 ---")

        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()

        for col in numeric_cols[:5]:  # 限制分析前5个变量
            print(f"\n分析变量: {col}")

            values = self.data[col].dropna().values
            if len(values) == 0:
                continue

            result = self.bayesian_mean_estimate(values)

            self.results[f"{col}_descriptive"] = result

            print(f"后验均值: {result['mean']:.4f} (SD: {result['std']:.4f})")
            print(f"95% 可信区间: [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")

    def bayesian_ttest(self, col1, col2):
        """
        使用贝叶斯因子的简化贝叶斯t检验
        """
        print(f"\n--- 贝叶斯t检验: {col1} vs {col2} ---")

        group1 = self.data[col1].dropna().values
        group2 = self.data[col2].dropna().values

        # 计算基本统计量
        n1, n2 = len(group1), len(group2)
        mean1, mean2 = np.mean(group1), np.mean(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

        # 合并方差
        pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        se_diff = np.sqrt(pooled_var * (1/n1 + 1/n2))

        # 效应量
        mean_diff = mean2 - mean1
        cohens_d = mean_diff / np.sqrt(pooled_var)

        # 贝叶斯因子近似（使用BIC近似）
        t_stat = mean_diff / se_diff
        df = n1 + n2 - 2

        # 计算p值用于参考
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))

        # BIC近似的贝叶斯因子
        bic_h0 = n1 + n2 * np.log(2 * np.pi * pooled_var)
        bic_h1 = bic_h0 - t_stat**2
        bf_10 = np.exp((bic_h0 - bic_h1) / 2)

        self.results[f"{col1}_vs_{col2}_ttest"] = {
            'group1_mean': mean1,
            'group2_mean': mean2,
            'mean_diff': mean_diff,
            'cohens_d': cohens_d,
            'bayes_factor': bf_10,
            'p_value': p_value
        }

        print(f"{col1} 均值: {mean1:.4f}")
        print(f"{col2} 均值: {mean2:.4f}")
        print(f"均值差异: {mean_diff:.4f}")
        print(f"效应量 (Cohen's d): {cohens_d:.4f}")
        print(f"贝叶斯因子 (BF10): {bf_10:.4f}")
        print(f"传统p值: {p_value:.4f}")

    def bayesian_correlation(self, var1, var2):
        """贝叶斯相关性分析"""
        print(f"\n--- 贝叶斯相关性分析: {var1} vs {var2} ---")

        # 移除缺失值
        data = self.data[[var1, var2]].dropna()
        x = data[var1].values
        y = data[var2].values

        if len(x) < 3:
            print("数据点太少，无法计算相关性")
            return

        # 计算相关系数
        r, p_value = stats.pearsonr(x, y)
        n = len(x)

        # Fisher z变换用于区间估计
        z = np.arctanh(r)
        se_z = 1 / np.sqrt(n - 3)

        # 95% 可信区间
        z_lower = z - 1.96 * se_z
        z_upper = z + 1.96 * se_z
        r_lower = np.tanh(z_lower)
        r_upper = np.tanh(z_upper)

        # 贝叶斯因子（Jeffreys, 1961）
        bf_10 = (1 + r**2)**(-(n-1)/2) / stats.beta.pdf((r+1)/2, 0.5, 0.5)

        self.results[f"{var1}_vs_{var2}_correlation"] = {
            'correlation': r,
            'ci_lower': r_lower,
            'ci_upper': r_upper,
            'bayes_factor': bf_10,
            'p_value': p_value
        }

        print(f"相关系数: {r:.4f}")
        print(f"95% 可信区间: [{r_lower:.4f}, {r_upper:.4f}]")
        print(f"贝叶斯因子 (BF10): {bf_10:.4f}")
        print(f"传统p值: {p_value:.4f}")

    def bayesian_anova(self, group_col, value_col):
        """
        贝叶斯ANOVA分析
        使用贝叶斯因子比较多组均值
        """
        print(f"\n--- 贝叶斯ANOVA: {value_col} across {group_col} ---")

        # 准备数据
        data = self.data[[group_col, value_col]].dropna()
        groups = data[group_col].unique()

        if len(groups) < 2:
            print("需要至少2个组进行ANOVA分析")
            return

        # 收集各组数据
        group_data = []
        group_means = []
        group_vars = []
        group_ns = []

        for group in groups:
            group_values = data[data[group_col] == group][value_col].values
            if len(group_values) > 1:
                group_data.append(group_values)
                group_means.append(np.mean(group_values))
                group_vars.append(np.var(group_values, ddof=1))
                group_ns.append(len(group_values))

        if len(group_data) < 2:
            print("没有足够的有效组进行分析")
            return

        # 传统ANOVA
        f_stat, p_value = stats.f_oneway(*group_data)

        # 计算效应量 (eta-squared)
        grand_mean = np.mean(np.concatenate(group_data))
        ss_between = sum(n * (mean - grand_mean)**2 for n, mean in zip(group_ns, group_means))
        ss_total = sum(np.sum((data - grand_mean)**2) for data in group_data)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0

        # 计算贝叶斯因子 (使用BIC近似)
        n_total = sum(group_ns)
        k = len(group_data)  # 组数

        # BIC for null model (所有组均值相等)
        pooled_var = sum((n-1)*var for n, var in zip(group_ns, group_vars)) / (n_total - k)
        bic_null = n_total * np.log(2 * np.pi * pooled_var) + n_total

        # BIC for alternative model (组均值不同)
        bic_alt = sum(n * np.log(2 * np.pi * var) + n for n, var in zip(group_ns, group_vars))

        # 贝叶斯因子 (限制最大值以避免inf)
        diff = (bic_null - bic_alt) / 2
        if diff > 500:  # 避免exp overflow
            bf_10 = 1e10
        else:
            bf_10 = np.exp(diff)

        # 保存结果
        self.results[f"{group_col}_{value_col}_anova"] = {
            'f_statistic': f_stat,
            'p_value': p_value,
            'eta_squared': eta_squared,
            'bayes_factor': bf_10,
            'n_groups': k,
            'group_means': {str(g): m for g, m in zip(groups[:len(group_means)], group_means)}
        }

        print(f"F统计量: {f_stat:.4f}")
        print(f"p值: {p_value:.4f}")
        print(f"效应量 (η²): {eta_squared:.4f}")
        print(f"贝叶斯因子 (BF10): {bf_10:.4f}")
        print(f"\n组均值:")
        for group, mean in zip(groups[:len(group_means)], group_means):
            print(f"  {group}: {mean:.4f}")

        # 解释贝叶斯因子
        if bf_10 > 10:
            print("\n强证据支持组间差异")
        elif bf_10 > 3:
            print("\n中等证据支持组间差异")
        elif bf_10 > 1:
            print("\n弱证据支持组间差异")
        else:
            print("\n证据支持组间无差异")

    def bayesian_regression(self, x_cols, y_col):
        """
        贝叶斯线性回归分析
        使用共轭先验的解析解
        """
        print(f"\n--- 贝叶斯线性回归: {y_col} ~ {x_cols} ---")

        # 准备数据
        if isinstance(x_cols, str):
            x_cols = [x_cols]

        data = self.data[x_cols + [y_col]].dropna()

        if len(data) < len(x_cols) + 2:
            print("数据点太少，无法进行回归分析")
            return

        X = data[x_cols].values
        y = data[y_col].values

        # 添加截距项
        X = np.column_stack([np.ones(len(X)), X])

        # 使用正态-逆伽马共轭先验
        # 先验参数
        n, p = X.shape
        alpha_0 = 1.0  # 逆伽马分布的形状参数
        beta_0 = 1.0   # 逆伽马分布的尺度参数
        mu_0 = np.zeros(p)  # 系数的先验均值
        lambda_0 = np.eye(p) * 0.01  # 先验精度矩阵（低精度=无信息先验）

        # 后验参数
        lambda_n = lambda_0 + X.T @ X
        lambda_n_inv = np.linalg.inv(lambda_n)
        mu_n = lambda_n_inv @ (lambda_0 @ mu_0 + X.T @ y)
        alpha_n = alpha_0 + n / 2
        beta_n = beta_0 + 0.5 * (y.T @ y + mu_0.T @ lambda_0 @ mu_0 - mu_n.T @ lambda_n @ mu_n)

        # 后验均值和标准差
        coefficients = mu_n
        sigma2_posterior = beta_n / alpha_n  # 方差的后验均值
        coef_std = np.sqrt(np.diag(lambda_n_inv) * sigma2_posterior)

        # 计算R²和调整R²
        y_pred = X @ coefficients
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p else 0

        # 计算贝叶斯信息准则 (BIC)
        log_likelihood = -n/2 * np.log(2*np.pi*sigma2_posterior) - ss_res/(2*sigma2_posterior)
        bic = -2 * log_likelihood + p * np.log(n)

        # 95% 可信区间
        ci_lower = coefficients - 1.96 * coef_std
        ci_upper = coefficients + 1.96 * coef_std

        # 保存结果
        coef_names = ['Intercept'] + x_cols
        self.results[f"{y_col}_regression"] = {
            'coefficients': {name: coef for name, coef in zip(coef_names, coefficients)},
            'std_errors': {name: std for name, std in zip(coef_names, coef_std)},
            'ci_lower': {name: ci_l for name, ci_l in zip(coef_names, ci_lower)},
            'ci_upper': {name: ci_u for name, ci_u in zip(coef_names, ci_upper)},
            'r_squared': r2,
            'adj_r_squared': adj_r2,
            'bic': bic,
            'sigma_squared': sigma2_posterior
        }

        print(f"\n回归系数:")
        for name, coef, std, ci_l, ci_u in zip(coef_names, coefficients, coef_std, ci_lower, ci_upper):
            print(f"  {name}: {coef:.4f} ± {std:.4f}")
            print(f"    95% CI: [{ci_l:.4f}, {ci_u:.4f}]")

        print(f"\nR²: {r2:.4f}")
        print(f"调整R²: {adj_r2:.4f}")
        print(f"BIC: {bic:.2f}")
        print(f"残差方差: {sigma2_posterior:.4f}")

    def bayesian_chisquare(self, cause_col='cause_name', age_col='age_group', count_col='both_sexes'):
        """简化的贝叶斯卡方检验"""
        print("\n--- 贝叶斯卡方检验: 死因 vs 年龄组 ---")

        try:
            # 创建列联表
            contingency_table = pd.crosstab(
                self.data[cause_col],
                self.data[age_col],
                values=self.data[count_col],
                aggfunc='sum'
            ).fillna(0)

            # 限制大小以避免计算过慢
            if contingency_table.shape[0] > 10:
                contingency_table = contingency_table.iloc[:10, :]
            if contingency_table.shape[1] > 10:
                contingency_table = contingency_table.iloc[:, :10]

            # 移除全零的行和列
            contingency_table = contingency_table.loc[(contingency_table != 0).any(axis=1), :]
            contingency_table = contingency_table.loc[:, (contingency_table != 0).any(axis=0)]

            # 添加小的伪计数避免零频数
            contingency_table = contingency_table + 0.5

            print(f"列联表大小: {contingency_table.shape}")

            # 传统卡方检验
            chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table.values)

            # 计算贝叶斯因子（使用BIC近似）
            n = contingency_table.values.sum()
            bic_independence = -2 * np.log(p_value) + dof * np.log(n)
            bic_dependence = chi2_stat - dof * np.log(n)
            diff = (bic_independence - bic_dependence) / 2
            if diff > 500:  # 避免exp overflow
                bf_10 = 1e10
            else:
                bf_10 = np.exp(diff)

            # Cramér's V 作为效应量
            cramers_v = np.sqrt(chi2_stat / (n * min(contingency_table.shape[0] - 1,
                                                     contingency_table.shape[1] - 1)))

            self.results['chisquare_test'] = {
                'chi2_statistic': chi2_stat,
                'p_value': p_value,
                'cramers_v': cramers_v,
                'bayes_factor': bf_10,
                'degrees_of_freedom': dof
            }

            print(f"卡方统计量: {chi2_stat:.4f}")
            print(f"自由度: {dof}")
            print(f"p值: {p_value:.4f}")
            print(f"Cramér's V: {cramers_v:.4f}")
            print(f"贝叶斯因子 (BF10): {bf_10:.4f}")

            # 解释贝叶斯因子
            if bf_10 > 10:
                print("强证据支持关联性")
            elif bf_10 > 3:
                print("中等证据支持关联性")
            elif bf_10 > 1:
                print("弱证据支持关联性")
            else:
                print("证据支持独立性")

        except Exception as e:
            print(f"卡方检验失败: {e}")

    def run_all_tests(self):
        """运行所有贝叶斯统计检验"""
        print("\n" + "="*60)
        print("开始简化贝叶斯统计分析")
        print("="*60)

        # 1. 描述性统计
        self.descriptive_statistics()

        # 2. t检验
        if 'male' in self.data.columns and 'female' in self.data.columns:
            self.bayesian_ttest('male', 'female')

        # 3. ANOVA分析
        if 'age_group' in self.data.columns and 'both_sexes' in self.data.columns:
            self.bayesian_anova('age_group', 'both_sexes')

        # 4. 相关性分析（限制数量）
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 2:
            self.bayesian_correlation(numeric_cols[0], numeric_cols[1])

        # 5. 回归分析
        if 'age_numeric' in self.data.columns and 'both_sexes' in self.data.columns:
            self.bayesian_regression(['age_numeric'], 'both_sexes')
        elif len(numeric_cols) >= 2:
            # 使用前两个数值列进行演示
            self.bayesian_regression([numeric_cols[0]], numeric_cols[1])

        # 6. 卡方检验
        if all(col in self.data.columns for col in ['cause_name', 'age_group', 'both_sexes']):
            self.bayesian_chisquare()

        return self.results


if __name__ == "__main__":
    try:
        # 检查数据文件
        if not os.path.exists(DATA_PATH):
            print("Error: Processed data not found at", DATA_PATH)
            print("Please run dataprocessing.py first.")
            exit(1)

        print(f"Loading data from {DATA_PATH}...")
        data = pd.read_csv(DATA_PATH)
        print(f"Loaded {len(data)} records")

        # 添加age_numeric编码
        from config import AGE_ENCODING
        if 'age_group' in data.columns:
            data['age_numeric'] = data['age_group'].map(AGE_ENCODING)
            print("Added age_numeric encoding")

        # 初始化分析器
        analyzer = SimpleBayesianStatistics(data)

        # 运行所有检验
        results = analyzer.run_all_tests()

        print("\n" + "="*60)
        print("简化贝叶斯分析完成")
        print("="*60)
        print("\n分析结果已保存在 analyzer.results 字典中")

        # 打印结果摘要
        print("\n--- 结果摘要 ---")
        for key, value in results.items():
            print(f"\n{key}:")
            if isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, (int, float)):
                        print(f"  {k}: {v:.4f}")
                    else:
                        print(f"  {k}: {v}")

    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()
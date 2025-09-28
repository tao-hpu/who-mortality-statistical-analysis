"""
贝叶斯统计分析模块
Bayesian Statistical Analysis Module

主要功能:
1. 贝叶斯描述性统计分析
2. 贝叶斯假设检验（贝叶斯t检验、ANOVA、卡方检验）
3. 贝叶斯相关性分析
4. 贝叶斯回归分析
"""

import pandas as pd
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy import stats


warnings.filterwarnings("ignore")


class BayesianStatistics:
    """
    贝叶斯统计分析类
    用于执行各种贝叶斯统计检验和分析
    """

    def __init__(self, data):
        """
        初始化贝叶斯统计分析器

        Parameters:
        -----------
        data : pd.DataFrame
            处理后的WHO死亡率数据
        """
        self.data = data.copy()
        self.results = {}
        self.trace = {}
        self.idata = None

    def descriptive_statistics(self):
        """
        执行贝叶斯描述性统计分析
        """
        print("\n--- 贝叶斯描述性统计分析 ---")

        # 获取数值型列
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()

        for col in numeric_cols:
            print(f"\n分析变量: {col}")

            # 移除缺失值
            values = self.data[col].dropna().values

            # 贝叶斯模型估计均值和标准差
            with pm.Model() as model:
                # 先验分布
                mu = pm.Normal('mu', mu=np.mean(values), sigma=np.std(values)*10)
                sigma = pm.HalfNormal('sigma', sigma=np.std(values)*10)

                # 似然函数
                pm.Normal('obs', mu=mu, sigma=sigma, observed=values)

                # MCMC采样 - 减少采样量以提高速度
                trace = pm.sample(500, tune=500, cores=1, random_seed=42)

            # 获取摘要统计
            summary = az.summary(trace, var_names=["mu", "sigma"])

            # 存储结果
            self.results[f"{col}_descriptive"] = {
                "mean": summary.loc["mu", "mean"],
                "sd": summary.loc["mu", "sd"],
                "hdi_3%": summary.loc["mu", "hdi_3%"],
                "hdi_97%": summary.loc["mu", "hdi_97%"],
                "sigma_mean": summary.loc["sigma", "mean"],
                "sigma_sd": summary.loc["sigma", "sd"]
            }

            # 打印结果
            print(f"均值: {summary.loc['mu', 'mean']:.4f} (SD: {summary.loc['mu', 'sd']:.4f})")
            print(f"95% HDI: [{summary.loc['mu', 'hdi_3%']:.4f}, {summary.loc['mu', 'hdi_97%']:.4f}]")
            print(f"标准差: {summary.loc['sigma', 'mean']:.4f} (SD: {summary.loc['sigma', 'sd']:.4f})")

    def bayesian_ttest(self, col1, col2):
        """
        执行贝叶斯独立样本t检验，直接比较两列数据

        Parameters:
        -----------
        col1 : str
            第一列数据列名
        col2 : str
            第二列数据列名
        """
        print(f"\n--- 贝叶斯t检验: {col1} vs {col2} ---")

        # 获取两列数据
        group1 = self.data[col1].dropna().values
        group2 = self.data[col2].dropna().values

        # 贝叶斯t检验模型
        with pm.Model() as model:
            # 先验分布
            mu1 = pm.Normal('mu1', mu=np.mean(group1), sigma=np.std(group1)*10)
            mu2 = pm.Normal('mu2', mu=np.mean(group2), sigma=np.std(group2)*10)
            sigma1 = pm.HalfNormal('sigma1', sigma=np.std(group1)*10)
            sigma2 = pm.HalfNormal('sigma2', sigma=np.std(group2)*10)

            # 效应量
            effect_size = pm.Deterministic('effect_size', (mu2 - mu1) / np.sqrt((sigma1**2 + sigma2**2) / 2))

            # 均值差异
            mean_diff = pm.Deterministic('mean_diff', mu2 - mu1)

            # 似然函数
            pm.Normal('group1', mu=mu1, sigma=sigma1, observed=group1)
            pm.Normal('group2', mu=mu2, sigma=sigma2, observed=group2)

            # MCMC采样 - 减少采样量以提高速度
            trace = pm.sample(500, tune=500, cores=1, random_seed=42)

        # 获取摘要统计
        summary = az.summary(trace, var_names=["mu1", "mu2", "effect_size", "mean_diff"])

        # 存储结果
        self.results[f"{col1}_vs_{col2}_ttest"] = {
            "group1_mean": summary.loc["mu1", "mean"],
            "group2_mean": summary.loc["mu2", "mean"],
            "mean_diff": summary.loc["mean_diff", "mean"],
            "mean_diff_hdi_3%": summary.loc["mean_diff", "hdi_3%"],
            "mean_diff_hdi_97%": summary.loc["mean_diff", "hdi_97%"],
            "effect_size": summary.loc["effect_size", "mean"],
            "effect_size_hdi_3%": summary.loc["effect_size", "hdi_3%"],
            "effect_size_hdi_97%": summary.loc["effect_size", "hdi_97%"]
        }

        # 打印结果
        print(f"{col1} 均值: {summary.loc['mu1', 'mean']:.4f}")
        print(f"{col2} 均值: {summary.loc['mu2', 'mean']:.4f}")
        print(f"均值差异: {summary.loc['mean_diff', 'mean']:.4f}")
        print(f"均值差异 95% HDI: [{summary.loc['mean_diff', 'hdi_3%']:.4f}, {summary.loc['mean_diff', 'hdi_97%']:.4f}]")
        print(f"效应量 (Cohen's d): {summary.loc['effect_size', 'mean']:.4f}")
        print(f"效应量 95% HDI: [{summary.loc['effect_size', 'hdi_3%']:.4f}, {summary.loc['effect_size', 'hdi_97%']:.4f}]")

        # 计算效应量大于0的概率
        prob_positive = np.mean(trace.posterior['effect_size'].values > 0)
        print(f"效应量 > 0 的概率: {prob_positive:.4f}")

        # 绘制效应量的后验分布
        az.plot_posterior(trace, var_names=["effect_size", "mean_diff"])
        plt.suptitle(f"{col1} vs {col2} 的贝叶斯t检验结果", y=1.02)
        plt.tight_layout()
        plt.show()

    def bayesian_anova(self, group_col, value_col):
        """
        执行贝叶斯ANOVA

        Parameters:
        -----------
        group_col : str
            分组变量列名
        value_col : str
            数值变量列名
        """
        print(f"\n--- 贝叶斯ANOVA: {value_col} by {group_col} ---")

        # 获取各组数据
        groups = self.data[group_col].unique()
        group_data = [self.data[self.data[group_col] == g][value_col].dropna().values for g in groups]

        # 贝叶斯ANOVA模型
        with pm.Model() as model:
            # 超先验
            mu_global = pm.Normal('mu_global', mu=np.mean(self.data[value_col].dropna()),
                                  sigma=np.std(self.data[value_col].dropna())*10)
            sigma_global = pm.HalfNormal('sigma_global', sigma=np.std(self.data[value_col].dropna())*10)

            # 各组参数
            mu = pm.Normal('mu', mu=mu_global, sigma=sigma_global, shape=len(groups))
            sigma = pm.HalfNormal('sigma', sigma=sigma_global, shape=len(groups))

            # 似然函数
            for i, data in enumerate(group_data):
                pm.Normal(f'group_{i}', mu=mu[i], sigma=sigma[i], observed=data)

            # MCMC采样 - 减少采样量以提高速度
            trace = pm.sample(500, tune=500, cores=1, random_seed=42)

        # 获取摘要统计
        summary = az.summary(trace, var_names=["mu"])

        # 存储结果
        anova_results = {}
        for i, group in enumerate(groups):
            anova_results[f"group_{group}_mean"] = summary.loc[f"mu[{i}]", "mean"]
            anova_results[f"group_{group}_hdi_3%"] = summary.loc[f"mu[{i}]", "hdi_3%"]
            anova_results[f"group_{group}_hdi_97%"] = summary.loc[f"mu[{i}]", "hdi_97%"]

        self.results[f"{value_col}_by_{group_col}_anova"] = anova_results

        # 打印结果
        for i, group in enumerate(groups):
            print(f"{group} 均值: {summary.loc[f'mu[{i}]', 'mean']:.4f}")
            print(f"95% HDI: [{summary.loc[f'mu[{i}]', 'hdi_3%']:.4f}, {summary.loc[f'mu[{i}]', 'hdi_97%']:.4f}]")

        # 绘制各组均值的后验分布
        az.plot_forest(trace, var_names=["mu"])
        plt.title(f"{value_col} 在 {group_col} 各组均值的后验分布")
        plt.show()

    def bayesian_chisquare(self, cause_col='cause_name', age_col='age_group', count_col='both_sexes'):
        """
        执行贝叶斯卡方检验，使用分层模型优化
        修复了 log_likelihood 缺失、收敛性差和计算速度慢的问题。

        Parameters:
        -----------
        cause_col : str
            死因列名
        age_col : str
            年龄组列名
        count_col : str
            死亡人数列名

        Returns:
        --------
        dict
            包含独立性检验结果的字典
        """
        print("\n--- 贝叶斯卡方检验: 死因 vs 年龄组 ---")
        print("H0: 死因分布与年龄组独立")
        print("H1: 死因分布与年龄组相关")

        # 准备数据 - 创建列联表
        contingency_table = pd.crosstab(
            self.data[cause_col],
            self.data[age_col],
            values=self.data[count_col],
            aggfunc='sum'
        ).fillna(0)

        print("\n观测到的死亡人数列联表:")
        print(contingency_table)

        # 获取维度信息
        n_causes, n_age_groups = contingency_table.shape
        total_observed = contingency_table.values.sum()

        # 计算行和列的边际和
        cause_sums = contingency_table.sum(axis=1).values
        age_sums = contingency_table.sum(axis=0).values

        # 计算期望频数
        expected = np.outer(cause_sums, age_sums) / total_observed

        # --- 修复1: 创建独立模型并确保存储log_likelihood ---
        print("\n正在采样独立模型...")
        with pm.Model() as independence_model:
            # 超参数 - 控制行和列效应的变异程度
            row_sd = pm.HalfNormal('row_sd', sigma=1.0)
            col_sd = pm.HalfNormal('col_sd', sigma=1.0)

            # 非中心化参数化
            row_raw = pm.Normal('row_raw', mu=0, sigma=1, shape=n_causes)
            col_raw = pm.Normal('col_raw', mu=0, sigma=1, shape=n_age_groups)

            # 行和列效应
            alpha_row = pm.Deterministic('alpha_row', row_raw * row_sd)
            alpha_col = pm.Deterministic('alpha_col', col_raw * col_sd)

            # 构建独立模型
            log_expected = pm.math.log(expected)
            log_rates = log_expected + alpha_row[:, None] + alpha_col[None, :]
            rates = pm.math.exp(log_rates)

            # 定义似然
            obs = pm.Poisson('obs', mu=rates, observed=contingency_table.values)

            pm.Deterministic('log_likelihood', pm.logp(obs, contingency_table.values))

            trace_independence = pm.sample(
                tune=500,
                draws=500,
                chains=2,
                cores=1,
                target_accept=0.95,
                random_seed=42
            )

        # --- 修复3: 创建交互模型并确保存储log_likelihood ---
        print("\n正在采样交互模型...")
        with pm.Model() as interaction_model:
            # 超参数
            row_sd = pm.HalfNormal('row_sd', sigma=1.0)
            col_sd = pm.HalfNormal('col_sd', sigma=1.0)
            interaction_sd = pm.HalfNormal('interaction_sd', sigma=0.5)

            # 非中心化参数化
            row_raw = pm.Normal('row_raw', mu=0, sigma=1, shape=n_causes)
            col_raw = pm.Normal('col_raw', mu=0, sigma=1, shape=n_age_groups)

            # 低秩交互项 - 使用秩为1的近似
            z_cause = pm.Normal('z_cause', mu=0, sigma=1, shape=n_causes)
            z_age = pm.Normal('z_age', mu=0, sigma=1, shape=n_age_groups)

            # 行和列效应
            alpha_row = pm.Deterministic('alpha_row', row_raw * row_sd)
            alpha_col = pm.Deterministic('alpha_col', col_raw * col_sd)

            # 交互效应
            interaction = pm.Deterministic('interaction', interaction_sd * pt.outer(z_cause, z_age))

            # 构建交互模型
            log_expected = pm.math.log(expected)
            log_rates = log_expected + alpha_row[:, None] + alpha_col[None, :] + interaction
            rates = pm.math.exp(log_rates)

            # 定义似然
            obs = pm.Poisson('obs', mu=rates, observed=contingency_table.values)

            # --- 关键修复：为模型比较添加log_likelihood ---
            pm.Deterministic('log_likelihood', pm.logp(obs, contingency_table.values))

            # --- 修复4: 增加采样量以提高收敛性 ---
            trace_interaction = pm.sample(
                tune=500,
                draws=500,
                chains=2,
                cores=1,
                target_accept=0.95,
                random_seed=42
            )

        # 计算降维比例
        original_params = n_causes * n_age_groups
        reduced_params = n_causes + n_age_groups
        reduction_ratio = (1 - reduced_params / original_params) * 100

        print(f"\n降维效果:")
        print(f"原始参数数量: {original_params}")
        print(f"降维后参数数量: {reduced_params}")
        print(f"降维比例: {reduction_ratio:.1f}%")

        # 存储结果
        test_name = f"chisquare_{cause_col}_vs_{age_col}"
        self.trace[test_name] = {
            'independence': trace_independence,
            'interaction': trace_interaction
        }

        print("\n进行模型比较...")
        model_compare = az.compare({
            'independence': trace_independence,
            'interaction': trace_interaction
        }, ic='loo', var_name='log_likelihood')

        print("\n模型比较结果:")
        print(model_compare)

        # 计算后验预测检查
        print("\n计算后验预测检查...")
        with independence_model:
            # 生成后验预测样本
            ppc_independence = pm.sample_posterior_predictive(trace_independence, var_names=['obs'])

        with interaction_model:
            # 生成后验预测样本
            ppc_interaction = pm.sample_posterior_predictive(trace_interaction, var_names=['obs'])

        # 计算残差
        posterior_mean_independence = ppc_independence.posterior_predictive['obs'].mean(("chain", "draw")).values
        posterior_mean_interaction = ppc_interaction.posterior_predictive['obs'].mean(("chain", "draw")).values

        residuals_independence = contingency_table.values - posterior_mean_independence
        residuals_interaction = contingency_table.values - posterior_mean_interaction

        # 准备结果
        results = {
            'model_comparison': model_compare.to_dict(),
            'reduction_ratio': reduction_ratio,
            'residuals_independence': residuals_independence,
            'residuals_interaction': residuals_interaction,
            'observed': contingency_table.values,
            'expected_independence': posterior_mean_independence,
            'expected_interaction': posterior_mean_interaction,
            'traditional_chi2': {
                'statistic': chi2_stat,
                'df': chi2_df,
                'pvalue': chi2_pvalue
            }
        }

        self.results[test_name] = results

        # 可视化结果
        plt.figure(figsize=(15, 10))

        # 1. 观测值 vs 独立模型预测值
        plt.subplot(2, 2, 1)
        plt.scatter(contingency_table.values.flatten(), posterior_mean_independence.flatten(), alpha=0.5)
        plt.plot([0, contingency_table.values.max()], [0, contingency_table.values.max()], 'r--')
        plt.xlabel('Observed')
        plt.ylabel('Predicted (Independence Model)')
        plt.title('Observed vs Predicted (Independence)')

        # 2. 观测值 vs 交互模型预测值
        plt.subplot(2, 2, 2)
        plt.scatter(contingency_table.values.flatten(), posterior_mean_interaction.flatten(), alpha=0.5)
        plt.plot([0, contingency_table.values.max()], [0, contingency_table.values.max()], 'r--')
        plt.xlabel('Observed')
        plt.ylabel('Predicted (Interaction Model)')
        plt.title('Observed vs Predicted (Interaction)')

        # 3. 独立模型残差
        plt.subplot(2, 2, 3)
        plt.hist(residuals_independence.flatten(), bins=30, alpha=0.7)
        plt.axvline(x=0, color='r', linestyle='--')
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Residuals (Independence Model)')

        # 4. 交互模型残差
        plt.subplot(2, 2, 4)
        plt.hist(residuals_interaction.flatten(), bins=30, alpha=0.7)
        plt.axvline(x=0, color='r', linestyle='--')
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Residuals (Interaction Model)')

        plt.tight_layout()
        plt.savefig(f'bayesian_chisquare_{cause_col}_vs_{age_col}.png')
        plt.close()

        # 可视化行和列效应
        plt.figure(figsize=(15, 6))

        # 行效应
        plt.subplot(1, 2, 1)
        row_effects = trace_independence.posterior['alpha_row'].mean(("chain", "draw")).values
        plt.barh(range(n_causes), row_effects)
        plt.yticks(range(n_causes), contingency_table.index)
        plt.xlabel('Effect Size')
        plt.title('Cause Effects (Independence Model)')

        # 列效应
        plt.subplot(1, 2, 2)
        col_effects = trace_independence.posterior['alpha_col'].mean(("chain", "draw")).values
        plt.barh(range(n_age_groups), col_effects)
        plt.yticks(range(n_age_groups), contingency_table.columns)
        plt.xlabel('Effect Size')
        plt.title('Age Group Effects (Independence Model)')

        plt.tight_layout()
        plt.savefig(f'bayesian_chisquare_effects_{cause_col}_vs_{age_col}.png')
        plt.close()

        return results

    def bayesian_correlation(self, var1, var2):
        """
        执行贝叶斯相关性分析

        Parameters:
        -----------
        var1 : str
            第一个变量列名
        var2 : str
            第二个变量列名
        """
        print(f"\n--- 贝叶斯相关性分析: {var1} vs {var2} ---")

        # 移除缺失值
        data = self.data[[var1, var2]].dropna()
        x = data[var1].values
        y = data[var2].values

        # 标准化数据
        x_std = (x - np.mean(x)) / np.std(x)
        y_std = (y - np.mean(y)) / np.std(y)

        # 贝叶斯相关性模型
        with pm.Model() as model:
            # 先验分布
            mu = pm.Normal('mu', mu=0, sigma=1, shape=2)
            sigma = pm.HalfNormal('sigma', sigma=1, shape=2)

            # 相关系数
            rho = pm.Uniform('rho', lower=-1, upper=1)

            # 构建协方差矩阵
            cov = pm.Deterministic('cov', pt.stack([
                [sigma[0]**2, rho * sigma[0] * sigma[1]],
                [rho * sigma[0] * sigma[1], sigma[1]**2]
            ]))

            # 似然函数
            pm.MvNormal('obs', mu=mu, cov=cov, observed=np.column_stack([x_std, y_std]))

            # MCMC采样 - 减少采样量以提高速度
            trace = pm.sample(500, tune=500, cores=1, random_seed=42)

        # 获取摘要统计
        summary = az.summary(trace, var_names=["rho"])

        # 存储结果
        self.results[f"{var1}_vs_{var2}_correlation"] = {
            "correlation": summary.loc["rho", "mean"],
            "correlation_hdi_3%": summary.loc["rho", "hdi_3%"],
            "correlation_hdi_97%": summary.loc["rho", "hdi_97%"]
        }

        # 打印结果
        print(f"相关系数: {summary.loc['rho', 'mean']:.4f}")
        print(f"95% HDI: [{summary.loc['rho', 'hdi_3%']:.4f}, {summary.loc['rho', 'hdi_97%']:.4f}]")

        # 计算相关系数大于0的概率
        prob_positive = np.mean(trace.posterior['rho'].values > 0)
        print(f"相关系数 > 0 的概率: {prob_positive:.4f}")

        # 绘制相关系数的后验分布
        # az.plot_posterior(trace, var_names=["rho"])
        # plt.title(f"{var1} vs {var2} 的相关系数后验分布")
        # plt.show()

    def run_all_tests(self):
        """
        运行所有贝叶斯统计检验
        """
        print("\n" + "="*60)
        print("开始贝叶斯统计分析")
        print("="*60)

        # 1. 描述性统计分析
        self.descriptive_statistics()

        # 2. 假设检验
        # 这里需要根据你的数据特点选择合适的变量
        # 示例: 如果有分类变量和数值变量
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()

        # 对 male 和 female 列进行贝叶斯 t 检验
        if 'male' in self.data.columns and 'female' in self.data.columns:
            self.bayesian_ttest('male', 'female')

        # 对每对分类变量 age_group 和数值变量 both_sexes 执行ANOVA
        if 'age_group' in self.data.columns and 'both_sexes' in self.data.columns:
            self.bayesian_anova('age_group', 'both_sexes')

        # if 'cause_name' in self.data.columns and 'age_group' in self.data.columns and 'both_sexes' in self.data.columns:
        #     self.bayesian_chisquare(cause_col='cause_name', age_col='age_group', count_col='both_sexes')
        # else:
        #     print("\n警告: 缺少执行贝叶斯卡方检验所需的列 ('cause_name', 'age_group', 'both_sexes')。")

        # 对每对数值变量执行相关性分析
        for i, num_col1 in enumerate(numeric_cols):
            for num_col2 in numeric_cols[i+1:]:
                self.bayesian_correlation(num_col1, num_col2)

        return self.results


if __name__ == "__main__":
    try:
        # 数据路径
        datapath = "data/processed/who_mortality_clean.csv"

        # 检查数据文件是否存在
        import os
        if not os.path.exists(datapath):
            print("Error: Processed data not found at", datapath)
            print("Please run dataprocessing.py first.")
            exit(1)

        print(f"Loading data from {datapath}...")
        data = pd.read_csv(datapath)
        print(f"Loaded {len(data)} records")

        # 初始化贝叶斯统计分析器
        analyzer = BayesianStatistics(data)

        # 运行所有检验
        results = analyzer.run_all_tests()

        print("\n" + "="*60)
        print("BAYESIAN ANALYSIS COMPLETE")
        print("="*60)
        print("\nAll Bayesian statistical tests completed successfully!")
        print("Results have been stored in the analyzer.results dictionary")
        print("MCMC traces have been stored in analyzer.trace variables")

    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()

"""
机器学习分析可视化模块
Machine Learning Analysis Visualization Module

生成PART3机器学习分析的所有可视化图表
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
import os
import warnings

warnings.filterwarnings("ignore")

# 设置中文字体和样式
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams['figure.dpi'] = 100
sns.set_style("whitegrid")
sns.set_palette("husl")

# 创建输出目录
output_dir = "figures/dsml"
os.makedirs(output_dir, exist_ok=True)


class DSMLVisualizer:
    """机器学习分析可视化器"""

    def __init__(self, analyzer):
        """
        初始化可视化器

        Parameters:
        -----------
        analyzer : MLStatistics
            已完成分析的MLStatistics实例
        """
        self.analyzer = analyzer
        self.data = analyzer.data
        self.results = analyzer.results

    def plot_gender_classification(self):
        """
        绘制性别分类分析图表
        包括：模型性能对比、混淆矩阵
        """
        print("\n📊 Generating Gender Classification Visualizations...")

        gender_results = self.results.get('gender_ml', {})
        model_results = gender_results.get('model_results', {})

        if not model_results:
            print("   ⚠️  No gender classification results found")
            return

        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. 模型性能对比 - CV准确率
        ax1 = fig.add_subplot(gs[0, :2])
        models = list(model_results.keys())
        cv_means = [model_results[m]['cv_mean'] for m in models]
        cv_stds = [model_results[m]['cv_std'] for m in models]

        bars = ax1.bar(models, cv_means, yerr=cv_stds, capsize=5, alpha=0.7)
        # 标记最佳模型
        best_idx = cv_means.index(max(cv_means))
        bars[best_idx].set_color('red')
        bars[best_idx].set_alpha(1.0)

        ax1.set_ylabel('Cross-Validation Accuracy', fontsize=12)
        ax1.set_title('Gender Classification Model Comparison (CV)', fontsize=14, fontweight='bold')
        ax1.set_ylim([0.6, 0.8])
        ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random Baseline')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        # 2. 测试集准确率
        ax2 = fig.add_subplot(gs[0, 2])
        test_accs = [model_results[m]['test_accuracy'] for m in models]
        bars2 = ax2.barh(models, test_accs, alpha=0.7)
        best_idx = test_accs.index(max(test_accs))
        bars2[best_idx].set_color('red')
        bars2[best_idx].set_alpha(1.0)

        ax2.set_xlabel('Test Accuracy', fontsize=11)
        ax2.set_title('Test Set Performance', fontsize=12, fontweight='bold')
        ax2.set_xlim([0.6, 0.8])
        ax2.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
        ax2.grid(axis='x', alpha=0.3)

        # 3. 年龄组性别比例
        ax3 = fig.add_subplot(gs[1, :])
        age_gender_stats = gender_results.get('age_gender_stats')
        if age_gender_stats is not None:
            age_order = ['0-28 days', '1-59 months', '5-14', '15-29', '30-49', '50-59', '60-69', '70+']
            age_gender_stats = age_gender_stats.set_index('age_group').reindex(age_order).reset_index()

            x = np.arange(len(age_gender_stats))
            width = 0.35

            ax3.bar(x - width/2, age_gender_stats['male_ratio'], width,
                   label='Male Ratio', color='#3498db', alpha=0.8)
            ax3.bar(x + width/2, age_gender_stats['female_ratio'], width,
                   label='Female Ratio', color='#e74c3c', alpha=0.8)

            ax3.set_xlabel('Age Group', fontsize=12)
            ax3.set_ylabel('Death Ratio', fontsize=12)
            ax3.set_title('Gender Death Ratios by Age Group', fontsize=14, fontweight='bold')
            ax3.set_xticks(x)
            ax3.set_xticklabels(age_gender_stats['age_group'], rotation=45, ha='right')
            ax3.legend()
            ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
            ax3.grid(axis='y', alpha=0.3)

        # 4. 性别特异性疾病 Top 10
        ax4 = fig.add_subplot(gs[2, :])
        cause_gender_stats = gender_results.get('cause_gender_stats')
        if cause_gender_stats is not None:
            # 筛选性别差异大的疾病
            extreme_causes = cause_gender_stats[
                (cause_gender_stats['m_f_ratio'] > 2.0) |
                (cause_gender_stats['m_f_ratio'] < 0.5)
            ].copy()

            # 限制比例范围以便可视化
            extreme_causes['log_ratio'] = np.log10(extreme_causes['m_f_ratio'].clip(0.01, 100))
            extreme_causes = extreme_causes.nlargest(10, 'abs_diff')

            colors = ['#3498db' if r > 0 else '#e74c3c' for r in extreme_causes['log_ratio']]
            bars = ax4.barh(range(len(extreme_causes)), extreme_causes['log_ratio'], color=colors, alpha=0.7)

            ax4.set_yticks(range(len(extreme_causes)))
            ax4.set_yticklabels([name[:35] + '...' if len(name) > 35 else name
                                 for name in extreme_causes['cause_name']], fontsize=9)
            ax4.set_xlabel('Log10(Male/Female Ratio)', fontsize=12)
            ax4.set_title('Top 10 Gender-Specific Causes of Death', fontsize=14, fontweight='bold')
            ax4.axvline(x=0, color='black', linestyle='-', linewidth=1)
            ax4.text(0.5, 9.5, 'Male-dominant →', fontsize=10, color='#3498db', fontweight='bold')
            ax4.text(-0.5, 9.5, '← Female-dominant', fontsize=10, color='#e74c3c',
                    fontweight='bold', ha='right')
            ax4.grid(axis='x', alpha=0.3)

        plt.suptitle('Gender Classification Analysis', fontsize=16, fontweight='bold', y=0.995)
        plt.savefig(f"{output_dir}/gender_classification_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved: {output_dir}/gender_classification_analysis.png")

    def plot_clustering_analysis(self):
        """
        绘制聚类分析图表
        包括：K值选择、聚类散点图、聚类特征对比
        """
        print("\n📊 Generating Clustering Analysis Visualizations...")

        clustering_results = self.results.get('age_clustering', {})
        k_selection = self.results.get('k_selection', {})

        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # 1. K值选择（轮廓系数）
        ax1 = fig.add_subplot(gs[0, 0])
        if k_selection:
            K_range = k_selection['K_range']
            scores = k_selection['silhouette_scores']
            best_k = k_selection['best_k']

            ax1.plot(K_range, scores, 'bo-', linewidth=2, markersize=8)
            best_idx = K_range.index(best_k)
            ax1.plot(best_k, scores[best_idx], 'r*', markersize=20, label=f'Best K={best_k}')

            ax1.set_xlabel('Number of Clusters (K)', fontsize=12)
            ax1.set_ylabel('Silhouette Score', fontsize=12)
            ax1.set_title('Optimal K Selection (Silhouette Method)', fontsize=13, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

        # 2. 聚类散点图（PCA降维）
        ax2 = fig.add_subplot(gs[0, 1])
        if 'cluster' in self.data.columns:
            # 使用PCA降维到2D
            cluster_features = ['both_sexes', 'male', 'female', 'male_ratio', 'log_deaths']
            X = self.data[cluster_features].values

            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)

            scatter = ax2.scatter(X_pca[:, 0], X_pca[:, 1],
                                 c=self.data['cluster'], cmap='viridis',
                                 alpha=0.6, s=30, edgecolors='black', linewidth=0.5)

            ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=11)
            ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=11)
            ax2.set_title('Cluster Visualization (PCA)', fontsize=13, fontweight='bold')
            plt.colorbar(scatter, ax=ax2, label='Cluster')

            # 添加聚类中心
            if clustering_results.get('centers') is not None:
                centers_pca = pca.transform(clustering_results['centers'])
                ax2.scatter(centers_pca[:, 0], centers_pca[:, 1],
                           c='red', marker='X', s=300, edgecolors='black',
                           linewidth=2, label='Centroids', zorder=10)
                ax2.legend()

        # 3. 聚类大小分布
        ax3 = fig.add_subplot(gs[1, 0])
        if 'cluster' in self.data.columns:
            cluster_sizes = self.data['cluster'].value_counts().sort_index()
            bars = ax3.bar(cluster_sizes.index, cluster_sizes.values,
                          alpha=0.7, color=sns.color_palette("viridis", len(cluster_sizes)))

            ax3.set_xlabel('Cluster ID', fontsize=12)
            ax3.set_ylabel('Number of Samples', fontsize=12)
            ax3.set_title('Cluster Size Distribution', fontsize=13, fontweight='bold')
            ax3.grid(axis='y', alpha=0.3)

            # 添加数值标签
            for i, v in enumerate(cluster_sizes.values):
                ax3.text(i, v + 5, str(v), ha='center', va='bottom', fontweight='bold')

        # 4. 聚类特征均值对比
        ax4 = fig.add_subplot(gs[1, 1])
        if 'cluster' in self.data.columns:
            cluster_means = self.data.groupby('cluster').agg({
                'both_sexes': 'mean',
                'male_ratio': 'mean',
                'log_deaths': 'mean'
            })

            cluster_means_norm = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min())

            x = np.arange(len(cluster_means))
            width = 0.25

            ax4.bar(x - width, cluster_means_norm['both_sexes'], width,
                   label='Deaths (normalized)', alpha=0.8)
            ax4.bar(x, cluster_means_norm['male_ratio'], width,
                   label='Male Ratio (normalized)', alpha=0.8)
            ax4.bar(x + width, cluster_means_norm['log_deaths'], width,
                   label='Log Deaths (normalized)', alpha=0.8)

            ax4.set_xlabel('Cluster ID', fontsize=12)
            ax4.set_ylabel('Normalized Mean Value', fontsize=12)
            ax4.set_title('Cluster Feature Comparison', fontsize=13, fontweight='bold')
            ax4.set_xticks(x)
            ax4.set_xticklabels([f'C{i}' for i in cluster_means.index])
            ax4.legend(fontsize=9)
            ax4.grid(axis='y', alpha=0.3)

        plt.suptitle('Age Pattern Clustering Analysis', fontsize=16, fontweight='bold', y=0.995)
        plt.savefig(f"{output_dir}/clustering_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved: {output_dir}/clustering_analysis.png")

    def plot_regression_analysis(self):
        """
        绘制回归分析图表
        包括：模型性能对比、残差分析、特征系数
        """
        print("\n📊 Generating Regression Analysis Visualizations...")

        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        # 1. 正则化回归模型对比
        ax1 = fig.add_subplot(gs[0, :2])
        reg_results = self.results.get('regularized_correlation', {})
        if reg_results:
            models = list(reg_results.keys())
            rmse_means = [reg_results[m]['cv_rmse_mean'] for m in models]
            rmse_stds = [reg_results[m]['cv_rmse_std'] for m in models]

            bars = ax1.bar(models, rmse_means, yerr=rmse_stds, capsize=5, alpha=0.7)
            # 标记最佳模型
            best_idx = rmse_means.index(min(rmse_means))
            bars[best_idx].set_color('green')
            bars[best_idx].set_alpha(1.0)

            ax1.set_ylabel('Cross-Validation RMSE', fontsize=12)
            ax1.set_title('Regularized Regression Model Comparison', fontsize=14, fontweight='bold')
            ax1.grid(axis='y', alpha=0.3)

        # 2. R² 得分对比
        ax2 = fig.add_subplot(gs[0, 2])
        if reg_results:
            r2_scores = [reg_results[m]['r2_score'] for m in models]
            bars2 = ax2.barh(models, r2_scores, alpha=0.7, color=sns.color_palette("RdYlGn", len(models)))

            ax2.set_xlabel('R² Score', fontsize=12)
            ax2.set_title('Model R² Comparison', fontsize=13, fontweight='bold')
            ax2.set_xlim([0, 1.05])
            ax2.grid(axis='x', alpha=0.3)

        # 3. 死亡预测模型对比
        ax3 = fig.add_subplot(gs[1, :2])
        pred_results = self.results.get('death_prediction', {})
        if pred_results:
            pred_models = list(pred_results.keys())
            test_rmses = [pred_results[m]['test_rmse'] for m in pred_models]

            bars3 = ax3.bar(pred_models, test_rmses, alpha=0.7, color='skyblue')
            # 标记最佳模型
            best_idx = test_rmses.index(min(test_rmses))
            bars3[best_idx].set_color('green')
            bars3[best_idx].set_alpha(1.0)

            ax3.set_ylabel('Test RMSE', fontsize=12)
            ax3.set_title('Death Prediction Model Performance', fontsize=14, fontweight='bold')
            ax3.set_yscale('log')
            ax3.grid(axis='y', alpha=0.3)
            ax3.set_xticklabels(pred_models, rotation=15, ha='right')

        # 4. R² 得分对比（死亡预测）
        ax4 = fig.add_subplot(gs[1, 2])
        if pred_results:
            pred_r2 = [pred_results[m]['test_r2'] for m in pred_models]
            colors = ['green' if r2 > 0.8 else 'orange' if r2 > 0.5 else 'red' for r2 in pred_r2]
            bars4 = ax4.barh(pred_models, pred_r2, alpha=0.7, color=colors)

            ax4.set_xlabel('Test R² Score', fontsize=12)
            ax4.set_title('Prediction R² Scores', fontsize=13, fontweight='bold')
            ax4.axvline(x=0.8, color='green', linestyle='--', alpha=0.5, label='Good (>0.8)')
            ax4.axvline(x=0.5, color='orange', linestyle='--', alpha=0.5, label='Fair (>0.5)')
            ax4.legend(fontsize=9)
            ax4.grid(axis='x', alpha=0.3)

        plt.suptitle('Regression Analysis Results', fontsize=16, fontweight='bold', y=0.995)
        plt.savefig(f"{output_dir}/regression_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved: {output_dir}/regression_analysis.png")

    def plot_bias_variance_analysis(self):
        """
        绘制偏差-方差权衡分析图表
        """
        print("\n📊 Generating Bias-Variance Analysis Visualizations...")

        bv_results = self.results.get('bias_variance', {})
        if not bv_results:
            print("   ⚠️  No bias-variance results found")
            return

        fig = plt.figure(figsize=(16, 8))
        gs = fig.add_gridspec(1, 3, hspace=0.3, wspace=0.3)

        models = list(bv_results.keys())
        train_r2 = [bv_results[m]['train_r2'] for m in models]
        val_r2 = [bv_results[m]['val_r2'] for m in models]
        overfitting_gap = [bv_results[m]['overfitting_gap'] for m in models]

        # 1. Train vs Validation R²
        ax1 = fig.add_subplot(gs[0, 0])
        x = np.arange(len(models))
        width = 0.35

        ax1.bar(x - width/2, train_r2, width, label='Train R²', alpha=0.8, color='#3498db')
        ax1.bar(x + width/2, val_r2, width, label='Validation R²', alpha=0.8, color='#e74c3c')

        ax1.set_ylabel('R² Score', fontsize=12)
        ax1.set_title('Train vs Validation Performance', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels([m.split('(')[0].strip() for m in models], rotation=45, ha='right')
        ax1.legend()
        ax1.axhline(y=0.8, color='green', linestyle='--', alpha=0.5)
        ax1.grid(axis='y', alpha=0.3)

        # 2. Overfitting Gap
        ax2 = fig.add_subplot(gs[0, 1])
        colors = ['red' if gap > 0.2 else 'orange' if gap > 0.1 else 'green'
                 for gap in overfitting_gap]
        bars = ax2.barh(models, overfitting_gap, alpha=0.7, color=colors)

        ax2.set_xlabel('Overfitting Gap (Train R² - Val R²)', fontsize=12)
        ax2.set_title('Overfitting Analysis', fontsize=14, fontweight='bold')
        ax2.axvline(x=0.1, color='orange', linestyle='--', alpha=0.5)
        ax2.axvline(x=0.2, color='red', linestyle='--', alpha=0.5)
        ax2.grid(axis='x', alpha=0.3)

        # 3. CV R² with std
        ax3 = fig.add_subplot(gs[0, 2])
        cv_r2_means = [bv_results[m]['cv_r2_mean'] for m in models]
        cv_r2_stds = [bv_results[m]['cv_r2_std'] for m in models]

        ax3.errorbar(cv_r2_means, range(len(models)), xerr=cv_r2_stds,
                    fmt='o', markersize=8, capsize=5, alpha=0.7)

        ax3.set_yticks(range(len(models)))
        ax3.set_yticklabels(models)
        ax3.set_xlabel('Cross-Validation R² (mean ± std)', fontsize=12)
        ax3.set_title('Model Stability (CV Results)', fontsize=14, fontweight='bold')
        ax3.axvline(x=0.8, color='green', linestyle='--', alpha=0.5, label='Good (>0.8)')
        ax3.legend()
        ax3.grid(axis='x', alpha=0.3)

        plt.suptitle('Bias-Variance Tradeoff Analysis', fontsize=16, fontweight='bold', y=0.98)
        plt.savefig(f"{output_dir}/bias_variance_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved: {output_dir}/bias_variance_analysis.png")

    def plot_summary_dashboard(self):
        """
        生成总结仪表板
        """
        print("\n📊 Generating Summary Dashboard...")

        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

        # 1. 最佳模型汇总
        ax1 = fig.add_subplot(gs[0, :])
        summary_data = []

        # 性别分类
        gender_results = self.results.get('gender_ml', {}).get('model_results', {})
        if gender_results:
            best = max(gender_results.items(), key=lambda x: x[1]['test_accuracy'])
            summary_data.append({
                'Task': 'Gender\nClassification',
                'Best Model': best[0],
                'Score': best[1]['test_accuracy'],
                'Metric': 'Accuracy'
            })

        # 正则化回归
        reg_results = self.results.get('regularized_correlation', {})
        if reg_results:
            best = min(reg_results.items(), key=lambda x: x[1]['cv_rmse_mean'])
            summary_data.append({
                'Task': 'Regularized\nRegression',
                'Best Model': best[0],
                'Score': best[1]['cv_rmse_mean'],
                'Metric': 'RMSE'
            })

        # 死亡预测
        pred_results = self.results.get('death_prediction', {})
        if pred_results:
            best = min(pred_results.items(), key=lambda x: x[1]['test_rmse'])
            summary_data.append({
                'Task': 'Death\nPrediction',
                'Best Model': best[0],
                'Score': best[1]['test_rmse'],
                'Metric': 'RMSE'
            })

        if summary_data:
            tasks = [d['Task'] for d in summary_data]
            models = [d['Best Model'] for d in summary_data]
            scores = [d['Score'] for d in summary_data]

            # 创建表格显示
            ax1.axis('tight')
            ax1.axis('off')
            table_data = [[d['Task'], d['Best Model'], f"{d['Score']:.3f}", d['Metric']]
                         for d in summary_data]
            table = ax1.table(cellText=table_data,
                            colLabels=['Analysis Task', 'Best Model', 'Performance', 'Metric'],
                            cellLoc='center',
                            loc='center',
                            colWidths=[0.25, 0.25, 0.25, 0.25])
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1, 3)

            # 样式设置
            for i in range(4):
                table[(0, i)].set_facecolor('#3498db')
                table[(0, i)].set_text_props(weight='bold', color='white')

            for i in range(1, len(summary_data) + 1):
                for j in range(4):
                    table[(i, j)].set_facecolor('#ecf0f1' if i % 2 == 0 else 'white')

            ax1.set_title('Machine Learning Analysis Summary - Best Models',
                         fontsize=16, fontweight='bold', pad=20)

        # 2. 数据集统计
        ax2 = fig.add_subplot(gs[1, 0])
        stats = [
            ['Total Records', len(self.data)],
            ['Age Groups', self.data['age_group'].nunique()],
            ['Causes', self.data['cause_name'].nunique()],
            ['Total Deaths', f"{self.data['both_sexes'].sum():.2e}"],
            ['Mean Deaths', f"{self.data['both_sexes'].mean():.0f}"],
            ['Median Deaths', f"{self.data['both_sexes'].median():.0f}"]
        ]

        ax2.axis('tight')
        ax2.axis('off')
        table2 = ax2.table(cellText=stats,
                          colLabels=['Statistic', 'Value'],
                          cellLoc='left',
                          loc='center',
                          colWidths=[0.6, 0.4])
        table2.auto_set_font_size(False)
        table2.set_fontsize(11)
        table2.scale(1, 2.5)

        table2[(0, 0)].set_facecolor('#2ecc71')
        table2[(0, 1)].set_facecolor('#2ecc71')
        table2[(0, 0)].set_text_props(weight='bold', color='white')
        table2[(0, 1)].set_text_props(weight='bold', color='white')

        ax2.set_title('Dataset Statistics', fontsize=13, fontweight='bold')

        # 3. 聚类信息
        ax3 = fig.add_subplot(gs[1, 1])
        if 'cluster' in self.data.columns:
            cluster_info = []
            for cid in sorted(self.data['cluster'].unique()):
                cluster_data = self.data[self.data['cluster'] == cid]
                cluster_info.append([
                    f'Cluster {cid}',
                    len(cluster_data),
                    f"{cluster_data['both_sexes'].mean():.0f}",
                    f"{cluster_data['male_ratio'].mean():.2f}"
                ])

            ax3.axis('tight')
            ax3.axis('off')
            table3 = ax3.table(cellText=cluster_info,
                              colLabels=['Cluster', 'Size', 'Avg Deaths', 'Male Ratio'],
                              cellLoc='center',
                              loc='center',
                              colWidths=[0.25, 0.25, 0.25, 0.25])
            table3.auto_set_font_size(False)
            table3.set_fontsize(10)
            table3.scale(1, 2)

            table3[(0, 0)].set_facecolor('#9b59b6')
            table3[(0, 1)].set_facecolor('#9b59b6')
            table3[(0, 2)].set_facecolor('#9b59b6')
            table3[(0, 3)].set_facecolor('#9b59b6')
            for i in range(4):
                table3[(0, i)].set_text_props(weight='bold', color='white')

            ax3.set_title('Clustering Results', fontsize=13, fontweight='bold')

        # 4. 性别差异摘要
        ax4 = fig.add_subplot(gs[1, 2])
        gender_stats = self.results.get('gender_ml', {})
        if 'age_gender_stats' in gender_stats:
            age_stats = gender_stats['age_gender_stats']
            male_dominant = (age_stats['male_ratio'] > 0.5).sum()
            female_dominant = (age_stats['female_ratio'] > 0.5).sum()

            sizes = [male_dominant, female_dominant]
            labels = ['Male-dominant\nAge Groups', 'Female-dominant\nAge Groups']
            colors = ['#3498db', '#e74c3c']

            ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.0f%%',
                   startangle=90, textprops={'fontsize': 11, 'weight': 'bold'})
            ax4.set_title('Age Group Gender Dominance', fontsize=13, fontweight='bold')

        plt.suptitle('WHO Mortality Machine Learning Analysis - Dashboard',
                    fontsize=18, fontweight='bold', y=0.98)
        plt.savefig(f"{output_dir}/ml_analysis_dashboard.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved: {output_dir}/ml_analysis_dashboard.png")

    def generate_all_visualizations(self):
        """生成所有可视化图表"""
        print("\n" + "=" * 60)
        print("  GENERATING ALL ML VISUALIZATIONS")
        print("=" * 60)

        self.plot_gender_classification()
        self.plot_clustering_analysis()
        self.plot_regression_analysis()
        self.plot_bias_variance_analysis()
        self.plot_summary_dashboard()

        print("\n" + "=" * 60)
        print("  ALL VISUALIZATIONS GENERATED SUCCESSFULLY")
        print("=" * 60)
        print(f"\n📁 Output directory: {output_dir}/")


# 主程序
if __name__ == "__main__":
    print("=" * 60)
    print("  WHO Mortality ML Analysis Visualizations")
    print("=" * 60)

    try:
        # 需要先运行dsml_analysis.py生成结果
        from dsml_analysis import MLStatistics

        # 加载数据
        data = pd.read_csv("data/processed/who_mortality_clean.csv")

        # 运行分析
        print("\n📊 Running ML Analysis...")
        analyzer = MLStatistics(data)
        analyzer.run_all_ml_tests()

        # 生成可视化
        visualizer = DSMLVisualizer(analyzer)
        visualizer.generate_all_visualizations()

        print("\n✅ Visualization generation complete!")
        print(f"   Check the '{output_dir}' directory for generated figures.")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

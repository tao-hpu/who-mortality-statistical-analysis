"""
机器学习统计分析模块（修复版）
Machine Learning Statistical Analysis Module (Fixed Version)

主要功能:
1. 描述性统计分析（保留）
2. 性别差异分析（使用ML分类器）
3. 年龄组模式识别（使用聚类和决策树）
4. 相关性分析（使用正则化回归）
5. 死亡数预测（使用多种ML回归算法）
6. 偏差-方差均衡分析（回归任务）
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings("ignore")


class MLStatistics:
    """
    机器学习统计分析类
    使用ML技术执行统计分析和预测
    """

    def __init__(self, data):
        """
        初始化ML统计分析器

        Parameters:
        -----------
        data : pd.DataFrame
            处理后的WHO死亡率数据
        """
        self.data = data.copy()
        self.results = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}

        # 预处理数据
        self._preprocess_data()

    def _preprocess_data(self):
        """数据预处理和特征工程"""
        # 年龄组数值编码
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
        self.data["age_numeric"] = self.data["age_group"].map(age_encoding)

        # 性别比例特征
        self.data["male_ratio"] = self.data["male"] / (self.data["both_sexes"] + 1e-10)
        self.data["female_ratio"] = self.data["female"] / (self.data["both_sexes"] + 1e-10)

        # 对数变换处理偏态分布
        self.data["log_deaths"] = np.log1p(self.data["both_sexes"])
        self.data["log_male"] = np.log1p(self.data["male"])
        self.data["log_female"] = np.log1p(self.data["female"])

        # 编码分类变量
        self.label_encoders['cause_name'] = LabelEncoder()
        self.data['cause_encoded'] = self.label_encoders['cause_name'].fit_transform(self.data['cause_name'])

        self.label_encoders['age_group'] = LabelEncoder()
        self.data['age_encoded'] = self.label_encoders['age_group'].fit_transform(self.data['age_group'])

    def descriptive_statistics(self):
        """
        保留描述性统计分析（与原版相同）
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

        print("\n📊 Overall Death Statistics:")
        print(f"   Mean deaths per category: {desc_stats['total_deaths']['mean']:,.1f}")
        print(f"   Median deaths: {desc_stats['total_deaths']['median']:,.1f}")
        print(f"   Std deviation: {desc_stats['total_deaths']['std']:,.1f}")
        print(f"   Range: {desc_stats['total_deaths']['min']:,.0f} - {desc_stats['total_deaths']['max']:,.0f}")

        self.results["descriptive"] = desc_stats
        return desc_stats

    def gender_classification_analysis(self):
        """
        使用机器学习分类器分析性别差异模式，并提供详细的性别对比分析
        """
        print("\n" + "=" * 50)
        print("GENDER DIFFERENCE ML ANALYSIS")
        print("=" * 50)

        # 1. 首先进行传统的性别差异统计分析
        print("\n📊 Traditional Gender Difference Analysis:")

        # 整体性别差异
        total_male = self.data['male'].sum()
        total_female = self.data['female'].sum()
        total_both = self.data['both_sexes'].sum()

        print(f"   Total Deaths - Male: {total_male:,.0f}, Female: {total_female:,.0f}")
        print(f"   Male Ratio: {total_male/total_both:.3f}, Female Ratio: {total_female/total_both:.3f}")
        print(f"   Male/Female Ratio: {total_male/total_female:.3f}")

        # 按年龄组的性别差异
        print("\n📈 Gender Differences by Age Group:")
        age_gender_stats = self.data.groupby('age_group').agg({
            'male': 'sum',
            'female': 'sum',
            'both_sexes': 'sum'
        }).reset_index()

        # 安全计算比例，避免除零
        age_gender_stats['male_ratio'] = age_gender_stats['male'] / (age_gender_stats['both_sexes'] + 1e-10)
        age_gender_stats['female_ratio'] = age_gender_stats['female'] / (age_gender_stats['both_sexes'] + 1e-10)
        age_gender_stats['m_f_ratio'] = age_gender_stats['male'] / (age_gender_stats['female'] + 1e-10)

        for _, row in age_gender_stats.iterrows():
            print(f"   {row['age_group']:15} M:{row['male_ratio']:.3f} F:{row['female_ratio']:.3f} M/F:{row['m_f_ratio']:.2f}")

        # 按死因的性别差异（Top 10）
        print("\n🔬 Top 10 Causes with Largest Gender Differences:")
        cause_gender_stats = self.data.groupby('cause_name').agg({
            'male': 'sum',
            'female': 'sum',
            'both_sexes': 'sum'
        }).reset_index()

        # 安全计算比例，避免除零
        cause_gender_stats['male_ratio'] = cause_gender_stats['male'] / (cause_gender_stats['both_sexes'] + 1e-10)
        cause_gender_stats['female_ratio'] = cause_gender_stats['female'] / (cause_gender_stats['both_sexes'] + 1e-10)
        cause_gender_stats['m_f_ratio'] = cause_gender_stats['male'] / (cause_gender_stats['female'] + 1e-10)
        cause_gender_stats['abs_diff'] = abs(cause_gender_stats['male'] - cause_gender_stats['female'])

        top_causes = cause_gender_stats.nlargest(10, 'abs_diff')
        for _, row in top_causes.iterrows():
            if row['female'] == 0:
                gender_trend = "Male-only"
                ratio_text = "∞"
            elif row['male'] == 0:
                gender_trend = "Female-only"
                ratio_text = "∞"
            else:
                gender_trend = "Male-dominant" if row['male'] > row['female'] else "Female-dominant"
                ratio_text = f"{row['m_f_ratio']:.2f}"

            print(f"   {row['cause_name'][:40]:40} M/F:{ratio_text} ({gender_trend})")

        # 2. 机器学习分类分析
        print("\n" + "=" * 50)
        print("MACHINE LEARNING CLASSIFICATION ANALYSIS")
        print("=" * 50)

        # 准备特征和标签
        features = ['age_numeric', 'both_sexes', 'log_deaths', 'cause_encoded']
        X = self.data[features]
        y = (self.data['male'] > self.data['female']).astype(int)  # 1=男性死亡更多

        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)

        # 多种分类器比较
        classifiers = {
            'KNN': KNeighborsClassifier(),
            'SVM': SVC(probability=True),
            'Decision Tree': DecisionTreeClassifier(max_depth=5),
            'Random Forest': RandomForestClassifier(n_estimators=100),
            'LDA': LinearDiscriminantAnalysis()
        }

        results = {}

        for name, clf in classifiers.items():
            # 交叉验证评估
            cv_scores = cross_val_score(clf, X_scaled, y, cv=5, scoring='accuracy')

            # 训练模型
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.3, random_state=42, stratify=y
            )
            clf.fit(X_train, y_train)

            # 测试集评估
            test_score = clf.score(X_test, y_test)

            # 预测概率（对于支持概率的模型）
            if hasattr(clf, 'predict_proba'):
                y_proba = clf.predict_proba(X_test)
                confidence = np.mean(np.max(y_proba, axis=1))
            else:
                confidence = "N/A"

            results[name] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'test_accuracy': test_score,
                'confidence': confidence,
                'model': clf
            }

            print(f"📈 {name}:")
            print(f"   CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
            print(f"   Test Accuracy: {test_score:.3f}")
            if confidence != "N/A":
                print(f"   Avg Confidence: {confidence:.3f}")

        # 找出最佳模型
        best_model = max(results.items(), key=lambda x: x[1]['test_accuracy'])
        print(f"\n🏆 Best Model: {best_model[0]} (Accuracy: {best_model[1]['test_accuracy']:.3f})")

        # 3. 使用最佳模型进行深入分析
        print("\n" + "=" * 50)
        print("INSIGHTS FROM BEST MODEL")
        print("=" * 50)

        best_clf = best_model[1]['model']

        # 特征重要性分析（对于树模型）
        if best_model[0] in ['Decision Tree', 'Random Forest']:
            importances = best_clf.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': features,
                'importance': importances
            }).sort_values('importance', ascending=False)

            print("\n📊 Feature Importance for Gender Prediction:")
            for _, row in feature_importance.iterrows():
                print(f"   {row['feature']}: {row['importance']:.3f}")

        # 预测分析：找出模型最容易预测和最难预测的案例
        X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        best_clf.fit(X_train_full, y_train_full)
        y_pred = best_clf.predict(X_test_full)

        # 分析预测正确的案例
        correct_predictions = X_test_full[y_pred == y_test_full]
        incorrect_predictions = X_test_full[y_pred != y_test_full]

        print(f"\n📈 Model Performance Insights:")
        print(f"   Correctly predicted: {len(correct_predictions)}/{len(y_test_full)} ({len(correct_predictions)/len(y_test_full):.3f})")
        print(f"   Incorrectly predicted: {len(incorrect_predictions)}/{len(y_test_full)} ({len(incorrect_predictions)/len(y_test_full):.3f})")

        # 分析最容易混淆的情况
        if len(incorrect_predictions) > 0:
            print("\n🤔 Most Confusing Cases (Incorrect Predictions):")
            # 获取原始数据中对应的记录
            incorrect_indices = incorrect_predictions.index
            confusing_cases = self.data.iloc[incorrect_indices].head(5)

            for _, row in confusing_cases.iterrows():
                actual = "Male-dominant" if row['male'] > row['female'] else "Female-dominant"
                print(f"   {row['cause_name'][:30]} ({row['age_group']}): {actual}")
                print(f"      Male: {row['male']:,.0f}, Female: {row['female']:,.0f}")

        # 4. 性别模式的业务洞察
        print("\n" + "=" * 50)
        print("GENDER PATTERN INSIGHTS")
        print("=" * 50)

        # 计算不同年龄段的性别主导模式
        age_dominance = self.data.groupby('age_group').apply(
            lambda x: pd.Series({
                'male_dominant_cases': (x['male'] > x['female']).sum(),
                'female_dominant_cases': (x['male'] <= x['female']).sum(),
                'total_cases': len(x)
            })
        ).reset_index()

        age_dominance['male_dominance_rate'] = age_dominance['male_dominant_cases'] / age_dominance['total_cases']

        print("\n📊 Gender Dominance by Age Group:")
        for _, row in age_dominance.iterrows():
            dominance = "Male-dominant" if row['male_dominance_rate'] > 0.5 else "Female-dominant"
            print(f"   {row['age_group']:15} {dominance} ({row['male_dominance_rate']:.3f})")

        # 找出性别差异最显著的死因（修复显示问题）
        print("\n🎯 Most Gender-Specific Causes:")

        # 处理无穷大值，创建一个更友好的显示
        extreme_causes = cause_gender_stats.copy()
        extreme_causes['display_ratio'] = extreme_causes.apply(
            lambda row: '∞' if (row['female'] == 0 or row['male'] == 0) else f"{row['m_f_ratio']:.2f}",
            axis=1
        )

        # 筛选性别特异性强的疾病（比例>2或<0.5）
        gender_specific = extreme_causes[
            (extreme_causes['m_f_ratio'] > 2.0) | (extreme_causes['m_f_ratio'] < 0.5)
            ].sort_values('m_f_ratio', ascending=False)

        for _, row in gender_specific.head(10).iterrows():
            if row['female'] == 0:
                print(f"   🚹 {row['cause_name'][:40]}: Male-only disease ({row['male']:,.0f} deaths)")
            elif row['male'] == 0:
                print(f"   🚺 {row['cause_name'][:40]}: Female-only disease ({row['female']:,.0f} deaths)")
            elif row['m_f_ratio'] > 2.0:
                print(f"   🚹 {row['cause_name'][:40]}: {row['display_ratio']}x more male deaths")
            else:
                female_multiple = 1 / row['m_f_ratio']
                print(f"   🚺 {row['cause_name'][:40]}: {female_multiple:.2f}x more female deaths")

        # 5. 额外的性别差异洞察
        print("\n" + "=" * 50)
        print("ADDITIONAL GENDER INSIGHTS")
        print("=" * 50)

        # 计算性别差异的统计显著性
        print("\n📊 Statistical Significance of Gender Differences:")

        # 按年龄组进行配对t检验
        from scipy.stats import ttest_rel

        age_groups = self.data['age_group'].unique()
        significant_age_groups = []

        for age_group in age_groups:
            age_data = self.data[self.data['age_group'] == age_group]
            if len(age_data) > 1:
                t_stat, p_value = ttest_rel(age_data['male'], age_data['female'])
                if p_value < 0.05:
                    direction = "Male > Female" if age_data['male'].mean() > age_data['female'].mean() else "Female > Male"
                    significant_age_groups.append({
                        'age_group': age_group,
                        'p_value': p_value,
                        'direction': direction,
                        'mean_diff': age_data['male'].mean() - age_data['female'].mean()
                    })

        if significant_age_groups:
            print("   Age groups with significant gender differences (p < 0.05):")
            for item in sorted(significant_age_groups, key=lambda x: x['p_value']):
                print(f"   {item['age_group']:15} {item['direction']} (p={item['p_value']:.6f}, diff={item['mean_diff']:.1f})")
        else:
            print("   No significant gender differences found by age group")

        self.results["gender_ml"] = {
            'model_results': results,
            'age_gender_stats': age_gender_stats,
            'cause_gender_stats': cause_gender_stats,
            'age_dominance': age_dominance,
            'significant_age_groups': significant_age_groups
        }

        return self.results["gender_ml"]



    def age_pattern_clustering(self):
        """
        使用聚类分析识别年龄组模式
        """
        print("\n" + "=" * 50)
        print("AGE GROUP PATTERN CLUSTERING")
        print("=" * 50)

        # 准备聚类特征
        cluster_features = ['both_sexes', 'male', 'female', 'male_ratio', 'log_deaths']
        X_cluster = self.data[cluster_features].copy()

        # 标准化
        X_cluster_scaled = self.scaler.fit_transform(X_cluster)

        # 使用肘部法则确定最佳聚类数
        inertias = []
        K_range = range(2, 8)

        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X_cluster_scaled)
            inertias.append(kmeans.inertia_)

        # 选择最佳K（简化版，选择拐点）
        best_k = 3  # 可以根据肘部法则动态选择

        # 执行聚类
        kmeans = KMeans(n_clusters=best_k, random_state=42)
        cluster_labels = kmeans.fit_predict(X_cluster_scaled)
        self.data['cluster'] = cluster_labels

        # 分析聚类结果
        print(f"📈 Clustering Results (K={best_k}):")

        for cluster_id in range(best_k):
            cluster_data = self.data[self.data['cluster'] == cluster_id]
            print(f"\n   Cluster {cluster_id} (n={len(cluster_data)}):")
            print(f"   Mean deaths: {cluster_data['both_sexes'].mean():.1f}")
            print(f"   Male ratio: {cluster_data['male_ratio'].mean():.3f}")
            print(f"   Common age groups: {cluster_data['age_group'].mode().values[:3]}")

        # 聚类中心解释
        print("\n📊 Cluster Centers Interpretation:")
        centers = self.scaler.inverse_transform(kmeans.cluster_centers_)
        center_df = pd.DataFrame(centers, columns=cluster_features)
        print(center_df.round(2))

        self.results["age_clustering"] = {
            'model': kmeans,
            'labels': cluster_labels,
            'centers': centers
        }

        return self.results["age_clustering"]

    def regularized_correlation_analysis(self):
        """
        使用正则化回归分析变量关系
        """
        print("\n" + "=" * 50)
        print("REGULARIZED CORRELATION ANALYSIS")
        print("=" * 50)

        # 准备回归数据
        X = self.data[['age_numeric', 'male', 'female', 'cause_encoded']]
        y = self.data['both_sexes']

        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)

        # 多种正则化回归模型
        models = {
            'Lasso': Lasso(alpha=0.1),
            'Ridge': Ridge(alpha=1.0),
            'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5)
        }

        results = {}

        for name, model in models.items():
            # 交叉验证评估
            cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='neg_mean_squared_error')
            rmse_scores = np.sqrt(-cv_scores)

            # 训练模型
            model.fit(X_scaled, y)

            # 预测和评估
            y_pred = model.predict(X_scaled)
            train_rmse = np.sqrt(mean_squared_error(y, y_pred))
            r2 = r2_score(y, y_pred)

            results[name] = {
                'cv_rmse_mean': rmse_scores.mean(),
                'cv_rmse_std': rmse_scores.std(),
                'train_rmse': train_rmse,
                'r2_score': r2,
                'coefficients': model.coef_,
                'model': model
            }

            print(f"📈 {name}:")
            print(f"   CV RMSE: {rmse_scores.mean():.1f} ± {rmse_scores.std():.1f}")
            print(f"   Train RMSE: {train_rmse:.1f}")
            print(f"   R² Score: {r2:.3f}")

        # 分析最佳模型的系数
        best_model = min(results.items(), key=lambda x: x[1]['cv_rmse_mean'])
        print(f"\n🏆 Best Model: {best_model[0]} (RMSE: {best_model[1]['cv_rmse_mean']:.1f})")

        print("\n📊 Feature Coefficients:")
        coef_df = pd.DataFrame({
            'feature': X.columns,
            'coefficient': best_model[1]['coefficients']
        }).sort_values('coefficient', key=abs, ascending=False)

        for _, row in coef_df.iterrows():
            print(f"   {row['feature']}: {row['coefficient']:.3f}")

        self.results["regularized_correlation"] = results
        return results

    def death_prediction_analysis(self):
        """
        使用多种ML回归算法预测死亡数
        """
        print("\n" + "=" * 50)
        print("DEATH PREDICTION ML ANALYSIS")
        print("=" * 50)

        # 准备预测数据
        features = ['age_numeric', 'male', 'female', 'male_ratio', 'cause_encoded']
        X = self.data[features]
        y = self.data['both_sexes']

        # 标准化
        X_scaled = self.scaler.fit_transform(X)

        # 多种回归器
        regressors = {
            'KNN': KNeighborsRegressor(),
            'SVR': SVR(),
            'Decision Tree': DecisionTreeRegressor(max_depth=10),
            'Random Forest': RandomForestRegressor(n_estimators=100),
            'Ridge': Ridge(alpha=1.0)
        }

        results = {}

        for name, reg in regressors.items():
            # 交叉验证
            cv_scores = cross_val_score(reg, X_scaled, y, cv=5, scoring='neg_mean_squared_error')
            rmse_scores = np.sqrt(-cv_scores)

            # 训练测试分割
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.3, random_state=42
            )
            reg.fit(X_train, y_train)

            # 测试评估
            y_pred = reg.predict(X_test)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            test_r2 = r2_score(y_test, y_pred)

            results[name] = {
                'cv_rmse_mean': rmse_scores.mean(),
                'cv_rmse_std': rmse_scores.std(),
                'test_rmse': test_rmse,
                'test_r2': test_r2,
                'model': reg
            }

            print(f"📈 {name}:")
            print(f"   CV RMSE: {rmse_scores.mean():.1f} ± {rmse_scores.std():.1f}")
            print(f"   Test RMSE: {test_rmse:.1f}")
            print(f"   Test R²: {test_r2:.3f}")

        # 最佳模型详细分析
        best_model = min(results.items(), key=lambda x: x[1]['test_rmse'])
        print(f"\n🏆 Best Model: {best_model[0]} (RMSE: {best_model[1]['test_rmse']:.1f})")

        # 特征重要性（对于树模型）
        if best_model[0] in ['Decision Tree', 'Random Forest']:
            model = best_model[1]['model']
            importances = model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': features,
                'importance': importances
            }).sort_values('importance', ascending=False)

            print("\n📊 Feature Importance:")
            for _, row in feature_importance.iterrows():
                print(f"   {row['feature']}: {row['importance']:.3f}")

        self.results["death_prediction"] = results
        return results

    def bias_variance_analysis(self):
        """
        偏差-方差权衡分析（回归任务）
        """
        print("\n" + "=" * 50)
        print("BIAS-VARIANCE TRADEOFF ANALYSIS")
        print("=" * 50)

        # 准备数据
        X = self.data[['age_numeric', 'male', 'female', 'cause_encoded']]
        y = self.data['both_sexes']
        X_scaled = self.scaler.fit_transform(X)

        # 不同复杂度的回归模型
        models = {
            'High Bias (Linear)': Ridge(alpha=10.0),
            'Balanced (RF-50)': RandomForestRegressor(n_estimators=50, max_depth=5),
            'High Variance (RF-200)': RandomForestRegressor(n_estimators=200, max_depth=None),
            'Medium (KNN-3)': KNeighborsRegressor(n_neighbors=3),
            'Low Variance (KNN-10)': KNeighborsRegressor(n_neighbors=10)
        }

        results = {}

        for name, model in models.items():
            # 训练集和验证集性能
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y, test_size=0.3, random_state=42
            )

            model.fit(X_train, y_train)
            train_score = model.score(X_train, y_train)
            val_score = model.score(X_val, y_val)

            # 交叉验证
            cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')

            # RMSE计算
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)
            train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
            val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))

            results[name] = {
                'train_r2': train_score,
                'val_r2': val_score,
                'train_rmse': train_rmse,
                'val_rmse': val_rmse,
                'cv_r2_mean': cv_scores.mean(),
                'cv_r2_std': cv_scores.std(),
                'overfitting_gap': train_score - val_score  # 过拟合指标
            }

            print(f"📈 {name}:")
            print(f"   Train R²: {train_score:.3f}, Val R²: {val_score:.3f}")
            print(f"   Train RMSE: {train_rmse:.1f}, Val RMSE: {val_rmse:.1f}")
            print(f"   Overfitting Gap: {train_score - val_score:.3f}")
            print(f"   CV R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

        # 找出最佳偏差-方差平衡
        best_balanced = min(results.items(), key=lambda x: abs(x[1]['overfitting_gap']))
        print(f"\n🏆 Best Bias-Variance Balance: {best_balanced[0]}")
        print(f"   Overfitting Gap: {abs(best_balanced[1]['overfitting_gap']):.3f}")
        print(f"   Validation R²: {best_balanced[1]['val_r2']:.3f}")

        # 偏差-方差可视化建议
        print("\n📊 Bias-Variance Interpretation:")
        for name, result in results.items():
            if result['overfitting_gap'] > 0.2:
                print(f"   {name}: High variance (overfitting)")
            elif result['val_r2'] < 0.3:
                print(f"   {name}: High bias (underfitting)")
            else:
                print(f"   {name}: Good balance")

        self.results["bias_variance"] = results
        return results

    def run_all_ml_tests(self):
        """
        运行所有机器学习分析
        """
        print("\n" + "=" * 60)
        print("  RUNNING ALL MACHINE LEARNING ANALYSES")
        print("=" * 60)

        # 1. 描述性统计
        self.descriptive_statistics()

        # 2. 性别差异ML分析
        self.gender_classification_analysis()

        # 3. 年龄模式聚类
        self.age_pattern_clustering()

        # 4. 正则化相关性分析
        self.regularized_correlation_analysis()

        # 5. 死亡数预测分析
        self.death_prediction_analysis()

        # 6. 偏差-方差分析
        self.bias_variance_analysis()

        # 创建结果摘要
        print("\n" + "=" * 60)
        print("  MACHINE LEARNING ANALYSIS SUMMARY")
        print("=" * 60)

        summary_data = []

        # 性别分析摘要 - 修复访问路径
        if 'gender_ml' in self.results and 'model_results' in self.results['gender_ml']:
            model_results = self.results['gender_ml']['model_results']
            if model_results:
                best_gender = max(model_results.items(),
                                  key=lambda x: x[1]['test_accuracy'])
                summary_data.append({
                    'Analysis': 'Gender Classification',
                    'Best Model': best_gender[0],
                    'Performance': f"{best_gender[1]['test_accuracy']:.3f}"
                })

        # 正则化回归摘要
        if 'regularized_correlation' in self.results:
            best_reg = min(self.results['regularized_correlation'].items(),
                           key=lambda x: x[1]['cv_rmse_mean'])
            summary_data.append({
                'Analysis': 'Regularized Regression',
                'Best Model': best_reg[0],
                'Performance': f"RMSE: {best_reg[1]['cv_rmse_mean']:.1f}"
            })

        # 死亡预测摘要
        if 'death_prediction' in self.results:
            best_pred = min(self.results['death_prediction'].items(),
                            key=lambda x: x[1]['test_rmse'])
            summary_data.append({
                'Analysis': 'Death Prediction',
                'Best Model': best_pred[0],
                'Performance': f"RMSE: {best_pred[1]['test_rmse']:.1f}"
            })

        summary_df = pd.DataFrame(summary_data)
        if not summary_df.empty:
            print("\n📊 ML Analysis Summary:")
            print(summary_df.to_string(index=False))

        # 额外的性别分析摘要
        if 'gender_ml' in self.results:
            print("\n" + "=" * 60)
            print("  GENDER ANALYSIS HIGHLIGHTS")
            print("=" * 60)

            gender_results = self.results['gender_ml']

            # 显示最佳模型
            if 'model_results' in gender_results:
                best_model = max(gender_results['model_results'].items(),
                                 key=lambda x: x[1]['test_accuracy'])
                print(f"\n🏆 Best Gender Classification Model: {best_model[0]}")
                print(f"   Accuracy: {best_model[1]['test_accuracy']:.3f}")
                print(f"   CV Score: {best_model[1]['cv_mean']:.3f} ± {best_model[1]['cv_std']:.3f}")

            # 显示性别差异统计
            if 'age_dominance' in gender_results:
                print(f"\n📊 Age Group Gender Dominance:")
                for _, row in gender_results['age_dominance'].iterrows():
                    dominance = "Male" if row['male_dominance_rate'] > 0.5 else "Female"
                    print(f"   {row['age_group']:15}: {dominance}-dominant ({row['male_dominance_rate']:.3f})")

            # 显示显著的年龄组
            if 'significant_age_groups' in gender_results and gender_results['significant_age_groups']:
                print(f"\n🔬 Significant Gender Differences by Age:")
                for item in sorted(gender_results['significant_age_groups'], key=lambda x: x['p_value'])[:3]:
                    print(f"   {item['age_group']}: {item['direction']} (p={item['p_value']:.6f})")

        return self.results



# 主程序
if __name__ == "__main__":
    print("=" * 60)
    print("  WHO Mortality Machine Learning Analysis")
    print("=" * 60)

    try:
        # 加载数据
        data_path = "data/processed/who_mortality_clean.csv"

        if not pd.io.common.file_exists(data_path):
            print(f"❌ Error: Processed data not found at {data_path}")
            print("   Please run data_processing.py first.")
            exit(1)

        print(f"📂 Loading data from {data_path}...")
        data = pd.read_csv(data_path)
        print(f"✅ Loaded {len(data)} records")

        # 初始化ML分析器
        analyzer = MLStatistics(data)

        # 运行所有分析
        results = analyzer.run_all_ml_tests()

        print("\n" + "=" * 60)
        print("  MACHINE LEARNING ANALYSIS COMPLETE")
        print("=" * 60)
        print("\n✅ All ML analyses completed successfully!")
        print("   Results have been stored in the analyzer.results dictionary")

    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

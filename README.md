# WHO Mortality Statistical Analysis
## Classical, Bayesian & Machine Learning Framework

## 📊 Project Overview
**Comprehensive Statistical & Machine Learning Analysis Framework for WHO Global Health Estimates 2021**

This repository implements classical statistics, Bayesian inference, and machine learning approaches for analyzing WHO mortality data, providing a complete computational framework for data processing, statistical testing, ML modeling, and visualization.

## 🎯 Analysis Framework
The project implements three complementary analytical paradigms:

### Classical Statistics (Part 1)
- Hypothesis testing (t-tests, ANOVA, chi-square)
- Correlation analysis
- Post-hoc comparisons (Tukey HSD, Games-Howell)
- Effect size calculations

### Bayesian Statistics (Part 2)
- Bayesian t-tests with Bayes factors
- Bayesian ANOVA with posterior distributions
- Bayesian contingency analysis
- MCMC sampling with PyMC

### Machine Learning & Data Science (Part 3)
- **Classification**: Gender prediction using multiple ML algorithms
- **Regression**: Death count prediction with regularization (Ridge, Lasso, ElasticNet)
- **Clustering**: Age-cause mortality pattern discovery (K-Means)
- **Bias-Variance Analysis**: Model complexity vs. generalization trade-offs
- **Feature Engineering**: Ratio calculations, logarithmic transformations, encoding

## 📁 Data Source
**WHO Global Health Estimates 2021: Deaths by Cause, Age, and Sex**
- Dataset: `ghe2021_deaths_global_new2.xlsx`
- Processed records: 1,024 observations
- Features: 128 causes of death × 8 age groups × 2 genders
- Total deaths analyzed: 64,337,460
- Source: [WHO GHO Database](https://www.who.int/data/gho/data/themes/mortality-and-global-health-estimates/ghe-leading-causes-of-death)

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- pip package manager
- C++ compiler (for PyTensor/PyMC)

### Installation & Execution

```bash
# 1. Clone repository
git clone https://github.com/tao-hpu/who-mortality-statistical-analysis.git
cd who-mortality-statistical-analysis

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt           # Classical analysis
pip install -r requirements_bayes.txt     # Bayesian analysis

# 4. Run analysis modules

## Data Processing
python src/data_processing.py             # Process raw WHO data

## Classical Analysis
python src/statistical_analysis.py        # Run statistical tests
python src/visualize_classical.py         # Generate classical visualizations

## Bayesian Analysis
# Set environment for PyTensor if needed
export PYTENSOR_FLAGS='optimizer=fast_compile,floatX=float32,device=cpu,cxx='
python src/bayes_analysis.py             # Run Bayesian analysis
python src/visualize_bayes.py            # Generate Bayesian visualizations

## Machine Learning & Data Science
python src/dsml_analysis.py              # Run ML analysis
python src/visualize_dsml.py             # Generate ML visualizations
```

## 📂 Project Structure

```
who-mortality-statistical-analysis/
│
├── src/                          # Core Python modules
│   ├── config.py                # Unified configuration
│   ├── data_processing.py       # WHO data ETL pipeline
│   ├── statistical_analysis.py  # Classical statistical tests
│   ├── bayes_analysis.py        # Bayesian statistical analysis
│   ├── dsml_analysis.py         # Machine learning analysis
│   ├── visualize_classical.py   # Classical visualization
│   ├── visualize_bayes.py       # Bayesian visualization
│   ├── visualize_dsml.py        # ML/DSML visualization
│   └── __init__.py
│
├── data/
│   ├── raw/                     # Original WHO Excel file
│   │   └── ghe2021_deaths_global_new2.xlsx
│   └── processed/               # Cleaned CSV output
│       └── who_mortality_clean.csv
│
├── figures/                      # Generated visualizations
│   ├── classical/                # Classical analysis outputs
│   │   ├── statistical_analysis_overview.png
│   │   ├── correlation_heatmap.png
│   │   └── age_mortality_pattern.png
│   ├── bayesian/                 # Bayesian analysis outputs
│   │   ├── posterior_distributions.png
│   │   ├── bayes_factors.png
│   │   ├── anova_results.png
│   │   ├── chi_square_results.png
│   │   └── regression_diagnostics.png
│   ├── dsml/                     # ML/DSML analysis outputs
│   │   ├── ml_analysis_dashboard.png
│   │   ├── gender_classification_analysis.png
│   │   ├── regression_analysis.png
│   │   ├── clustering_analysis.png
│   │   └── bias_variance_analysis.png
│   └── screenshots/              # JASP validation screenshots
│
├── notebooks/                    # Jupyter exploration notebooks
│   └── 01_initial_exploration.ipynb
│
├── assignment-docs/              # Course materials (reference)
│   ├── 期末项目-第一部分-经典统计学.md
│   ├── 期末项目-第二部分-贝叶斯统计学.md
│   └── 期末项目-第三部分-数据科学与机器学习.md
│
├── requirements.txt             # Classical dependencies
├── requirements_bayes.txt       # Bayesian dependencies
├── .pytensorrc                  # PyTensor configuration
└── README.md                    # This file
```

## 🔬 Statistical Analysis Pipeline

### 1. Data Processing (`src/data_processing.py`)
- Parse multi-level WHO Excel structure
- Extract 8 age groups × 128 causes × 2 genders
- Handle missing values and data validation
- Calculate gender ratios and age distributions
- Export standardized CSV (1,024 records)

### 2. Classical Analysis (`src/statistical_analysis.py`)

#### Hypothesis Testing Results
| Test          | Statistic   | p-value | Effect Size | Interpretation         |
| ------------- | ----------- | ------- | ----------- | ---------------------- |
| Gender T-test | t = 3.16    | 0.002   | d = 0.031   | Significant difference |
| Age ANOVA     | F = 8.78    | <0.001  | η² = 0.057  | Significant variation  |
| Chi-square    | χ² = 6.37e7 | <0.001  | V = 0.434   | Strong dependency      |

#### Key Findings
- **J-shaped mortality curve**: Exponential increase with age
- **Peak mortality**: 70+ age group (252,819 mean deaths)
- **Lowest mortality**: 5-14 age group (6,486 mean deaths)
- **Gender ratio**: Male/Female = 1.17

### 3. Bayesian Analysis (`src/bayes_analysis.py`)

#### Bayesian Test Results
| Analysis           | Bayes Factor | Evidence       | Posterior Mean | 95% HDI           |
| ------------------ | ------------ | -------------- | -------------- | ----------------- |
| Gender Comparison  | BF₁₀ = 28.5  | Strong for H₁  | δ = 0.098      | [0.031, 0.165]    |
| Age Group Effect   | BF₁₀ > 1000  | Decisive for H₁| Multiple       | See distributions |
| Independence Test  | BF₁₀ > 1000  | Decisive       | -              | -                 |

#### MCMC Configuration
- Sampling: 1000 draws, 500 tuning steps
- Chains: 4 parallel chains
- Convergence: R̂ < 1.01 for all parameters
- Effective samples: > 800 per parameter

### 4. Machine Learning Analysis (`src/dsml_analysis.py`)

#### Gender Classification Task
| Model | CV Accuracy | Test Accuracy | Best Features |
| ----- | ----------- | ------------- | ------------- |
| Logistic Regression | 72.1% ± 2.3% | 71.8% | age_numeric, log_deaths |
| Random Forest | 73.5% ± 2.1% | 73.2% | Multiple features |
| SVM | 71.8% ± 2.4% | 71.5% | age_numeric, cause_encoded |

#### Death Count Prediction (Regression)
| Model | Test R² | Test RMSE | Regularization |
| ----- | ------- | --------- | -------------- |
| Ridge | 0.845 | 48,321 | α = 1.0 |
| Lasso | 0.843 | 48,567 | α = 0.1 |
| ElasticNet | 0.844 | 48,442 | α = 0.5, l1_ratio = 0.5 |

#### Clustering Analysis
- **Optimal K**: 3 clusters (Silhouette Score: 0.452)
- **Cluster Characteristics**: Age-based mortality patterns identified
- **Feature Set**: age_numeric, log_deaths, male_ratio, cause_encoded

#### Key ML Insights
- **Data Leakage Prevention**: Removed direct male/female features from both_sexes prediction
- **Bias-Variance Trade-off**: Ridge regression provides optimal balance
- **Feature Importance**: Age is the strongest predictor across all models
- **Model Selection**: Random Forest achieves highest classification accuracy

### 5. Visualization Modules

#### Classical Visualizations (`src/visualize_classical.py`)
- Multi-panel statistical overview
- Correlation heatmaps
- Age-mortality J-curve visualization

#### Bayesian Visualizations (`src/visualize_bayes.py`)
- Posterior distributions with HDI
- Bayes factor comparisons
- MCMC trace plots
- Regression diagnostics

#### ML/DSML Visualizations (`src/visualize_dsml.py`)
- ML analysis dashboard (5 comprehensive visualizations)
- Gender classification model comparison
- Regression performance with residual analysis
- Clustering patterns with PCA projections
- Bias-variance decomposition curves

## 📊 Top 5 Causes of Death (2021)
1. **Ischaemic heart disease**: 9,033,116 deaths
2. **COVID-19**: 8,721,899 deaths
3. **Stroke**: 6,972,662 deaths
4. **COPD**: 3,519,685 deaths
5. **Lower respiratory infections**: 2,453,675 deaths

## 🛠 Technologies Used

### Core Libraries
- **Data Processing**: pandas, numpy, openpyxl
- **Classical Statistics**: scipy, statsmodels, pingouin
- **Bayesian Statistics**: PyMC, ArviZ, PyTensor
- **Machine Learning**: scikit-learn (classification, regression, clustering)
- **Visualization**: matplotlib, seaborn
- **Environment**: Python 3.9+

### Configuration Files
- `src/config.py`: Unified settings for all analyses
- `.pytensorrc`: PyTensor compiler configuration

## 🔍 Validation & Reproducibility

### Cross-validation with JASP
- Kruskal-Wallis H: **1086.00** (exact match)
- All p-values < 0.001 (consistent)
- Effect sizes within 0.01 tolerance

### Reproducibility Commands
```bash
# Classical analysis reproducibility
python src/statistical_analysis.py > classical_results.txt

# Bayesian analysis reproducibility
export PYTENSOR_FLAGS='optimizer=fast_compile,floatX=float32'
python src/bayes_analysis.py > bayesian_results.txt
```

## 📈 Analysis Workflow

1. **Data Preparation**: Process WHO Excel → Standardized CSV
2. **Classical Analysis**: Frequentist hypothesis testing
3. **Bayesian Analysis**: Posterior distributions and Bayes factors
4. **Machine Learning Analysis**: Classification, regression, clustering
5. **Visualization**: Generate publication-quality figures
6. **Validation**: Cross-check with JASP software (classical & Bayesian)

## ⚠️ Troubleshooting

### PyTensor Compilation Issues
If encountering C++ compilation errors:
```bash
# Set environment flags and clear cache
export PYTENSOR_FLAGS='optimizer=fast_compile,floatX=float32,device=cpu,cxx='
rm -rf ~/.pytensor  # Clear cache
python src/bayes_analysis.py
```

### Memory Issues with MCMC
Reduce sampling parameters in `src/config.py`:
- `MCMC_SAMPLES = 500` (from 1000)
- `MCMC_TUNE = 200` (from 500)

## 📋 Module API Documentation

### config.py
```python
# Unified configuration for all analyses
AGE_GROUP_ORDER      # Standard age group ordering
AGE_ENCODING         # Age group to numeric mapping
SIGNIFICANCE_LEVEL   # α = 0.05
MCMC_SAMPLES        # Bayesian sampling parameters
```

### data_processing.py
```python
WHODataProcessor:
  - load_data()          # Load Excel file
  - process_data()       # Clean and transform
  - save_processed_data() # Export to CSV
```

### statistical_analysis.py
```python
ClassicalStatistics:
  - descriptive_statistics()  # Summary statistics
  - gender_ttest()           # Gender comparison
  - age_group_anova()        # Age analysis
  - chi_square_test()        # Independence testing
```

### bayes_analysis.py
```python
BayesianAnalysis:
  - bayesian_ttest()         # Bayesian t-test
  - bayesian_anova()         # Bayesian ANOVA
  - bayesian_contingency()   # Bayesian chi-square
  - calculate_bayes_factors() # Evidence strength
```

### dsml_analysis.py
```python
MLStatistics:
  - gender_classification_ml()      # Multi-model gender prediction
  - unsupervised_clustering()       # K-Means with silhouette analysis
  - regularized_correlation_analysis() # Ridge/Lasso/ElasticNet
  - death_prediction_analysis()     # ML regression models
  - bias_variance_analysis()        # Model complexity evaluation
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-analysis`)
3. Commit changes (`git commit -m 'Add new analysis'`)
4. Push to branch (`git push origin feature/new-analysis`)
5. Open Pull Request

## 📝 Citation

```bibtex
@software{who_mortality_analysis_2025,
  title = {WHO Mortality Statistical Analysis: Classical, Bayesian & Machine Learning Framework},
  author = {Tao et al.},
  year = {2025},
  url = {https://github.com/tao-hpu/who-mortality-statistical-analysis},
  version = {3.0}
}
```

## 📄 License
Academic use only - MSAI Program 2025

## 📞 Support
For technical issues, please open a GitHub Issue.

---
**Latest Changes (2025-10-12):**
- Fixed data leakage issues in ML regression models (removed male/female features from both_sexes prediction)
- Added comprehensive ML visualizations (`src/visualize_dsml.py`)
- Improved clustering analysis with silhouette-based K selection
- Generated 5 publication-quality ML analysis figures

*Last Updated: 2025-10-12*
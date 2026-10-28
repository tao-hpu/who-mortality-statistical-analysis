# WHO Mortality Statistical Analysis
## Classical, Bayesian & Machine Learning Framework

[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Academic-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Statistical Framework](https://img.shields.io/badge/stats-Classical%20%7C%20Bayesian%20%7C%20ML-orange.svg)](README.md)
[![WHO Data](https://img.shields.io/badge/data-WHO%20GHE%202021-red.svg)](https://www.who.int/data/gho)

## 📑 Table of Contents
- [Project Overview](#-project-overview)
- [Analysis Framework](#-analysis-framework)
- [Demo & Visualizations](#-demo--visualizations)
- [Data Source](#-data-source)
- [System Requirements](#-system-requirements)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Statistical Analysis Pipeline](#-statistical-analysis-pipeline)
- [Top Causes of Death](#-top-5-causes-of-death-2021)
- [Technologies Used](#-technologies-used)
- [Validation & Reproducibility](#-validation--reproducibility)
- [FAQ](#-frequently-asked-questions-faq)
- [Data Ethics](#-data-ethics--responsible-use)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [Citation](#-citation)

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

## 🎨 Demo & Visualizations

### Sample Output Gallery

<details>
<summary><b>📊 Classical Statistical Analysis</b></summary>

**Key Visualizations:**
- Age-mortality J-curve showing exponential increase with age
- Correlation heatmap revealing relationships between variables
- Statistical test results with confidence intervals
- Gender comparison distributions

**Output:** `figures/classical/statistical_analysis_overview.png`
</details>

<details>
<summary><b>🔮 Bayesian Analysis Results</b></summary>

**Key Visualizations:**
- Posterior distributions with 95% Highest Density Intervals (HDI)
- Bayes factor comparison charts (evidence strength)
- MCMC trace plots demonstrating convergence
- Regression diagnostics with credible intervals

**Output:** `figures/bayesian/posterior_distributions.png`, `bayes_factors.png`
</details>

<details>
<summary><b>🤖 Machine Learning Dashboard</b></summary>

**Key Visualizations:**
- Multi-model classification performance comparison
- Regression analysis with residual plots
- Clustering patterns with PCA projections
- Bias-variance trade-off curves
- Feature importance rankings

**Output:** `figures/dsml/ml_analysis_dashboard.png`
</details>

### Quick Preview
```
Total Deaths Analyzed: 64,337,460 (WHO 2021)
Age Groups: 8 categories (0-4 to 70+)
Causes of Death: 128 distinct causes
Gender Patterns: Male/Female ratio = 1.17
Peak Mortality: 70+ age group (252,819 mean deaths per cause)
```

## 📁 Data Source
**WHO Global Health Estimates 2021: Deaths by Cause, Age, and Sex**
- Dataset: `ghe2021_deaths_global_new2.xlsx`
- Processed records: 1,024 observations
- Features: 128 causes of death × 8 age groups × 2 genders
- Total deaths analyzed: 64,337,460
- Source: [WHO GHO Database](https://www.who.int/data/gho/data/themes/mortality-and-global-health-estimates/ghe-leading-causes-of-death)

## 💻 System Requirements

### Hardware Recommendations
| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **CPU** | 2 cores | 4+ cores (for parallel MCMC) |
| **RAM** | 4 GB | 8+ GB (Bayesian MCMC sampling) |
| **Storage** | 500 MB | 1 GB (including figures) |
| **OS** | Linux/macOS/Windows | Linux (best PyTensor support) |

### Software Prerequisites
- **Python**: 3.11 (Pillow 13 requires 3.11+; `scipy==1.10.1` requires Python below 3.12)
- **Package Manager**: pip (included with Python)
- **C++ Compiler**:
  - Linux: `gcc/g++` (install via `apt-get install build-essential`)
  - macOS: Xcode Command Line Tools (`xcode-select --install`)
  - Windows: Microsoft Visual C++ Build Tools
- **Optional**: Jupyter Notebook for interactive exploration

### Estimated Execution Time
| Analysis Module | Runtime | Notes |
|-----------------|---------|-------|
| Data Processing | ~5 seconds | One-time setup |
| Classical Statistics | ~10 seconds | Fast frequentist tests |
| Bayesian Analysis | ~2-5 minutes | MCMC sampling (4 chains × 1000 draws) |
| Machine Learning | ~30 seconds | CV + multiple models |
| All Visualizations | ~15 seconds | Generate publication figures |
| **Total Pipeline** | **~3-6 minutes** | Full end-to-end analysis |

> **Note**: MCMC runtime scales with sample size. Reduce `MCMC_SAMPLES` in `config.py` for faster testing.

## 🚀 Quick Start

### One-Command Installation
```bash
# Clone and setup in one go
git clone https://github.com/tao-hpu/who-mortality-statistical-analysis.git
cd who-mortality-statistical-analysis
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt && pip install -r requirements_bayes.txt
```

### Step-by-Step Installation

```bash
# 1. Clone repository
git clone https://github.com/tao-hpu/who-mortality-statistical-analysis.git
cd who-mortality-statistical-analysis

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install all dependencies
pip install -r requirements.txt           # Classical + ML analysis (~30 packages)
pip install -r requirements_bayes.txt     # Bayesian analysis (~15 packages)
```

### Running the Full Analysis Pipeline

```bash
# 🔧 Step 1: Data Processing (~5 seconds)
python src/data_processing.py
# Output: data/processed/who_mortality_clean.csv (1,024 records)

# 📊 Step 2: Classical Statistical Analysis (~10 seconds)
python src/statistical_analysis.py
python src/visualize_classical.py
# Output: figures/classical/*.png (3 figures)

# 🔮 Step 3: Bayesian Analysis (~2-5 minutes)
export PYTENSOR_FLAGS='optimizer=fast_compile,floatX=float32,device=cpu,cxx='
python src/bayes_analysis.py
python src/visualize_bayes.py
# Output: figures/bayesian/*.png (5 figures)

# 🤖 Step 4: Machine Learning Analysis (~30 seconds)
python src/dsml_analysis.py
python src/visualize_dsml.py
# Output: figures/dsml/*.png (5 figures)
```

### Quick Test Run
```bash
# For a fast test (skip time-consuming MCMC):
python src/data_processing.py
python src/statistical_analysis.py
python src/dsml_analysis.py
# Total time: ~45 seconds
```

### Expected Output Example
```
=== Classical Statistics Results ===
Gender T-test: t=3.16, p=0.002, Cohen's d=0.031
Age ANOVA: F=8.78, p<0.001, η²=0.057
Chi-square: χ²=6.37e7, p<0.001, Cramér's V=0.434

=== Machine Learning Results ===
Best Classification Model: Random Forest (73.5% accuracy)
Best Regression Model: Ridge (R²=0.845, RMSE=48,321)
Optimal Clusters: K=3 (Silhouette=0.452)
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
- **Environment**: Python 3.11

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

## ❓ Frequently Asked Questions (FAQ)

### General Questions

**Q: Which analysis method should I use for my research question?**
- **Classical Statistics**: Use for hypothesis testing with clear null/alternative hypotheses, when you need p-values for publication, or when reviewers expect frequentist methods
- **Bayesian Statistics**: Use when you want to quantify evidence strength (Bayes factors), incorporate prior knowledge, or get probability distributions for parameters
- **Machine Learning**: Use for prediction tasks, pattern discovery (clustering), or when relationships are complex/non-linear

**Q: How long does the full analysis take?**
- Classical + ML: ~1 minute
- Bayesian MCMC: 2-5 minutes (can be reduced by lowering `MCMC_SAMPLES` in `config.py`)
- Total pipeline: 3-6 minutes

**Q: Can I use this framework with my own dataset?**
Yes! Modify `src/data_processing.py` to match your data structure. The analysis pipeline expects a CSV with columns: `age_group`, `sex`, `cause`, and `deaths`.

### Statistical Interpretation

**Q: How do I interpret Bayes Factors?**
| Bayes Factor (BF₁₀) | Evidence Strength |
|---------------------|-------------------|
| 1-3 | Anecdotal |
| 3-10 | Moderate |
| 10-30 | Strong |
| 30-100 | Very Strong |
| >100 | Decisive |

Example: BF₁₀ = 28.5 means data are 28.5× more likely under H₁ than H₀ (strong evidence)

**Q: What's the difference between effect size and p-value?**
- **p-value**: Probability of observing data if null hypothesis is true (does NOT measure effect magnitude)
- **Effect size**: Magnitude of the difference/relationship (e.g., Cohen's d, η², Cramér's V)
- Always report both! Small effects can be significant with large samples.

**Q: Why is my MCMC sampling slow?**
MCMC speed depends on:
1. Sample size (`MCMC_SAMPLES` in config.py)
2. Number of chains (4 by default)
3. Model complexity
4. CPU cores available

Solutions:
- Reduce `MCMC_SAMPLES` to 500 for testing
- Reduce `MCMC_TUNE` to 200
- Ensure C++ compiler is properly installed

### Machine Learning

**Q: Why is classification accuracy only ~73%?**
Gender classification from mortality data is inherently challenging because:
- Strong overlap in cause-of-death patterns between genders
- Age is the dominant feature (gender has smaller effect)
- 73% is significantly better than random (50%) and reflects real biological patterns

**Q: How was data leakage prevented in ML models?**
For `both_sexes` prediction:
- Removed `male_deaths` and `female_deaths` features (would cause perfect prediction)
- Only used `age_group`, `cause_of_death`, and derived features
- Proper train/test split with stratification

**Q: What's the optimal K for clustering?**
We use silhouette analysis to select K=3, representing:
1. Low-mortality causes (injuries, infectious diseases in young)
2. Medium-mortality causes (chronic diseases in middle-age)
3. High-mortality causes (cardiovascular, cancer in elderly)

### Technical Issues

**Q: I get "ModuleNotFoundError: No module named 'pytensor'"**
```bash
pip install -r requirements_bayes.txt
```
PyTensor is only needed for Bayesian analysis. You can run classical and ML analyses without it.

**Q: Bayesian analysis fails with C++ compilation error**
Set PyTensor to use fast compilation:
```bash
export PYTENSOR_FLAGS='optimizer=fast_compile,floatX=float32,device=cpu,cxx='
rm -rf ~/.pytensor
python src/bayes_analysis.py
```

**Q: How do I reproduce exact results?**
All analyses use fixed random seeds:
- Classical: scipy default behavior (deterministic)
- Bayesian: `random_seed=42` in PyMC models
- ML: `random_state=42` in scikit-learn

Run Python 3.11 and the package versions from `requirements.txt`.

**Q: Can I run this on Windows?**
Yes, but:
- Install Microsoft Visual C++ Build Tools for PyTensor
- Use `venv\Scripts\activate` instead of `source venv/bin/activate`
- Linux/macOS have better PyTensor support

### Data & Ethics

**Q: Is this real data or simulated?**
Real data from WHO Global Health Estimates 2021. Dataset represents actual global mortality statistics aggregated by WHO.

**Q: Can I publish results from this analysis?**
Yes, but:
- Cite WHO as data source
- Acknowledge this repository if using the code framework
- Follow your institution's research ethics guidelines

**Q: How often is WHO data updated?**
WHO typically releases GHE updates annually. Check [WHO GHO Database](https://www.who.int/data/gho) for latest versions.

## 🔒 Data Ethics & Responsible Use

### Data Source & Licensing
- **Source**: World Health Organization (WHO) Global Health Estimates 2021
- **Scope**: Aggregated population-level mortality statistics (no individual records)
- **License**: WHO data is publicly available for research and educational purposes
- **Attribution**: All publications must cite WHO as the primary data source

### Ethical Considerations

**Privacy Protection**
- Dataset contains only aggregated counts (no personally identifiable information)
- Minimum cell sizes ensure individual-level de-identification
- Complies with epidemiological data sharing standards

**Appropriate Use**
✅ **Permitted**:
- Academic research and education
- Public health policy analysis
- Statistical methodology development
- Epidemiological trend analysis

❌ **Prohibited**:
- Commercial use without WHO permission
- Misrepresentation of findings
- Drawing causal conclusions without appropriate study design
- Using data to stigmatize populations

**Interpretation Guidelines**
- Correlation does not imply causation (observational data)
- Aggregated data may mask within-group heterogeneity
- Cultural, socioeconomic, and healthcare access factors influence mortality patterns
- Always contextualize findings with domain expertise

**Bias & Limitations Awareness**
- Data quality varies by country reporting infrastructure
- Some causes of death may be underreported (e.g., mental health, substance abuse)
- Age groupings mask fine-grained patterns
- Gender binary classification may not reflect all populations

### Responsible Reporting
When publishing analyses:
1. Acknowledge data limitations explicitly
2. Avoid sensationalist language about mortality risks
3. Consider public health messaging implications
4. Consult domain experts for clinical/policy interpretation

### Contact for Ethical Concerns
For questions about data use ethics:
- WHO Data Repository: https://www.who.int/data
- Institutional Review Board (IRB) for research involving human subjects data

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
**Latest Changes (2025-10-21):**
- ✨ Enhanced README with comprehensive documentation improvements
- 📊 Added project badges for quick status overview
- 📑 Added navigable Table of Contents for better user experience
- 🎨 Added Demo & Visualizations section with collapsible examples
- 💻 Added detailed System Requirements with hardware/software specs
- ⚡ Improved Quick Start with one-command installation and expected outputs
- ❓ Added comprehensive FAQ covering 15+ common questions
- 🔒 Added Data Ethics & Responsible Use guidelines
- 📈 Added execution time estimates for all analysis modules

**Previous Changes (2025-10-12):**
- Fixed data leakage issues in ML regression models (removed male/female features from both_sexes prediction)
- Added comprehensive ML visualizations (`src/visualize_dsml.py`)
- Improved clustering analysis with silhouette-based K selection
- Generated 5 publication-quality ML analysis figures

*Last Updated: 2025-10-21*
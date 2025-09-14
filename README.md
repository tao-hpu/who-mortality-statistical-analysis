# WHO Mortality Statistical Analysis (Python Implementation)

## 📊 Project Overview
**Python-based Statistical Analysis Framework for WHO Global Health Estimates 2021**

This repository contains the Python implementation for analyzing WHO mortality data using classical statistical methods. It serves as a computational framework for data processing, statistical testing, and visualization.

> **Note**: This is the pure Python implementation. Course assignment documents (including JASP analyses) are located in the `assignment-docs/` directory for reference only.

## 🎯 Project Focus
This repository focuses on **programmatic statistical analysis** using Python:
- Automated data processing pipeline
- Reproducible statistical tests
- Code-based visualization
- Open-source implementation

## 📁 Data Source
WHO Global Health Estimates 2021: Deaths by Cause, Age, and Sex
- Dataset: `ghe2021_deaths_global_new2.xlsx`
- Processed records: 1,024 observations
- Features: 128 causes of death × 8 age groups × 2 genders
- Total deaths analyzed: 64,337,460
- Source: [WHO GHO Database](https://www.who.int/data/gho/data/themes/mortality-and-global-health-estimates/ghe-leading-causes-of-death)

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- pip package manager

### Installation & Execution

```bash
# 1. Clone repository
git clone https://github.com/tao-hpu/who-mortality-statistical-analysis.git
cd who-mortality-statistical-analysis

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run complete analysis pipeline
python run_analysis.py

# Or run modules separately:
python src/data_processing.py        # Process raw data
python src/statistical_analysis.py   # Run statistical tests
python visualize_results.py          # Generate visualizations
```

## 📂 Project Structure

```
who-mortality-statistical-analysis/
│
├── src/                          # Core Python modules
│   ├── data_processing.py       # WHO data ETL pipeline
│   ├── statistical_analysis.py  # Statistical tests implementation
│   └── __init__.py
│
├── data/
│   ├── raw/                     # Original WHO Excel file
│   │   └── ghe2021_deaths_global_new2.xlsx
│   └── processed/               # Cleaned CSV output
│       └── who_mortality_clean.csv
│
├── figures/                      # Generated visualizations
│   ├── statistical_analysis_overview.png
│   ├── correlation_heatmap.png
│   └── age_mortality_pattern.png
│
├── notebooks/                    # Jupyter exploration notebooks
│   └── 01_initial_exploration.ipynb
│
├── assignment-docs/              # Course assignment materials (reference only)
│   └── 期末项目-第一部分-经典统计学.md
│
├── run_analysis.py              # Main execution script
├── visualize_results.py         # Visualization generator
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🔬 Statistical Analysis Pipeline

### 1. Data Processing Module (`src/data_processing.py`)
- **Input**: WHO Excel file with mortality data
- **Processing**:
  - Parse multi-level Excel structure
  - Extract 8 age groups × 128 causes
  - Handle missing values and data validation
  - Calculate gender ratios
- **Output**: Standardized CSV with 1,024 records

### 2. Statistical Analysis Module (`src/statistical_analysis.py`)
Implements comprehensive statistical testing:

#### Descriptive Statistics
- Mean, median, standard deviation
- Quartiles and range
- Distribution characteristics

#### Hypothesis Testing
- **Gender Differences**: Paired t-test (t=3.16, p=0.002)
- **Age Group Comparisons**: One-way ANOVA (F=8.78, p<0.001)
- **Independence Testing**: Chi-square test (χ²=6.37e7, p<0.001)
- **Correlation Analysis**: Pearson correlations (age vs deaths: r=0.189)
- **Post-hoc Analysis**: Tukey HSD for pairwise comparisons

### 3. Visualization Module (`visualize_results.py`)
Generates publication-quality figures:
- Multi-panel statistical overview
- Correlation heatmaps
- U-shaped mortality curve visualization

## 📊 Key Findings

### Statistical Results Summary
| Test | Statistic | p-value | Effect Size | Interpretation |
|------|-----------|---------|-------------|----------------|
| Gender T-test | t = 3.16 | 0.002 | d = 0.031 | Significant difference |
| Age ANOVA | F = 8.78 | <0.001 | η² = 0.057 | Significant variation |
| Chi-square | χ² = 6.37e7 | <0.001 | V = 0.434 | Strong dependency |

### Mortality Patterns
- **U-shaped distribution**: High in infancy → Low in youth → Rising with age
- **Peak mortality**: 70+ age group (252,819 mean deaths)
- **Lowest mortality**: 15-29 age group (14,847 mean deaths)
- **Gender ratio**: Male/Female = 1.17

### Top 5 Causes of Death (2021)
1. Ischaemic heart disease: 9,033,116
2. COVID-19: 8,721,899
3. Stroke: 6,972,662
4. COPD: 3,519,685
5. Lower respiratory infections: 2,453,675

## 🛠 Technologies Used
- **Data Processing**: pandas, numpy, openpyxl
- **Statistical Analysis**: scipy, statsmodels
- **Visualization**: matplotlib, seaborn
- **Environment**: Python 3.9+

## 📈 Progress Status

### ✅ Completed Tasks
- [x] Project framework setup
- [x] Data processing pipeline
- [x] Statistical analysis implementation
- [x] Visualization generation
- [x] Results validation
- [x] Documentation

### 🔄 Current Status
- All Week 1 deliverables completed
- Python implementation fully functional
- Results validated against JASP software

## 🔍 Validation & Reproducibility

This implementation has been validated against JASP statistical software:
- Kruskal-Wallis H statistic: **1086.00** (exact match)
- All p-values < 0.001 (consistent)
- Effect sizes within 0.01 tolerance

To reproduce results:
```bash
python src/statistical_analysis.py > results.txt
```

## 📋 Module Documentation

### data_processing.py
```python
WHODataProcessor: Main class for data handling
- load_data(): Load Excel file
- parse_structure(): Identify data regions
- process_data(): Clean and transform
- save_processed_data(): Export to CSV
```

### statistical_analysis.py
```python
ClassicalStatistics: Statistical testing suite
- descriptive_statistics(): Summary stats
- gender_ttest(): Gender comparison
- age_group_anova(): Age analysis
- correlation_analysis(): Variable relationships
- chi_square_test(): Independence testing
```

## ⚠️ Important Notes

1. **This repository contains only Python implementations** - No JASP or other proprietary software dependencies
2. **Assignment documents** in `assignment-docs/` are for reference only
3. **Data file required**: Place WHO Excel file in `data/raw/` before running
4. **Virtual environment recommended** to ensure package compatibility

## 🤝 Contributing

For code contributions:
1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-analysis`)
3. Commit changes (`git commit -m 'Add new analysis'`)
4. Push to branch (`git push origin feature/new-analysis`)
5. Open Pull Request

## 📝 Citation

If using this code for research:
```
WHO Mortality Statistical Analysis (2025).
Python Implementation for WHO Global Health Estimates 2021.
https://github.com/tao-hpu/who-mortality-statistical-analysis
```

## 📄 License
Academic use only - MSAI Program 2025

## 📞 Support
For technical issues, please open a GitHub Issue or contact the repository maintainer.

---
*Last Updated: 2025-09-14*
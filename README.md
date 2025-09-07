# WHO Mortality Statistical Analysis

## 📊 Project Overview
MSAI Course Final Project - Statistical Analysis of WHO Global Health Estimates 2021

## 👥 Team Members
- [Tao] - Team Lead & Technical
- [木南] - Technical 
- [陈宝成] - Technical
- [余晗] - Research & Documentation
- [李春旭] - Analysis & Visualization

## 🎯 Objectives
Analyze global mortality patterns using three statistical approaches:
1. Classical Statistics
2. Bayesian Statistics  
3. Machine Learning & Data Science

## 📁 Data Source
WHO Global Health Estimates 2021: Deaths by Cause, Age, and Sex
- Dataset: Global summary estimates (ghe2021_deaths_global_new2.xlsx)
- Records: 1,400+ observations
- Dimensions: Cause of death, Age groups, Sex
- Download: [WHO GHO Leading Causes of Death](https://www.who.int/data/gho/data/themes/mortality-and-global-health-estimates/ghe-leading-causes-of-death)

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.9 or higher
- Git

### Setup Instructions

#### Windows Users
```bash
# 1. Clone the repository
git clone https://github.com/[username]/who-mortality-statistical-analysis.git
cd who-mortality-statistical-analysis

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Place data file
# Put ghe2021_deaths_global_new2.xlsx in data/raw/ folder

# 6. Run data processing
python src/data_processing.py

# 7. Run statistical analysis
python src/statistical_analysis.py
```

#### Mac/Linux Users
```bash
# 1. Clone the repository
git clone https://github.com/[username]/who-mortality-statistical-analysis.git
cd who-mortality-statistical-analysis

# 2. Create virtual environment
python3 -m venv venv

# 3. Activate virtual environment
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Place data file
# Put ghe2021_deaths_global_new2.xlsx in data/raw/ folder

# 6. Run data processing
python3 src/data_processing.py

# 7. Run statistical analysis
python3 src/statistical_analysis.py
```

#### Deactivate Virtual Environment
```bash
deactivate
```

## 📂 Project Structure
```
who-mortality-statistical-analysis/
│
├── data/
│   ├── raw/                   # Original data files (place Excel here)
│   │   └── ghe2021_deaths_global_new2.xlsx
│   └── processed/              # Cleaned data (auto-generated)
│       └── who_mortality_clean.csv
│
├── src/
│   ├── data_processing.py     # Data loading and cleaning module
│   ├── statistical_analysis.py # Statistical tests (t-test, ANOVA, etc.)
│   ├── visualization.py        # Visualization module (TBD)
│   └── utils.py               # Utility functions (TBD)
│
├── notebooks/                  # Jupyter notebooks (optional)
│   └── 01_initial_exploration.ipynb
│
├── reports/
│   ├── figures/               # Generated plots
│   └── tables/                # Statistical tables
│
├── docs/
│   ├── meeting_notes/         # Team meeting records
│   └── references/            # Literature and references
│
├── requirements.txt           # Python dependencies
├── .gitignore                # Git ignore file
├── run_analysis.py           # One-click analysis script
└── README.md                 # This file
```

## 🔧 Technologies
- Python 3.9+
- Pandas, NumPy, SciPy
- Matplotlib, Seaborn
- Scikit-learn
- PyMC3 (for Bayesian analysis)
- Statsmodels

## 📊 Analysis Pipeline

### Part 1: Classical Statistics (Week 1)
- **Data Processing** (`src/data_processing.py`)
  - Load WHO Excel data
  - Clean and reshape data
  - Handle missing values
  - Export to CSV format

- **Statistical Analysis** (`src/statistical_analysis.py`)
  - Descriptive statistics
  - T-tests for gender differences
  - ANOVA for age group comparisons
  - Chi-square test for independence
  - Correlation analysis

### Part 2: Bayesian Statistics (Week 2)
- Prior distribution selection
- Posterior estimation
- Credible intervals
- Model comparison

### Part 3: Machine Learning (Week 3)
- Feature engineering
- Model selection
- Cross-validation
- Performance evaluation

## 📋 Week 1 Task Assignment

| Team Member | Task                                                  | Deadline | Status |
| ----------- | ----------------------------------------------------- | -------- | ------ |
| Tao (Lead)  | Project setup, framework, initial data exploration    | Day 3    | ✅      |
| 木南        | Data cleaning, descriptive statistics implementation  | Day 5    | ⏳      |
| 陈宝成      | Statistical methods research, hypothesis testing code | Day 5    | ⏳      |
| 余晗        | Literature review, WHO methodology documentation      | Day 4    | ⏳      |
| 李春旭      | Report template, visualization planning               | Day 4    | ⏳      |

## 🔍 Research Questions

### Proposed Questions for Part 1
1. **Gender Disparities**: Are there significant differences in mortality rates between males and females across different causes of death?
2. **Age Patterns**: How do leading causes of death vary across age groups?
3. **Disease Transitions**: What are the dominant mortality patterns (communicable vs non-communicable diseases)?
4. **Risk Factors**: Can we identify correlations between age, gender, and specific causes of death?

## 📈 Progress
- [x] Data collection
- [x] Project setup
- [x] Basic code framework
- [ ] Data cleaning
- [ ] Exploratory analysis
- [ ] Classical statistical analysis
- [ ] Bayesian analysis
- [ ] Machine learning models
- [ ] Final report

## 🤝 Collaboration Guidelines

### Git Workflow
1. Pull latest changes: `git pull origin main`
2. Create your branch: `git checkout -b dev-yourname`
3. Make changes and commit: `git add .` and `git commit -m "description"`
4. Push to GitHub: `git push origin dev-yourname`
5. Create Pull Request for review

### Code Standards
- Add docstrings to all functions
- Comment complex logic
- Follow PEP 8 style guide
- Test code before committing

### Communication
- Daily progress updates in WeChat group
- Weekly team meetings (Thursdays 8 PM)
- Use GitHub Issues for bug tracking

## ❓ FAQ

**Q: Why use a virtual environment?**  
A: To ensure consistent package versions across all team members and avoid conflicts.

**Q: Where to place the Excel data file?**  
A: Put `ghe2021_deaths_global_new2.xlsx` in the `data/raw/` folder.

**Q: How to update dependencies?**  
A: Run `pip install -r requirements.txt` after pulling new code.

**Q: What if I get import errors?**  
A: Make sure your virtual environment is activated and all packages are installed.

## 📞 Contact
- WeChat Group: MSAI Project Team
- Emergency: Contact Tao (Team Lead)

## 📝 License
Academic use only - MSAI Program 2025
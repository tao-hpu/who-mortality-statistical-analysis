"""
统一配置文件，确保所有脚本使用相同的标签和顺序
"""

# 年龄组的标准顺序（从年轻到年老）
AGE_GROUP_ORDER = [
    "0-28 days",
    "1-59 months",
    "5-14",
    "15-29",
    "30-49",
    "50-59",
    "60-69",
    "70+"
]

# 年龄组到数值的映射（用于统计分析）
AGE_ENCODING = {
    "0-28 days": 0.04,      # 约0.04年（14天）
    "1-59 months": 2.5,     # 约2.5年（30个月）
    "5-14": 9.5,            # 9.5年（中位数）
    "15-29": 22,            # 22年（中位数）
    "30-49": 39.5,          # 39.5年（中位数）
    "50-59": 54.5,          # 54.5年（中位数）
    "60-69": 64.5,          # 64.5年（中位数）
    "70+": 75,              # 75年（代表值）
}

# 死因代码映射（如需要）
CAUSE_CATEGORIES = {
    "communicable": "传染性疾病",
    "noncommunicable": "非传染性疾病",
    "injuries": "伤害"
}

# 数据文件路径
DATA_PATH = "data/processed/who_mortality_clean.csv"
RAW_DATA_PATH = "data/raw/WHOMortalityDatabase.csv"

# 图表保存路径
FIGURES_PATH = "figures/"

# 统计分析参数
SIGNIFICANCE_LEVEL = 0.05  # 显著性水平
CONFIDENCE_INTERVAL = 0.95  # 置信区间

# 贝叶斯分析参数
MCMC_SAMPLES = 1000  # MCMC采样数
MCMC_TUNE = 500      # MCMC调优步数
RANDOM_SEED = 42     # 随机种子
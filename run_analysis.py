#!/usr/bin/env python
"""
一键运行完整分析流程
One-click script to run the complete analysis pipeline

Usage:
    python run_analysis.py
"""

import os
import sys
from pathlib import Path


def print_header(text):
    """打印格式化的标题"""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def check_environment():
    """检查运行环境"""
    print_header("环境检查 / Environment Check")

    issues = []

    # 检查Python版本
    python_version = sys.version_info
    if python_version.major < 3 or (
        python_version.major == 3 and python_version.minor < 9
    ):
        issues.append(f"❌ Python版本过低: {sys.version.split()[0]} (需要 3.9+)")
    else:
        print(f"✅ Python版本: {sys.version.split()[0]}")

    # 检查数据文件
    data_file = Path("data/raw/ghe2021_deaths_global_new2.xlsx")
    if not data_file.exists():
        issues.append(f"❌ 未找到数据文件: {data_file}")
        print(f"❌ 数据文件未找到")
        print(f"   请将Excel文件放到: data/raw/")
    else:
        print(f"✅ 数据文件已找到: {data_file.name}")

    # 检查必要的包
    required_packages = {
        "pandas": "数据处理",
        "numpy": "数值计算",
        "scipy": "统计分析",
        "matplotlib": "可视化",
        "openpyxl": "Excel读取",
    }

    print("\n检查依赖包:")
    for package, description in required_packages.items():
        try:
            __import__(package)
            print(f"  ✅ {package:<12} - {description}")
        except ImportError:
            issues.append(f"❌ 缺少包: {package} ({description})")
            print(f"  ❌ {package:<12} - 未安装")

    # 汇总问题
    if issues:
        print("\n" + "⚠️  发现以下问题 / Issues Found:")
        for issue in issues:
            print(f"  {issue}")
        print("\n解决方案:")
        print("  1. 确保Python版本 >= 3.9")
        print(
            "  2. 激活虚拟环境: venv\\Scripts\\activate (Windows) 或 source venv/bin/activate (Mac/Linux)"
        )
        print("  3. 安装依赖: pip install -r requirements.txt")
        print("  4. 下载数据文件到 data/raw/ 文件夹")
        return False

    print("\n✅ 环境检查通过！")
    return True


def create_directories():
    """创建必要的目录结构"""
    directories = [
        "data/raw",
        "data/processed",
        "reports/figures",
        "reports/tables",
        "docs/meeting_notes",
        "docs/references",
    ]

    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        # 创建.gitkeep文件保持空文件夹
        gitkeep = Path(dir_path) / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.touch()

    print("✅ 目录结构已创建")


def run_data_processing():
    """运行数据处理模块"""
    print_header("步骤 1/3: 数据处理 / Data Processing")

    try:
        # 导入数据处理模块
        from src.data_processing import WHODataProcessor

        # 初始化处理器
        processor = WHODataProcessor("data/raw/ghe2021_deaths_global_new2.xlsx")

        # 加载数据
        print("📂 加载原始数据...")
        processor.load_data()

        # 处理数据
        print("🔧 处理数据...")
        df = processor.process_data()

        # 保存处理后的数据
        print("💾 保存清洗后的数据...")
        processor.save_processed_data()

        # 显示摘要
        summary = processor.get_summary_stats()
        print(f"\n📊 数据摘要:")
        print(f"   - 总记录数: {len(df):,}")
        print(f"   - 死因数量: {df['cause_name'].nunique()}")
        print(f"   - 年龄组数: {df['age_group'].nunique()}")
        print(f"   - 总死亡人数: {summary['total_deaths']:,.0f}")

        print("\n✅ 数据处理完成！")
        return df

    except Exception as e:
        print(f"\n❌ 数据处理失败: {e}")
        return None


def run_statistical_analysis(df):
    """运行统计分析模块"""
    print_header("步骤 2/3: 统计分析 / Statistical Analysis")

    try:
        # 导入统计分析模块
        from src.statistical_analysis import ClassicalStatistics

        # 初始化分析器
        print("📈 执行统计检验...")
        stats_analyzer = ClassicalStatistics(df)

        # 运行所有检验
        results = stats_analyzer.run_all_tests()

        # 显示结果摘要
        print(f"\n📊 统计检验结果:")
        print(
            f"   - 性别差异 (T-test): {'显著' if results['gender_ttest']['significant'] else '不显著'} (p={results['gender_ttest']['p_value']:.4f})"
        )
        print(
            f"   - 年龄组差异 (ANOVA): {'显著' if results['age_anova']['significant'] else '不显著'} (p={results['age_anova']['p_value']:.4f})"
        )
        print(
            f"   - 死因-年龄关联 (Chi²): {'相关' if results['chi_square']['significant'] else '独立'} (p={results['chi_square']['p_value']:.4f})"
        )

        print("\n✅ 统计分析完成！")
        return results

    except Exception as e:
        print(f"\n❌ 统计分析失败: {e}")
        return None


def generate_report(df, results):
    """生成分析报告摘要"""
    print_header("步骤 3/3: 生成报告 / Generate Report")

    try:
        # 创建报告
        report_path = Path("reports/analysis_summary.txt")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("WHO MORTALITY STATISTICAL ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")

            f.write("1. DATA SUMMARY\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total Records: {len(df):,}\n")
            f.write(f"Unique Causes: {df['cause_name'].nunique()}\n")
            f.write(f"Age Groups: {', '.join(df['age_group'].unique())}\n")
            f.write(f"Total Deaths: {df['both_sexes'].sum():,.0f}\n")
            f.write(f"Male Deaths: {df['male'].sum():,.0f}\n")
            f.write(f"Female Deaths: {df['female'].sum():,.0f}\n\n")

            f.write("2. TOP 10 CAUSES OF DEATH\n")
            f.write("-" * 30 + "\n")
            top_causes = df.groupby("cause_name")["both_sexes"].sum().nlargest(10)
            for i, (cause, deaths) in enumerate(top_causes.items(), 1):
                f.write(f"{i:2}. {cause}: {deaths:,.0f}\n")

            f.write("\n3. STATISTICAL TEST RESULTS\n")
            f.write("-" * 30 + "\n")
            f.write(
                f"Gender T-Test: p-value = {results['gender_ttest']['p_value']:.4f}\n"
            )
            f.write(f"Age ANOVA: p-value = {results['age_anova']['p_value']:.4f}\n")
            f.write(f"Chi-Square: p-value = {results['chi_square']['p_value']:.4f}\n")

        print(f"📄 报告已生成: {report_path}")
        print("\n✅ 报告生成完成！")
        return True

    except Exception as e:
        print(f"\n❌ 报告生成失败: {e}")
        return False


def main():
    """主函数"""
    print("\n" + "🔬" * 30)
    print("  WHO MORTALITY STATISTICAL ANALYSIS")
    print("  MSAI 小组项目 - 自动分析脚本")
    print("🔬" * 30)

    # 1. 检查环境
    if not check_environment():
        print("\n⚠️  请修复上述问题后重试")
        sys.exit(1)

    # 2. 创建目录
    create_directories()

    # 3. 运行数据处理
    df = run_data_processing()
    if df is None:
        print("\n⚠️  数据处理失败，请检查错误信息")
        sys.exit(1)

    # 4. 运行统计分析
    results = run_statistical_analysis(df)
    if results is None:
        print("\n⚠️  统计分析失败，请检查错误信息")
        sys.exit(1)

    # 5. 生成报告
    success = generate_report(df, results)

    # 完成
    if success:
        print("\n" + "🎉" * 30)
        print("  恭喜！所有分析已成功完成！")
        print("  Congratulations! All analyses completed successfully!")
        print("🎉" * 30)
        print("\n下一步:")
        print("  1. 查看处理后的数据: data/processed/who_mortality_clean.csv")
        print("  2. 查看分析报告: reports/analysis_summary.txt")
        print("  3. 运行可视化脚本 (开发中)")
    else:
        print("\n⚠️  分析过程中出现错误，请检查日志")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断运行")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 程序异常: {e}")
        sys.exit(1)

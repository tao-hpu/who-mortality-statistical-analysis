"""
WHO死亡率数据处理模块
Data processing module for WHO mortality data

主要功能:
1. 加载Excel原始数据
2. 数据清洗和转换
3. 生成标准化数据格式
4. 导出处理后的数据
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")


class WHODataProcessor:
    """
    WHO死亡数据处理器
    用于处理Global Health Estimates 2021数据
    """

    def __init__(self, filepath="data/raw/ghe2021_deaths_global_new2.xlsx"):
        """
        初始化数据处理器

        Parameters:
        -----------
        filepath : str or Path
            Excel数据文件路径
        """
        self.filepath = Path(filepath)
        self.raw_data = None
        self.processed_data = None
        self.metadata = {}

    def load_data(self, sheet_name="Global 2021"):
        """
        加载Excel数据文件

        Parameters:
        -----------
        sheet_name : str
            Excel工作表名称，默认为'Global 2021'

        Returns:
        --------
        pd.DataFrame
            原始数据框
        """
        print(f"📂 Loading data from {self.filepath}...")

        if not self.filepath.exists():
            raise FileNotFoundError(f"Data file not found: {self.filepath}")

        try:
            # 读取整个Excel文件，不设置header
            self.raw_data = pd.read_excel(
                self.filepath, sheet_name=sheet_name, header=None
            )

            print(
                f"✅ Loaded {self.raw_data.shape[0]} rows, {self.raw_data.shape[1]} columns"
            )

        except Exception as e:
            raise Exception(f"Failed to load Excel file: {e}")

        return self.raw_data

    def parse_structure(self):
        """
        解析数据结构，识别元数据和数据区域

        Returns:
        --------
        int
            实际数据开始的行号
        """
        if self.raw_data is None:
            raise ValueError("Please load data first using load_data()")

        print("🔍 Parsing data structure...")

        # 查找包含'Population'的行，这通常是数据开始的标志
        data_start_row = None
        for idx in range(min(20, len(self.raw_data))):  # 只检查前20行
            row = self.raw_data.iloc[idx]
            if pd.notna(row[0]) and "Population" in str(row[0]):
                data_start_row = idx
                break

        # 如果没找到，使用默认值
        if data_start_row is None:
            # 查找包含数字数据的第一行
            for idx in range(5, min(20, len(self.raw_data))):
                row = self.raw_data.iloc[idx]
                # 检查是否有多个数值列
                numeric_count = sum(
                    [isinstance(val, (int, float)) for val in row[4:10]]
                )
                if numeric_count >= 3:
                    data_start_row = idx
                    break

        if data_start_row is None:
            data_start_row = 6  # 默认值

        # 提取元数据（从前几行）
        try:
            # 通常第2行包含地区信息，第3行包含年份
            for i in range(min(5, len(self.raw_data))):
                row = self.raw_data.iloc[i]
                if pd.notna(row[0]):
                    if "Region" in str(row[0]):
                        self.metadata["region"] = (
                            row[1] if pd.notna(row[1]) else "Global"
                        )
                    elif "Year" in str(row[0]):
                        self.metadata["year"] = (
                            int(row[1]) if pd.notna(row[1]) else 2021
                        )
        except:
            # 使用默认值
            self.metadata["region"] = "Global"
            self.metadata["year"] = 2021

        print(f"📍 Region: {self.metadata.get('region', 'Global')}")
        print(f"📅 Year: {self.metadata.get('year', 2021)}")
        print(
            f"📊 Data starts at row: {data_start_row + 1} (0-indexed: {data_start_row})"
        )

        return data_start_row

    def process_data(self):
        """
        处理原始数据为标准化格式

        Returns:
        --------
        pd.DataFrame
            处理后的数据框，包含以下列：
            - cause_code: 死因代码
            - cause_name: 死因名称
            - age_group: 年龄组
            - both_sexes: 总死亡人数
            - male: 男性死亡人数
            - female: 女性死亡人数
            - male_female_ratio: 男女比例
        """
        print("🔧 Processing data...")

        # 获取数据开始位置
        data_start = self.parse_structure()

        # 定义年龄组
        age_groups = [
            "0-28 days",
            "1-59 months",
            "5-14",
            "15-29",
            "30-49",
            "50-59",
            "60-69",
            "70+",
        ]

        # 存储处理后的数据
        processed_rows = []

        # 遍历数据行
        current_row = data_start
        valid_rows = 0

        while (
            current_row < len(self.raw_data) and valid_rows < 200
        ):  # 限制最多200个死因
            row = self.raw_data.iloc[current_row]

            # 检查第一列是否有值（死因代码）
            if pd.isna(row[0]) or row[0] == "" or str(row[0]).strip() == "":
                current_row += 1
                continue

            # 检查是否是注释行或标题行
            if any(
                keyword in str(row[0]).lower()
                for keyword in ["note", "source", "total", "sum"]
            ):
                current_row += 1
                continue

            cause_code = str(row[0]).strip()
            cause_name = str(row[1]).strip() if pd.notna(row[1]) else "Unknown"

            # 跳过无效的死因名称
            if cause_name == "Unknown" or cause_name == "" or pd.isna(row[1]):
                current_row += 1
                continue

            try:
                # 数据列的位置（根据Excel结构调整）
                # Both sexes: 通常在第5-13列
                # Male: 通常在第16-24列
                # Female: 通常在第27-35列

                # 提取Both sexes数据（跳过Total列）
                both_data = []
                for i, age in enumerate(age_groups):
                    col_idx = 6 + i  # 从第7列开始（0-indexed为6）
                    val = row[col_idx] if col_idx < len(row) else 0
                    both_data.append(float(val) if pd.notna(val) and val != "" else 0)

                # 提取Male数据
                male_data = []
                for i, age in enumerate(age_groups):
                    col_idx = 17 + i  # Male数据起始列
                    val = row[col_idx] if col_idx < len(row) else 0
                    male_data.append(float(val) if pd.notna(val) and val != "" else 0)

                # 提取Female数据
                female_data = []
                for i, age in enumerate(age_groups):
                    col_idx = 28 + i  # Female数据起始列
                    val = row[col_idx] if col_idx < len(row) else 0
                    female_data.append(float(val) if pd.notna(val) and val != "" else 0)

                # 为每个年龄组创建一条记录
                for i, age_group in enumerate(age_groups):
                    processed_rows.append(
                        {
                            "cause_code": cause_code,
                            "cause_name": cause_name,
                            "age_group": age_group,
                            "both_sexes": both_data[i],
                            "male": male_data[i],
                            "female": female_data[i],
                        }
                    )

                valid_rows += 1

            except Exception as e:
                print(f"⚠️  Warning: Error processing row {current_row}: {e}")

            current_row += 1

        # 创建DataFrame
        self.processed_data = pd.DataFrame(processed_rows)

        if len(self.processed_data) == 0:
            raise ValueError(
                "No valid data found. Please check the Excel file structure."
            )

        # 数据类型转换和清洗
        numeric_columns = ["both_sexes", "male", "female"]
        for col in numeric_columns:
            self.processed_data[col] = pd.to_numeric(
                self.processed_data[col], errors="coerce"
            ).fillna(0)
            # 确保没有负数
            self.processed_data[col] = self.processed_data[col].abs()

        # 计算男女比例
        self.processed_data["male_female_ratio"] = np.where(
            self.processed_data["female"] > 0,
            self.processed_data["male"] / self.processed_data["female"],
            np.nan,
        )

        # 数据验证
        print("\n📊 Data Processing Summary:")
        print(f"   - Processed records: {len(self.processed_data)}")
        print(f"   - Unique causes: {self.processed_data['cause_name'].nunique()}")
        print(f"   - Age groups: {self.processed_data['age_group'].nunique()}")
        print(f"   - Total deaths: {self.processed_data['both_sexes'].sum():,.0f}")

        # 检查数据质量
        zero_rows = (
            self.processed_data[["both_sexes", "male", "female"]].sum(axis=1) == 0
        ).sum()
        if zero_rows > 0:
            print(f"   ⚠️  Warning: {zero_rows} rows with zero deaths")

        return self.processed_data

    def save_processed_data(self, output_path="data/processed/who_mortality_clean.csv"):
        """
        保存处理后的数据到CSV文件

        Parameters:
        -----------
        output_path : str or Path
            输出文件路径
        """
        if self.processed_data is None:
            raise ValueError("No processed data to save. Run process_data() first.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 保存CSV
        self.processed_data.to_csv(output_path, index=False, encoding="utf-8")
        print(f"💾 Saved processed data to {output_path}")
        print(f"   File size: {output_path.stat().st_size / 1024:.1f} KB")

    def get_summary_stats(self):
        """
        获取数据的摘要统计信息

        Returns:
        --------
        dict
            包含各种统计摘要的字典
        """
        if self.processed_data is None:
            raise ValueError("No processed data. Run process_data() first.")

        df = self.processed_data

        summary = {
            "total_deaths": df["both_sexes"].sum(),
            "male_deaths": df["male"].sum(),
            "female_deaths": df["female"].sum(),
            "num_causes": df["cause_name"].nunique(),
            "num_age_groups": df["age_group"].nunique(),
            "num_records": len(df),
            "top_causes": df.groupby("cause_name")["both_sexes"].sum().nlargest(10),
            "age_distribution": df.groupby("age_group")["both_sexes"]
            .sum()
            .sort_values(ascending=False),
            "gender_ratio": (
                df["male"].sum() / df["female"].sum()
                if df["female"].sum() > 0
                else np.nan
            ),
        }

        # 添加元数据
        summary["metadata"] = self.metadata

        return summary

    def validate_data(self):
        """
        验证处理后的数据质量

        Returns:
        --------
        dict
            验证结果
        """
        if self.processed_data is None:
            raise ValueError("No processed data to validate.")

        df = self.processed_data
        validation = {
            "has_nulls": df.isnull().any().any(),
            "has_negatives": (df[["both_sexes", "male", "female"]] < 0).any().any(),
            "gender_consistency": [],
            "age_groups_complete": [],
            "total_checks": [],
        }

        # 检查性别一致性（男+女应该约等于总体）
        tolerance = 0.1  # 10%容差
        for _, row in df.iterrows():
            if row["both_sexes"] > 0:
                expected = row["male"] + row["female"]
                diff = abs(row["both_sexes"] - expected) / row["both_sexes"]
                if diff > tolerance:
                    validation["gender_consistency"].append(
                        {
                            "cause": row["cause_name"],
                            "age": row["age_group"],
                            "both": row["both_sexes"],
                            "sum_m_f": expected,
                            "diff_pct": diff * 100,
                        }
                    )

        # 检查每个死因是否有完整的年龄组数据
        expected_age_groups = df["age_group"].nunique()
        for cause in df["cause_name"].unique():
            cause_data = df[df["cause_name"] == cause]
            if len(cause_data) != expected_age_groups:
                validation["age_groups_complete"].append(
                    {
                        "cause": cause,
                        "found": len(cause_data),
                        "expected": expected_age_groups,
                    }
                )

        return validation


# 主程序
if __name__ == "__main__":
    """
    直接运行此脚本进行数据处理
    """
    print("=" * 60)
    print("  WHO Mortality Data Processing")
    print("=" * 60)

    try:
        # 初始化处理器
        processor = WHODataProcessor()

        # 加载数据
        raw_data = processor.load_data()

        # 处理数据
        processed_data = processor.process_data()

        # 保存数据
        processor.save_processed_data()

        # 显示摘要统计
        summary = processor.get_summary_stats()

        print("\n" + "=" * 60)
        print("  Summary Statistics")
        print("=" * 60)
        print(f"Total deaths: {summary['total_deaths']:,.0f}")
        print(f"Male deaths: {summary['male_deaths']:,.0f}")
        print(f"Female deaths: {summary['female_deaths']:,.0f}")
        print(f"Gender ratio (M/F): {summary['gender_ratio']:.2f}")
        print(f"\nTop 5 Causes of Death:")
        for i, (cause, deaths) in enumerate(summary["top_causes"].head().items(), 1):
            print(f"  {i}. {cause}: {deaths:,.0f}")

        # 数据验证
        print("\n" + "=" * 60)
        print("  Data Validation")
        print("=" * 60)
        validation = processor.validate_data()
        print(f"Has null values: {validation['has_nulls']}")
        print(f"Has negative values: {validation['has_negatives']}")
        print(f"Gender inconsistencies: {len(validation['gender_consistency'])}")
        print(f"Incomplete age groups: {len(validation['age_groups_complete'])}")

        print("\n✅ Data processing completed successfully!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()

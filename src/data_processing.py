"""
数据处理模块
用于加载和清洗WHO全球死亡率数据
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")


class WHODataProcessor:
    """WHO死亡数据处理器"""

    def __init__(self, filepath="data/raw/ghe2021_deaths_global_new2.xlsx"):
        self.filepath = Path(filepath)
        self.raw_data = None
        self.processed_data = None
        self.metadata = {}

    def load_data(self, sheet_name="Global 2021"):
        """
        加载Excel数据

        Parameters:
        -----------
        sheet_name : str
            Excel工作表名称

        Returns:
        --------
        pd.DataFrame
            原始数据框
        """
        print(f"Loading data from {self.filepath}...")

        # 读取Excel文件
        self.raw_data = pd.read_excel(self.filepath, sheet_name=sheet_name, header=None)

        print(f"Loaded {self.raw_data.shape[0]} rows, {self.raw_data.shape[1]} columns")
        return self.raw_data

    def parse_structure(self):
        """
        解析数据结构，提取元数据和实际数据
        """
        if self.raw_data is None:
            raise ValueError("Please load data first using load_data()")

        # 找到数据开始的行（通常在第6-7行）
        data_start_row = None
        for idx, row in self.raw_data.iterrows():
            if pd.notna(row[0]) and "Population" in str(row[0]):
                data_start_row = idx
                break

        if data_start_row is None:
            data_start_row = 6  # 默认值

        # 提取元数据
        self.metadata["region"] = self.raw_data.iloc[1, 1]  # Global
        self.metadata["year"] = self.raw_data.iloc[2, 1]  # 2021

        print(f"Region: {self.metadata['region']}")
        print(f"Year: {self.metadata['year']}")
        print(f"Data starts at row: {data_start_row}")

        return data_start_row

    def process_data(self):
        """
        处理数据为标准格式
        """
        data_start = self.parse_structure()

        # 构建处理后的数据结构
        processed_rows = []

        # 年龄组列名
        age_columns = {
            "both_sexes": [
                "Total",
                "0-28 days",
                "1-59 months",
                "5-14",
                "15-29",
                "30-49",
                "50-59",
                "60-69",
                "70+",
            ],
            "male": [
                "Total",
                "0-28 days",
                "1-59 months",
                "5-14",
                "15-29",
                "30-49",
                "50-59",
                "60-69",
                "70+",
            ],
            "female": [
                "Total",
                "0-28 days",
                "1-59 months",
                "5-14",
                "15-29",
                "30-49",
                "50-59",
                "60-69",
                "70+",
            ],
        }

        # 遍历数据行
        current_row = data_start
        while current_row < len(self.raw_data):
            row = self.raw_data.iloc[current_row]

            # 检查是否是有效数据行
            if pd.isna(row[0]):
                current_row += 1
                continue

            cause_code = row[0]
            cause_name = row[1] if pd.notna(row[1]) else "Unknown"

            # 跳过非数据行
            if not isinstance(cause_code, (int, float, str)) or cause_name == "Unknown":
                current_row += 1
                continue

            # 提取各年龄组的数据
            try:
                # Both sexes数据
                both_total = row[4] if pd.notna(row[4]) else 0
                both_ages = [row[i] for i in range(6, 14)]

                # Male数据
                male_total = row[15] if pd.notna(row[15]) else 0
                male_ages = [row[i] for i in range(17, 25)]

                # Female数据
                female_total = row[26] if pd.notna(row[26]) else 0
                female_ages = [row[i] for i in range(28, 36)]

                # 为每个年龄组创建一行
                for i, age_group in enumerate(
                    age_columns["both_sexes"][1:]
                ):  # 跳过Total
                    processed_rows.append(
                        {
                            "cause_code": cause_code,
                            "cause_name": str(cause_name).strip(),
                            "age_group": age_group,
                            "both_sexes": (
                                float(both_ages[i]) if pd.notna(both_ages[i]) else 0
                            ),
                            "male": (
                                float(male_ages[i]) if pd.notna(male_ages[i]) else 0
                            ),
                            "female": (
                                float(female_ages[i]) if pd.notna(female_ages[i]) else 0
                            ),
                        }
                    )

            except Exception as e:
                print(f"Error processing row {current_row}: {e}")

            current_row += 1

        # 创建DataFrame
        self.processed_data = pd.DataFrame(processed_rows)

        # 数据清洗
        self.processed_data["both_sexes"] = pd.to_numeric(
            self.processed_data["both_sexes"], errors="coerce"
        ).fillna(0)
        self.processed_data["male"] = pd.to_numeric(
            self.processed_data["male"], errors="coerce"
        ).fillna(0)
        self.processed_data["female"] = pd.to_numeric(
            self.processed_data["female"], errors="coerce"
        ).fillna(0)

        # 添加计算字段
        self.processed_data["male_female_ratio"] = np.where(
            self.processed_data["female"] > 0,
            self.processed_data["male"] / self.processed_data["female"],
            np.nan,
        )

        print(f"\nProcessed data shape: {self.processed_data.shape}")
        print(f"Unique causes: {self.processed_data['cause_name'].nunique()}")
        print(f"Age groups: {self.processed_data['age_group'].unique()}")

        return self.processed_data

    def save_processed_data(self, output_path="data/processed/who_mortality_clean.csv"):
        """
        保存处理后的数据
        """
        if self.processed_data is None:
            raise ValueError("No processed data to save. Run process_data() first.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.processed_data.to_csv(output_path, index=False)
        print(f"Saved processed data to {output_path}")

    def get_summary_stats(self):
        """
        获取摘要统计
        """
        if self.processed_data is None:
            raise ValueError("No processed data. Run process_data() first.")

        summary = {
            "total_deaths": self.processed_data["both_sexes"].sum(),
            "male_deaths": self.processed_data["male"].sum(),
            "female_deaths": self.processed_data["female"].sum(),
            "top_causes": self.processed_data.groupby("cause_name")["both_sexes"]
            .sum()
            .nlargest(10),
            "age_distribution": self.processed_data.groupby("age_group")[
                "both_sexes"
            ].sum(),
        }

        return summary


# 主程序
if __name__ == "__main__":
    # 初始化处理器
    processor = WHODataProcessor()

    # 加载数据
    processor.load_data()

    # 处理数据
    df = processor.process_data()

    # 保存清洗后的数据
    processor.save_processed_data()

    # 显示摘要统计
    summary = processor.get_summary_stats()
    print("\n=== Summary Statistics ===")
    print(f"Total deaths: {summary['total_deaths']:,.0f}")
    print(f"Male deaths: {summary['male_deaths']:,.0f}")
    print(f"Female deaths: {summary['female_deaths']:,.0f}")
    print("\nTop 10 Causes of Death:")
    print(summary["top_causes"])

"""
示例：计算集中趋势指标（均值、中位数、众数）。

本例演示三种企鹅体重的集中趋势指标计算，以及分组统计。

运行方式：python3 chapters/week_02/examples/01_central_tendency.py
预期输出：
- 按物种分组的体重统计表（均值、中位数、计数）
- 整体均值、中位数、众数
"""
from __future__ import annotations

import seaborn as sns
import pandas as pd


def main() -> None:
    """主函数：计算并输出集中趋势指标"""
    penguins = sns.load_dataset("penguins")

    # 计算三种企鹅的平均体重和中位数体重
    print("按物种分组的体重统计：")
    stats_by_species = penguins.groupby("species")["body_mass_g"].agg(
        mean="mean",
        median="median",
        count="count"
    ).round(1)
    print(stats_by_species)
    print()

    # 计算整体的均值和中位数
    print("整体统计：")
    mean_mass = penguins["body_mass_g"].mean()
    median_mass = penguins["body_mass_g"].median()
    mode_species = penguins["species"].mode().tolist()

    print(f"均值：{mean_mass:.1f} g")
    print(f"中位数：{median_mass:.1f} g")
    print(f"众数（物种）：{mode_species}")

    # 额外输出：演示极端值对均值的影响
    print("\n极端值影响演示：")
    sample_data = pd.Series([3000, 3500, 4000, 4500, 100000])
    print(f"数据：{sample_data.tolist()}")
    print(f"均值：{sample_data.mean():.1f}（被极端值拉偏）")
    print(f"中位数：{sample_data.median():.1f}（稳健）")


if __name__ == "__main__":
    main()

"""
示例：计算离散程度指标（标准差、IQR、极差）。

本例演示如何使用 pandas 计算三种企鹅体重的离散程度指标，
包括标准差、分位数和四分位距（IQR）。

运行方式：python3 chapters/week_02/examples/02_dispersion_demo.py
预期输出：
- 按物种分组的离散程度统计表
- 整体 IQR 值
"""
from __future__ import annotations

import seaborn as sns
import pandas as pd


def compute_iqr(series: pd.Series) -> float:
    """计算四分位距（IQR）"""
    q25 = series.quantile(0.25)
    q75 = series.quantile(0.75)
    return q75 - q25


def main() -> None:
    """主函数：计算并输出离散程度指标"""
    penguins = sns.load_dataset("penguins")

    # 计算三种企鹅体重的标准差和分位数
    print("按物种分组的离散程度：")
    dispersion_by_species = penguins.groupby("species")["body_mass_g"].agg(
        std="std",
        min="min",
        q25=lambda x: x.quantile(0.25),
        median="median",
        q75=lambda x: x.quantile(0.75),
        max="max"
    ).round(1)
    print(dispersion_by_species)
    print()

    # 计算整体 IQR
    print("整体离散程度：")
    iqr = compute_iqr(penguins["body_mass_g"].dropna())
    mass_range = penguins["body_mass_g"].max() - penguins["body_mass_g"].min()
    std = penguins["body_mass_g"].std()

    print(f"标准差：{std:.1f} g")
    print(f"四分位距（IQR）：{iqr:.1f} g")
    print(f"极差：{mass_range:.1f} g")

    # 演示方差 vs 标准差（单位差异）
    variance = penguins["body_mass_g"].var()
    print(f"\n方差：{variance:.1f} g²（单位：平方克）")
    print(f"标准差：{std:.1f} g（单位：克，更易理解）")


if __name__ == "__main__":
    main()

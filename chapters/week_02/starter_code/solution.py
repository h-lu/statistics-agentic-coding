"""
Week 02 作业参考实现。

本文件提供作业的参考答案，学生在遇到困难时可以查看。
建议先自己尝试完成作业，实在想不出来再参考本文件。

作业要求概述：
1. 计算描述统计量（均值、中位数、标准差、IQR）
2. 绘制直方图和箱线图
3. 分析分布形状和异常值
4. 应用诚实可视化原则

注意：本实现只包含基础要求，不覆盖进阶/挑战部分。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def setup_output_dir() -> Path:
    """设置输出目录"""
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    return output_dir


# =============================================================================
# 核心函数（测试期望的函数）
# =============================================================================

def calculate_central_tendency(series: pd.Series) -> Dict[str, float]:
    """
    计算集中趋势指标

    参数：
        series: pandas Series，数值型数据

    返回：
        包含均值、中位数、众数的字典
    """
    return {
        "mean": series.mean(),
        "median": series.median(),
        # 众数可能返回多个值，取第一个
        "mode": series.mode().iloc[0] if len(series.mode()) > 0 else np.nan,
    }


def calculate_dispersion(series: pd.Series) -> Dict[str, float]:
    """
    计算离散程度指标

    参数：
        series: pandas Series，数值型数据

    返回：
        包含方差、标准差、IQR、极差的字典
    """
    q25 = series.quantile(0.25)
    q75 = series.quantile(0.75)
    return {
        "variance": series.var(),
        "std": series.std(),
        "iqr": q75 - q25,
        "range": series.max() - series.min(),
    }


def generate_descriptive_summary(df: pd.DataFrame) -> Dict[str, any]:
    """
    生成描述统计摘要

    参数：
        df: pandas DataFrame，包含要分析的数据

    返回：
        包含描述统计摘要的字典
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    summary = {}

    for col in numeric_cols:
        data = df[col].dropna()
        summary[col] = {
            **calculate_central_tendency(data),
            **calculate_dispersion(data),
            "count": len(data),
            "min": data.min(),
            "max": data.max(),
        }

    return summary


def exercise_1_central_tendency() -> None:
    """
    作业题 1：计算集中趋势指标

    要求：
    - 加载 penguins 数据集
    - 计算所有数值型列的均值和中位数
    - 按物种分组，计算每种企鹅的平均体重
    """
    penguins = sns.load_dataset("penguins")

    print("=== 集中趋势指标 ===")

    # 整体统计
    print("\n整体统计：")
    numeric_cols = penguins.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        mean_val = penguins[col].mean()
        median_val = penguins[col].median()
        print(f"{col}: 均值={mean_val:.2f}, 中位数={median_val:.2f}")

    # 按物种分组
    print("\n按物种分组的平均体重：")
    species_mean = penguins.groupby("species")["body_mass_g"].mean().round(1)
    print(species_mean)

    # 众数
    print("\n众数：")
    print(f"物种众数: {penguins['species'].mode().tolist()}")


def exercise_2_dispersion() -> None:
    """
    作业题 2：计算离散程度指标

    要求：
    - 计算体重的标准差和方差
    - 计算体重的 IQR（四分位距）
    - 按物种分组，比较哪种企鹅的体重波动最大
    """
    penguins = sns.load_dataset("penguins")

    print("\n=== 离散程度指标 ===")

    # 整体统计
    mass = penguins["body_mass_g"].dropna()
    std = mass.std()
    var = mass.var()
    q25 = mass.quantile(0.25)
    q75 = mass.quantile(0.75)
    iqr = q75 - q25

    print(f"\n整体体重统计：")
    print(f"标准差: {std:.2f} g")
    print(f"方差: {var:.2f} g²")
    print(f"IQR: {iqr:.2f} g")
    print(f"极差: {mass.max() - mass.min():.2f} g")

    # 按物种分组
    print("\n按物种分组的标准差：")
    species_std = penguins.groupby("species")["body_mass_g"].std().round(1)
    print(species_std)
    print(f"\n波动最大的是: {species_std.idxmax()}")


def exercise_3_histogram(output_dir: Path) -> None:
    """
    作业题 3：绘制直方图

    要求：
    - 绘制所有企鹅的体重分布直方图
    - 按物种分组，绘制三张子图，每张图一个物种的体重分布
    - 图表要清晰标注坐标轴和标题
    """
    penguins = sns.load_dataset("penguins")

    # 单图：整体分布
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 左图：整体分布
    axes[0].hist(penguins["body_mass_g"].dropna(), bins=20, edgecolor="black", alpha=0.7)
    axes[0].set_xlabel("Body Mass (g)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Overall Body Mass Distribution")

    # 右图：按物种分组
    for species in penguins["species"].unique():
        data = penguins[penguins["species"] == species]["body_mass_g"].dropna()
        axes[1].hist(data, bins=15, alpha=0.5, label=species, edgecolor="black")
    axes[1].set_xlabel("Body Mass (g)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Body Mass Distribution by Species")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_dir / "histogram_solution.png", dpi=100, facecolor="white")
    plt.close()
    print(f"\n直方图已保存到 {output_dir / 'histogram_solution.png'}")


def exercise_4_boxplot(output_dir: Path) -> None:
    """
    作业题 4：绘制箱线图并识别异常值

    要求：
    - 绘制按物种分组的体重箱线图
    - 识别并输出可能的异常值（使用 1.5×IQR 规则）
    - 分析哪些物种有异常值
    """
    penguins = sns.load_dataset("penguins")

    # 绘制箱线图
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=penguins, x="species", y="body_mass_g")
    plt.xlabel("Species")
    plt.ylabel("Body Mass (g)")
    plt.title("Body Mass Boxplot by Species")
    plt.savefig(output_dir / "boxplot_solution.png", dpi=100, facecolor="white")
    plt.close()
    print(f"\n箱线图已保存到 {output_dir / 'boxplot_solution.png'}")

    # 识别异常值
    print("\n异常值检测（1.5×IQR 规则）：")
    for species in penguins["species"].unique():
        data = penguins[penguins["species"] == species]["body_mass_g"].dropna()
        q25 = data.quantile(0.25)
        q75 = data.quantile(0.75)
        iqr = q75 - q25
        lower = q25 - 1.5 * iqr
        upper = q75 + 1.5 * iqr
        outliers = data[(data < lower) | (data > upper)]
        if len(outliers) > 0:
            print(f"{species}: 发现 {len(outliers)} 个异常值")
        else:
            print(f"{species}: 无异常值")


def exercise_5_honest_visualization(output_dir: Path) -> None:
    """
    作业题 5：诚实可视化对比

    要求：
    - 创建两张并排的柱状图，比较三种企鹅的平均体重
    - 左图截断 Y 轴（误导性），右图从 0 开始（诚实）
    - 在图上标注实际数值和样本量
    """
    penguins = sns.load_dataset("penguins")

    # 计算平均体重
    mean_mass = penguins.groupby("species")["body_mass_g"].mean().reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    colors = ["steelblue", "orange", "green"]

    # 左图：截断 Y 轴
    axes[0].bar(mean_mass["species"], mean_mass["body_mass_g"], color=colors)
    axes[0].set_ylim(3000, 5500)
    axes[0].set_ylabel("Body Mass (g)")
    axes[0].set_title("Misleading: Truncated Y-axis")

    # 右图：完整 Y 轴
    axes[1].bar(mean_mass["species"], mean_mass["body_mass_g"], color=colors)
    axes[1].set_ylim(0, 6000)
    axes[1].set_ylabel("Body Mass (g)")
    axes[1].set_title("Honest: Full Y-axis")

    # 标注数值
    for ax in axes:
        for i, row in mean_mass.iterrows():
            ax.text(i, row["body_mass_g"] + 100, f"{row['body_mass_g']:.0f}",
                    ha="center", va="bottom")

    # 标注样本量
    sample_sizes = penguins["species"].value_counts().to_dict()
    for ax in axes:
        ax.text(0.5, 0.02, f"n: Adelie={sample_sizes['Adelie']}, "
                           f"Chinstrap={sample_sizes['Chinstrap']}, "
                           f"Gentoo={sample_sizes['Gentoo']}",
                transform=ax.transAxes, ha="center", fontsize=9, style="italic")

    plt.tight_layout()
    plt.savefig(output_dir / "honest_viz_solution.png", dpi=100, facecolor="white")
    plt.close()
    print(f"\n诚实可视化对比图已保存到 {output_dir / 'honest_viz_solution.png'}")


def exercise_6_analysis_report() -> None:
    """
    作业题 6：分布分析报告

    要求：
    - 基于上述分析，写一份简短的分布分析报告
    - 报告应包含：典型值、波动、分布形状、异常值、诚实可视化说明
    """
    penguins = sns.load_dataset("penguins")

    print("\n=== 分布分析报告 ===")

    # 计算关键统计量
    adelie_stats = penguins[penguins["species"] == "Adelie"]["body_mass_g"].describe()
    gentoo_stats = penguins[penguins["species"] == "Gentoo"]["body_mass_g"].describe()

    report = """
    ## Palmer Penguins 体重分布分析

    ### 1. 集中趋势
    - Adelie 企鹅平均体重: {:.0f} g
    - Gentoo 企鹅平均体重: {:.0f} g
    - Gentoo 比 Adelie 重约 {:.0f} g（{:.0f}%）

    ### 2. 离散程度
    - 整体标准差: {:.1f} g
    - Adelie 标准差: {:.1f} g
    - Gentoo 标准差: {:.1f} g
    - Gentoo 的体重波动更大

    ### 3. 分布形状
    - 整体分布呈现轻微右偏（偏度: {:.2f}）
    - 按物种分组后，各组分布更接近正态

    ### 4. 异常值
    - 箱线图显示存在少量异常值
    - 这些异常值需要进一步核实（数据错误 vs 真实极端值）

    ### 5. 诚实可视化
    - 所有图表 Y 轴从 0 开始（除非说明理由）
    - 图上标注实际数值
    - 图例注明样本量
    """.format(
        adelie_stats["mean"],
        gentoo_stats["mean"],
        gentoo_stats["mean"] - adelie_stats["mean"],
        (gentoo_stats["mean"] - adelie_stats["mean"]) / adelie_stats["mean"] * 100,
        penguins["body_mass_g"].std(),
        adelie_stats["std"],
        gentoo_stats["std"],
        penguins["body_mass_g"].skew()
    )

    print(report)


def main() -> None:
    """运行所有作业题的参考解答"""
    print("="*60)
    print("Week 02 作业参考实现")
    print("="*60)

    output_dir = setup_output_dir()

    exercise_1_central_tendency()
    exercise_2_dispersion()
    exercise_3_histogram(output_dir)
    exercise_4_boxplot(output_dir)
    exercise_5_honest_visualization(output_dir)
    exercise_6_analysis_report()

    print("\n" + "="*60)
    print("所有作业题完成！")
    print(f"图表已保存到: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()

"""
示例：生成一页分布报告（包含箱线图）。

本例整合本周所学：集中趋势、离散程度、分布形状、箱线图、诚实可视化，
生成一份完整的"一页分布报告"。

运行方式：python3 chapters/week_02/examples/05_one_page_report.py
预期输出：
- output/one_page_report.png：四合一报告（摘要统计 + 直方图 + 箱线图 + 密度图）
- 控制台输出：完整的统计摘要
"""
from __future__ import annotations

from pathlib import Path

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def setup_output_dir() -> Path:
    """设置输出目录"""
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    return output_dir


def generate_summary_stats(df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    """生成描述统计表"""
    stats = df[numeric_cols].agg([
        ("count", "count"),
        ("mean", "mean"),
        ("median", "median"),
        ("std", "std"),
        ("min", "min"),
        ("Q25", lambda x: x.quantile(0.25)),
        ("Q75", lambda x: x.quantile(0.75)),
        ("max", "max")
    ]).round(1)
    return stats


def print_summary_stats(df: pd.DataFrame) -> None:
    """打印摘要统计"""
    print("="*60)
    print("一页分布报告：Palmer Penguins 数据集")
    print("="*60)

    # 整体统计
    print("\n【数据概览】")
    print(f"样本量：{len(df)} 只企鹅")
    print(f"物种：{', '.join(df['species'].unique().tolist())}")
    print(f"岛屿：{', '.join(df['island'].unique().tolist())}")

    # 按物种分组的体重统计
    print("\n【按物种分组的体重统计】")
    species_stats = df.groupby("species")["body_mass_g"].agg(
        n="count",
        mean="mean",
        median="median",
        std="std",
        min="min",
        max="max"
    ).round(1)
    print(species_stats)

    # 嘴峰长度统计
    print("\n【按物种分组的嘴峰长度统计】")
    bill_stats = df.groupby("species")["bill_length_mm"].agg(
        n="count",
        mean="mean",
        median="median",
        std="std"
    ).round(1)
    print(bill_stats)

    # 异常值检测
    print("\n【异常值检测（基于 1.5×IQR 规则）】")
    for species in df["species"].unique():
        data = df[df["species"] == species]["body_mass_g"].dropna()
        q25 = data.quantile(0.25)
        q75 = data.quantile(0.75)
        iqr = q75 - q25
        lower = q25 - 1.5 * iqr
        upper = q75 + 1.5 * iqr
        outliers = data[(data < lower) | (data > upper)]
        if len(outliers) > 0:
            print(f"{species}: 发现 {len(outliers)} 个异常值（{outliers.tolist()[:5]}...）")
        else:
            print(f"{species}: 无异常值")

    print("\n" + "="*60)


def plot_one_page_report(df: pd.DataFrame, output_dir: Path) -> None:
    """生成一页报告（四合一图）"""
    fig = plt.figure(figsize=(14, 10))

    # 1. 摘要统计表（左上）
    ax1 = plt.subplot(2, 2, 1)
    ax1.axis("tight")
    ax1.axis("off")

    # 按物种分组统计
    species_stats = df.groupby("species")["body_mass_g"].agg(
        n="count",
        mean="mean",
        median="median",
        std="std"
    ).round(1)

    table_data = []
    for species in species_stats.index:
        row = species_stats.loc[species]
        table_data.append([
            species,
            f"{int(row['n'])}",
            f"{row['mean']:.0f}",
            f"{row['median']:.0f}",
            f"{row['std']:.0f}"
        ])

    table = ax1.table(
        cellText=table_data,
        colLabels=["Species", "n", "Mean", "Median", "SD"],
        cellLoc="center",
        loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    ax1.set_title("Summary Statistics by Species", fontsize=12, fontweight="bold")

    # 2. 直方图（右上）
    ax2 = plt.subplot(2, 2, 2)
    species_colors = {"Adelie": "steelblue", "Chinstrap": "orange", "Gentoo": "green"}
    for species in df["species"].unique():
        data = df[df["species"] == species]["body_mass_g"].dropna()
        ax2.hist(data, bins=15, alpha=0.5, label=species, edgecolor="black",
                color=species_colors.get(species))
    ax2.set_xlabel("Body Mass (g)")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Distribution by Species", fontsize=12, fontweight="bold")
    ax2.legend()

    # 3. 箱线图（左下）
    ax3 = plt.subplot(2, 2, 3)
    sns.boxplot(data=df, x="species", y="body_mass_g", hue="species", ax=ax3,
               palette={"Adelie": "steelblue", "Chinstrap": "orange", "Gentoo": "green"}, legend=False)
    ax3.set_xlabel("Species")
    ax3.set_ylabel("Body Mass (g)")
    ax3.set_title("Boxplot: Detecting Outliers", fontsize=12, fontweight="bold")

    # 4. 密度图（右下）
    ax4 = plt.subplot(2, 2, 4)
    for species in df["species"].unique():
        data = df[df["species"] == species]["body_mass_g"].dropna()
        ax4.hist(data, bins=15, alpha=0.3, density=True, edgecolor="black",
                color=species_colors.get(species))
        # 叠加密度曲线
        from scipy import stats
        kde = stats.gaussian_kde(data)
        x_min, x_max = data.min() - 500, data.max() + 500
        x = [x_min + i * (x_max - x_min) / 200 for i in range(200)]
        ax4.plot(x, kde(x), linewidth=2, label=species, color=species_colors.get(species))
    ax4.set_xlabel("Body Mass (g)")
    ax4.set_ylabel("Density")
    ax4.set_title("Density Plot by Species", fontsize=12, fontweight="bold")
    ax4.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "one_page_report.png", dpi=100, facecolor="white")
    plt.close()
    print(f"\n一页报告图已保存到 {output_dir / 'one_page_report.png'}")


def plot_boxplot_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    """生成箱线图：单变量 vs 按物种分组"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 左图：单变量箱线图
    axes[0].boxplot(df["body_mass_g"].dropna(), vert=True)
    axes[0].set_ylabel("Body Mass (g)")
    axes[0].set_title("Overall Distribution")
    axes[0].set_xticks([])

    # 右图：按物种分组的箱线图
    species_colors = {"Adelie": "steelblue", "Chinstrap": "orange", "Gentoo": "green"}
    sns.boxplot(data=df, x="species", y="body_mass_g", hue="species", ax=axes[1],
               palette=species_colors, legend=False)
    axes[1].set_xlabel("Species")
    axes[1].set_ylabel("Body Mass (g)")
    axes[1].set_title("By Species (Gentoo is clearly heavier)")

    plt.tight_layout()
    plt.savefig(output_dir / "boxplot_comparison.png", dpi=100, facecolor="white")
    plt.close()
    print(f"箱线图对比已保存到 {output_dir / 'boxplot_comparison.png'}")


def main() -> None:
    """主函数：生成完整的一页分布报告"""
    penguins = sns.load_dataset("penguins")
    output_dir = setup_output_dir()

    # 打印摘要统计
    print_summary_stats(penguins)

    # 生成可视化
    plot_one_page_report(penguins, output_dir)
    plot_boxplot_comparison(penguins, output_dir)

    print("\n✓ 一页分布报告生成完成！")
    print("  包含：摘要统计 + 直方图 + 箱线图 + 密度图")


if __name__ == "__main__":
    main()

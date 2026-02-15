"""
示例：绘制分布图（直方图和密度图）。

本例演示如何用 matplotlib 和 seaborn 绘制体重分布的直方图和密度图，
揭示三种企鹅的分布差异。

运行方式：python3 chapters/week_02/examples/03_distribution_plots.py
预期输出：
- output/distribution_plots.png：双图（整体分布 + 按物种分组）
- output/density_plots.png：按物种分组的密度图
- output/distribution_with_stats.png：带统计量的分布图
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


def plot_histograms(penguins: pd.DataFrame, output_dir: Path) -> None:
    """绘制直方图：整体分布和按物种分组"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 左图：所有企鹅的体重分布
    axes[0].hist(penguins["body_mass_g"].dropna(), bins=20, edgecolor="black", alpha=0.7, color="steelblue")
    axes[0].set_xlabel("Body Mass (g)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Distribution of Penguin Body Mass")

    # 右图：按物种分组的体重分布
    species_colors = {"Adelie": "steelblue", "Chinstrap": "orange", "Gentoo": "green"}
    for species in penguins["species"].unique():
        data = penguins[penguins["species"] == species]["body_mass_g"].dropna()
        axes[1].hist(data, bins=15, alpha=0.5, label=species,
                    edgecolor="black", color=species_colors.get(species))

    axes[1].set_xlabel("Body Mass (g)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Body Mass by Species")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_dir / "distribution_plots.png", dpi=100, facecolor="white")
    plt.close()
    print(f"直方图已保存到 {output_dir / 'distribution_plots.png'}")


def plot_density(penguins: pd.DataFrame, output_dir: Path) -> None:
    """绘制密度图：按物种分组的平滑分布"""
    plt.figure(figsize=(8, 5))
    sns.kdeplot(data=penguins, x="body_mass_g", hue="species", fill=True, alpha=0.3)
    plt.xlabel("Body Mass (g)")
    plt.ylabel("Density")
    plt.title("Density Plot of Body Mass by Species")
    plt.savefig(output_dir / "density_plots.png", dpi=100, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"密度图已保存到 {output_dir / 'density_plots.png'}")


def plot_distribution_with_stats(penguins: pd.DataFrame, output_dir: Path) -> None:
    """绘制带统计量的分布图"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    species_list = ["Adelie", "Chinstrap", "Gentoo"]
    colors = ["steelblue", "orange", "green"]

    for idx, (species, color) in enumerate(zip(species_list, colors)):
        data = penguins[penguins["species"] == species]["body_mass_g"].dropna()

        # 直方图
        axes[idx].hist(data, bins=15, alpha=0.7, edgecolor="black", color=color)

        # 添加统计量
        mean_val = data.mean()
        median_val = data.median()
        std_val = data.std()

        axes[idx].axvline(mean_val, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_val:.0f}")
        axes[idx].axvline(median_val, color="blue", linestyle="-", linewidth=2, label=f"Median: {median_val:.0f}")

        axes[idx].set_xlabel("Body Mass (g)")
        axes[idx].set_ylabel("Frequency")
        axes[idx].set_title(f"{species}\n(n={len(data)}, SD={std_val:.0f})")
        axes[idx].legend()

    plt.tight_layout()
    plt.savefig(output_dir / "distribution_with_stats.png", dpi=100, facecolor="white")
    plt.close()
    print(f"带统计量的分布图已保存到 {output_dir / 'distribution_with_stats.png'}")


def print_skewness_kurtosis(penguins: pd.DataFrame) -> None:
    """打印偏度和峰度"""
    print("\n分布形状指标：")
    skewness = penguins["body_mass_g"].skew()
    kurtosis = penguins["body_mass_g"].kurtosis()
    print(f"偏度（Skewness）：{skewness:.2f}")
    print(f"  - 接近 0 表示分布对称")
    print(f"  - 正值表示右偏（右尾较长）")
    print(f"  - 负值表示左偏（左尾较长）")
    print(f"峰度（Kurtosis）：{kurtosis:.2f}")
    print(f"  - 正态分布的峰度约为 0（或 3，取决于定义）")
    print(f"  - 正值表示分布更尖峭")
    print(f"  - 负值表示分布更平坦")


def main() -> None:
    """主函数：生成所有分布可视化"""
    penguins = sns.load_dataset("penguins")
    output_dir = setup_output_dir()

    plot_histograms(penguins, output_dir)
    plot_density(penguins, output_dir)
    plot_distribution_with_stats(penguins, output_dir)
    print_skewness_kurtosis(penguins)


if __name__ == "__main__":
    main()

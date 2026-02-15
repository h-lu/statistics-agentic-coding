"""
示例：诚实可视化对比（Y 轴截断问题）。

本例演示 Y 轴截断如何误导读者，以及如何生成诚实的可视化。
对比同一数据的两种画法，展示视觉差异。

运行方式：python3 chapters/week_02/examples/04_honest_visualization.py
预期输出：
- output/honest_visualization.png：并排对比（误导 vs 诚实）
- output/area_trap_demo.png：面积陷阱演示
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


def plot_y_axis_comparison(penguins: pd.DataFrame, output_dir: Path) -> None:
    """绘制 Y 轴截断对比图"""
    # 计算各物种的平均体重
    mean_mass = penguins.groupby("species")["body_mass_g"].mean().reset_index()
    mean_mass = mean_mass.sort_values("body_mass_g")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    colors = {"Adelie": "steelblue", "Chinstrap": "orange", "Gentoo": "green"}
    bar_colors = [colors[species] for species in mean_mass["species"]]

    # 左图：截断 Y 轴（误导性）
    axes[0].bar(mean_mass["species"], mean_mass["body_mass_g"], color=bar_colors)
    axes[0].set_ylim(3000, 5500)  # 截断 Y 轴
    axes[0].set_ylabel("Body Mass (g)")
    axes[0].set_title("❌ Misleading: Truncated Y-axis")
    axes[0].set_xlabel("Species")

    # 右图：完整 Y 轴（诚实）
    axes[1].bar(mean_mass["species"], mean_mass["body_mass_g"], color=bar_colors)
    axes[1].set_ylim(0, 6000)  # 从 0 开始
    axes[1].set_ylabel("Body Mass (g)")
    axes[1].set_title("✅ Honest: Full Y-axis (from 0)")
    axes[1].set_xlabel("Species")

    # 标注实际数值
    for ax in axes:
        for i, row in mean_mass.iterrows():
            ax.text(i, row["body_mass_g"] + 100, f"{row['body_mass_g']:.0f}",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

    # 标注样本量
    sample_sizes = penguins["species"].value_counts().to_dict()
    for ax in axes:
        ax.text(0.5, 0.02, f"Sample sizes: Adelie={sample_sizes['Adelie']}, "
                           f"Chinstrap={sample_sizes['Chinstrap']}, Gentoo={sample_sizes['Gentoo']}",
                transform=ax.transAxes, ha="center", fontsize=9, style="italic")

    plt.tight_layout()
    plt.savefig(output_dir / "honest_visualization.png", dpi=100, facecolor="white")
    plt.close()
    print(f"Y 轴对比图已保存到 {output_dir / 'honest_visualization.png'}")

    # 打印实际差异
    adelie_mean = mean_mass[mean_mass["species"] == "Adelie"]["body_mass_g"].values[0]
    gentoo_mean = mean_mass[mean_mass["species"] == "Gentoo"]["body_mass_g"].values[0]
    diff = gentoo_mean - adelie_mean
    pct_diff = (diff / adelie_mean) * 100

    print(f"\n实际差异：")
    print(f"  Gentoo 比 Adelie 重 {diff:.0f} g（{pct_diff:.1f}%）")
    print(f"  但截断 Y 轴的图看起来像是'两倍'的差异！")


def plot_area_trap(output_dir: Path) -> None:
    """演示面积陷阱：饼图 vs 柱状图的感知差异"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 模拟数据：只有 10% 的差异
    categories = ["A", "B"]
    values = [100, 110]

    # 左图：饼图（面积感知）
    axes[0].pie(values, labels=categories, autopct="%1.0f%%", colors=["steelblue", "orange"])
    axes[0].set_title("❌ Pie Chart: Area perception (10% difference)")

    # 右图：柱状图（高度感知）
    axes[1].bar(categories, values, color=["steelblue", "orange"])
    axes[1].set_ylim(0, 120)
    axes[1].set_ylabel("Value")
    axes[1].set_title("✅ Bar Chart: Height perception (10% difference)")

    # 标注实际数值
    for i, v in enumerate(values):
        axes[1].text(i, v + 3, f"{v}", ha="center", va="bottom", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_dir / "area_trap_demo.png", dpi=100, facecolor="white")
    plt.close()
    print(f"面积陷阱演示图已保存到 {output_dir / 'area_trap_demo.png'}")


def print_honest_principles() -> None:
    """打印诚实可视化三原则"""
    print("\n" + "="*50)
    print("诚实可视化三原则：")
    print("="*50)
    print("1. Y 轴从 0 开始（除非有充分理由，并说明）")
    print("2. 标注实际数值（不要让读者'猜'差异）")
    print("3. 标注样本量（n=?）")
    print("\n其他常见陷阱：")
    print("  - 面积陷阱：用二维图形表示一维数据")
    print("  - 颜色陷阱：用颜色强度表示数值但没有图例")
    print("  - 时间陷阱：X 轴间隔不均匀但画成等距")
    print("="*50)


def main() -> None:
    """主函数：生成诚实可视化对比"""
    penguins = sns.load_dataset("penguins")
    output_dir = setup_output_dir()

    plot_y_axis_comparison(penguins, output_dir)
    plot_area_trap(output_dir)
    print_honest_principles()


if __name__ == "__main__":
    main()

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
import matplotlib.font_manager as fm
import pandas as pd


SPECIES_COLORS = {
    "Adelie": "#2563EB",      # blue
    "Chinstrap": "#F97316",   # orange
    "Gentoo": "#10B981",      # emerald
}


def setup_plot_style() -> None:
    """配置课件友好的可视化风格：中文字体、柔和配色、浅色网格。"""
    preferred_fonts = [
        "Noto Sans CJK SC",
        "Noto Sans SC",
        "Noto Sans CJK JP",  # many Linux distros expose the CJK TTC under the JP family name
        "Source Han Sans SC",
        "Microsoft YaHei",
        "PingFang SC",
        "WenQuanYi Micro Hei",
        "DejaVu Sans",
    ]
    available_fonts = {font.name for font in fm.fontManager.ttflist}
    for font in preferred_fonts:
        if font in available_fonts:
            plt.rcParams["font.sans-serif"] = [font, "DejaVu Sans"]
            break
    else:
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]

    plt.rcParams.update({
        "axes.unicode_minus": False,
        "figure.facecolor": "#F8FAFC",
        "axes.facecolor": "#FFFFFF",
        "axes.edgecolor": "#CBD5E1",
        "axes.labelcolor": "#334155",
        "axes.titlecolor": "#0F172A",
        "xtick.color": "#475569",
        "ytick.color": "#475569",
        "text.color": "#0F172A",
        "grid.color": "#E2E8F0",
        "grid.linewidth": 0.8,
        "legend.frameon": False,
        "savefig.facecolor": "#F8FAFC",
        "savefig.bbox": "tight",
    })
    sns.set_theme(style="whitegrid", context="notebook", rc={
        "font.sans-serif": plt.rcParams["font.sans-serif"],
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def style_axis(ax, title: str, subtitle: str | None = None) -> None:
    """统一单个子图的标题、网格和边框。"""
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold", pad=12)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9.5,
                color="#64748B", va="bottom", ha="left")
    ax.grid(True, axis="y", alpha=0.75)
    ax.grid(False, axis="x")
    for spine in ax.spines.values():
        spine.set_color("#E2E8F0")
        spine.set_linewidth(0.8)


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
    setup_plot_style()

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.5))
    fig.subplots_adjust(top=0.86, hspace=0.36, wspace=0.24)
    fig.suptitle("Palmer Penguins 体重分布一页报告", x=0.06, y=0.97,
                 ha="left", fontsize=24, fontweight="bold", color="#0F172A")
    fig.text(0.06, 0.925, "用一张图同时看样本规模、中心位置、离散程度和分布形状",
             ha="left", fontsize=12.5, color="#64748B")

    # 1. 摘要统计表（左上）
    ax1 = axes[0, 0]
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
        colLabels=["物种", "样本量", "均值", "中位数", "标准差"],
        cellLoc="center",
        loc="center",
        colColours=["#E0F2FE"] * 5,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.08, 2.28)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#CBD5E1")
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_text_props(weight="bold", color="#0F172A")
        elif col == 0:
            cell.set_text_props(weight="bold", color=SPECIES_COLORS.get(table_data[row - 1][0], "#0F172A"))
            cell.set_facecolor("#F8FAFC")
        else:
            cell.set_facecolor("#FFFFFF")
    style_axis(ax1, "01 摘要统计", "按物种汇总体重：先看 n、均值、中位数与波动")

    # 2. 直方图（右上）
    ax2 = axes[0, 1]
    for species in df["species"].unique():
        data = df[df["species"] == species]["body_mass_g"].dropna()
        ax2.hist(data, bins=14, alpha=0.48, label=species, edgecolor="white",
                 linewidth=1.1, color=SPECIES_COLORS.get(species))
    ax2.set_xlabel("体重 Body Mass (g)")
    ax2.set_ylabel("频数 Frequency")
    style_axis(ax2, "02 分布形状", "颜色区分物种，透明叠加便于比较重叠区域")
    ax2.legend(title="Species", loc="upper right")

    # 3. 箱线图（左下）
    ax3 = axes[1, 0]
    sns.boxplot(data=df, x="species", y="body_mass_g", hue="species", ax=ax3,
                palette=SPECIES_COLORS, width=0.55, linewidth=1.2, fliersize=4,
                legend=False)
    sns.stripplot(data=df, x="species", y="body_mass_g", ax=ax3,
                  color="#0F172A", alpha=0.18, size=2.6, jitter=0.18)
    ax3.set_xlabel("物种 Species")
    ax3.set_ylabel("体重 Body Mass (g)")
    style_axis(ax3, "03 离散程度与异常点", "箱体看 IQR，散点保留原始观测，不只看均值")

    # 4. 密度图（右下）
    ax4 = axes[1, 1]
    from scipy import stats
    for species in df["species"].unique():
        data = df[df["species"] == species]["body_mass_g"].dropna()
        ax4.hist(data, bins=14, alpha=0.14, density=True, edgecolor="white",
                 color=SPECIES_COLORS.get(species))
        kde = stats.gaussian_kde(data)
        x_min, x_max = data.min() - 500, data.max() + 500
        x = [x_min + i * (x_max - x_min) / 220 for i in range(220)]
        ax4.plot(x, kde(x), linewidth=2.8, label=species, color=SPECIES_COLORS.get(species))
        ax4.fill_between(x, kde(x), alpha=0.08, color=SPECIES_COLORS.get(species))
    ax4.set_xlabel("体重 Body Mass (g)")
    ax4.set_ylabel("密度 Density")
    style_axis(ax4, "04 平滑密度", "看整体趋势：Gentoo 明显更重，Adelie 与 Chinstrap 有重叠")
    ax4.legend(title="Species", loc="upper right")

    output_path = output_dir / "one_page_report.png"
    plt.savefig(output_path, dpi=160, facecolor="#F8FAFC")
    plt.close()
    print(f"\n一页报告图已保存到 {output_path}")


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

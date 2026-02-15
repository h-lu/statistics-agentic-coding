"""
示例：多变量可视化 - 散点图矩阵与热力图。

本例演示：
1. 使用 pairplot 创建散点图矩阵
2. 创建相关热力图
3. 在散点图矩阵中使用 hue 参数分组
4. 讨论热力图的局限性（非线性关系）

运行方式：python3 chapters/week_04/examples/03_multivariate_viz.py
预期输出：
- stdout 输出相关矩阵
- images/multivariate_pairplot.png：散点图矩阵
- images/multivariate_heatmap.png：相关热力图
- images/multivariate_pairplot_hue.png：带分组的散点图矩阵
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def setup_chinese_font() -> str:
    """配置中文字体，返回使用的字体名称"""
    chinese_fonts = ['SimHei', 'Noto Sans CJK SC', 'Arial Unicode MS',
                     'PingFang SC', 'Microsoft YaHei']
    available = [f.name for f in fm.fontManager.ttflist]
    for font in chinese_fonts:
        if font in available:
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False
            return font
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    return 'DejaVu Sans'


def create_multivariate_data(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """创建多变量数据"""
    rng = np.random.default_rng(seed)

    # 基础变量
    age = rng.integers(18, 70, n)
    income = rng.lognormal(10, 0.5, n)  # 对数正态分布（右偏）
    education_years = rng.integers(8, 20, n)

    # 创建有关系的变量
    # 购买金额与年龄、收入正相关
    purchase_amount = 50 + age * 2 + (np.log(income) - 8) * 30 + rng.normal(0, 20, n)

    # 停留时长（与购买金额弱相关）
    time_on_site = rng.exponential(180, n)

    # 添加一个类别变量
    customer_segment = rng.choice(["A", "B", "C"], n, p=[0.4, 0.35, 0.25])

    df = pd.DataFrame({
        "age": age,
        "income": income,
        "education_years": education_years,
        "purchase_amount": purchase_amount,
        "time_on_site": time_on_site,
        "customer_segment": customer_segment
    })
    return df


def print_correlation_matrices(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """对比三种相关系数矩阵"""
    numeric_cols = ["age", "income", "education_years", "purchase_amount", "time_on_site"]

    print("=" * 60)
    print("相关系数矩阵对比")
    print("=" * 60)

    pearson_corr = df[numeric_cols].corr(method="pearson")
    spearman_corr = df[numeric_cols].corr(method="spearman")
    kendall_corr = df[numeric_cols].corr(method="kendall")

    print("\n1. Pearson 相关系数")
    print(pearson_corr.round(2))

    print("\n2. Spearman 相关系数")
    print(spearman_corr.round(2))

    # 找出最强的相关
    corr_unstacked = pearson_corr.abs().unstack()
    corr_unstacked = corr_unstacked[corr_unstacked < 1]  # 排除自相关
    max_corr_pair = corr_unstacked.idxmax()
    max_corr_value = corr_unstacked.max()

    print(f"\n最强相关：{max_corr_pair[0]} 与 {max_corr_pair[1]} (Pearson r = {max_corr_value:.2f})")

    return pearson_corr, spearman_corr, kendall_corr


def create_pairplot(df: pd.DataFrame, output_dir: Path) -> None:
    """创建散点图矩阵"""
    font = setup_chinese_font()
    print(f"使用字体：{font}")

    numeric_cols = ["age", "income", "education_years", "purchase_amount", "time_on_site"]

    # 创建散点图矩阵
    sns.pairplot(df[numeric_cols], diag_kind="hist", plot_kws={"alpha": 0.6})

    plt.suptitle("Scatter Plot Matrix: All Variables", y=1.02, fontsize=14)
    plt.savefig(output_dir / "multivariate_pairplot.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n图表已保存到 {output_dir / 'multivariate_pairplot.png'}")


def create_pairplot_with_hue(df: pd.DataFrame, output_dir: Path) -> None:
    """创建带分组（hue）的散点图矩阵"""
    font = setup_chinese_font()

    numeric_cols = ["age", "income", "education_years", "purchase_amount", "time_on_site"]

    # 创建带分组的散点图矩阵
    sns.pairplot(df, vars=numeric_cols, hue="customer_segment",
                diag_kind="hist", plot_kws={"alpha": 0.7},
                palette="Set2")

    plt.suptitle("Scatter Plot Matrix: Grouped by Customer Segment", y=1.02, fontsize=14)
    plt.savefig(output_dir / "multivariate_pairplot_hue.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"图表已保存到 {output_dir / 'multivariate_pairplot_hue.png'}")


def create_heatmap(corr_matrix: pd.DataFrame, output_dir: Path) -> None:
    """创建相关热力图"""
    font = setup_chinese_font()

    plt.figure(figsize=(8, 6))

    # 使用 seaborn 绘制热力图
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8, "label": "Correlation"})

    plt.title("Correlation Heatmap\n(Pearson Correlation Coefficient)", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / "multivariate_heatmap.png", dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()

    print(f"图表已保存到 {output_dir / 'multivariate_heatmap.png'}")


def demonstrate_nonlinear_relationship(output_dir: Path) -> None:
    """演示非线性关系：U 型关系在热力图中的问题"""
    font = setup_chinese_font()

    # 创建 U 型关系数据
    np.random.seed(42)
    x = np.random.uniform(-5, 5, 150)
    # U 型关系：y = x^2 + noise
    y = x ** 2 + np.random.normal(0, 1, 150)

    df_u = pd.DataFrame({"x": x, "y": y})

    # 计算相关系数
    pearson_r = df_u["x"].corr(df_u["y"])
    spearman_rho = df_u["x"].corr(df_u["y"], method="spearman")

    print("\n" + "=" * 60)
    print("非线性关系示例：U 型关系")
    print("=" * 60)
    print(f"Pearson r = {pearson_r:.3f}（接近 0，但实际有强关系！）")
    print(f"Spearman \u03c1 = {spearman_rho:.3f}（也无法捕捉非单调关系）")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 左图：散点图
    axes[0].scatter(df_u["x"], df_u["y"], alpha=0.6, edgecolors='black', linewidth=0.5)
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")
    axes[0].set_title(f"U-Shaped Relationship (y = x\u00b2 + noise)\nScatter Plot Shows the Pattern")
    axes[0].grid(True, alpha=0.3)

    # 右图：相关热力图（误导）
    corr_matrix = df_u.corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap="coolwarm",
                center=0, square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8}, ax=axes[1])
    axes[1].set_title("Heatmap Shows 'No Correlation'\n(Misleading for nonlinear relationships)")

    plt.tight_layout()
    plt.savefig(output_dir / "multivariate_nonlinear_limitation.png", dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()

    print(f"图表已保存到 {output_dir / 'multivariate_nonlinear_limitation.png'}")


def main() -> None:
    """主函数"""
    # 设置输出路径
    output_dir = Path(__file__).parent.parent / "images"
    output_dir.mkdir(exist_ok=True)

    # 创建数据
    df = create_multivariate_data()

    # 1. 计算相关矩阵
    pearson_corr, spearman_corr, kendall_corr = print_correlation_matrices(df)

    # 2. 生成散点图矩阵
    create_pairplot(df, output_dir)

    # 3. 生成带分组的散点图矩阵
    create_pairplot_with_hue(df, output_dir)

    # 4. 生成热力图
    create_heatmap(pearson_corr, output_dir)

    # 5. 演示非线性关系的局限
    demonstrate_nonlinear_relationship(output_dir)

    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print("多变量可视化工具选择：")
    print("  1. 探索阶段：用散点图矩阵（能看见关系形状）")
    print("  2. 汇报阶段：用相关热力图（更简洁）")
    print("  3. 变量很多时：只用热力图（散点图矩阵会太密）")
    print("\n老潘的经验：")
    print("  '热力图是扫描仪，不是结论机'")
    print("  任何要在报告里展示的相关性，必须先看散点图")
    print("\n注意：")
    print("  - 热力图只能显示线性相关")
    print("  - U 型关系会被热力图标记为'无相关'")
    print("  - 散点图能发现非线性模式，热力图不能")


if __name__ == "__main__":
    main()

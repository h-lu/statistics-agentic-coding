"""
示例：相关分析入门 - 散点图、Pearson/Spearman/Kendall 相关系数。

本例演示：
1. 用散点图直观观察两个变量之间的关系
2. 计算 Pearson 相关系数（线性相关）
3. 计算 Spearman 相关系数（对极端值稳健）
4. 计算 Kendall 相关系数（小样本适用）
5. 展示异常值对相关系数的影响

运行方式：python3 chapters/week_04/examples/01_correlation_intro.py
预期输出：
- stdout 输出数据概览、相关系数矩阵、异常值影响对比
- images/correlation_scatter.png：散点图对比
- images/correlation_outlier_impact.png：异常值影响对比图
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.stats import pearsonr, spearmanr, kendalltau


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


def create_ecommerce_data(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """创建模拟电商用户行为数据"""
    rng = np.random.default_rng(seed)

    # 年龄与购买金额（正相关，但有噪声）
    age = rng.integers(18, 70, n)
    purchase_amount = 50 + age * 2 + rng.normal(0, 30, n)

    # 停留时长与购买金额（弱正相关）
    time_on_site = rng.exponential(180, n)  # 平均 180 秒
    purchase_amount_from_time = 20 + time_on_site * 0.5 + rng.normal(0, 40, n)

    # 创建一个非线性关系的例子（U型关系）
    satisfaction = rng.uniform(1, 10, n)
    # 满意度过低或过高时，复购意愿都高（中间低两头高）
    repurchase_intent = 2 * (satisfaction - 5) ** 2 / 25 + rng.normal(0.5, 0.2, n)
    repurchase_intent = np.clip(repurchase_intent, 0, 1)

    df = pd.DataFrame({
        "age": age,
        "time_on_site": time_on_site,
        "purchase_amount": purchase_amount,
        "satisfaction": satisfaction,
        "repurchase_intent": repurchase_intent
    })
    return df


def print_data_overview(df: pd.DataFrame) -> None:
    """打印数据概览"""
    print("=" * 60)
    print("数据概览")
    print("=" * 60)
    print(df.describe().round(1))
    print()


def calculate_correlation_matrix(df: pd.DataFrame,
                                 method: Literal["pearson", "spearman", "kendall"] = "pearson"
                                 ) -> pd.DataFrame:
    """计算相关系数矩阵"""
    return df.corr(method=method)


def print_correlation_comparison(df: pd.DataFrame) -> None:
    """对比三种相关系数"""
    numeric_cols = ["age", "time_on_site", "purchase_amount"]
    pearson_corr = df[numeric_cols].corr(method="pearson")
    spearman_corr = df[numeric_cols].corr(method="spearman")
    kendall_corr = df[numeric_cols].corr(method="kendall")

    print("=" * 60)
    print("三种相关系数对比")
    print("=" * 60)

    print("\n1. Pearson 相关系数（线性相关）")
    print(pearson_corr.round(2))
    print("适用：数据近似正态分布、关系线性、无极端值")

    print("\n2. Spearman 相关系数（排名相关）")
    print(spearman_corr.round(2))
    print("适用：有偏态、有极端值、非线性单调关系")

    print("\n3. Kendall 相关系数（序数相关）")
    print(kendall_corr.round(2))
    print("适用：小样本（< 20）、有并列排名")

    # scipy.stats 也可以计算
    print("\n" + "-" * 60)
    print("使用 scipy.stats 计算单个相关系数及其 p 值")
    print("-" * 60)
    x, y = df["age"].values, df["purchase_amount"].values

    r_pearson, p_pearson = pearsonr(x, y)
    rho_spearman, p_spearman = spearmanr(x, y)
    tau_kendall, p_kendall = kendalltau(x, y)

    print(f"\nage vs purchase_amount:")
    print(f"  Pearson:  r = {r_pearson:.3f}, p = {p_pearson:.2e}")
    print(f"  Spearman: \u03c1 = {rho_spearman:.3f}, p = {p_spearman:.2e}")
    print(f"  Kendall:  \u03c4 = {tau_kendall:.3f}, p = {p_kendall:.2e}")
    print()


def demonstrate_outlier_impact(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """演示异常值对相关系数的影响"""
    df_clean = df[["age", "purchase_amount"]].copy()
    df_with_outlier = df[["age", "purchase_amount"]].copy()

    # 添加一个极端值：年龄=20，购买金额=1000（异常高）
    df_with_outlier = pd.concat([
        df_with_outlier,
        pd.DataFrame({"age": [20], "purchase_amount": [1000]})
    ], ignore_index=True)

    # 计算相关系数
    corr_clean = df_clean["age"].corr(df_clean["purchase_amount"])
    corr_with_outlier = df_with_outlier["age"].corr(df_with_outlier["purchase_amount"])

    print("=" * 60)
    print("异常值对相关系数的影响")
    print("=" * 60)
    print(f"不含异常值：Pearson r = {corr_clean:.3f}")
    print(f"含异常值：  Pearson r = {corr_with_outlier:.3f}")
    print(f"\n一个极端值改变了相关系数 {abs(corr_clean - corr_with_outlier):.3f}")
    print("\n结论：异常值可以显著扭曲相关系数，因此必须先看散点图！")

    return df_clean, df_with_outlier


def create_scatter_plots(df: pd.DataFrame,
                        output_dir: Path) -> None:
    """创建散点图"""
    font = setup_chinese_font()
    print(f"使用字体：{font}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 左图：年龄 vs 购买金额（正相关）
    axes[0].scatter(df["age"], df["purchase_amount"], alpha=0.6, edgecolors='black', linewidth=0.5)
    axes[0].set_xlabel("Age (years)")
    axes[0].set_ylabel("Purchase Amount ($)")
    axes[0].set_title("Age vs Purchase Amount\n(Positive Correlation)")
    axes[0].grid(True, alpha=0.3)

    # 右图：停留时长 vs 购买金额（弱正相关）
    axes[1].scatter(df["time_on_site"], df["purchase_amount"], alpha=0.6, edgecolors='black', linewidth=0.5)
    axes[1].set_xlabel("Time on Site (seconds)")
    axes[1].set_ylabel("Purchase Amount ($)")
    axes[1].set_title("Time on Site vs Purchase Amount\n(Weak Positive Correlation)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "correlation_scatter.png", dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()

    print(f"\n图表已保存到 {output_dir / 'correlation_scatter.png'}")


def create_outlier_impact_plot(df_clean: pd.DataFrame,
                                df_with_outlier: pd.DataFrame,
                                output_dir: Path) -> None:
    """创建异常值影响对比图"""
    font = setup_chinese_font()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 左图：不含异常值
    axes[0].scatter(df_clean["age"], df_clean["purchase_amount"],
                    alpha=0.6, edgecolors='black', linewidth=0.5, color='steelblue')
    r_clean = df_clean["age"].corr(df_clean["purchase_amount"])
    axes[0].set_xlabel("Age (years)")
    axes[0].set_ylabel("Purchase Amount ($)")
    axes[0].set_title(f"Without Outlier\nr = {r_clean:.3f}")
    axes[0].grid(True, alpha=0.3)

    # 右图：含异常值
    axes[1].scatter(df_with_outlier["age"], df_with_outlier["purchase_amount"],
                    alpha=0.6, edgecolors='black', linewidth=0.5, color='steelblue')
    # 标出异常值
    outlier_mask = df_with_outlier["purchase_amount"] > 800
    axes[1].scatter(df_with_outlier.loc[outlier_mask, "age"],
                    df_with_outlier.loc[outlier_mask, "purchase_amount"],
                    color='red', s=100, edgecolors='darkred', linewidth=2, zorder=5, label='Outlier')
    r_with_outlier = df_with_outlier["age"].corr(df_with_outlier["purchase_amount"])
    axes[1].set_xlabel("Age (years)")
    axes[1].set_ylabel("Purchase Amount ($)")
    axes[1].set_title(f"With One Outlier\nr = {r_with_outlier:.3f}")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "correlation_outlier_impact.png", dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()

    print(f"图表已保存到 {output_dir / 'correlation_outlier_impact.png'}")


def main() -> None:
    """主函数"""
    # 设置输出路径
    output_dir = Path(__file__).parent.parent / "images"
    output_dir.mkdir(exist_ok=True)

    # 创建数据
    df = create_ecommerce_data()

    # 1. 数据概览
    print_data_overview(df)

    # 2. 计算并对比三种相关系数
    print_correlation_comparison(df)

    # 3. 演示异常值影响
    df_clean, df_with_outlier = demonstrate_outlier_impact(df)

    # 4. 生成图表
    create_scatter_plots(df, output_dir)
    create_outlier_impact_plot(df_clean, df_with_outlier, output_dir)

    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print("相关系数选择指南（老潘的经验法则）：")
    print("  1. 先用散点图看一眼：关系大致是线性的？用 Pearson")
    print("  2. 如果有明显极端值或偏态：用 Spearman")
    print("  3. 如果是小样本（< 20）：用 Kendall")
    print("\n记住：相关系数只是一个总结数字，散点图才是'讲故事的人'！")


if __name__ == "__main__":
    main()

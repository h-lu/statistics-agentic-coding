"""
示例：时间序列初步 - 趋势与季节性识别。

本例演示：
1. 创建模拟时间序列数据（趋势 + 季节性 + 噪声）
2. 可视化时间序列
3. 通过聚合识别趋势和季节性
4. 移动平均平滑

运行方式：python3 chapters/week_04/examples/04_time_series.py
预期输出：
- stdout 输出时间序列数据概览
- images/time_series_components.png：时间序列分解图
- images/time_series_aggregation.png：聚合对比图
- images/time_series_moving_avg.png：移动平均平滑图
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
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


def create_time_series_data(start_date: str = "2025-01-01",
                           end_date: str = "2025-12-31",
                           seed: int = 42) -> pd.DataFrame:
    """创建模拟时间序列数据"""
    rng = np.random.default_rng(seed)

    dates = pd.date_range(start_date, end_date, freq="D")
    n = len(dates)

    # 1. 趋势：线性上升
    trend = np.linspace(100, 200, n)

    # 2. 季节性：周末高、工作日低
    seasonality_weekly = np.where(dates.dayofweek >= 5, 30, 0)

    # 3. 年度季节性：夏季高、冬季低
    month_factor = np.sin(2 * np.pi * (dates.month - 1) / 12) * 20

    # 4. 随机噪声
    noise = rng.normal(0, 15, n)

    # 组合
    value = trend + seasonality_weekly + month_factor + noise

    df = pd.DataFrame({
        "date": dates,
        "sales": value
    })

    return df


def print_time_series_overview(df: pd.DataFrame) -> None:
    """打印时间序列数据概览"""
    print("=" * 60)
    print("时间序列数据概览")
    print("=" * 60)
    print(f"时间范围：{df['date'].min()} 到 {df['date'].max()}")
    print(f"数据点数：{len(df)}")
    print(f"频率：每日")
    print()
    print(df.head(10).to_string(index=False))
    print()
    print("描述统计：")
    print(df["sales"].describe().round(1))
    print()


def visualize_components(df: pd.DataFrame, output_dir: Path) -> None:
    """可视化时间序列的组成成分"""
    font = setup_chinese_font()

    fig, axes = plt.subplots(4, 1, figsize=(12, 10))

    # 原始数据
    axes[0].plot(df["date"], df["sales"], alpha=0.7, linewidth=0.8)
    axes[0].set_ylabel("Sales ($)")
    axes[0].set_title("Original Time Series (Trend + Seasonality + Noise)")
    axes[0].grid(True, alpha=0.3)

    # 趋势（通过移动平均估计）
    df["trend_est"] = df["sales"].rolling(window=30, center=True).mean()
    axes[1].plot(df["date"], df["trend_est"], color="steelblue", linewidth=2)
    axes[1].set_ylabel("Sales ($)")
    axes[1].set_title("Estimated Trend (30-day Moving Average)")
    axes[1].grid(True, alpha=0.3)

    # 季节性：周末 vs 工作日
    df["is_weekend"] = df["date"].dt.dayofweek >= 5
    weekend_means = df.groupby("is_weekend")["sales"].mean()
    axes[2].bar(["Weekday", "Weekend"], weekend_means.values,
                color=["steelblue", "orange"], edgecolor="black")
    axes[2].set_ylabel("Average Sales ($)")
    axes[2].set_title("Weekly Seasonality: Weekend vs Weekday")
    axes[2].grid(True, alpha=0.3, axis="y")

    # 噪声（残差）
    df["residual"] = df["sales"] - df["trend_est"]
    axes[3].plot(df["date"], df["residual"], alpha=0.5, linewidth=0.5)
    axes[3].axhline(0, color="red", linestyle="--", alpha=0.7)
    axes[3].set_xlabel("Date")
    axes[3].set_ylabel("Residual ($)")
    axes[3].set_title("Residual (Noise)")
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "time_series_components.png", dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()

    print(f"图表已保存到 {output_dir / 'time_series_components.png'}")


def demonstrate_aggregation(df: pd.DataFrame, output_dir: Path) -> None:
    """演示聚合如何让趋势更清晰"""
    font = setup_chinese_font()

    # 按周聚合
    df["week"] = df["date"].dt.isocalendar().week
    weekly_sales = df.groupby("week")["sales"].agg(["mean", "std"]).reset_index()

    # 按月聚合
    df["month"] = df["date"].dt.month
    monthly_sales = df.groupby("month")["sales"].agg(["mean", "std"]).reset_index()

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # 上图：每日 vs 周平均
    axes[0].plot(df["date"], df["sales"], alpha=0.4, linewidth=0.5,
                label="Daily", color="lightgray")
    axes[0].plot(df["date"], df["trend_est"], alpha=0.8, linewidth=1.5,
                label="7-day Moving Average", color="steelblue")
    axes[0].set_ylabel("Sales ($)")
    axes[0].set_title("Daily Data vs Moving Average")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 下图：周平均数据
    axes[1].plot(weekly_sales["week"], weekly_sales["mean"],
                marker="o", linewidth=2, color="steelblue")
    axes[1].fill_between(weekly_sales["week"],
                        weekly_sales["mean"] - weekly_sales["std"],
                        weekly_sales["mean"] + weekly_sales["std"],
                        alpha=0.3, label="+/- 1 SD")
    axes[1].set_xlabel("Week")
    axes[1].set_ylabel("Average Sales ($)")
    axes[1].set_title("Weekly Average Sales (Trend + Seasonality Clearer)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "time_series_aggregation.png", dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()

    print(f"图表已保存到 {output_dir / 'time_series_aggregation.png'}")


def demonstrate_moving_average(df: pd.DataFrame, output_dir: Path) -> None:
    """演示不同窗口的移动平均"""
    font = setup_chinese_font()

    # 计算不同窗口的移动平均
    windows = [7, 14, 30]
    for w in windows:
        df[f"ma_{w}"] = df["sales"].rolling(window=w, center=True).mean()

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(df["date"], df["sales"], alpha=0.3, linewidth=0.5,
            label="Daily", color="lightgray")

    colors = ["steelblue", "orange", "green"]
    for w, color in zip(windows, colors):
        ax.plot(df["date"], df[f"ma_{w}"], linewidth=1.5,
                label=f"{w}-day MA", color=color)

    ax.set_xlabel("Date")
    ax.set_ylabel("Sales ($)")
    ax.set_title("Moving Average Smoothing: Different Windows")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "time_series_moving_avg.png", dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()

    print(f"图表已保存到 {output_dir / 'time_series_moving_avg.png'}")


def analyze_seasonality_patterns(df: pd.DataFrame) -> None:
    """分析季节性模式"""
    print("=" * 60)
    print("季节性模式分析")
    print("=" * 60)

    # 1. 周末 vs 工作日
    df["is_weekend"] = df["date"].dt.dayofweek >= 5
    weekend_comparison = df.groupby("is_weekend")["sales"].agg(["mean", "std"])
    print("\n1. 周末 vs 工作日")
    print(weekend_comparison.round(1))

    # 2. 月份模式
    monthly_stats = df.groupby("month")["sales"].agg(["mean", "std", "count"])
    print("\n2. 按月统计")
    print(monthly_stats.round(1))

    # 3. 找出最高和最低的月份
    max_month = monthly_stats["mean"].idxmax()
    min_month = monthly_stats["mean"].idxmin()
    print(f"\n最高月份：{max_month}月（平均 ${monthly_stats.loc[max_month, 'mean']:.1f}）")
    print(f"最低月份：{min_month}月（平均 ${monthly_stats.loc[min_month, 'mean']:.1f}）")


def main() -> None:
    """主函数"""
    # 设置输出路径
    output_dir = Path(__file__).parent.parent / "images"
    output_dir.mkdir(exist_ok=True)

    # 创建数据
    df = create_time_series_data()

    # 1. 数据概览
    print_time_series_overview(df)

    # 2. 可视化组成成分
    visualize_components(df, output_dir)

    # 3. 演示聚合效果
    demonstrate_aggregation(df, output_dir)

    # 4. 演示移动平均
    demonstrate_moving_average(df, output_dir)

    # 5. 分析季节性模式
    analyze_seasonality_patterns(df)

    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print("时间序列分析要点：")
    print("  1. 趋势（Trend）：长期方向（上升/下降/持平）")
    print("  2. 季节性（Seasonality）：固定周期模式（周末/月份）")
    print("  3. 噪声（Noise）：随机波动")
    print("\n识别方法：")
    print("  - 聚合（按周/月）能减少噪声，让趋势更清晰")
    print("  - 移动平均能平滑短期波动")
    print("  - 按周期分组比较能发现季节性模式")
    print("\n注意：")
    print("  - 聚合会掩盖细节（如'周五晚上 vs 周六早上'）")
    print("  - 时间序列中的'相关'容易受混杂变量影响")
    print("  - 如'冰淇淋销量与溺水'都和温度相关，而非因果关系")


if __name__ == "__main__":
    main()

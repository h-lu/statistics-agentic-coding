"""
示例：分组比较 - groupby 与透视表。

本例演示：
1. 按 category 变量分组统计数值变量
2. 使用 pandas groupby 进行分组聚合
3. 使用 pivot_table 进行多维度交叉分析
4. 分组箱线图可视化

运行方式：python3 chapters/week_04/examples/02_groupby_analysis.py
预期输出：
- stdout 输出各渠道的统计描述、groupby 聚合结果、透视表
- images/groupby_boxplot.png：分组箱线图
- images/groupby_pivot_heatmap.png：透视表热力图
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


def create_grouped_data(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """创建带分组信息的电商数据"""
    rng = np.random.default_rng(seed)

    # 三个来源渠道：direct, search, social
    source = rng.choice(["direct", "search", "social"], n)

    # 不同渠道的购买金额分布不同
    # direct: 均值 100，标准差 20
    # search: 均值 120，标准差 25
    # social: 均值 80，标准差 15
    purchase_by_source = {
        "direct": rng.normal(100, 20, n),
        "search": rng.normal(120, 25, n),
        "social": rng.normal(80, 15, n)
    }

    # 构建完整数据
    data_list = []
    source_list = []
    for s in ["direct", "search", "social"]:
        data_list.extend(purchase_by_source[s])
        source_list.extend([s] * n)

    df = pd.DataFrame({
        "source": source_list[:len(data_list)],
        "purchase_amount": data_list
    })

    # 添加地区维度（用于透视表示例）
    regions = ["North", "South", "East", "West"]
    df["region"] = rng.choice(regions, len(df))

    # 添加一个复购标签
    df["is_returning"] = rng.choice([0, 1], len(df), p=[0.7, 0.3])

    return df


def print_grouped_descriptives(df: pd.DataFrame) -> None:
    """打印各分组的描述统计"""
    print("=" * 60)
    print("各渠道购买金额统计")
    print("=" * 60)
    print(df.groupby("source")["purchase_amount"].describe().round(1))
    print()


def demonstrate_groupby(df: pd.DataFrame) -> pd.DataFrame:
    """演示 groupby 的多种聚合操作"""
    print("=" * 60)
    print("groupby 聚合示例")
    print("=" * 60)

    # 1. 基础聚合
    print("\n1. 按渠道分组的均值")
    print(df.groupby("source")["purchase_amount"].mean().round(1))

    # 2. 多重聚合
    print("\n2. 按渠道分组的多种统计量")
    group_stats = df.groupby("source")["purchase_amount"].agg([
        ("count", "count"),
        ("mean", "mean"),
        ("median", "median"),
        ("std", "std"),
        ("cv", lambda x: x.std() / x.mean())  # 变异系数
    ]).round(1)
    print(group_stats)

    # 3. 按多个维度分组
    print("\n3. 按渠道和地区分组的均值")
    multi_group = df.groupby(["source", "region"])["purchase_amount"].mean().round(1)
    print(multi_group)
    print()

    return group_stats


def demonstrate_pivot_table(df: pd.DataFrame) -> pd.DataFrame:
    """演示透视表的创建和解读"""
    print("=" * 60)
    print("透视表（pivot_table）示例")
    print("=" * 60)

    # 1. 二维透视表：渠道 × 地区
    print("\n1. 按渠道和地区的交叉平均购买金额")
    pivot_mean = df.pivot_table(
        values="purchase_amount",
        index="source",
        columns="region",
        aggfunc="mean"
    ).round(1)
    print(pivot_mean)

    # 2. 二维透视表：渠道 × 是否复购
    print("\n2. 按渠道和复购状态的交叉统计")
    pivot_returning = df.pivot_table(
        values="purchase_amount",
        index="source",
        columns="is_returning",
        aggfunc="mean"
    ).round(1)
    pivot_returning.columns = ["New Customer", "Returning Customer"]
    print(pivot_returning)
    print("观察：复购客户的平均购买金额普遍高于新客户")

    # 3. 多种聚合函数
    print("\n3. 透视表使用多种聚合函数")
    pivot_multi = df.pivot_table(
        values="purchase_amount",
        index="source",
        columns="is_returning",
        aggfunc=["mean", "count", "std"]
    ).round(1)
    print(pivot_multi)
    print()

    return pivot_mean


def create_boxplot(df: pd.DataFrame, output_dir: Path) -> None:
    """创建分组箱线图"""
    font = setup_chinese_font()
    print(f"使用字体：{font}")

    plt.figure(figsize=(8, 5))

    # 使用 seaborn 绘制箱线图
    sns.boxplot(data=df, x="source", y="purchase_amount",
                hue="source", palette="Set2", legend=False)

    plt.xlabel("Source Channel", fontsize=11)
    plt.ylabel("Purchase Amount ($)", fontsize=11)
    plt.title("Purchase Amount by Source Channel\n(Grouped Boxplot)", fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / "groupby_boxplot.png", dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()

    print(f"\n图表已保存到 {output_dir / 'groupby_boxplot.png'}")


def create_pivot_heatmap(pivot_df: pd.DataFrame, output_dir: Path) -> None:
    """创建透视表热力图"""
    font = setup_chinese_font()

    plt.figure(figsize=(8, 6))

    # 使用 seaborn 绘制热力图
    sns.heatmap(pivot_df, annot=True, fmt=".1f", cmap="YlOrRd",
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})

    plt.xlabel("Region", fontsize=11)
    plt.ylabel("Source Channel", fontsize=11)
    plt.title("Average Purchase Amount: Channel x Region\n(Pivot Table as Heatmap)", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_dir / "groupby_pivot_heatmap.png", dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()

    print(f"图表已保存到 {output_dir / 'groupby_pivot_heatmap.png'}")


def main() -> None:
    """主函数"""
    # 设置输出路径
    output_dir = Path(__file__).parent.parent / "images"
    output_dir.mkdir(exist_ok=True)

    # 创建数据
    df = create_grouped_data()

    # 1. 描述统计
    print_grouped_descriptives(df)

    # 2. groupby 聚合
    group_stats = demonstrate_groupby(df)

    # 3. 透视表
    pivot_mean = demonstrate_pivot_table(df)

    # 4. 生成图表
    create_boxplot(df, output_dir)
    create_pivot_heatmap(pivot_mean, output_dir)

    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print("分组比较工具选择：")
    print("  - groupby：按一个或多个维度分组聚合")
    print("  - pivot_table：创建二维交叉表，适合展示和热力图")
    print("\n老潘的经验：")
    print("  '分组统计不只是算数字，而是问：这个差异是真的吗？'")
    print("  Week 06-08 我们会学习用统计检验来回答这个问题")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
示例：F 分布模拟与可视化

本例演示如何通过模拟理解 F 分布，帮助读者建立 ANOVA 的统计直觉。
内容：
1. 模拟 H0 为真时的 F 统计量分布
2. 对比理论 F 分布
3. 标注临界值与拒绝域

运行方式：python3 chapters/week_07/examples/01_f_distribution.py
预期输出：
  - F 临界值（控制台输出）
  - f_distribution_intuition.png（可视化图）

作者：Week 07 示例代码
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path


def simulate_f_statistic(
    k: int = 5,
    n_per_group: int = 50,
    n_sim: int = 10000,
    seed: int = 42
) -> np.ndarray:
    """
    模拟 F 统计量在 H0 为真时的分布。

    参数：
        k: 组数
        n_per_group: 每组样本量
        n_sim: 模拟次数
        seed: 随机种子

    返回：
        np.ndarray: F 统计量数组
    """
    np.random.seed(seed)
    f_stats = []

    for _ in range(n_sim):
        # 在 H0 为真时（所有组均值相等），生成数据
        groups = [np.random.normal(loc=100, scale=15, size=n_per_group) for _ in range(k)]

        # 合并数据
        all_data = np.concatenate(groups)
        grand_mean = np.mean(all_data)

        # 计算 SSB（组间平方和）
        group_means = [np.mean(g) for g in groups]
        ssb = sum(
            len(g) * (group_mean - grand_mean) ** 2
            for g, group_mean in zip(groups, group_means)
        )

        # 计算 SSW（组内平方和）
        ssw = sum(
            sum((x - group_mean) ** 2 for x in g)
            for g, group_mean in zip(groups, group_means)
        )

        # 计算均方
        df_between = k - 1
        df_within = k * n_per_group - k
        msb = ssb / df_between
        msw = ssw / df_within

        # F 统计量
        f = msb / msw
        f_stats.append(f)

    return np.array(f_stats)


def plot_f_distribution(f_stats: np.ndarray, df1: int, df2: int) -> Path:
    """
    绘制 F 分布对比图。

    参数：
        f_stats: 模拟的 F 统计量
        df1: 分子自由度
        df2: 分母自由度

    返回：
        Path: 保存的图片路径
    """
    # 理论 F 分布
    x = np.linspace(0, 6, 500)
    theoretical_pdf = stats.f.pdf(x, df1, df2)
    critical_value = stats.f.ppf(0.95, df1, df2)

    # 可视化
    plt.figure(figsize=(10, 6))

    # 模拟分布直方图
    plt.hist(
        f_stats,
        bins=50,
        density=True,
        alpha=0.7,
        color='skyblue',
        edgecolor='black',
        label='模拟 F 统计量'
    )

    # 理论分布曲线
    plt.plot(
        x,
        theoretical_pdf,
        'r-',
        linewidth=2,
        label=f'理论 F 分布 (df1={df1}, df2={df2})'
    )

    # 临界值线
    plt.axvline(
        critical_value,
        color='orange',
        linestyle='--',
        linewidth=2,
        label=f'临界值 F(0.95)={critical_value:.3f}'
    )

    # 标记拒绝域
    x_reject = np.linspace(critical_value, 6, 100)
    plt.fill_between(
        x_reject,
        stats.f.pdf(x_reject, df1, df2),
        alpha=0.3,
        color='red',
        label='拒绝域 (α=0.05)'
    )

    plt.xlabel('F 统计量', fontsize=12)
    plt.ylabel('密度', fontsize=12)
    plt.title('F 分布的直观理解：H0 为真时 F 的抽样分布', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # 保存图片
    output_path = Path('f_distribution_intuition.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path


def main() -> None:
    """主函数"""
    print("=" * 70)
    print("F 分布模拟")
    print("=" * 70)

    # 模拟参数
    k = 5  # 组数
    n_per_group = 50  # 每组样本量
    df1 = k - 1  # 组间自由度
    df2 = k * n_per_group - k  # 组内自由度

    print(f"\n模拟参数：")
    print(f"  组数：{k}")
    print(f"  每组样本量：{n_per_group}")
    print(f"  组间自由度 (df1)：{df1}")
    print(f"  组内自由度 (df2)：{df2}")

    # 模拟 F 分布
    print(f"\n模拟 F 统计量分布...")
    f_stats = simulate_f_statistic(k=k, n_per_group=n_per_group, n_sim=10000)

    # 计算临界值
    critical_value = stats.f.ppf(0.95, df1, df2)
    print(f"\nF 临界值 (α=0.05, df1={df1}, df2={df2})：{critical_value:.3f}")
    print(f"结论：如果 F > {critical_value:.3f}，拒绝 H0（各组均值不全相等）")

    # 可视化
    print(f"\n生成可视化...")
    output_path = plot_f_distribution(f_stats, df1, df2)
    print(f"图片已保存：{output_path}")

    # 验证：检查模拟分布与理论分布的一致性
    simulated_mean = np.mean(f_stats)
    theoretical_mean = df2 / (df2 - 2) if df2 > 2 else np.nan
    print(f"\n验证：")
    print(f"  模拟 F 均值：{simulated_mean:.3f}")
    print(f"  理论 F 均值：{theoretical_mean:.3f} (df2>2)")

    print("\n" + "=" * 70)
    print("模拟完成")
    print("=" * 70)


if __name__ == "__main__":
    main()

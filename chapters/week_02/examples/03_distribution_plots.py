#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分布可视化示例：直方图、密度图、箱线图

本示例展示如何用 seaborn 绘制不同类型的分布图，
以及每张图适合回答什么问题。

运行方式：python 03_distribution_plots.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def setup_style():
    """设置绘图风格"""
    sns.set_theme(style="whitegrid")
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False


def demonstrate_histogram():
    """演示直方图"""
    print("=" * 60)
    print("直方图 (Histogram)：看数据的'分布形状'")
    print("=" * 60)

    # 生成示例数据：正态分布 + 长尾
    np.random.seed(42)
    data_normal = np.random.normal(loc=100, scale=20, size=500)
    data_skewed = np.random.exponential(scale=50, size=500)

    print("\n说明：")
    print("• 直方图把数据切成'箱子'（bins），统计每个箱子里的数据点数")
    print("• 高度 = 频数，宽度 = 数值范围")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 左图：正态分布
    sns.histplot(data_normal, bins=30, kde=True, ax=axes[0], color='steelblue')
    axes[0].set_title('正态分布数据', fontsize=12)
    axes[0].set_xlabel('数值')
    axes[0].set_ylabel('频数')

    # 右图：偏态分布
    sns.histplot(data_skewed, bins=30, kde=True, ax=axes[1], color='coral')
    axes[1].set_title('右偏分布数据（长尾）', fontsize=12)
    axes[1].set_xlabel('数值')
    axes[1].set_ylabel('频数')

    plt.tight_layout()
    print("\n输出：distribution_comparison.png")
    plt.savefig('distribution_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("解读：")
    print("• 左图：对称的钟形，均值≈中位数")
    print("• 右图：长尾向右，均值会被拉高")


def demonstrate_boxplot():
    """演示箱线图"""
    print("\n" + "=" * 60)
    print("箱线图 (Boxplot)：一眼看'中心、波动、异常值'")
    print("=" * 60)

    # 生成三组对比数据
    np.random.seed(42)
    group_a = np.random.normal(50, 10, 100)
    group_b = np.random.normal(55, 15, 100)
    group_c = np.random.normal(52, 8, 100)

    # 给组C添加一些异常值
    group_c_with_outliers = np.append(group_c, [15, 95])

    df = pd.DataFrame({
        'value': list(group_a) + list(group_b) + list(group_c_with_outliers),
        'group': (['A'] * 100) + (['B'] * 100) + (['C'] * 102)
    })

    print("\n说明：")
    print("• 箱子：中间 50% 数据（Q1 到 Q3）")
    print("• 中线：中位数")
    print("• 须：延伸到 1.5×IQR 或最远点（取较小者）")
    print("• 圆点：须外的异常值")

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.boxplot(data=df, x='group', y='value', ax=ax, palette='Set2')

    ax.set_title('三组数据分布对比', fontsize=14)
    ax.set_xlabel('分组', fontsize=12)
    ax.set_ylabel('数值', fontsize=12)

    # 添加均值标记（对比中位数）
    means = df.groupby('group')['value'].mean()
    for i, (group, mean_val) in enumerate(means.items()):
        ax.plot(i, mean_val, marker='D', markersize=10, color='red', alpha=0.7)
        ax.text(i, mean_val + 3, f'μ={mean_val:.1f}', ha='center', fontsize=10, color='red')

    plt.tight_layout()
    print("\n输出：boxplot_comparison.png")
    plt.savefig('boxplot_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("解读：")
    print("• A 组最稳定（箱子窄）")
    print("• B 组波动最大（箱子高）")
    print("• C 组有两个异常值（圆点）")
    print("• 红色菱形 = 各组均值，对比中位数的位置")


def demonstrate_combined_plot():
    """演示组合图：直方图 + 箱线图"""
    print("\n" + "=" * 60)
    print("组合图：直方图 + 箱线图，同时看分布和异常值")
    print("=" * 60)

    # 生成双峰分布
    np.random.seed(42)
    data = np.concatenate([
        np.random.normal(30, 5, 200),
        np.random.normal(60, 8, 200)
    ])

    df = pd.DataFrame({'value': data})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # 左图：直方图 + KDE
    sns.histplot(data, bins=40, kde=True, ax=ax1, color='teal')
    ax1.set_title('分布形状（直方图 + 密度曲线）', fontsize=12)
    ax1.set_xlabel('数值')
    ax1.set_ylabel('频数')

    # 右图：箱线图
    sns.boxplot(data=df, x=['value'] * len(data), y='value', ax=ax2, color='lightblue')
    ax2.set_title('分布概览（箱线图）', fontsize=12)
    ax2.set_ylabel('数值')
    ax2.set_xlabel('')

    # 在箱线图上标注统计量
    median = df['value'].median()
    q1 = df['value'].quantile(0.25)
    q3 = df['value'].quantile(0.75)
    iqr = q3 - q1

    ax2.annotate(f'中位数={median:.1f}', xy=(0.5, median), xytext=(0.7, median),
                arrowprops=dict(arrowstyle='->', color='black'), fontsize=10)
    ax2.annotate(f'IQR={iqr:.1f}', xy=(0.5, q3), xytext=(0.7, q3),
                arrowprops=dict(arrowstyle='->', color='black'), fontsize=10)

    plt.tight_layout()
    print("\n输出：combined_view.png")
    plt.savefig('combined_view.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("组合解读：")
    print("• 直方图揭示双峰（两个峰值：约30和约60）")
    print("• 箱线图给出中位数和 IQR")
    print("• 两者结合，你对数据的理解更全面")


def demonstrate_rugplot():
    """演示 Rugplot（地毯图）"""
    print("\n" + "=" * 60)
    print("地毯图 (Rugplot)：每个数据点的'位置记号'")
    print("=" * 60)

    # 小数据集：每个点都很重要
    np.random.seed(42)
    data = np.random.normal(50, 15, 30)

    fig, ax = plt.subplots(figsize=(10, 4))

    # 地毯图 + KDE
    sns.kdeplot(data, fill=True, color='lightblue', alpha=0.5, ax=ax)
    sns.rugplot(data, color='darkblue', height=0.05, ax=ax)

    ax.set_title('小数据集的分布可视化', fontsize=14)
    ax.set_xlabel('数值', fontsize=12)
    ax.set_ylabel('密度', fontsize=12)
    ax.set_yticks([])  # 隐藏 Y 轴刻度

    plt.tight_layout()
    print("\n输出：rugplot_example.png")
    plt.savefig('rugplot_example.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("解读：")
    print("• 地毯上的每个小竖线 = 一个数据点")
    print("• 适合小数据集，能看到'实际点在哪里'")
    print("• 配合 KDE 曲线，能看到密度分布')


if __name__ == "__main__":
    setup_style()

    # 运行所有演示
    demonstrate_histogram()
    demonstrate_boxplot()
    demonstrate_combined_plot()
    demonstrate_rugplot()

    print("\n" + "=" * 60)
    print("核心要点总结")
    print("=" * 60)
    print("1. 直方图：看分布形状（对称/偏态/双峰）")
    print("2. 箱线图：一眼看中心、波动、异常值")
    print("3. 密度图 (KDE)：平滑的分布曲线")
    print("4. 地毯图：显示每个数据点的位置")
    print("5. 选择原则：根据问题选图，不是为了画而画")

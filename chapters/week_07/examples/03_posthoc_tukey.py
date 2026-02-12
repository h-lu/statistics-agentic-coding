#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
示例：Tukey HSD 事后检验

本例演示如何在 ANOVA 显著后进行事后检验，找出具体哪些组对存在差异。
内容：
1. Tukey HSD 检验执行
2. 提取显著结果
3. 可视化均值差异与置信区间

运行方式：python3 chapters/week_07/examples/03_posthoc_tukey.py
预期输出：
  - Tukey HSD 检验结果表
  - 显著城市对列表
  - tukey_hsd_results.png（可视化图）

作者：Week 07 示例代码
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
from pathlib import Path


def generate_city_data(n_per_city: int = 100, seed: int = 42) -> pd.DataFrame:
    """生成模拟城市消费数据（与 02_anova_example.py 一致）"""
    np.random.seed(seed)

    cities = ['北京', '上海', '广州', '深圳', '杭州']
    data = {'城市': [], '消费': []}

    means = [280, 310, 270, 320, 290]
    common_std = 50

    for city, mean in zip(cities, means):
        consumptions = np.random.normal(loc=mean, scale=common_std, size=n_per_city)
        data['城市'].extend([city] * n_per_city)
        data['消费'].extend(consumptions.tolist())

    return pd.DataFrame(data)


def tukey_hsd_test(
    data: pd.DataFrame,
    group_col: str,
    value_col: str,
    alpha: float = 0.05
) -> tuple:
    """
    执行 Tukey HSD 事后检验。

    参数：
        data: DataFrame
        group_col: 分组列名
        value_col: 数值列名
        alpha: 显著性水平

    返回：
        tuple: (tukey_results, tukey_df, significant_pairs)
    """
    groups = data[group_col].values
    values = data[value_col].values

    # 执行 Tukey HSD
    tukey = pairwise_tukeyhsd(endog=values, groups=groups, alpha=alpha)

    # 提取结果为 DataFrame
    tukey_df = pd.DataFrame(
        data=tukey._results_table.data[1:],
        columns=tukey._results_table.data[0]
    )

    # 提取显著结果
    significant_pairs = tukey_df[tukey_df['reject'] == True]

    return tukey, tukey_df, significant_pairs


def print_tukey_results(
    tukey_results,
    tukey_df: pd.DataFrame,
    significant_pairs: pd.DataFrame
) -> None:
    """打印 Tukey HSD 结果"""
    print("\n=== Tukey HSD 事后检验 ===")
    print(tukey_results)

    print(f"\n显著的城市对（α=0.05）：{len(significant_pairs)} 对")
    if len(significant_pairs) > 0:
        print(significant_pairs[['group1', 'group2', 'meandiff', 'p-adj', 'lower', 'upper']])
    else:
        print("  未发现显著差异的城市对")


def plot_tukey_results(tukey_df: pd.DataFrame) -> Path:
    """
    绘制 Tukey HSD 结果可视化。

    参数：
        tukey_df: Tukey HSD 结果 DataFrame

    返回：
        Path: 保存的图片路径
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # 绘制所有城市对的均值差异与 95% CI
    for idx, row in tukey_df.iterrows():
        group1, group2 = row['group1'], row['group2']
        meandiff = row['meandiff']
        lower, upper = row['lower'], row['upper']
        p_adj = row['p-adj']
        reject = row['reject']

        # Y 坐标
        y_pos = idx

        # 绘制误差条
        color = '#e74c3c' if reject else '#95a5a6'  # 红色显著，灰色不显著
        ax.errorbar(
            meandiff,
            y_pos,
            xerr=[[meandiff - lower], [upper - meandiff]],
            fmt='o',
            capsize=5,
            capthick=2,
            linewidth=2,
            color=color,
            markersize=8
        )

        # 标注 p 值
        ax.text(
            meandiff,
            y_pos + 0.4,
            f'p={p_adj:.3f}',
            ha='center',
            fontsize=9,
            color=color
        )

    # 零线
    ax.axvline(0, color='black', linestyle='--', alpha=0.5, linewidth=2)

    # Y 轴标签
    pair_labels = [
        f"{row['group1']} vs {row['group2']}"
        for _, row in tukey_df.iterrows()
    ]
    ax.set_yticks(range(len(tukey_df)))
    ax.set_yticklabels(pair_labels, fontsize=10)
    ax.set_xlabel('均值差异（元）', fontsize=12)
    ax.set_title(
        'Tukey HSD 事后检验结果\n(红色表示显著，灰色表示不显著)',
        fontsize=13
    )
    ax.grid(True, alpha=0.3, axis='x')

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', label='显著差异 (p<0.05)'),
        Patch(facecolor='#95a5a6', label='不显著 (p≥0.05)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.tight_layout()

    # 保存图片
    output_path = Path('tukey_hsd_results.png')
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path


def main() -> None:
    """主函数"""
    print("=" * 70)
    print("Tukey HSD 事后检验")
    print("=" * 70)

    # 1. 生成数据
    print("\n[1/3] 生成模拟数据...")
    df = generate_city_data()
    print(f"总样本量：{len(df)}")

    # 2. 执行 Tukey HSD
    print("\n[2/3] 执行 Tukey HSD 检验...")
    tukey_results, tukey_df, significant_pairs = tukey_hsd_test(
        df, '城市', '消费', alpha=0.05
    )

    # 3. 打印结果
    print("\n[3/3] 输出结果...")
    print_tukey_results(tukey_results, tukey_df, significant_pairs)

    # 4. 可视化
    print("\n生成可视化...")
    output_path = plot_tukey_results(tukey_df)
    print(f"图片已保存：{output_path}")

    # 5. 解读
    print("\n" + "=" * 70)
    print("结果解读")
    print("=" * 70)

    if len(significant_pairs) > 0:
        print(f"\n发现 {len(significant_pairs)} 对显著差异：")
        for _, row in significant_pairs.iterrows():
            direction = "高于" if row['meandiff'] > 0 else "低于"
            print(
                f"  - {row['group1']} {direction} {row['group2']} "
                f"（差={abs(row['meandiff']):.1f}元, p={row['p-adj']:.4f}）"
            )
    else:
        print("\n未发现显著差异的城市对。")

    print("\n" + "=" * 70)
    print("分析完成")
    print("=" * 70)


if __name__ == "__main__":
    main()

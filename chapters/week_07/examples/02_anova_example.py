#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
示例：单因素 ANOVA 完整实战

本例演示完整的 ANOVA 分析流程，包括：
1. 前提假设检查（正态性、方差齐性、独立性）
2. ANOVA 检验（scipy 和 statsmodels 两种方法）
3. 效应量计算（η²）
4. 结果可视化（箱线图 + 均值误差条）

运行方式：python3 chapters/week_07/examples/02_anova_example.py
预期输出：
  - 前提假设检查结果
  - ANOVA 表
  - η² 效应量
  - anova_results.png（可视化图）

作者：Week 07 示例代码
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def generate_city_data(
    n_per_city: int = 100,
    seed: int = 42
) -> pd.DataFrame:
    """
    生成模拟城市消费数据。

    参数：
        n_per_city: 每个城市样本量
        seed: 随机种子

    返回：
        pd.DataFrame: 包含城市和消费的数据
    """
    np.random.seed(seed)

    cities = ['北京', '上海', '广州', '深圳', '杭州']
    data = {'城市': [], '消费': []}

    # 各组均值略有不同，方差相等
    means = [280, 310, 270, 320, 290]
    common_std = 50

    for city, mean in zip(cities, means):
        consumptions = np.random.normal(loc=mean, scale=common_std, size=n_per_city)
        data['城市'].extend([city] * n_per_city)
        data['消费'].extend(consumptions.tolist())

    return pd.DataFrame(data)


def check_assumptions(df: pd.DataFrame) -> dict:
    """
    检查 ANOVA 前提假设。

    参数：
        df: 包含'城市'和'消费'列的数据框

    返回：
        dict: 假设检查结果
    """
    print("\n=== 前提假设检查 ===")

    cities = df['城市'].unique()
    results = {}

    # 1. 正态性检验（Shapiro-Wilk 检验）
    print("\n1. 正态性检验（Shapiro-Wilk）：")
    normality_results = {}
    for city in cities:
        city_data = df[df['城市'] == city]['消费']
        _, p_value = stats.shapiro(city_data)
        normality_results[city] = p_value
        status = '✓ 正态性假设满足' if p_value > 0.05 else '✗ 偏离正态'
        print(f"  {city}：p = {p_value:.4f} {status}")
    results['normality'] = normality_results

    # 2. 方差齐性检验（Levene 检验）
    city_groups = [df[df['城市'] == city]['消费'].values for city in cities]
    _, p_levene = stats.levene(*city_groups)
    print(f"\n2. 方差齐性检验（Levene）：")
    print(f"  p = {p_levene:.4f}")
    status = '✓ 方差齐性假设满足' if p_levene > 0.05 else '✗ 方差不齐（需使用 Welch ANOVA）'
    print(f"  结论：{status}")
    results['levene_p'] = p_levene

    # 3. 独立性检验（设计检查）
    print(f"\n3. 独立性：")
    print(f"  ✓ 用户随机抽样，各城市互不干扰")
    results['independence'] = True

    return results


def perform_anova(df: pd.DataFrame) -> tuple:
    """
    执行 ANOVA 检验。

    参数：
        df: 包含'城市'和'消费'列的数据框

    返回：
        tuple: (f_stat, p_value, anova_table)
    """
    print("\n=== 单因素 ANOVA ===")

    cities = df['城市'].unique()
    city_groups = [df[df['城市'] == city]['消费'].values for city in cities]

    # 方法 1：使用 scipy.stats.f_oneway
    f_stat, p_value = stats.f_oneway(*city_groups)
    print(f"\n方法 1：scipy.stats.f_oneway")
    print(f"  F 统计量：{f_stat:.4f}")
    print(f"  p 值：{p_value:.6f}")
    decision = '拒绝 H0（各组均值不全相等）' if p_value < 0.05 else '无法拒绝 H0（各组均值可能相等）'
    print(f"  结论：{decision}")

    # 方法 2：使用 statsmodels（生成完整 ANOVA 表）
    print(f"\n方法 2：statsmodels OLS + ANOVA 表")
    model = ols('消费 ~ C(城市)', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)

    return f_stat, p_value, anova_table


def calculate_eta_squared(anova_table: pd.DataFrame) -> float:
    """
    计算 η²（eta-squared）效应量。

    参数：
        anova_table: statsmodels 生成的 ANOVA 表

    返回：
        float: η² 值
    """
    ssb = anova_table.loc['C(城市)', 'sum_sq']
    ssw = anova_table.loc['Residual', 'sum_sq']
    sst = ssb + ssw
    eta2 = ssb / sst
    return eta2


def interpret_eta_squared(eta2: float) -> str:
    """
    解释 η² 效应量。

    参数：
        eta2: η² 值

    返回：
        str: 解释文字
    """
    if eta2 < 0.01:
        return "效应量极小（< 1% 的变异由组间差异解释）"
    elif eta2 < 0.06:
        return "效应量小（1%-6% 的变异由组间差异解释）"
    elif eta2 < 0.14:
        return "效应量中等（6%-14% 的变异由组间差异解释）"
    else:
        return "效应量大（≥ 14% 的变异由组间差异解释）"


def plot_anova_results(
    df: pd.DataFrame,
    f_stat: float,
    p_value: float,
    eta2: float
) -> Path:
    """
    绘制 ANOVA 结果可视化。

    参数：
        df: 数据框
        f_stat: F 统计量
        p_value: p 值
        eta2: η² 效应量

    返回：
        Path: 保存的图片路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：箱线图比较
    sns.boxplot(data=df, x='城市', y='消费', ax=axes[0], palette='Set2')
    axes[0].set_xlabel('城市', fontsize=12)
    axes[0].set_ylabel('消费（元）', fontsize=12)
    axes[0].set_title(
        f'各城市用户消费分布\nF={f_stat:.2f}, p={p_value:.6f}, η²={eta2:.3f}',
        fontsize=12
    )
    axes[0].grid(True, alpha=0.3, axis='y')

    # 右图：均值与 95% CI
    city_stats = df.groupby('城市')['消费'].agg(['mean', 'std', 'count'])
    city_stats['sem'] = city_stats['std'] / np.sqrt(city_stats['count'])
    city_stats['ci_low'] = city_stats['mean'] - 1.96 * city_stats['sem']
    city_stats['ci_high'] = city_stats['mean'] + 1.96 * city_stats['sem']

    axes[1].errorbar(
        city_stats.index,
        city_stats['mean'],
        yerr=[
            city_stats['mean'] - city_stats['ci_low'],
            city_stats['ci_high'] - city_stats['mean']
        ],
        fmt='o',
        capsize=10,
        capthick=2,
        linewidth=2,
        markersize=8,
        color='steelblue'
    )
    axes[1].axhline(
        df['消费'].mean(),
        color='red',
        linestyle='--',
        alpha=0.5,
        label='总均值'
    )
    axes[1].set_xlabel('城市', fontsize=12)
    axes[1].set_ylabel('消费（元）', fontsize=12)
    axes[1].set_title('各城市均值与 95% 置信区间', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # 保存图片
    output_path = Path('anova_results.png')
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path


def main() -> None:
    """主函数"""
    print("=" * 70)
    print("单因素 ANOVA 实战")
    print("=" * 70)

    # 1. 生成数据
    print("\n[1/5] 生成模拟数据...")
    df = generate_city_data()
    print(f"总样本量：{len(df)}")
    print(f"各组均值：\n{df.groupby('城市')['消费'].mean()}")

    # 2. 前提假设检查
    print("\n[2/5] 检查前提假设...")
    assumptions = check_assumptions(df)

    # 3. 执行 ANOVA
    print("\n[3/5] 执行 ANOVA...")
    f_stat, p_value, anova_table = perform_anova(df)

    # 4. 计算效应量
    print("\n[4/5] 计算效应量...")
    eta2 = calculate_eta_squared(anova_table)
    print(f"\nη² 效应量：{eta2:.3f}")
    interpretation = interpret_eta_squared(eta2)
    print(f"解释：{interpretation}")

    # 5. 可视化
    print("\n[5/5] 生成可视化...")
    output_path = plot_anova_results(df, f_stat, p_value, eta2)
    print(f"图片已保存：{output_path}")

    print("\n" + "=" * 70)
    print("分析完成")
    print("=" * 70)


if __name__ == "__main__":
    main()

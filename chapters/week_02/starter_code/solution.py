#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Week 02 起始代码：一页分布报告

本文件是学生练习的起始模板。
你可以基于此代码完成本周作业。

运行方式：python solution.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def setup_style():
    """设置绘图风格"""
    sns.set_theme(style="whitegrid")
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False


# ============ 第 1 节：集中趋势 ============

def calculate_central_tendency(series):
    """
    计算集中趋势指标

    参数：
        series: pd.Series - 数值型数据

    返回：
        dict: 包含均值、中位数、众数
    """
    series = pd.Series(series).dropna()

    result = {
        'mean': series.mean(),
        'median': series.median(),
        'mode': series.mode()[0] if len(series.mode()) > 0 else None,
        'count': len(series)
    }

    return result


def recommend_central_tendency(series):
    """
    根据数据特征推荐合适的集中趋势指标

    返回：
        str: 推荐的指标和理由
    """
    series = pd.Series(series).dropna()
    mean_val = series.mean()
    median_val = series.median()
    skewness = series.skew()

    # 判断偏态
    if abs(skewness) < 0.3:
        return f"建议使用均值（{mean_val:.2f}）：分布较为对称"
    elif skewness > 0.3:
        return f"建议使用中位数（{median_val:.2f}）：分布右偏，均值被拉高"
    else:  # skewness < -0.3
        return f"建议使用中位数（{median_val:.2f}）：分布左偏，均值被拉低"


# ============ 第 2 节：离散程度 ============

def calculate_dispersion(series):
    """
    计算离散程度指标

    参数：
        series: pd.Series - 数值型数据

    返回：
        dict: 包含标准差、方差、IQR
    """
    series = pd.Series(series).dropna()

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    result = {
        'std': series.std(),
        'variance': series.var(),
        'q1': q1,
        'q3': q3,
        'iqr': q3 - q1,
        'range': series.max() - series.min()
    }

    return result


def detect_outliers_iqr(series, multiplier=1.5):
    """
    使用 IQR 规则检测异常值

    参数：
        series: pd.Series - 数值型数据
        multiplier: float - IQR 倍数（默认 1.5）

    返回：
        dict: 包含边界和异常值列表
    """
    series = pd.Series(series).dropna()

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr

    outliers = series[(series < lower_bound) | (series > upper_bound)]

    return {
        'q1': q1,
        'q3': q3,
        'iqr': iqr,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'outliers': outliers.tolist(),
        'outlier_count': len(outliers)
    }


# ============ 第 3 节：分布可视化 ============

def create_histogram(data, column, bins=30, kde=True):
    """
    创建直方图

    参数：
        data: pd.DataFrame - 数据集
        column: str - 要绘制的列名
        bins: int - 箱子数量
        kde: bool - 是否添加 KDE 曲线

    返回：
        matplotlib.figure.Figure: 图表对象
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    sns.histplot(data[column].dropna(), bins=bins, kde=kde, ax=ax, color='steelblue')

    mean_val = data[column].mean()
    median_val = data[column].median()
    ax.axvline(mean_val, color='red', linestyle='--', alpha=0.7, label=f'均值={mean_val:.1f}')
    ax.axvline(median_val, color='green', linestyle='--', alpha=0.7, label=f'中位数={median_val:.1f}')

    ax.set_title(f'{column} 分布', fontsize=12)
    ax.set_xlabel(column)
    ax.set_ylabel('频数')
    ax.legend(fontsize=9)

    return fig


def create_boxplot(data, x_column, y_column):
    """
    创建箱线图

    参数：
        data: pd.DataFrame - 数据集
        x_column: str - 分组列名
        y_column: str - 数值列名

    返回：
        matplotlib.figure.Figure: 图表对象
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    sns.boxplot(data=data, x=x_column, y=y_column, ax=ax, palette='Set2')

    ax.set_title(f'{y_column} 按 {x_column} 分组', fontsize=12)
    ax.set_ylabel(y_column)

    return fig


# ============ 第 4 节：可视化诚实性 ============

def check_y_axis_baseline(y_limits, data_range, tolerance=0.1):
    """
    检查 Y 轴基线是否诚实

    参数：
        y_limits: tuple - (ymin, ymax)
        data_range: float - 数据的实际范围
        tolerance: float - 容忍比例（默认 10%）

    返回：
        dict: 包含 is_honest 和说明
    """
    ymin, ymax = y_limits
    data_min, data_max = 0, data_range

    if ymin > tolerance * data_range:
        return {
            'is_honest': False,
            'issue': 'Y 轴截断：不从 0 开始',
            'severity': 'high' if ymin > 0.5 * data_range else 'medium'
        }
    else:
        return {
            'is_honest': True,
            'issue': None
        }


def create_honest_bar_chart(categories, values, ylabel='数值'):
    """
    创建诚实的柱状图（Y 轴从 0 开始）

    参数：
        categories: list - 类别标签
        values: list - 数值列表
        ylabel: str - Y 轴标签

    返回：
        matplotlib.figure.Figure: 图表对象
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(categories, values, color='steelblue', alpha=0.7)

    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(values) * 1.1)

    # 添加数值标签
    for i, (cat, val) in enumerate(zip(categories, values)):
        ax.text(i, val + max(values) * 0.02, f'{val}', ha='center', fontsize=10)

    return fig


# ============ 第 5 节：一页报告 ============

def generate_descriptive_summary(data, numeric_columns=None):
    """
    生成描述统计摘要

    参数：
        data: pd.DataFrame - 数据集
        numeric_columns: list - 要统计的数值列列表

    返回：
        dict: 以列名为键，统计指标为值的字典
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()

    summary = {}

    for col in numeric_columns[:6]:  # 最多 6 列
        col_data = data[col].dropna()

        if len(col_data) == 0:
            continue

        summary[col] = {
            'mean': col_data.mean(),
            'median': col_data.median(),
            'std': col_data.std(),
            'q1': col_data.quantile(0.25),
            'q3': col_data.quantile(0.75),
            'iqr': col_data.quantile(0.75) - col_data.quantile(0.25),
        }

    return summary


def append_to_report(report_path='report.md', content=''):
    """
    追加内容到报告

    参数：
        report_path: str - 报告文件路径
        content: str - 要追加的内容

    返回：
        None
    """
    with open(report_path, 'a', encoding='utf-8') as f:
        f.write(content)

    print(f"已追加内容到 {report_path}")


# ============ 主函数演示 ============

def main():
    """主函数：演示所有功能"""
    print("=" * 60)
    print("Week 02 起始代码演示")
    print("=" * 60)

    # 创建示例数据
    np.random.seed(42)
    n = 200

    df = pd.DataFrame({
        'age': np.random.normal(35, 10, n),
        'income': np.random.lognormal(8, 1, n),
        'score': np.random.uniform(60, 100, n),
        'category': np.random.choice(['A', 'B', 'C'], n)
    })

    # 确保数据为正
    df['age'] = df['age'].clip(lower=18).round(0).astype(int)
    df['income'] = df['income'].clip(lower=0).round(2)

    print(f"\n示例数据（前 5 行）：")
    print(df.head())

    # 演示第 1 节：集中趋势
    print("\n--- 第 1 节：集中趋势 ---")
    income_central = calculate_central_tendency(df['income'])
    print(f"收入均值: {income_central['mean']:.2f}")
    print(f"收入中位数: {income_central['median']:.2f}")
    print(f"推荐: {recommend_central_tendency(df['income'])}")

    # 演示第 2 节：离散程度
    print("\n--- 第 2 节：离散程度 ---")
    income_dispersion = calculate_dispersion(df['income'])
    print(f"收入标准差: {income_dispersion['std']:.2f}")
    print(f"收入IQR: {income_dispersion['iqr']:.2f}")

    outliers = detect_outliers_iqr(df['income'])
    print(f"检测到 {outliers['outlier_count']} 个异常值")

    # 演示第 4 节：诚实可视化检查
    print("\n--- 第 4 节：可视化诚实性 ---")
    y_limits = (1000, 10000)
    data_range = df['income'].max() - df['income'].min()
    check_result = check_y_axis_baseline(y_limits, data_range)
    print(f"Y 轴诚实性检查: {check_result['is_honest']}")
    if not check_result['is_honest']:
        print(f"问题: {check_result['issue']}")

    print("\n" + "=" * 60)
    print("提示：运行此文件会演示所有函数，但不会生成文件")
    print("请在本文件基础上完成作业")
    print("=" * 60)


if __name__ == "__main__":
    setup_style()
    main()

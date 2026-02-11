#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
离散程度示例：标准差与 IQR 的对比

本示例展示如何计算和解释两种离散程度指标，
以及它们对异常值的敏感度差异。

运行方式：python 02_dispersion_demo.py
"""

import numpy as np
import pandas as pd


def demonstrate_dispersion():
    """演示离散程度的计算和对比"""

    print("=" * 60)
    print("离散程度示例：两组'均值相同'但波动完全不同的数据")
    print("=" * 60)

    # 两组数据：均值相同，但波动差异巨大
    group_a = [48, 49, 50, 51, 52]  # 稳定
    group_b = [10, 30, 50, 70, 90]  # 波动大

    print(f"\n组 A (稳定): {group_a}")
    print(f"组 B (波动): {group_b}")

    # 计算集中趋势
    mean_a = np.mean(group_a)
    mean_b = np.mean(group_b)
    print(f"\n组 A 均值: {mean_a}")
    print(f"组 B 均值: {mean_b}")
    print("→ 均值相同！但这是故事的一半...")

    # 计算标准差
    std_a = np.std(group_a, ddof=0)  # 总体标准差
    std_b = np.std(group_b, ddof=0)

    print("\n--- 标准差 (Standard Deviation) ---")
    print(f"组 A 标准差: {std_a:.2f}")
    print(f"组 B 标准差: {std_b:.2f}")
    print("\n解释：")
    print("• 标准差衡量'平均偏离程度'")
    print("• 组 B 的标准差约是组 A 的 7 倍")
    print("• 标准差对异常值敏感（基于所有数据点）")

    # 计算方差
    var_a = np.var(group_a, ddof=0)
    var_b = np.var(group_b, ddof=0)

    print("\n--- 方差 (Variance) ---")
    print(f"组 A 方差: {var_a:.2f}")
    print(f"组 B 方差: {var_b:.2f}")
    print("• 方差是标准差的平方，单位是'原单位的平方'")

    # 计算四分位距 IQR
    q1_a, q3_a = np.percentile(group_a, [25, 75])
    q1_b, q3_b = np.percentile(group_b, [25, 75])
    iqr_a = q3_a - q1_a
    iqr_b = q3_b - q1_b

    print("\n--- 四分位距 IQR (Interquartile Range) ---")
    print(f"组 A: Q1={q1_a}, Q3={q3_a}, IQR={iqr_a:.2f}")
    print(f"组 B: Q1={q1_b}, Q3={q3_b}, IQR={iqr_b:.2f}")
    print("\n解释：")
    print("• IQR 是中间 50% 数据的宽度")
    print("• 组 B 的 IQR 约是组 A 的 7 倍")
    print("• IQR 忽略两端的极端值，更稳健")

    # 对比表格
    print("\n" + "=" * 60)
    print("对比总结")
    print("=" * 60)
    print(f"{'指标':<15} {'组 A':<15} {'组 B':<15}")
    print("-" * 45)
    print(f"{'均值':<15} {mean_a:<15.2f} {mean_b:<15.2f}")
    print(f"{'标准差':<15} {std_a:<15.2f} {std_b:<15.2f}")
    print(f"{'IQR':<15} {iqr_a:<15.2f} {iqr_b:.2f}")
    print("-" * 45)

    return {
        'group_a': group_a,
        'group_b': group_b,
        'std_a': std_a,
        'std_b': std_b,
        'iqr_a': iqr_a,
        'iqr_b': iqr_b
    }


def demonstrate_outlier_detection():
    """演示用 IQR 检测异常值"""
    print("\n\n" + "=" * 60)
    print("异常值检测：基于 IQR 的 1.5×IQR 规则")
    print("=" * 60)

    # 包含异常值的数据
    data = [10, 12, 15, 18, 20, 22, 25, 28, 30, 35, 120]
    df = pd.DataFrame({'value': data})

    # 计算 IQR
    q1 = df['value'].quantile(0.25)
    q3 = df['value'].quantile(0.75)
    iqr = q3 - q1

    # 定义异常值边界（1.5倍 IQR 规则）
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    print(f"\n原始数据: {data}")
    print(f"\nQ1 (25%): {q1}")
    print(f"Q3 (75%): {q3}")
    print(f"IQR: {iqr:.2f}")

    print(f"\n异常值检测边界:")
    print(f"  下界: {lower_bound:.2f}")
    print(f"  上界: {upper_bound:.2f}")

    # 识别异常值
    outliers = df[(df['value'] < lower_bound) | (df['value'] > upper_bound)]

    print(f"\n检测到的异常值: {outliers['value'].tolist()}")
    print(f"异常值数量: {len(outliers)}")

    # 阿码的追问
    print("\n--- 阿码的困惑 ---")
    print("阿码问：'标准差和 IQR 该用哪个？'")
    print("\n回答：")
    print("• 想知道'整体波动范围' → 用标准差")
    print("• 想稳健地'检测异常值' → 用 IQR")
    print("• 如果发现标准差远大于 IQR → 说明有极端值在'捣乱'")

    return {
        'q1': q1,
        'q3': q3,
        'iqr': iqr,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'outliers': outliers['value'].tolist()
    }


def demonstrate_with_real_data():
    """用更真实的数据演示"""
    print("\n\n" + "=" * 60)
    print("真实场景：不同城市的房价波动")
    print("=" * 60)

    # 模拟三个城市的房价（单位：万元）
    cities = {
        'A城': [50, 52, 55, 58, 60, 62, 65, 68, 70, 75],
        'B城': [30, 45, 60, 75, 90, 105, 120, 150, 200],
        'C城': [48, 49, 50, 51, 52]
    }

    results = []

    for city, prices in cities.items():
        mean_val = np.mean(prices)
        std_val = np.std(prices, ddof=0)
        cv = std_val / mean_val  # 变异系数

        q1, q3 = np.percentile(prices, [25, 75])
        iqr_val = q3 - q1

        results.append({
            'city': city,
            'mean': mean_val,
            'std': std_val,
            'cv': cv,
            'iqr': iqr_val
        })

    # 输出对比表
    print(f"\n{'城市':<8} {'均值':<10} {'标准差':<10} {'变异系数':<10} {'IQR':<10}")
    print("-" * 50)
    for r in results:
        print(f"{r['city']:<8} {r['mean']:<10.2f} {r['std']:<10.2f} {r['cv']:<10.3f} {r['iqr']:<10.2f}")

    print("\n解释：")
    print("• B城均值与 C城相近，但标准差是 3 倍")
    print("• 变异系数（CV = 标准差/均值）可用于比较不同水平的波动")
    print("• 如果只看均值，会误以为 A城和 B城房价'差不多'")

    return results


if __name__ == "__main__":
    # 运行所有演示
    demonstrate_dispersion()
    demonstrate_outlier_detection()
    demonstrate_with_real_data()

    print("\n\n" + "=" * 60)
    print("核心要点总结")
    print("=" * 60)
    print("1. 波动不只是'噪音'——它告诉你结论有多可靠")
    print("2. 标准差：对异常值敏感，衡量整体波动")
    print("3. IQR：稳健，基于中间 50% 数据，适合异常值检测")
    print("4. 如果标准差远大于 IQR → 警告：有极端值在捣乱")

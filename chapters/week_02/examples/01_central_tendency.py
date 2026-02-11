#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
集中趋势示例：均值、中位数、众数的选择

本示例展示如何计算和解释三种集中趋势指标，
以及在不同场景下应该选择哪一个。

运行方式：python 01_central_tendency.py
"""

import numpy as np
import pandas as pd


def demonstrate_central_tendency():
    """演示集中趋势的计算和选择"""

    # 示例数据：一组用户消费金额（单位：元）
    # 包含一个极端值（超级大户）
    spending = [120, 150, 180, 200, 220, 250, 280, 300, 350, 1800]

    print("=" * 50)
    print("集中趋势示例：用户消费数据")
    print("=" * 50)
    print(f"\n原始数据: {spending}")
    print(f"数据点数: {len(spending)}")

    # 计算均值
    mean_val = np.mean(spending)
    print(f"\n--- 均值 (Mean) ---")
    print(f"均值 = {mean_val:.2f} 元")
    print("含义：如果所有消费平均分摊，每个人得到的金额")
    print("注意：受到极端值 1800 的强烈影响！")

    # 计算中位数
    median_val = np.median(spending)
    print(f"\n--- 中位数 (Median) ---")
    print(f"中位数 = {median_val:.2f} 元")
    print("含义：排序后中间位置的人的消费金额")
    print("优点：不受极端值影响")

    # 计算众数
    mode_val = pd.Series(spending).mode()
    if len(mode_val) > 0:
        mode_val = mode_val.iloc[0]
        print(f"\n--- 众数 (Mode) ---")
        print(f"众数 = {mode_val:.2f} 元")
        print("含义：出现频率最高的值")
        print("注意：本数据集中每个值只出现一次，众数意义不大")
    else:
        print("\n--- 众数 (Mode) ---")
        print("本数据集中没有重复值，众数不存在")

    # 对比与解释
    print("\n" + "=" * 50)
    print("如何选择？")
    print("=" * 50)
    print("场景 A：老板问'典型用户消费多少？'")
    print(f"  → 用均值 ({mean_val:.2f})：适合描述'整体规模'")
    print("  → 用中位数 ({:.2f})：适合描述'典型用户'".format(median_val))

    print("\n场景 B：预算制定，怕被极端值拉偏")
    print(f"  → 推荐中位数 ({median_val:.2f})：更稳健")

    print("\n场景 C：分析会员等级分布（如青铜/白银/黄金）")
    print("  → 推荐众数：最常见的等级")

    # 演示：有极端值 vs 无极端值
    print("\n" + "=" * 50)
    print("极端值的影响对比")
    print("=" * 50)

    spending_no_outlier = [120, 150, 180, 200, 220, 250, 280, 300, 350]
    mean_no_outlier = np.mean(spending_no_outlier)
    median_no_outlier = np.median(spending_no_outlier)

    print(f"\n去掉极端值后的数据: {spending_no_outlier}")
    print(f"均值: {mean_no_outlier:.2f} → 从 {mean_val:.2f} 降到 {mean_no_outlier:.2f}")
    print(f"中位数: {median_no_outlier:.2f} → 从 {median_val:.2f} 降到 {median_no_outlier:.2f}")
    print("\n结论：均值对极端值敏感，中位数更稳健")

    # 小北的常见错误
    print("\n" + "=" * 50)
    print("小北的错误：只看均值就下结论")
    print("=" * 50)
    print(f"小北说：'平均消费 {mean_val:.2f} 元，我们用户很富裕！'")
    print(f"老潘提醒：'但你没看到中位数只有 {median_val:.2f} 元，'")
    print(f"           说明 90% 的用户实际消费不到 {median_val:.2f} 元'")

    return {
        'mean': mean_val,
        'median': median_val,
        'data': spending
    }


def demonstrate_with_skewed_distribution():
    """演示偏态分布下均值与中位数的差异"""
    print("\n\n" + "=" * 50)
    print("偏态分布示例：网站访问时长")
    print("=" * 50)

    # 模拟右偏数据：大多数用户停留短，少数用户停留很长
    visit_duration = [1, 2, 3, 5, 8, 12, 15, 20, 30, 45, 60, 120, 180]

    mean_val = np.mean(visit_duration)
    median_val = np.median(visit_duration)

    print(f"数据: {visit_duration} 分钟")
    print(f"均值 = {mean_val:.2f} 分钟")
    print(f"中位数 = {median_val:.2f} 分钟")

    print("\n解释：")
    print("• 均值被少数'超长停留'用户拉高了")
    print("• 中位数更能反映'典型用户'的体验")

    # 计算偏度
    from scipy import stats
    skewness = stats.skew(visit_duration)
    print(f"\n偏度 (Skewness) = {skewness:.3f}")
    print("• 偏度 > 0：右偏（长尾向右）")
    print("• 右偏时：均值 > 中位数")

    return {
        'mean': mean_val,
        'median': median_val,
        'skewness': skewness
    }


if __name__ == "__main__":
    # 运行演示
    result1 = demonstrate_central_tendency()
    result2 = demonstrate_with_skewed_distribution()

    print("\n\n" + "=" * 50)
    print("核心要点总结")
    print("=" * 50)
    print("1. 均值：受极端值影响，适合描述'整体规模'")
    print("2. 中位数：稳健，适合描述'典型情况'")
    print("3. 众数：描述'最常见'，适合分类型数据")
    print("4. 选择前先看分布！")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
反例：EDA 和假设生成中的常见错误

本文件展示 Week 04 概念的错误用法，供读者对比学习。
这些代码可以运行，但会产生误导性结果或隐藏问题。

运行方式：python3 chapters/week_04/examples/00_bad_examples.py
预期输出：错误示范的输出，以及正确做法的对比说明
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def generate_sample_data(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """生成示例数据"""
    rng = np.random.default_rng(seed)

    df = pd.DataFrame({
        'age': np.clip(rng.normal(35, 10, n), 18, 65).astype(int),
        'income': np.clip(rng.lognormal(8.5, 0.5, n), 3000, None).astype(int),
        'spend': 0,
        'gender': rng.choice(['男', '女'], n),
    })

    # 消费与收入相关
    df['spend'] = (df['income'] * rng.uniform(0.2, 0.4, n)).astype(int)

    return df


# ==================== 错误 1：相关等于因果 ====================

def bad_correlation_to_causation(df: pd.DataFrame) -> None:
    """
    ❌ 错误做法：看到相关就推断因果

    问题：
    1. 相关不等于因果，可能有混杂变量
    2. 方向性不确定（A 导致 B 还是 B 导致 A？）
    3. 可能是虚假相关（两者都由第三个变量驱动）
    """
    corr = df['income'].corr(df['spend'])
    print(f"❌ 错误结论：'收入和消费相关系数 {corr:.3f}，说明收入增加导致消费增加'")
    print("   问题：相关不等于因果！可能有混杂变量（如年龄、城市级别）")


def good_correlation_cautious(df: pd.DataFrame) -> None:
    """
    ✅ 正确做法：谨慎解释相关性

    步骤：
    1. 报告相关系数和置信区间
    2. 提出可能的解释（包括混杂）
    3. 明确需要进一步验证
    """
    corr = df['income'].corr(df['spend'])
    print(f"✅ 谨慎表述：'收入和消费相关系数 {corr:.3f}，存在正相关关系'")
    print("   可能解释：")
    print("     1. 收入增加导致消费能力提升（因果）")
    print("     2. 年龄影响收入和消费（混杂）")
    print("     3. 两者由其他因素共同驱动（虚假相关）")
    print("   下一步：控制年龄等变量，进行更深入分析")


# ==================== 错误 2：忽视异常值的影响 ====================

def bad_ignore_outliers(df: pd.DataFrame) -> None:
    """
    ❌ 错误做法：不检查异常值直接计算 Pearson 相关

    问题：
    1. Pearson 对异常值敏感，一个极端值可能扭曲相关系数
    2. 可能错过数据质量问题
    3. 异常值可能是真实的 VIP 用户，不应简单删除
    """
    # 添加异常值
    df_outlier = df.copy()
    df_outlier.loc[0, 'income'] = 500000  # 极端高收入
    df_outlier.loc[0, 'spend'] = 200000   # 极端高消费

    corr = df_outlier['income'].corr(df_outlier['spend'])
    print(f"\n❌ 忽视异常值：Pearson r = {corr:.3f}")
    print("   问题：一个异常值可能大幅改变相关系数")


def good_handle_outliers(df: pd.DataFrame) -> None:
    """
    ✅ 正确做法：检查异常值并选择合适的相关系数

    步骤：
    1. 绘制散点图检查异常值
    2. 比较 Pearson 和 Spearman
    3. 对异常值进行分类（可疑/VIP/正常）
    """
    # 添加异常值
    df_outlier = df.copy()
    df_outlier.loc[0, 'income'] = 500000
    df_outlier.loc[0, 'spend'] = 200000

    pearson_r = df_outlier['income'].corr(df_outlier['spend'], method='pearson')
    spearman_r = df_outlier['income'].corr(df_outlier['spend'], method='spearman')

    print(f"\n✅ 正确处理异常值：")
    print(f"   Pearson r = {pearson_r:.3f}（对异常值敏感）")
    print(f"   Spearman ρ = {spearman_r:.3f}（更稳健）")
    print(f"   差异：{abs(pearson_r - spearman_r):.3f}")
    print("   如果差异大，需要检查异常值")


# ==================== 错误 3：分组比较不控制混杂 ====================

def bad_group_comparison_no_control(df: pd.DataFrame) -> None:
    """
    ❌ 错误做法：分组比较不控制混杂变量

    问题：
    1. 整体差异可能由混杂变量驱动
    2. 可能得出错误结论
    3. 忽略分层分析的重要性
    """
    # 创建收入分层（模拟女性收入更高）
    df['income_tier'] = pd.qcut(df['income'], q=2, labels=['低收入', '高收入'])

    # 整体比较
    gender_diff = df.groupby('gender')['spend'].mean()
    diff_pct = (gender_diff['女'] - gender_diff['男']) / gender_diff['男'] * 100

    print(f"\n❌ 不控制混杂：")
    print(f"   女性平均消费: {gender_diff['女']:.0f}")
    print(f"   男性平均消费: {gender_diff['男']:.0f}")
    print(f"   差异: {diff_pct:+.1f}%")
    print("   结论：'女性消费更高'")
    print("   问题：没有控制收入，可能收入才是驱动因素")


def good_group_comparison_controlled(df: pd.DataFrame) -> None:
    """
    ✅ 正确做法：分层分析控制混杂变量

    步骤：
    1. 识别潜在混杂变量（与分组变量和结果都相关）
    2. 在混杂变量的每个层内进行比较
    3. 比较层内差异与整体差异
    """
    df['income_tier'] = pd.qcut(df['income'], q=2, labels=['低收入', '高收入'])

    # 整体比较
    overall = df.groupby('gender')['spend'].mean()
    overall_diff = (overall['女'] - overall['男']) / overall['男'] * 100

    # 分层比较
    stratified = df.groupby(['income_tier', 'gender'])['spend'].mean().unstack()
    stratified['差异%'] = (stratified['女'] - stratified['男']) / stratified['男'] * 100

    print(f"\n✅ 控制混杂后：")
    print(f"   整体差异: {overall_diff:+.1f}%")
    print(f"   分层差异:")
    for tier in stratified.index:
        diff = stratified.loc[tier, '差异%']
        print(f"     {tier}: {diff:+.1f}%")

    avg_stratified_diff = abs(stratified['差异%']).mean()
    if avg_stratified_diff < abs(overall_diff) / 2:
        print("   结论：收入是混杂变量，控制后性别差异大幅减小")
    else:
        print("   结论：控制收入后，性别差异仍然存在")


# ==================== 错误 4：假设没有数据支持 ====================

def bad_hypothesis_without_evidence() -> list[dict]:
    """
    ❌ 错误做法：假设没有 EDA 数据支持

    问题：
    1. 假设变成拍脑袋，不是基于数据发现
    2. 浪费资源检验没有依据的假设
    3. 增加多重比较问题
    """
    hypotheses = [
        {
            'id': 'H1',
            'description': '用户星座与消费金额相关',
            'H0': '星座与消费无关',
            'H1': '星座与消费有关',
            'data_support': '无',  # ❌ 没有数据支持
            'proposed_test': 'ANOVA'
        }
    ]
    print("\n❌ 无数据支持的假设：")
    print("   '星座与消费相关'")
    print("   问题：没有 EDA 发现支持，属于拍脑袋")
    return hypotheses


def good_hypothesis_with_evidence() -> list[dict]:
    """
    ✅ 正确做法：每个假设都有 EDA 发现支持

    标准：
    1. 假设必须基于具体的 EDA 发现
    2. 记录发现的具体数据（相关系数、差异大小等）
    3. 区分探索性假设和验证性假设
    """
    hypotheses = [
        {
            'id': 'H1',
            'description': '用户收入与月消费存在正相关',
            'H0': '收入与消费相关系数 = 0',
            'H1': '收入与消费相关系数 > 0',
            'data_support': 'EDA 发现 r=0.52, n=200, p<0.001',  # ✅ 具体数据
            'proposed_test': 'Pearson 相关性检验'
        }
    ]
    print("\n✅ 有数据支持的假设：")
    print("   '收入与消费正相关'")
    print("   数据支持：EDA 发现 r=0.52, n=200, p<0.001")
    return hypotheses


# ==================== 错误 5：H0 和 H1 不对称 ====================

def bad_asymmetric_hypothesis() -> dict:
    """
    ❌ 错误做法：H0 和 H1 不对称

    问题：
    1. H0: r = 0, H1: r = 0.5（H1 太具体，无法检验）
    2. 应该使用复合备择假设
    """
    hypothesis = {
        'H0': '相关系数 = 0',
        'H1': '相关系数 = 0.5'  # ❌ 太具体
    }
    print("\n❌ 不对称假设：")
    print(f"   H0: {hypothesis['H0']}")
    print(f"   H1: {hypothesis['H1']}")
    print("   问题：H1 太具体，几乎不可能恰好等于 0.5")
    return hypothesis


def good_symmetric_hypothesis() -> dict:
    """
    ✅ 正确做法：H0 和 H1 互斥且穷尽

    标准：
    1. H0: 参数 = 某个值（通常是 0）
    2. H1: 参数 ≠ 某个值（双尾）或 >/<（单尾）
    """
    hypothesis = {
        'H0': '相关系数 = 0',
        'H1': '相关系数 ≠ 0'  # ✅ 复合备择假设
    }
    print("\n✅ 对称假设：")
    print(f"   H0: {hypothesis['H0']}")
    print(f"   H1: {hypothesis['H1']}")
    print("   正确：H1 包含所有 H0 以外的情况")
    return hypothesis


# ==================== 错误 6：不报告效应量 ====================

def bad_only_pvalue() -> None:
    """
    ❌ 错误做法：只关注 p 值，不报告效应量

    问题：
    1. 大样本下微小差异也会显著（p < 0.05）
    2. 统计显著 ≠ 实际重要
    3. 无法与其他研究比较
    """
    print("\n❌ 只报告显著性：")
    print("   '两组差异显著（p < 0.05）'")
    print("   问题：不知道差异有多大，是否实际重要")


def good_report_effect_size() -> None:
    """
    ✅ 正确做法：同时报告效应量

    标准：
    1. 相关性：报告 r（小 0.1，中 0.3，大 0.5）
    2. 均值差异：报告 Cohen's d（小 0.2，中 0.5，大 0.8）
    3. 方差分析：报告 eta-squared
    """
    print("\n✅ 报告效应量：")
    print("   '两组差异显著（p < 0.05），效应量中等（d = 0.6）'")
    print("   正确：既知道统计显著，又知道实际重要性")


# ==================== 主函数：对比展示 ====================

def main() -> None:
    """主函数：展示错误做法与正确做法的对比"""
    print("=" * 70)
    print("EDA 和假设生成常见错误（反例）")
    print("=" * 70)

    df = generate_sample_data(n=200, seed=42)
    print(f"\n示例数据: {len(df)} 行")

    # 错误 1
    print("\n" + "=" * 70)
    print("错误 1：相关等于因果")
    print("=" * 70)
    bad_correlation_to_causation(df)
    print()
    good_correlation_cautious(df)

    # 错误 2
    print("\n" + "=" * 70)
    print("错误 2：忽视异常值的影响")
    print("=" * 70)
    bad_ignore_outliers(df)
    good_handle_outliers(df)

    # 错误 3
    print("\n" + "=" * 70)
    print("错误 3：分组比较不控制混杂")
    print("=" * 70)
    bad_group_comparison_no_control(df)
    print()
    good_group_comparison_controlled(df)

    # 错误 4
    print("\n" + "=" * 70)
    print("错误 4：假设没有数据支持")
    print("=" * 70)
    bad_hypothesis_without_evidence()
    print()
    good_hypothesis_with_evidence()

    # 错误 5
    print("\n" + "=" * 70)
    print("错误 5：H0 和 H1 不对称")
    print("=" * 70)
    bad_asymmetric_hypothesis()
    print()
    good_symmetric_hypothesis()

    # 错误 6
    print("\n" + "=" * 70)
    print("错误 6：不报告效应量")
    print("=" * 70)
    bad_only_pvalue()
    good_report_effect_size()

    # 总结
    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print("好的 EDA 不是跑代码出图，而是：")
    print("  1. 谨慎解释相关性，不随意推断因果")
    print("  2. 检查异常值，选择合适的统计量")
    print("  3. 分组比较时控制混杂变量")
    print("  4. 每个假设都要有数据支持")
    print("  5. H0 和 H1 必须形式化、可检验")
    print("  6. 同时报告统计显著性和效应量")


if __name__ == "__main__":
    main()

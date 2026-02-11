#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
示例：缺失值概览与机制分析

本例演示如何系统性地分析缺失值：
1. 计算缺失率并排序
2. 分析缺失与观测值的关系（判断 MAR/MCAR）
3. 生成缺失概览表供报告使用

运行方式：python3 chapters/week_03/examples/01_missing_overview.py
预期输出：stdout 输出缺失概览表，以及 MAR 分析结果
"""
from __future__ import annotations

import pandas as pd
import numpy as np


def generate_sample_data_with_missing(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    生成带有不同缺失机制的示例数据

    模拟场景：用户消费数据分析
    - MCAR: age 随机缺失（模拟数据录入错误）
    - MAR: income 的缺失与 registration_days 相关（老用户更不愿意填收入）
    - MNAR: vip_score 在高消费用户中缺失更多（高消费用户隐私意识强）
    """
    rng = np.random.default_rng(seed)

    # 基础数据
    df = pd.DataFrame({
        'user_id': range(1, n + 1),
        'registration_days': rng.exponential(100, n),
        'total_spend': rng.lognormal(7, 1, n),
        'city': rng.choice(['北京', '上海', '广州', '深圳'], n),
    })

    df['registration_days'] = np.clip(df['registration_days'], 1, None).round(0).astype(int)
    df['total_spend'] = np.clip(df['total_spend'], 0, None).round(2)

    # 生成 age（MCAR：完全随机缺失）
    df['age'] = np.clip(rng.normal(35, 10, n), 18, None).round(0).astype(int)
    mcar_mask = rng.random(n) < 0.05  # 5% 随机缺失
    df.loc[mcar_mask, 'age'] = np.nan

    # 生成 income（MAR：与 registration_days 相关）
    df['income'] = np.clip(df['total_spend'] * 0.3 + rng.normal(5000, 2000, n), 3000, None)
    # 注册天数越长，越不愿意填收入
    mar_prob = 1 / (1 + np.exp(-(df['registration_days'] - 150) / 50))  # sigmoid
    mar_mask = rng.random(n) < mar_prob * 0.4  # 最高 40% 缺失率
    df.loc[mar_mask, 'income'] = np.nan

    # 生成 vip_score（MNAR：与自身值相关）
    df['vip_score'] = np.clip(df['total_spend'] / 100 + rng.normal(50, 15, n), 0, None)
    # 高消费用户（vip_score 高）更不愿意透露
    mnar_threshold = df['vip_score'].quantile(0.7)
    mnar_mask = (df['vip_score'] > mnar_threshold) & (rng.random(n) < 0.5)
    df.loc[mnar_mask, 'vip_score'] = np.nan

    return df


def missing_overview(df: pd.DataFrame) -> pd.DataFrame:
    """
    生成缺失值概览表

    返回 DataFrame，包含：
    - 缺失数量
    - 缺失率（百分比）
    - 数据类型
    """
    missing_count = df.isna().sum()
    missing_rate = (df.isna().mean() * 100).round(2)

    overview = pd.DataFrame({
        'missing_count': missing_count,
        'missing_rate_%': missing_rate,
        'dtype': df.dtypes,
        'total_count': len(df)
    })

    # 按缺失率降序排列
    overview = overview.sort_values('missing_rate_%', ascending=False)

    return overview


def analyze_mar_indicator(df: pd.DataFrame, missing_col: str, indicator_col: str) -> dict:
    """
    分析缺失是否与某列的取值相关（MAR 判断）

    原理：如果缺失组的 indicator_col 均值与观测组显著不同，
    则可能是 MAR（Missing At Random，依赖其他观测变量）

    参数：
        missing_col: 可能缺失的列
        indicator_col: 用于判断相关性的列

    返回：
        dict 包含分析结果
    """
    missing_mask = df[missing_col].isna()

    observed_mean = df.loc[~missing_mask, indicator_col].mean()
    missing_group_mean = df.loc[missing_mask, indicator_col].mean()

    # 计算差异百分比
    if observed_mean != 0:
        diff_pct = ((missing_group_mean - observed_mean) / observed_mean) * 100
    else:
        diff_pct = float('inf') if missing_group_mean > 0 else 0

    return {
        'missing_col': missing_col,
        'indicator_col': indicator_col,
        'observed_mean': observed_mean,
        'missing_group_mean': missing_group_mean,
        'difference_%': round(diff_pct, 2),
        'is_likely_mar': abs(diff_pct) > 10,  # 差异超过 10% 认为可能是 MAR
        'missing_count': missing_mask.sum(),
        'observed_count': (~missing_mask).sum()
    }


def print_missing_report(df: pd.DataFrame) -> None:
    """打印完整的缺失值分析报告"""
    print("=" * 60)
    print("缺失值概览报告")
    print("=" * 60)

    # 1. 基础概览
    overview = missing_overview(df)
    print("\n【缺失概览表】")
    print(overview[overview['missing_count'] > 0].to_string())

    # 2. MAR 分析
    print("\n【MAR 机制分析】")
    print("分析缺失是否与观测变量相关...")

    # 分析 income 的缺失是否与 registration_days 相关
    mar_result = analyze_mar_indicator(df, 'income', 'registration_days')
    print(f"\n字段: {mar_result['missing_col']}")
    print(f"  观测组平均注册天数: {mar_result['observed_mean']:.1f}")
    print(f"  缺失组平均注册天数: {mar_result['missing_group_mean']:.1f}")
    print(f"  差异: {mar_result['difference_%']:+.1f}%")
    if mar_result['is_likely_mar']:
        print(f"  → 判断: 可能是 MAR（与 {mar_result['indicator_col']} 相关）")
    else:
        print(f"  → 判断: 可能是 MCAR（与 {mar_result['indicator_col']} 无关）")

    # 3. 缺失模式总结
    print("\n【缺失模式总结】")
    complete_rows = df.dropna().shape[0]
    print(f"完整行数: {complete_rows} / {len(df)} ({100*complete_rows/len(df):.1f}%)")

    any_missing = df.isna().any(axis=1).sum()
    print(f"至少一个缺失: {any_missing} / {len(df)} ({100*any_missing/len(df):.1f}%)")


def main() -> None:
    """主函数"""
    # 生成示例数据
    df = generate_sample_data_with_missing(n=500, seed=42)

    print("数据样本（前 5 行）：")
    print(df.head())
    print(f"\n数据形状: {df.shape}")

    # 生成缺失报告
    print_missing_report(df)

    print("\n" + "=" * 60)
    print("小北的困惑：")
    print("=" * 60)
    print("'原来缺失不是随机的！老用户更不愿意填收入...'")
    print("'那我填充的时候是不是应该按注册时长分组填？'")

    print("\n阿码的追问：")
    print("'vip_score 的缺失跟消费金额有关，这是不是 MNAR？'")
    print("'MNAR 是不是最难处理，因为缺失本身携带信息？'")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
示例：异常值检测与分类处理

本例演示：
1. IQR 方法检测异常值（无分布假设）
2. Z-score 方法（假设正态分布）
3. 结合业务规则对异常值分类（suspicious/VIP/high_spend）
4. 不同处理策略的选择

运行方式：python3 chapters/week_03/examples/03_outlier_detection.py
预期输出：异常值检测结果、分类统计、处理建议
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from scipy import stats


def generate_sample_data(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    生成包含不同类型异常值的示例数据

    模拟场景：用户消费数据
    - 正常用户：消费 1000-10000 元
    - VIP 高消费：消费 50000+ 元（真实但极端）
    - 可疑值：消费为负或极其异常的高值（可能是数据错误）
    """
    rng = np.random.default_rng(seed)

    # 基础用户数据
    df = pd.DataFrame({
        'user_id': range(1, n + 1),
        'age': np.clip(rng.normal(35, 10, n), 18, None).round(0).astype(int),
        'registration_days': np.clip(rng.exponential(100, n), 1, None).round(0).astype(int),
        'city': rng.choice(['北京', '上海', '广州', '深圳'], n),
    })

    # 生成消费数据（对数正态分布）
    base_spend = rng.lognormal(8, 0.8, n)

    # 添加一些 VIP 用户（真实高消费，约 2%）
    vip_mask = rng.random(n) < 0.02
    base_spend[vip_mask] = base_spend[vip_mask] * 5 + 50000

    # 添加一些可疑值（数据错误，约 1%）
    error_mask = rng.random(n) < 0.01
    base_spend[error_mask] = -base_spend[error_mask]  # 负数消费

    df['total_spend'] = base_spend.round(2)

    return df


def detect_outliers_iqr(df: pd.DataFrame, column: str, multiplier: float = 1.5) -> pd.DataFrame:
    """
    使用 IQR 方法检测异常值

    原理：
    - Q1 = 第 25 百分位数
    - Q3 = 第 75 百分位数
    - IQR = Q3 - Q1
    - 下界 = Q1 - multiplier * IQR
    - 上界 = Q3 + multiplier * IQR

    优点：不假设分布，对极端值稳健
    缺点：可能标记过多小极端值

    参数：
        multiplier: 1.5 为常规异常值，3.0 为极端异常值
    """
    data = df[column].dropna()

    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr

    df_result = df.copy()
    df_result['is_outlier_iqr'] = (
        (df[column] < lower_bound) | (df[column] > upper_bound)
    )
    df_result['outlier_bounds'] = f'[{lower_bound:.2f}, {upper_bound:.2f}]'

    return df_result


def detect_outliers_zscore(df: pd.DataFrame, column: str, threshold: float = 3.0) -> pd.DataFrame:
    """
    使用 Z-score 方法检测异常值

    原理：
    - Z = (X - mean) / std
    - |Z| > threshold 认为是异常值

    优点：直观，有概率解释（正态假设下）
    缺点：对非正态分布数据效果差，受极端值影响大

    注意：本实现使用 MAD（中位数绝对偏差）修正版，更稳健
    """
    data = df[column].dropna()

    # 使用 MAD 修正的 Z-score（更稳健）
    median = data.median()
    mad = np.median(np.abs(data - median))

    if mad == 0:
        # 如果 MAD 为 0，使用标准 Z-score
        z_scores = np.abs(stats.zscore(data, nan_policy='omit'))
    else:
        # 修正 Z-score: 0.6745 * (x - median) / MAD
        z_scores = np.abs(0.6745 * (df[column] - median) / mad)

    df_result = df.copy()
    df_result['is_outlier_zscore'] = z_scores > threshold
    df_result['z_score'] = z_scores

    return df_result


def classify_outlier(row: pd.Series) -> str:
    """
    根据业务规则对异常值进行分类

    分类逻辑：
    - suspicious: 负数或极其异常的高值（可能是数据错误）
    - VIP: 高消费但合理的值（真实用户）
    - high_spend: 高于平均水平但不算异常
    - normal: 正常值
    """
    spend = row['total_spend']

    if spend < 0:
        return 'suspicious'  # 数据错误

    if spend > 100000:
        return 'suspicious'  # 极其异常，需要核实

    if spend > 50000:
        return 'VIP'  # 高消费真实用户

    if spend > 20000:
        return 'high_spend'  # 较高消费

    return 'normal'


def analyze_outliers(df: pd.DataFrame, column: str) -> dict:
    """
    综合分析异常值

    返回包含统计信息和处理建议的字典
    """
    # 应用两种检测方法
    df_iqr = detect_outliers_iqr(df, column, multiplier=1.5)
    df_zscore = detect_outliers_zscore(df, column, threshold=3.5)

    # 合并结果
    df['is_outlier_iqr'] = df_iqr['is_outlier_iqr']
    df['is_outlier_zscore'] = df_zscore['is_outlier_zscore']
    df['outlier_category'] = df.apply(classify_outlier, axis=1)

    # 统计
    iqr_outliers = df['is_outlier_iqr'].sum()
    zscore_outliers = df['is_outlier_zscore'].sum()

    category_counts = df['outlier_category'].value_counts().to_dict()

    # 计算两种方法的一致性
    agreement = (df['is_outlier_iqr'] == df['is_outlier_zscore']).mean()

    return {
        'total_samples': len(df),
        'iqr_outliers': int(iqr_outliers),
        'iqr_outlier_rate': round(iqr_outliers / len(df) * 100, 2),
        'zscore_outliers': int(zscore_outliers),
        'zscore_outlier_rate': round(zscore_outliers / len(df) * 100, 2),
        'method_agreement_%': round(agreement * 100, 2),
        'category_distribution': category_counts,
        'suspicious_values': df[df['outlier_category'] == 'suspicious'][column].tolist(),
        'vip_users': len(df[df['outlier_category'] == 'VIP']),
        'data_with_outlier_flags': df
    }


def recommend_outlier_treatment(category: str) -> str:
    """
    根据异常值类别推荐处理策略
    """
    recommendations = {
        'suspicious': {
            'action': '核实或删除',
            'method': '标记为缺失，人工核实；确认错误后删除或修正',
            'rationale': '数据录入错误，不应参与统计'
        },
        'VIP': {
            'action': '保留',
            'method': '单独分组分析，或在报告中说明存在高消费用户',
            'rationale': '真实用户行为，删除会扭曲业务理解'
        },
        'high_spend': {
            'action': '视分析目的而定',
            'method': '如果关注典型用户，可 Winsorize；如果关注总体，保留',
            'rationale': '边缘案例，处理方式取决于研究问题'
        },
        'normal': {
            'action': '无需处理',
            'method': '正常参与分析',
            'rationale': '非异常值'
        }
    }

    return recommendations.get(category, {'action': '未知', 'method': '', 'rationale': ''})


def winsorize_series(series: pd.Series, limits: tuple = (0.05, 0.05)) -> pd.Series:
    """
    Winsorize 处理：将极端值替换为指定分位数的值

    参数：
        limits: (下限, 上限) 要截断的比例
    """
    lower_limit = series.quantile(limits[0])
    upper_limit = series.quantile(1 - limits[1])

    return series.clip(lower=lower_limit, upper=upper_limit)


def main() -> None:
    """主函数"""
    df = generate_sample_data(n=500, seed=42)

    print("=" * 70)
    print("异常值检测与分析")
    print("=" * 70)

    print("\n【数据概览】")
    print(f"总样本量: {len(df)}")
    print(f"消费金额统计:")
    print(df['total_spend'].describe().round(2))

    # 异常值分析
    print("\n【异常值检测结果】")
    result = analyze_outliers(df, 'total_spend')

    print(f"IQR 方法检测到: {result['iqr_outliers']} 个异常值 ({result['iqr_outlier_rate']}%)")
    print(f"Z-score 方法检测到: {result['zscore_outliers']} 个异常值 ({result['zscore_outlier_rate']}%)")
    print(f"两种方法一致性: {result['method_agreement_%']}%")

    # 分类统计
    print("\n【异常值分类统计】")
    for category, count in result['category_distribution'].items():
        pct = count / len(df) * 100
        print(f"  {category}: {count} ({pct:.1f}%)")

    # 可疑值详情
    if result['suspicious_values']:
        print(f"\n【可疑值详情】")
        print(f"  发现 {len(result['suspicious_values'])} 个可疑值:")
        for val in result['suspicious_values'][:5]:  # 最多显示 5 个
            print(f"    - {val}")

    # VIP 用户
    print(f"\n【VIP 用户】")
    print(f"  发现 {result['vip_users']} 个 VIP 高消费用户")
    print(f"  建议：保留并单独分析，不要删除")

    # 处理建议
    print("\n" + "=" * 70)
    print("各类别处理建议")
    print("=" * 70)

    for category in ['suspicious', 'VIP', 'high_spend', 'normal']:
        rec = recommend_outlier_treatment(category)
        print(f"\n【{category}】")
        print(f"  建议操作: {rec['action']}")
        print(f"  处理方法: {rec['method']}")
        print(f"  理由: {rec['rationale']}")

    # Winsorize 演示
    print("\n【Winsorize 效果演示】")
    original_std = df['total_spend'].std()
    winsorized = winsorize_series(df['total_spend'], limits=(0.02, 0.02))
    winsorized_std = winsorized.std()

    print(f"  原始标准差: {original_std:.2f}")
    print(f"  Winsorize 后标准差: {winsorized_std:.2f}")
    print(f"  标准差变化: {(winsorized_std - original_std)/original_std*100:+.1f}%")

    print("\n" + "=" * 70)
    print("小北的发现：")
    print("=" * 70)
    print("'原来异常值不能一概而论！'")
    print("'负数消费肯定是错的，但那些高消费可能是真的 VIP 用户...'")
    print("'删除 VIP 用户会让我们的平均消费看起来很低，误导决策。'")

    print("\n阿码的追问：")
    print("'Winsorize 是不是比直接删除更好？'")
    print("'因为它保留了样本量，只是压缩了极端值？'")


if __name__ == "__main__":
    main()

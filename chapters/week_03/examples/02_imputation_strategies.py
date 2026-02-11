#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
示例：填充策略对比与决策记录

本例演示不同缺失值填充策略的效果对比：
1. 删除法（listwise deletion）
2. 均值/中位数填充
3. 分组填充（利用 MAR 信息）
4. 记录决策理由的模板

运行方式：python3 chapters/week_03/examples/02_imputation_strategies.py
预期输出：不同填充策略的对比结果，以及决策建议
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer


def generate_sample_data(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """生成带有缺失值的示例数据"""
    rng = np.random.default_rng(seed)

    df = pd.DataFrame({
        'user_id': range(1, n + 1),
        'registration_days': np.clip(rng.exponential(100, n), 1, None).round(0).astype(int),
        'total_spend': np.clip(rng.lognormal(7, 1, n), 0, None).round(2),
        'city': rng.choice(['北京', '上海', '广州', '深圳'], n),
    })

    # age: 5% MCAR
    df['age'] = np.clip(rng.normal(35, 10, n), 18, None).round(0).astype(int)
    df.loc[rng.random(n) < 0.05, 'age'] = np.nan

    # income: MAR（与老用户相关）
    df['income'] = np.clip(df['total_spend'] * 0.3 + rng.normal(5000, 2000, n), 3000, None).round(2)
    mar_prob = 1 / (1 + np.exp(-(df['registration_days'] - 150) / 50))
    df.loc[rng.random(n) < mar_prob * 0.4, 'income'] = np.nan

    return df


def strategy_delete(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    策略 1：删除含有缺失值的行

    优点：简单，不引入人工偏差
    缺点：损失样本，可能引入选择偏差（如果缺失非随机）
    """
    return df.dropna(subset=cols).copy()


def strategy_mean_imputation(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    策略 2：均值填充

    优点：保持总体均值不变
    缺点：降低方差，破坏相关性结构
    """
    df_filled = df.copy()
    mean_val = df[col].mean()
    df_filled[col] = df[col].fillna(mean_val)
    return df_filled


def strategy_median_imputation(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    策略 3：中位数填充

    优点：对异常值稳健，保持中位数不变
    缺点：同样会降低方差
    """
    df_filled = df.copy()
    median_val = df[col].median()
    df_filled[col] = df[col].fillna(median_val)
    return df_filled


def strategy_group_imputation(
    df: pd.DataFrame,
    target_col: str,
    group_col: str,
    method: str = 'median'
) -> pd.DataFrame:
    """
    策略 4：分组填充

    利用 MAR 信息：如果缺失与某分组变量相关，
    则按组分别计算填充值

    参数：
        target_col: 要填充的列
        group_col: 分组依据的列
        method: 'mean' 或 'median'
    """
    df_filled = df.copy()

    for group in df[group_col].unique():
        if pd.isna(group):
            continue

        mask = df[group_col] == group
        group_data = df.loc[mask, target_col]

        if method == 'mean':
            fill_val = group_data.mean()
        else:
            fill_val = group_data.median()

        df_filled.loc[mask & df[target_col].isna(), target_col] = fill_val

    return df_filled


def compare_imputation_strategies(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    对比不同填充策略对数据分布的影响

    返回对比表，包含：
    - 均值、中位数、标准差的变化
    - 样本量变化
    """
    results = []

    # 原始数据（仅观测值）
    observed = df[col].dropna()
    results.append({
        'strategy': '原始观测值',
        'n': len(observed),
        'mean': observed.mean(),
        'median': observed.median(),
        'std': observed.std(),
        'missing_filled': 0
    })

    # 删除法
    df_deleted = strategy_delete(df, [col])
    results.append({
        'strategy': '删除法',
        'n': len(df_deleted),
        'mean': df_deleted[col].mean(),
        'median': df_deleted[col].median(),
        'std': df_deleted[col].std(),
        'missing_filled': 0
    })

    # 均值填充
    df_mean = strategy_mean_imputation(df, col)
    results.append({
        'strategy': '均值填充',
        'n': len(df_mean),
        'mean': df_mean[col].mean(),
        'median': df_mean[col].median(),
        'std': df_mean[col].std(),
        'missing_filled': df[col].isna().sum()
    })

    # 中位数填充
    df_median = strategy_median_imputation(df, col)
    results.append({
        'strategy': '中位数填充',
        'n': len(df_median),
        'mean': df_median[col].mean(),
        'median': df_median[col].median(),
        'std': df_median[col].std(),
        'missing_filled': df[col].isna().sum()
    })

    # 分组中位数填充（按 city）
    df_group = strategy_group_imputation(df, col, 'city', method='median')
    results.append({
        'strategy': '分组中位数（按城市）',
        'n': len(df_group),
        'mean': df_group[col].mean(),
        'median': df_group[col].median(),
        'std': df_group[col].std(),
        'missing_filled': df[col].isna().sum()
    })

    return pd.DataFrame(results).round(2)


def create_imputation_decision_record(
    column: str,
    missing_rate: float,
    missing_mechanism: str,
    chosen_strategy: str,
    rationale: str,
    alternatives_considered: list[str]
) -> dict:
    """
    创建填充决策记录

    这是可复现分析的关键：你必须记录为什么选择某种填充策略
    """
    return {
        'column': column,
        'missing_rate_%': round(missing_rate * 100, 2),
        'missing_mechanism': missing_mechanism,
        'chosen_strategy': chosen_strategy,
        'rationale': rationale,
        'alternatives_considered': alternatives_considered,
        'timestamp': pd.Timestamp.now().isoformat()
    }


def print_decision_template() -> None:
    """打印决策记录模板"""
    template = """
【缺失值处理决策记录模板】

字段：__________
缺失率：__________%
缺失机制判断：
  □ MCAR（完全随机缺失）
  □ MAR（随机缺失，与其他变量相关）
  □ MNAR（非随机缺失，与自身值相关）
  □ 不确定

选择策略：__________
理由：
  ___________________________________
  ___________________________________

考虑过但未采用的策略：
  1. __________ 原因：__________
  2. __________ 原因：__________

对后续分析的潜在影响：
  ___________________________________

记录人：__________ 日期：__________
"""
    print(template)


def main() -> None:
    """主函数"""
    df = generate_sample_data(n=500, seed=42)

    print("=" * 70)
    print("缺失值填充策略对比")
    print("=" * 70)

    print("\n【数据概览】")
    print(f"总样本量: {len(df)}")
    print(f"age 缺失: {df['age'].isna().sum()} ({df['age'].isna().mean()*100:.1f}%)")
    print(f"income 缺失: {df['income'].isna().sum()} ({df['income'].isna().mean()*100:.1f}%)")

    # 对比 income 的不同填充策略
    print("\n【income 字段填充策略对比】")
    comparison = compare_imputation_strategies(df, 'income')
    print(comparison.to_string(index=False))

    print("\n【关键发现】")
    print("1. 删除法损失了 {:.1f}% 的样本".format(
        (1 - comparison[comparison['strategy']=='删除法']['n'].values[0]/len(df)) * 100
    ))
    print("2. 均值/中位数填充会降低标准差（方差收缩）")
    print("3. 分组填充保留了城市间的收入差异")

    # 演示分组填充的细节
    print("\n【分组填充详情（income 按 city）】")
    df_grouped = strategy_group_imputation(df, 'income', 'city', method='median')
    city_medians = df.groupby('city')['income'].median().round(2)
    print("各城市中位数（用于填充）：")
    for city, median in city_medians.items():
        filled_count = df[df['city']==city]['income'].isna().sum()
        print(f"  {city}: {median:.2f}（填充了 {filled_count} 个缺失值）")

    # 决策记录示例
    print("\n" + "=" * 70)
    print("决策记录示例")
    print("=" * 70)

    decision = create_imputation_decision_record(
        column='income',
        missing_rate=df['income'].isna().mean(),
        missing_mechanism='MAR（与 registration_days 相关）',
        chosen_strategy='分组中位数填充（按 city）',
        rationale='income 存在城市差异，分组填充能保留这种差异；'
                  '中位数对异常值稳健',
        alternatives_considered=[
            '删除法：损失样本过多（约 25%）',
            '均值填充：会低估方差，且忽略城市差异'
        ]
    )

    for key, value in decision.items():
        print(f"{key}: {value}")

    print("\n" + "=" * 70)
    print("决策记录模板")
    print("=" * 70)
    print_decision_template()

    print("\n老潘的点评：")
    print("'没有最好的填充策略，只有最适合你数据的策略。'")
    print("'关键是把决策过程写下来，让读者能评判你的选择。'")


if __name__ == "__main__":
    main()

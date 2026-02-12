#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
示例：相关性分析——Pearson vs Spearman 与缺失值处理

本例演示：
1. 计算 Pearson 和 Spearman 相关系数
2. 绘制相关性热力图
3. 比较成对删除 vs 完全删除的差异
4. 展示异常值对 Pearson 的影响

运行方式：python3 chapters/week_04/examples/04_correlation_analysis.py
预期输出：stdout 输出相关系数对比表，并保存热力图到 correlation_heatmap.png
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def generate_ecommerce_data(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """
    生成模拟电商用户数据

    变量：
    - age: 年龄（18-65岁）
    - monthly_income: 月收入（对数正态分布）
    - monthly_spend: 月消费（与收入相关）
    - vip_score: VIP 评分（与消费相关）
    """
    rng = np.random.default_rng(seed)

    df = pd.DataFrame({
        'user_id': range(1, n + 1),
        'age': np.clip(rng.normal(35, 10, n), 18, 65).astype(int),
        'monthly_income': np.clip(rng.lognormal(8.5, 0.5, n), 3000, 50000).astype(int),
    })

    # 消费与收入相关（模拟真实业务逻辑）
    df['monthly_spend'] = (
        df['monthly_income'] * 0.3 * rng.uniform(0.5, 1.5, n) +
        rng.lognormal(6, 0.5, n) * 100
    ).astype(int)

    # VIP 评分与消费相关
    df['vip_score'] = (
        df['monthly_spend'] / 100 +
        rng.normal(50, 15, n)
    ).clip(0, 100).round(1)

    return df


def add_missing_values(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    添加缺失值用于演示删除策略

    - income: 5% 随机缺失
    - spend: 3% 随机缺失（与 income 部分重叠）
    """
    rng = np.random.default_rng(seed)
    df = df.copy()

    # 5% income 缺失
    income_missing = rng.choice(df.index, size=int(len(df) * 0.05), replace=False)
    df.loc[income_missing, 'monthly_income'] = np.nan

    # 3% spend 缺失（部分与 income 重叠）
    spend_missing = rng.choice(df.index, size=int(len(df) * 0.03), replace=False)
    df.loc[spend_missing, 'monthly_spend'] = np.nan

    return df


def calculate_correlations(df: pd.DataFrame, cols: list[str]) -> dict:
    """
    计算 Pearson 和 Spearman 相关系数

    返回：
        dict: 包含 pearson 和 spearman 矩阵的字典
    """
    return {
        'pearson': df[cols].corr(method='pearson'),
        'spearman': df[cols].corr(method='spearman')
    }


def compare_deletion_strategies(df: pd.DataFrame, cols: list[str]) -> dict:
    """
    比较成对删除 vs 完全删除的差异

    成对删除：每对变量独立计算，使用各自可用的样本
    完全删除：只使用所有变量都完整的样本
    """
    # 成对删除（pandas 默认行为）
    corr_pairwise = df[cols].corr(method='pearson')

    # 完全删除
    df_complete = df[cols].dropna()
    corr_complete = df_complete.corr(method='pearson')

    return {
        'pairwise': corr_pairwise,
        'complete': corr_complete,
        'pairwise_n': df[cols].count(),
        'complete_n': len(df_complete)
    }


def plot_correlation_heatmap(corr_matrix: pd.DataFrame, title: str, output_path: str) -> None:
    """绘制相关性热力图并保存"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        fmt='.3f',
        annot_kws={'size': 10}
    )
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"热力图已保存: {output_path}")
    plt.close()


def demonstrate_outlier_effect() -> None:
    """
    演示异常值对 Pearson vs Spearman 的影响

    添加一个极端异常值，观察两种系数的变化
    """
    # 生成基础数据（弱相关）
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y = np.array([2, 4, 3, 5, 6, 5, 7, 8, 7, 9])

    # 添加异常值
    x_outlier = np.append(x, 100)
    y_outlier = np.append(y, 100)

    # 计算系数
    pearson_normal = np.corrcoef(x, y)[0, 1]
    spearman_normal = pd.Series(x).corr(pd.Series(y), method='spearman')

    pearson_outlier = np.corrcoef(x_outlier, y_outlier)[0, 1]
    spearman_outlier = pd.Series(x_outlier).corr(pd.Series(y_outlier), method='spearman')

    print("\n" + "=" * 60)
    print("异常值影响演示")
    print("=" * 60)
    print(f"正常数据（10个样本）：")
    print(f"  Pearson r = {pearson_normal:.3f}")
    print(f"  Spearman ρ = {spearman_normal:.3f}")
    print(f"\n添加异常值 (100, 100) 后：")
    print(f"  Pearson r = {pearson_outlier:.3f} (变化: {pearson_outlier - pearson_normal:+.3f})")
    print(f"  Spearman ρ = {spearman_outlier:.3f} (变化: {spearman_outlier - spearman_normal:+.3f})")
    print("\n结论：Pearson 对异常值敏感，Spearman 更稳健")


def print_correlation_report(corrs: dict, deletion_comparison: dict, cols: list[str]) -> None:
    """打印相关性分析报告"""
    print("\n" + "=" * 60)
    print("相关性分析报告")
    print("=" * 60)

    # Pearson vs Spearman 对比
    print("\n【Pearson vs Spearman 对比】")
    print("-" * 60)
    print(f"{'变量对':<25} {'Pearson r':<12} {'Spearman ρ':<12} {'差异':<10}")
    print("-" * 60)

    for i, col1 in enumerate(cols):
        for col2 in cols[i+1:]:
            pearson = corrs['pearson'].loc[col1, col2]
            spearman = corrs['spearman'].loc[col1, col2]
            diff = abs(pearson - spearman)
            print(f"{col1} - {col2:<15} {pearson:>8.3f}    {spearman:>8.3f}    {diff:>8.3f}")

    # 删除策略对比
    print("\n【成对删除 vs 完全删除对比】")
    print("-" * 60)
    print(f"各变量有效样本量（成对删除）：")
    for col in cols:
        print(f"  {col}: {deletion_comparison['pairwise_n'][col]} 个")

    print(f"\n完全删除后样本量: {deletion_comparison['complete_n']} 个")
    print(f"原始数据样本量: {deletion_comparison['pairwise_n'].iloc[0]} 个")

    print("\n收入-消费相关系数对比：")
    income_spend_pw = deletion_comparison['pairwise'].loc['monthly_income', 'monthly_spend']
    income_spend_cm = deletion_comparison['complete'].loc['monthly_income', 'monthly_spend']
    print(f"  成对删除: r = {income_spend_pw:.3f}")
    print(f"  完全删除: r = {income_spend_cm:.3f}")
    print(f"  差异: {abs(income_spend_pw - income_spend_cm):.3f}")


def main() -> None:
    """主函数"""
    print("=" * 60)
    print("相关性分析示例")
    print("=" * 60)

    # 1. 生成数据
    print("\n[1/5] 生成模拟电商数据...")
    df = generate_ecommerce_data(n=200, seed=42)
    df = add_missing_values(df, seed=42)
    print(f"  数据形状: {df.shape}")
    print(f"  缺失情况:\n{df[['age', 'monthly_income', 'monthly_spend', 'vip_score']].isna().sum()}")

    # 2. 计算相关性
    print("\n[2/5] 计算相关系数...")
    numeric_cols = ['age', 'monthly_income', 'monthly_spend', 'vip_score']
    corrs = calculate_correlations(df, numeric_cols)

    # 3. 比较删除策略
    print("\n[3/5] 比较删除策略...")
    deletion_comparison = compare_deletion_strategies(df, numeric_cols)

    # 4. 打印报告
    print("\n[4/5] 生成报告...")
    print_correlation_report(corrs, deletion_comparison, numeric_cols)

    # 5. 绘制热力图
    print("\n[5/5] 绘制热力图...")
    plot_correlation_heatmap(
        corrs['pearson'],
        '变量相关性热力图 (Pearson)',
        'correlation_heatmap_pearson.png'
    )
    plot_correlation_heatmap(
        corrs['spearman'],
        '变量相关性热力图 (Spearman)',
        'correlation_heatmap_spearman.png'
    )

    # 6. 异常值影响演示
    demonstrate_outlier_effect()

    # 角色对话
    print("\n" + "=" * 60)
    print("小北的困惑：")
    print("=" * 60)
    print("'原来 Pearson 和 Spearman 差别这么大！'")
    print("'那我有异常值的时候应该用 Spearman 吗？'")

    print("\n老潘的建议：")
    print("'看你想测什么。Pearson 测线性关系，Spearman 测单调关系。'")
    print("'如果有异常值，两个都算算，看看差异大不大。'")
    print("'差异大说明异常值在作祟，需要仔细审查。'")

    print("\n" + "=" * 60)
    print("关键结论")
    print("=" * 60)
    print("1. Pearson 对异常值敏感，Spearman 更稳健")
    print("2. 成对删除保留更多样本，但不同相关系数基于不同样本")
    print("3. 完全删除保证一致性，但可能损失大量样本")
    print("4. 报告相关系数时必须说明样本量和缺失处理方式")


if __name__ == "__main__":
    main()

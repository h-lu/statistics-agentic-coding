#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
示例：分组比较——Groupby 与透视表

本例演示：
1. 使用 groupby 进行分组统计
2. 创建透视表进行交叉分析
3. 分组可视化（箱线图、小提琴图）
4. 发现群体间差异并解释业务含义

运行方式：python3 chapters/week_04/examples/04_group_comparison.py
预期输出：stdout 输出分组统计表，并保存可视化图表
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def generate_ecommerce_data(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """
    生成模拟电商用户数据，包含分组变量

    变量：
    - age, monthly_income, monthly_spend: 数值变量
    - user_level: 用户等级（普通/银卡/金卡/钻石）
    - city_tier: 城市级别（一线/二线/三线）
    - gender: 性别（男/女）
    """
    rng = np.random.default_rng(seed)

    # 基础数据
    df = pd.DataFrame({
        'user_id': range(1, n + 1),
        'age': np.clip(rng.normal(35, 12, n), 18, 70).astype(int),
        'monthly_income': np.clip(rng.lognormal(8.5, 0.6, n), 3000, 80000).astype(int),
        'gender': rng.choice(['男', '女'], n, p=[0.48, 0.52]),
        'city_tier': rng.choice(['一线', '二线', '三线'], n, p=[0.3, 0.45, 0.25]),
    })

    # 消费与收入、城市级别相关
    base_spend = df['monthly_income'] * rng.uniform(0.15, 0.35, n)
    city_multiplier = df['city_tier'].map({'一线': 1.3, '二线': 1.0, '三线': 0.8})
    df['monthly_spend'] = (base_spend * city_multiplier + rng.normal(0, 500, n)).astype(int).clip(100, None)

    # 根据消费划分用户等级
    spend_bins = [0, 1000, 3000, 8000, float('inf')]
    spend_labels = ['普通', '银卡', '金卡', '钻石']
    df['user_level'] = pd.cut(df['monthly_spend'], bins=spend_bins, labels=spend_labels)

    return df


def groupby_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    使用 groupby 进行分组统计分析

    按用户等级分组，计算年龄、收入、消费的描述统计
    """
    group_stats = df.groupby('user_level').agg({
        'age': ['mean', 'std', 'median', 'count'],
        'monthly_income': ['mean', 'std', 'median'],
        'monthly_spend': ['mean', 'std', 'median', 'min', 'max']
    }).round(2)

    # 扁平化列名
    group_stats.columns = ['_'.join(col).strip() for col in group_stats.columns.values]

    return group_stats


def create_pivot_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    创建透视表：城市级别 × 用户等级 → 平均消费
    """
    pivot = pd.pivot_table(
        df,
        values='monthly_spend',
        index='city_tier',
        columns='user_level',
        aggfunc='mean',
        margins=True,
        margins_name='总计'
    ).round(0)

    return pivot


def create_cross_tabulation(df: pd.DataFrame) -> pd.DataFrame:
    """
    创建交叉表：城市级别 vs 用户等级的频数分布
    """
    cross_tab = pd.crosstab(
        df['city_tier'],
        df['user_level'],
        margins=True,
        margins_name='总计'
    )

    # 计算百分比
    cross_tab_pct = pd.crosstab(
        df['city_tier'],
        df['user_level'],
        normalize='index'
    ) * 100

    return cross_tab, cross_tab_pct.round(1)


def plot_group_comparison(df: pd.DataFrame, output_dir: str = '.') -> None:
    """
    绘制分组比较可视化图表
    """
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 图1: 按用户等级的消费分布（箱线图）
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 箱线图
    sns.boxplot(data=df, x='user_level', y='monthly_spend', ax=axes[0, 0])
    axes[0, 0].set_title('消费分布 by 用户等级（箱线图）')
    axes[0, 0].set_xlabel('用户等级')
    axes[0, 0].set_ylabel('月消费（元）')

    # 小提琴图
    sns.violinplot(data=df, x='city_tier', y='monthly_income', ax=axes[0, 1])
    axes[0, 1].set_title('收入分布 by 城市级别（小提琴图）')
    axes[0, 1].set_xlabel('城市级别')
    axes[0, 1].set_ylabel('月收入（元）')

    # 分组柱状图：平均消费 by 城市级别和性别
    grouped = df.groupby(['city_tier', 'gender'])['monthly_spend'].mean().unstack()
    grouped.plot(kind='bar', ax=axes[1, 0], rot=0)
    axes[1, 0].set_title('平均消费 by 城市级别和性别')
    axes[1, 0].set_xlabel('城市级别')
    axes[1, 0].set_ylabel('平均月消费（元）')
    axes[1, 0].legend(title='性别')

    # 热力图：城市级别 × 用户等级的平均消费
    pivot = create_pivot_table(df)
    # 去掉总计行/列用于热力图
    pivot_clean = pivot.iloc[:-1, :-1]
    sns.heatmap(pivot_clean, annot=True, fmt='.0f', cmap='YlOrRd', ax=axes[1, 1])
    axes[1, 1].set_title('平均消费热力图（城市 × 等级）')
    axes[1, 1].set_xlabel('用户等级')
    axes[1, 1].set_ylabel('城市级别')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/group_comparison_plots.png', dpi=150, bbox_inches='tight')
    print(f"分组比较图表已保存: {output_dir}/group_comparison_plots.png")
    plt.close()


def analyze_group_differences(df: pd.DataFrame) -> dict:
    """
    分析群体间差异，提取关键发现
    """
    findings = {}

    # 发现1: 不同等级的消费差异
    level_spend = df.groupby('user_level')['monthly_spend'].mean()
    findings['level_spend_ratio'] = level_spend['钻石'] / level_spend['普通']

    # 发现2: 城市级别差异
    city_spend = df.groupby('city_tier')['monthly_spend'].mean()
    findings['city_spend_ratio'] = city_spend['一线'] / city_spend['三线']

    # 发现3: 性别差异
    gender_spend = df.groupby('gender')['monthly_spend'].mean()
    findings['gender_diff'] = gender_spend['女'] - gender_spend['男']
    findings['gender_diff_pct'] = (gender_spend['女'] - gender_spend['男']) / gender_spend['男'] * 100

    # 发现4: 年龄差异
    age_by_level = df.groupby('user_level')['age'].mean()
    findings['age_diff'] = age_by_level['钻石'] - age_by_level['普通']

    return findings


def print_group_report(df: pd.DataFrame, group_stats: pd.DataFrame,
                       pivot: pd.DataFrame, cross_tab: pd.DataFrame,
                       cross_tab_pct: pd.DataFrame, findings: dict) -> None:
    """打印分组比较报告"""
    print("\n" + "=" * 70)
    print("分组比较分析报告")
    print("=" * 70)

    # Groupby 统计
    print("\n【按用户等级的分组统计】")
    print("-" * 70)
    print(group_stats.to_string())

    # 透视表
    print("\n【城市级别 × 用户等级的平均消费透视表】")
    print("-" * 70)
    print(pivot.to_string())

    # 交叉表
    print("\n【城市级别 × 用户等级的频数分布】")
    print("-" * 70)
    print(cross_tab.to_string())

    print("\n【城市级别 × 用户等级的百分比分布（行百分比）】")
    print("-" * 70)
    print(cross_tab_pct.to_string())

    # 关键发现
    print("\n【关键发现】")
    print("-" * 70)
    print(f"1. 等级消费差异：钻石用户消费是普通用户的 {findings['level_spend_ratio']:.1f} 倍")
    print(f"2. 城市消费差异：一线城市消费是三线城市的 {findings['city_spend_ratio']:.1f} 倍")
    print(f"3. 性别消费差异：女性比男性平均高 {findings['gender_diff']:.0f} 元 ({findings['gender_diff_pct']:+.1f}%)")
    print(f"4. 年龄差异：钻石用户平均年龄比普通用户大 {findings['age_diff']:.1f} 岁")


def main() -> None:
    """主函数"""
    print("=" * 70)
    print("分组比较示例")
    print("=" * 70)

    # 1. 生成数据
    print("\n[1/5] 生成模拟电商数据...")
    df = generate_ecommerce_data(n=300, seed=42)
    print(f"  数据形状: {df.shape}")
    print(f"  样本分布:")
    print(f"    用户等级: {df['user_level'].value_counts().to_dict()}")
    print(f"    城市级别: {df['city_tier'].value_counts().to_dict()}")

    # 2. Groupby 分析
    print("\n[2/5] 执行 groupby 分组统计...")
    group_stats = groupby_analysis(df)

    # 3. 透视表
    print("\n[3/5] 创建透视表...")
    pivot = create_pivot_table(df)

    # 4. 交叉表
    print("\n[4/5] 创建交叉表...")
    cross_tab, cross_tab_pct = create_cross_tabulation(df)

    # 5. 分析差异
    print("\n[5/5] 分析群体差异...")
    findings = analyze_group_differences(df)

    # 打印报告
    print_group_report(df, group_stats, pivot, cross_tab, cross_tab_pct, findings)

    # 6. 可视化
    print("\n[6/5] 生成可视化图表...")
    plot_group_comparison(df)

    # 角色对话
    print("\n" + "=" * 70)
    print("阿码的追问：")
    print("=" * 70)
    print("'一线城市的金卡用户消费特别高，这说明什么？'")
    print("'是城市效应还是收入效应？'")

    print("\n老潘的提醒：")
    print("=" * 70)
    print("'现象不等于解释。'")
    print("'可能的原因有很多：生活成本高、收入高、运营策略不同...'")
    print("'EDA 的任务是发现现象，验证解释需要更严格的研究设计。'")

    print("\n" + "=" * 70)
    print("关键结论")
    print("=" * 70)
    print("1. Groupby 适合一维分组聚合，透视表适合二维交叉分析")
    print("2. 箱线图展示分布形状，小提琴图展示密度估计")
    print("3. 发现群体差异后，要追问'为什么'并标记潜在混杂变量")
    print("4. 交叉表的行/列百分比可以揭示不同的分布模式")


if __name__ == "__main__":
    main()

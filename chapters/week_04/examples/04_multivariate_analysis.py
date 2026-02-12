#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
示例：多变量分析——混杂变量识别与辛普森悖论

本例演示：
1. 识别混杂变量（收入对性别-消费关系的影响）
2. 分层分析（stratified analysis）
3. 辛普森悖论演示
4. 多变量可视化（散点图矩阵）

运行方式：python3 chapters/week_04/examples/04_multivariate_analysis.py
预期输出：stdout 输出分层分析结果，展示辛普森悖论
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def generate_confounding_data(n: int = 400, seed: int = 42) -> pd.DataFrame:
    """
    生成包含混杂变量的数据

    场景：分析性别与消费的关系，但收入是混杂变量
    - 女性平均收入高于男性（模拟特定行业/场景）
    - 消费与收入强相关
    - 控制收入后，性别差异消失甚至反转
    """
    rng = np.random.default_rng(seed)

    # 性别（女性收入更高，模拟特定行业）
    gender = rng.choice(['男', '女'], n, p=[0.5, 0.5])

    # 收入与性别相关（女性收入高）
    income = np.where(
        gender == '女',
        rng.lognormal(8.6, 0.5, n),  # 女性收入分布
        rng.lognormal(8.3, 0.5, n)   # 男性收入分布
    ) * 1000

    # 消费主要由收入决定，与性别无直接关系
    spend = income * rng.uniform(0.25, 0.35, n) + rng.normal(0, 300, n)

    df = pd.DataFrame({
        'user_id': range(1, n + 1),
        'gender': gender,
        'monthly_income': income.clip(3000, 100000).astype(int),
        'monthly_spend': spend.clip(100, None).astype(int),
        'age': np.clip(rng.normal(35, 10, n), 18, 65).astype(int),
    })

    return df


def generate_simpsons_paradox_data(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """
    生成展示辛普森悖论的数据

    辛普森悖论：整体趋势与分组趋势相反
    场景：两种治疗方法（A/B），两个医院（中心/社区）
    - A 疗法在每家医院都更好
    - 但整体看 B 疗法似乎更好（因为重症患者更多选择 A）
    """
    rng = np.random.default_rng(seed)

    # 医院类型：中心医院接收更多重症患者
    hospital = ['中心'] * (n // 2) + ['社区'] * (n // 2)
    hospital = rng.permutation(hospital)

    # 病情严重程度
    severity = np.where(
        np.array(hospital) == '中心',
        rng.beta(2, 1, n),  # 中心医院：更多重症
        rng.beta(1, 2, n)   # 社区医院：更多轻症
    )

    # 治疗方法：重症更倾向选择 A 疗法
    treatment = np.where(
        severity > 0.5,
        rng.choice(['A', 'B'], n, p=[0.7, 0.3]),
        rng.choice(['A', 'B'], n, p=[0.3, 0.7])
    )

    # 治愈率：A 疗法更好，但重症患者治愈率本身就低
    base_rate = 0.9 - severity * 0.4  # 基础治愈率（随病情降低）
    treatment_effect = np.where(treatment == 'A', 0.08, 0)  # A 疗法 +8% 治愈率
    cure_prob = base_rate + treatment_effect

    cured = rng.random(n) < cure_prob

    df = pd.DataFrame({
        'patient_id': range(1, n + 1),
        'hospital': hospital,
        'treatment': treatment,
        'severity': severity.round(3),
        'cured': cured,
    })

    return df


def stratified_analysis(df: pd.DataFrame, target_col: str,
                        group_col: str, stratify_col: str) -> pd.DataFrame:
    """
    分层分析：在控制 stratify_col 后，看 target_col 与 group_col 的关系

    参数：
        target_col: 目标变量（如消费）
        group_col: 分组变量（如性别）
        stratify_col: 分层变量（如收入分层）
    """
    # 创建收入分层
    df = df.copy()
    # 使用 rank 方法避免重复边界问题
    df['income_tier'] = pd.qcut(df[stratify_col].rank(method='first'), q=3, labels=['低收入', '中收入', '高收入'])

    # 整体统计
    overall = df.groupby(group_col)[target_col].mean()

    # 分层统计
    stratified = df.groupby(['income_tier', group_col])[target_col].mean().unstack()

    # 计算层内差异
    if '女' in stratified.columns and '男' in stratified.columns:
        stratified['差异(女-男)'] = stratified['女'] - stratified['男']
        stratified['差异%'] = (stratified['女'] - stratified['男']) / stratified['男'] * 100

    return overall, stratified


def analyze_simpsons_paradox(df: pd.DataFrame) -> dict:
    """
    分析辛普森悖论

    比较整体治愈率和分层（按医院）治愈率
    """
    # 整体治愈率
    overall = df.groupby('treatment')['cured'].mean()

    # 分层治愈率（按医院）
    by_hospital = df.groupby(['hospital', 'treatment'])['cured'].mean().unstack()

    # 各医院样本量
    counts = df.groupby(['hospital', 'treatment']).size().unstack()

    return {
        'overall': overall,
        'by_hospital': by_hospital,
        'counts': counts
    }


def plot_multivariate_relationships(df: pd.DataFrame, output_path: str = 'pairplot.png') -> None:
    """
    绘制多变量关系图（散点图矩阵）
    """
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 选择数值列和分组变量
    plot_cols = ['age', 'monthly_income', 'monthly_spend']

    # 散点图矩阵
    g = sns.pairplot(
        df[plot_cols + ['gender']],
        hue='gender',
        diag_kind='kde',
        plot_kws={'alpha': 0.6, 's': 30},
        height=2.5
    )
    g.fig.suptitle('变量关系散点图矩阵（按性别分色）', y=1.02, fontsize=14)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"散点图矩阵已保存: {output_path}")
    plt.close()


def plot_confounding_illustration(df: pd.DataFrame, output_path: str = 'confounding.png') -> None:
    """
    绘制混杂变量示意图
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：整体散点图
    for gender, color in [('男', 'blue'), ('女', 'red')]:
        subset = df[df['gender'] == gender]
        axes[0].scatter(subset['monthly_income'], subset['monthly_spend'],
                       alpha=0.5, label=gender, color=color)

    axes[0].set_xlabel('月收入（元）')
    axes[0].set_ylabel('月消费（元）')
    axes[0].set_title('收入-消费关系（按性别分色）')
    axes[0].legend()

    # 右图：收入分层后的箱线图
    df['income_tier'] = pd.qcut(df['monthly_income'], q=3, labels=['低收入', '中收入', '高收入'])
    sns.boxplot(data=df, x='income_tier', y='monthly_spend', hue='gender', ax=axes[1])
    axes[1].set_xlabel('收入分层')
    axes[1].set_ylabel('月消费（元）')
    axes[1].set_title('控制收入后的性别消费差异')
    axes[1].legend(title='性别')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"混杂变量示意图已保存: {output_path}")
    plt.close()


def print_multivariate_report(df: pd.DataFrame, overall: pd.Series,
                              stratified: pd.DataFrame, simpsons: dict) -> None:
    """打印多变量分析报告"""
    print("\n" + "=" * 70)
    print("多变量分析报告：混杂变量与辛普森悖论")
    print("=" * 70)

    # 混杂变量分析
    print("\n【混杂变量分析：性别-消费关系】")
    print("-" * 70)

    print("\n整体统计（未控制收入）：")
    print(f"  男性平均消费: {overall['男']:.0f} 元")
    print(f"  女性平均消费: {overall['女']:.0f} 元")
    print(f"  差异: {overall['女'] - overall['男']:+.0f} 元 ({(overall['女']/overall['男']-1)*100:+.1f}%)")

    print("\n分层统计（控制收入后）：")
    print(stratified.to_string())

    print("\n结论：")
    if abs(stratified.loc[:, '差异%'].mean()) < 5:
        print("  控制收入后，性别差异基本消失！")
        print("  → 收入是性别-消费关系的混杂变量")
    else:
        print("  控制收入后，性别差异仍然存在")
        print("  → 可能有其他混杂变量或真实的性别效应")

    # 辛普森悖论分析
    print("\n" + "=" * 70)
    print("【辛普森悖论演示：医院治愈率】")
    print("=" * 70)

    print("\n整体治愈率：")
    for treatment in simpsons['overall'].index:
        rate = simpsons['overall'][treatment]
        print(f"  {treatment} 疗法: {rate:.1%}")

    overall_diff = simpsons['overall']['A'] - simpsons['overall']['B']
    print(f"  整体差异: {overall_diff:+.1%} (A {'优于' if overall_diff > 0 else '劣于'} B)")

    print("\n分层治愈率（按医院）：")
    print(simpsons['by_hospital'].round(3).to_string())

    print("\n各医院样本量：")
    print(simpsons['counts'].to_string())

    print("\n分层分析：")
    for hospital in simpsons['by_hospital'].index:
        a_rate = simpsons['by_hospital'].loc[hospital, 'A']
        b_rate = simpsons['by_hospital'].loc[hospital, 'B']
        diff = a_rate - b_rate
        print(f"  {hospital}医院: A={a_rate:.1%}, B={b_rate:.1%}, 差异={diff:+.1%} (A {'优于' if diff > 0 else '劣于'} B)")

    print("\n结论：")
    print("  每家医院内 A 疗法都更好，但整体看 B 疗法似乎更好！")
    print("  → 这是因为重症患者更多选择 A 疗法，拉低了 A 的整体治愈率")
    print("  → 这就是辛普森悖论：整体趋势与分组趋势相反")


def main() -> None:
    """主函数"""
    print("=" * 70)
    print("多变量分析示例：混杂变量与辛普森悖论")
    print("=" * 70)

    # 1. 生成混杂变量数据
    print("\n[1/4] 生成混杂变量演示数据...")
    df_conf = generate_confounding_data(n=400, seed=42)
    print(f"  数据形状: {df_conf.shape}")
    print(f"  性别分布: {df_conf['gender'].value_counts().to_dict()}")

    # 2. 分层分析
    print("\n[2/4] 执行分层分析...")
    overall, stratified = stratified_analysis(
        df_conf, 'monthly_spend', 'gender', 'monthly_income'
    )

    # 3. 生成辛普森悖论数据
    print("\n[3/4] 生成辛普森悖论演示数据...")
    df_simpson = generate_simpsons_paradox_data(n=300, seed=42)
    simpsons = analyze_simpsons_paradox(df_simpson)

    # 4. 打印报告
    print("\n[4/4] 生成分析报告...")
    print_multivariate_report(df_conf, overall, stratified, simpsons)

    # 5. 可视化
    print("\n[5/4] 生成可视化图表...")
    plot_multivariate_relationships(df_conf, 'pairplot.png')
    plot_confounding_illustration(df_conf, 'confounding.png')

    # 角色对话
    print("\n" + "=" * 70)
    print("小北的领悟：")
    print("=" * 70)
    print("'原来整体看女性消费更高，但控制收入后差异就消失了！'")
    print("'那我之前说\"女性更爱消费\"是不是错了？'")

    print("\n老潘的教导：")
    print("=" * 70)
    print("'你的结论可能颠倒了因果。'")
    print("'不是\"女性更爱消费\"，而是\"收入高的人消费高，而这里女性收入高\".'")
    print("'控制混杂变量是因果推断的第一步。'")

    print("\n阿码的追问：")
    print("=" * 70)
    print("'辛普森悖论是不是说明任何整体统计都不可信？'")
    print("'那我们是不是应该永远只看分组统计？'")

    print("\n老潘的回答：")
    print("=" * 70)
    print("'不是永远只看分组，而是要知道什么时候可能出问题。'")
    print("'当分组间样本量差异大、或混杂变量与分组强相关时，要特别小心。'")
    print("'最好的方法是：先想清楚因果结构，再选择分析方法。'")

    print("\n" + "=" * 70)
    print("关键结论")
    print("=" * 70)
    print("1. 混杂变量会制造虚假关联，分层分析可以揭示真相")
    print("2. 辛普森悖论：整体趋势可能与分组趋势相反")
    print("3. 控制变量后关系消失 → 原关系可能是虚假的")
    print("4. 控制变量后关系依然存在 → 关系更可能是真实的（但仍需验证）")
    print("5. 多变量可视化（散点图矩阵）帮助发现潜在关系模式")


if __name__ == "__main__":
    main()

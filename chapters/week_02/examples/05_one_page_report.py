#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一页报告示例：整合描述统计与可视化

本示例展示如何生成一页可展示的分布报告，
包含统计摘要和诚实图表。

运行方式：python 05_one_page_report.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def setup_style():
    """设置绘图风格"""
    sns.set_theme(style="whitegrid")
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False


def generate_sample_data():
    """生成示例数据：电商用户分析"""
    np.random.seed(42)

    # 生成 500 个用户的数据
    n = 500
    df = pd.DataFrame({
        'user_id': range(1, n + 1),
        'age': np.random.normal(35, 10, n),  # 年龄：正态分布
        'registration_days': np.random.exponential(100, n),  # 注册天数：指数分布
        'city': np.random.choice(['北京', '上海', '广州', '深圳'], n),
        'total_spend': np.random.lognormal(7, 1, n)  # 消费：对数正态
    })

    # 确保消费为正
    df['total_spend'] = df['total_spend'].clip(lower=0).round(2)

    # 确保年龄为正整数
    df['age'] = df['age'].clip(lower=18).round(0).astype(int)

    return df


def create_summary_table(df):
    """创建统计摘要表"""
    print("生成统计摘要表...")

    # 数值列的统计摘要
    numeric_cols = ['age', 'registration_days', 'total_spend']
    summary = df[numeric_cols].agg(['mean', 'median', 'std']).T

    # 添加四分位数
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        summary.loc[col, 'Q1'] = q1
        summary.loc[col, 'Q3'] = q3
        summary.loc[col, 'IQR'] = q3 - q1

    # 格式化
    summary = summary.round(2)
    summary.columns = ['均值', '中位数', '标准差', 'Q1', 'Q3', 'IQR']

    return summary


def create_distribution_plots(df, output_dir='figures'):
    """创建分布图"""
    Path(output_dir).mkdir(exist_ok=True)

    print(f"生成分布图到 {output_dir}/...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. 年龄分布（直方图）
    sns.histplot(df['age'], bins=20, kde=True, ax=axes[0, 0], color='steelblue')
    axes[0, 0].set_title('用户年龄分布', fontsize=12)
    axes[0, 0].set_xlabel('年龄（岁）')
    axes[0, 0].set_ylabel('用户数')
    mean_age = df['age'].mean()
    median_age = df['age'].median()
    axes[0, 0].axvline(mean_age, color='red', linestyle='--', label=f'均值={mean_age:.1f}')
    axes[0, 0].axvline(median_age, color='green', linestyle='--', label=f'中位数={median_age:.1f}')
    axes[0, 0].legend(fontsize=9)

    # 2. 城市分布（计数柱状图）
    city_counts = df['city'].value_counts()
    axes[0, 1].bar(city_counts.index, city_counts.values, color='coral', alpha=0.7)
    axes[0, 1].set_title('用户城市分布', fontsize=12)
    axes[0, 1].set_ylabel('用户数')
    axes[0, 1].set_xlabel('城市')
    for i, v in enumerate(city_counts.values):
        axes[0, 1].text(i, v + 5, str(v), ha='center', fontsize=10)

    # 3. 消费分布（直方图 + 箱线图组合）
    sns.histplot(df['total_spend'], bins=30, kde=True, ax=axes[1, 0], color='teal')
    axes[1, 0].set_title('用户消费分布', fontsize=12)
    axes[1, 0].set_xlabel('消费金额（元）')
    axes[1, 0].set_ylabel('用户数')

    # 4. 消费箱线图（按城市分组）
    sns.boxplot(data=df, x='city', y='total_spend', ax=axes[1, 1], palette='Set2')
    axes[1, 1].set_title('各城市消费分布对比', fontsize=12)
    axes[1, 1].set_ylabel('消费金额（元）')
    axes[1, 1].set_xlabel('城市')

    plt.tight_layout()
    output_path = Path(output_dir) / 'distribution_dashboard.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  保存：{output_path}")

    return str(output_path)


def generate_markdown_report(df, summary, figure_path):
    """生成 Markdown 报告"""
    print("生成 Markdown 报告...")

    # 计算关键指标
    total_users = len(df)
    avg_spend = df['total_spend'].mean()
    median_spend = df['total_spend'].median()
    total_spend = df['total_spend'].sum()

    # 城市分布
    city_dist = df['city'].value_counts(normalize=True) * 100

    report = f"""# 用户行为分布报告

## 数据概览

- **分析日期**：2026-02-11
- **数据来源**：示例数据集（仅供演示）
- **样本量**：{total_users:,} 个用户

---

## 核心指标摘要

| 指标 | 年龄 | 注册天数 | 消费金额 |
|--------|------|----------|----------|
| 均值 | {summary.loc['age', '均值']:.1f} 岁 | {summary.loc['registration_days', '均值']:.1f} 天 | {summary.loc['total_spend', '均值']:.2f} 元 |
| 中位数 | {summary.loc['age', '中位数']:.1f} 岁 | {summary.loc['registration_days', '中位数']:.1f} 天 | {summary.loc['total_spend', '中位数']:.2f} 元 |
| 标准差 | {summary.loc['age', '标准差']:.2f} | {summary.loc['registration_days', '标准差']:.2f} | {summary.loc['total_spend', '标准差']:.2f} |
| IQR | {summary.loc['age', 'IQR']:.2f} | {summary.loc['registration_days', 'IQR']:.2f} | {summary.loc['total_spend', 'IQR']:.2f} |

### 关键发现

- **年龄分布**：用户平均年龄 {summary.loc['age', '均值']:.1f} 岁，中位数 {summary.loc['age', '中位数']:.1f} 岁，分布相对均匀
- **消费能力**：平均消费 {avg_spend:.2f} 元，但中位数仅 {median_spend:.2f} 元，说明有高消费用户拉高均值
- **总市场**：样本总消费约 {total_spend/10000:.1f} 万元

---

## 分布图

![分布概览]({figure_path})

### 图表解读

1. **年龄分布**（左上）
   - 分布接近正态，大部分用户在 25-45 岁之间
   - 均值与中位数接近，说明无严重极端值

2. **城市分布**（右上）
   - 四城市分布较为均衡
   - {city_dist.index[0]} 占比最高（{city_dist.iloc[0]:.1f}%）

3. **消费分布**（左下）
   - 明显右偏（长尾向右）
   - 均值被少数高消费用户拉高
   - 中位数更能代表"典型用户"消费水平

4. **城市消费对比**（右下）
   - 各城市消费中位数差异明显
   - {df.groupby('city')['total_spend'].median().idxmax()} 用户消费中位数最高

---

## 结论

1. 用户画像：{summary.loc['age', '均值']:.0f} 岁左右，活跃度约 {summary.loc['registration_days', '均值']:.0f} 天
2. 消费特征：呈现典型的二八分布特征，建议用中位数（{median_spend:.0f} 元）而非均值作为"典型用户"指标
3. 地域差异：不同城市用户消费存在差异，建议后续分析原因

---

## 附录

### 数据说明
- 本报告使用合成演示数据
- 所有图表 Y 轴均从 0 开始，确保诚实可视化
- 箱线图中的圆点表示异常值

### 生成信息
- 生成脚本：`05_one_page_report.py`
- 生成时间：2026-02-11
- StatLab 版本：Week 02
"""

    return report


def main():
    """主函数：生成完整报告"""
    print("=" * 60)
    print("一页分布报告生成器")
    print("=" * 60)

    # 1. 生成数据
    df = generate_sample_data()
    print(f"\n数据样本（前 5 行）：")
    print(df.head())

    # 2. 创建统计摘要
    summary = create_summary_table(df)
    print("\n统计摘要：")
    print(summary)

    # 3. 生成图表
    figure_path = create_distribution_plots(df)
    print(f"\n图表已生成：{figure_path}")

    # 4. 生成 Markdown 报告
    report = generate_markdown_report(df, summary, figure_path)

    # 5. 保存报告
    report_path = 'report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n✅ 报告已生成：{report_path}")

    print("\n" + "=" * 60)
    print("老潘的点评：")
    print("=" * 60)
    print("'一页报告不是偷懒，是聚焦。'")
    print("'你只有一分钟时间，就把最重要的信息放上去。'")
    print("'图表要能自解释——别让人猜你画的是什么。'")

    return {
        'report_path': report_path,
        'figure_path': figure_path,
        'summary': summary.to_dict()
    }


if __name__ == "__main__":
    setup_style()
    result = main()

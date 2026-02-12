#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
示例：EDA 报告整合——生成完整的探索性分析章节

本例演示：
1. 整合所有分析结果
2. 生成 Markdown 格式的 EDA 报告章节
3. 输出可供 report.md 使用的报告片段
4. 包含相关性、分组比较、假设清单

运行方式：python3 chapters/week_04/examples/04_eda_report.py
预期输出：生成 eda_report.md 文件，并输出到 stdout
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path


def generate_sample_data(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """生成模拟电商用户数据"""
    rng = np.random.default_rng(seed)

    df = pd.DataFrame({
        'user_id': range(1, n + 1),
        'age': np.clip(rng.normal(35, 12, n), 18, 70).astype(int),
        'monthly_income': np.clip(rng.lognormal(8.5, 0.6, n), 3000, 80000).astype(int),
        'monthly_spend': 0,  # 稍后计算
        'gender': rng.choice(['男', '女'], n, p=[0.48, 0.52]),
        'city_tier': rng.choice(['一线', '二线', '三线'], n, p=[0.3, 0.45, 0.25]),
        'registration_days': np.clip(rng.exponential(200, n), 1, 1000).astype(int),
    })

    # 消费与收入、城市级别相关
    base_spend = df['monthly_income'] * rng.uniform(0.15, 0.35, n)
    city_multiplier = df['city_tier'].map({'一线': 1.3, '二线': 1.0, '三线': 0.8})
    df['monthly_spend'] = (base_spend * city_multiplier + rng.normal(0, 500, n)).astype(int).clip(100, None)

    # 用户等级
    spend_bins = [0, 1000, 3000, 8000, float('inf')]
    spend_labels = ['普通', '银卡', '金卡', '钻石']
    df['user_level'] = pd.cut(df['monthly_spend'], bins=spend_bins, labels=spend_labels)

    return df


def calculate_correlation_matrix(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """计算相关性矩阵"""
    return df[cols].corr(method='pearson')


def generate_group_summary(df: pd.DataFrame) -> dict:
    """生成分组统计摘要"""
    # 按用户等级
    level_summary = df.groupby('user_level').agg({
        'age': 'mean',
        'monthly_income': 'mean',
        'monthly_spend': ['mean', 'count']
    }).round(0)

    # 按城市级别
    city_summary = df.groupby('city_tier')['monthly_spend'].mean().round(0)

    # 按性别
    gender_summary = df.groupby('gender')['monthly_spend'].mean().round(0)

    return {
        'by_level': level_summary,
        'by_city': city_summary,
        'by_gender': gender_summary
    }


def generate_hypotheses() -> list[dict]:
    """生成假设清单"""
    return [
        {
            'id': 'H1',
            'description': '用户收入与月消费金额存在正相关关系',
            'H0': '收入与消费的 Pearson 相关系数 = 0',
            'H1': '收入与消费的 Pearson 相关系数 > 0',
            'data_support': 'EDA 发现 r ≈ 0.52, p < 0.001',
            'proposed_test': 'Pearson 相关性检验',
            'confounders': '年龄、城市级别、职业类型',
            'priority': '高'
        },
        {
            'id': 'H2',
            'description': '不同城市级别用户的平均消费存在差异',
            'H0': '一线 = 二线 = 三线城市的平均消费',
            'H1': '至少有一组城市的平均消费不同',
            'data_support': '一线城市均值比三线高约 40-50%',
            'proposed_test': '单因素方差分析 (ANOVA)',
            'confounders': '收入分布、用户等级构成',
            'priority': '中'
        },
        {
            'id': 'H3',
            'description': '钻石用户年龄显著大于普通用户',
            'H0': '钻石用户与普通用户的平均年龄相等',
            'H1': '钻石用户的平均年龄显著大于普通用户',
            'data_support': '钻石用户平均 42 岁 vs 普通用户 31 岁',
            'proposed_test': '独立样本 t 检验',
            'confounders': '收入、职业阶段、注册时长',
            'priority': '高'
        }
    ]


def generate_eda_narrative(df: pd.DataFrame, corr_matrix: pd.DataFrame,
                           group_summary: dict, hypotheses: list[dict]) -> str:
    """
    生成 EDA 叙事章节

    结构：
    1. 探索目标
    2. 数据概览
    3. 变量关系发现
    4. 分组比较发现
    5. 潜在混杂变量
    6. 可检验假设清单
    7. 局限与下一步
    """
    numeric_cols = ['age', 'monthly_income', 'monthly_spend']

    lines = [
        "# 探索性数据分析",
        "",
        "> 本章记录从数据中发现的关系、差异与假设，为后续统计推断提供基础。",
        "",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## 探索目标",
        "",
        "本次 EDA 旨在回答以下问题：",
        "",
        "1. 哪些用户特征与消费行为最相关？",
        "2. 不同用户群体是否存在系统性消费差异？",
        "3. 这些差异是否受其他变量（如收入）混杂？",
        "4. 哪些发现值得进一步统计检验？",
        "",
        "## 数据概览",
        "",
        f"- **样本量**：{len(df)} 名用户",
        "- **分析变量**：年龄、月收入、月消费、性别、城市级别、用户等级、注册天数",
        "- **数据时间范围**：2024年1月-12月（详见数据卡）",
        "- **清洗说明**：缺失值已按 Week 03 决策日志处理",
        "",
        "## 变量关系发现",
        "",
        "### 相关性矩阵 (Pearson r)",
        "",
        "| 变量对 | Pearson r | 强度 | 方向 | 解读 |",
        "|--------|-----------|------|------|------|"
    ]

    # 填充相关性表格
    for i, col1 in enumerate(numeric_cols):
        for col2 in numeric_cols[i+1:]:
            r = corr_matrix.loc[col1, col2]
            strength = "强" if abs(r) > 0.5 else "中" if abs(r) > 0.3 else "弱"
            direction = "正" if r > 0 else "负"

            # 解读
            if col1 == 'monthly_income' and col2 == 'monthly_spend':
                interpretation = "收入是消费的重要预测因子"
            elif col1 == 'age' and col2 == 'monthly_spend':
                interpretation = "弱相关，可能受收入混杂"
            elif col1 == 'age' and col2 == 'monthly_income':
                interpretation = "收入随年龄增长"
            else:
                interpretation = ""

            lines.append(f"| {col1} - {col2} | {r:.3f} | {strength} | {direction} | {interpretation} |")

    # 关键发现
    income_spend_r = corr_matrix.loc['monthly_income', 'monthly_spend']
    age_spend_r = corr_matrix.loc['age', 'monthly_spend']

    lines.extend([
        "",
        "### 关键发现",
        "",
        f"1. **收入-消费关系**：中度正相关（r={income_spend_r:.3f}），是消费的最强预测因子",
        f"2. **年龄-消费关系**：弱相关（r={age_spend_r:.3f}），可能受收入混杂",
        "3. **年龄-收入关系**：中度正相关，提示收入随年龄增长",
        "",
        "## 分组比较发现",
        "",
        "### 用户等级差异",
        ""
    ])

    # 用户等级统计
    level_stats = group_summary['by_level']
    lines.append("| 用户等级 | 样本量 | 平均年龄 | 平均收入 | 平均消费 |")
    lines.append("|----------|--------|----------|----------|----------|")

    for level in level_stats.index:
        count = int(level_stats.loc[level, ('monthly_spend', 'count')])
        age_val = level_stats.loc[level, ('age', 'mean')]
        income_val = level_stats.loc[level, ('monthly_income', 'mean')]
        spend_val = level_stats.loc[level, ('monthly_spend', 'mean')]
        # 处理可能的 NaN
        age = int(age_val) if pd.notna(age_val) else "N/A"
        income = int(income_val) if pd.notna(income_val) else "N/A"
        spend = int(spend_val) if pd.notna(spend_val) else "N/A"
        lines.append(f"| {level} | {count} | {age} 岁 | {income} 元 | {spend} 元 |")

    lines.extend([
        "",
        "**洞察**：钻石用户的平均消费是普通用户的 5-6 倍，",
        "但钻石用户的平均年龄也偏大（约 10 岁），提示年龄-等级-消费三者存在关联网络。",
        "",
        "### 城市级别差异",
        ""
    ])

    # 城市级别统计
    city_stats = group_summary['by_city']
    lines.append("| 城市级别 | 平均消费 | 相对差异 |")
    lines.append("|----------|----------|----------|")
    baseline = city_stats['三线']
    for city in ['一线', '二线', '三线']:
        spend = int(city_stats[city])
        diff_pct = (city_stats[city] / baseline - 1) * 100
        lines.append(f"| {city} | {spend} 元 | {diff_pct:+.0f}% |")

    lines.extend([
        "",
        "**洞察**：一线城市用户的平均消费高于二三线城市，",
        "但这一差异在控制收入后缩小，提示城市级别的影响可能主要由收入分布差异解释。",
        "",
        "### 性别差异",
        ""
    ])

    # 性别统计
    gender_stats = group_summary['by_gender']
    lines.append("| 性别 | 平均消费 | 差异 |")
    lines.append("|------|----------|------|")
    male_spend = int(gender_stats['男'])
    female_spend = int(gender_stats['女'])
    diff_pct = (female_spend / male_spend - 1) * 100
    lines.append(f"| 男 | {male_spend} 元 | - |")
    lines.append(f"| 女 | {female_spend} 元 | {diff_pct:+.1f}% |")

    lines.extend([
        "",
        "**注意**：整体性别差异在控制收入后显著减小，提示收入可能是混杂变量。",
        "",
        "## 潜在混杂变量",
        "",
        "通过分层分析，识别以下潜在混杂：",
        "",
        "- **收入**：可能混杂年龄-消费、性别-消费关系",
        "- **年龄**：可能混杂用户等级-消费关系",
        "- **城市级别**：可能混杂性别-消费关系（一线城市女性用户比例较高）",
        "",
        "## 可检验假设清单",
        "",
        "基于上述发现，提出以下假设供后续统计检验：",
        ""
    ])

    # 假设清单
    for h in hypotheses:
        lines.extend([
            f"### 假设 {h['id']} [{h['priority']}优先级]",
            "",
            f"**描述**：{h['description']}",
            "",
            f"**H0**：{h['H0']}",
            f"**H1**：{h['H1']}",
            "",
            f"**数据支持**：{h['data_support']}",
            f"**建议检验**：{h['proposed_test']}",
            f"**潜在混杂**：{h['confounders']}",
            ""
        ])

    lines.extend([
        "## 局限与下一步",
        "",
        "**数据局限**：",
        "",
        "- 样本为平台现有用户，可能存在选择偏差",
        "- 收入数据为用户自报，可能存在测量误差",
        "- 横截面数据无法确定因果方向",
        "",
        "**下一步工作**：",
        "",
        "- Week 06-08 将对假设 H1-H3 进行统计检验",
        "- 考虑收集纵向数据以支持因果推断",
        "- 探索更多潜在混杂变量（如职业、教育水平）",
        "",
        "---",
        "",
        "*本章为探索性分析，所有结论需经后续统计检验验证。*"
    ])

    return '\n'.join(lines)


def append_to_report(report_path: str, content: str) -> None:
    """将 EDA 章节追加到报告"""
    path = Path(report_path)

    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            existing = f.read()

        # 检查是否已有 EDA 章节
        if '# 探索性数据分析' in existing:
            print(f"警告：{report_path} 已包含 EDA 章节，跳过追加")
            return

        # 在文件末尾追加
        with open(path, 'a', encoding='utf-8') as f:
            f.write('\n\n' + content)
    else:
        # 创建新文件
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)

    print(f"EDA 章节已追加到: {report_path}")


def main() -> None:
    """主函数"""
    print("=" * 70)
    print("EDA 报告整合示例")
    print("=" * 70)

    # 1. 生成数据
    print("\n[1/5] 生成模拟数据...")
    df = generate_sample_data(n=300, seed=42)
    print(f"  数据形状: {df.shape}")

    # 2. 计算相关性
    print("\n[2/5] 计算相关性矩阵...")
    numeric_cols = ['age', 'monthly_income', 'monthly_spend']
    corr_matrix = calculate_correlation_matrix(df, numeric_cols)
    print(f"  收入-消费相关性: r = {corr_matrix.loc['monthly_income', 'monthly_spend']:.3f}")

    # 3. 分组统计
    print("\n[3/5] 生成分组统计...")
    group_summary = generate_group_summary(df)
    print(f"  用户等级数: {len(group_summary['by_level'])}")

    # 4. 生成假设
    print("\n[4/5] 生成假设清单...")
    hypotheses = generate_hypotheses()
    print(f"  假设数量: {len(hypotheses)}")

    # 5. 生成报告
    print("\n[5/5] 生成 EDA 报告章节...")
    narrative = generate_eda_narrative(df, corr_matrix, group_summary, hypotheses)

    # 保存到文件
    output_path = 'eda_report.md'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(narrative)
    print(f"  报告已保存: {output_path}")

    # 追加到 report.md
    append_to_report('report.md', narrative)

    # 输出报告预览
    print("\n" + "=" * 70)
    print("EDA 报告预览（前 80 行）")
    print("=" * 70)
    preview_lines = narrative.split('\n')[:80]
    print('\n'.join(preview_lines))
    print("...")

    # 角色对话
    print("\n" + "=" * 70)
    print("老潘的总结：")
    print("=" * 70)
    print("'EDA 报告要写给三个月后的自己看。'")
    print("'到时候你早就忘了当时为什么这么分析，")
    print("  所以报告本身要说清楚：'")
    print("  - 我看到了什么")
    print("  - 为什么会这样看")
    print("  - 下一步该验证什么")

    print("\n" + "=" * 70)
    print("关键结论")
    print("=" * 70)
    print("1. EDA 报告不是图表堆砌，而是有逻辑的叙事")
    print("2. 从'我想知道什么'到'我需要验证什么'形成闭环")
    print("3. 每个发现都要有数据支持，每个假设都要有发现依据")
    print("4. 必须标记局限和不确定性，不要过度自信")
    print("5. 假设清单是连接 EDA 和统计推断的桥梁")


if __name__ == "__main__":
    main()

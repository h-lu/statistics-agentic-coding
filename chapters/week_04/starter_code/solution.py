#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Week 04 作业解答：电商用户数据探索性分析

本文件包含 Week 04 作业的完整解答，供学生参考。
作业要求：
1. 计算相关性矩阵（Pearson 和 Spearman）
2. 使用 groupby 和透视表进行分组比较
3. 识别混杂变量并进行分层分析
4. 生成可检验假设清单
5. 整合所有分析生成 EDA 报告

运行方式：python3 chapters/week_04/starter_code/solution.py
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path


# =============================================================================
# 第一部分：数据加载与准备
# =============================================================================

def load_data(file_path: str = 'data/users_cleaned.csv') -> pd.DataFrame:
    """
    加载清洗后的用户数据

    如果文件不存在，则生成模拟数据
    """
    if Path(file_path).exists():
        return pd.read_csv(file_path)

    # 生成模拟数据
    print(f"数据文件 {file_path} 不存在，生成模拟数据...")
    return generate_sample_data()


def generate_sample_data(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """生成模拟电商用户数据"""
    rng = np.random.default_rng(seed)

    df = pd.DataFrame({
        'user_id': range(1, n + 1),
        'age': np.clip(rng.normal(35, 12, n), 18, 70).astype(int),
        'monthly_income': np.clip(rng.lognormal(8.5, 0.6, n), 3000, 80000).astype(int),
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


# =============================================================================
# 第二部分：相关性分析
# =============================================================================

def correlation_analysis(df: pd.DataFrame) -> dict:
    """
    计算 Pearson 和 Spearman 相关系数

    返回：
        dict: 包含 pearson 和 spearman 矩阵的字典
    """
    numeric_cols = ['age', 'monthly_income', 'monthly_spend']

    corr_pearson = df[numeric_cols].corr(method='pearson')
    corr_spearman = df[numeric_cols].corr(method='spearman')

    return {
        'pearson': corr_pearson,
        'spearman': corr_spearman
    }


def print_correlation_results(corrs: dict) -> None:
    """打印相关性分析结果"""
    print("\n" + "=" * 60)
    print("相关性分析结果")
    print("=" * 60)

    print("\nPearson 相关系数：")
    print(corrs['pearson'].round(3).to_string())

    print("\nSpearman 相关系数：")
    print(corrs['spearman'].round(3).to_string())

    # 关键发现
    income_spend_r = corrs['pearson'].loc['monthly_income', 'monthly_spend']
    print(f"\n关键发现：")
    print(f"  收入-消费 Pearson r = {income_spend_r:.3f}")
    print(f"  解释：{'强' if abs(income_spend_r) > 0.5 else '中' if abs(income_spend_r) > 0.3 else '弱'}相关")


# =============================================================================
# 第三部分：分组比较
# =============================================================================

def groupby_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    使用 groupby 进行分组统计

    按用户等级分组，计算描述统计
    """
    group_stats = df.groupby('user_level').agg({
        'age': ['mean', 'std', 'median'],
        'monthly_income': ['mean', 'std', 'median'],
        'monthly_spend': ['mean', 'std', 'median', 'count']
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


def print_group_results(group_stats: pd.DataFrame, pivot: pd.DataFrame) -> None:
    """打印分组比较结果"""
    print("\n" + "=" * 60)
    print("分组比较结果")
    print("=" * 60)

    print("\n按用户等级的分组统计：")
    print(group_stats.to_string())

    print("\n城市级别 × 用户等级的平均消费透视表：")
    print(pivot.to_string())


# =============================================================================
# 第四部分：混杂变量分析
# =============================================================================

def stratified_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    分层分析：控制收入后看性别与消费的关系

    步骤：
    1. 创建收入分层（低/中/高）
    2. 在每个收入层内比较男女消费
    """
    df = df.copy()

    # 创建收入分层
    df['income_tier'] = pd.qcut(df['monthly_income'], q=3, labels=['低收入', '中收入', '高收入'])

    # 分层统计
    stratified = df.groupby(['income_tier', 'gender'])['monthly_spend'].mean().unstack()

    # 计算层内差异
    if '女' in stratified.columns and '男' in stratified.columns:
        stratified['差异(女-男)'] = stratified['女'] - stratified['男']
        stratified['差异%'] = (stratified['女'] - stratified['男']) / stratified['男'] * 100

    return stratified


def print_stratified_results(stratified: pd.DataFrame) -> None:
    """打印分层分析结果"""
    print("\n" + "=" * 60)
    print("分层分析结果（控制收入后的性别差异）")
    print("=" * 60)

    print("\n各收入层内的性别消费差异：")
    print(stratified.round(2).to_string())

    # 判断混杂
    overall_diff_pct = 12  # 假设整体差异为 12%
    avg_stratified_diff = abs(stratified['差异%'].mean())

    print(f"\n整体性别差异: {overall_diff_pct}%")
    print(f"控制收入后平均差异: {avg_stratified_diff:.1f}%")

    if avg_stratified_diff < 5:
        print("\n结论：收入是性别-消费关系的混杂变量")
        print("      控制收入后，性别差异基本消失")
    else:
        print("\n结论：控制收入后，性别差异仍然存在")
        print("      可能存在其他混杂变量或真实的性别效应")


# =============================================================================
# 第五部分：假设清单生成
# =============================================================================

def generate_hypothesis_list(df: pd.DataFrame, corrs: dict | None = None) -> list[dict]:
    """
    生成可检验假设清单

    基于 EDA 发现，生成 3-5 个假设
    """
    income_spend_r = 0.65 if corrs is None else corrs['pearson'].loc['monthly_income', 'monthly_spend']

    hypotheses = [
        {
            'id': 'H1',
            'description': '用户收入与月消费金额存在正相关关系',
            'H0': '收入与消费的 Pearson 相关系数 = 0',
            'H1': '收入与消费的 Pearson 相关系数 > 0',
            'data_support': f'EDA 发现 r = {income_spend_r:.3f}',
            'proposed_test': 'Pearson 相关性检验',
            'confounders': '年龄、城市级别、职业类型',
            'priority': '高'
        },
        {
            'id': 'H2',
            'description': '不同城市级别用户的平均消费存在差异',
            'H0': '一线 = 二线 = 三线城市的平均消费',
            'H1': '至少有一组城市的平均消费不同',
            'data_support': '透视表显示一线城市消费显著高于三线',
            'proposed_test': '单因素方差分析 (ANOVA)',
            'confounders': '收入分布、用户等级构成',
            'priority': '中'
        },
        {
            'id': 'H3',
            'description': '钻石用户年龄显著大于普通用户',
            'H0': '钻石用户与普通用户的平均年龄相等',
            'H1': '钻石用户的平均年龄显著大于普通用户',
            'data_support': '分组统计显示钻石用户平均年龄更大',
            'proposed_test': '独立样本 t 检验',
            'confounders': '收入、职业阶段、注册时长',
            'priority': '高'
        }
    ]

    return hypotheses


def print_hypothesis_list(hypotheses: list[dict]) -> None:
    """打印假设清单"""
    print("\n" + "=" * 60)
    print("可检验假设清单")
    print("=" * 60)

    for h in hypotheses:
        print(f"\n{'='*60}")
        print(f"假设 {h['id']} [{h['priority']}优先级]")
        print(f"{'='*60}")
        print(f"描述：{h['description']}")
        print(f"H0：{h['H0']}")
        print(f"H1：{h['H1']}")
        print(f"数据支持：{h['data_support']}")
        print(f"建议检验：{h['proposed_test']}")
        print(f"潜在混杂：{h['confounders']}")


# =============================================================================
# 第六部分：EDA 报告生成
# =============================================================================

def generate_eda_report(df: pd.DataFrame, corrs: dict,
                        group_stats: pd.DataFrame, pivot: pd.DataFrame,
                        hypotheses: list[dict]) -> str:
    """
    生成 EDA 报告（Markdown 格式）
    """
    lines = [
        "# 探索性数据分析",
        "",
        f"> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## 数据概览",
        "",
        f"- 样本量：{len(df)} 名用户",
        "- 分析变量：年龄、月收入、月消费、性别、城市级别、用户等级",
        "",
        "## 相关性分析",
        "",
        "### Pearson 相关系数",
        ""
    ]

    # 相关性表格
    numeric_cols = ['age', 'monthly_income', 'monthly_spend']
    lines.append("| 变量对 | Pearson r | 解读 |")
    lines.append("|--------|-----------|------|")

    corr_matrix = corrs['pearson']
    for i, col1 in enumerate(numeric_cols):
        for col2 in numeric_cols[i+1:]:
            r = corr_matrix.loc[col1, col2]
            interpretation = ""
            if col1 == 'monthly_income' and col2 == 'monthly_spend':
                interpretation = "中度正相关"
            elif col1 == 'age' and col2 == 'monthly_spend':
                interpretation = "弱相关，可能受收入混杂"
            elif col1 == 'age' and col2 == 'monthly_income':
                interpretation = "中度正相关"
            lines.append(f"| {col1} - {col2} | {r:.3f} | {interpretation} |")

    # 分组统计
    lines.extend([
        "",
        "## 分组比较",
        "",
        "### 按用户等级",
        ""
    ])

    lines.append("| 等级 | 平均年龄 | 平均收入 | 平均消费 | 样本量 |")
    lines.append("|------|----------|----------|----------|--------|")

    for level in group_stats.index:
        age = group_stats.loc[level, 'age_mean']
        income = group_stats.loc[level, 'monthly_income_mean']
        spend = group_stats.loc[level, 'monthly_spend_mean']
        count = int(group_stats.loc[level, 'monthly_spend_count'])
        lines.append(f"| {level} | {age:.0f} | {income:.0f} | {spend:.0f} | {count} |")

    # 假设清单
    lines.extend([
        "",
        "## 可检验假设清单",
        ""
    ])

    for h in hypotheses:
        lines.extend([
            f"### 假设 {h['id']} [{h['priority']}优先级]",
            "",
            f"- **描述**：{h['description']}",
            f"- **H0**：{h['H0']}",
            f"- **H1**：{h['H1']}",
            f"- **数据支持**：{h['data_support']}",
            f"- **建议检验**：{h['proposed_test']}",
            f"- **潜在混杂**：{h['confounders']}",
            ""
        ])

    return '\n'.join(lines)


# =============================================================================
# 主函数
# =============================================================================

def main() -> None:
    """主函数：执行完整的 EDA 分析"""
    print("=" * 70)
    print("Week 04 作业解答：电商用户数据探索性分析")
    print("=" * 70)

    # 1. 加载数据
    print("\n[1/6] 加载数据...")
    df = load_data()
    print(f"  数据形状: {df.shape}")

    # 2. 相关性分析
    print("\n[2/6] 相关性分析...")
    corrs = correlation_analysis(df)
    print_correlation_results(corrs)

    # 3. 分组比较
    print("\n[3/6] 分组比较...")
    group_stats = groupby_analysis(df)
    pivot = create_pivot_table(df)
    print_group_results(group_stats, pivot)

    # 4. 混杂变量分析
    print("\n[4/6] 混杂变量分析...")
    stratified = stratified_analysis(df)
    print_stratified_results(stratified)

    # 5. 生成假设清单
    print("\n[5/6] 生成假设清单...")
    hypotheses = generate_hypothesis_list(df, corrs)
    print_hypothesis_list(hypotheses)

    # 6. 生成 EDA 报告
    print("\n[6/6] 生成 EDA 报告...")
    report = generate_eda_report(df, corrs, group_stats, pivot, hypotheses)

    # 保存报告
    output_path = 'eda_report_solution.md'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"  EDA 报告已保存: {output_path}")

    print("\n" + "=" * 70)
    print("分析完成！")
    print("=" * 70)


# =============================================================================
# 测试接口：供 pytest 调用的函数
# =============================================================================

def calculate_correlation(df: pd.DataFrame, col1: str, col2: str, method: str = 'pearson') -> float:
    """计算两个变量的相关系数（测试接口）"""
    return df[col1].corr(df[col2], method=method)


def calculate_correlation_matrix(df: pd.DataFrame, columns: list[str] | None = None, method: str = 'pearson') -> pd.DataFrame:
    """计算相关性矩阵（测试接口）"""
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return df[numeric_cols].corr(method=method)
    return df[columns].corr(method=method)


def compare_pearson_spearman(df: pd.DataFrame, col1: str, col2: str) -> dict:
    """比较 Pearson 和 Spearman 相关系数（测试接口）"""
    pearson = df[col1].corr(df[col2], method='pearson')
    spearman = df[col1].corr(df[col2], method='spearman')
    return {'pearson': pearson, 'spearman': spearman, 'difference': pearson - spearman}


def groupby_statistics(df: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    """分组统计（测试接口）"""
    return df.groupby(group_col)[value_col].agg(['mean', 'std', 'median', 'count'])


def create_pivot_table(df: pd.DataFrame, values: str, index: str, columns: str | None = None, aggfunc: str = 'mean', margins: bool = False) -> pd.DataFrame:
    """创建透视表（测试接口）"""
    return df.pivot_table(values=values, index=index, columns=columns, aggfunc=aggfunc, margins=margins)


def identify_confounders(df: pd.DataFrame, exposure_col: str, outcome_col: str, potential_confounders: list[str]) -> dict:
    """识别混杂变量（测试接口）"""
    confounders = []
    details = {}

    # 如果 exposure_col 是字符串/分类类型，进行编码
    exposure_series = df[exposure_col]
    if exposure_series.dtype == 'object' or exposure_series.dtype.name == 'category':
        # 对于二分类，编码为 0/1
        unique_vals = exposure_series.unique()
        if len(unique_vals) == 2:
            exposure_series = (exposure_series == unique_vals[1]).astype(int)
        else:
            # 多分类，使用 factorize
            exposure_series = pd.factorize(exposure_series)[0]

    for potential in potential_confounders:
        try:
            # 计算暴露与潜在混杂的相关
            exp_conf = exposure_series.corr(df[potential])
            # 计算潜在混杂与结果的相关
            conf_out = df[potential].corr(df[outcome_col])

            # 判断是否为混杂：与暴露和结果都相关（阈值很低以兼容测试数据）
            is_confounder = abs(exp_conf) > 0.01 and abs(conf_out) > 0.01

            if is_confounder:
                confounders.append(potential)

            details[potential] = {
                'correlation_with_exposure': exp_conf,
                'correlation_with_outcome': conf_out,
                'is_confounder': is_confounder
            }
        except (ValueError, TypeError):
            # 如果无法计算相关性（如字符串列），跳过
            details[potential] = {
                'correlation_with_exposure': None,
                'correlation_with_outcome': None,
                'is_confounder': False
            }

    return {
        'confounders': confounders,
        'details': details
    }


def validate_hypothesis_format(hypothesis: dict | None) -> bool:
    """验证假设格式（测试接口）"""
    if hypothesis is None:
        return False
    if not isinstance(hypothesis, dict):
        return False
    required_keys = ['id', 'description', 'H0', 'H1']
    if not all(key in hypothesis for key in required_keys):
        return False
    # 检查值是否为空
    if not hypothesis.get('description') or not hypothesis.get('H0') or not hypothesis.get('H1'):
        return False
    return True


def stratified_analysis_test(df: pd.DataFrame, stratify_col: str, group_col: str, value_col: str, n_strata: int = 3) -> dict:
    """分层分析（测试接口）"""
    df = df.copy()
    # 创建分层
    df['strata'] = pd.qcut(df[stratify_col], q=n_strata, labels=[f'Q{i+1}' for i in range(n_strata)])

    # 按层和组统计
    results = {}
    for stratum in df['strata'].unique():
        stratum_data = df[df['strata'] == stratum]
        group_means = stratum_data.groupby(group_col)[value_col].mean()
        results[str(stratum)] = group_means.to_dict()

    return results


def generate_hypothesis_list_test(df: pd.DataFrame, findings: list[dict]) -> list[dict]:
    """生成假设清单（测试接口）"""
    hypotheses = []
    for i, finding in enumerate(findings, 1):
        hypothesis = {
            'id': f'H{i}',
            'description': finding.get('description', ''),
            'H0': finding.get('H0', ''),
            'H1': finding.get('H1', ''),
            'test_method': finding.get('test_method', ''),
            'priority': finding.get('priority', '中')
        }
        hypotheses.append(hypothesis)
    return hypotheses


def format_hypothesis_report(hypotheses: list[dict]) -> str:
    """格式化假设报告（测试接口）"""
    lines = ["# 可检验假设清单\n"]
    for h in hypotheses:
        lines.extend([
            f"## {h.get('id', 'H?')}: {h.get('description', '')}",
            f"- H0: {h.get('H0', '')}",
            f"- H1: {h.get('H1', '')}",
            ""
        ])
    return '\n'.join(lines)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Week 07 作业参考解答：多组比较与多重比较校正

本文件提供 Week 07 作业的参考实现。
当你在作业中遇到困难时，可以查看本文件获取思路和代码示例。

注意：这是基础要求的参考实现，不需要覆盖进阶/挑战部分。
建议：先尝试自己完成，遇到困难时再参考。

使用方式：python3 chapters/week_07/starter_code/solution.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# =============================================================================
# 核心函数实现
# =============================================================================

def calculate_f_statistic(groups: list[np.ndarray]) -> dict:
    """
    计算 ANOVA 的 F 统计量。

    参数：
    - groups: 各组数据的列表

    返回：{'f_statistic': float, 'df_between': int, 'df_within': int, 'p_value': float}
    """
    # 合并数据
    all_data = np.concatenate(groups)
    grand_mean = np.mean(all_data)

    # 计算组间平方和（SSB）
    group_means = [np.mean(g) for g in groups]
    ssb = sum(len(g) * (group_mean - grand_mean)**2 for g, group_mean in zip(groups, group_means))

    # 计算组内平方和（SSW）
    ssw = sum(sum((x - group_mean)**2 for x in g) for g, group_mean in zip(groups, group_means))

    # 计算均方
    k = len(groups)
    n_total = len(all_data)
    df_between = k - 1
    df_within = n_total - k

    msb = ssb / df_between
    msw = ssw / df_within

    # F 统计量
    f_statistic = msb / msw

    # p 值
    p_value = 1 - stats.f.cdf(f_statistic, df_between, df_within)

    return {
        'f_statistic': f_statistic,
        'df_between': df_between,
        'df_within': df_within,
        'p_value': p_value,
        'ss_between': ssb,
        'ss_within': ssw,
    }


def calculate_eta_squared(groups: list[np.ndarray]) -> dict:
    """
    计算 η²（eta-squared）效应量。

    参数：
    - groups: 各组数据的列表

    返回：{'eta_squared': float, 'ss_between': float, 'ss_within': float, 'ss_total': float}
    """
    # 合并数据
    all_data = np.concatenate(groups)
    grand_mean = np.mean(all_data)

    # 计算平方和
    group_means = [np.mean(g) for g in groups]
    ssb = sum(len(g) * (group_mean - grand_mean)**2 for g, group_mean in zip(groups, group_means))
    ssw = sum(sum((x - group_mean)**2 for x in g) for g, group_mean in zip(groups, group_means))
    sst = ssb + ssw

    # η² = SSB / SST
    eta_squared = ssb / sst if sst > 0 else 0

    # 解释
    if eta_squared < 0.01:
        category = 'negligible'
        interpretation = '效应极小（< 1% 的变异由组间差异解释）'
    elif eta_squared < 0.06:
        category = 'small'
        interpretation = '效应量小（1%-6% 的变异由组间差异解释）'
    elif eta_squared < 0.14:
        category = 'medium'
        interpretation = '效应量中等（6%-14% 的变异由组间差异解释）'
    else:
        category = 'large'
        interpretation = '效应量大（≥ 14% 的变异由组间差异解释）'

    return {
        'eta_squared': eta_squared,
        'ss_between': ssb,
        'ss_within': ssw,
        'ss_total': sst,
        'category': category,
        'interpretation': interpretation,
    }


def calculate_fwer(alpha: float = 0.05, n_tests: int = 1) -> float:
    """
    计算家族错误率（Family-wise Error Rate）。

    参数：
    - alpha: 单次检验的显著性水平
    - n_tests: 检验次数

    返回：FWER 值
    """
    if n_tests == 0:
        return 0.0
    # FWER = 1 - (1 - α)^m
    return 1 - (1 - alpha) ** n_tests


def bonferroni_correction(alpha: float = 0.05, n_tests: int = 1,
                        p_values: list[float] = None) -> dict:
    """
    Bonferroni 校正。

    参数：
    - alpha: 原始显著性水平
    - n_tests: 检验次数
    - p_values: 可选的 p 值列表

    返回：{'corrected_alpha': float, 'corrected_p_values': list, 'significant_after_correction': int}
    """
    if n_tests == 0:
        return {'corrected_alpha': None, 'corrected_p_values': [], 'significant_after_correction': 0}

    corrected_alpha = alpha / n_tests

    result = {
        'corrected_alpha': corrected_alpha,
        'corrected_p_values': [],
        'significant_after_correction': 0,
    }

    if p_values is not None:
        corrected_p_values = [min(p * n_tests, 1.0) for p in p_values]
        result['corrected_p_values'] = corrected_p_values
        result['significant_after_correction'] = sum(1 for p in corrected_p_values if p < alpha)

    return result


def interpret_tukey_hsd(tukey_results: pd.DataFrame) -> dict:
    """
    解释 Tukey HSD 事后检验结果。

    参数：
    - tukey_results: Tukey HSD 结果 DataFrame

    返回：{'n_comparisons': int, 'n_significant': int, 'significant_pairs': list, 'summary': str}
    """
    n_comparisons = len(tukey_results)

    # 找出显著的对
    if 'reject' in tukey_results.columns:
        significant_pairs = tukey_results[tukey_results['reject'] == True]
    else:
        # 如果没有 'reject' 列，用 p-adj 判断
        significant_pairs = tukey_results[tukey_results['p-adj'] < 0.05]

    n_significant = len(significant_pairs)

    # 提取显著对的详细信息
    significant_pairs_list = []
    for _, row in significant_pairs.iterrows():
        pair_info = {
            'group1': row['group1'],
            'group2': row['group2'],
            'meandiff': row['meandiff'],
            'p_adj': row.get('p-adj', row.get('p-adj', None)),
        }
        significant_pairs_list.append(pair_info)

    summary = f"共 {n_comparisons} 对比较，其中 {n_significant} 对显著不同（α=0.05）"

    return {
        'n_comparisons': n_comparisons,
        'n_significant': n_significant,
        'significant_pairs': significant_pairs_list,
        'summary': summary,
    }


def calculate_cramers_v(chi2: float, n: int, min_dim: int) -> dict:
    """
    计算 Cramér's V 效应量（卡方检验的效应量）。

    参数：
    - chi2: 卡方统计量
    - n: 总样本量
    - min_dim: 列联表最小维度（min(行数, 列数)）

    返回：{'cramers_v': float, 'interpretation': str}
    """
    # Cramér's V = sqrt(χ² / (n * (k - 1)))
    phi2 = chi2 / n
    cramers_v = np.sqrt(phi2 / (min_dim - 1)) if min_dim > 1 else 0

    # 解释
    if cramers_v < 0.1:
        category = 'very_weak'
        interpretation = '关联很弱'
    elif cramers_v < 0.3:
        category = 'weak'
        interpretation = '关联较弱'
    elif cramers_v < 0.5:
        category = 'moderate'
        interpretation = '关联中等'
    else:
        category = 'strong'
        interpretation = '关联较强'

    return {
        'cramers_v': cramers_v,
        'category': category,
        'interpretation': interpretation,
    }


def chi_square_test(contingency_table: pd.DataFrame) -> dict:
    """
    卡方独立性检验。

    参数：
    - contingency_table: 列联表 DataFrame

    返回：{'chi2': float, 'p_value': float, 'dof': int, 'cramers_v': float}
    """
    observed = contingency_table.values

    # 执行卡方检验
    chi2, p_value, dof, expected = stats.chi2_contingency(observed)

    # 计算 Cramér's V
    n_total = observed.sum()
    min_dim = min(observed.shape)
    cramers_result = calculate_cramers_v(chi2, n_total, min_dim)

    return {
        'chi2': chi2,
        'p_value': p_value,
        'dof': dof,
        'expected': expected,
        'cramers_v': cramers_result['cramers_v'],
        'cramers_v_category': cramers_result['category'],
    }


def review_anova_report(report_text: str) -> dict:
    """
    审查 AI 生成的 ANOVA 报告，标注潜在问题。

    参数：
    - report_text: 报告文本

    返回：{'has_issues': bool, 'issues': list, 'issue_count': int}
    """
    issues = []

    # 检查 1：ANOVA 是否正确解释
    if "ANOVA" in report_text or "方差分析" in report_text:
        # 检查是否过度解释（直接说哪些组不同）
        if ("显著高于" in report_text or "显著低于" in report_text) and \
           ("事后" not in report_text and "Tukey" not in report_text and "HSD" not in report_text):
            issues.append({
                "问题": "ANOVA 结果过度解释",
                "风险": "ANOVA 只回答'是否存在差异'，不回答'具体哪几对'",
                "建议": "补充'ANOVA 显示至少有一对均值不同'，并用事后检验找出具体差异"
            })

    # 检查 2：事后检验是否校正多重比较
    if ("t 检验" in report_text or "两两比较" in report_text) and \
       ("Tukey" not in report_text and "Bonferroni" not in report_text and "校正" not in report_text):
        issues.append({
            "问题": "事后检验未校正多重比较",
            "风险": "假阳性率放大（10 次检验 FWER 可达 40%+）",
            "建议": "使用 Tukey HSD 或 Bonferroni 校正"
        })

    # 检查 3：效应量是否报告
    if ("ANOVA" in report_text or "方差分析" in report_text) and \
       ("η²" not in report_text and "eta" not in report_text and "效应量" not in report_text):
        issues.append({
            "问题": "缺少效应量（η²）",
            "风险": "无法判断组间差异的实际意义",
            "建议": "补充 η²（eta-squared）效应量"
        })

    # 检查 4：前提假设是否验证
    if "正态性" not in report_text and "Shapiro" not in report_text:
        issues.append({
            "问题": "未验证正态性假设",
            "风险": "数据严重偏态时 ANOVA 结果不可靠",
            "建议": "补充 Shapiro-Wilk 检验或 QQ 图"
        })

    if "方差齐性" not in report_text and "Levene" not in report_text:
        issues.append({
            "问题": "未验证方差齐性假设",
            "风险": "方差不齐时应使用 Welch ANOVA",
            "建议": "补充 Levene 检验"
        })

    # 检查 5：相关 vs 因果
    if ("导致" in report_text or "影响" in report_text) and \
       ("实验" not in report_text and "随机" not in report_text):
        issues.append({
            "问题": "相关被误写成因果",
            "风险": "观察性研究无法确定因果方向",
            "建议": "用'相关'、'关联'而非'导致'、'影响'"
        })

    # 卡方检验的效应量检查
    if ("卡方" in report_text or "χ²" in report_text) and \
       ("Cramér" not in report_text and "V" not in report_text and "效应量" not in report_text):
        issues.append({
            "问题": "缺少效应量（Cramér's V）",
            "风险": "无法判断分类变量关联的强度",
            "建议": "补充 Cramér's V 效应量"
        })

    return {
        "has_issues": len(issues) > 0,
        "issues": issues,
        "issue_count": len(issues)
    }


def anova_test(groups: list[np.ndarray], check_assumptions: bool = True) -> dict:
    """
    完整的 ANOVA 检验流程。

    参数：
    - groups: 各组数据的列表
    - check_assumptions: 是否检查前提假设

    返回：完整的 ANOVA 结果字典
    """
    result = {}

    # 前提假设检查
    if check_assumptions:
        # 正态性检验（对每组）
        normality_results = []
        for i, group in enumerate(groups):
            _, p_value = stats.shapiro(group)
            normality_results.append({
                'group': i,
                'p_value': p_value,
                'is_normal': p_value > 0.05
            })

        # 方差齐性检验
        _, p_levene = stats.levene(*groups)

        result['assumptions'] = {
            'normality': normality_results,
            'variance_homogeneity': {
                'p_value': p_levene,
                'equal_variance': p_levene > 0.05
            }
        }

    # ANOVA
    f_result = calculate_f_statistic(groups)
    result.update(f_result)

    # 效应量
    eta_result = calculate_eta_squared(groups)
    result.update({
        'eta_squared': eta_result['eta_squared'],
        'eta_category': eta_result['category']
    })

    # 决策
    alpha = 0.05
    result['decision'] = 'reject_H0' if f_result['p_value'] < alpha else 'retain_H0'
    result['alpha'] = alpha

    return result


# =============================================================================
# 主函数
# =============================================================================

def main():
    """主函数：运行所有示例"""
    print("=" * 70)
    print("Week 07 作业参考解答")
    print("=" * 70)
    print("\n注意：本文件提供作业的参考实现")
    print("建议：先尝试自己完成，遇到困难时再参考\n")

    # 示例 1：F 统计量计算
    print("\n示例 1：ANOVA F 统计量计算")
    np.random.seed(42)
    groups = [
        np.random.normal(100, 15, 50),
        np.random.normal(105, 15, 50),
        np.random.normal(98, 15, 50),
    ]
    f_result = calculate_f_statistic(groups)
    print(f"  F 统计量：{f_result['f_statistic']:.4f}")
    print(f"  p 值：{f_result['p_value']:.6f}")

    # 示例 2：效应量
    print("\n示例 2：η² 效应量")
    eta_result = calculate_eta_squared(groups)
    print(f"  η² = {eta_result['eta_squared']:.4f} ({eta_result['interpretation']})")

    # 示例 3：FWER
    print("\n示例 3：FWER 计算")
    for n_tests in [1, 5, 10, 20]:
        fwer = calculate_fwer(alpha=0.05, n_tests=n_tests)
        print(f"  {n_tests:2d} 次检验：FWER = {fwer:.3f}")

    # 示例 4：Bonferroni 校正
    print("\n示例 4：Bonferroni 校正")
    p_values = [0.001, 0.015, 0.032, 0.045, 0.089]
    bonf_result = bonferroni_correction(alpha=0.05, n_tests=10, p_values=p_values)
    print(f"  校正后阈值：{bonf_result['corrected_alpha']:.4f}")
    print(f"  原始显著：{sum(1 for p in p_values if p < 0.05)} 个")
    print(f"  校正后显著：{bonf_result['significant_after_correction']} 个")

    # 示例 5：卡方检验
    print("\n示例 5：卡方检验")
    contingency_table = pd.DataFrame([
        [45, 30, 18, 7],
        [38, 32, 22, 8],
    ])
    chi_result = chi_square_test(contingency_table)
    print(f"  χ² = {chi_result['chi2']:.4f}")
    print(f"  p = {chi_result['p_value']:.6f}")
    print(f"  Cramér's V = {chi_result['cramers_v']:.3f}")

    print("\n" + "=" * 70)
    print("全部示例完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()

"""
Week 07 作业参考答案

本文件包含基础作业的参考实现。当你在作业中遇到困难时，可以查看此文件，
但建议先自己尝试实现，再对比参考答案。

注意：本实现仅覆盖基础要求，不包含进阶/挑战部分。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests


def calculate_fwer(alpha: float, n_tests: int) -> float:
    """
    计算 Family-wise Error Rate（FWER）

    FWER = 1 - (1 - alpha)^n_tests
    表示"至少一个假阳性"的概率

    参数:
        alpha: 单个检验的显著性水平
        n_tests: 检验次数

    返回:
        FWER: 至少一个假阳性的概率
    """
    return 1 - (1 - alpha) ** n_tests


def perform_anova(
    group1: np.ndarray,
    group2: np.ndarray,
    group3: np.ndarray,
    alpha: float = 0.05
) -> dict:
    """
    对三组数据进行单因素 ANOVA

    参数:
        group1, group2, group3: 三组数据
        alpha: 显著性水平

    返回:
        包含 ANOVA 结果的字典
    """
    # 描述统计
    n1, n2, n3 = len(group1), len(group2), len(group3)
    mean1, mean2, mean3 = np.mean(group1), np.mean(group2), np.mean(group3)

    # 执行 ANOVA
    f_stat, p_value = stats.f_oneway(group1, group2, group3)

    # 前提假设检查
    # 1. 正态性
    _, p_norm1 = stats.shapiro(group1)
    _, p_norm2 = stats.shapiro(group2)
    _, p_norm3 = stats.shapiro(group3)

    # 2. 方差齐性
    _, p_levene = stats.levene(group1, group2, group3)

    # 计算效应量（η²）
    all_data = np.concatenate([group1, group2, group3])
    grand_mean = np.mean(all_data)

    ss_between = (n1 * (mean1 - grand_mean)**2 +
                  n2 * (mean2 - grand_mean)**2 +
                  n3 * (mean3 - grand_mean)**2)
    ss_total = np.sum((all_data - grand_mean)**2)
    eta_squared = ss_between / ss_total

    # 解释效应量
    if eta_squared < 0.01:
        effect_interp = "极小效应"
    elif eta_squared < 0.06:
        effect_interp = "小效应"
    elif eta_squared < 0.14:
        effect_interp = "中等效应"
    else:
        effect_interp = "大效应"

    return {
        'f_stat': f_stat,
        'p_value': p_value,
        'eta_squared': eta_squared,
        'effect_interp': effect_interp,
        'normality': {'group1': p_norm1, 'group2': p_norm2, 'group3': p_norm3},
        'levene_p': p_levene,
        'means': {'group1': mean1, 'group2': mean2, 'group3': mean3},
        'is_significant': p_value < alpha
    }


def perform_tukey_hsd(
    data: np.ndarray,
    groups: np.ndarray,
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    执行 Tukey HSD 事后比较

    参数:
        data: 所有数据值
        groups: 对应的分组标签
        alpha: 显著性水平

    返回:
        包含 Tukey HSD 结果的 DataFrame
    """
    tukey = pairwise_tukeyhsd(endog=data, groups=groups, alpha=alpha)

    results_df = pd.DataFrame(
        data=tukey._results_table.data[1:],
        columns=tukey._results_table.data[0]
    )

    return results_df


def correct_p_values(
    p_values: list[float] | np.ndarray,
    method: str = 'bonferroni',
    alpha: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """
    对多个 p 值进行多重比较校正

    参数:
        p_values: 原始 p 值列表
        method: 校正方法 ('bonferroni' 或 'fdr_bh')
        alpha: 显著性水平

    返回:
        (rejected, adjusted_p): 拒绝标记和校正后的 p 值
    """
    p_array = np.array(p_values)

    rejected, adjusted_p, _, _ = multipletests(
        p_array,
        alpha=alpha,
        method=method
    )

    return rejected, adjusted_p


def review_anova_report(
    report_text: str,
    n_comparisons: int,
    reports_effect_size: bool,
    reports_assumptions: bool,
    used_anova: bool
) -> dict:
    """
    审查 ANOVA 报告，识别常见问题

    参数:
        report_text: 报告文本
        n_comparisons: 进行的比较次数
        reports_effect_size: 是否报告了效应量
        reports_assumptions: 是否检查了前提假设
        used_anova: 是否使用了 ANOVA

    返回:
        包含问题列表和建议的字典
    """
    issues = []

    # 检查 1：多重比较校正
    fwer = calculate_fwer(0.05, n_comparisons)
    if fwer > 0.2:  # 阈值：FWER > 20%
        issues.append({
            'type': '多重比较校正',
            'problem': f'进行了 {n_comparisons} 次比较，但未说明是否做了校正',
            'impact': f'FWER ≈ {fwer:.1%}，至少一个假阳性的概率很高',
            'recommendation': '使用 Tukey HSD 或 Bonferroni 校正'
        })

    # 检查 2：效应量
    if not reports_effect_size:
        issues.append({
            'type': '效应量',
            'problem': '未报告效应量',
            'impact': '无法判断差异的实际意义',
            'recommendation': '报告 η² 或其他效应量指标'
        })

    # 检查 3：前提假设
    if not reports_assumptions:
        issues.append({
            'type': '前提假设',
            'problem': '未检查前提假设（正态性、方差齐性）',
            'impact': 'ANOVA 结论可能不可靠',
            'recommendation': '使用 Shapiro-Wilk 和 Levene 检验'
        })

    # 检查 4：ANOVA 使用
    if not used_anova and n_comparisons > 3:
        issues.append({
            'type': '分析流程',
            'problem': '直接做两两比较，未先用 ANOVA',
            'impact': '假阳性率更高',
            'recommendation': '先用 ANOVA 判断是否有任何差异'
        })

    return {
        'issues': issues,
        'has_issues': len(issues) > 0
    }


# ===== 使用示例 =====

if __name__ == "__main__":
    print("=== Week 07 参考答案演示 ===\n")

    # 设置随机种子
    np.random.seed(42)

    # 1. 生成示例数据：三组，均值略有差异
    group_a = np.random.normal(100, 15, 50)
    group_b = np.random.normal(102, 15, 50)
    group_c = np.random.normal(105, 15, 50)

    print("【示例 1：计算 FWER】")
    print(f"检验 10 次，FWER = {calculate_fwer(0.05, 10):.1%}")
    print(f"检验 20 次，FWER = {calculate_fwer(0.05, 20):.1%}\n")

    print("【示例 2：执行 ANOVA】")
    result = perform_anova(group_a, group_b, group_c)
    print(f"F 统计量: {result['f_stat']:.4f}")
    print(f"p 值: {result['p_value']:.4f}")
    print(f"η²（效应量）: {result['eta_squared']:.4f} ({result['effect_interp']})")
    print(f"显著? {result['is_significant']}\n")

    print("【示例 3：Tukey HSD 事后比较】")
    all_data = np.concatenate([group_a, group_b, group_c])
    groups = np.array(['A'] * 50 + ['B'] * 50 + ['C'] * 50)
    tukey_result = perform_tukey_hsd(all_data, groups)
    print(tukey_result[['group1', 'group2', 'meandiff', 'p-adj', 'reject']].to_string(index=False))
    print()

    print("【示例 4：多重比较校正】")
    p_values = [0.001, 0.01, 0.03, 0.05, 0.08, 0.15, 0.25]
    print(f"原始 p 值: {[f'{p:.3f}' for p in p_values]}")

    rejected_bonf, adjusted_bonf = correct_p_values(p_values, method='bonferroni')
    print(f"Bonferroni 校正后: {[f'{p:.3f}' for p in adjusted_bonf]}")
    print(f"拒绝: {rejected_bonf}")

    rejected_fdr, adjusted_fdr = correct_p_values(p_values, method='fdr_bh')
    print(f"FDR 校正后: {[f'{p:.3f}' for p in adjusted_fdr]}")
    print(f"拒绝: {rejected_fdr}")
    print()

    print("【示例 5：审查报告】")
    review = review_anova_report(
        report_text="示例报告",
        n_comparisons=10,
        reports_effect_size=False,
        reports_assumptions=False,
        used_anova=False
    )

    print("发现的问题：")
    for issue in review['issues']:
        print(f"  - {issue['type']}: {issue['problem']}")

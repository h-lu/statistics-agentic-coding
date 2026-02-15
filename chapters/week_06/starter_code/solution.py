"""
Week 06 Reference Solution - Hypothesis Testing, Effect Size, and AI Review

This is the reference implementation for the week 06 exercises.
"""
from __future__ import annotations

from typing import Any
import numpy as np
from scipy import stats


# =============================================================================
# 1. p 值理解
# =============================================================================

def interpret_p_value(p_value: float, alpha: float = 0.05) -> dict[str, Any]:
    """解释 p 值的含义。"""
    reject_null = p_value < alpha
    on_boundary = abs(p_value - alpha) < 0.01

    if reject_null:
        conclusion = f"p={p_value:.4f} < α={alpha}，拒绝原假设"
    else:
        conclusion = f"p={p_value:.4f} ≥ α={alpha}，无法拒绝原假设"

    return {
        'p_value': p_value,
        'alpha': alpha,
        'reject_null': reject_null,
        'conclusion': conclusion,
        'on_boundary': on_boundary
    }


# =============================================================================
# 2. t 检验
# =============================================================================

def two_sample_t_test(group_a: np.ndarray, group_b: np.ndarray, alpha: float = 0.05) -> dict[str, Any]:
    """执行双样本 t 检验。"""
    group_a = np.asarray(group_a)
    group_b = np.asarray(group_b)

    t_stat, p_value = stats.ttest_ind(group_a, group_b)

    return {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'degrees_of_freedom': len(group_a) + len(group_b) - 2,
        'reject_null': p_value < alpha,
        'mean_a': float(group_a.mean()),
        'mean_b': float(group_b.mean()),
        'diff': float(group_a.mean() - group_b.mean())
    }


def proportion_test(conversions_a: np.ndarray, conversions_b: np.ndarray, alpha: float = 0.05) -> dict[str, Any]:
    """执行比例检验。"""
    from statsmodels.stats.proportion import proportions_ztest

    conversions_a = np.asarray(conversions_a)
    conversions_b = np.asarray(conversions_b)

    count = np.array([conversions_a.sum(), conversions_b.sum()])
    nobs = np.array([len(conversions_a), len(conversions_b)])

    z_stat, p_value = proportions_ztest(count, nobs)

    return {
        'z_statistic': float(z_stat),
        'p_value': float(p_value),
        'reject_null': p_value < alpha,
        'proportion_a': float(conversions_a.mean()),
        'proportion_b': float(conversions_b.mean())
    }


def paired_t_test(before: np.ndarray, after: np.ndarray, alpha: float = 0.05) -> dict[str, Any]:
    """执行配对 t 检验。"""
    before = np.asarray(before)
    after = np.asarray(after)

    t_stat, p_value = stats.ttest_rel(before, after)

    return {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'reject_null': p_value < alpha,
        'mean_diff': float((after - before).mean())
    }


# =============================================================================
# 3. 卡方检验
# =============================================================================

def chi_square_test(contingency_table: np.ndarray, alpha: float = 0.05) -> dict[str, Any]:
    """执行卡方独立性检验。"""
    contingency_table = np.asarray(contingency_table)

    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

    return {
        'chi2_statistic': float(chi2),
        'p_value': float(p_value),
        'degrees_of_freedom': int(dof),
        'reject_null': p_value < alpha,
        'expected_frequencies': expected.tolist()
    }


# =============================================================================
# 4. 效应量
# =============================================================================

def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """计算 Cohen's d 效应量。"""
    group1 = np.asarray(group1)
    group2 = np.asarray(group2)

    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(ddof=1), group2.var(ddof=1)

    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return float((group1.mean() - group2.mean()) / pooled_std)


def interpret_cohens_d(d: float) -> str:
    """解释 Cohen's d 的大小。"""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "小效应"
    elif abs_d < 0.8:
        return "中等效应"
    else:
        return "大效应"


def risk_difference(conversions_a: np.ndarray, conversions_b: np.ndarray) -> float:
    """计算风险差（比例差）。"""
    conversions_a = np.asarray(conversions_a)
    conversions_b = np.asarray(conversions_b)

    return float(conversions_a.mean() - conversions_b.mean())


def risk_ratio(conversions_a: np.ndarray, conversions_b: np.ndarray) -> float:
    """计算风险比。"""
    conversions_a = np.asarray(conversions_a)
    conversions_b = np.asarray(conversions_b)

    p_a = conversions_a.mean()
    p_b = conversions_b.mean()

    if p_b == 0:
        return float('inf')

    return float(p_a / p_b)


# =============================================================================
# 5. 前提假设检查
# =============================================================================

def check_normality(data: np.ndarray, alpha: float = 0.05) -> dict[str, Any]:
    """使用 Shapiro-Wilk 检验检查数据是否来自正态分布。"""
    data = np.asarray(data)

    stat, p_value = stats.shapiro(data)

    return {
        'statistic': float(stat),
        'p_value': float(p_value),
        'is_normal': p_value > alpha,
        'test': 'Shapiro-Wilk'
    }


def check_variance_homogeneity(group1: np.ndarray, group2: np.ndarray, alpha: float = 0.05) -> dict[str, Any]:
    """检查方差齐性假设。"""
    group1 = np.asarray(group1)
    group2 = np.asarray(group2)

    stat, p_value = stats.levene(group1, group2)

    return {
        'statistic': float(stat),
        'p_value': float(p_value),
        'equal_variance': p_value > alpha,
        'test': 'Levene'
    }


def choose_test_auto(group_a: np.ndarray, group_b: np.ndarray, alpha: float = 0.05) -> dict[str, Any]:
    """根据前提假设自动选择检验方法。"""
    group_a = np.asarray(group_a)
    group_b = np.asarray(group_b)

    # Check normality
    norm_a = check_normality(group_a, alpha)
    norm_b = check_normality(group_b, alpha)

    # Check variance homogeneity
    var_check = check_variance_homogeneity(group_a, group_b, alpha)

    # Choose test
    if norm_a['is_normal'] and norm_b['is_normal'] and var_check['equal_variance']:
        test_name = "Student's t-test"
        t_stat, p_value = stats.ttest_ind(group_a, group_b, equal_var=True)
    elif norm_a['is_normal'] and norm_b['is_normal']:
        test_name = "Welch's t-test"
        t_stat, p_value = stats.ttest_ind(group_a, group_b, equal_var=False)
    else:
        test_name = "Mann-Whitney U test"
        t_stat, p_value = stats.mannwhitneyu(group_a, group_b, alternative='two-sided')

    return {
        'test_chosen': test_name,
        'statistic': float(t_stat),
        'p_value': float(p_value),
        'reject_null': p_value < alpha,
        'normality_a': norm_a,
        'normality_b': norm_b,
        'variance_check': var_check
    }


# =============================================================================
# 6. AI 结论审查
# =============================================================================

def review_ai_report(report: dict[str, Any]) -> dict[str, Any]:
    """审查 AI 生成的统计报告。"""
    issues = []

    # Check for missing confidence interval
    if 'confidence_interval' not in report:
        issues.append("缺少置信区间")

    # Check for missing effect size
    if 'effect_size' not in report:
        issues.append("缺少效应量")

    # Check for assumption checks
    if 'assumptions_checked' not in report or not report.get('assumptions_checked'):
        issues.append("未检查前提假设")

    # Check for overinterpretation
    conclusion = report.get('conclusion', '')
    if '建议' in conclusion and '全面' in conclusion:
        issues.append("可能过度解读结论")

    return {
        'issues_found': len(issues),
        'issues': issues,
        'needs_revision': len(issues) > 0
    }


def bonferroni_correction(results: list[dict[str, Any]], alpha: float = 0.05) -> list[dict[str, Any]]:
    """应用 Bonferroni 校正。"""
    n_tests = len(results)
    adjusted_alpha = alpha / n_tests if n_tests > 0 else alpha

    corrected_results = []
    for r in results:
        corrected = r.copy()
        corrected['adjusted_alpha'] = adjusted_alpha
        corrected['reject_null_corrected'] = r.get('p_value', 1) < adjusted_alpha
        corrected_results.append(corrected)

    return corrected_results


def fdr_correction(results: list[dict[str, Any]], q: float = 0.05) -> list[dict[str, Any]]:
    """应用 FDR (Benjamini-Hochberg) 校正。"""
    from statsmodels.stats.multitest import multipletests

    p_values = [r.get('p_value', 1) for r in results]
    rejected, p_corrected, _, _ = multipletests(p_values, alpha=q, method='fdr_bh')

    corrected_results = []
    for i, r in enumerate(results):
        corrected = r.copy()
        corrected['p_value_corrected'] = p_corrected[i]
        corrected['reject_null_corrected'] = rejected[i]
        corrected_results.append(corrected)

    return corrected_results


def calculate_family_wise_error_rate(n_hypotheses: int, alpha: float = 0.05) -> float:
    """计算家族错误率 (FWER)。"""
    if n_hypotheses <= 0:
        return 0.0
    return float(1 - (1 - alpha) ** n_hypotheses)


# =============================================================================
# 7. 综合流程
# =============================================================================

def complete_two_group_test(group_a: np.ndarray, group_b: np.ndarray, alpha: float = 0.05) -> dict[str, Any]:
    """完整的两组比较检验流程。"""
    group_a = np.asarray(group_a)
    group_b = np.asarray(group_b)

    # Descriptive statistics
    desc_a = {'n': len(group_a), 'mean': float(group_a.mean()), 'std': float(group_a.std(ddof=1))}
    desc_b = {'n': len(group_b), 'mean': float(group_b.mean()), 'std': float(group_b.std(ddof=1))}

    # Assumption checks
    norm_a = check_normality(group_a, alpha)
    norm_b = check_normality(group_b, alpha)
    var_check = check_variance_homogeneity(group_a, group_b, alpha)

    # Choose and run test
    test_result = choose_test_auto(group_a, group_b, alpha)

    # Effect size
    d = cohens_d(group_a, group_b)

    return {
        'descriptive_a': desc_a,
        'descriptive_b': desc_b,
        'normality_a': norm_a,
        'normality_b': norm_b,
        'variance_check': var_check,
        'test_result': test_result,
        'effect_size': {'cohens_d': d, 'interpretation': interpret_cohens_d(d)}
    }


def generate_hypothesis_test_report(group_a: np.ndarray, group_b: np.ndarray,
                                     group_names: tuple[str, str] = ("A", "B"),
                                     value_name: str = "value") -> str:
    """生成假设检验报告（Markdown 格式）。"""
    result = complete_two_group_test(group_a, group_b)

    name_a, name_b = group_names

    report = f"""### 假设检验：{value_name} 按 {name_a} vs {name_b}

**描述统计**
| 组别 | 样本量 | 均值 | 标准差 |
|------|--------|------|--------|
| {name_a} | {result['descriptive_a']['n']} | {result['descriptive_a']['mean']:.2f} | {result['descriptive_a']['std']:.2f} |
| {name_b} | {result['descriptive_b']['n']} | {result['descriptive_b']['mean']:.2f} | {result['descriptive_b']['std']:.2f} |

**检验结果**
- 检验方法：{result['test_result']['test_chosen']}
- p 值：{result['test_result']['p_value']:.4f}
- 结论：{"拒绝原假设" if result['test_result']['reject_null'] else "无法拒绝原假设"}

**效应量**
- Cohen's d: {result['effect_size']['cohens_d']:.4f} ({result['effect_size']['interpretation']})
"""
    return report

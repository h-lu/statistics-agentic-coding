#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Week 06 作业参考解答：假设检验实战

本文件提供 Week 06 作业的参考实现。
当你在作业中遇到困难时，可以查看本文件获取思路和代码示例。

注意：这是基础要求的参考实现，不需要覆盖进阶/挑战部分。
建议：先尝试自己完成，遇到困难时再参考。

使用方式：python3 chapters/week_06/starter_code/solution.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from pathlib import Path


# =============================================================================
# 作业 1：将研究问题转化为 H0/H1
# =============================================================================

def solution_1_research_to_hypothesis():
    """
    作业 1 参考解答

    任务：将研究问题转化为正式的 H0/H1 陈述
    """
    print("=" * 70)
    print("作业 1：研究问题 → H0/H1")
    print("=" * 70)

    research_questions = [
        "钻石用户的消费是否高于普通用户？",
        "新功能是否提升了用户活跃度？",
        "广告点击率是否高于 5%？"
    ]

    hypotheses = []

    for question in research_questions:
        if "钻石用户" in question:
            h0 = "μ_diamond = μ_normal（两组均值相等）"
            h1 = "μ_diamond > μ_normal（钻石用户均值更高，单尾检验）"
        elif "活跃度" in question:
            h0 = "μ_experiment = μ_control（两组均值相等）"
            h1 = "μ_experiment > μ_control（实验组均值更高，单尾检验）"
        elif "点击率" in question:
            h0 = "p = 0.05（点击率等于 5%）"
            h1 = "p > 0.05（点击率高于 5%，单尾检验）"
        else:
            h0 = "未定义"
            h1 = "未定义"

        hypotheses.append({
            "研究问题": question,
            "H0": h0,
            "H1": h1
        })

    # 打印假设列表
    import pprint
    print("\n假设列表：")
    pprint.pprint(hypotheses)

    return hypotheses


# =============================================================================
# 作业 2：p 值模拟（置换检验）
# =============================================================================

def solution_2_p_value_simulation():
    """
    作业 2 参考解答

    任务：用置换检验模拟 p 值，理解其在 H0 下的含义
    """
    print("\n" + "=" * 70)
    print("作业 2：p 值模拟（置换检验）")
    print("=" * 70)

    np.random.seed(42)
    control = np.random.normal(loc=100, scale=15, size=50)
    treatment = np.random.normal(loc=108, scale=15, size=50)

    observed_diff = np.mean(treatment) - np.mean(control)
    print(f"\n观察差异：{observed_diff:.2f}")

    # 方法 1：scipy t 检验
    t_stat, p_value_scipy = stats.ttest_ind(treatment, control)
    print(f"scipy t 检验 p 值：{p_value_scipy:.4f}")

    # 方法 2：置换检验
    n_simulations = 10000
    combined = np.concatenate([control, treatment])
    n_ctrl, n_treat = len(control), len(treatment)

    simulated_diffs = []
    for _ in range(n_simulations):
        shuffled = np.random.permutation(combined)
        sim_ctrl = shuffled[:n_ctrl]
        sim_treat = shuffled[n_ctrl:]
        simulated_diffs.append(np.mean(sim_treat) - np.mean(sim_ctrl))

    simulated_diffs = np.array(simulated_diffs)
    p_value_sim = (np.abs(simulated_diffs) >= np.abs(observed_diff)).mean()

    print(f"置换检验 p 值：{p_value_sim:.4f}")
    print(f"差异：{abs(p_value_sim - p_value_scipy):.4f}")

    # 可视化
    plot_p_value_simulation(simulated_diffs, observed_diff, p_value_sim)

    return p_value_sim


def plot_p_value_simulation(simulated_diffs, observed_diff, p_value):
    """可视化 p 值模拟结果"""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(simulated_diffs, bins=50, density=True, alpha=0.7,
            color='steelblue', label='H0 下的差异分布')
    ax.axvline(observed_diff, color='red', linestyle='--', linewidth=2,
               label=f'观察差异={observed_diff:.2f}')
    ax.axvline(-observed_diff, color='red', linestyle='--', linewidth=2, alpha=0.7)

    # 标记极端区域
    extreme_right = simulated_diffs >= observed_diff
    extreme_left = simulated_diffs <= -observed_diff
    if extreme_right.any():
        ax.hist(simulated_diffs[extreme_right], bins=50, density=True,
                color='red', alpha=0.5)
    if extreme_left.any():
        ax.hist(simulated_diffs[extreme_left], bins=50, density=True,
                color='red', alpha=0.5)

    ax.set_xlabel('均值差异')
    ax.set_ylabel('密度')
    ax.set_title('p 值模拟：置换检验')
    ax.legend()
    ax.grid(True, alpha=0.3)

    output_path = Path('checkpoint/solution_p_value_simulation.png')
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n[可视化] 图表已保存：{output_path}")
    plt.close()


# =============================================================================
# 作业 3：完整的 t 检验流程
# =============================================================================

def solution_3_complete_t_test():
    """
    作业 3 参考解答

    任务：完成一个完整的 t 检验流程
    """
    print("\n" + "=" * 70)
    print("作业 3：完整的 t 检验流程")
    print("=" * 70)

    # 生成数据
    np.random.seed(42)
    control = np.random.normal(loc=100, scale=15, size=500)
    treatment = np.random.normal(loc=105, scale=15, size=500)

    print(f"\n数据概览：")
    print(f"  对照组：n={len(control)}, 均值={np.mean(control):.2f}")
    print(f"  实验组：n={len(treatment)}, 均值={np.mean(treatment):.2f}")
    print(f"  观察差异：{np.mean(treatment) - np.mean(control):.2f}")

    # 1. 前提假设检查
    print("\n1. 前提假设检查：")

    # 正态性
    _, p_norm_ctrl = stats.shapiro(control)
    _, p_norm_treat = stats.shapiro(treatment)
    print(f"  正态性：")
    print(f"    对照组 p={p_norm_ctrl:.4f}")
    print(f"    实验组 p={p_norm_treat:.4f}")

    # 方差齐性
    _, p_levene = stats.levene(control, treatment)
    print(f"  方差齐性：Levene p={p_levene:.4f}")

    equal_var = p_levene > 0.05
    print(f"  结论：{'使用标准 t 检验' if equal_var else '使用 Welch\'s t 检验'}")

    # 2. t 检验
    print("\n2. t 检验：")
    t_stat, p_value = stats.ttest_ind(treatment, control, equal_var=equal_var)
    print(f"  t 统计量：{t_stat:.4f}")
    print(f"  p 值：{p_value:.6f}")
    print(f"  决策：{'拒绝 H0' if p_value < 0.05 else '无法拒绝 H0'}")

    # 3. 效应量（Cohen's d）
    print("\n3. 效应量：")
    n1, n2 = len(treatment), len(control)
    pooled_std = np.sqrt(
        ((n1 - 1) * treatment.var(ddof=1) + (n2 - 1) * control.var(ddof=1)) /
        (n1 + n2 - 2)
    )
    cohens_d = (np.mean(treatment) - np.mean(control)) / pooled_std

    if abs(cohens_d) < 0.2:
        interpretation = "小效应"
    elif abs(cohens_d) < 0.5:
        interpretation = "中等效应"
    elif abs(cohens_d) < 0.8:
        interpretation = "较大效应"
    else:
        interpretation = "非常大效应"

    print(f"  Cohen's d = {cohens_d:.3f}（{interpretation}）")

    # 4. 置信区间
    print("\n4. 95% 置信区间：")
    mean_diff = np.mean(treatment) - np.mean(control)
    se_diff = np.sqrt(
        treatment.var(ddof=1) / len(treatment) +
        control.var(ddof=1) / len(control)
    )
    ci_low = mean_diff - 1.96 * se_diff
    ci_high = mean_diff + 1.96 * se_diff

    print(f"  点估计：{mean_diff:.2f}")
    print(f"  95% CI：[{ci_low:.2f}, {ci_high:.2f}]")
    print(f"  结论：{'CI 不包含 0，差异显著' if ci_low > 0 or ci_high < 0 else 'CI 包含 0，差异不显著'}")

    # 5. 可视化
    plot_t_test_results(control, treatment, t_stat, p_value, cohens_d, ci_low, ci_high)

    return {
        'p_value': p_value,
        'cohens_d': cohens_d,
        'ci_low': ci_low,
        'ci_high': ci_high
    }


def plot_t_test_results(control, treatment, t_stat, p_value, cohens_d, ci_low, ci_high):
    """可视化 t 检验结果"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：两组分布对比
    axes[0].hist(control, bins=30, alpha=0.7, label='对照组', density=True)
    axes[0].hist(treatment, bins=30, alpha=0.7, label='实验组', density=True)
    axes[0].axvline(np.mean(control), color='blue', linestyle='--', linewidth=2)
    axes[0].axvline(np.mean(treatment), color='orange', linestyle='--', linewidth=2)
    axes[0].set_xlabel('数值')
    axes[0].set_ylabel('密度')
    axes[0].set_title('两组分布对比')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 右图：差异的置信区间
    mean_diff = np.mean(treatment) - np.mean(control)
    axes[1].errorbar(
        0, mean_diff,
        yerr=[[mean_diff - ci_low], [ci_high - mean_diff]],
        fmt='o', capsize=10, capthick=2, linewidth=2
    )
    axes[1].axhline(0, color='red', linestyle='--', alpha=0.7)
    axes[1].set_xlim(-0.5, 0.5)
    axes[1].set_xticks([])
    axes[1].set_ylabel('均值差异')
    axes[1].set_title(f'均值差异的 95% CI\nd={cohens_d:.3f}, p={p_value:.6f}')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    output_path = Path('checkpoint/solution_ttest_results.png')
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n[可视化] 图表已保存：{output_path}")
    plt.close()


# =============================================================================
# 作业 4：功效分析
# =============================================================================

def solution_4_power_analysis():
    """
    作业 4 参考解答

    任务：分析不同样本量下的功效
    """
    print("\n" + "=" * 70)
    print("作业 4：功效分析")
    print("=" * 70)

    sample_sizes = [20, 50, 100, 200]
    true_diff = 8.0
    n_simulations = 5000

    powers = []

    for n in sample_sizes:
        # 模拟功效
        rejections = 0
        for _ in range(n_simulations):
            control = np.random.normal(loc=100, scale=15, size=n)
            treatment = np.random.normal(loc=100 + true_diff, scale=15, size=n)

            _, p_value = stats.ttest_ind(treatment, control)
            if p_value < 0.05:
                rejections += 1

        power = rejections / n_simulations
        powers.append(power)
        print(f"  样本量 n={n:3d}：功效 = {power:.3f}")

    # 可视化
    plot_power_curve(sample_sizes, powers)

    return powers


def plot_power_curve(sample_sizes, powers):
    """可视化功效曲线"""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(sample_sizes, powers, 'o-', linewidth=2, markersize=10, color='green')
    ax.axhline(0.8, color='red', linestyle='--', alpha=0.7, label='推荐功效 ≥ 80%')
    ax.set_xlabel('样本量')
    ax.set_ylabel('功效（1-β）')
    ax.set_title('功效随样本量增加而提升')
    ax.legend()
    ax.grid(True, alpha=0.3)

    output_path = Path('checkpoint/solution_power_curve.png')
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n[可视化] 图表已保存：{output_path}")
    plt.close()


# =============================================================================
# 作业 5：AI 报告审查
# =============================================================================

def solution_5_ai_report_review():
    """
    作业 5 参考解答

    任务：审查 AI 生成的统计报告，识别问题并修订
    """
    print("\n" + "=" * 70)
    print("作业 5：AI 报告审查")
    print("=" * 70)

    # 示例 AI 报告（含问题）
    ai_report = """
统计检验报告：

我们对实验组和对照组进行了 t 检验，结果 t=2.15, p=0.03。

结论：
1. 新功能显著提升了用户活跃度（H0 为真的概率是 3%）。
2. 两组均值差异为 5.2 分。

建议：
- 上线新功能，因为效果显著。
"""

    print("\n原始 AI 报告：")
    print(ai_report)

    # 审查
    issues = []

    # 检查 1：H0/H1
    if "H0" not in ai_report:
        issues.append({
            "问题": "未明确说明 H0",
            "建议": "补充 H0/H1 的正式陈述"
        })

    # 检查 2：p 值误解释
    if "H0 为真的概率" in ai_report:
        issues.append({
            "问题": "p 值误解释",
            "建议": "正确解释：在 H0 为真时看到当前数据的概率"
        })

    # 检查 3：缺少效应量
    if "Cohen" not in ai_report and "效应量" not in ai_report:
        issues.append({
            "问题": "缺少效应量",
            "建议": "补充 Cohen's d"
        })

    # 检查 4：缺少置信区间
    if "CI" not in ai_report and "置信区间" not in ai_report:
        issues.append({
            "问题": "缺少置信区间",
            "建议": "补充 95% CI"
        })

    # 检查 5：未验证前提假设
    if "正态性" not in ai_report:
        issues.append({
            "问题": "未验证正态性假设",
            "建议": "补充 Shapiro-Wilk 检验"
        })

    print("\n发现的问题：")
    for i, issue in enumerate(issues, 1):
        print(f"{i}. {issue['问题']}")
        print(f"   建议：{issue['建议']}")

    print(f"\n共发现 {len(issues)} 个问题")

    return issues


# =============================================================================
# 主函数
# =============================================================================

def main():
    """主函数：运行所有作业参考解答"""
    print("=" * 70)
    print("Week 06 作业参考解答")
    print("=" * 70)
    print("\n注意：本文件提供作业的参考实现")
    print("建议：先尝试自己完成，遇到困难时再参考\n")

    # 作业 1
    solution_1_research_to_hypothesis()

    # 作业 2
    solution_2_p_value_simulation()

    # 作业 3
    solution_3_complete_t_test()

    # 作业 4
    solution_4_power_analysis()

    # 作业 5
    solution_5_ai_report_review()

    print("\n" + "=" * 70)
    print("全部作业参考解答完成！")
    print("=" * 70)
    print("\n提示：")
    print("- 这些是基础要求的参考实现")
    print("- 进阶/挑战部分需要你自主完成")
    print("- 理解代码背后的概念比复制更重要")
    print("- 有疑问时回顾 CHAPTER.md 的讲解")
    print("=" * 70)


# =============================================================================
# 测试辅助函数（用于 pytest）
# =============================================================================

def formulate_hypothesis(question: str, test_type: str = "auto") -> dict:
    """
    将研究问题转化为 H0/H1 假设。

    参数：
    - question: 研究问题（字符串）
    - test_type: 检验类型（'auto', 'one_tailed', 'two_tailed'）

    返回：包含 H0, H1, test_type 的字典
    """
    # 根据关键词判断方向
    if "大于" in question or "高于" in question or "提升" in question:
        direction = "greater"
    elif "小于" in question or "低于" in question or "减少" in question:
        direction = "less"
    else:
        direction = "two_sided"

    # 确定检验类型
    if test_type == "auto":
        test_type = "one_tailed" if direction in ["greater", "less"] else "two_tailed"

    # 构建 H0 和 H1
    if direction == "greater":
        h0 = "μ1 = μ2（两组均值相等）"
        h1 = "μ1 > μ2（第一组均值大于第二组）"
        test_type = "one_tailed"
    elif direction == "less":
        h0 = "μ1 = μ2（两组均值相等）"
        h1 = "μ1 < μ2（第一组均值小于第二组）"
        test_type = "one_tailed"
    else:
        h0 = "μ1 = μ2（两组均值相等）"
        h1 = "μ1 ≠ μ2（两组均值不等）"
        test_type = "two_tailed"

    return {
        "H0": h0,
        "H1": h1,
        "test_type": test_type
    }


def validate_hypothesis(hypothesis: dict) -> dict:
    """
    验证假设的有效性。

    参数：
    - hypothesis: 包含 H0, H1, test_type 的字典

    返回：{'valid': bool, 'issues': list}
    """
    issues = []

    # 检查 H0
    if "H0" not in hypothesis or not hypothesis["H0"]:
        issues.append("缺少 H0 或 H0 为空")

    # 检查 H1
    if "H1" not in hypothesis or not hypothesis["H1"]:
        issues.append("缺少 H1 或 H1 为空")

    # 检查 H0 是否包含"相等"概念
    if "H0" in hypothesis and hypothesis["H0"]:
        h0 = hypothesis["H0"]
        if "=" not in h0 and "相等" not in h0 and "等于" not in h0:
            issues.append("H0 应该表示'相等'或'无差异'")

    return {
        "valid": len(issues) == 0,
        "issues": issues
    }


def interpret_p_value(p_value: float, alpha: float = 0.05) -> dict:
    """
    解释 p 值。

    参数：
    - p_value: p 值
    - alpha: 显著性水平（默认 0.05）

    返回：解释字典
    """
    is_significant = p_value < alpha

    # 证据强度
    if p_value == 0.0:
        evidence_strength = "强"
    elif p_value < 0.001:
        evidence_strength = "极强"
    elif p_value < 0.01:
        evidence_strength = "强"
    elif p_value < 0.05:
        evidence_strength = "中等"
    else:
        evidence_strength = "弱"

    return {
        "p_value": p_value,
        "alpha": alpha,
        "is_significant": is_significant,
        "decision": "拒绝 H0" if is_significant else "无法拒绝 H0",
        "evidence_strength": evidence_strength,
        "correct_interpretation": (
            f"在 H0 为真时，看到当前或更极端数据的概率为 {p_value:.4f}"
        ),
        "interpretation": (
            f"在 H0 为真时，看到当前或更极端数据的概率为 {p_value:.4f}"
        ),
        "common_misinterpretation": (
            "p 值不是 P(H0|data)，不能解释为'原假设为真的概率'"
        )
    }


def check_normality(data: np.ndarray, method: str = "shapiro") -> dict:
    """
    检查数据正态性。

    参数：
    - data: 数据数组
    - method: 检验方法（'shapiro', 'k2', 'anderson'）

    返回：{'p_value': float, 'is_normal': bool, 'method': str}
    """
    from scipy import stats

    if method == 'shapiro':
        # Shapiro-Wilk 检验（样本量 < 5000 时更准确）
        if len(data) > 5000:
            # 大样本时取前 5000 个
            _, p_value = stats.shapiro(data[:5000])
        else:
            _, p_value = stats.shapiro(data)
        method_name = "Shapiro-Wilk"
        is_normal = p_value > 0.05
    elif method == 'anderson':
        # Anderson-Darling 检验
        ad_result = stats.anderson(data)
        # 使用 5% 显著性水平的临界值
        critical_value = ad_result.critical_values[2]  # 5% level
        statistic = ad_result.statistic
        is_normal = statistic < critical_value
        # Anderson-Darling 不返回 p 值，我们用统计量与临界值比较
        p_value = None  # Anderson-Darling 不提供 p 值
        method_name = "Anderson-Darling"
        return {
            "p_value": p_value,
            "is_normal": is_normal,
            "method": method_name,
            "test": method_name,
            "statistic": statistic,
            "critical_value": critical_value
        }
    else:
        # D'Agostino's K2 检验（适合大样本）
        _, p_value = stats.normaltest(data)
        method_name = "D'Agostino's K2"
        is_normal = p_value > 0.05

    return {
        "p_value": p_value,
        "is_normal": is_normal,
        "method": method_name,
        "test": method_name,
        "statistic": p_value  # p_value from test
    }


def check_variance_homogeneity(group1: np.ndarray, group2: np.ndarray, method: str = "levene") -> dict:
    """
    检查两组方差齐性。

    参数：
    - group1: 第一组数据
    - group2: 第二组数据
    - method: 检验方法（'levene', 'bartlett', 'fligner'）

    返回：{'p_value': float, 'equal_variance': bool, 'method': str}
    """
    from scipy import stats

    if method == 'levene':
        # Levene 检验（更稳健，对偏态不敏感）
        _, p_value = stats.levene(group1, group2)
        method_name = "Levene"
    elif method == 'bartlett':
        # Bartlett 检验（假设数据正态）
        _, p_value = stats.bartlett(group1, group2)
        method_name = "Bartlett"
    elif method == 'fligner':
        # Fligner-Killeen 检验（非参数，最稳健）
        _, p_value = stats.fligner(group1, group2)
        method_name = "Fligner-Killeen"
    else:
        raise ValueError(f"未知的检验方法: {method}")

    return {
        "p_value": p_value,
        "equal_variance": p_value > 0.05,
        "method": method_name,
        "test": method_name
    }


def t_test_independent(
    group1: np.ndarray,
    group2: np.ndarray,
    check_assumptions: bool = True,
    equal_var: bool = None
) -> dict:
    """
    独立样本 t 检验。

    参数：
    - group1: 第一组数据
    - group2: 第二组数据
    - check_assumptions: 是否检查前提假设
    - equal_var: 是否假设方差相等（None 表示自动检测）

    返回：检验结果字典
    """
    from scipy import stats

    result = {}

    # 前提假设检查
    if check_assumptions:
        norm1 = check_normality(group1)
        norm2 = check_normality(group2)
        var_result = check_variance_homogeneity(group1, group2)

        result["assumptions"] = {
            "group1_normal": norm1["is_normal"],
            "group2_normal": norm2["is_normal"],
            "equal_variance": var_result["equal_variance"]
        }

        if equal_var is None:
            equal_var = var_result["equal_variance"]
    elif equal_var is None:
        equal_var = True

    # t 检验
    t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=equal_var)

    result.update({
        "t_statistic": t_stat,
        "p_value": p_value,
        "equal_var": equal_var,
        "test_type": "Welch's t 检验" if not equal_var else "独立样本 t 检验",
        "mean1": group1.mean(),
        "mean2": group2.mean(),
        "n1": len(group1),
        "n2": len(group2)
    })

    # 兼容测试期望的键名
    if check_assumptions:
        result["normality_group1"] = result["assumptions"]["group1_normal"]
        result["normality_group2"] = result["assumptions"]["group2_normal"]
        result["variance_homogeneity"] = result["assumptions"]["equal_variance"]
        result["variance_test_p"] = var_result.get("p_value")

    return result


def cohens_d(group1: np.ndarray, group2: np.ndarray, paired: bool = False) -> dict:
    """
    计算 Cohen's d 效应量。

    参数：
    - group1: 第一组数据
    - group2: 第二组数据
    - paired: 是否为配对样本

    返回：{'cohens_d': float, 'abs_d': float, 'effect_size': str, 'category': str}
    """
    n1, n2 = len(group1), len(group2)

    if paired:
        # 配对样本：使用差值的标准差
        differences = group1 - group2
        d = differences.mean() / differences.std(ddof=1)
    else:
        # 独立样本：使用合并标准差
        var1, var2 = group1.var(ddof=1), group2.var(ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        d = (group1.mean() - group2.mean()) / pooled_std

    abs_d = abs(d)

    # 解释效应量
    if abs_d < 0.2:
        effect_size = "negligible"
        category = "negligible"
    elif abs_d < 0.5:
        effect_size = "small"
        category = "small"
    elif abs_d < 0.8:
        effect_size = "medium"
        category = "medium"
    else:
        effect_size = "large"
        category = "large"

    return {
        "cohens_d": d,
        "abs_d": abs_d,
        "effect_size": effect_size,
        "category": category,
        "magnitude": abs_d
    }


def calculate_type_errors(true_state: str, decision: str) -> dict:
    """
    判断错误类型。

    参数：
    - true_state: 真实状态（'H0_true' 或 'H0_false'）
    - decision: 决策（'reject_H0' 或 'retain_H0'）

    返回：{'error_type': str, 'description': str}
    """
    # 错误矩阵
    # H0 true + reject = Type I
    # H0 true + retain = Correct
    # H0 false + reject = Correct
    # H0 false + retain = Type II

    if true_state == "H0_true":
        if decision == "reject_H0":
            return {
                "error_type": "Type I",
                "description": "第一类错误（假阳性）：H0 为真但被拒绝"
            }
        else:
            return {
                "error_type": "Correct",
                "description": "正确决策：H0 为真且被保留"
            }
    else:  # H0_false
        if decision == "reject_H0":
            return {
                "error_type": "Correct",
                "description": "正确决策：H0 为假且被拒绝"
            }
        else:
            return {
                "error_type": "Type II",
                "description": "第二类错误（假阴性）：H0 为假但被保留"
            }


def review_statistical_report(report_text: str) -> dict:
    """
    审查 AI 生成的统计报告。

    参数：
    - report_text: 报告文本

    返回：{'has_issues': bool, 'issues': list, 'issue_count': int}
    """
    issues = []

    # 检查 H0/H1
    if "H0" not in report_text and "原假设" not in report_text:
        issues.append({
            "问题": "未明确说明原假设 H0",
            "风险": "不清楚'显著'是拒绝什么假设",
            "建议": "补充 H0/H1 的正式陈述"
        })

    # 检查 p 值误解释
    if "H0 为真的概率" in report_text or "结论为真的概率" in report_text:
        issues.append({
            "问题": "p 值误解释",
            "风险": "严重逻辑错误（p ≠ P(H0|data）",
            "建议": "正确解释：在 H0 为真时，看到当前数据的概率"
        })

    # 检查效应量
    if "Cohen" not in report_text and "效应量" not in report_text:
        issues.append({
            "问题": "缺少效应量",
            "风险": "无法判断实际意义（p<0.05 可能是微小效应）",
            "建议": "补充 Cohen's d 或 η²"
        })

    # 检查置信区间
    if "CI" not in report_text and "置信区间" not in report_text:
        issues.append({
            "问题": "缺少置信区间",
            "风险": "无法量化不确定性",
            "建议": "补充 95% CI"
        })

    # 检查前提假设
    if "正态性" not in report_text and "Shapiro" not in report_text:
        issues.append({
            "问题": "未验证正态性假设",
            "风险": "数据严重偏态时 t 检验不可靠",
            "建议": "补充 Shapiro-Wilk 检验或 QQ 图"
        })

    if "方差齐性" not in report_text and "Levene" not in report_text:
        issues.append({
            "问题": "未验证方差齐性假设",
            "风险": "方差不齐时应使用 Welch's t 检验",
            "建议": "补充 Levene 检验"
        })

    # 检查多重比较
    if ("多次检验" in report_text or "多个指标" in report_text) and \
       ("校正" not in report_text and "Bonferroni" not in report_text):
        issues.append({
            "问题": "多重比较未校正",
            "风险": "假阳性风险放大（跑 20 次检验总会有 1 次碰巧显著）",
            "建议": "使用 Bonferroni 或 FDR 校正"
        })

    # 检查功效/样本量
    if "功效" not in report_text and "样本量" not in report_text:
        issues.append({
            "问题": "未讨论样本量/功效",
            "风险": "小样本检测小效应时假阴性风险高",
            "建议": "补充功效分析或说明样本量限制"
        })

    # 检查相关被误写成因果
    if ("导致" in report_text or "因果" in report_text) and \
       ("实验" not in report_text and "随机" not in report_text):
        issues.append({
            "问题": "相关被误写成因果",
            "风险": "观察性研究无法确定因果方向",
            "建议": "用'相关'、'关联'而非'导致'、'因果'"
        })

    return {
        "has_issues": len(issues) > 0,
        "issues": issues,
        "issue_count": len(issues)
    }


def check_p_value_interpretation(statement: str) -> dict:
    """
    检查 p 值解释的正确性。

    参数：
    - statement: p 值解释陈述

    返回：{'is_correct': bool, 'issues': list, 'positive_signals': list}
    """
    issues = []
    positive_signals = []

    # 常见误解释模式
    misinterpretation_patterns = [
        ("H0 为真的概率", "H0 为真的概率"),
        ("证明 H0", "证明 H0"),
        ("证明 H1", "证明 H1"),
    ]

    for pattern, description in misinterpretation_patterns:
        if pattern in statement:
            issues.append({
                "type": "误解释",
                "pattern": description,
                "explanation": f"不应使用'{description}'的表述"
            })

    # 正确解释的关键词
    correct_patterns = [
        "H0 为真时",
        "在 H0 为真时",
        "如果 H0 是真的",
        "观察到当前",
    ]

    for pattern in correct_patterns:
        if pattern in statement:
            positive_signals.append(pattern)

    # 判断是否有问题
    has_issues = len(issues) > 0

    return {
        "is_correct": len(issues) == 0 and len(positive_signals) > 0,
        "has_issues": has_issues,
        "issues": issues,
        "positive_signals": positive_signals
    }


def calculate_type_errors(true_state: str, decision: str, alpha: float = 0.05) -> dict:
    """
    判断错误类型。

    参数：
    - true_state: 真实状态（'H0_true' 或 'H0_false'）
    - decision: 决策（'reject_H0' 或 'retain_H0'）
    - alpha: 显著性水平

    返回：{'error_type': str, 'result': str, 'probability': str}
    """
    # 错误矩阵
    if true_state == "H0_true":
        if decision == "reject_H0":
            return {
                "error_type": "Type I",
                "result": "第一类错误（假阳性）：H0 为真但被拒绝",
                "probability": f"α = {alpha}"
            }
        else:
            return {
                "error_type": None,
                "result": "正确决策（真阴性）：H0 为真且被保留",
                "probability": f"1-α = {1-alpha}"
            }
    else:  # H0_false
        if decision == "reject_H0":
            return {
                "error_type": None,
                "result": "正确决策（真阳性）：H0 为假且被拒绝",
                "probability": "1-β（功效）"
            }
        else:
            return {
                "error_type": "Type II",
                "result": "第二类错误（假阴性）：H0 为假但被保留",
                "probability": "β"
            }


def calculate_power(
    n_per_group: int,
    effect_size: float,
    alpha: float = 0.05,
    ratio: float = 1.0
) -> dict:
    """
    计算统计功效。

    参数：
    - n_per_group: 每组样本量
    - effect_size: Cohen's d（标准化效应量）
    - alpha: 显著性水平
    - ratio: 两组样本量比例

    返回：{'power': float, 'interpretation': str}
    """
    from scipy import stats

    # 简化的功效计算（基于 Z 检验近似）
    z_alpha = stats.norm.ppf(1 - alpha / 2)  # 双尾检验
    z_beta = effect_size * np.sqrt(n_per_group / (1 + ratio)) - z_alpha

    # 转换为功效
    power = stats.norm.cdf(z_beta)

    # 解释
    if power >= 0.8:
        interpretation = "功效充足（≥ 80%）"
    elif power >= 0.6:
        interpretation = "功效中等（60-80%）"
    else:
        interpretation = "功效不足（< 60%），建议增加样本量"

    return {
        "power": power,
        "effect_size": effect_size,
        "n_per_group": n_per_group,
        "interpretation": interpretation,
        "beta": 1 - power,
        "recommendation": "增加样本量" if power < 0.8 else "样本量充足",
        "is_adequate": power >= 0.8
    }


def simulate_type_error_rates(
    n_sim: int = 10000,
    n_sample: int = 50,
    true_diff: float = 0,
    alpha: float = 0.05,
    seed: int = 42
) -> dict:
    """
    模拟两类错误率。

    参数：
    - n_sim: 模拟次数
    - n_sample: 每组样本量
    - true_diff: 真实均值差异
    - alpha: 显著性水平
    - seed: 随机种子

    返回：{'type_i_error_rate': float, 'type_ii_error_rate': float, 'power': float}
    """
    np.random.seed(seed)

    type_i_count = 0
    type_ii_count = 0
    total_true_H0 = 0
    total_false_H0 = 0

    for _ in range(n_sim):
        # 生成数据
        control = np.random.normal(loc=100, scale=15, size=n_sample)
        treatment = np.random.normal(loc=100 + true_diff, scale=15, size=n_sample)

        # t 检验
        _, p_value = stats.ttest_ind(treatment, control)

        # 决策
        if true_diff == 0:
            # H0 为真
            total_true_H0 += 1
            if p_value < alpha:
                type_i_count += 1
        else:
            # H0 为假
            total_false_H0 += 1
            if p_value >= alpha:
                type_ii_count += 1

    # 计算率
    type_i_rate = type_i_count / total_true_H0 if total_true_H0 > 0 else None
    type_ii_rate = type_ii_count / total_false_H0 if total_false_H0 > 0 else None
    power = 1 - type_ii_rate if type_ii_rate is not None else None

    return {
        "type_i_error_rate": type_i_rate,
        "type_ii_error_rate": type_ii_rate,
        "type_i_rate": type_i_rate,  # 保留旧字段兼容性
        "type_ii_rate": type_ii_rate,  # 保留旧字段兼容性
        "power": power,
        "n_simulations": n_sim
    }


def interpret_effect_size(cohens_d: float, context: str = None) -> str:
    """
    解释效应量。

    参数：
    - cohens_d: Cohen's d 值
    - context: 应用上下文（可选）

    返回：效应量解释字符串
    """
    abs_d = abs(cohens_d)

    # 效应量分类
    if abs_d < 0.2:
        category = "可忽略"
        description = "效应极小，实际意义有限"
    elif abs_d < 0.5:
        category = "小"
        description = "小效应，但在某些场景下可能有实际意义"
    elif abs_d < 0.8:
        category = "中等"
        description = "中等效应，通常有实际意义"
    else:
        category = "大"
        description = "大效应，具有明确的实际意义"

    # 方向描述
    if cohens_d > 0:
        direction = "高于"
    elif cohens_d < 0:
        direction = "低于"
    else:
        direction = "等于"

    # 构建解释字符串
    if abs_d == 0:
        interpretation = f"Cohen's d = {cohens_d:.3f}（{category}效应），两组均值{direction}"
    else:
        interpretation = f"Cohen's d = {cohens_d:.3f}（{category}效应），第一组均值{direction}第二组"

    if context:
        interpretation += f"，在{context}下，{description}"
    else:
        interpretation += f"，{description}"

    return interpretation


def t_test_one_sample(data: np.ndarray, pop_mean: float = 0.0, check_assumptions: bool = True) -> dict:
    """
    单样本 t 检验。

    参数：
    - data: 样本数据
    - pop_mean: 假设的总体均值
    - check_assumptions: 是否检查前提假设

    返回：检验结果字典
    """
    from scipy import stats

    result = {}

    # 前提假设检查
    if check_assumptions:
        norm_result = check_normality(data)
        result["assumptions"] = {
            "normality": norm_result["is_normal"],
            "normality_p_value": norm_result["p_value"]
        }

    # t 检验
    t_stat, p_value = stats.ttest_1samp(data, pop_mean)

    sample_mean = data.mean()
    mean_diff = sample_mean - pop_mean

    result.update({
        "t_statistic": t_stat,
        "p_value": p_value,
        "pop_mean": pop_mean,
        "hypothesized_mean": pop_mean,
        "sample_mean": sample_mean,
        "sample_std": data.std(ddof=1),
        "mean_difference": mean_diff,
        "n": len(data),
        "test_type": "单样本 t 检验"
    })

    # 兼容测试期望的键名
    if check_assumptions:
        result["normality_assumption"] = result["assumptions"]["normality"]
        result["normality_p"] = result["assumptions"]["normality_p_value"]

    return result


def t_test_paired(before: np.ndarray, after: np.ndarray, check_assumptions: bool = True) -> dict:
    """
    配对样本 t 检验。

    参数：
    - before: 处理前数据
    - after: 处理后数据
    - check_assumptions: 是否检查前提假设

    返回：检验结果字典
    """
    from scipy import stats

    result = {}
    differences = after - before

    # 前提假设检查
    if check_assumptions:
        norm_result = check_normality(differences)
        result["assumptions"] = {
            "normality_differences": norm_result["is_normal"],
            "differences_normal": norm_result["is_normal"],
            "differences_normal_p_value": norm_result["p_value"]
        }

    # 配对 t 检验
    t_stat, p_value = stats.ttest_rel(after, before)

    mean_diff = differences.mean()
    std_diff = differences.std(ddof=1)
    df = len(before) - 1  # 自由度

    result.update({
        "t_statistic": t_stat,
        "p_value": p_value,
        "mean_difference": mean_diff,
        "std_difference": std_diff,
        "mean_before": before.mean(),
        "mean_after": after.mean(),
        "n_pairs": len(before),
        "df": df,
        "test_type": "配对样本 t 检验"
    })

    # 兼容测试期望的键名
    if check_assumptions:
        result["normality_differences"] = result["assumptions"]["differences_normal"]
        result["normality_p"] = result["assumptions"]["differences_normal_p_value"]

    return result


if __name__ == "__main__":
    main()

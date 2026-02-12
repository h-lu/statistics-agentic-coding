#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
示例：完整的 t 检验流程——从前提检查到结果解释

本例演示一个完整的独立样本 t 检验流程：
1. 数据加载（模拟 A/B 测试）
2. 前提假设检查（正态性、方差齐性）
3. t 检验（根据方差齐性选择标准 or Welch's）
4. 效应量（Cohen's d）计算
5. 95% 置信区间
6. 可视化（两组分布对比 + 差异 CI）

运行方式：python3 chapters/week_06/examples/02_ttest_example.py
预期输出：终端显示完整检验报告、生成 ttest_results.png 可视化图

核心概念：
- t 检验不是万能的，必须先检查前提假设
- 方差不齐时使用 Welch's t 检验（而非标准 t 检验）
- 同时报告 p 值、效应量和置信区间才是完整的统计报告

作者：StatLab Week 06
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from typing import Tuple, Dict


# =============================================================================
# 数据生成
# =============================================================================

def load_ab_test_data(
    n_control: int = 500,
    n_treatment: int = 500,
    control_mean: float = 100.0,
    treatment_mean: float = 105.0,
    std_dev: float = 15.0,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成模拟 A/B 测试数据。

    参数：
        n_control: 对照组样本量
        n_treatment: 实验组样本量
        control_mean: 对照组均值
        treatment_mean: 实验组均值
        std_dev: 标准差（两组相同）
        seed: 随机种子

    返回：
        tuple: (对照组数据, 实验组数据)
    """
    np.random.seed(seed)

    control = np.random.normal(loc=control_mean, scale=std_dev, size=n_control)
    treatment = np.random.normal(loc=treatment_mean, scale=std_dev, size=n_treatment)

    return control, treatment


# =============================================================================
# 前提假设检查
# =============================================================================

def check_normality(data: np.ndarray, alpha: float = 0.05) -> Dict[str, any]:
    """
    检查数据的正态性假设（Shapiro-Wilk 检验）。

    H0: 数据来自正态分布

    参数：
        data: 待检验数据
        alpha: 显著性水平

    返回：
        dict: 包含检验统计量、p 值、结论
    """
    # Shapiro-Wilk 检验
    # 注意：当样本量 > 5000 时，shapiro 函数会警告
    if len(data) > 5000:
        data = np.random.choice(data, 5000, replace=False)

    statistic, p_value = stats.shapiro(data)

    return {
        'statistic': statistic,
        'p_value': p_value,
        'is_normal': p_value > alpha,
        'conclusion': '✓ 正态性假设满足' if p_value > alpha else '✗ 数据可能偏离正态（考虑非参数检验）'
    }


def check_variance_homogeneity(
    group1: np.ndarray,
    group2: np.ndarray,
    alpha: float = 0.05
) -> Dict[str, any]:
    """
    检查两组数据的方差齐性（Levene 检验）。

    H0: 两组方差相等

    参数：
        group1: 第一组数据
        group2: 第二组数据
        alpha: 显著性水平

    返回：
        dict: 包含检验统计量、p 值、结论和建议
    """
    statistic, p_value = stats.levene(group1, group2)

    return {
        'statistic': statistic,
        'p_value': p_value,
        'equal_variance': p_value > alpha,
        'conclusion': '✓ 方差齐性假设满足' if p_value > alpha else '✗ 方差不齐（使用 Welch\'s t 检验）',
        'recommended_test': '标准 t 检验' if p_value > alpha else 'Welch\'s t 检验'
    }


# =============================================================================
# t 检验与效应量
# =============================================================================

def perform_t_test(
    group1: np.ndarray,
    group2: np.ndarray,
    equal_var: bool = True,
    alternative: str = 'two-sided'
) -> Dict[str, any]:
    """
    执行独立样本 t 检验。

    参数：
        group1: 第一组数据
        group2: 第二组数据
        equal_var: 是否假设方差齐性
        alternative: 备择假设类型 ('two-sided', 'less', 'greater')

    返回：
        dict: 包含 t 统计量、p 值、自由度、决策
    """
    t_stat, p_value = stats.ttest_ind(
        group1, group2,
        equal_var=equal_var,
        alternative=alternative
    )

    # 自由度计算
    n1, n2 = len(group1), len(group2)
    if equal_var:
        df = n1 + n2 - 2
    else:
        # Welch-Satterthwaite 自由度
        var1, var2 = group1.var(ddof=1), group2.var(ddof=1)
        df = (var1/n1 + var2/n2)**2 / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))

    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'degrees_of_freedom': df,
        'equal_var': equal_var,
        'alpha': 0.05,
        'reject_h0': p_value < 0.05,
        'decision': '拒绝 H0（差异显著）' if p_value < 0.05 else '无法拒绝 H0（差异不显著）'
    }


def calculate_cohens_d(group1: np.ndarray, group2: np.ndarray) -> Dict[str, any]:
    """
    计算 Cohen's d 效应量。

    Cohen's d = (均值1 - 均值2) / 合并标准差

    参数：
        group1: 第一组数据
        group2: 第二组数据

    返回：
        dict: 包含 Cohen's d、解释、定性评估
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(ddof=1), group2.var(ddof=1)

    # 合并标准差（pooled standard deviation）
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    # Cohen's d
    d = (np.mean(group1) - np.mean(group2)) / pooled_std

    # 解释（Cohen's 经验标准）
    abs_d = abs(d)
    if abs_d < 0.2:
        interpretation = "效应量很小（small）"
        magnitude = "small"
    elif abs_d < 0.5:
        interpretation = "效应量中等（medium）"
        magnitude = "medium"
    elif abs_d < 0.8:
        interpretation = "效应量较大（large）"
        magnitude = "large"
    else:
        interpretation = "效应量非常大（very large）"
        magnitude = "very large"

    return {
        'cohens_d': d,
        'magnitude': magnitude,
        'interpretation': interpretation,
        'pooled_std': pooled_std
    }


def calculate_confidence_interval(
    group1: np.ndarray,
    group2: np.ndarray,
    confidence: float = 0.95
) -> Dict[str, any]:
    """
    计算均值差异的置信区间。

    参数：
        group1: 第一组数据
        group2: 第二组数据
        confidence: 置信水平

    返回：
        dict: 包含点估计、CI 下界、CI 上界、是否包含0
    """
    mean_diff = np.mean(group1) - np.mean(group2)

    # 标准误
    se_diff = np.sqrt(
        group1.var(ddof=1) / len(group1) +
        group2.var(ddof=1) / len(group2)
    )

    # Z 值（大样本）或 t 值（小样本）
    # 这里简化使用 Z 值
    z_value = stats.norm.ppf((1 + confidence) / 2)

    margin_of_error = z_value * se_diff
    ci_low = mean_diff - margin_of_error
    ci_high = mean_diff + margin_of_error

    return {
        'point_estimate': mean_diff,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'confidence_level': confidence,
        'margin_of_error': margin_of_error,
        'contains_zero': ci_low <= 0 <= ci_high,
        'conclusion': 'CI 不包含 0，差异显著' if not (ci_low <= 0 <= ci_high) else 'CI 包含 0，差异不显著'
    }


# =============================================================================
# 可视化
# =============================================================================

def plot_ttest_results(
    group1: np.ndarray,
    group2: np.ndarray,
    t_result: Dict,
    effect_size: Dict,
    ci_result: Dict,
    output_path: str = 'checkpoint/ttest_results.png'
) -> None:
    """
    可视化 t 检验结果。

    包含两个子图：
    1. 两组分布对比（直方图 + 均值线）
    2. 均值差异的置信区间（误差条图）

    参数：
        group1: 第一组数据
        group2: 第二组数据
        t_result: t 检验结果字典
        effect_size: 效应量字典
        ci_result: 置信区间字典
        output_path: 输出文件路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：两组分布对比
    axes[0].hist(group1, bins=30, alpha=0.7, label='对照组', density=True, color='steelblue')
    axes[0].hist(group2, bins=30, alpha=0.7, label='实验组', density=True, color='orange')
    axes[0].axvline(np.mean(group1), color='darkblue', linestyle='--', linewidth=2, label=f'对照组均值={np.mean(group1):.1f}')
    axes[0].axvline(np.mean(group2), color='darkorange', linestyle='--', linewidth=2, label=f'实验组均值={np.mean(group2):.1f}')

    axes[0].set_xlabel('活跃度', fontsize=12)
    axes[0].set_ylabel('密度', fontsize=12)
    axes[0].set_title('两组分布对比', fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # 右图：差异的置信区间
    mean_diff = ci_result['point_estimate']
    ci_low = ci_result['ci_low']
    ci_high = ci_result['ci_high']
    d = effect_size['cohens_d']
    p = t_result['p_value']

    axes[1].errorbar(
        0, mean_diff,
        yerr=[[mean_diff - ci_low], [ci_high - mean_diff]],
        fmt='o', capsize=10, capthick=2, linewidth=2,
        markersize=12, color='steelblue'
    )
    axes[1].axhline(0, color='red', linestyle='--', alpha=0.7, label='零差异线')
    axes[1].set_xlim(-0.5, 0.5)
    axes[1].set_ylim(ci_low - abs(ci_low)*0.5, ci_high + abs(ci_high)*0.5)
    axes[1].set_xticks([])
    axes[1].set_ylabel('均值差异', fontsize=12)
    axes[1].set_title(
        f'均值差异的 95% CI\n'
        f'd={d:.3f} ({effect_size["interpretation"]}），p={p:.6f}',
        fontsize=14, fontweight='bold'
    )
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # 保存图表
    Path(output_path).parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n[可视化] 图表已保存：{output_path}")
    plt.close()


# =============================================================================
# 完整流程
# =============================================================================

def complete_ttest_pipeline() -> Dict:
    """
    执行完整的 t 检验流程：从数据加载到结果解释。

    返回：
        dict: 包含所有结果的字典
    """
    print("=" * 70)
    print("完整 t 检验流程：从前提检查到结果解释")
    print("=" * 70)

    # 1. 加载数据
    print("\n[1] 数据加载")
    control, treatment = load_ab_test_data()
    print(f"  对照组：n={len(control)}, 均值={np.mean(control):.2f}, 标准差={np.std(control, ddof=1):.2f}")
    print(f"  实验组：n={len(treatment)}, 均值={np.mean(treatment):.2f}, 标准差={np.std(treatment, ddof=1):.2f}")
    print(f"  观察差异：{np.mean(treatment) - np.mean(control):.2f}")

    # 2. 前提假设检查
    print("\n[2] 前提假设检查")

    # 正态性检验
    print("  正态性检验（Shapiro-Wilk）：")
    norm_ctrl = check_normality(control)
    norm_treat = check_normality(treatment)
    print(f"    对照组：W={norm_ctrl['statistic']:.4f}, p={norm_ctrl['p_value']:.4f} - {norm_ctrl['conclusion']}")
    print(f"    实验组：W={norm_treat['statistic']:.4f}, p={norm_treat['p_value']:.4f} - {norm_treat['conclusion']}")

    # 方差齐性检验
    print("  方差齐性检验（Levene）：")
    levene = check_variance_homogeneity(control, treatment)
    print(f"    统计量={levene['statistic']:.4f}, p={levene['p_value']:.4f}")
    print(f"    结论：{levene['conclusion']}")
    print(f"    建议使用：{levene['recommended_test']}")

    # 3. t 检验
    print("\n[3] t 检验")
    t_result = perform_t_test(treatment, control, equal_var=levene['equal_variance'])
    print(f"  检验方法：{'Welch\'s t 检验' if not t_result['equal_var'] else '标准 t 检验'}")
    print(f"  t 统计量：t({t_result['degrees_of_freedom']:.1f}) = {t_result['t_statistic']:.4f}")
    print(f"  p 值（双尾）：{t_result['p_value']:.6f}")
    print(f"  显著性水平 α=0.05：{t_result['decision']}")

    # 4. 效应量
    print("\n[4] 效应量（Cohen's d）")
    effect_size = calculate_cohens_d(treatment, control)
    print(f"  Cohen's d = {effect_size['cohens_d']:.3f}")
    print(f"  解释：{effect_size['interpretation']}")

    # 5. 置信区间
    print("\n[5] 95% 置信区间")
    ci_result = calculate_confidence_interval(treatment, control)
    print(f"  点估计：{ci_result['point_estimate']:.2f}")
    print(f"  95% CI：[{ci_result['ci_low']:.2f}, {ci_result['ci_high']:.2f}]")
    print(f"  边际误差：{ci_result['margin_of_error']:.2f}")
    print(f"  结论：{ci_result['conclusion']}")

    # 6. 可视化
    print("\n[6] 生成可视化")
    plot_ttest_results(control, treatment, t_result, effect_size, ci_result)

    # 7. 综合结论
    print("\n[7] 综合结论")
    print("  " + "-" * 66)
    if t_result['reject_h0']:
        print(f"  ✓ 在 α=0.05 水平下拒绝 H0，认为实验组与对照组的均值存在显著差异")
        print(f"  ✓ p={t_result['p_value']:.6f} < 0.05，统计显著性成立")
        print(f"  ✓ Cohen's d={effect_size['cohens_d']:.3f}，{effect_size['interpretation']}")
        print(f"  ✓ 95% CI [{ci_result['ci_low']:.2f}, {ci_result['ci_high']:.2f}] 不包含 0，支持差异为正")
    else:
        print(f"  ✗ 无法拒绝 H0，差异可能由抽样误差导致")
        print(f"  ✗ p={t_result['p_value']:.6f} > 0.05，统计显著性不成立")
        print(f"  ✗ 95% CI [{ci_result['ci_low']:.2f}, {ci_result['ci_high']:.2f}] 包含 0，不能排除零差异")

    print("  " + "-" * 66)

    # 返回所有结果
    return {
        'control': control,
        'treatment': treatment,
        'normality_control': norm_ctrl,
        'normality_treatment': norm_treat,
        'levene': levene,
        't_test': t_result,
        'effect_size': effect_size,
        'confidence_interval': ci_result
    }


# =============================================================================
# 反例：常见的 t 检验错误
# =============================================================================

def bad_t_test_example() -> None:
    """
    反例：常见的 t 检验错误

    这个函数展示了小北和阿码常犯的错误，
    帮助读者避免类似问题。
    """
    print("\n" + "=" * 70)
    print("反例：小北和阿码的 t 检验错误")
    print("=" * 70)

    # 生成数据（对照组方差大）
    np.random.seed(42)
    control = np.random.normal(100, 25, 50)  # 大方差
    treatment = np.random.normal(110, 10, 50)  # 小方差

    print("\n[数据]")
    print(f"  对照组：均值={np.mean(control):.2f}, 标准差={np.std(control, ddof=1):.2f}")
    print(f"  实验组：均值={np.mean(treatment):.2f}, 标准差={np.std(treatment, ddof=1):.2f}")

    # 小北的错误1：不做方差齐性检验
    print("\n[小北的错误1：不做前提假设检查]")
    t_stat1, p_value1 = stats.ttest_ind(treatment, control, equal_var=True)
    print(f"  ❌ 直接用标准 t 检验：t={t_stat1:.3f}, p={p_value1:.4f}")
    print(f"  ❌ 没检查方差是否相等")

    # 正确做法
    levene_stat, levene_p = stats.levene(control, treatment)
    print(f"\n  ✓ 正确做法：先做 Levene 检验")
    print(f"    Levene: statistic={levene_stat:.4f}, p={levene_p:.4f}")
    t_stat2, p_value2 = stats.ttest_ind(treatment, control, equal_var=False)
    print(f"    使用 Welch's t 检验：t={t_stat2:.3f}, p={p_value2:.4f}")
    print(f"    p 值差异：{abs(p_value1 - p_value2):.4f}（方差不齐时不可忽略！）")

    # 小北的错误2：只看 p 值
    print("\n[小北的错误2：只报告 p 值，不报告效应量]")
    print(f'  ❌ "结论：p={p_value2:.4f} < 0.05，差异显著"')
    print(f"  ❌ 没说明效应有多大")

    # 正确做法
    d = (np.mean(treatment) - np.mean(control)) / np.sqrt(
        ((len(treatment)-1)*treatment.var(ddof=1) + (len(control)-1)*control.var(ddof=1)) /
        (len(treatment) + len(control) - 2)
    )
    print(f"\n  ✓ 正确做法：同时报告效应量")
    print(f"    Cohen's d = {d:.3f}（{'小' if abs(d)<0.2 else '中' if abs(d)<0.5 else '大'}效应）")
    print(f"    统计显著 ≠ 实际显著，需要结合业务判断")

    # 阿码的错误3：样本量很小时也用 Shapiro
    print("\n[阿码的错误3：样本量很小时也用 Shapiro 检验正态性]")
    tiny_sample = np.random.normal(100, 15, 10)
    shapiro_stat, shapiro_p = stats.shapiro(tiny_sample)
    print(f"  ❌ 样本量 n=10 时做 Shapiro：W={shapiro_stat:.4f}, p={shapiro_p:.4f}")
    print(f"  ❌ 小样本时 Shapiro 检验功效很低，即使不正态也可能 p>0.05")
    print(f"\n  ✓ 正确做法：小样本时结合 QQ 图和业务判断")


# =============================================================================
# 主函数
# =============================================================================

def main() -> None:
    """主函数：运行完整 t 检验示例"""
    # 完整流程
    result = complete_ttest_pipeline()

    # 反例
    bad_t_test_example()

    print("\n" + "=" * 70)
    print("要点总结")
    print("=" * 70)
    print("1. t 检验前必须检查前提假设：正态性、方差齐性、样本独立性")
    print("2. 方差不齐时使用 Welch's t 检验（而非标准 t 检验）")
    print("3. p 值告诉你'是否显著'，效应量告诉你'有多大'")
    print("4. 同时报告 p 值、效应量和置信区间才是完整的统计报告")
    print("5. 统计显著（p<0.05）≠ 实际显著（效应量大）")
    print("=" * 70)


if __name__ == "__main__":
    main()

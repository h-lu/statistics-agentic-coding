"""
示例：效应量计算——区分"统计显著"与"实际意义"。

本例演示如何计算 Cohen's d（两组均值差异的效应量），并解释为什么
p 值小不一定意味着效应量大。效应量描述"差异有多大"，不受样本量影响。

运行方式：python3 chapters/week_06/examples/03_effect_size_demo.py
预期输出：
  - stdout 输出 Cohen's d 值和解释
  - 对比不同样本量下 p 值的变化（效应量不变，p 值变化）
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def cohen_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    计算 Cohen's d（两组均值差异的效应量）

    d = (mean1 - mean2) / pooled_standard_deviation

    参数：
        group1, group2: 两组数据

    返回：
        Cohen's d 值
    """
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = group1.mean(), group2.mean()
    var1, var2 = group1.var(ddof=1), group2.var(ddof=1)

    # 合并标准差
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    return (mean1 - mean2) / pooled_std


def interpret_cohens_d(d: float) -> str:
    """
    解释 Cohen's d 的效应量大小

    参数：
        d: Cohen's d 值

    返回：
        效应量解释字符串
    """
    abs_d = abs(d)
    if abs_d < 0.2:
        return "小效应"
    elif abs_d < 0.5:
        return "小到中等效应"
    elif abs_d < 0.8:
        return "中等效应"
    else:
        return "大效应"


def main() -> None:
    """运行效应量演示"""
    np.random.seed(42)

    print("=== 效应量演示：统计显著 ≠ 实际意义 ===\n")

    # 场景 1：小样本，差异 3%
    print("=== 场景 1：小样本（每组 120 用户），差异 3% ===")
    conversions_a_small = np.array([1] * 14 + [0] * (120 - 14))  # 11.7%
    conversions_b_small = np.array([1] * 11 + [0] * (120 - 11))  # 9.2%

    rate_a_small = conversions_a_small.mean()
    rate_b_small = conversions_b_small.mean()
    diff_small = rate_a_small - rate_b_small

    t_stat_small, p_value_small = stats.ttest_ind(conversions_a_small, conversions_b_small)
    d_small = cohen_d(conversions_a_small, conversions_b_small)

    print(f"A 渠道转化率：{rate_a_small:.2%}")
    print(f"B 渠道转化率：{rate_b_small:.2%}")
    print(f"差异：{diff_small:.2%}")
    print(f"p 值：{p_value_small:.4f}")
    print(f"Cohen's d：{d_small:.4f}（{interpret_cohens_d(d_small)}）")

    if p_value_small < 0.05:
        print(f"结论：p < 0.05，统计显著")
    else:
        print(f"结论：p ≥ 0.05，不显著")

    print()

    # 场景 2：大样本，相同差异 3%
    print("=== 场景 2：大样本（每组 1200 用户），相同差异 3% ===")
    conversions_a_large = np.array([1] * 144 + [0] * (1200 - 144))  # 12%
    conversions_b_large = np.array([1] * 108 + [0] * (1200 - 108))  # 9%

    rate_a_large = conversions_a_large.mean()
    rate_b_large = conversions_b_large.mean()
    diff_large = rate_a_large - rate_b_large

    t_stat_large, p_value_large = stats.ttest_ind(conversions_a_large, conversions_b_large)
    d_large = cohen_d(conversions_a_large, conversions_b_large)

    print(f"A 渠道转化率：{rate_a_large:.2%}")
    print(f"B 渠道转化率：{rate_b_large:.2%}")
    print(f"差异：{diff_large:.2%}")
    print(f"p 值：{p_value_large:.4f}")
    print(f"Cohen's d：{d_large:.4f}（{interpret_cohens_d(d_large)}）")

    if p_value_large < 0.05:
        print(f"结论：p < 0.05，统计显著")
    else:
        print(f"结论：p ≥ 0.05，不显著")

    print()

    # 对比总结
    print("=== 对比总结 ===")
    print(f"场景 1：差异 {diff_small:.2%}，n=120，p={p_value_small:.4f}，{'显著' if p_value_small < 0.05 else '不显著'}")
    print(f"场景 2：差异 {diff_large:.2%}，n=1200，p={p_value_large:.4f}，{'显著' if p_value_large < 0.05 else '不显著'}")
    print()
    print("关键发现：")
    print("  - 差异大小相似（约 3%）")
    print("  - 效应量相似（Cohen's d ≈ 0.1，小效应）")
    print("  - 但 p 值差异巨大：小样本不显著，大样本高度显著")
    print()
    print("结论：p 值受样本量影响，效应量不受样本量影响。")
    print("      决策时应该优先看效应量，而不是只看 p 值。")

    print("\n=== Cohen's d 的解释惯例 ===")
    print("  | d < 0.2  ：小效应")
    print("  | 0.2 ≤ d < 0.5 ：小到中等效应")
    print("  | 0.5 ≤ d < 0.8 ：中等效应")
    print("  | d ≥ 0.8  ：大效应")


if __name__ == "__main__":
    main()

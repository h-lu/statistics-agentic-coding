"""
Week 14 作业参考实现

本文件是作业的参考解答，供学生在遇到困难时查看。
包含基础作业的完整实现，不涉及进阶/挑战部分。

运行方式：python3 chapters/week_14/starter_code/solution.py
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest


def exercise_1_frequentist_ab(conversions_A: int, exposures_A: int,
                             conversions_B: int, exposures_B: int) -> dict:
    """
    练习 1：频率学派 A/B 测试

    实现：
    1. 计算两组的转化率
    2. 使用 scipy.stats.proportions_ztest 进行 z 检验
    3. 计算 p 值和置信区间
    4. 给出"显著/不显著"的结论

    参数:
        conversions_A: A 组转化数
        exposures_A: A 组曝光数
        conversions_B: B 组转化数
        exposures_B: B 组曝光数

    返回:
        包含转化率、p 值、置信区间和结论的字典
    """
    # 1. 计算转化率
    p_A = conversions_A / exposures_A
    p_B = conversions_B / exposures_B

    # 2. z 检验
    from statsmodels.stats.proportion import proportions_ztest
    count = np.array([conversions_B, conversions_A])
    nobs = np.array([exposures_B, exposures_A])
    z_stat, p_value = proportions_ztest(count, nobs)

    # 3. 计算置信区间（差异的置信区间）
    # 使用正态近似
    se_A = np.sqrt(p_A * (1 - p_A) / exposures_A)
    se_B = np.sqrt(p_B * (1 - p_B) / exposures_B)
    se_diff = np.sqrt(se_A**2 + se_B**2)

    ci_low = (p_B - p_A) - 1.96 * se_diff
    ci_high = (p_B - p_A) + 1.96 * se_diff

    # 4. 结论
    alpha = 0.05
    if p_value < alpha:
        conclusion = f"拒绝原假设，差异显著（p={p_value:.4f} < {alpha}）"
    else:
        conclusion = f"无法拒绝原假设，差异不显著（p={p_value:.4f} >= {alpha}）"

    return {
        'conversion_rate_A': p_A,
        'conversion_rate_B': p_B,
        'difference': p_B - p_A,
        'p_value': p_value,
        'z_statistic': z_stat,
        'ci_difference': (ci_low, ci_high),
        'conclusion': conclusion
    }


def exercise_2_bayesian_ab(conversions_A: int, exposures_A: int,
                          conversions_B: int, exposures_B: int,
                          alpha_prior: float = 1, beta_prior: float = 1) -> dict:
    """
    练习 2：贝叶斯 A/B 测试（共轭先验）

    实现：
    1. 使用 Beta(1,1) 作为无信息先验
    2. 计算后验分布（共轭先验有解析解）
    3. 采样 10000 次，估计 P(B > A)
    4. 计算提升幅度的中位数和 95% 可信区间

    参数:
        conversions_A: A 组转化数
        exposures_A: A 组曝光数
        conversions_B: B 组转化数
        exposures_B: B 组曝光数
        alpha_prior: Beta 先验的 alpha 参数
        beta_prior: Beta 先验的 beta 参数

    返回:
        包含后验统计量的字典
    """
    # 1. 先验：Beta(alpha_prior, beta_prior)
    prior = stats.beta(alpha_prior, beta_prior)

    # 2. 后验：共轭先验的解析解
    # Beta(alpha_prior + successes, beta_prior + failures)
    posterior_A = stats.beta(
        alpha_prior + conversions_A,
        beta_prior + exposures_A - conversions_A
    )
    posterior_B = stats.beta(
        alpha_prior + conversions_B,
        beta_prior + exposures_B - conversions_B
    )

    # 3. 采样并计算 P(B > A)
    n_samples = 10000
    samples_A = posterior_A.rvs(n_samples, random_state=42)
    samples_B = posterior_B.rvs(n_samples, random_state=43)

    prob_B_better = (samples_B > samples_A).mean()

    # 4. 提升幅度的统计量
    lift = (samples_B - samples_A) / samples_A * 100
    median_lift = np.median(lift)
    ci_low, ci_high = np.percentile(lift, [2.5, 97.5])

    return {
        'posterior_A_mean': posterior_A.mean(),
        'posterior_B_mean': posterior_B.mean(),
        'prob_B_better': prob_B_better,
        'median_lift_percent': median_lift,
        'ci_lift_percent': (ci_low, ci_high)
    }


def exercise_3_prior_comparison(conversions: int, exposures: int) -> dict:
    """
    练习 3：对比不同先验的影响

    实现：
    1. 定义三种先验：无信息、弱信息、强信息
    2. 对比后验均值、标准差
    3. 分析样本量如何影响先验的作用

    参数:
        conversions: 转化数
        exposures: 曝光数

    返回:
        包含三种先验下后验统计量的字典
    """
    # 定义三种先验
    priors = {
        '无信息': (1, 1),           # Beta(1,1) = 均匀
        '弱信息': (2, 40),          # 均值约 4.8%
        '强信息': (50, 1000),        # 均值约 4.8%
    }

    results = {}

    for name, (alpha, beta) in priors.items():
        # 计算后验
        posterior = stats.beta(
            alpha + conversions,
            beta + exposures - conversions
        )

        results[name] = {
            'prior': (alpha, beta),
            'posterior_mean': posterior.mean(),
            'posterior_std': posterior.std(),
            'ci_95': posterior.interval(0.95)
        }

    return results


def exercise_4_interpret_credible_interval(
    posterior_mean: float,
    ci_low: float,
    ci_high: float
) -> str:
    """
    练习 4：解释可信区间

    区分 95% 置信区间（CI）和 95% 可信区间（Credible Interval）

    参数:
        posterior_mean: 后验均值
        ci_low: 区间下界
        ci_high: 区间上界

    返回:
        正确的可信区间解释（字符串）
    """
    # 错误解释（置信区间）：
    # "重复抽样 100 次，95 个区间会覆盖真值"

    # 正确解释（可信区间）：
    # "给定数据和先验，参数有 95% 的概率在区间内"

    correct_interpretation = (
        f"95% 可信区间 [{ci_low:.4f}, {ci_high:.4f}] 的正确解释是：\n"
        f"  '给定观测数据和先验分布，参数有 95% 的概率落在 "
        f"[{ci_low:.4f}, {ci_high:.4f}] 之间。'\n\n"
        f"这与频率学派的置信区间不同：\n"
        f"  - 置信区间：重复抽样 100 次，95 个区间会覆盖真值\n"
        f"  - 可信区间：参数在区间内的概率是 95%\n\n"
        f"贝叶斯的可信区间更直观，因为它直接回答了决策者的问题："
        f"'参数有多大概率在这个范围内？'"
    )

    return correct_interpretation


def exercise_5_decision_rule(prob_B_better: float,
                          median_lift: float,
                          prob_threshold: float = 0.9,
                          lift_threshold: float = 5.0) -> str:
    """
    练习 5：基于贝叶斯后验的决策规则

    实现：
    1. 如果 P(B > A) >= 90% 且提升幅度 >= 5%，推荐 B
    2. 如果 70% <= P(B > A) < 90%，倾向于 B，建议继续收集数据
    3. 如果 P(B > A) < 70%，不推荐 B

    参数:
        prob_B_better: B 比 A 好的概率
        median_lift: 提升幅度的中位数
        prob_threshold: 推荐的概率阈值（默认 90%）
        lift_threshold: 推荐的提升幅度阈值（默认 5%）

    返回:
        决策建议（字符串）
    """
    if prob_B_better >= prob_threshold and median_lift >= lift_threshold:
        decision = (
            f"✅ 推荐上线 B 版本\n"
            f"  - B 比 A 好的概率: {prob_B_better:.1%} >= {prob_threshold:.0%}\n"
            f"  - 提升幅度中位数: {median_lift:.2f}% >= {lift_threshold:.1f}%"
        )
    elif prob_B_better >= 0.70:
        decision = (
            f"⚠️  倾向于 B 版本，建议继续收集数据\n"
            f"  - B 比 A 好的概率: {prob_B_better:.1%}（介于 70%-90%）\n"
            f"  - 提升幅度中位数: {median_lift:.2f}%\n"
            f"  建议：继续观察，等概率提升到 90% 再决策"
        )
    else:
        decision = (
            f"❌ 不推荐 B 版本\n"
            f"  - B 比 A 好的概率: {prob_B_better:.1%} < 70%\n"
            f"  - 证据不足，建议保持 A 或重新设计"
        )

    return decision


def main() -> None:
    """运行所有练习并打印结果"""
    print("=" * 60)
    print("Week 14 作业参考实现")
    print("=" * 60)

    # 练习 1：频率学派 A/B 测试
    print("\n" + "=" * 60)
    print("练习 1：频率学派 A/B 测试")
    print("=" * 60)

    result1 = exercise_1_frequentist_ab(52, 1000, 58, 1000)
    print(f"\nA 组转化率: {result1['conversion_rate_A']:.3f}")
    print(f"B 组转化率: {result1['conversion_rate_B']:.3f}")
    print(f"差异: {result1['difference']:.4f}")
    print(f"p 值: {result1['p_value']:.4f}")
    print(f"95% CI（差异）: [{result1['ci_difference'][0]:.4f}, {result1['ci_difference'][1]:.4f}]")
    print(f"\n结论: {result1['conclusion']}")

    # 练习 2：贝叶斯 A/B 测试
    print("\n" + "=" * 60)
    print("练习 2：贝叶斯 A/B 测试")
    print("=" * 60)

    result2 = exercise_2_bayesian_ab(52, 1000, 58, 1000)
    print(f"\nA 后验均值: {result2['posterior_A_mean']:.4f}")
    print(f"B 后验均值: {result2['posterior_B_mean']:.4f}")
    print(f"P(B > A): {result2['prob_B_better']:.1%}")
    print(f"提升幅度中位数: {result2['median_lift_percent']:.2f}%")
    print(f"95% CI（提升）: [{result2['ci_lift_percent'][0]:.2f}%, {result2['ci_lift_percent'][1]:.2f}%]")

    # 练习 3：先验对比
    print("\n" + "=" * 60)
    print("练习 3：对比不同先验")
    print("=" * 60)

    result3 = exercise_3_prior_comparison(58, 1000)
    print(f"\n{'先验类型':<12} {'后验均值':>10} {'后验标准差':>12}")
    print("-" * 40)
    for name, stats in result3.items():
        print(f"{name:<12} {stats['posterior_mean']:>10.4f} {stats['posterior_std']:>12.4f}")

    # 练习 4：解释可信区间
    print("\n" + "=" * 60)
    print("练习 4：解释可信区间")
    print("=" * 60)

    interpretation = exercise_4_interpret_credible_interval(
        posterior_mean=0.058,
        ci_low=0.044,
        ci_high=0.072
    )
    print(f"\n{interpretation}")

    # 练习 5：决策规则
    print("\n" + "=" * 60)
    print("练习 5：贝叶斯决策")
    print("=" * 60)

    # 测试三种场景
    scenarios = [
        (0.92, 8.5, "推荐场景"),
        (0.75, 3.2, "继续观察场景"),
        (0.55, 1.8, "不推荐场景")
    ]

    for prob, lift, desc in scenarios:
        print(f"\n{desc}: P(B>A)={prob:.1%}, 提升={lift:.1f}%")
        decision = exercise_5_decision_rule(prob, lift)
        print(decision)

    print("\n" + "=" * 60)
    print("所有练习完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()

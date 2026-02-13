"""
示例：贝叶斯 A/B 测试——共轭先验解析解

运行方式：python3 chapters/week_14/examples/03_bayesian_ab.py
预期输出：B 比 A 好的概率、提升幅度的中位数和 95% 可信区间
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def bayesian_ab_test(conversions_A: int, exposures_A: int,
                    conversions_B: int, exposures_B: int,
                    alpha_prior: float = 1, beta_prior: float = 1,
                    n_samples: int = 10000,
                    random_seed: int = 42) -> dict:
    """
    贝叶斯 A/B 测试（共轭先验解析解）

    使用 Beta-Binomial 共轭先验，后验有解析解

    参数:
        conversions_A: A 版本转化数
        exposures_A: A 版本曝光数
        conversions_B: B 版本转化数
        exposures_B: B 版本曝光数
        alpha_prior: Beta 先验的 alpha 参数（默认 1 = 无信息先验）
        beta_prior: Beta 先验的 beta 参数（默认 1 = 无信息先验）
        n_samples: 后验采样数量
        random_seed: 随机种子

    返回:
        包含后验统计量的字典
    """
    np.random.seed(random_seed)

    # 计算后验（Beta-Binomial 共轭）
    # 后验：Beta(alpha_prior + conversions, beta_prior + failures)
    posterior_A = stats.beta(
        alpha_prior + conversions_A,
        beta_prior + exposures_A - conversions_A
    )
    posterior_B = stats.beta(
        alpha_prior + conversions_B,
        beta_prior + exposures_B - conversions_B
    )

    # 从后验分布采样
    samples_A = posterior_A.rvs(n_samples)
    samples_B = posterior_B.rvs(n_samples)

    # 计算"B 比 A 好"的概率
    prob_B_better = (samples_B > samples_A).mean()

    # 计算提升幅度的分布
    lift = (samples_B - samples_A) / samples_A * 100

    # 95% 可信区间
    ci_low = np.percentile(lift, 2.5)
    ci_high = np.percentile(lift, 97.5)
    median_lift = np.median(lift)

    return {
        'posterior_A_mean': posterior_A.mean(),
        'posterior_B_mean': posterior_B.mean(),
        'prob_B_better': prob_B_better,
        'median_lift_percent': median_lift,
        'ci_low_percent': ci_low,
        'ci_high_percent': ci_high,
        'samples_A': samples_A,
        'samples_B': samples_B,
        'lift_samples': lift
    }


# 坏例子：只看均值，忽略不确定性
def bad_bayesian_decision(posterior_A_mean: float, posterior_B_mean: float) -> str:
    """
    反例：只用后验均值做决策

    问题：
    1. 忽略了后验分布的不确定性
    2. 无法评估决策风险
    3. 丢失了贝叶斯方法的核心优势
    """
    if posterior_B_mean > posterior_A_mean:
        return f"B 更好（均值 {posterior_B_mean:.3f} > {posterior_A_mean:.3f}）"
    else:
        return f"A 更好（均值 {posterior_A_mean:.3f} > {posterior_B_mean:.3f}）"


def compare_frequentist_vs_bayesian():
    """对比频率学派和贝叶斯学派的结果"""
    # 示例数据
    conversions_A, exposures_A = 52, 1000
    conversions_B, exposures_B = 58, 1000

    print("=" * 60)
    print("频率学派 vs 贝叶斯学派对比")
    print("=" * 60)
    print(f"\n数据：A={conversions_A}/{exposures_A}, B={conversions_B}/{exposures_B}")
    print()

    # 频率学派结果
    p_A = conversions_A / exposures_A
    p_B = conversions_B / exposures_B
    relative_lift = (p_B - p_A) / p_A * 100

    print("【频率学派】")
    print(f"  转化率 A: {p_A:.3f}")
    print(f"  转化率 B: {p_B:.3f}")
    print(f"  提升幅度: {relative_lift:.2f}%")
    print(f"  结论: p值需要额外计算，且只能给出'显著/不显著'")
    print()

    # 贝叶斯学派结果
    result = bayesian_ab_test(conversions_A, exposures_A,
                            conversions_B, exposures_B)

    print("【贝叶斯学派】")
    print(f"  A 后验均值: {result['posterior_A_mean']:.3f}")
    print(f"  B 后验均值: {result['posterior_B_mean']:.3f}")
    print(f"  B 比 A 好的概率: {result['prob_B_better']:.1%}")
    print(f"  提升幅度中位数: {result['median_lift_percent']:.2f}%")
    print(f"  95% 可信区间: [{result['ci_low_percent']:.2f}%, {result['ci_high_percent']:.2f}%]")
    print()

    # 解释差异
    print("【关键差异】")
    print("  频率学派：")
    print("    - 输出：p 值、置信区间")
    print("    - 解释：'重复抽样下，95% 的区间会覆盖真值'")
    print("    - 无法直接回答'B 比 A 好的概率'")
    print()
    print("  贝叶斯学派：")
    print("    - 输出：后验分布、可信区间")
    print("    - 解释：'参数有 95% 的概率在区间内'")
    print(f"    - 直接回答：B 有 {result['prob_B_better']:.1%} 的概率更好")
    print()


def decision_with_probability(result: dict, threshold: float = 0.9) -> str:
    """
    基于概率陈述做决策

    这是贝叶斯方法的核心优势：决策者可以用业务阈值

    参数:
        result: bayesian_ab_test 的返回结果
        threshold: 决策阈值（如 90%）

    返回:
        决策建议
    """
    prob = result['prob_B_better']
    median_lift = result['median_lift_percent']

    if prob >= threshold:
        return (f"✅ 推荐 B：B 有 {prob:.1%} 的概率更好，"
                f"提升中位数 {median_lift:.2f}%")
    elif prob >= 0.6:
        return (f"⚠️  倾向 B：B 有 {prob:.1%} 的概率更好，"
                f"但不确定度较高，建议继续收集数据")
    else:
        return (f"❌ 不推荐 B：B 只有 {prob:.1%} 的概率更好，"
                f"建议保持 A 或重新设计")


def main() -> None:
    # 对比频率学派和贝叶斯学派
    compare_frequentist_vs_bayesian()

    # 示例：贝叶斯决策
    conversions_A, exposures_A = 52, 1000
    conversions_B, exposures_B = 58, 1000

    result = bayesian_ab_test(conversions_A, exposures_A,
                            conversions_B, exposures_B)

    print("=" * 60)
    print("贝叶斯决策示例")
    print("=" * 60)

    # 尝试不同的决策阈值
    for threshold in [0.5, 0.75, 0.90, 0.95]:
        decision = decision_with_probability(result, threshold)
        print(f"阈值 {threshold:.0%}: {decision}")

    print()
    print("注意：贝叶斯方法让决策者可以根据业务风险调整阈值！")
    print("      频率学派的 p<0.05 是固定的，无法灵活调整。")


if __name__ == "__main__":
    main()

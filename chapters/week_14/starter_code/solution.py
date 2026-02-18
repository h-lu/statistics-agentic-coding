"""
Week 14 作业参考实现

本文件提供了作业的参考实现，供学生在遇到困难时查看。
包含基础作业的完整代码，但不包含进阶和挑战部分的答案。

注意：建议学生先自己尝试完成作业，仅在遇到困难时参考此文件。
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from typing import Tuple, Dict


# ===== 练习题 1：贝叶斯定理计算 =====

def bayes_theorem(prior: float, sensitivity: float, false_positive: float) -> float:
    """
    练习 1：实现贝叶斯定理

    计算 P(患病|检测阳性) = P(阳性|患病) × P(患病) / P(阳性)

    参数:
        prior: P(患病)，先验概率（患病率）
        sensitivity: P(检测阳性|患病)，灵敏度/真阳性率
        false_positive: P(检测阳性|健康)，假阳性率

    返回:
        P(患病|检测阳性)，后验概率
    """
    # P(检测阳性) = P(阳性|患病)×P(患病) + P(阳性|健康)×P(健康)
    p_positive = sensitivity * prior + false_positive * (1 - prior)

    # 贝叶斯定理
    posterior = (sensitivity * prior) / p_positive

    return posterior


# ===== 练习题 2：Beta-Binomial 后验计算 =====

def beta_binomial_posterior(
    alpha_prior: float,
    beta_prior: float,
    successes: int,
    failures: int
) -> Tuple[float, float, float]:
    """
    练习 2：计算 Beta-Binomial 后验分布

    后验参数：alpha_post = alpha_prior + successes
              beta_post = beta_prior + failures

    参数:
        alpha_prior: 先验 Beta 分布的 alpha 参数
        beta_prior: 先验 Beta 分布的 beta 参数
        successes: 成功次数（如流失客户数）
        failures: 失败次数（如未流失客户数）

    返回:
        (后验均值, 95%可信区间下界, 95%可信区间上界)
    """
    # 后验参数（共轭先验的优美性质）
    alpha_post = alpha_prior + successes
    beta_post = beta_prior + failures

    # 后验均值
    posterior_mean = alpha_post / (alpha_post + beta_post)

    # 95% 可信区间
    ci_lower, ci_upper = stats.beta.interval(0.95, alpha_post, beta_post)

    return posterior_mean, ci_lower, ci_upper


# ===== 练习题 3：先验敏感性分析 =====

def prior_sensitivity_analysis(
    n: int,
    successes: int,
    priors: Dict[str, Tuple[float, float]]
) -> Dict[str, Dict[str, float]]:
    """
    练习 3：先验敏感性分析

    比较不同先验下的后验分布，评估结论对先验的依赖程度

    参数:
        n: 总样本数
        successes: 成功次数
        priors: 字典 {先验名称: (alpha, beta)}

    返回:
        包含每个先验后验统计的字典
    """
    failures = n - successes
    results = {}

    for name, (alpha, beta) in priors.items():
        post_mean, ci_low, ci_high = beta_binomial_posterior(
            alpha, beta, successes, failures
        )

        results[name] = {
            'prior_mean': alpha / (alpha + beta),
            'posterior_mean': post_mean,
            'ci_lower': ci_low,
            'ci_upper': ci_high
        }

    return results


# ===== 练习题 4：判断敏感性 =====

def is_sensitive(results: Dict[str, Dict[str, float]],
                threshold: float = 0.02) -> bool:
    """
    练习 4：判断后验均值是否对先验敏感

    参数:
        results: prior_sensitivity_analysis 的返回结果
        threshold: 判断阈值（默认 0.02 = 2%）

    返回:
        True 表示敏感，False 表示不敏感
    """
    means = [r['posterior_mean'] for r in results.values()]
    mean_range = max(means) - min(means)

    return mean_range >= threshold


# ===== 示例：流失率分析 =====

def analyze_churn_rate() -> None:
    """
    示例：分析流失率的贝叶斯估计

    场景：公司有 1000 个客户，其中 180 个流失
          不同部门对流失率有不同的先验看法
    """
    print("=" * 50)
    print("流失率贝叶斯分析示例")
    print("=" * 50)

    # 数据
    n = 1000
    churned = 180

    # 不同部门的先验
    priors = {
        '无信息': (1, 1),        # Beta(1,1) 均匀分布
        '弱信息': (15, 85),      # Beta(15,85) 均值 15%
        '市场部': (180, 820),    # 基于历史数据
        '产品部': (5, 15),       # 基于近期趋势
    }

    print(f"\n数据：{churned}/{n} = {churned/n:.1%}")

    # 先验敏感性分析
    results = prior_sensitivity_analysis(n, churned, priors)

    print(f"\n{'先验':<12} {'先验均值':<12} {'后验均值':<12} {'95% CI'}")
    print("-" * 55)

    for name, res in results.items():
        ci_str = f"[{res['ci_lower']:.1%}, {res['ci_upper']:.1%}]"
        print(f"{name:<12} {res['prior_mean']:>10.1%}   "
             f"{res['posterior_mean']:>10.1%}   {ci_str}")

    # 判断敏感性
    sensitive = is_sensitive(results)
    print(f"\n结论对先验{'敏感' if sensitive else '不敏感'}")


# ===== 主函数 =====

def main() -> None:
    """运行所有示例"""
    print("\n" + "=" * 50)
    print("Week 14 作业参考实现")
    print("=" * 50)

    # 示例 1：贝叶斯定理
    print("\n### 示例 1：医疗检测")
    prior = 0.01       # 患病率 1%
    sensitivity = 0.99 # 灵敏度 99%
    false_positive = 0.05  # 假阳性率 5%

    posterior = bayes_theorem(prior, sensitivity, false_positive)
    print(f"P(患病|阳性) = {posterior:.2%}")

    # 示例 2：流失率分析
    print("\n### 示例 2：流失率分析")
    analyze_churn_rate()


if __name__ == "__main__":
    main()

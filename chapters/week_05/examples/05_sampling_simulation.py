#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
示例：概率与模拟——从贝叶斯定理到中心极限定理

本例演示：
1. 用模拟验证贝叶斯定理（医疗检测假阳性问题）
2. 常见概率分布生成与可视化
3. 中心极限定理模拟实验
4. Bootstrap 抽样分布估计

运行方式：python3 chapters/week_05/examples/05_sampling_simulation.py
预期输出：多个可视化图表 + stdout 统计报告
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# ============================================================================
# 第一部分：贝叶斯定理模拟验证
# ============================================================================

def simulate_disease_test(
    population: int = 100000,
    prevalence: float = 0.01,
    sensitivity: float = 0.99,
    specificity: float = 0.99,
    seed: int = 42
) -> dict:
    """
    模拟疾病检测实验，验证贝叶斯计算

    参数：
        population: 模拟人口数量
        prevalence: 基础发病率（先验概率）
        sensitivity: 敏感性 = P(阳性|患病)
        specificity: 特异性 = P(阴性|健康)
        seed: 随机种子

    返回：
        dict: 包含各种统计量的字典
    """
    rng = np.random.default_rng(seed)

    # 第一步：生成真实的患病状态
    true_status = rng.random(population) < prevalence

    # 第二步：根据真实状态生成检测结果
    # 患病的人：sensitivity 概率检测阳性
    # 健康的人：(1 - specificity) 概率误报阳性
    test_result = np.where(
        true_status,
        rng.random(population) < sensitivity,
        rng.random(population) < (1 - specificity)
    )

    # 第三步：计算条件概率
    positive_mask = test_result
    total_positives = positive_mask.sum()
    true_positives = (true_status & positive_mask).sum()
    false_positives = (~true_status & positive_mask).sum()

    p_sick_given_positive = true_positives / total_positives if total_positives > 0 else 0

    # 理论值计算（贝叶斯公式）
    p_sick = prevalence
    p_positive_given_sick = sensitivity
    p_positive_given_healthy = 1 - specificity
    p_positive = (p_positive_given_sick * p_sick +
                 p_positive_given_healthy * (1 - p_sick))
    p_sick_given_positive_theory = (p_positive_given_sick * p_sick) / p_positive

    return {
        'population': population,
        'true_sick': true_status.sum(),
        'total_positives': total_positives,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'p_sick_given_positive_sim': p_sick_given_positive,
        'p_sick_given_positive_theory': p_sick_given_positive_theory
    }


def print_bayes_report(results: dict) -> None:
    """打印贝叶斯定理报告"""
    print("\n" + "=" * 70)
    print("贝叶斯定理验证：医疗检测假阳性问题")
    print("=" * 70)

    print(f"\n模拟人口：{results['population']:,} 人")
    print(f"真实患病：{results['true_sick']:,} 人 ({results['true_sick']/results['population']:.1%})")
    print(f"检测阳性：{results['total_positives']:,} 人")
    print(f"  其中真阳性：{results['true_positives']:,} 人")
    print(f"  其中假阳性：{results['false_positives']:,} 人")

    print(f"\n关键问题：检测阳性的情况下，真正患病的概率是多少？")
    print(f"  模拟结果：P(患病|阳性) = {results['p_sick_given_positive_sim']:.1%}")
    print(f"  理论计算：P(患病|阳性) = {results['p_sick_given_positive_theory']:.1%}")

    print(f"\n小北的困惑：")
    print(f"  '检测准确率 99%，为什么阳性后真正得病的概率只有 50%？'")

    print(f"\n老潘的解释：")
    print(f"  '因为健康人基数太大，即使 1% 误报也会产生大量假阳性。'")
    print(f"  '假阳性数 ≈ 真阳性数时，P(患病|阳性) ≈ 50%。'")


# ============================================================================
# 第二部分：常见概率分布
# ============================================================================

def plot_common_distributions() -> dict:
    """
    绘制常见概率分布：正态、二项、泊松

    返回：
        dict: 各分布的参数和统计量
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 正态分布
    mu, sigma = 100, 15
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 200)
    y = stats.norm.pdf(x, mu, sigma)

    axes[0, 0].plot(x, y, 'b-', lw=2, label=f'N({mu}, {sigma}²)')
    axes[0, 0].fill_between(x, y, alpha=0.3)
    axes[0, 0].axvline(mu, color='r', linestyle='--', label=f'μ={mu}')
    axes[0, 0].axvline(mu + sigma, color='orange', linestyle='--', alpha=0.7, label='±σ')
    axes[0, 0].axvline(mu - sigma, color='orange', linestyle='--', alpha=0.7)
    axes[0, 0].set_xlabel('数值')
    axes[0, 0].set_ylabel('概率密度')
    axes[0, 0].set_title('正态分布：对称的钟形曲线')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 计算正态分布的 3σ 原则
    within_1sigma = stats.norm.cdf(mu + sigma, mu, sigma) - stats.norm.cdf(mu - sigma, mu, sigma)
    within_2sigma = stats.norm.cdf(mu + 2*sigma, mu, sigma) - stats.norm.cdf(mu - 2*sigma, mu, sigma)
    within_3sigma = stats.norm.cdf(mu + 3*sigma, mu, sigma) - stats.norm.cdf(mu - 3*sigma, mu, sigma)

    # 2. 二项分布
    n, p = 100, 0.05
    x_binom = np.arange(0, n + 1)
    pmf = stats.binom.pmf(x_binom, n, p)

    axes[0, 1].bar(x_binom, pmf, alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(n * p, color='r', linestyle='--', label=f'期望={n*p}')
    axes[0, 1].set_xlabel('成功次数')
    axes[0, 1].set_ylabel('概率')
    axes[0, 1].set_title(f'二项分布 B({n}, {p})：100 次试验中的成功次数')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. 泊松分布
    lam = 3
    x_poisson = np.arange(0, 15)
    pmf_poisson = stats.poisson.pmf(x_poisson, lam)

    axes[1, 0].bar(x_poisson, pmf_poisson, alpha=0.7, color='green', edgecolor='black')
    axes[1, 0].axvline(lam, color='r', linestyle='--', label=f'λ={lam}')
    axes[1, 0].set_xlabel('事件发生次数')
    axes[1, 0].set_ylabel('概率')
    axes[1, 0].set_title(f'泊松分布 Poisson(λ={lam})：单位时间内的稀有事件')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. 对比图：不同参数的正态分布
    x_compare = np.linspace(-10, 10, 200)
    axes[1, 1].plot(x_compare, stats.norm.pdf(x_compare, 0, 1), 'b-', lw=2, label='N(0, 1)')
    axes[1, 1].plot(x_compare, stats.norm.pdf(x_compare, 0, 2), 'g-', lw=2, label='N(0, 4)')
    axes[1, 1].plot(x_compare, stats.norm.pdf(x_compare, 3, 1), 'r-', lw=2, label='N(3, 1)')
    axes[1, 1].set_xlabel('数值')
    axes[1, 1].set_ylabel('概率密度')
    axes[1, 1].set_title('均值和标准差的影响')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('common_distributions.png', dpi=150)
    print(f"\n[图表已保存] common_distributions.png")
    plt.close()

    return {
        'normal': {
            'mu': mu,
            'sigma': sigma,
            'within_1sigma': within_1sigma,
            'within_2sigma': within_2sigma,
            'within_3sigma': within_3sigma
        },
        'binomial': {'n': n, 'p': p, 'mean': n * p},
        'poisson': {'lambda': lam, 'mean': lam, 'var': lam}
    }


def print_distribution_report(dist_params: dict) -> None:
    """打印分布统计报告"""
    print("\n" + "=" * 70)
    print("常见概率分布统计")
    print("=" * 70)

    print("\n【正态分布】")
    normal = dist_params['normal']
    print(f"  参数：N({normal['mu']}, {normal['sigma']}²)")
    print(f"  68-95-99.7 原则：")
    print(f"    落在 μ±1σ 的概率：{normal['within_1sigma']:.2%}")
    print(f"    落在 μ±2σ 的概率：{normal['within_2sigma']:.2%}")
    print(f"    落在 μ±3σ 的概率：{normal['within_3sigma']:.2%}")

    print("\n【二项分布】")
    binom = dist_params['binomial']
    print(f"  参数：B({binom['n']}, {binom['p']})")
    print(f"  期望成功次数：{binom['mean']}")
    print(f"  标准差：{np.sqrt(binom['n'] * binom['p'] * (1 - binom['p'])):.2f}")

    print("\n【泊松分布】")
    pois = dist_params['poisson']
    print(f"  参数：Poisson(λ={pois['lambda']})")
    print(f"  期望：{pois['mean']}")
    print(f"  方差：{pois['var']} （泊松分布的期望 = 方差）")


# ============================================================================
# 第三部分：中心极限定理模拟
# ============================================================================

def demonstrate_clt(
    population_size: int = 100000,
    sample_sizes: list[int] = [5, 30, 100],
    n_simulations: int = 10000,
    seed: int = 42
) -> dict:
    """
    模拟中心极限定理：从任意分布抽样，观察样本均值分布

    使用严重右偏的指数分布作为总体
    """
    rng = np.random.default_rng(seed)

    # 原始数据：严重右偏的指数分布
    population = rng.exponential(scale=10, size=population_size)
    pop_mean = np.mean(population)
    pop_std = np.std(population, ddof=1)

    # 为每个样本量生成样本均值分布
    results = {}
    for n in sample_sizes:
        sample_means = np.array([
            np.mean(rng.choice(population, n, replace=False))
            for _ in range(n_simulations)
        ])
        results[n] = {
            'means': sample_means,
            'mean_of_means': np.mean(sample_means),
            'std_of_means': np.std(sample_means, ddof=1),
            'theoretical_se': pop_std / np.sqrt(n)
        }

    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 原始分布
    axes[0, 0].hist(population, bins=100, density=True, alpha=0.7, color='orange')
    axes[0, 0].set_title('原始分布（严重右偏的指数分布）')
    axes[0, 0].set_xlabel('数值')
    axes[0, 0].set_ylabel('密度')

    # 样本均值分布
    for idx, n in enumerate(sample_sizes, 1):
        ax = axes[(idx) // 2, (idx) % 2]
        means = results[n]['means']
        ax.hist(means, bins=50, density=True, alpha=0.7, color='steelblue')

        # 叠加理论正态分布
        x = np.linspace(means.min(), means.max(), 200)
        theoretical_norm = stats.norm.pdf(
            x, results[n]['mean_of_means'], results[n]['std_of_means']
        )
        ax.plot(x, theoretical_norm, 'r-', lw=2, label='理论正态分布')

        ax.axvline(pop_mean, color='g', linestyle='--', label=f'总体均值={pop_mean:.1f}')
        ax.set_title(f'样本均值分布（n={n}）')
        ax.set_xlabel('样本均值')
        ax.set_ylabel('密度')
        ax.legend()

    plt.tight_layout()
    plt.savefig('clt_demo.png', dpi=150)
    print(f"\n[图表已保存] clt_demo.png")
    plt.close()

    return {
        'population_mean': pop_mean,
        'population_std': pop_std,
        'sample_results': results
    }


def print_clt_report(clt_results: dict) -> None:
    """打印 CLT 模拟报告"""
    print("\n" + "=" * 70)
    print("中心极限定理模拟")
    print("=" * 70)

    pop_mean = clt_results['population_mean']
    pop_std = clt_results['population_std']

    print(f"\n总体分布：指数分布")
    print(f"  总体均值：{pop_mean:.2f}")
    print(f"  总体标准差：{pop_std:.2f}")

    print(f"\n样本均值分布统计：")
    print(f"{'样本量 n':<10} {'均值':<12} {'标准差(实际)':<15} {'标准误(理论)':<15}")
    print("-" * 70)

    for n, res in clt_results['sample_results'].items():
        actual_se = res['std_of_means']
        theoretical_se = res['theoretical_se']
        print(f"{n:<10} {res['mean_of_means']:<12.3f} {actual_se:<15.3f} {theoretical_se:<15.3f}")

    print(f"\n阿码的追问：")
    print(f"  '为什么样本量越大，样本均值的分布越接近正态？'")

    print(f"\n老潘的解释：")
    print(f"  '因为每个样本里的随机误差会相互抵消。'")
    print(f"  '样本量越大，中心极限定理的作用越强。'")
    print(f"  '标准误 SE = σ/√n，所以 n 越大，均值越稳定。'")


# ============================================================================
# 第四部分：Bootstrap 抽样分布
# ============================================================================

def bootstrap_distribution(
    sample: np.ndarray,
    statistic: str = 'mean',
    n_bootstrap: int = 10000,
    seed: int = 42
) -> dict:
    """
    Bootstrap 估计统计量的抽样分布

    参数：
        sample: 原始样本
        statistic: 统计量类型 ('mean', 'median', 'std')
        n_bootstrap: Bootstrap 重复次数
        seed: 随机种子

    返回：
        dict: Bootstrap 分布和置信区间
    """
    rng = np.random.default_rng(seed)
    n = len(sample)
    boot_stats = []

    for _ in range(n_bootstrap):
        # 有放回重采样
        resample = rng.choice(sample, size=n, replace=True)

        if statistic == 'mean':
            boot_stats.append(np.mean(resample))
        elif statistic == 'median':
            boot_stats.append(np.median(resample))
        elif statistic == 'std':
            boot_stats.append(np.std(resample, ddof=1))

    boot_stats = np.array(boot_stats)

    # 计算 95% 置信区间（percentile 方法）
    ci_low = np.percentile(boot_stats, 2.5)
    ci_high = np.percentile(boot_stats, 97.5)

    return {
        'bootstrap_distribution': boot_stats,
        'observed': sample.mean() if statistic == 'mean' else
                    (np.median(sample) if statistic == 'median' else np.std(sample, ddof=1)),
        'ci_low': ci_low,
        'ci_high': ci_high,
        'se': boot_stats.std(ddof=1)
    }


def demonstrate_bootstrap() -> None:
    """演示 Bootstrap 用于估计均值和中位数的抽样分布"""
    # 生成右偏样本
    np.random.seed(42)
    sample = np.random.lognormal(7, 0.8, 100)

    # Bootstrap 均值
    boot_mean = bootstrap_distribution(sample, 'mean', n_bootstrap=10000)

    # Bootstrap 中位数
    boot_median = bootstrap_distribution(sample, 'median', n_bootstrap=10000)

    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 均值分布
    axes[0].hist(boot_mean['bootstrap_distribution'], bins=50, density=True, alpha=0.7)
    axes[0].axvline(boot_mean['observed'], color='r', linestyle='--',
                    label=f"观察均值={boot_mean['observed']:.0f}")
    axes[0].axvline(boot_mean['ci_low'], color='orange', linestyle='--', alpha=0.7)
    axes[0].axvline(boot_mean['ci_high'], color='orange', linestyle='--', alpha=0.7,
                    label='95% CI')
    axes[0].set_xlabel('样本均值')
    axes[0].set_ylabel('密度')
    axes[0].set_title('Bootstrap 分布：样本均值')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 中位数分布
    axes[1].hist(boot_median['bootstrap_distribution'], bins=50, density=True, alpha=0.7, color='green')
    axes[1].axvline(boot_median['observed'], color='r', linestyle='--',
                    label=f"观察中位数={boot_median['observed']:.0f}")
    axes[1].axvline(boot_median['ci_low'], color='orange', linestyle='--', alpha=0.7)
    axes[1].axvline(boot_median['ci_high'], color='orange', linestyle='--', alpha=0.7,
                    label='95% CI')
    axes[1].set_xlabel('样本中位数')
    axes[1].set_ylabel('密度')
    axes[1].set_title('Bootstrap 分布：样本中位数')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('bootstrap_sampling_distribution.png', dpi=150)
    print(f"\n[图表已保存] bootstrap_sampling_distribution.png")
    plt.close()

    # 打印报告
    print("\n" + "=" * 70)
    print("Bootstrap 抽样分布估计")
    print("=" * 70)

    print(f"\n【均值】")
    print(f"  点估计：{boot_mean['observed']:.0f}")
    print(f"  95% CI：[{boot_mean['ci_low']:.0f}, {boot_mean['ci_high']:.0f}]")
    print(f"  标准误：{boot_mean['se']:.0f}")

    print(f"\n【中位数】")
    print(f"  点估计：{boot_median['observed']:.0f}")
    print(f"  95% CI：[{boot_median['ci_low']:.0f}, {boot_median['ci_high']:.0f}]")
    print(f"  标准误：{boot_median['se']:.0f}")

    print(f"\n关键发现：")
    print(f"  '均值分布的宽度比中位数窄 → 均值是更稳定的统计量'")
    print(f"  '但如果数据有极端异常值，中位数会更稳健'")


# ============================================================================
# 主函数
# ============================================================================

def main() -> None:
    """主函数：运行所有示例"""
    print("=" * 70)
    print("概率与模拟：从贝叶斯定理到中心极限定理")
    print("=" * 70)

    # 1. 贝叶斯定理验证
    print("\n" + "=" * 70)
    print("第一部分：贝叶斯定理验证（医疗检测假阳性问题）")
    print("=" * 70)

    bayes_results = simulate_disease_test(
        population=100000,
        prevalence=0.01,
        sensitivity=0.99,
        specificity=0.99,
        seed=42
    )
    print_bayes_report(bayes_results)

    # 2. 常见分布
    print("\n" + "=" * 70)
    print("第二部分：常见概率分布")
    print("=" * 70)

    dist_params = plot_common_distributions()
    print_distribution_report(dist_params)

    # 3. 中心极限定理
    print("\n" + "=" * 70)
    print("第三部分：中心极限定理模拟")
    print("=" * 70)

    clt_results = demonstrate_clt()
    print_clt_report(clt_results)

    # 4. Bootstrap
    print("\n" + "=" * 70)
    print("第四部分：Bootstrap 抽样分布估计")
    print("=" * 70)

    demonstrate_bootstrap()

    # 总结
    print("\n" + "=" * 70)
    print("本周核心要点")
    print("=" * 70)
    print("1. 贝叶斯定理：P(A|B) ≠ P(B|A)，医疗检测假阳性是经典例子")
    print("2. 常见分布：正态（对称）、二项（计数）、泊松（稀有事件）")
    print("3. 中心极限定理：样本均值的分布在大样本时近似正态")
    print("4. Bootstrap：用重采样估计任意统计量的抽样分布")
    print("\n从'算一个数'到'给一个范围'——这就是统计推断的核心跳跃。")
    print("=" * 70)


if __name__ == "__main__":
    main()

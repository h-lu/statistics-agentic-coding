#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立示例 2：常见概率分布生成与可视化

本示例独立演示三种常见分布（正态、二项、泊松）的
生成、参数估计和可视化。

运行方式：python3 chapters/week_05/examples/05_distributions_demo.py
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

OUTPUT_DIR = Path(__file__).parent.parent.parent / "output" / "week_05"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def demo_normal_distribution():
    """演示正态分布"""
    print("\n" + "=" * 70)
    print("正态分布：N(μ, σ²)")
    print("=" * 70)

    mu = 100
    sigma = 15

    # 生成数据
    np.random.seed(42)
    data = np.random.normal(mu, sigma, 10000)

    # 理论计算
    within_1s = stats.norm.cdf(mu + sigma, mu, sigma) - stats.norm.cdf(mu - sigma, mu, sigma)
    within_2s = stats.norm.cdf(mu + 2*sigma, mu, sigma) - stats.norm.cdf(mu - 2*sigma, mu, sigma)
    within_3s = stats.norm.cdf(mu + 3*sigma, mu, sigma) - stats.norm.cdf(mu - 3*sigma, mu, sigma)

    print(f"\n参数：μ = {mu}, σ = {sigma}")
    print(f"\n68-95-99.7 原则：")
    print(f"  - 落在 μ ± 1σ 的概率：{within_1s:.2%}")
    print(f"  - 落在 μ ± 2σ 的概率：{within_2s:.2%}")
    print(f"  - 落在 μ ± 3σ 的概率：{within_3s:.2%}")

    # 可视化
    fig, ax = plt.subplots(figsize=(10, 6))

    # 直方图
    ax.hist(data, bins=50, density=True, alpha=0.7, color='steelblue', label='模拟数据')

    # 理论曲线
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 200)
    y = stats.norm.pdf(x, mu, sigma)
    ax.plot(x, y, 'r-', lw=2, label=f'理论 N({mu}, {sigma}²)')

    # 标记区域
    for i, color in enumerate(['g', 'orange', 'r'], 1):
        ax.axvline(mu - i*sigma, color=color, linestyle='--', alpha=0.5)
        ax.axvline(mu + i*sigma, color=color, linestyle='--', alpha=0.5)

    ax.axvline(mu, color='k', linestyle='-', lw=1, label=f'μ={mu}')

    ax.set_xlabel('数值')
    ax.set_ylabel('概率密度')
    ax.set_title('正态分布：对称的钟形曲线')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'demo_normal_distribution.png', dpi=150)
    print(f"\n[图表已保存] /tmp/demo_normal_distribution.png")
    plt.close()

    return {'mu': mu, 'sigma': sigma}


def demo_binomial_distribution():
    """演示二项分布"""
    print("\n" + "=" * 70)
    print("二项分布：B(n, p)")
    print("=" * 70)

    n = 100  # 试验次数
    p = 0.05  # 单次成功概率

    # 生成数据
    np.random.seed(42)
    data = np.random.binomial(n, p, 10000)

    # 理论值
    mean = n * p
    variance = n * p * (1 - p)
    std = np.sqrt(variance)

    print(f"\n参数：n = {n}, p = {p}")
    print(f"\n理论值：")
    print(f"  - 期望 E[X] = n × p = {mean}")
    print(f"  - 方差 Var(X) = n × p × (1-p) = {variance:.2f}")
    print(f"  - 标准差 σ = {std:.2f}")

    print(f"\n模拟统计：")
    print(f"  - 样本均值 = {data.mean():.2f}")
    print(f"  - 样本标准差 = {data.std(ddof=1):.2f}")

    # 可视化
    fig, ax = plt.subplots(figsize=(10, 6))

    # 直方图
    ax.hist(data, bins=np.arange(-0.5, n+1.5, 1), density=True,
            alpha=0.7, edgecolor='black', label='模拟数据')

    # 理论 PMF
    x = np.arange(0, n+1)
    pmf = stats.binom.pmf(x, n, p)
    ax.plot(x, pmf, 'ro-', markersize=4, label='理论二项分布')

    ax.axvline(mean, color='r', linestyle='--', lw=2, label=f'期望={mean}')
    ax.axvline(mean + 2*std, color='orange', linestyle='--', alpha=0.7)
    ax.axvline(mean - 2*std, color='orange', linestyle='--', alpha=0.7, label='±2σ')

    ax.set_xlabel('成功次数')
    ax.set_ylabel('概率')
    ax.set_title(f'二项分布 B({n}, {p})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'demo_binomial_distribution.png', dpi=150)
    print(f"\n[图表已保存] /tmp/demo_binomial_distribution.png")
    plt.close()

    return {'n': n, 'p': p, 'mean': mean, 'std': std}


def demo_poisson_distribution():
    """演示泊松分布"""
    print("\n" + "=" * 70)
    print("泊松分布：Poisson(λ)")
    print("=" * 70)

    lam = 3  # 平均发生率

    # 生成数据
    np.random.seed(42)
    data = np.random.poisson(lam, 10000)

    # 理论值
    mean = lam
    variance = lam
    std = np.sqrt(lam)

    print(f"\n参数：λ = {lam}")
    print(f"\n理论值：")
    print(f"  - 期望 E[X] = λ = {mean}")
    print(f"  - 方差 Var(X) = λ = {variance}")
    print(f"  - 标准差 σ = √λ = {std:.3f}")
    print(f"\n注意：泊松分布的期望 = 方差")

    print(f"\n模拟统计：")
    print(f"  - 样本均值 = {data.mean():.3f}")
    print(f"  - 样本方差 = {data.var(ddof=1):.3f}")

    # 可视化
    fig, ax = plt.subplots(figsize=(10, 6))

    # 直方图
    ax.hist(data, bins=np.arange(-0.5, max(data)+1.5, 1),
            density=True, alpha=0.7, color='green', edgecolor='black',
            label='模拟数据')

    # 理论 PMF
    x = np.arange(0, max(data)+1)
    pmf = stats.poisson.pmf(x, lam)
    ax.plot(x, pmf, 'ro-', markersize=4, label='理论泊松分布')

    ax.axvline(lam, color='r', linestyle='--', lw=2, label=f'λ={lam}')

    ax.set_xlabel('事件发生次数')
    ax.set_ylabel('概率')
    ax.set_title(f'泊松分布 Poisson(λ={lam})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'demo_poisson_distribution.png', dpi=150)
    print(f"\n[图表已保存] /tmp/demo_poisson_distribution.png")
    plt.close()

    # 打印具体概率
    print(f"\n具体概率：")
    print(f"  - P(X = 0) = {stats.poisson.pmf(0, lam):.4f}")
    print(f"  - P(X = 1) = {stats.poisson.pmf(1, lam):.4f}")
    print(f"  - P(X = 2) = {stats.poisson.pmf(2, lam):.4f}")
    print(f"  - P(X = 3) = {stats.poisson.pmf(3, lam):.4f}")
    print(f"  - P(X > 5) = {1 - stats.poisson.cdf(5, lam):.4f}")

    return {'lambda': lam, 'mean': mean, 'std': std}


def compare_distributions():
    """对比三种分布的特点"""
    print("\n" + "=" * 70)
    print("三种分布的特点对比")
    print("=" * 70)

    comparison = """
    ┌─────────────┬──────────────┬─────────────┬──────────────┐
    │  分布类型   │   参数      │   支持范围  │   典型应用  │
    ├─────────────┼──────────────┼─────────────┼──────────────┤
    │ 正态分布    │ μ, σ        │   (-∞, +∞)  │ 身高、误差   │
    │ Normal      │ 连续对称     │   对称钟形  │ 测量误差    │
    ├─────────────┼──────────────┼─────────────┼──────────────┤
    │ 二项分布    │ n, p        │   {0,1,...,n}│ 成功计数    │
    │ Binomial    │ 离散        │   有限范围  │ 次品数、点击数│
    ├─────────────┼──────────────┼─────────────┼──────────────┤
    │ 泊松分布    │ λ           │   {0,1,2,...}│ 稀有事件计数│
    │ Poisson     │ 离散        │   无限范围  │ 投诉数、崩溃数│
    └─────────────┴──────────────┴─────────────┴──────────────┘

    关键区别：

    1. 数据类型：
       - 正态：连续数据（身高、体重、温度）
       - 二项/泊松：离散计数（整数）

    2. 参数含义：
       - 正态 μ：中心位置，σ：分散程度
       - 二项 n：试验次数，p：单次成功概率
       - 泊松 λ：单位时间/空间内平均发生次数

    3. 方特点：
       - 正态：对称，均值=中位数=众数
       - 二项：n×p 较大时接近对称
       - 泊松：期望=方差（重要特征！）

    4. 何时使用哪个？
       - 数据是连续且对称 → 正态分布
       - n 次试验中的成功次数 → 二项分布
       - 单位时间内稀有事件次数 → 泊松分布
    """

    print(comparison)


def main() -> None:
    """主函数：演示所有分布"""
    print("=" * 70)
    print("独立示例 2：常见概率分布")
    print("=" * 70)

    demo_normal_distribution()
    demo_binomial_distribution()
    demo_poisson_distribution()
    compare_distributions()

    print("\n" + "=" * 70)
    print("总结：如何选择合适的分布？")
    print("=" * 70)
    print("1. 先看数据类型：连续 vs 离散")
    print("2. 画直方图：对称、偏态、多峰？")
    print("3. 计算统计量：均值 vs 方差的关系")
    print("4. 用 QQ 图检验：数据是否接近理论分布")
    print("5. 不确定时，用 Bootstrap（不依赖分布假设）")
    print("=" * 70)


if __name__ == "__main__":
    main()

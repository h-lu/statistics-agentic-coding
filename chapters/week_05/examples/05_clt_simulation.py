#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立示例 3：中心极限定理（CLT）模拟

本示例独立演示中心极限定理：
无论总体分布是什么形状，样本均值的分布
在大样本时都会近似正态分布。

运行方式：python3 chapters/week_05/examples/05_clt_simulation.py
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def demonstrate_clt(
    population_size: int = 100000,
    sample_sizes: list[int] = [5, 10, 30, 100],
    n_simulations: int = 10000,
    seed: int = 42
) -> dict:
    """
    模拟中心极限定理

    参数：
        population_size: 总体大小
        sample_sizes: 要测试的样本量列表
        n_simulations: 每个样本量的模拟次数
        seed: 随机种子

    返回：
        dict: 包含总体统计和各样本量的结果
    """
    rng = np.random.default_rng(seed)

    # 创建严重右偏的总体（指数分布）
    population = rng.exponential(scale=10, size=population_size)
    pop_mean = np.mean(population)
    pop_std = np.std(population, ddof=1)
    pop_skew = stats.skew(population)

    print("=" * 70)
    print("中心极限定理（CLT）模拟")
    print("=" * 70)
    print(f"\n总体分布：指数分布（严重右偏）")
    print(f"  - 总体均值 μ = {pop_mean:.2f}")
    print(f"  - 总体标准差 σ = {pop_std:.2f}")
    print(f"  - 总体偏度 = {pop_skew:.2f} （正值表示右偏）")

    # 为每个样本量生成样本均值分布
    results = {}
    for n in sample_sizes:
        sample_means = np.array([
            np.mean(rng.choice(population, n, replace=False))
            for _ in range(n_simulations)
        ])

        # 计算统计量
        mean_of_means = np.mean(sample_means)
        std_of_means = np.std(sample_means, ddof=1)
        theoretical_se = pop_std / np.sqrt(n)

        # 正态性检验
        _, normality_p = stats.shapiro(sample_means[:5000])  # Shapiro 限制样本量

        results[n] = {
            'means': sample_means,
            'mean_of_means': mean_of_means,
            'std_of_means': std_of_means,
            'theoretical_se': theoretical_se,
            'normality_p': normality_p
        }

        print(f"\n样本量 n = {n}:")
        print(f"  - 样本均值分布的均值 = {mean_of_means:.3f} (理论: {pop_mean:.3f})")
        print(f"  - 样本均值分布的标准差 (SE) = {std_of_means:.3f}")
        print(f"  - 理论 SE = σ/√n = {theoretical_se:.3f}")
        print(f"  - 正态性检验 p 值 = {normality_p:.4f}")

        if normality_p > 0.05:
            print(f"  ✓ 不能拒绝正态假设，样本均值分布接近正态")
        else:
            print(f"  ✗ 拒绝正态假设，样本均值分布偏离正态")

    return {
        'population_mean': pop_mean,
        'population_std': pop_std,
        'sample_results': results
    }


def plot_clt_results(clt_data: dict) -> None:
    """绘制 CLT 模拟结果"""
    sample_sizes = sorted(clt_data['sample_results'].keys())
    pop_mean = clt_data['population_mean']

    # 创建子图
    n_plots = len(sample_sizes) + 1
    n_cols = 2
    n_rows = (n_plots + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    axes = axes.flatten()

    # 第一个图：总体分布
    ax = axes[0]
    population = np.random.exponential(scale=10, size=10000)
    ax.hist(population, bins=50, density=True, alpha=0.7, color='orange')
    ax.set_title('原始总体分布（严重右偏的指数分布）')
    ax.set_xlabel('数值')
    ax.set_ylabel('密度')

    # 后续图：不同样本量的均值分布
    for idx, n in enumerate(sample_sizes, 1):
        ax = axes[idx]
        means = clt_data['sample_results'][n]['means']
        se = clt_data['sample_results'][n]['std_of_means']

        # 直方图
        ax.hist(means, bins=50, density=True, alpha=0.7, color='steelblue')

        # 叠加理论正态分布
        x = np.linspace(means.min(), means.max(), 200)
        theoretical_norm = stats.norm.pdf(
            x,
            clt_data['sample_results'][n]['mean_of_means'],
            se
        )
        ax.plot(x, theoretical_norm, 'r-', lw=2, label='理论正态分布')

        ax.axvline(pop_mean, color='g', linestyle='--',
                   label=f'总体均值={pop_mean:.1f}')
        ax.set_title(f'样本均值分布（n={n}）')
        ax.set_xlabel('样本均值')
        ax.set_ylabel('密度')
        ax.legend()

    # 隐藏多余的子图
    for idx in range(len(sample_sizes) + 1, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig('/tmp/demo_clt_simulation.png', dpi=150)
    print(f"\n[图表已保存] /tmp/demo_clt_simulation.png")
    plt.close()


def plot_se_vs_sample_size() -> None:
    """绘制标准误与样本量的关系"""
    print("\n" + "=" * 70)
    print("标准误与样本量的关系")
    print("=" * 70)

    sigma = 15  # 假设总体标准差
    sample_sizes = np.array([5, 10, 20, 30, 50, 100, 200, 500, 1000])
    standard_errors = sigma / np.sqrt(sample_sizes)

    print(f"\n假设总体标准差 σ = {sigma}")
    print(f"\n{'样本量 n':<12} {'标准误 SE':<15} {'相对变化'}")
    print("-" * 50)

    prev_se = None
    for n, se in zip(sample_sizes, standard_errors):
        if prev_se:
            rel_change = (prev_se - se) / prev_se
            print(f"{n:<12} {se:<15.3f} {-rel_change:.1%}")
        else:
            print(f"{n:<12} {se:<15.3f} -")
        prev_se = se

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sample_sizes, standard_errors, 'o-', lw=2)
    ax.set_xlabel('样本量 n')
    ax.set_ylabel('标准误 SE = σ/√n')
    ax.set_title('标准误与样本量的关系：SE 随 √n 增长而减小')
    ax.grid(True, alpha=0.3)

    # 添加关键点标注
    for n in [30, 100]:
        se = sigma / np.sqrt(n)
        ax.scatter([n], [se], s=100, color='red', zorder=5)
        ax.annotate(f'n={n}\nSE={se:.2f}', xy=(n, se),
                   xytext=(10, 20), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    plt.tight_layout()
    plt.savefig('/tmp/demo_se_vs_sample_size.png', dpi=150)
    print(f"\n[图表已保存] /tmp/demo_se_vs_sample_size.png")
    plt.close()


def demonstrate_different_populations():
    """演示不同总体分布下的 CLT"""
    print("\n" + "=" * 70)
    print("不同总体分布下的 CLT")
    print("=" * 70)

    np.random.seed(42)
    n_simulations = 5000
    sample_size = 30

    distributions = {
        '正态分布': lambda: np.random.normal(100, 15, 100000),
        '均匀分布': lambda: np.random.uniform(0, 200, 100000),
        '指数分布（右偏）': lambda: np.random.exponential(50, 100000),
        '双峰分布': lambda: np.concatenate([
            np.random.normal(50, 10, 50000),
            np.random.normal(150, 10, 50000)
        ])
    }

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for idx, (name, pop_func) in enumerate(distributions.items()):
        population = pop_func()
        pop_mean = np.mean(population)

        # 生成样本均值分布
        sample_means = np.array([
            np.mean(np.random.choice(population, sample_size, replace=False))
            for _ in range(n_simulations)
        ])

        # 总体分布图
        axes[0, idx].hist(population, bins=50, density=True, alpha=0.7)
        axes[0, idx].set_title(f'{name}\n(原始总体)')
        axes[0, idx].set_ylabel('密度')

        # 样本均值分布图
        axes[1, idx].hist(sample_means, bins=30, density=True, alpha=0.7, color='steelblue')
        axes[1, idx].axvline(pop_mean, color='r', linestyle='--', label=f'μ={pop_mean:.1f}')
        axes[1, idx].set_title(f'样本均值分布 (n={sample_size})')
        axes[1, idx].set_xlabel('样本均值')
        axes[1, idx].legend()

    plt.tight_layout()
    plt.savefig('/tmp/demo_clt_different_populations.png', dpi=150)
    print(f"\n[图表已保存] /tmp/demo_clt_different_populations.png")
    plt.close()


def main() -> None:
    """主函数：运行所有 CLT 演示"""
    print("=" * 70)
    print("独立示例 3：中心极限定理（CLT）")
    print("=" * 70)

    # 主实验
    clt_data = demonstrate_clt()
    plot_clt_results(clt_data)

    # 标准误与样本量关系
    plot_se_vs_sample_size()

    # 不同总体下的 CLT
    demonstrate_different_populations()

    print("\n" + "=" * 70)
    print("CLT 核心要点")
    print("=" * 70)
    print("1. 中心极限定理（CLT）：样本量够大时，")
    print("   无论总体分布是什么形状，样本均值的分布都近似正态。")
    print()
    print("2. 标准误 SE = σ/√n：样本量越大，均值估计越稳定。")
    print("   - 样本量翻倍 → SE 缩小到原来的 1/√2 ≈ 0.71")
    print("   - 样本量 4 倍 → SE 缩小到原来的 1/2")
    print()
    print("3. '够大'取决于总体分布：")
    print("   - 总体接近正态：n=30 可能够")
    print("   - 总体严重偏态：可能需要 n=100 或更多")
    print()
    print("4. CLT 只适用于均值（及和）这类'可加'统计量，")
    print("   不适用于中位数、方差等。")
    print("=" * 70)


if __name__ == "__main__":
    main()

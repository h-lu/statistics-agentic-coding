#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立示例 4：Bootstrap 抽样分布估计

本示例独立演示如何用 Bootstrap 方法
估计任意统计量的抽样分布和置信区间。

运行方式：python3 chapters/week_05/examples/05_bootstrap_demo.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def bootstrap(
    sample: np.ndarray,
    statistic: str = 'mean',
    n_bootstrap: int = 10000,
    seed: int = 42
) -> dict:
    """
    通用的 Bootstrap 函数

    参数：
        sample: 原始样本
        statistic: 统计量类型 ('mean', 'median', 'std', 'var')
        n_bootstrap: Bootstrap 重复次数
        seed: 随机种子

    返回：
        dict: Bootstrap 分布和置信区间
    """
    rng = np.random.default_rng(seed)
    n = len(sample)

    # 统计量函数映射
    stat_functions = {
        'mean': np.mean,
        'median': np.median,
        'std': lambda x: np.std(x, ddof=1),
        'var': lambda x: np.var(x, ddof=1)
    }

    if statistic not in stat_functions:
        raise ValueError(f"不支持的统计量: {statistic}")

    stat_func = stat_functions[statistic]

    # Bootstrap 重采样
    boot_stats = np.array([
        stat_func(rng.choice(sample, size=n, replace=True))
        for _ in range(n_bootstrap)
    ])

    # 计算置信区间（percentile 方法）
    ci_low = np.percentile(boot_stats, 2.5)
    ci_high = np.percentile(boot_stats, 97.5)

    # 计算标准误
    se = boot_stats.std(ddof=1)

    # 观察到的统计量值
    observed = stat_func(sample)

    return {
        'bootstrap_distribution': boot_stats,
        'observed': observed,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'se': se,
        'ci_width': ci_high - ci_low
    }


def compare_statistics_stability():
    """比较不同统计量的稳定性（均值 vs 中位数）"""
    print("\n" + "=" * 70)
    print("Bootstrap：比较不同统计量的稳定性")
    print("=" * 70)

    # 生成右偏样本
    np.random.seed(42)
    sample = np.random.lognormal(7, 0.8, 100)

    print(f"\n样本统计：")
    print(f"  - 样本量：{len(sample)}")
    print(f"  - 均值：{np.mean(sample):.0f}")
    print(f"  - 中位数：{np.median(sample):.0f}")
    print(f"  - 标准差：{np.std(sample, ddof=1):.0f}")

    # Bootstrap 均值
    boot_mean = bootstrap(sample, 'mean', n_bootstrap=10000)

    # Bootstrap 中位数
    boot_median = bootstrap(sample, 'median', n_bootstrap=10000)

    print(f"\nBootstrap 均值结果：")
    print(f"  - 点估计：{boot_mean['observed']:.0f}")
    print(f"  - 95% CI：[{boot_mean['ci_low']:.0f}, {boot_mean['ci_high']:.0f}]")
    print(f"  - CI 宽度：{boot_mean['ci_width']:.0f}")
    print(f"  - 标准误：{boot_mean['se']:.0f}")

    print(f"\nBootstrap 中位数结果：")
    print(f"  - 点估计：{boot_median['observed']:.0f}")
    print(f"  - 95% CI：[{boot_median['ci_low']:.0f}, {boot_median['ci_high']:.0f}]")
    print(f"  - CI 宽度：{boot_median['ci_width']:.0f}")
    print(f"  - 标准误：{boot_median['se']:.0f}")

    print(f"\n稳定性比较：")
    print(f"  - 均值 SE = {boot_mean['se']:.0f}")
    print(f"  - 中位数 SE = {boot_median['se']:.0f}")
    print(f"  → 均值是更稳定的统计量（SE 更小）")

    # 可视化
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
    axes[1].hist(boot_median['bootstrap_distribution'], bins=50, density=True,
                 alpha=0.7, color='green')
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
    plt.savefig('/tmp/demo_bootstrap_comparison.png', dpi=150)
    print(f"\n[图表已保存] /tmp/demo_bootstrap_comparison.png")
    plt.close()


def bootstrap_correlation():
    """Bootstrap 相关系数的抽样分布"""
    print("\n" + "=" * 70)
    print("Bootstrap：相关系数的抽样分布")
    print("=" * 70)

    np.random.seed(42)

    # 创建相关数据
    x = np.random.lognormal(8.5, 0.5, 100)
    y = x * 0.3 + np.random.lognormal(6, 0.5, 100)

    # 观察到的相关系数
    observed_corr = np.corrcoef(x, y)[0, 1]
    print(f"\n观察到的相关系数：r = {observed_corr:.3f}")

    # Bootstrap 相关系数
    n = len(x)
    n_bootstrap = 10000
    boot_corrs = []

    for _ in range(n_bootstrap):
        # 成对重采样（保持配对关系）
        idx = np.random.choice(n, size=n, replace=True)
        boot_x = x[idx]
        boot_y = y[idx]
        boot_corrs.append(np.corrcoef(boot_x, boot_y)[0, 1])

    boot_corrs = np.array(boot_corrs)

    # 计算置信区间
    ci_low = np.percentile(boot_corrs, 2.5)
    ci_high = np.percentile(boot_corrs, 97.5)
    se = boot_corrs.std(ddof=1)

    print(f"\nBootstrap 95% CI：[{ci_low:.3f}, {ci_high:.3f}]")
    print(f"标准误：{se:.3f}")

    # 判断是否显著
    is_significant = not (ci_low <= 0 <= ci_high)
    print(f"\n显著性判断：")
    print(f"  - CI 包含 0？{'是' if not is_significant else '否'}")
    print(f"  - 结论：{'不显著' if not is_significant else '显著'}")

    # 可视化
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(boot_corrs, bins=50, density=True, alpha=0.7, color='steelblue')
    ax.axvline(observed_corr, color='r', linestyle='--',
               label=f'r = {observed_corr:.3f}')
    ax.axvline(ci_low, color='orange', linestyle='--', alpha=0.7)
    ax.axvline(ci_high, color='orange', linestyle='--', alpha=0.7,
               label='95% CI')
    ax.axvline(0, color='k', linestyle='-', linewidth=0.5)
    ax.set_xlabel('相关系数 r')
    ax.set_ylabel('密度')
    ax.set_title('Bootstrap 分布：Pearson 相关系数')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/tmp/demo_bootstrap_correlation.png', dpi=150)
    print(f"\n[图表已保存] /tmp/demo_bootstrap_correlation.png")
    plt.close()


def bootstrap_mean_diff():
    """Bootstrap 两组均值差异的抽样分布"""
    print("\n" + "=" * 70)
    print("Bootstrap：两组均值差异的抽样分布")
    print("=" * 70)

    np.random.seed(42)

    # 创建两组数据
    group1 = np.random.lognormal(8, 0.6, 50)    # 钻石用户
    group2 = np.random.lognormal(6.5, 0.5, 200)  # 普通用户

    obs_diff = np.mean(group1) - np.mean(group2)
    print(f"\n观察到的均值差异：{obs_diff:.0f} 元")
    print(f"  - 钻石用户均值：{np.mean(group1):.0f} 元 (n={len(group1)})")
    print(f"  - 普通用户均值：{np.mean(group2):.0f} 元 (n={len(group2)})")

    # Bootstrap 均值差异
    n1, n2 = len(group1), len(group2)
    n_bootstrap = 10000
    boot_diffs = []

    for _ in range(n_bootstrap):
        resample1 = np.random.choice(group1, size=n1, replace=True)
        resample2 = np.random.choice(group2, size=n2, replace=True)
        boot_diffs.append(np.mean(resample1) - np.mean(resample2))

    boot_diffs = np.array(boot_diffs)

    # 计算置信区间
    ci_low = np.percentile(boot_diffs, 2.5)
    ci_high = np.percentile(boot_diffs, 97.5)
    se = boot_diffs.std(ddof=1)

    print(f"\nBootstrap 95% CI：[{ci_low:.0f}, {ci_high:.0f}]")
    print(f"标准误：{se:.0f}")

    # 判断是否显著
    is_significant = not (ci_low <= 0 <= ci_high)
    print(f"\n显著性判断：")
    print(f"  - CI 包含 0？{'是' if not is_significant else '否'}")
    print(f"  - 结论：差异{'不显著' if not is_significant else '显著'}")

    # 可视化
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(boot_diffs, bins=50, density=True, alpha=0.7, color='steelblue')
    ax.axvline(obs_diff, color='r', linestyle='--',
               label=f'差异 = {obs_diff:.0f}')
    ax.axvline(ci_low, color='orange', linestyle='--', alpha=0.7)
    ax.axvline(ci_high, color='orange', linestyle='--', alpha=0.7,
               label='95% CI')
    ax.axvline(0, color='k', linestyle='-', linewidth=1, label='零假设')
    ax.set_xlabel('均值差异（元）')
    ax.set_ylabel('density')
    ax.set_title('Bootstrap 分布：两组均值差异')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/tmp/demo_bootstrap_mean_diff.png', dpi=150)
    print(f"\n[图表已保存] /tmp/demo_bootstrap_mean_diff.png")
    plt.close()


def bootstrap_ci_methods():
    """演示不同的 Bootstrap 置信区间计算方法"""
    print("\n" + "=" * 70)
    print("Bootstrap 置信区间的不同计算方法")
    print("=" * 70)

    np.random.seed(42)
    sample = np.random.normal(100, 15, 50)
    n_bootstrap = 10000

    # Percentile 方法
    boot_means = np.array([
        np.mean(np.random.choice(sample, len(sample), replace=True))
        for _ in range(n_bootstrap)
    ])

    # 方法 1: Percentile
    ci_percentile_low = np.percentile(boot_means, 2.5)
    ci_percentile_high = np.percentile(boot_means, 97.5)

    # 方法 2: Basic (Reverse Percentile)
    observed = np.mean(sample)
    se = boot_means.std(ddof=1)
    ci_basic_low = 2 * observed - ci_percentile_high
    ci_basic_high = 2 * observed - ci_percentile_low

    # 方法 3: SE-based (Normal approximation)
    ci_se_low = observed - 1.96 * se
    ci_se_high = observed + 1.96 * se

    print(f"\n观察均值：{observed:.2f}")
    print(f"\n{'方法':<20} {'95% CI':<25} {'宽度'}")
    print("-" * 70)
    print(f"{'Percentile':<20} [{ci_percentile_low:.2f}, {ci_percentile_high:.2f}] {ci_percentile_high - ci_percentile_low:.2f}")
    print(f"{'Basic':<20} [{ci_basic_low:.2f}, {ci_basic_high:.2f}] {ci_basic_high - ci_basic_low:.2f}")
    print(f"{'SE-based':<20} [{ci_se_low:.2f}, {ci_se_high:.2f}] {ci_se_high - ci_se_low:.2f}")

    print(f"\n老潘的建议：")
    print(f"  'Percentile 方法最简单，适合大多数情况。'")
    print(f"  'Basic 方法在小样本时有偏差。'")
    print(f"  'SE-based 假设 Bootstrap 分布正态，可能不成立。'")
    print(f"  '工程上推荐用 Percentile，简单可靠。'")


def main() -> None:
    """主函数：运行所有 Bootstrap 演示"""
    print("=" * 70)
    print("独立示例 4：Bootstrap 抽样分布估计")
    print("=" * 70)

    compare_statistics_stability()
    bootstrap_correlation()
    bootstrap_mean_diff()
    bootstrap_ci_methods()

    print("\n" + "=" * 70)
    print("Bootstrap 核心要点")
    print("=" * 70)
    print("1. Bootstrap 通过有放回重采样估计统计量的抽样分布")
    print("2. 不依赖总体分布的强假设（非参数方法）")
    print("3. 可以估计任意统计量（均值、中位数、相关系数等）")
    print("4. 置信区间告诉你'如果重新抽样，结果会波动多大'")
    print("5. CI 不包含 0（差异）或 0（相关）= 初步显著")
    print()
    print("小北的疑问：")
    print("  '为什么要重采样？直接算不行吗？'")
    print()
    print("老潘的解释：")
    print("  '你只有一个样本，不知道'如果重新抽样会怎样'。'")
    print("  'Bootstrap 用重采样模拟这个过程，给你不确定性估计。'")
    print("  '这是比'只报一个数'专业得多的做法。'")
    print("=" * 70)


if __name__ == "__main__":
    main()

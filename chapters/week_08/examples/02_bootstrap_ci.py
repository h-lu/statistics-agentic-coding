"""
示例：Bootstrap 均值差置信区间——用重采样估计不确定性。

本例演示如何用 Bootstrap 构造两组均值差的 95% CI：
- 有放回地重采样（Bootstrap 样本）
- 计算每个 Bootstrap 样本的均值差
- 用百分位数法构造 CI

运行方式：python3 chapters/week_08/examples/02_bootstrap_ci.py
预期输出：
  - stdout 输出原始均值差、Bootstrap 95% CI、显著性判断
  - 生成 bootstrap_distribution.png（Bootstrap 分布直方图）
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def bootstrap_ci_diff(group1: np.ndarray,
                      group2: np.ndarray,
                      n_bootstrap: int = 10000,
                      conf_level: float = 0.95,
                      seed: int = 42) -> tuple[np.ndarray, float, float]:
    """
    用 Bootstrap 构造两组均值差的置信区间。

    参数：
        group1, group2: 两组数据
        n_bootstrap: Bootstrap 次数
        conf_level: 置信水平（如 0.95）
        seed: 随机种子

    返回：
        bootstrap_diffs: Bootstrap 均值差列表
        ci_low: CI 下界
        ci_high: CI 上界
    """
    np.random.seed(seed)
    n1, n2 = len(group1), len(group2)

    # 原始均值差
    observed_diff = np.mean(group1) - np.mean(group2)

    # Bootstrap 重采样
    bootstrap_diffs = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        # 有放回地抽取 Bootstrap 样本
        boot_sample1 = np.random.choice(group1, size=n1, replace=True)
        boot_sample2 = np.random.choice(group2, size=n2, replace=True)

        # 计算 Bootstrap 均值差
        boot_diff = np.mean(boot_sample1) - np.mean(boot_sample2)
        bootstrap_diffs[i] = boot_diff

    # 计算 CI（百分位数法）
    alpha = 1 - conf_level
    ci_low = np.percentile(bootstrap_diffs, 100 * alpha / 2)
    ci_high = np.percentile(bootstrap_diffs, 100 * (1 - alpha / 2))

    return bootstrap_diffs, ci_low, ci_high


def plot_bootstrap_distribution(bootstrap_diffs: np.ndarray,
                                 ci_low: float,
                                 ci_high: float,
                                 observed_diff: float,
                                 output_path: str = 'bootstrap_distribution.png') -> None:
    """
    可视化 Bootstrap 均值差分布。

    参数：
        bootstrap_diffs: Bootstrap 均值差列表
        ci_low: CI 下界
        ci_high: CI 上界
        observed_diff: 原始均值差
        output_path: 输出图片路径
    """
    plt.figure(figsize=(10, 6))
    plt.hist(bootstrap_diffs, bins=50, density=True, alpha=0.7, color='steelblue')
    plt.axvline(ci_low, color='red', linestyle='--', linewidth=2, label=f'2.5% ({ci_low:.2f})')
    plt.axvline(ci_high, color='red', linestyle='--', linewidth=2, label=f'97.5% ({ci_high:.2f})')
    plt.axvline(0, color='black', linestyle='-', linewidth=2, label='零线（无差异）')
    plt.axvline(observed_diff, color='green', linestyle='-', linewidth=2,
                label=f'原始均值差 ({observed_diff:.2f})')

    plt.xlabel('均值差（元）')
    plt.ylabel('密度')
    plt.title('Bootstrap 均值差分布（10000 次重采样）')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()
    print(f"图片已保存：{output_path}")


def main() -> None:
    """主函数：Bootstrap 均值差 CI。"""
    print("=== Bootstrap 均值差置信区间 ===\n")

    # 模拟数据：新用户 vs 老用户的消费
    np.random.seed(42)
    new_users = np.random.normal(loc=315, scale=50, size=100)   # 新用户
    old_users = np.random.normal(loc=300, scale=50, size=100)  # 老用户

    print("数据描述：")
    print(f"  新用户：n={len(new_users)}, 均值={np.mean(new_users):.2f}, 标准差={np.std(new_users, ddof=1):.2f}")
    print(f"  老用户：n={len(old_users)}, 均值={np.mean(old_users):.2f}, 标准差={np.std(old_users, ddof=1):.2f}")

    # Bootstrap CI
    bootstrap_diffs, ci_low, ci_high = bootstrap_ci_diff(new_users, old_users)

    print(f"\nBootstrap 结果（10000 次重采样）：")
    print(f"  原始均值差：{np.mean(new_users) - np.mean(old_users):.2f} 元")
    print(f"  Bootstrap 95% CI：[{ci_low:.2f}, {ci_high:.2f}]")
    print(f"  CI 宽度：{ci_high - ci_low:.2f} 元")

    # 判断是否显著
    if ci_low > 0:
        print(f"\n结论：新用户显著高于老用户（95% CI 不包含 0）")
    elif ci_high < 0:
        print(f"\n结论：新用户显著低于老用户（95% CI 不包含 0）")
    else:
        print(f"\n结论：两组差异不显著（95% CI 包含 0）")

    # Bootstrap 优势说明
    print("\nBootstrap 的优势：")
    print("  - 不依赖正态分布假设")
    print("  - 适用于任何统计量（均值、中位数、相关系数等）")
    print("  - 用计算换取理论假设")

    # 可视化
    observed_diff = np.mean(new_users) - np.mean(old_users)
    plot_bootstrap_distribution(bootstrap_diffs, ci_low, ci_high, observed_diff)


if __name__ == "__main__":
    main()

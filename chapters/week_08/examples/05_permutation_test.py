"""
示例：置换检验（Permutation Test）——分布无关的假设检验。

本例演示如何用置换检验判断两组均值差异是否显著：
- 通过随机打乱标签构造零分布（null distribution）
- 不依赖正态分布假设
- 适用于任何分布的数据

运行方式：python3 chapters/week_08/examples/05_permutation_test.py
预期输出：
  - stdout 输出观测均值差、p 值、显著性判断
  - 生成 permutation_distribution.png（置换分布 + 观测值标注）
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def permutation_test(group1: np.ndarray,
                     group2: np.ndarray,
                     n_permutations: int = 10000,
                     stat_func: callable = np.mean,
                     seed: int = 42) -> tuple[float, np.ndarray, float]:
    """
    执行置换检验。

    参数：
        group1, group2: 两组数据
        n_permutations: 置换次数
        stat_func: 统计量函数（默认是均值）
        seed: 随机种子

    返回：
        observed_stat: 观测统计量
        perm_stats: 置换统计量列表
        p_value: p 值（双尾）
    """
    np.random.seed(seed)
    n1, n2 = len(group1), len(group2)

    # 合并数据
    combined = np.concatenate([group1, group2])

    # 观测统计量
    observed_stat = stat_func(group1) - stat_func(group2)

    # 置换检验
    perm_stats = np.empty(n_permutations)

    for i in range(n_permutations):
        # 随机打乱标签
        permuted = np.random.permutation(combined)

        # 重新分组
        perm_group1 = permuted[:n1]
        perm_group2 = permuted[n1:n1 + n2]

        # 计算置换统计量
        perm_stat = stat_func(perm_group1) - stat_func(perm_group2)
        perm_stats[i] = perm_stat

    # 计算 p 值（双尾）
    if observed_stat >= 0:
        p_value = (np.mean(perm_stats >= observed_stat) +
                   np.mean(perm_stats <= -observed_stat))
    else:
        p_value = (np.mean(perm_stats <= observed_stat) +
                   np.mean(perm_stats >= -observed_stat))

    return observed_stat, perm_stats, p_value


def plot_permutation_distribution(perm_stats: np.ndarray,
                                   observed_stat: float,
                                   p_value: float,
                                   output_path: str = 'permutation_distribution.png') -> None:
    """
    可视化置换分布。

    参数：
        perm_stats: 置换统计量列表
        observed_stat: 观测统计量
        p_value: p 值
        output_path: 输出图片路径
    """
    plt.figure(figsize=(10, 6))
    plt.hist(perm_stats, bins=50, density=True, alpha=0.7,
             color='steelblue', label='置换分布（H0 为真时）')

    # 标注观测统计量
    plt.axvline(observed_stat, color='red', linestyle='-',
                linewidth=3, label=f'观测值 ({observed_stat:.2f})')
    plt.axvline(-observed_stat, color='red', linestyle='--',
                linewidth=2, label=f'负观测值 ({-observed_stat:.2f})')

    # 标注极端区域（p < 0.05）
    perm_95_low = np.percentile(perm_stats, 2.5)
    perm_95_high = np.percentile(perm_stats, 97.5)
    plt.axvline(perm_95_low, color='orange', linestyle=':',
                linewidth=2, label=f'2.5% ({perm_95_low:.2f})')
    plt.axvline(perm_95_high, color='orange', linestyle=':',
                linewidth=2, label=f'97.5% ({perm_95_high:.2f})')

    # 填充极端区域
    plt.axvspan(perm_stats.min(), perm_95_low, alpha=0.3,
                color='red', label='极端区域（α=0.05）')
    plt.axvspan(perm_95_high, perm_stats.max(), alpha=0.3, color='red')

    plt.xlabel('均值差（元）')
    plt.ylabel('密度')
    plt.title(f'置换检验：均值差的零分布（{len(perm_stats)} 次置换）\np = {p_value:.6f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()
    print(f"图片已保存：{output_path}")


def main() -> None:
    """主函数：置换检验。"""
    print("=== 置换检验（Permutation Test） ===\n")

    # 模拟数据：新用户 vs 老用户
    np.random.seed(42)
    new_users = np.random.normal(loc=315, scale=50, size=100)
    old_users = np.random.normal(loc=300, scale=50, size=100)

    print("数据描述：")
    print(f"  新用户：n={len(new_users)}, 均值={np.mean(new_users):.2f}")
    print(f"  老用户：n={len(old_users)}, 均值={np.mean(old_users):.2f}")
    print(f"  均值差：{np.mean(new_users) - np.mean(old_users):.2f}")

    # 置换检验
    observed_diff, perm_stats, p_value = permutation_test(new_users, old_users)

    print(f"\n置换检验结果（{len(perm_stats)} 次置换）：")
    print(f"  观测均值差：{observed_diff:.2f} 元")
    print(f"  p 值：{p_value:.6f}")

    # 显著性判断
    alpha = 0.05
    if p_value < alpha:
        print(f"  结论：拒绝 H0（差异显著，p < {alpha}）")
    else:
        print(f"  结论：无法拒绝 H0（差异不显著，p >= {alpha}）")

    # 置换分布统计
    print(f"\n置换分布统计：")
    print(f"  均值：{np.mean(perm_stats):.4f}（理论应接近 0）")
    print(f"  标准差：{np.std(perm_stats):.4f}")
    print(f"  95% 区间：[{np.percentile(perm_stats, 2.5):.2f}, {np.percentile(perm_stats, 97.5):.2f}]")

    # 置换检验的优势
    print(f"\n置换检验的优势：")
    print(f"  - 不依赖正态分布假设")
    print(f"  - 适用于偏态数据、有离群点的数据")
    print(f"  - 分布无关（distribution-free）")
    print(f"  - 只需满足可交换性（exchangeability）")

    # 局限性
    print(f"\n置换检验的局限性：")
    print(f"  - 计算成本高（需大量置换）")
    print(f"  - 不适用于有依赖关系的数据（如时间序列）")
    print(f"  - 小样本时结果可能不稳定")

    # 可视化
    plot_permutation_distribution(perm_stats, observed_diff, p_value)


if __name__ == "__main__":
    main()

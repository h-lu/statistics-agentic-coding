"""
示例：置换 ANOVA——多组比较的分布无关方法。

本例演示如何用置换检验扩展到多组比较场景：
- 类似 Week 07 的 ANOVA，但不依赖正态假设
- 通过随机打乱标签构造 F 统计量的零分布
- 适用于 3 组以上的比较

运行方式：python3 chapters/week_08/examples/06_permutation_anova.py
预期输出：
  - stdout 输出观测 F 值、p 值、显著性判断
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def permutation_anova(groups: list[np.ndarray],
                      n_permutations: int = 10000,
                      seed: int = 42) -> tuple[float, np.ndarray, float]:
    """
    执行置换 ANOVA（F 检验）。

    参数：
        groups: 多组数据的列表
        n_permutations: 置换次数
        seed: 随机种子

    返回：
        observed_f: 观测 F 统计量
        perm_fs: 置换 F 统计量列表
        p_value: p 值
    """
    np.random.seed(seed)

    # 合并数据
    combined = np.concatenate(groups)
    group_sizes = [len(g) for g in groups]

    # 观测 F 统计量（Week 07 的方法）
    observed_f, _ = stats.f_oneway(*groups)

    # 置换检验
    perm_fs = np.empty(n_permutations)

    for i in range(n_permutations):
        # 随机打乱标签
        permuted = np.random.permutation(combined)

        # 重新分组
        perm_groups = []
        start = 0
        for size in group_sizes:
            perm_groups.append(permuted[start:start + size])
            start += size

        # 计算置换 F 统计量
        perm_f, _ = stats.f_oneway(*perm_groups)
        perm_fs[i] = perm_f

    # 计算 p 值
    p_value = np.mean(perm_fs >= observed_f)

    return observed_f, perm_fs, p_value


def main() -> None:
    """主函数：置换 ANOVA。"""
    print("=== 置换 ANOVA（多组比较） ===\n")

    # 模拟 5 个城市的消费数据（Week 07 的例子）
    np.random.seed(42)
    cities = ['北京', '上海', '广州', '深圳', '杭州']
    true_means = [280, 310, 270, 320, 290]
    groups = [np.random.normal(loc=mean, scale=50, size=100) for mean in true_means]

    print("数据描述：")
    for city, group, true_mean in zip(cities, groups, true_means):
        print(f"  {city}：n={len(group)}, 均值={np.mean(group):.2f}, 真实均值={true_mean}")

    # 传统 ANOVA（Week 07 的方法）
    observed_f, p_value_traditional = stats.f_oneway(*groups)

    print(f"\n传统 ANOVA（Week 07）：")
    print(f"  F 统计量：{observed_f:.4f}")
    print(f"  p 值：{p_value_traditional:.6f}")
    print(f"  结论：{'拒绝 H0（至少有一对不同）' if p_value_traditional < 0.05 else '无法拒绝 H0'}")

    # 置换 ANOVA
    observed_f_perm, perm_fs, p_value_perm = permutation_anova(groups)

    print(f"\n置换 ANOVA（{len(perm_fs)} 次置换）：")
    print(f"  F 统计量：{observed_f_perm:.4f}")
    print(f"  p 值：{p_value_perm:.6f}")
    print(f"  结论：{'拒绝 H0（至少有一对不同）' if p_value_perm < 0.05 else '无法拒绝 H0'}")

    # 两种方法对比
    print(f"\n传统 ANOVA vs 置换 ANOVA：")
    print(f"  F 统计量：{observed_f:.4f} vs {observed_f_perm:.4f}")
    print(f"  p 值：{p_value_traditional:.6f} vs {p_value_perm:.6f}")

    if abs(p_value_traditional - p_value_perm) < 0.01:
        print(f"  两种方法结果一致（数据满足正态假设）")
    else:
        print(f"  两种方法结果有差异（置换 ANOVA 更稳健）")

    # 置换分布统计
    print(f"\n置换 F 分布统计：")
    print(f"  均值：{np.mean(perm_fs):.4f}")
    print(f"  标准差：{np.std(perm_fs):.4f}")
    print(f"  95% 分位数：{np.percentile(perm_fs, 95):.4f}")
    print(f"  观测 F 值：{observed_f_perm:.4f}")

    # 置换 ANOVA 的优势
    print(f"\n置换 ANOVA 的优势：")
    print(f"  - 不依赖正态分布假设")
    print(f"  - 适用于偏态数据、方差不齐的数据")
    print(f"  - 当传统 ANOVA 假设严重违反时，是稳健的替代方案")

    # 适用场景
    print(f"\n何时使用置换 ANOVA：")
    print(f"  - 数据严重偏态")
    print(f"  - 有极端离群点")
    print(f"  - 方差不齐（Levene 检验显著）")
    print(f"  - 样本量较小（n < 30 每组）")


if __name__ == "__main__":
    main()

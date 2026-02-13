"""
示例：Bootstrap Cohen's d 置信区间——效应量的不确定性。

本例演示如何用 Bootstrap 构造 Cohen's d（效应量）的 CI。
效应量的 CI 比 p 值更重要，它告诉你"效应有多稳定"。

Cohen's d 解释：
- |d| < 0.2：效应量极小
- |d| < 0.5：效应量小
- |d| < 0.8：效应量中等
- |d| >= 0.8：效应量大

运行方式：python3 chapters/week_08/examples/04_bootstrap_effect_size.py
预期输出：
  - stdout 输出 Cohen's d 及其 95% CI、效应量解释
"""
from __future__ import annotations

import numpy as np


def bootstrap_ci_cohens_d(group1: np.ndarray,
                           group2: np.ndarray,
                           n_bootstrap: int = 10000,
                           conf_level: float = 0.95,
                           seed: int = 42) -> tuple[np.ndarray, tuple[float, float], float]:
    """
    用 Bootstrap 构造 Cohen's d 的置信区间。

    参数：
        group1, group2: 两组数据
        n_bootstrap: Bootstrap 次数
        conf_level: 置信水平
        seed: 随机种子

    返回：
        boot_ds: Cohen's d 的 Bootstrap 分布
        (ci_low, ci_high): 95% CI
        observed_d: 原始 Cohen's d
    """
    np.random.seed(seed)
    n1, n2 = len(group1), len(group2)

    # 原始 Cohen's d
    pooled_std = np.sqrt(((n1 - 1) * np.var(group1, ddof=1) +
                          (n2 - 1) * np.var(group2, ddof=1)) / (n1 + n2 - 2))
    observed_d = (np.mean(group1) - np.mean(group2)) / pooled_std

    # Bootstrap 重采样
    boot_ds = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        boot_sample1 = np.random.choice(group1, size=n1, replace=True)
        boot_sample2 = np.random.choice(group2, size=n2, replace=True)

        # 计算 Bootstrap Cohen's d
        boot_pooled_std = np.sqrt(((n1 - 1) * np.var(boot_sample1, ddof=1) +
                                   (n2 - 1) * np.var(boot_sample2, ddof=1)) / (n1 + n2 - 2))
        boot_d = (np.mean(boot_sample1) - np.mean(boot_sample2)) / boot_pooled_std
        boot_ds[i] = boot_d

    # 计算 CI
    alpha = 1 - conf_level
    ci_low = np.percentile(boot_ds, 100 * alpha / 2)
    ci_high = np.percentile(boot_ds, 100 * (1 - alpha / 2))

    return boot_ds, (ci_low, ci_high), observed_d


def interpret_cohens_d(d: float) -> str:
    """
    解释 Cohen's d 的效应量大小。

    参数：
        d: Cohen's d 值

    返回：
        效应量解释文本
    """
    abs_d = abs(d)
    if abs_d < 0.2:
        return "效应量极小（negligible）"
    elif abs_d < 0.5:
        return "效应量小（small）"
    elif abs_d < 0.8:
        return "效应量中等（medium）"
    else:
        return "效应量大（large）"


def main() -> None:
    """主函数：Bootstrap Cohen's d CI。"""
    print("=== Bootstrap Cohen's d 置信区间 ===\n")

    # 模拟数据
    np.random.seed(42)
    new_users = np.random.normal(loc=315, scale=50, size=100)
    old_users = np.random.normal(loc=300, scale=50, size=100)

    print("数据描述：")
    print(f"  新用户：n={len(new_users)}, 均值={np.mean(new_users):.2f}, 标准差={np.std(new_users, ddof=1):.2f}")
    print(f"  老用户：n={len(old_users)}, 均值={np.mean(old_users):.2f}, 标准差={np.std(old_users, ddof=1):.2f}")

    # Bootstrap Cohen's d
    boot_ds, (ci_low, ci_high), observed_d = bootstrap_ci_cohens_d(new_users, old_users)

    print(f"\nCohen's d 结果（10000 次重采样）：")
    print(f"  原始 Cohen's d：{observed_d:.3f}")
    print(f"  Bootstrap 95% CI：[{ci_low:.3f}, {ci_high:.3f}]")
    print(f"  CI 宽度：{ci_high - ci_low:.3f}")

    # 解释效应量
    interpretation = interpret_cohens_d(observed_d)
    print(f"\n效应量解释：{interpretation}")

    # Cohen's d 的实际意义
    mean_diff = np.mean(new_users) - np.mean(old_users)
    pooled_std = np.sqrt(((len(new_users) - 1) * np.var(new_users, ddof=1) +
                          (len(old_users) - 1) * np.var(old_users, ddof=1)) /
                         (len(new_users) + len(old_users) - 2))
    print(f"\n实际意义：")
    print(f"  均值差：{mean_diff:.2f} 元")
    print(f"  合并标准差：{pooled_std:.2f} 元")
    print(f"  Cohen's d = 均值差 / 合并标准差 = {mean_diff:.2f} / {pooled_std:.2f} = {observed_d:.3f}")
    print(f"  解释：新用户比老用户平均高 {mean_diff:.2f} 元，相当于 {observed_d:.3f} 个标准差")

    # CI 是否包含 0
    print(f"\n显著性判断：")
    if ci_low > 0:
        print(f"  95% CI 不包含 0 → 新用户显著高于老用户")
    elif ci_high < 0:
        print(f"  95% CI 不包含 0 → 新用户显著低于老用户")
    else:
        print(f"  95% CI 包含 0 → 差异不显著")

    # 效应量的稳定性
    print(f"\n效应量的稳定性：")
    if ci_low > 0.2:
        print(f"  即使 CI 下界也 > 0.2 → 效应稳定且有意义")
    elif ci_low > 0:
        print(f"  CI 下界 > 0 但 < 0.2 → 有统计显著性，但实际意义可能有限")
    else:
        print(f"  CI 包含 0 或负值 → 效应不稳定，需要更多数据")

    print(f"\n总结：")
    print(f"  - p 值告诉你'是否显著'")
    print(f"  - Cohen's d 告诉你'效应有多大'")
    print(f"  - Cohen's d 的 CI 告诉你'效应有多确定'")
    print(f"  - 三者结合才能完整回答'差异是否重要'")


if __name__ == "__main__":
    main()

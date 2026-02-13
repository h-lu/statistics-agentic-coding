"""
Week 08 Starter Code: 区间估计与重采样

这是作业的基础解决方案模板，学生可以在此基础上完成作业。

本周核心技能：
1. 计算并解释置信区间（频率学派）
2. 实现 Bootstrap 方法估计统计量的不确定性
3. 实现置换检验（分布无关的假设检验）
4. 区分置信区间与贝叶斯可信区间
5. 审查 AI 生成的推断报告
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


# =============================================================================
# 题 1：置信区间计算与解释
# =============================================================================

def ci_mean(data, conf_level=0.95):
    """
    计算均值的置信区间（t 公式）。

    参数：
    - data: 数值型数据
    - conf_level: 置信水平（默认 0.95）

    返回：
    - mean: 样本均值
    - ci_low: CI 下界
    - ci_high: CI 上界
    """
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    se = std / np.sqrt(n)

    # t 临界值
    df = n - 1
    t_critical = stats.t.ppf((1 + conf_level) / 2, df)

    # 置信区间
    margin = t_critical * se
    ci_low = mean - margin
    ci_high = mean + margin

    return mean, ci_low, ci_high


# =============================================================================
# 题 2：Bootstrap 实现
# =============================================================================

def bootstrap_ci(data, stat_func=np.mean, n_bootstrap=10000,
                 conf_level=0.95, seed=42):
    """
    用 Bootstrap 构造统计量的置信区间。

    参数：
    - data: 原始数据
    - stat_func: 统计量函数（默认是均值）
    - n_bootstrap: Bootstrap 次数
    - conf_level: 置信水平
    - seed: 随机种子

    返回：
    - observed: 观测统计量
    - ci_low: CI 下界
    - ci_high: CI 上界
    - boot_stats: Bootstrap 统计量列表
    """
    np.random.seed(seed)
    n = len(data)

    # 观测统计量
    observed = stat_func(data)

    # Bootstrap 重采样
    boot_stats = []
    for _ in range(n_bootstrap):
        boot_sample = np.random.choice(data, size=n, replace=True)
        boot_stats.append(stat_func(boot_sample))

    # 计算 CI（百分位数法）
    alpha = 1 - conf_level
    ci_low = np.percentile(boot_stats, 100 * alpha / 2)
    ci_high = np.percentile(boot_stats, 100 * (1 - alpha / 2))

    return observed, ci_low, ci_high, boot_stats


# =============================================================================
# 题 3：置换检验
# =============================================================================

def permutation_test(group1, group2, n_permutations=10000,
                    stat_func=np.mean, seed=42):
    """
    执行置换检验。

    参数：
    - group1, group2: 两组数据
    - n_permutations: 置换次数
    - stat_func: 统计量函数（默认是均值）
    - seed: 随机种子

    返回：
    - observed_stat: 观测统计量
    - p_value: p 值
    - perm_stats: 置换统计量列表
    """
    np.random.seed(seed)
    n1, n2 = len(group1), len(group2)

    # 合并数据
    combined = np.concatenate([group1, group2])

    # 观测统计量
    observed_stat = stat_func(group1) - stat_func(group2)

    # 置换检验
    perm_stats = []
    for _ in range(n_permutations):
        permuted = np.random.permutation(combined)
        perm_group1 = permuted[:n1]
        perm_group2 = permuted[n1:n1+n2]
        perm_stat = stat_func(perm_group1) - stat_func(perm_group2)
        perm_stats.append(perm_stat)

    # 计算 p 值（双尾）
    perm_stats = np.array(perm_stats)
    if observed_stat >= 0:
        p_value = np.mean(perm_stats >= observed_stat) + \
                  np.mean(perm_stats <= -observed_stat)
    else:
        p_value = np.mean(perm_stats <= observed_stat) + \
                  np.mean(perm_stats >= -observed_stat)

    return observed_stat, p_value, perm_stats


# =============================================================================
# 示例使用
# =============================================================================

if __name__ == "__main__":
    # 模拟数据
    np.random.seed(42)
    group_a = np.random.normal(loc=100, scale=15, size=50)
    group_b = np.random.normal(loc=95, scale=15, size=50)

    print("=" * 50)
    print("Week 08 示例：区间估计与重采样")
    print("=" * 50)

    # 1. 置信区间
    print("\n【1】置信区间（t 公式）")
    mean_a, ci_low_a, ci_high_a = ci_mean(group_a)
    print(f"Group A 均值: {mean_a:.2f}, 95% CI: [{ci_low_a:.2f}, {ci_high_a:.2f}]")

    mean_b, ci_low_b, ci_high_b = ci_mean(group_b)
    print(f"Group B 均值: {mean_b:.2f}, 95% CI: [{ci_low_b:.2f}, {ci_high_b:.2f}]")

    # 2. Bootstrap
    print("\n【2】Bootstrap 均值差 CI")
    diff = np.mean(group_a) - np.mean(group_b)
    obs, ci_low, ci_high, _ = bootstrap_ci(
        np.concatenate([group_a, group_b]),
        stat_func=lambda x: np.mean(x[:50]) - np.mean(x[50:])
    )
    print(f"均值差: {diff:.2f}")
    print(f"Bootstrap 95% CI: [{ci_low:.2f}, {ci_high:.2f}]")

    # 3. 置换检验
    print("\n【3】置换检验")
    obs_diff, p_val, _ = permutation_test(group_a, group_b)
    print(f"观测均值差: {obs_diff:.2f}")
    print(f"p 值: {p_val:.4f}")
    print(f"结论: {'拒绝 H0' if p_val < 0.05 else '无法拒绝 H0'}")

    print("\n" + "=" * 50)
    print("完成！请在基础上完成作业要求。")
    print("=" * 50)

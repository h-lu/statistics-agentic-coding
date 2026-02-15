"""
Week 08 作业参考实现：区间估计与重采样

本文件是作业的参考实现，供学生在遇到困难时查看。
只实现基础作业要求，不覆盖进阶/挑战部分。

运行方式：python3 chapters/week_08/starter_code/solution.py
预期输出：
  - 完成 calculate_confidence_interval 函数：计算均值和 95% CI
  - 完成 bootstrap_mean 函数：Bootstrap 均值估计
  - 完成 bootstrap_ci 函数：用 Bootstrap 计算 CI
  - 完成 permutation_test 函数：执行置换检验
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from typing import Callable, Union, Optional


def calculate_confidence_interval(data: np.ndarray,
                                  confidence: float = 0.95) -> dict:
    """
    计算均值及其置信区间

    参数:
        data: 数据数组
        confidence: 置信水平（默认 0.95）

    返回:
        dict 包含:
            - point_estimate: 点估计（均值）
            - ci_low: CI 下界
            - ci_high: CI 上界
            - standard_error: 标准误
            - n: 样本量
            - confidence: 置信水平

    示例:
        >>> data = np.array([3.2, 2.8, 3.5, 4.1, 2.9, 3.3, 3.0, 3.7])
        >>> result = calculate_confidence_interval(data)
        >>> print(f"均值: {result['point_estimate']:.2f}")
    """
    data = np.asarray(data)

    if len(data) == 0:
        raise ValueError("数据不能为空")

    # 计算均值
    mean = float(np.mean(data))

    # 单值数据特殊处理
    if len(data) == 1:
        return {
            'point_estimate': mean,
            'ci_low': mean,
            'ci_high': mean,
            'standard_error': 0.0,
            'n': 1,
            'confidence': confidence
        }

    # 计算标准误
    se = float(stats.sem(data))

    # 计算自由度
    df = len(data) - 1

    # 使用 t 分布计算 CI
    # 处理常量数据（SE = 0）的情况
    if se == 0 or np.isnan(se):
        ci_low = mean
        ci_high = mean
    else:
        ci_low, ci_high = stats.t.interval(confidence, df=df, loc=mean, scale=se)

    return {
        'point_estimate': mean,
        'ci_low': float(ci_low),
        'ci_high': float(ci_high),
        'standard_error': se,
        'n': len(data),
        'confidence': confidence
    }


def interpret_confidence_interval(ci_result: Union[dict, tuple]) -> str:
    """
    解释置信区间的含义

    参数:
        ci_result: calculate_confidence_interval 的返回结果

    返回:
        解释字符串

    示例:
        >>> result = calculate_confidence_interval(data)
        >>> print(interpret_confidence_interval(result))
    """
    if isinstance(ci_result, tuple):
        ci_low, ci_high = ci_result
        mean = None
        confidence = 0.95
    else:
        ci_low = ci_result.get('ci_low', ci_result.get('lower'))
        ci_high = ci_result.get('ci_high', ci_result.get('upper'))
        mean = ci_result.get('point_estimate', ci_result.get('mean'))
        confidence = ci_result.get('confidence', 0.95)

    confidence_pct = int(confidence * 100)

    interpretation = (
        f"置信区间 [{ci_low:.2f}, {ci_high:.2f}] 的正确解释：\n"
        f"如果我们从同一总体中重复抽样很多次，并每次计算 {confidence_pct}% CI，\n"
        f"那么大约有 {confidence_pct}% 的区间会包含真实的总体参数。\n"
        f"注意：不能说'参数有 {confidence_pct}% 概率落在这个区间内'，\n"
        f"因为参数是固定值，不是随机变量。"
    )

    if mean is not None:
        interpretation += f"\n本次样本的点估计（均值）为 {mean:.2f}。"

    return interpretation


def bootstrap_mean(data: np.ndarray,
                   n_bootstrap: int = 10000,
                   random_state: Optional[int] = None) -> dict:
    """
    用 Bootstrap 方法估计均值和标准误

    参数:
        data: 数据数组
        n_bootstrap: Bootstrap 次数（默认 10000）
        random_state: 随机种子（默认 None）

    返回:
        dict 包含:
            - bootstrap_mean: Bootstrap 均值
            - mean: 同 bootstrap_mean（兼容别名）
            - standard_error: Bootstrap 标准误
            - se: 同 standard_error（兼容别名）

    示例:
        >>> data = np.array([3.2, 2.8, 3.5, 4.1, 2.9, 3.3, 3.0, 3.7])
        >>> result = bootstrap_mean(data, n_bootstrap=1000, random_state=42)
        >>> print(f"Bootstrap 均值: {result['bootstrap_mean']:.2f}")
    """
    data = np.asarray(data)

    if len(data) == 0:
        raise ValueError("数据不能为空")

    if random_state is not None:
        np.random.seed(random_state)

    # Bootstrap 重采样
    n = len(data)
    boot_means = []
    for _ in range(n_bootstrap):
        # 有放回抽样
        boot_sample = np.random.choice(data, size=n, replace=True)
        boot_means.append(np.mean(boot_sample))

    boot_means = np.array(boot_means)

    # 计算统计量
    bootstrap_mean_val = float(np.mean(boot_means))
    standard_error = float(np.std(boot_means, ddof=1))

    return {
        'bootstrap_mean': bootstrap_mean_val,
        'mean': bootstrap_mean_val,
        'standard_error': standard_error,
        'se': standard_error
    }


def bootstrap_ci(data: np.ndarray,
                 statistic: Union[Callable, str] = np.mean,
                 confidence: float = 0.95,
                 n_bootstrap: int = 10000,
                 method: str = 'percentile',
                 random_state: Optional[int] = None) -> dict:
    """
    用 Bootstrap 方法计算置信区间

    参数:
        data: 数据数组
        statistic: 统计量函数或字符串（'mean', 'median'）
        confidence: 置信水平（默认 0.95）
        n_bootstrap: Bootstrap 次数（默认 10000）
        method: CI 方法（'percentile'）
        random_state: 随机种子（默认 None）

    返回:
        dict 包含:
            - point_estimate: 点估计
            - ci_low: CI 下界
            - ci_high: CI 上界
            - lower/ci_high: 别名

    示例:
        >>> data = np.array([3.2, 2.8, 3.5, 4.1, 2.9, 3.3, 3.0, 3.7])
        >>> result = bootstrap_ci(data, random_state=42)
        >>> print(f"95% CI: [{result['ci_low']:.2f}, {result['ci_high']:.2f}]")
    """
    data = np.asarray(data)

    if len(data) == 0:
        raise ValueError("数据不能为空")

    if random_state is not None:
        np.random.seed(random_state)

    # 处理 statistic 参数
    if isinstance(statistic, str):
        if statistic == 'median':
            stat_func = np.median
        elif statistic == 'mean':
            stat_func = np.mean
        else:
            raise ValueError(f"不支持的统计量: {statistic}")
    else:
        stat_func = statistic

    # 计算点估计
    point_estimate = float(stat_func(data))

    # Bootstrap 重采样
    n = len(data)
    boot_stats = []
    for _ in range(n_bootstrap):
        # 有放回抽样
        boot_sample = np.random.choice(data, size=n, replace=True)
        boot_stats.append(stat_func(boot_sample))

    boot_stats = np.array(boot_stats)

    # 计算分位数 CI
    alpha = 1 - confidence
    ci_low = float(np.percentile(boot_stats, alpha / 2 * 100))
    ci_high = float(np.percentile(boot_stats, (1 - alpha / 2) * 100))

    return {
        'point_estimate': point_estimate,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'lower': ci_low,
        'upper': ci_high
    }


def bootstrap_ci_bca(data: np.ndarray,
                     n_bootstrap: int = 10000,
                     confidence: float = 0.95,
                     random_state: Optional[int] = None) -> dict:
    """
    用 BCa (Bias-Corrected and Accelerated) Bootstrap 方法计算置信区间

    BCa 方法对偏差和加速进行校正，通常比 Percentile Bootstrap 更准确。

    参数:
        data: 数据数组
        n_bootstrap: Bootstrap 次数（默认 10000）
        confidence: 置信水平（默认 0.95）
        random_state: 随机种子（默认 None）

    返回:
        dict 包含:
            - point_estimate: 点估计
            - ci_low: CI 下界
            - ci_high: CI 上界

    示例:
        >>> data = np.array([3.2, 2.8, 3.5, 4.1, 2.9, 3.3, 3.0, 3.7])
        >>> result = bootstrap_ci_bca(data, random_state=42)
    """
    data = np.asarray(data)

    if len(data) == 0:
        raise ValueError("数据不能为空")

    if random_state is not None:
        np.random.seed(random_state)

    n = len(data)

    # 计算点估计（均值）
    theta_hat = np.mean(data)

    # Bootstrap 重采样
    boot_means = []
    for _ in range(n_bootstrap):
        boot_sample = np.random.choice(data, size=n, replace=True)
        boot_means.append(np.mean(boot_sample))

    boot_means = np.array(boot_means)

    # 计算偏差校正因子 z0
    # z0 = Phi^(-1)(proportion of boot_means < theta_hat)
    prop_less = np.mean(boot_means < theta_hat)
    # 避免极端值
    prop_less = np.clip(prop_less, 0.001, 0.999)
    z0 = stats.norm.ppf(prop_less)

    # 计算加速因子 a（使用 jackknife）
    jackknife_means = []
    for i in range(n):
        jack_sample = np.delete(data, i)
        jackknife_means.append(np.mean(jack_sample))

    jackknife_means = np.array(jackknife_means)
    jack_mean = np.mean(jackknife_means)

    # 加速因子
    numerator = np.sum((jack_mean - jackknife_means) ** 3)
    denominator = np.sum((jack_mean - jackknife_means) ** 2) ** 1.5

    if denominator > 0:
        a = numerator / (6 * denominator)
    else:
        a = 0

    # 计算 BCa 校正的分位数
    alpha = 1 - confidence
    z_alpha_lower = stats.norm.ppf(alpha / 2)
    z_alpha_upper = stats.norm.ppf(1 - alpha / 2)

    # BCa 校正公式
    def bca_quantile(z_alpha):
        adjusted = z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha))
        return stats.norm.cdf(adjusted)

    alpha1 = bca_quantile(z_alpha_lower)
    alpha2 = bca_quantile(z_alpha_upper)

    # 确保 alpha 值在有效范围内
    alpha1 = np.clip(alpha1, 0.001, 0.999)
    alpha2 = np.clip(alpha2, 0.001, 0.999)

    ci_low = float(np.percentile(boot_means, alpha1 * 100))
    ci_high = float(np.percentile(boot_means, alpha2 * 100))

    return {
        'point_estimate': float(theta_hat),
        'ci_low': ci_low,
        'ci_high': ci_high,
        'lower': ci_low,
        'upper': ci_high
    }


def bootstrap_median(data: np.ndarray,
                     n_bootstrap: int = 10000,
                     random_state: Optional[int] = None) -> dict:
    """
    用 Bootstrap 方法估计中位数

    参数:
        data: 数据数组
        n_bootstrap: Bootstrap 次数（默认 10000）
        random_state: 随机种子（默认 None）

    返回:
        dict 包含:
            - bootstrap_median: Bootstrap 中位数
            - median: 同 bootstrap_median（兼容别名）

    示例:
        >>> data = np.array([3.2, 2.8, 3.5, 4.1, 2.9, 3.3, 3.0, 3.7])
        >>> result = bootstrap_median(data, random_state=42)
    """
    data = np.asarray(data)

    if len(data) == 0:
        raise ValueError("数据不能为空")

    if random_state is not None:
        np.random.seed(random_state)

    # Bootstrap 重采样
    n = len(data)
    boot_medians = []
    for _ in range(n_bootstrap):
        boot_sample = np.random.choice(data, size=n, replace=True)
        boot_medians.append(np.median(boot_sample))

    boot_medians = np.array(boot_medians)
    bootstrap_median_val = float(np.mean(boot_medians))

    return {
        'bootstrap_median': bootstrap_median_val,
        'median': bootstrap_median_val
    }


def permutation_test(group1: np.ndarray,
                     group2: np.ndarray,
                     n_permutations: int = 10000,
                     alternative: str = 'two-sided',
                     random_state: Optional[int] = None) -> dict:
    """
    执行置换检验，比较两组均值是否有差异

    参数:
        group1, group2: 两组数据
        n_permutations: 置换次数（默认 10000）
        alternative: 备择假设类型 ('two-sided', 'greater', 'less')
        random_state: 随机种子（默认 None）

    返回:
        dict 包含:
            - observed_difference: 观测差异
            - p_value: p 值
            - n_permutations: 置换次数

    示例:
        >>> group1 = np.array([3.2, 2.8, 3.5, 4.1])
        >>> group2 = np.array([3.8, 4.2, 3.9, 4.5])
        >>> result = permutation_test(group1, group2, random_state=42)
        >>> print(f"差异: {result['observed_difference']:.3f}, p = {result['p_value']:.4f}")
    """
    group1 = np.asarray(group1)
    group2 = np.asarray(group2)

    if len(group1) == 0 or len(group2) == 0:
        raise ValueError("两组数据都不能为空")

    if random_state is not None:
        np.random.seed(random_state)

    # 计算观测差异
    obs_diff = float(np.mean(group2) - np.mean(group1))

    # 合并数据
    combined = np.concatenate([group1, group2])
    n1 = len(group1)

    # 置换检验
    perm_diffs = []
    for _ in range(n_permutations):
        # 打乱标签
        permuted = np.random.permutation(combined)
        # 重新分组
        perm_group1 = permuted[:n1]
        perm_group2 = permuted[n1:]
        # 计算差异
        perm_diffs.append(np.mean(perm_group2) - np.mean(perm_group1))

    perm_diffs = np.array(perm_diffs)

    # 计算 p 值（根据备择假设类型）
    if alternative == 'two-sided':
        p_value = float(np.mean(np.abs(perm_diffs) >= np.abs(obs_diff)))
    elif alternative == 'greater':
        p_value = float(np.mean(perm_diffs >= obs_diff))
    elif alternative == 'less':
        p_value = float(np.mean(perm_diffs <= obs_diff))
    else:
        raise ValueError(f"不支持的备择假设类型: {alternative}")

    return {
        'observed_difference': obs_diff,
        'p_value': p_value,
        'n_permutations': n_permutations
    }


def permutation_test_ci(group1: np.ndarray,
                        group2: np.ndarray,
                        n_permutations: int = 10000,
                        n_bootstrap: int = 5000,
                        random_state: Optional[int] = None) -> dict:
    """
    置换检验 + Bootstrap 置信区间

    参数:
        group1, group2: 两组数据
        n_permutations: 置换次数（默认 10000）
        n_bootstrap: Bootstrap 次数（默认 5000）
        random_state: 随机种子（默认 None）

    返回:
        dict 包含:
            - observed_difference: 观测差异
            - p_value: p 值
            - ci_low: 差异的 CI 下界
            - ci_high: 差异的 CI 上界

    示例:
        >>> group1 = np.array([3.2, 2.8, 3.5, 4.1])
        >>> group2 = np.array([3.8, 4.2, 3.9, 4.5])
        >>> result = permutation_test_ci(group1, group2, random_state=42)
    """
    group1 = np.asarray(group1)
    group2 = np.asarray(group2)

    if len(group1) == 0 or len(group2) == 0:
        raise ValueError("两组数据都不能为空")

    if random_state is not None:
        np.random.seed(random_state)

    # 计算观测差异
    obs_diff = float(np.mean(group2) - np.mean(group1))

    # 置换检验
    combined = np.concatenate([group1, group2])
    n1 = len(group1)

    perm_diffs = []
    for _ in range(n_permutations):
        permuted = np.random.permutation(combined)
        perm_diffs.append(np.mean(permuted[n1:]) - np.mean(permuted[:n1]))

    perm_diffs = np.array(perm_diffs)
    p_value = float(np.mean(np.abs(perm_diffs) >= np.abs(obs_diff)))

    # Bootstrap CI for difference
    boot_diffs = []
    for _ in range(n_bootstrap):
        boot_g1 = np.random.choice(group1, size=len(group1), replace=True)
        boot_g2 = np.random.choice(group2, size=len(group2), replace=True)
        boot_diffs.append(np.mean(boot_g2) - np.mean(boot_g1))

    boot_diffs = np.array(boot_diffs)
    ci_low = float(np.percentile(boot_diffs, 2.5))
    ci_high = float(np.percentile(boot_diffs, 97.5))

    return {
        'observed_difference': obs_diff,
        'p_value': p_value,
        'ci_low': ci_low,
        'ci_high': ci_high
    }


def compare_groups_with_uncertainty(group1: np.ndarray,
                                    group2: np.ndarray,
                                    n_bootstrap: int = 5000,
                                    random_state: Optional[int] = None) -> dict:
    """
    比较两组数据，包含不确定性量化

    参数:
        group1, group2: 两组数据
        n_bootstrap: Bootstrap 次数（默认 5000）
        random_state: 随机种子（默认 None）

    返回:
        dict 包含:
            - mean_difference: 均值差异
            - ci_low: 差异的 CI 下界
            - ci_high: 差异的 CI 上界
            - group1_mean: 组1 均值
            - group2_mean: 组2 均值
            - permutation_p: 置换检验 p 值

    示例:
        >>> group1 = np.array([3.2, 2.8, 3.5, 4.1])
        >>> group2 = np.array([3.8, 4.2, 3.9, 4.5])
        >>> result = compare_groups_with_uncertainty(group1, group2, random_state=42)
    """
    group1 = np.asarray(group1)
    group2 = np.asarray(group2)

    if len(group1) == 0 or len(group2) == 0:
        raise ValueError("两组数据都不能为空")

    if random_state is not None:
        np.random.seed(random_state)

    # 计算均值
    mean1 = float(np.mean(group1))
    mean2 = float(np.mean(group2))
    mean_diff = mean1 - mean2  # group1 - group2 (e.g., new - active)

    # Bootstrap CI for difference
    boot_diffs = []
    for _ in range(n_bootstrap):
        boot_g1 = np.random.choice(group1, size=len(group1), replace=True)
        boot_g2 = np.random.choice(group2, size=len(group2), replace=True)
        boot_diffs.append(np.mean(boot_g1) - np.mean(boot_g2))  # group1 - group2

    boot_diffs = np.array(boot_diffs)
    ci_low = float(np.percentile(boot_diffs, 2.5))
    ci_high = float(np.percentile(boot_diffs, 97.5))

    # 置换检验
    combined = np.concatenate([group1, group2])
    n1 = len(group1)

    perm_diffs = []
    for _ in range(5000):
        permuted = np.random.permutation(combined)
        perm_diffs.append(np.mean(permuted[:n1]) - np.mean(permuted[n1:]))  # group1 - group2

    perm_diffs = np.array(perm_diffs)
    p_value = float(np.mean(np.abs(perm_diffs) >= np.abs(mean_diff)))

    return {
        'mean_difference': mean_diff,
        'point_diff': mean_diff,
        'difference': mean_diff,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'group1_mean': mean1,
        'group2_mean': mean2,
        'permutation_p': p_value,
        'p_value': p_value,
        'p': p_value
    }


def add_ci_to_estimate(data: np.ndarray,
                       confidence: float = 0.95) -> dict:
    """
    为点估计添加置信区间

    参数:
        data: 数据数组
        confidence: 置信水平（默认 0.95）

    返回:
        dict 包含:
            - estimate: 点估计
            - ci_low: CI 下界
            - ci_high: CI 上界
            - confidence: 置信水平

    示例:
        >>> data = np.array([3.2, 2.8, 3.5, 4.1, 2.9, 3.3, 3.0, 3.7])
        >>> result = add_ci_to_estimate(data)
        >>> print(f"估计: {result['estimate']:.2f} [{result['ci_low']:.2f}, {result['ci_high']:.2f}]")
    """
    data = np.asarray(data)

    if len(data) == 0:
        raise ValueError("数据不能为空")

    # 计算 CI
    mean = float(np.mean(data))

    if len(data) == 1:
        return {
            'estimate': mean,
            'ci_low': mean,
            'ci_high': mean,
            'confidence': confidence
        }

    se = float(stats.sem(data))
    df = len(data) - 1

    # 处理常量数据（SE = 0）的情况
    if se == 0 or np.isnan(se):
        ci_low = mean
        ci_high = mean
    else:
        ci_low, ci_high = stats.t.interval(confidence, df=df, loc=mean, scale=se)

    return {
        'estimate': mean,
        'point_estimate': mean,
        'mean': mean,
        'ci_low': float(ci_low),
        'ci_high': float(ci_high),
        'confidence': confidence
    }


# 兼容旧函数名
def confidence_interval(data: np.ndarray,
                        confidence: float = 0.95) -> tuple:
    """
    计算均值及其置信区间（兼容旧 API）

    返回: (均值, CI下界, CI上界)
    """
    result = calculate_confidence_interval(data, confidence)
    return (result['point_estimate'], result['ci_low'], result['ci_high'])


def format_result_with_ci(mean: float, ci_low: float, ci_high: float,
                          label: str = "均值") -> str:
    """
    格式化结果为报告风格

    参数:
        mean: 点估计
        ci_low, ci_high: 置信区间
        label: 标签

    返回:
        格式化的字符串
    """
    ci_width = ci_high - ci_low
    return f"{label}: {mean:.2f} [95% CI: {ci_low:.2f}, {ci_high:.2f}] (宽度: {ci_width:.2f})"


def demonstrate_usage() -> None:
    """演示函数使用"""
    print("=" * 70)
    print("Week 08 作业参考实现演示")
    print("=" * 70)

    # 示例数据
    np.random.seed(42)
    data = np.random.normal(loc=3.2, scale=1.5, size=100)

    # 1. 计算置信区间（理论方法）
    print("\n1. 计算置信区间（理论 t 分布方法）")
    result = calculate_confidence_interval(data)
    print(f"   均值: {result['point_estimate']:.2f}")
    print(f"   95% CI: [{result['ci_low']:.2f}, {result['ci_high']:.2f}]")
    print(f"   标准误: {result['standard_error']:.3f}")

    # 2. 用 Bootstrap 估计均值
    print("\n2. Bootstrap 均值估计")
    boot_result = bootstrap_mean(data, n_bootstrap=1000, random_state=42)
    print(f"   Bootstrap 均值: {boot_result['bootstrap_mean']:.2f}")
    print(f"   Bootstrap SE: {boot_result['standard_error']:.3f}")

    # 3. 用 Bootstrap 计算置信区间
    print("\n3. 用 Bootstrap 计算置信区间")
    boot_ci = bootstrap_ci(data, random_state=42)
    print(f"   95% CI: [{boot_ci['ci_low']:.2f}, {boot_ci['ci_high']:.2f}]")

    # 4. 置换检验
    print("\n4. 置换检验（两组比较）")
    group1 = np.random.normal(loc=3.0, scale=1.5, size=50)
    group2 = np.random.normal(loc=3.5, scale=1.5, size=50)
    perm_result = permutation_test(group1, group2, random_state=42)
    print(f"   观测差异: {perm_result['observed_difference']:.3f}")
    print(f"   置换检验 p 值: {perm_result['p_value']:.4f}")
    if perm_result['p_value'] < 0.05:
        print(f"   结论: p < 0.05，差异显著")
    else:
        print(f"   结论: p >= 0.05，差异不显著")

    # 5. 解释 CI
    print("\n5. 解释置信区间")
    ci_result = calculate_confidence_interval(data)
    print(interpret_confidence_interval(ci_result))

    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print("\n可用函数：")
    print("  1. calculate_confidence_interval: 使用 t 分布计算 CI")
    print("  2. bootstrap_mean: Bootstrap 均值和标准误")
    print("  3. bootstrap_ci: Bootstrap 置信区间")
    print("  4. bootstrap_ci_bca: BCa Bootstrap 置信区间")
    print("  5. bootstrap_median: Bootstrap 中位数")
    print("  6. permutation_test: 置换检验")
    print("  7. permutation_test_ci: 置换检验 + Bootstrap CI")
    print("  8. compare_groups_with_uncertainty: 完整的组间比较")
    print("  9. add_ci_to_estimate: 为估计添加 CI")
    print("  10. interpret_confidence_interval: 解释 CI 含义")
    print()


if __name__ == "__main__":
    demonstrate_usage()

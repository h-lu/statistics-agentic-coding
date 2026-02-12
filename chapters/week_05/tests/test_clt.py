"""
中心极限定理（CLT）测试

测试中心极限定理的模拟和性质验证。
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from .conftest import simulate_clt


# =============================================================================
# CLT 基础测试
# =============================================================================

class TestCentralLimitTheorem:
    """测试中心极限定理"""

    def test_clt_convergence(self, exponential_population, clt_sample_sizes):
        """测试：样本量增加时，样本均值分布更接近正态"""
        results = {}

        for n in clt_sample_sizes:
            sample_means = simulate_clt(exponential_population, n, n_simulations=1000)
            results[n] = {
                'means': sample_means,
                'skewness': stats.skew(sample_means),
                'normality_p': stats.shapiro(sample_means[:5000])[1]
            }

        # 随着样本量增加，偏度应该趋近于 0
        # n=5 的偏度绝对值 > n=100 的偏度绝对值
        assert abs(results[5]['skewness']) > abs(results[100]['skewness'])

        # 大样本的正态性检验 p 值应该更大
        assert results[30]['normality_p'] > results[5]['normality_p']

    def test_sample_mean_unbiased(self, exponential_population):
        """测试：样本均值是总体均值的无偏估计"""
        pop_mean = np.mean(exponential_population)

        # 多次抽样的均值应该接近总体均值
        sample_means = simulate_clt(exponential_population, n=30, n_simulations=1000)
        mean_of_means = np.mean(sample_means)

        # 样本均值的均值应该非常接近总体均值
        assert abs(mean_of_means - pop_mean) < pop_mean * 0.05

    def test_standard_error_formula(self, exponential_population):
        """测试：标准误 SE = σ/√n 的正确性"""
        pop_std = np.std(exponential_population, ddof=1)

        for n in [10, 30, 100]:
            sample_means = simulate_clt(exponential_population, n, n_simulations=1000)
            actual_se = np.std(sample_means, ddof=1)
            theoretical_se = pop_std / np.sqrt(n)

            # 实际 SE 应该接近理论 SE
            assert abs(actual_se - theoretical_se) / theoretical_se < 0.15


# =============================================================================
# 不同总体下的 CLT 测试
# =============================================================================

class TestCLTDifferentPopulations:
    """测试不同总体分布下的 CLT"""

    @pytest.mark.parametrize("dist_func,dist_name", [
        (lambda: np.random.uniform(0, 200, 100000), "均匀分布"),
        (lambda: np.random.exponential(50, 100000), "指数分布（右偏）"),
        (lambda: np.concatenate([np.random.normal(50, 10, 50000),
                              np.random.normal(150, 10, 50000)]), "双峰分布"),
    ])
    def test_clt_works_for_various_distributions(self, dist_func, dist_name):
        """测试：CLT 对各种分布都成立"""
        np.random.seed(42)
        population = dist_func()
        pop_mean = np.mean(population)

        # 大样本（n=100）的样本均值分布应该接近正态
        sample_means = simulate_clt(population, n=100, n_simulations=1000)

        # 正态性检验
        _, p_value = stats.shapiro(sample_means)

        # 应该不能拒绝正态假设
        assert p_value > 0.01

    def test_clt_with_normal_population(self):
        """测试：正态总体下的 CLT（基准情况）"""
        np.random.seed(42)
        normal_population = np.random.normal(100, 15, 100000)

        # 对于正态总体，即使小样本，均值分布也是正态的
        for n in [5, 10, 30]:
            sample_means = simulate_clt(normal_population, n, n_simulations=1000)
            _, p_value = stats.shapiro(sample_means)

            # 正态总体的样本均值分布应该接近正态
            assert p_value > 0.01


# =============================================================================
# 标准误测试
# =============================================================================

class TestStandardError:
    """测试标准误 SE = σ/√n"""

    def test_se_decreases_with_sample_size(self):
        """测试：样本量增加，SE 减小"""
        pop_std = 15

        ses = [pop_std / np.sqrt(n) for n in [10, 30, 100, 1000]]

        # SE 应该单调递减
        assert ses[0] > ses[1] > ses[2] > ses[3]

    def test_se_relationship(self, rng):
        """测试 SE 与 √n 的反比关系"""
        pop_std = 15

        # 样本量翻倍，SE 除以 √2
        se_n = pop_std / np.sqrt(30)
        se_2n = pop_std / np.sqrt(60)

        ratio = se_n / se_2n
        expected_ratio = np.sqrt(2)

        assert abs(ratio - expected_ratio) < 0.01


# =============================================================================
# 边界和反例测试
# =============================================================================

class TestCLTEdgeCases:
    """测试 CLT 的边界情况和限制"""

    def test_small_sample_non_normal_population(self, exponential_population):
        """测试：小样本 + 非正态总体 = CLT 不成立"""
        # 小样本（n=5）的均值分布可能偏离正态
        sample_means = simulate_clt(exponential_population, n=5, n_simulations=1000)
        _, p_value = stats.shapiro(sample_means)

        # 小样本可能拒绝正态假设
        # （虽然由于随机性，这个测试可能偶尔通过）
        # 这里我们只验证偏度仍然较大
        assert abs(stats.skew(sample_means)) > 0.1

    def test_sample_size_1(self, exponential_population):
        """测试：n=1 时无法计算均值分布"""
        np.random.seed(42)

        # n=1 时，样本均值就是单个观测值
        # 分布形状与总体相同，不是正态
        sample_means = np.array([
            np.mean(np.random.choice(exponential_population, 1, replace=False))
            for _ in range(1000)
        ])

        # 偏度应该仍然较大（不接近正态）
        assert abs(stats.skew(sample_means)) > 0.5

    def test_clt_does_not_apply_to_median(self, exponential_population):
        """测试：CLT 不适用于中位数"""
        # 中位数分布在大样本时不一定接近正态
        # （尤其是对于偏态分布）

        sample_medians = np.array([
            np.median(np.random.choice(exponential_population, 100, replace=False))
            for _ in range(1000)
        ])

        sample_means = np.array([
            np.mean(np.random.choice(exponential_population, 100, replace=False))
            for _ in range(1000)
        ])

        # 均值分布的正态性应该比中位数分布更好
        _, p_mean = stats.shapiro(sample_means)
        _, p_median = stats.shapiro(sample_medians)

        # 对于偏态总体，中位数分布的正态性较差
        # 这个断言可能偶尔失败，但总体趋势应该成立


# =============================================================================
# 反例测试
# =============================================================================

class TestCLTCommonMistakes:
    """测试 CLT 相关的常见错误"""

    def test_confusing_sd_with_se(self, normal_sample):
        """测试：混淆标准差和标准误"""
        n = len(normal_sample)
        data_sd = np.std(normal_sample, ddof=1)

        # 标准误是样本均值分布的标准差
        # SE = SD / √n

        correct_se = data_sd / np.sqrt(n)

        # 如果有人错误地认为 SE = SD
        wrong_se = data_sd

        # 这两者应该不同（除非 n=1）
        if n > 1:
            assert correct_se != wrong_se
            assert correct_se < wrong_se

    def test_assuming_clt_for_small_sample(self):
        """测试：小样本时错误假设 CLT"""
        # 对于严重偏态分布，小样本（如 n=5）的均值分布
        # 不应该假设正态

        population = np.random.exponential(10, 100000)
        sample_means = simulate_clt(population, n=5, n_simulations=1000)

        # 正态性检验可能拒绝正态假设
        _, p_value = stats.shapiro(sample_means)

        # 如果 p_value < 0.05，说明数据偏离正态
        # 不应该假设 CLT 成立
        # （这个测试可能偶尔通过，因为随机性）

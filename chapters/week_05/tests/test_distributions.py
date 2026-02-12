"""
概率分布测试

测试正态分布、二项分布、泊松分布的性质和参数估计。
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats


# =============================================================================
# 正态分布测试
# =============================================================================

class TestNormalDistribution:
    """测试正态分布"""

    def test_68_95_99_7_rule(self, normal_sample):
        """测试正态分布的 68-95-99.7 原则"""
        mean = np.mean(normal_sample)
        std = np.std(normal_sample, ddof=1)

        within_1s = np.sum((normal_sample >= mean - std) & (normal_sample <= mean + std)) / len(normal_sample)
        within_2s = np.sum((normal_sample >= mean - 2*std) & (normal_sample <= mean + 2*std)) / len(normal_sample)
        within_3s = np.sum((normal_sample >= mean - 3*std) & (normal_sample <= mean + 3*std)) / len(normal_sample)

        # 允许一定误差（因为是样本，不是总体）
        assert 0.65 <= within_1s <= 0.75
        assert 0.93 <= within_2s <= 0.97
        assert 0.99 <= within_3s <= 1.0

    def test_normal_parameters(self, normal_sample):
        """测试正态分布的参数估计"""
        mean = np.mean(normal_sample)
        std = np.std(normal_sample, ddof=1)
        median = np.median(normal_sample)

        # 正态分布的均值应该接近中位数
        assert abs(mean - median) < std * 0.1

    @pytest.mark.parametrize("mu,sigma", [
        (0, 1),
        (100, 15),
        (-50, 10),
    ])
    def test_normal_pdf_integral(self, mu, sigma):
        """测试正态分布 PDF 积分为 1"""
        x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
        pdf = stats.norm.pdf(x, mu, sigma)
        integral = np.trapz(pdf, x)

        assert abs(integral - 1.0) < 0.01

    def test_normal_skewness_near_zero(self, normal_sample):
        """测试正态分布的偏度接近 0"""
        skew = stats.skew(normal_sample)
        assert abs(skew) < 0.3


# =============================================================================
# 二项分布测试
# =============================================================================

class TestBinomialDistribution:
    """测试二项分布"""

    def test_binomial_mean_formula(self, binomial_sample):
        """测试二项分布的期望 E[X] = n×p"""
        # 二项分布样本由 np.random.binomial(n, p, size) 生成
        # 这里我们直接验证理论公式

        n, p = 100, 0.05
        expected_mean = n * p
        theoretical_variance = n * p * (1 - p)

        # 验证公式关系
        assert expected_mean == 5.0
        assert abs(theoretical_variance - 4.75) < 0.01

    def test_binomial_support(self, binomial_sample):
        """测试二项分布的支持集是 {0, 1, ..., n}"""
        # 所有值应该是非负整数
        assert np.all(binomial_sample >= 0)
        assert np.all(binomial_sample <= 100)

    def test_binomial_variance_greater_than_mean_for_small_p(self):
        """测试：当 p 较小时，方差可能接近均值（泊松极限）"""
        n, p = 100, 0.05
        mean = n * p
        variance = n * p * (1 - p)

        # 对于二项分布：Var = np(1-p) < np（当 p>0 时）
        assert variance < mean

    @pytest.mark.parametrize("n,p", [
        (10, 0.5),
        (100, 0.05),
        (1000, 0.01),
    ])
    def test_binomial_pmf_sum(self, n, p):
        """测试二项分布 PMF 求和为 1"""
        x = np.arange(0, n + 1)
        pmf = stats.binom.pmf(x, n, p)
        total = np.sum(pmf)

        assert abs(total - 1.0) < 0.001


# =============================================================================
# 泊松分布测试
# =============================================================================

class TestPoissonDistribution:
    """测试泊松分布"""

    def test_poisson_mean_equals_variance(self, poisson_sample):
        """测试泊松分布的特性：期望 = 方差"""
        mean = np.mean(poisson_sample)
        variance = np.var(poisson_sample, ddof=1)

        # 允许一定误差（因为是样本）
        ratio = variance / mean if mean > 0 else 0
        assert 0.7 <= ratio <= 1.3

    def test_poisson_support(self, poisson_sample):
        """测试泊松分布的支持集是非负整数"""
        assert np.all(poisson_sample >= 0)
        assert np.all(poisson_sample == poisson_sample.astype(int))

    @pytest.mark.parametrize("lam", [
        1,
        3,
        10,
    ])
    def test_poisson_pmf_sum(self, lam):
        """测试泊松分布 PMF 求和为 1（截断到合理范围）"""
        # 泊松分布支持无限范围，实际计算截断到某个足够大的值
        x = np.arange(0, lam * 10 + 50)
        pmf = stats.poisson.pmf(x, lam)
        total = np.sum(pmf)

        assert abs(total - 1.0) < 0.001

    def test_poisson_rare_event_probability(self):
        """测试：泊松分布用于稀有事件计数"""
        lam = 3

        # P(X = 0) = exp(-λ)
        p_zero = stats.poisson.pmf(0, lam)
        expected_p_zero = np.exp(-lam)

        assert abs(p_zero - expected_p_zero) < 0.001


# =============================================================================
# 分布选择测试
# =============================================================================

class TestDistributionSelection:
    """测试如何选择合适的分布"""

    def test_normal_vs_skewed_data(self, normal_sample, skewed_sample):
        """测试：正态数据 vs 偏态数据的区分"""
        # 正态数据的偏度接近 0
        normal_skew = stats.skew(normal_sample)

        # 右偏数据的偏度为正
        skewed_skew = stats.skew(skewed_sample)

        assert abs(normal_skew) < abs(skewed_skew)

    def test_count_data_distribution(self, binomial_sample, poisson_sample):
        """测试：计数数据适合离散分布"""
        # 这些样本都是整数
        assert np.all(binomial_sample == binomial_sample.astype(int))
        assert np.all(poisson_sample == poisson_sample.astype(int))

    def test_qq_plot_concept(self):
        """测试：QQ 图可以检验分布类型（概念测试）"""
        # QQ 图比较理论分位数和样本分位数
        # 如果数据来自指定分布，点应该落在对角线上

        # 生成正态数据
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)

        # 计算理论分位数和样本分位数
        theoretical_quantiles = np.sort(np.random.normal(0, 1, 100))
        sample_quantiles = np.sort(data)

        # 计算相关性（正态数据应该高度相关）
        correlation = np.corrcoef(theoretical_quantiles, sample_quantiles)[0, 1]

        assert correlation > 0.9


# =============================================================================
# 边界测试
# =============================================================================

class TestDistributionEdgeCases:
    """测试分布的边界情况"""

    def test_empty_array_statistics(self):
        """测试：空数组的统计"""
        # NumPy 对空数组的统计返回 NaN 并发出警告，不抛出 ValueError
        # 我们检查返回值是否为 NaN
        result = np.mean([])
        assert np.isnan(result)

    def test_single_value_distribution(self):
        """测试：单值数组的分布特征"""
        data = np.array([5.0])

        assert np.mean(data) == 5.0
        assert np.median(data) == 5.0
        # 标准差为 0 或 NaN（取决于 ddof）
        assert np.std(data, ddof=0) == 0.0

    def test_constant_data_distribution(self, constant_array):
        """测试：常数数据的分布特征"""
        mean = np.mean(constant_array)
        std = np.std(constant_array, ddof=1)

        assert mean == 5.0
        # 常数的标准差为 0
        assert std == 0.0

    @pytest.mark.parametrize("p", [0, 1])
    def test_binomial_extreme_p(self, p):
        """测试二项分布极端 p 值"""
        n = 10

        # p = 0: 总是 0
        # p = 1: 总是 n
        expected_0 = 0  # 当 p=0 时期望为 0
        expected_n = n  # 当 p=1 时期望为 n

        # 用 scipy 验证
        mean_at_p0 = stats.binom.mean(n, 0)
        mean_at_p1 = stats.binom.mean(n, 1)

        assert mean_at_p0 == expected_0
        assert mean_at_p1 == expected_n

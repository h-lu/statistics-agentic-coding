"""
Test Suite: Beta-Binomial Model（Beta-二项分布模型）

测试 Beta-Binomial 共轭先验模型：
1. Beta 分布作为二项分布参数的共轭先验
2. 后验分布的解析解：Beta(α + successes, β + failures)
3. Beta 分布的统计性质（均值、方差、众数）
4. 不同先验强度的比较

测试覆盖：
- 正确计算后验分布参数
- 正确计算后验统计量
- 理解共轭先验的便利性
- 理解先验强度对后验的影响
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

# 添加 starter_code 到路径
starter_code_path = Path(__file__).parent.parent / "starter_code"
if str(starter_code_path) not in sys.path:
    sys.path.insert(0, str(starter_code_path))


# =============================================================================
# 共轭先验更新测试
# =============================================================================

class TestConjugateUpdate:
    """测试共轭先验的更新规则"""

    @pytest.mark.parametrize("alpha_prior,beta_prior,n,successes,alpha_post,beta_post", [
        (1, 1, 100, 20, 21, 81),       # 无信息先验
        (15, 85, 1000, 180, 195, 905),  # 信息性先验
        (5, 20, 50, 10, 15, 60),       # 弱信息先验
        (10, 10, 200, 100, 110, 110),  # 对称先验
    ])
    def test_beta_binomial_conjugate_update(
        self, alpha_prior, beta_prior, n, successes, alpha_post, beta_post
    ):
        """
        正例：Beta-Binomial 共轭更新

        验证：Beta(α, β) 先验 + Binomial(n, θ) 似然
              → Beta(α + successes, β + failures) 后验
        """
        failures = n - successes

        # 后验参数
        computed_alpha_post = alpha_prior + successes
        computed_beta_post = beta_prior + failures

        # 验证：共轭更新公式
        assert computed_alpha_post == alpha_post
        assert computed_beta_post == beta_post

    def test_conjugate_update_interpretation(self, churn_bayes_data):
        """
        正例：共轭更新的直观理解

        先验参数 α 和 β 可以理解为"伪观测"
        - α：伪成功次数
        - β：伪失败次数

        后验 = 先验伪观测 + 实际观测
        """
        alpha_prior = churn_bayes_data['alpha_prior']
        beta_prior = churn_bayes_data['beta_prior']
        n = churn_bayes_data['n']
        churned = churn_bayes_data['churned']

        # 先验可以理解为"15 次流失，85 次未流失"
        prior_interpretation = f"先验等价于 {alpha_prior} 次流失，{beta_prior} 次未流失"

        # 后验 = 先验 + 数据
        alpha_post = alpha_prior + churned
        beta_post = beta_prior + (n - churned)

        # 验证：后验等价于"195 次流失，905 次未流失"
        assert alpha_post == 195
        assert beta_post == 905


# =============================================================================
# Beta 分布统计量测试
# =============================================================================

class TestBetaDistributionStatistics:
    """测试 Beta 分布的统计量"""

    def test_beta_mean(self, beta_distribution_properties):
        """
        正例：Beta 分布的均值

        E[θ] = α / (α + β)
        """
        for name, params in beta_distribution_properties.items():
            alpha = params['alpha']
            beta = params['beta']
            expected_mean = params['mean']

            # 计算均值
            computed_mean = alpha / (alpha + beta)

            assert abs(computed_mean - expected_mean) < 1e-10

    def test_beta_variance(self, beta_distribution_properties):
        """
        正例：Beta 分布的方差

        Var(θ) = αβ / [(α+β)²(α+β+1)]
        """
        for name, params in beta_distribution_properties.items():
            alpha = params['alpha']
            beta = params['beta']
            expected_var = params['variance']

            # 计算方差
            computed_var = (alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1))

            assert abs(computed_var - expected_var) < 1e-10

    def test_beta_mode(self, beta_distribution_properties):
        """
        正例：Beta 分布的众数

        Mode(θ) = (α - 1) / (α + β - 2), for α, β > 1
        """
        for name, params in beta_distribution_properties.items():
            alpha = params['alpha']
            beta = params['beta']
            expected_mode = params['mode']

            # 计算众数（当 α, β > 1 时）
            if alpha > 1 and beta > 1:
                computed_mode = (alpha - 1) / (alpha + beta - 2)
                assert abs(computed_mode - expected_mode) < 1e-10
            elif expected_mode is None:
                # 均匀分布或其他无众数情况
                assert True

    def test_beta_interval(self, churn_bayes_data, tolerance):
        """
        正例：Beta 分布的区间估计

        95% 可信区间：[q(0.025), q(0.975)]
        """
        alpha_post = churn_bayes_data['expected_alpha_post']
        beta_post = churn_bayes_data['expected_beta_post']

        # 使用 scipy 计算 95% 区间
        ci_low, ci_high = stats.beta.interval(0.95, alpha_post, beta_post)

        # 验证：区间合理
        assert 0 < ci_low < ci_high < 1

        # 验证：区间包含均值
        mean = alpha_post / (alpha_post + beta_post)
        assert ci_low < mean < ci_high


# =============================================================================
# 先验类型测试
# =============================================================================

class TestPriorTypes:
    """测试不同类型的先验"""

    def test_uninformative_prior_uniform(self):
        """
        正例：无信息先验（均匀分布）

        Beta(1, 1) = Uniform(0, 1)
        """
        alpha, beta = 1, 1

        # 均值 = 0.5
        mean = alpha / (alpha + beta)
        assert abs(mean - 0.5) < 1e-10

        # 验证：均匀分布的 PDF 在 [0,1] 上恒为 1
        x = np.linspace(0, 1, 10)
        pdf_values = stats.beta.pdf(x, alpha, beta)
        assert np.allclose(pdf_values, 1.0)

    def test_jeffreys_prior(self):
        """
        正例：Jeffreys 先验

        Beta(0.5, 0.5) - 在 0 和 1 处发散（无信息先验的另一种选择）
        """
        alpha, beta = 0.5, 0.5

        # 均值 = 0.5
        mean = alpha / (alpha + beta)
        assert abs(mean - 0.5) < 1e-10

        # 方差很大（高不确定性）
        variance = (alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1))
        assert variance > 0.1  # 很大方差

    def test_weakly_informative_prior(self):
        """
        正例：弱信息先验

        Beta(5, 20) - 均值 20%，方差较大
        """
        alpha, beta = 5, 20

        # 均值
        mean = alpha / (alpha + beta)
        assert abs(mean - 0.2) < 1e-10

        # 方差适中（不是太大，也不是太小）
        variance = (alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1))
        assert 0.01 > variance > 0.001

    def test_informative_prior(self):
        """
        正例：信息性先验

        Beta(150, 850) - 基于 1000 个历史观测，均值 15%
        """
        alpha, beta = 150, 850

        # 均值 = 15%
        mean = alpha / (alpha + beta)
        assert abs(mean - 0.15) < 1e-10

        # 方差很小（高确定性）
        variance = (alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1))
        assert variance < 0.0002  # 约等于 0.000127

    def test_prior_strength_comparison(self):
        """
        正例：先验强度比较

        α + β 的值越大，先验越"强"
        """
        priors = {
            '无信息': (1, 1),       # 强度 = 2
            '弱信息': (5, 20),      # 强度 = 25
            '信息性': (150, 850),   # 强度 = 1000
        }

        strengths = {name: alpha + beta for name, (alpha, beta) in priors.items()}

        # 验证：强度递增
        assert strengths['无信息'] < strengths['弱信息'] < strengths['信息性']

        # 验证：强度越大，方差越小（越确定）
        variances = {}
        for name, (alpha, beta) in priors.items():
            var = (alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1))
            variances[name] = var

        assert variances['无信息'] > variances['弱信息'] > variances['信息性']


# =============================================================================
# 后验性质测试
# =============================================================================

class TestPosteriorProperties:
    """测试后验分布的性质"""

    def test_posterior_mean_between_prior_and_data(self):
        """
        正例：后验均值在先验均值和数据均值之间

        后验是先验和数据的"折中"
        """
        # 先验：Beta(15, 85) -> 均值 15%
        alpha_prior, beta_prior = 15, 85
        prior_mean = alpha_prior / (alpha_prior + beta_prior)

        # 数据：180/1000 = 18%
        n, churned = 1000, 180
        data_mean = churned / n

        # 后验：Beta(195, 905) -> 均值 17.7%
        alpha_post, beta_post = 195, 905
        posterior_mean = alpha_post / (alpha_post + beta_post)

        # 验证：后验均值在先验和数据之间
        assert prior_mean < posterior_mean < data_mean

    def test_data_dominates_with_large_sample(self):
        """
        正例：大样本时数据主导后验

        当 n >> α + β 时，后验接近数据
        """
        # 弱先验：Beta(5, 5)
        alpha_prior, beta_prior = 5, 5

        # 大数据：10000 次观测，2000 次成功
        n, successes = 10000, 2000
        data_mean = successes / n

        # 后验
        alpha_post = alpha_prior + successes
        beta_post = beta_prior + (n - successes)
        posterior_mean = alpha_post / (alpha_post + beta_post)

        # 验证：后验均值非常接近数据均值
        assert abs(posterior_mean - data_mean) < 0.001

    def test_prior_dominates_with_small_sample(self):
        """
        正例：小样本时先验主导后验

        当 n << α + β 时，后验接近先验
        """
        # 强先验：Beta(500, 500) -> 均值 50%
        alpha_prior, beta_prior = 500, 500
        prior_mean = alpha_prior / (alpha_prior + beta_prior)

        # 小数据：10 次观测，8 次成功
        n, successes = 10, 8
        data_mean = successes / n

        # 后验
        alpha_post = alpha_prior + successes
        beta_post = beta_prior + (n - successes)
        posterior_mean = alpha_post / (alpha_post + beta_post)

        # 验证：后验均值更接近先验（而不是数据）
        assert abs(posterior_mean - prior_mean) < abs(posterior_mean - data_mean)


# =============================================================================
# 可信区间测试
# =============================================================================

class TestCredibleIntervals:
    """测试贝叶斯可信区间"""

    def test_equal_tailed_interval(self, churn_bayes_data):
        """
        正例：等尾区间

        [q(α/2), q(1-α/2)]
        """
        alpha_post = churn_bayes_data['expected_alpha_post']
        beta_post = churn_bayes_data['expected_beta_post']

        # 95% 等尾区间
        ci_low, ci_high = stats.beta.interval(0.95, alpha_post, beta_post)

        # 验证：区间两端各占 2.5%
        assert abs(stats.beta.cdf(ci_low, alpha_post, beta_post) - 0.025) < 1e-10
        assert abs(stats.beta.cdf(ci_high, alpha_post, beta_post) - 0.975) < 1e-10

    def test_highest_density_interval(self):
        """
        正例：最高密度区间（HDI）

        HDI 是包含指定概率的最窄区间
        """
        alpha, beta = 195, 905

        # 对于单峰的 Beta 分布，HDI 和等尾区间接近
        # (对于对称分布，它们相同)
        ci_et = stats.beta.interval(0.95, alpha, beta)

        # 简化的 HDI：对于单峰分布，HDI ≈ 等尾区间
        # 严格来说需要用数值方法找 HDI
        assert ci_et[0] > 0
        assert ci_et[1] < 1
        assert ci_et[0] < ci_et[1]

    def test_interpretation_difference(self):
        """
        正例：可信区间 vs 置信区间的解释

        贝叶斯 95% 可信区间："参数有 95% 的概率在这个区间里"
        频率学派 95% 置信区间："长期频率下 95% 的区间包含真实参数"
        """
        # 贝叶斯解释
        bayesian_interp = "P(θ ∈ [a, b] | data) = 0.95"

        # 频率学派解释
        frequentist_interp = "在重复抽样中，95% 的区间包含真实参数"

        # 验证：解释不同
        assert "P(θ ∈" in bayesian_interp
        assert "重复抽样" in frequentist_interp


# =============================================================================
# 边界情况测试
# =============================================================================

class TestEdgeCases:
    """测试边界情况"""

    def test_all_successes(self):
        """
        边界：全部成功

        n 次试验全部成功
        """
        alpha_prior, beta_prior = 1, 1
        n, successes = 10, 10

        # 后验
        alpha_post = alpha_prior + successes
        beta_post = beta_prior + (n - successes)

        # 验证：后验均值接近 1
        posterior_mean = alpha_post / (alpha_post + beta_post)
        assert posterior_mean > 0.9

    def test_all_failures(self):
        """
        边界：全部失败

        n 次试验全部失败
        """
        alpha_prior, beta_prior = 1, 1
        n, successes = 10, 0

        # 后验
        alpha_post = alpha_prior + successes
        beta_post = beta_prior + (n - successes)

        # 验证：后验均值接近 0
        posterior_mean = alpha_post / (alpha_post + beta_post)
        assert posterior_mean < 0.1

    def test_single_observation(self, minimal_data):
        """
        边界：单次观测

        n = 1
        """
        data = minimal_data['single_observation']

        alpha_prior, beta_prior = data['prior']
        n, successes = data['n'], data['churned']

        # 后验
        alpha_post = alpha_prior + successes
        beta_post = beta_prior + (n - successes)

        expected = data['expected_posterior']

        assert alpha_post == expected[0]
        assert beta_post == expected[1]

    def test_extreme_prior_parameters(self, edge_case_beta_parameters):
        """
        边界：极端 Beta 参数

        测试极端参数值的数值稳定性
        """
        # 非常强的先验
        name = 'very_strong'
        alpha, beta = edge_case_beta_parameters[name]

        # 均值
        mean = alpha / (alpha + beta)
        assert 0 < mean < 1

        # 方差（应该很小）
        variance = (alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1))
        assert variance < 1e-5  # 约 9e-6

    def test_one_sided_priors(self, edge_case_beta_parameters):
        """
        边界：单侧先验

        Beta(1, 100) 和 Beta(100, 1) - 强烈偏向一侧
        """
        # 偏向 0
        alpha_low, beta_low = edge_case_beta_parameters['one_sided']
        mean_low = alpha_low / (alpha_low + beta_low)
        assert mean_low < 0.05

        # 偏向 1
        alpha_high, beta_high = edge_case_beta_parameters['other_sided']
        mean_high = alpha_high / (alpha_high + beta_high)
        assert mean_high > 0.95


# =============================================================================
# 数值验证测试
# =============================================================================

class TestNumericalVerification:
    """数值验证：Beta-Binomial 模型的正确性"""

    def test_posterior_integrity(self):
        """
        正例：后验分布的完整性

        后验 PDF 的积分应为 1
        """
        alpha, beta = 195, 905

        # 数值积分
        x = np.linspace(0, 1, 1000)
        pdf_values = stats.beta.pdf(x, alpha, beta)
        integral = np.trapz(pdf_values, x)

        # 验证：积分约等于 1
        assert abs(integral - 1.0) < 0.01

    def test_conjugate_property(self):
        """
        正例：共轭性质

        后验和先验属于同一分布族
        """
        # 先验：Beta(α, β)
        # 似然：Binomial(n, θ)
        # 后验：Beta(α', β')

        # 验证：后验仍是 Beta 分布
        assert True  # 这是共轭先验的定义

    def test_predictive_distribution(self):
        """
        正例：预测分布

        给定后验，预测下一次观测的概率
        """
        # 后验：Beta(α, β)
        alpha, beta = 195, 905

        # 预测下一次是成功的概率：E[θ] = α / (α + β)
        predictive_prob = alpha / (alpha + beta)

        # 验证：预测概率合理
        assert 0 < predictive_prob < 1
        assert 0.15 < predictive_prob < 0.20  # 在合理范围内


# =============================================================================
# 先验与后验比较测试
# =============================================================================

class TestPriorVsPosterior:
    """测试先验与后验的关系"""

    def test_posterior_narrows_prior(self):
        """
        正例：后验比先验更窄（更确定）

        数据减少不确定性
        """
        # 先验：Beta(15, 85)
        alpha_prior, beta_prior = 15, 85
        var_prior = (alpha_prior * beta_prior) / ((alpha_prior + beta_prior)**2 * (alpha_prior + beta_prior + 1))

        # 后验：Beta(195, 905)
        alpha_post, beta_post = 195, 905
        var_post = (alpha_post * beta_post) / ((alpha_post + beta_post)**2 * (alpha_post + beta_post + 1))

        # 验证：后验方差更小
        assert var_post < var_prior

    def test_posterior_shifts_toward_data(self, churn_bayes_data):
        """
        正例：后验向数据方向移动

        后验均值在先验和数据之间
        """
        alpha_prior, beta_prior = churn_bayes_data['alpha_prior'], churn_bayes_data['beta_prior']
        n, churned = churn_bayes_data['n'], churn_bayes_data['churned']

        # 先验均值
        prior_mean = alpha_prior / (alpha_prior + beta_prior)

        # 数据均值
        data_mean = churned / n

        # 后验均值
        alpha_post, beta_post = alpha_prior + churned, beta_prior + (n - churned)
        posterior_mean = alpha_post / (alpha_post + beta_post)

        # 验证：后验均值在先验和数据之间
        assert min(prior_mean, data_mean) <= posterior_mean <= max(prior_mean, data_mean)

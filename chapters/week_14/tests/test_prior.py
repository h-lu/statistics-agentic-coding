"""
测试先验（Prior）相关功能

正例：
- 创建无信息先验（Uniform/Beta(1,1)）
- 创建弱信息先验（Normal(0, large_sigma)）
- 创建共轭先验（Beta-Binomial 配对）

边界：
- 极端弱信息先验（sigma=1e6）
- 先验参数接近零

反例：
- 先验方差为负数（应失败）
- Beta 分布参数为负数（应失败）
"""
import pytest
import numpy as np
from scipy import stats


class TestPriorCreation:
    """测试先验创建"""

    def test_create_uniform_prior_beta_1_1(self):
        """
        正例：创建无信息先验 Beta(1, 1)

        Beta(1, 1) 在 [0, 1] 上是均匀分布
        """
        prior = stats.beta(1, 1)

        # 验证基本属性
        assert prior.mean() == 0.5  # 均匀分布均值
        assert prior.var() == 1/12  # 均匀分布方差

        # 验证分布形状（采样测试）
        samples = prior.rvs(1000, random_state=42)
        assert np.mean(samples) == pytest.approx(0.5, abs=0.05)
        assert 0 <= samples.min() <= 1
        assert 0 <= samples.max() <= 1

    def test_create_weakly_informative_prior(self):
        """
        正例：创建弱信息先验

        使用 Beta(2, 40)，均值约 0.047，方差较大
        """
        prior = stats.beta(2, 40)

        # 验证均值合理（约 4.7% 转化率）
        assert prior.mean() == pytest.approx(0.0476, abs=0.001)

        # 验证方差较大（弱信息）
        assert prior.var() > 0.001  # 方差不是极小

        # 验证 pdf 在合理范围
        assert prior.pdf(0.05) > 0  # 5% 转化率有非零概率
        assert prior.pdf(0.5) > 0  # 50% 转化率也有概率（虽然很小）

    def test_create_conjugate_prior_beta_binomial(self):
        """
        正例：创建共轭先验（Beta-Binomial 配对）

        Beta 分布是 Binomial 似然的共轭先验
        后验也是 Beta 分布：Beta(alpha + successes, beta + failures)
        """
        alpha_prior = 2
        beta_prior = 40
        prior = stats.beta(alpha_prior, beta_prior)

        # 模拟观测数据
        successes = 58
        trials = 1000
        failures = trials - successes

        # 计算后验
        posterior = stats.beta(
            alpha_prior + successes,
            beta_prior + failures
        )

        # 验证后验存在
        assert posterior.mean() > 0
        assert posterior.mean() < 1

        # 验证后验均值介于先验和 MLE 之间
        mle = successes / trials
        prior_mean = alpha_prior / (alpha_prior + beta_prior)

        assert prior_mean < posterior.mean() < mle or \
               mle < posterior.mean() < prior_mean

    @pytest.mark.parametrize("sigma", [10, 100, 1000, 1e6])
    def test_extreme_weakly_informative_prior(self, sigma):
        """
        边界：极端弱信息先验（sigma 很大）

        测试不同方差下的正态先验
        """
        prior = stats.norm(0, sigma)

        # 验证均值
        assert prior.mean() == 0

        # 验证方差
        assert prior.var() == sigma ** 2

        # 验证覆盖范围（3-sigma 规则）
        # 对于正态分布，3-sigma 范围覆盖约 99.7%
        cdf_value = prior.cdf(3 * sigma)
        assert cdf_value == pytest.approx(0.998, abs=0.01) or \
               cdf_value > 0.99  # 更宽松的检查

    def test_prior_parameters_near_zero(self):
        """
        边界：先验参数接近零（Jeffreys 先验）

        Beta(0.5, 0.5) 是 Jeffreys 先验，参数接近零但仍有效
        """
        prior = stats.beta(0.5, 0.5)

        # 验证基本属性
        assert prior.mean() == 0.5
        assert np.isfinite(prior.var())

        # 验证 pdf 在边界行为（U 型分布）
        assert prior.pdf(0.01) > prior.pdf(0.5)  # 边界概率更高


class TestPriorValidation:
    """测试先验验证（反例）"""

    @pytest.mark.parametrize("alpha,beta", [
        (-1, 1),  # alpha 负数
        (1, -1),  # beta 负数
        (-1, -1),  # 都负数
        (0, 0),   # 都为零（边界情况）
    ])
    def test_beta_prior_with_negative_parameters_raises_error(self, alpha, beta):
        """
        反例：Beta 分布参数为负数（应失败）

        scipy.stats.beta 不接受负参数
        """
        with pytest.raises(ValueError):
            stats.beta(alpha, beta).rvs()

    @pytest.mark.parametrize("sigma", [
        -1,      # 负标准差
        -10,     # 大负标准差
        0,       # 零标准差（退化分布）
    ])
    def test_normal_prior_with_invalid_sigma(self, sigma):
        """
        反例：正态分布标准差为负数或零（应失败或退化）

        sigma=0 会产生退化分布
        sigma<0 会产生错误
        """
        if sigma < 0:
            # 标准差必须非负
            with pytest.raises(Exception):
                stats.norm(0, sigma).rvs()
        else:
            # sigma=0 是退化分布，所有采样值都是均值
            prior = stats.norm(0, sigma)
            samples = prior.rvs(10)
            assert np.all(samples == 0)


class TestPriorInfluence:
    """测试先验对后验的影响"""

    def test_weak_prior_vs_strong_prior(self, simple_ab_data):
        """
        对比弱先验和强先验对后验的影响

        强先验会让后验更接近先验
        """
        conversions = simple_ab_data['conversions_B']
        exposures = simple_ab_data['exposures_B']

        # 弱先验
        prior_weak = stats.beta(2, 40)
        posterior_weak = stats.beta(
            prior_weak.args[0] + conversions,
            prior_weak.args[1] + exposures - conversions
        )

        # 强先验
        prior_strong = stats.beta(50, 1000)
        posterior_strong = stats.beta(
            prior_strong.args[0] + conversions,
            prior_strong.args[1] + exposures - conversions
        )

        # 强先验的后验应该更接近先验
        mle = conversions / exposures

        # 弱先验的后验更接近 MLE
        assert abs(posterior_weak.mean() - mle) < \
               abs(posterior_strong.mean() - mle)

    def test_prior_washes_out_with_large_sample(self):
        """
        测试：大样本下先验被"淹没"

        当样本量很大时，不同先验的后验应该接近
        """
        # 大样本数据
        conversions_large = 5800
        exposures_large = 100000

        priors = [
            stats.beta(1, 1),      # 无信息
            stats.beta(2, 40),     # 弱信息
            stats.beta(50, 1000),  # 强信息
        ]

        posteriors = [
            stats.beta(p.args[0] + conversions_large,
                      p.args[1] + exposures_large - conversions_large)
            for p in priors
        ]

        # 所有后验均值应该接近
        posterior_means = [p.mean() for p in posteriors]
        max_diff = max(posterior_means) - min(posterior_means)

        # 大样本下，先验差异被淹没
        assert max_diff < 0.001  # 差异小于 0.1%


class TestPriorPredictive:
    """测试先验预测分布"""

    def test_prior_predictive_check(self, simple_ab_data):
        """
        测试先验预测检查

        验证先验是否合理编码了领域知识
        """
        prior = stats.beta(2, 40)

        # 从先验采样
        prior_samples = prior.rvs(1000, random_state=42)

        # 验证先验预测的合理范围
        # 95% 的先验概率应该在 0% 到 20% 之间
        assert np.percentile(prior_samples, 97.5) < 0.20

        # 先验不应该支持极端值（如 90% 转化率）
        assert (prior_samples > 0.5).mean() < 0.01

    def test_prior_informativeness(self):
        """
        测试先验的信息量

        比较不同先验的"强度"
        """
        prior_uninformative = stats.beta(1, 1)
        prior_weak = stats.beta(2, 40)
        prior_strong = stats.beta(50, 1000)

        # 先验方差越小，信息越强
        var_uninformative = prior_uninformative.var()
        var_weak = prior_weak.var()
        var_strong = prior_strong.var()

        assert var_strong < var_weak < var_uninformative

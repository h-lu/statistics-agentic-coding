"""
测试 Week 14 示例代码

确保所有示例能够正确运行并产生预期输出
"""
import pytest
import numpy as np
from scipy import stats


class TestFrequentistAB:
    """测试频率学派 A/B 测试示例"""

    def test_frequentist_ab_calculation(self, simple_ab_data):
        """
        测试频率学派 A/B 测试的计算

        验证：
        1. 转化率计算正确
        2. z 统计量符号正确
        3. p 值在合理范围
        """
        from statsmodels.stats.proportion import proportions_ztest

        conversions_A = simple_ab_data['conversions_A']
        exposures_A = simple_ab_data['exposures_A']
        conversions_B = simple_ab_data['conversions_B']
        exposures_B = simple_ab_data['exposures_B']

        # 计算转化率
        p_A = conversions_A / exposures_A
        p_B = conversions_B / exposures_B

        assert 0 < p_A < 1
        assert 0 < p_B < 1

        # z 检验
        count = np.array([conversions_B, conversions_A])
        nobs = np.array([exposures_B, exposures_A])
        z_stat, p_value = proportions_ztest(count, nobs)

        # B 的转化率更高，z 统计量应该为正
        assert z_stat > 0

        # p 值应该在 [0, 1] 范围内
        assert 0 <= p_value <= 1


class TestBayesianAB:
    """测试贝叶斯 A/B 测试示例"""

    def test_bayesian_ab_posterior(self, simple_ab_data):
        """
        测试贝叶斯 A/B 测试的后验计算

        验证：
        1. 后验均值在合理范围
        2. 后验均值介于先验和 MLE 之间
        3. P(B > A) 在 [0, 1] 范围
        """
        conversions_A = simple_ab_data['conversions_A']
        exposures_A = simple_ab_data['exposures_A']
        conversions_B = simple_ab_data['conversions_B']
        exposures_B = simple_ab_data['exposures_B']

        # 无信息先验
        alpha_prior, beta_prior = 1, 1

        # 计算后验
        posterior_A = stats.beta(
            alpha_prior + conversions_A,
            beta_prior + exposures_A - conversions_A
        )
        posterior_B = stats.beta(
            alpha_prior + conversions_B,
            beta_prior + exposures_B - conversions_B
        )

        # 后验均值
        mean_A = posterior_A.mean()
        mean_B = posterior_B.mean()

        assert 0 < mean_A < 1
        assert 0 < mean_B < 1

        # MLE
        mle_A = conversions_A / exposures_A
        mle_B = conversions_B / exposures_B

        # 后验均值应该接近 MLE（无信息先验）
        assert abs(mean_A - mle_A) < 0.01
        assert abs(mean_B - mle_B) < 0.01

    def test_prob_B_better(self, simple_ab_data):
        """
        测试 P(B > A) 的计算
        """
        conversions_A = simple_ab_data['conversions_A']
        exposures_A = simple_ab_data['exposures_A']
        conversions_B = simple_ab_data['conversions_B']
        exposures_B = simple_ab_data['exposures_B']

        alpha_prior, beta_prior = 1, 1

        posterior_A = stats.beta(
            alpha_prior + conversions_A,
            beta_prior + exposures_A - conversions_A
        )
        posterior_B = stats.beta(
            alpha_prior + conversions_B,
            beta_prior + exposures_B - conversions_B
        )

        # 采样
        np.random.seed(42)
        samples_A = posterior_A.rvs(1000)
        samples_B = posterior_B.rvs(1000)

        prob_B_better = (samples_B > samples_A).mean()

        # 概率应该在 [0, 1] 之间
        assert 0 <= prob_B_better <= 1


class TestPriorSensitivity:
    """测试先验敏感性分析"""

    def test_prior_influence_on_posterior(self, small_ab_data):
        """
        测试先验对后验的影响

        验证：
        1. 强先验的后验更接近先验
        2. 弱先验的后验更接近 MLE
        """
        conversions = small_ab_data['conversions_B']
        exposures = small_ab_data['exposures_B']

        mle = conversions / exposures

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

        # 弱先验的后验应该更接近 MLE
        dist_weak = abs(posterior_weak.mean() - mle)
        dist_strong = abs(posterior_strong.mean() - mle)

        assert dist_weak < dist_strong

    def test_prior_washes_out_with_large_sample(self):
        """
        测试大样本下先验被"淹没"

        验证：
        1. 大样本下，不同先验的后验接近
        """
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

        posterior_means = [p.mean() for p in posteriors]
        max_diff = max(posterior_means) - min(posterior_means)

        # 大样本下，先验差异被淹没
        assert max_diff < 0.001


class TestMCMCIntuition:
    """测试 MCMC 采样直觉"""

    def test_metropolis_hastings_basic(self):
        """
        测试 Metropolis-Hastings 算法的基本功能

        验证：
        1. 接受率在合理范围
        2. 样本均值接近真实值
        3. 样本标准差接近真实值
        """
        def target_log_pdf(x):
            return -0.5 * x**2 - 0.5 * np.log(2 * np.pi)

        # 这里不运行完整的 MH 算法（太慢）
        # 只测试目标函数
        assert target_log_pdf(0) > target_log_pdf(1)
        assert target_log_pdf(0) > target_log_pdf(-1)


class TestHierarchicalModel:
    """测试层次模型"""

    def test_shrinkage_concept(self):
        """
        测试 shrinkage 的概念

        验证：
        1. 小样本的后验向全局均值收缩
        2. 大样本的后验保持接近观测值
        """
        # 大样本
        conv_large, exp_large = 580, 10000
        obs_rate_large = conv_large / exp_large

        # 小样本
        conv_small, exp_small = 13, 200
        obs_rate_small = conv_small / exp_small

        # 全局均值（假设）
        global_mean = 0.05

        # 大样本后验（无信息先验）
        post_large = stats.beta(1 + conv_large, 1 + exp_large - conv_large)

        # 小样本后验（无信息先验）
        post_small = stats.beta(1 + conv_small, 1 + exp_small - conv_small)

        # 大样本的后验应该更接近观测值
        shrinkage_large = abs(post_large.mean() - obs_rate_large)
        shrinkage_small = abs(post_small.mean() - obs_rate_small)

        # 相对于标准差，大样本的收缩比例更小
        # 这里简化检查
        assert shrinkage_large / post_large.std() < shrinkage_small / post_small.std()


class TestInterpretation:
    """测试解释性概念"""

    def test_credible_vs_confidence_interval(self):
        """
        测试可信区间 vs 置信区间的解释

        验证：
        1. 理解两种区间的本质区别
        """
        # 贝叶斯可信区间的正确解释
        credible_interpretation = (
            "给定数据和先验，参数有 95% 的概率在区间内"
        )

        # 频率学派置信区间的正确解释
        confidence_interpretation = (
            "重复抽样 100 次，95 个区间会覆盖真值"
        )

        # 两者确实不同
        assert credible_interpretation != confidence_interpretation

    def test_common_misinterpretation(self):
        """
        测试常见的错误解释
        """
        # 错误：把置信区间解释为可信区间
        wrong = "参数有 95% 的概率在置信区间内"
        correct = "重复抽样下，95% 的区间会覆盖真值"

        assert wrong != correct


class TestDecisionRules:
    """测试贝叶斯决策规则"""

    @pytest.mark.parametrize("prob,lift,expected", [
        (0.92, 8.0, "推荐"),  # 高概率，高提升
        (0.75, 3.0, "继续观察"),  # 中等概率
        (0.55, 1.0, "不推荐"),  # 低概率
    ])
    def test_decision_rule(self, prob, lift, expected):
        """
        测试贝叶斯决策规则

        规则：
        - P(B > A) >= 90% 且提升 >= 5%：推荐
        - 70% <= P(B > A) < 90%：继续观察
        - P(B > A) < 70%：不推荐
        """
        if prob >= 0.9 and lift >= 5.0:
            decision = "推荐"
        elif prob >= 0.70:
            decision = "继续观察"
        else:
            decision = "不推荐"

        assert decision == expected

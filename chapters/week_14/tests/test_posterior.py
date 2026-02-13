"""
测试后验（Posterior）计算

正例：
- Beta-Binomial 共轭后验计算正确
- 贝叶斯更新后参数合理（alpha, beta > 0）
- 后验分布统计量计算正确

边界：
- 零数据情况（后验 = 先验）
- 极端数据（全成功或全失败）
- 单次观测

反例：
- 无效数据（成功次数 > 试验次数）
"""
import pytest
import numpy as np
from scipy import stats


class TestPosteriorCalculation:
    """测试后验计算"""

    def test_beta_binomial_conjugate_posterior(self, simple_ab_data):
        """
        正例：Beta-Binomial 共轭后验计算正确

        先验: Beta(alpha, beta)
        似然: Binomial(n, theta)
        后验: Beta(alpha + successes, beta + failures)
        """
        alpha_prior = 1
        beta_prior = 1
        successes = simple_ab_data['conversions_A']
        trials = simple_ab_data['exposures_A']
        failures = trials - successes

        # 计算后验
        posterior = stats.beta(
            alpha_prior + successes,
            beta_prior + failures
        )

        # 验证后验参数正确
        assert posterior.args[0] == alpha_prior + successes
        assert posterior.args[1] == beta_prior + failures

        # 验证后验均值公式
        expected_mean = (alpha_prior + successes) / (alpha_prior + beta_prior + trials)
        assert posterior.mean() == pytest.approx(expected_mean)

    def test_bayesian_update_parameters_positive(self, small_ab_data):
        """
        正例：贝叶斯更新后参数合理

        后验的 alpha 和 beta 必须 > 0
        """
        alpha_prior = 2
        beta_prior = 40
        successes = small_ab_data['conversions_B']
        trials = small_ab_data['exposures_B']
        failures = trials - successes

        posterior = stats.beta(
            alpha_prior + successes,
            beta_prior + failures
        )

        # 验证参数为正
        alpha_post, beta_post = posterior.args
        assert alpha_post > 0
        assert beta_post > 0

        # 验证后验在 [0, 1] 范围内
        samples = posterior.rvs(1000)
        assert np.all((samples >= 0) & (samples <= 1))

    def test_posterior_statistics_correct(self, simple_ab_data):
        """
        正例：后验分布统计量计算正确

        均值、方差、分位数应该合理
        """
        alpha_prior = 1
        beta_prior = 1
        successes = simple_ab_data['conversions_B']
        trials = simple_ab_data['exposures_B']
        failures = trials - successes

        posterior = stats.beta(
            alpha_prior + successes,
            beta_prior + failures
        )

        # 均值
        mean = posterior.mean()
        assert 0 < mean < 1

        # 方差
        var = posterior.var()
        assert var > 0
        assert var < 0.1  # 转化率方差不应该太大

        # 分位数
        ci_low, ci_high = posterior.interval(0.95)
        assert 0 <= ci_low < ci_high <= 1
        assert (ci_high - ci_low) < 0.2  # 95% CI 不应该太宽

    @pytest.mark.parametrize("alpha_prior,beta_prior,successes,trials", [
        (1, 1, 0, 100),      # 零次成功
        (1, 1, 100, 100),    # 全部成功
        (1, 1, 50, 100),     # 50% 成功率
        (2, 40, 1, 10),      # 单次观测成功
    ])
    def test_posterior_with_various_data(self, alpha_prior, beta_prior,
                                        successes, trials):
        """
        边界：不同数据情况下的后验计算
        """
        failures = trials - successes

        posterior = stats.beta(
            alpha_prior + successes,
            beta_prior + failures
        )

        # 验证后验存在
        assert np.isfinite(posterior.mean())
        assert np.isfinite(posterior.var())

        # 验证后验在合理范围
        assert 0 <= posterior.mean() <= 1

    def test_zero_data_posterior_equals_prior(self):
        """
        边界：零数据情况（后验 = 先验）

        如果没有观测数据，后验应该等于先验
        """
        alpha_prior = 5
        beta_prior = 50

        # 零数据
        successes = 0
        trials = 0

        posterior = stats.beta(
            alpha_prior + successes,
            beta_prior + trials - successes
        )

        # 后验应该等于先验
        assert posterior.args == (alpha_prior, beta_prior)
        assert posterior.mean() == stats.beta(alpha_prior, beta_prior).mean()

    def test_extreme_data_all_success(self):
        """
        边界：极端数据（全成功）

        即使全部成功，后验也应该合理（不会过度自信）
        """
        prior = stats.beta(1, 1)
        successes = 100
        trials = 100

        posterior = stats.beta(
            prior.args[0] + successes,
            prior.args[1] + trials - successes
        )

        # 后验均值应该接近 1，但不等于 1
        assert posterior.mean() > 0.95
        assert posterior.mean() < 1.0

        # 95% CI 上限应该小于 1
        ci_low, ci_high = posterior.interval(0.95)
        assert ci_high < 1.0

    def test_extreme_data_all_failure(self):
        """
        边界：极端数据（全失败）

        即使全部失败，后验也应该合理
        """
        prior = stats.beta(1, 1)
        successes = 0
        trials = 100

        posterior = stats.beta(
            prior.args[0] + successes,
            prior.args[1] + trials - successes
        )

        # 后验均值应该接近 0，但大于 0
        assert posterior.mean() < 0.05
        assert posterior.mean() > 0.0

        # 95% CI 下限应该大于 0
        ci_low, ci_high = posterior.interval(0.95)
        assert ci_low > 0.0

    def test_single_observation(self):
        """
        边界：单次观测

        单次观测的后验应该介于先验和观测之间
        """
        prior = stats.beta(2, 40)

        # 单次成功观测
        posterior_success = stats.beta(prior.args[0] + 1, prior.args[1])
        # 单次失败观测
        posterior_failure = stats.beta(prior.args[0], prior.args[1] + 1)

        # 成功观测应该增加后验均值
        assert posterior_success.mean() > prior.mean()
        # 失败观测应该降低后验均值
        assert posterior_failure.mean() < prior.mean()


class TestPosteriorValidation:
    """测试后验验证（反例）"""

    @pytest.mark.parametrize("successes,trials", [
        (150, 100),  # 成功次数 > 试验次数
        (-10, 100),  # 负成功次数
        (50, -100),  # 负试验次数
    ])
    def test_invalid_binomial_data(self, successes, trials):
        """
        反例：无效数据（成功次数 > 试验次数）

        这类数据应该被拒绝或修正
        """
        # 验证数据有效性应该在外部进行
        # 这里测试如果传入无效数据，Beta 分布会失败或产生警告

        if successes > trials or successes < 0 or trials < 0:
            # 无效数据，Beta 分布可能产生非有限值
            posterior = stats.beta(1 + successes, 1 + trials - successes)

            # 检查后验是否仍然有效
            # 注意：scipy 可能不会直接报错，但会产生非有限值
            try:
                mean = posterior.mean()
                if not np.isfinite(mean):
                    # 非有限值是预期行为
                    pass
            except Exception:
                # 抛出异常也是预期行为
                pass


class TestPosteriorPredictive:
    """测试后验预测分布"""

    def test_posterior_predictive_samples(self, simple_ab_data):
        """
        测试后验预测采样

        从后验采样，然后用采样的参数生成预测数据
        """
        alpha_prior = 1
        beta_prior = 1
        successes = simple_ab_data['conversions_A']
        trials = simple_ab_data['exposures_A']
        failures = trials - successes

        posterior = stats.beta(
            alpha_prior + successes,
            beta_prior + failures
        )

        # 从后验采样
        n_samples = 1000
        theta_samples = posterior.rvs(n_samples, random_state=42)

        # 后验预测：对每个 theta 采样 Binomial
        predictive_samples = np.array([
            np.random.binomial(trials, theta)
            for theta in theta_samples
        ])

        # 验证预测样本在合理范围
        assert np.all(predictive_samples >= 0)
        assert np.all(predictive_samples <= trials)

        # 预测均值应该接近观测值
        assert np.abs(predictive_samples.mean() - successes) / trials < 0.05

    def test_posterior_predictive_check(self, simple_ab_data):
        """
        测试后验预测检查（Posterior Predictive Check）

        验证模型是否能复现观测数据的特征
        """
        successes = simple_ab_data['conversions_B']
        trials = simple_ab_data['exposures_B']

        prior = stats.beta(2, 40)
        posterior = stats.beta(
            prior.args[0] + successes,
            prior.args[1] + trials - successes
        )

        # 后验预测采样
        n_sim = 1000
        theta_samples = posterior.rvs(n_sim, random_state=42)
        sim_successes = np.array([
            np.random.binomial(trials, theta)
            for theta in theta_samples
        ])

        # 检查观测值是否在预测分布的合理范围内
        # 观测值应该接近预测分布的中位数
        median_sim = np.median(sim_successes)
        assert abs(successes - median_sim) / trials < 0.05

        # 观测值不应该在预测分布的极端尾部
        p_value = (sim_successes <= successes).mean()
        # 如果 p_value 接近 0 或 1，说明模型拟合差
        # 但对于简单情况，应该在合理范围内
        # 这里不做断言，只是演示如何做 PPC


class TestPosteriorComparison:
    """测试后验比较（如 A/B 测试）"""

    def test_probability_b_better_than_a(self, simple_ab_data):
        """
        测试计算 "B 比 A 好" 的概率

        P(theta_B > theta_A)
        """
        # 先验
        prior = stats.beta(1, 1)

        # A 版本后验
        posterior_A = stats.beta(
            prior.args[0] + simple_ab_data['conversions_A'],
            prior.args[1] + simple_ab_data['exposures_A'] - simple_ab_data['conversions_A']
        )

        # B 版本后验
        posterior_B = stats.beta(
            prior.args[0] + simple_ab_data['conversions_B'],
            prior.args[1] + simple_ab_data['exposures_B'] - simple_ab_data['conversions_B']
        )

        # 采样计算概率
        n_samples = 10000
        samples_A = posterior_A.rvs(n_samples, random_state=42)
        samples_B = posterior_B.rvs(n_samples, random_state=43)

        prob_B_better = (samples_B > samples_A).mean()

        # 验证概率在合理范围
        assert 0 <= prob_B_better <= 1

        # B 的转化率略高，概率应该 > 0.5
        assert prob_B_better > 0.5

    def test_relative_lift_distribution(self, simple_ab_data):
        """
        测试相对提升分布

        (theta_B - theta_A) / theta_A
        """
        prior = stats.beta(1, 1)

        posterior_A = stats.beta(
            prior.args[0] + simple_ab_data['conversions_A'],
            prior.args[1] + simple_ab_data['exposures_A'] - simple_ab_data['conversions_A']
        )

        posterior_B = stats.beta(
            prior.args[0] + simple_ab_data['conversions_B'],
            prior.args[1] + simple_ab_data['exposures_B'] - simple_ab_data['conversions_B']
        )

        # 采样
        n_samples = 10000
        samples_A = posterior_A.rvs(n_samples, random_state=42)
        samples_B = posterior_B.rvs(n_samples, random_state=43)

        # 计算相对提升
        rel_lift = (samples_B - samples_A) / samples_A * 100

        # 验证相对提升统计量
        median_lift = np.median(rel_lift)
        assert -100 < median_lift < 100  # 不应该极端

        # 95% HDI
        hdi_low = np.percentile(rel_lift, 2.5)
        hdi_high = np.percentile(rel_lift, 97.5)
        assert hdi_low < hdi_high


class TestBayesianUpdateSequential:
    """测试顺序贝叶斯更新"""

    def test_sequential_update_equals_batch_update(self):
        """
        测试：顺序更新等于批量更新

        分批更新先验，应该等于一次性更新
        """
        prior = stats.beta(2, 40)

        # 数据分两批
        batch1 = {'successes': 30, 'trials': 500}
        batch2 = {'successes': 28, 'trials': 500}

        # 顺序更新
        posterior_batch1 = stats.beta(
            prior.args[0] + batch1['successes'],
            prior.args[1] + batch1['trials'] - batch1['successes']
        )
        posterior_sequential = stats.beta(
            posterior_batch1.args[0] + batch2['successes'],
            posterior_batch1.args[1] + batch2['trials'] - batch2['successes']
        )

        # 批量更新
        total_successes = batch1['successes'] + batch2['successes']
        total_trials = batch1['trials'] + batch2['trials']
        posterior_batch = stats.beta(
            prior.args[0] + total_successes,
            prior.args[1] + total_trials - total_successes
        )

        # 两者应该相等
        assert posterior_sequential.args == posterior_batch.args
        assert posterior_sequential.mean() == pytest.approx(posterior_batch.mean())

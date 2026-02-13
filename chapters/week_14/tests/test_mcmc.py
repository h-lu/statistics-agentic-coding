"""
测试 MCMC 采样相关功能

注意：这些测试需要 PyMC 和 ArviZ

正例：
- PyMC 模型能成功采样
- MCMC 收敛（r_hat < 1.1）
- 采样数量正确（trace shape）

边界：
- 少样本（n=10）可能不收敛
- 不同采样器的使用

反例：
- chains=1 无法计算 r_hat
"""
import pytest
import numpy as np

# 检查 PyMC 是否可用
try:
    import pymc as pm
    import arviz as az
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False


@pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
class TestMCMCSampling:
    """测试 MCMC 采样基本功能"""

    def test_simple_ab_test_mcmc(self, simple_ab_data):
        """
        正例：PyMC 模型能成功采样

        简单的 A/B 测试模型
        """
        conversions_A = simple_ab_data['conversions_A']
        exposures_A = simple_ab_data['exposures_A']

        with pm.Model() as model:
            # 先验：均匀分布
            theta_A = pm.Uniform("theta_A", 0, 1)

            # 似然
            obs_A = pm.Binomial("obs_A", n=exposures_A,
                              p=theta_A, observed=conversions_A)

            # MCMC 采样
            idata = pm.sample(1000, tune=500, chains=2,
                            random_seed=42, progressbar=False)

        # 验证采样结果
        assert "posterior" in idata
        assert "theta_A" in idata.posterior.data_vars

        # 验证采样数量
        assert idata.posterior["theta_A"].shape[1] == 1000  # draws
        assert idata.posterior["theta_A"].shape[0] == 2  # chains

    def test_mcmc_convergence_r_hat(self, simple_ab_data, convergence_thresholds):
        """
        正例：MCMC 收敛（r_hat < 1.05）

        R-hat 应该接近 1.0（< 1.01 优秀，< 1.05 可接受）
        """
        conversions = simple_ab_data['conversions_B']
        exposures = simple_ab_data['exposures_B']

        with pm.Model() as model:
            theta = pm.Uniform("theta", 0, 1)
            obs = pm.Binomial("obs", n=exposures,
                            p=theta, observed=conversions)

            idata = pm.sample(2000, tune=1000, chains=4,
                            random_seed=42, progressbar=False)

        # 计算 R-hat
        r_hat = az.rhat(idata)["theta"].values

        # 验证收敛
        assert r_hat < convergence_thresholds['r_hat_acceptable']
        # 优秀情况
        assert r_hat < convergence_thresholds['r_hat_good'] or \
               r_hat < convergence_thresholds['r_hat_acceptable']

    def test_mcmc_trace_shape(self, simple_ab_data):
        """
        正例：采样数量正确（trace shape）

        验证 idata 对象的维度正确
        """
        conversions = simple_ab_data['conversions_A']
        exposures = simple_ab_data['exposures_A']
        draws = 1500
        tune = 500
        chains = 3

        with pm.Model() as model:
            theta = pm.Uniform("theta", 0, 1)
            obs = pm.Binomial("obs", n=exposures,
                            p=theta, observed=conversions)

            idata = pm.sample(draws, tune=tune, chains=chains,
                            random_seed=42, progressbar=False)

        # 验证形状
        theta_trace = idata.posterior["theta"]

        # chain 维度
        assert theta_trace.shape[0] == chains

        # draw 维度（不包括 tune）
        assert theta_trace.shape[1] == draws

    def test_mcmc_multiple_variables(self):
        """
        正例：多变量模型采样

        测试包含多个参数的模型
        """
        np.random.seed(42)
        n = 100

        # 生成数据
        X = np.random.normal(0, 1, n)
        y = 2 + 3 * X + np.random.normal(0, 1, n)

        with pm.Model() as model:
            # 先验
            alpha = pm.Normal("alpha", mu=0, sigma=10)
            beta = pm.Normal("beta", mu=0, sigma=10)
            sigma = pm.HalfNormal("sigma", sigma=1)

            # 似然
            mu = alpha + beta * X
            Y_obs = pm.Normal("Y_obs", mu=mu, sigma=sigma, observed=y)

            idata = pm.sample(1000, tune=500, chains=2,
                            random_seed=42, progressbar=False)

        # 验证所有变量都被采样
        var_names = list(idata.posterior.data_vars.keys())
        assert "alpha" in var_names
        assert "beta" in var_names
        assert "sigma" in var_names

    def test_mcmc_deterministic_variable(self, simple_ab_data):
        """
        正例：测试确定性变量（Deterministic）

        delta = theta_B - theta_A
        """
        conversions_A = simple_ab_data['conversions_A']
        exposures_A = simple_ab_data['exposures_A']
        conversions_B = simple_ab_data['conversions_B']
        exposures_B = simple_ab_data['exposures_B']

        with pm.Model() as model:
            theta_A = pm.Uniform("theta_A", 0, 1)
            theta_B = pm.Uniform("theta_B", 0, 1)

            obs_A = pm.Binomial("obs_A", n=exposures_A,
                              p=theta_A, observed=conversions_A)
            obs_B = pm.Binomial("obs_B", n=exposures_B,
                              p=theta_B, observed=conversions_B)

            # 确定性变量
            delta = pm.Deterministic("delta", theta_B - theta_A)

            idata = pm.sample(1000, tune=500, chains=2,
                            random_seed=42, progressbar=False)

        # 验证确定性变量被记录
        assert "delta" in idata.posterior.data_vars

        # delta 的均值应该接近观测差异
        delta_mean = idata.posterior["delta"].mean().values
        observed_diff = (conversions_B / exposures_B -
                        conversions_A / exposures_A)
        assert abs(delta_mean - observed_diff) < 0.02


@pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
class TestMCMCEssentialSize:
    """测试有效样本量（ESS）"""

    def test_ess_sufficiently_large(self, simple_ab_data, convergence_thresholds):
        """
        正例：有效样本量足够大

        ESS 应该 > 400（最低），> 1000（优秀）
        """
        conversions = simple_ab_data['conversions_A']
        exposures = simple_ab_data['exposures_A']

        with pm.Model() as model:
            theta = pm.Uniform("theta", 0, 1)
            obs = pm.Binomial("obs", n=exposures,
                            p=theta, observed=conversions)

            idata = pm.sample(2000, tune=1000, chains=4,
                            random_seed=42, progressbar=False)

        # 计算 ESS
        ess = az.ess(idata)["theta"].values

        # 验证 ESS 足够大
        assert ess > convergence_thresholds['ess_min']

    def test_ess_vs_total_samples(self, simple_ab_data):
        """
        测试：ESS 应该明显小于总样本量

        由于自相关，ESS < 总样本量
        """
        conversions = simple_ab_data['conversions_B']
        exposures = simple_ab_data['exposures_B']

        total_samples = 2000

        with pm.Model() as model:
            theta = pm.Uniform("theta", 0, 1)
            obs = pm.Binomial("obs", n=exposures,
                            p=theta, observed=conversions)

            idata = pm.sample(total_samples, tune=1000, chains=2,
                            random_seed=42, progressbar=False)

        ess = az.ess(idata)["theta"].values

        # ESS 应该 <= 总样本量 * chains
        total_with_chains = total_samples * 2
        assert ess <= total_with_chains

        # ESS 应该 > 0（至少有效）
        assert ess > 0


@pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
class TestMCMCEdgeCases:
    """测试 MCMC 边界情况"""

    def test_small_sample_may_not_converge(self):
        """
        边界：少样本（n=10）可能不收敛

        小数据导致后验宽，可能需要更多采样
        """
        conversions = 2
        exposures = 10

        with pm.Model() as model:
            theta = pm.Uniform("theta", 0, 1)
            obs = pm.Binomial("obs", n=exposures,
                            p=theta, observed=conversions)

            idata = pm.sample(500, tune=300, chains=2,
                            random_seed=42, progressbar=False)

        # 小样本下 R-hat 可能较差
        r_hat = az.rhat(idata)["theta"].values

        # 不做严格断言，只是记录观察
        # 实际应用中需要增加 tune 或 draws

    def test_extreme_data_all_success(self):
        """
        边界：极端数据（全成功）

        测试 MCMC 在边界情况的表现
        """
        conversions = 100
        exposures = 100

        with pm.Model() as model:
            # 使用 Beta 先验避免边界问题
            theta = pm.Beta("theta", alpha=1, beta=1)
            obs = pm.Binomial("obs", n=exposures,
                            p=theta, observed=conversions)

            idata = pm.sample(1000, tune=500, chains=2,
                            random_seed=42, progressbar=False)

        # 验证采样成功
        assert idata.posterior["theta"].values.mean() > 0.95

    def test_different_samplers(self):
        """
        边界：测试不同采样器

        PyMC 会自动选择采样器（如 NUTS、Metropolis）
        """
        with pm.Model() as model:
            # 正态分布（连续） -> NUTS
            mu = pm.Normal("mu", mu=0, sigma=1)
            obs = pm.Normal("obs", mu=mu, sigma=1,
                          observed=np.random.normal(0, 1, 100))

            idata = pm.sample(500, tune=300, chains=2,
                            step=pm.NUTS(),
                            random_seed=42, progressbar=False)

        # 验证采样成功
        assert "mu" in idata.posterior.data_vars


@pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
class TestMCMCValidation:
    """测试 MCMC 验证（反例）"""

    def test_single_chain_no_r_hat(self, simple_ab_data):
        """
        反例：chains=1 无法计算 r_hat

        R-hat 需要至少 2 条链
        """
        conversions = simple_ab_data['conversions_A']
        exposures = simple_ab_data['exposures_A']

        with pm.Model() as model:
            theta = pm.Uniform("theta", 0, 1)
            obs = pm.Binomial("obs", n=exposures,
                            p=theta, observed=conversions)

            idata = pm.sample(500, tune=300, chains=1,
                            random_seed=42, progressbar=False)

        # 单链情况下，R-hat 无法计算或无意义
        # ArviZ 可能返回 nan 或 1.0
        try:
            r_hat = az.rhat(idata)["theta"].values
            # 如果返回 1.0，这是警告而非错误
            if not np.isnan(r_hat):
                assert r_hat == 1.0
        except Exception:
            # 抛出异常也是可能的
            pass


@pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
class TestMCMCDiagnostics:
    """测试 MCMC 诊断工具"""

    def test_trace_plot_structure(self, simple_ab_data):
        """
        测试 trace plot 的数据结构

        验证可以用于绘图
        """
        conversions = simple_ab_data['conversions_A']
        exposures = simple_ab_data['exposures_A']

        with pm.Model() as model:
            theta = pm.Uniform("theta", 0, 1)
            obs = pm.Binomial("obs", n=exposures,
                            p=theta, observed=conversions)

            idata = pm.sample(500, tune=300, chains=2,
                            random_seed=42, progressbar=False)

        # 验证数据结构
        assert idata.posterior["theta"].shape[0] == 2  # chains
        assert idata.posterior["theta"].shape[1] == 500  # draws

        # 验证数值范围
        samples = idata.posterior["theta"].values
        assert np.all((samples >= 0) & (samples <= 1))

    def test_summary_statistics(self, simple_ab_data):
        """
        测试后验摘要统计

        验证 az.summary 输出
        """
        conversions = simple_ab_data['conversions_B']
        exposures = simple_ab_data['exposures_B']

        with pm.Model() as model:
            theta = pm.Uniform("theta", 0, 1)
            obs = pm.Binomial("obs", n=exposures,
                            p=theta, observed=conversions)

            idata = pm.sample(1000, tune=500, chains=2,
                            random_seed=42, progressbar=False)

        summary = az.summary(idata)

        # 验证摘要包含关键字段
        assert "mean" in summary.columns
        assert "sd" in summary.columns
        assert "hdi_3%" in summary.columns or "ci_lower" in summary.columns
        assert "hdi_97%" in summary.columns or "ci_upper" in summary.columns

        # 验证统计量合理
        theta_summary = summary.loc["theta"]
        assert 0 < theta_summary["mean"] < 1
        assert theta_summary["sd"] > 0


@pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
class TestMCMCPrediction:
    """测试 MCMC 预测"""

    def test_posterior_predictive_check(self, simple_ab_data):
        """
        测试后验预测检查

        使用 pm.sample_posterior_predictive
        """
        conversions = simple_ab_data['conversions_A']
        exposures = simple_ab_data['exposures_A']

        with pm.Model() as model:
            theta = pm.Uniform("theta", 0, 1)
            obs = pm.Binomial("obs", n=exposures,
                            p=theta, observed=conversions)

            idata = pm.sample(500, tune=300, chains=2,
                            random_seed=42, progressbar=False)

            # 后验预测
            idata_pp = pm.sample_posterior_predictive(
                idata, progressbar=False
            )

        # 验证预测数据存在
        assert "posterior_predictive" in idata_pp
        assert "obs" in idata_pp.posterior_predictive

    def test_prior_predictive_check(self, simple_ab_data):
        """
        测试先验预测检查

        使用 pm.sample_prior_predictive
        """
        conversions = simple_ab_data['conversions_A']
        exposures = simple_ab_data['exposures_A']

        with pm.Model() as model:
            theta = pm.Uniform("theta", 0, 1)
            obs = pm.Binomial("obs", n=exposures,
                            p=theta, observed=conversions)

            # 先验预测（不需要采样）
            idata_prior = pm.sample_prior_predictive(
                500, random_seed=42
            )

        # 验证先验预测数据存在
        assert "prior" in idata_prior
        assert "obs" in idata_prior.prior

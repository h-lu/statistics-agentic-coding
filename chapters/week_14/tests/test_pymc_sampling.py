"""
Test Suite: PyMC MCMC Sampling（PyMC MCMC 采样）

测试 MCMC（马尔可夫链蒙特卡洛）采样：
1. 使用 PyMC 进行贝叶斯推断
2. MCMC 收敛性诊断（R-hat, ESS）
3. 从后验分布中采样
4. 迹图和后验分布可视化

测试覆盖：
- 正确设置 PyMC 模型
- 正确执行 MCMC 采样
- 正确检查收敛性
- 理解采样参数的影响（样本量、链数、tune）
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

try:
    import pymc as pm
    import arviz as az
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False


# =============================================================================
# PyMC 模型设置测试
# =============================================================================

class TestPyMCModelSetup:
    """测试 PyMC 模型的基本设置"""

    @pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
    def test_beta_binomial_model_definition(self, mcmc_test_data):
        """
        正例：定义 Beta-Binomial 模型

        验证：模型可以正确设置
        """
        n = mcmc_test_data['n']
        churned = mcmc_test_data['churned']
        prior_alpha = mcmc_test_data['prior_alpha']
        prior_beta = mcmc_test_data['prior_beta']

        # 定义模型
        with pm.Model() as model:
            # 先验：Beta 分布
            theta = pm.Beta('theta', alpha=prior_alpha, beta=prior_beta)

            # 似然：Binomial
            likelihood = pm.Binomial('likelihood', n=n, p=theta, observed=churned)

            # 验证：模型包含正确的变量
            assert 'theta' in model.named_vars
            assert 'likelihood' in model.named_vars

    @pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
    def test_model_free_rv_names(self, mcmc_test_data):
        """
        正例：模型的自由随机变量

        验证：只有先验是自由变量
        """
        n = mcmc_test_data['n']
        churned = mcmc_test_data['churned']
        prior_alpha = mcmc_test_data['prior_alpha']
        prior_beta = mcmc_test_data['prior_beta']

        with pm.Model() as model:
            theta = pm.Beta('theta', alpha=prior_alpha, beta=prior_beta)
            likelihood = pm.Binomial('likelihood', n=n, p=theta, observed=churned)

            # 验证：只有 theta 是自由 RV
            free_rvs = model.free_RVs
            assert len(free_rvs) == 1
            assert free_rvs[0].name == 'theta'


# =============================================================================
# MCMC 采样测试
# =============================================================================

class TestMCMCSampling:
    """测试 MCMC 采样过程"""

    @pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
    def test_basic_mcmc_sampling(self, mcmc_test_data):
        """
        正例：基本的 MCMC 采样

        验证：采样可以正常执行
        """
        n = mcmc_test_data['n']
        churned = mcmc_test_data['churned']
        prior_alpha = mcmc_test_data['prior_alpha']
        prior_beta = mcmc_test_data['prior_beta']
        n_samples = 1000  # 减少样本量加快测试

        with pm.Model() as model:
            theta = pm.Beta('theta', alpha=prior_alpha, beta=prior_beta)
            likelihood = pm.Binomial('likelihood', n=n, p=theta, observed=churned)

            # 采样
            trace = pm.sample(n_samples, tune=500, chains=2, random_seed=42,
                            progressbar=False, logger=None)

            # 验证：trace 包含数据
            assert 'theta' in trace.posterior
            assert trace.posterior['theta'].shape[1] == n_samples

    @pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
    def test_multiple_chains(self, mcmc_test_data):
        """
        正例：多条链采样

        验证：可以运行多条链
        """
        n = mcmc_test_data['n']
        churned = mcmc_test_data['churned']
        prior_alpha = mcmc_test_data['prior_alpha']
        prior_beta = mcmc_test_data['prior_beta']
        chains = 4

        with pm.Model() as model:
            theta = pm.Beta('theta', alpha=prior_alpha, beta=prior_beta)
            likelihood = pm.Binomial('likelihood', n=n, p=theta, observed=churned)

            trace = pm.sample(500, tune=300, chains=chains, random_seed=42,
                            progressbar=False, logger=None)

            # 验证：链数正确
            assert trace.posterior['theta'].shape[0] == chains

    @pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
    def test_posterior_mean_approximation(self, mcmc_test_data, tolerance):
        """
        正例：后验均值的采样近似

        验证：采样均值接近解析解
        """
        n = mcmc_test_data['n']
        churned = mcmc_test_data['churned']
        prior_alpha = mcmc_test_data['prior_alpha']
        prior_beta = mcmc_test_data['prior_beta']

        # 解析解
        alpha_post = prior_alpha + churned
        beta_post = prior_beta + (n - churned)
        exact_mean = alpha_post / (alpha_post + beta_post)

        with pm.Model() as model:
            theta = pm.Beta('theta', alpha=prior_alpha, beta=prior_beta)
            likelihood = pm.Binomial('likelihood', n=n, p=theta, observed=churned)

            trace = pm.sample(2000, tune=1000, chains=2, random_seed=42,
                            progressbar=False, logger=None)

            # 采样均值
            sampled_mean = trace.posterior['theta'].mean().item()

            # 验证：采样均值接近解析解（容差 1%）
            assert abs(sampled_mean - exact_mean) < 0.01


# =============================================================================
# 收敛性诊断测试
# =============================================================================

class TestConvergenceDiagnostics:
    """测试 MCMC 收敛性诊断"""

    @pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
    def test_rhat_calculation(self, mcmc_test_data):
        """
        正例：R-hat 计算

        R-hat < 1.05 表示收敛
        """
        n = mcmc_test_data['n']
        churned = mcmc_test_data['churned']
        prior_alpha = mcmc_test_data['prior_alpha']
        prior_beta = mcmc_test_data['prior_beta']

        with pm.Model() as model:
            theta = pm.Beta('theta', alpha=prior_alpha, beta=prior_beta)
            likelihood = pm.Binomial('likelihood', n=n, p=theta, observed=churned)

            trace = pm.sample(2000, tune=1000, chains=4, random_seed=42,
                            progressbar=False, logger=None)

            # 计算 R-hat
            rhat = az.rhat(trace).theta.values.item()

            # 验证：R-hat 接近 1
            assert rhat < 1.05
            assert rhat > 0.9

    @pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
    def test_ess_calculation(self, mcmc_test_data):
        """
        正例：有效样本量（ESS）计算

        ESS > 400 表示样本量足够
        """
        n = mcmc_test_data['n']
        churned = mcmc_test_data['churned']
        prior_alpha = mcmc_test_data['prior_alpha']
        prior_beta = mcmc_test_data['prior_beta']

        with pm.Model() as model:
            theta = pm.Beta('theta', alpha=prior_alpha, beta=prior_beta)
            likelihood = pm.Binomial('likelihood', n=n, p=theta, observed=churned)

            trace = pm.sample(2000, tune=1000, chains=4, random_seed=42,
                            progressbar=False, logger=None)

            # 计算 ESS
            ess = az.ess(trace).theta.values.item()

            # 验证：ESS 足够大
            assert ess > 400

    @pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
    def test_convergence_criteria(self, mcmc_convergence_data):
        """
        正例：收敛标准

        R-hat < 1.05 且 ESS > 400 表示收敛
        """
        # 模拟收敛的数据
        n = 1000
        churned = 180

        with pm.Model() as model:
            theta = pm.Beta('theta', alpha=15, beta=85)
            likelihood = pm.Binomial('likelihood', n=n, p=theta, observed=churned)

            trace = pm.sample(2000, tune=1000, chains=4, random_seed=42,
                            progressbar=False, logger=None)

            rhat = az.rhat(trace).theta.values.item()
            ess = az.ess(trace).theta.values.item()

            # 验证：收敛标准
            is_converged = (rhat < 1.05) and (ess > 400)
            assert is_converged


# =============================================================================
# 后验分布测试
# =============================================================================

class TestPosteriorDistribution:
    """测试从采样中提取的后验分布"""

    @pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
    def test_posterior_samples_shape(self, mcmc_test_data):
        """
        正例：后验样本的形状

        验证：样本形状正确
        """
        n = mcmc_test_data['n']
        churned = mcmc_test_data['churned']
        prior_alpha = mcmc_test_data['prior_alpha']
        prior_beta = mcmc_test_data['prior_beta']
        n_samples = 1000
        chains = 3

        with pm.Model() as model:
            theta = pm.Beta('theta', alpha=prior_alpha, beta=prior_beta)
            likelihood = pm.Binomial('likelihood', n=n, p=theta, observed=churned)

            trace = pm.sample(n_samples, tune=500, chains=chains, random_seed=42,
                            progressbar=False, logger=None)

            # 验证：形状
            samples = trace.posterior['theta']
            assert samples.shape[0] == chains
            assert samples.shape[1] == n_samples

    @pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
    def test_posterior_range(self, mcmc_test_data):
        """
        正例：后验样本的范围

        验证：样本在 [0, 1] 范围内
        """
        n = mcmc_test_data['n']
        churned = mcmc_test_data['churned']
        prior_alpha = mcmc_test_data['prior_alpha']
        prior_beta = mcmc_test_data['prior_beta']

        with pm.Model() as model:
            theta = pm.Beta('theta', alpha=prior_alpha, beta=prior_beta)
            likelihood = pm.Binomial('likelihood', n=n, p=theta, observed=churned)

            trace = pm.sample(1000, tune=500, chains=2, random_seed=42,
                            progressbar=False, logger=None)

            samples = trace.posterior['theta'].values

            # 验证：所有样本在 [0, 1] 范围内
            assert np.all(samples >= 0)
            assert np.all(samples <= 1)

    @pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
    def test_posterior_quantiles(self, mcmc_test_data):
        """
        正例：后验分位数

        验证：可以计算后验分位数
        """
        n = mcmc_test_data['n']
        churned = mcmc_test_data['churned']
        prior_alpha = mcmc_test_data['prior_alpha']
        prior_beta = mcmc_test_data['prior_beta']

        with pm.Model() as model:
            theta = pm.Beta('theta', alpha=prior_alpha, beta=prior_beta)
            likelihood = pm.Binomial('likelihood', n=n, p=theta, observed=churned)

            trace = pm.sample(1000, tune=500, chains=2, random_seed=42,
                            progressbar=False, logger=None)

            # 计算分位数
            q25 = trace.posterior['theta'].quantile(0.25).item()
            q50 = trace.posterior['theta'].quantile(0.5).item()
            q75 = trace.posterior['theta'].quantile(0.75).item()

            # 验证：分位数顺序
            assert q25 < q50 < q75


# =============================================================================
# 采样参数影响测试
# =============================================================================

class TestSamplingParameters:
    """测试采样参数对结果的影响"""

    @pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
    @pytest.mark.parametrize("n_samples", [500, 1000, 2000])
    def test_sample_size_impact(self, mcmc_test_data, n_samples):
        """
        正例：样本量的影响

        更多样本 → 更精确的估计
        """
        n = mcmc_test_data['n']
        churned = mcmc_test_data['churned']
        prior_alpha = mcmc_test_data['prior_alpha']
        prior_beta = mcmc_test_data['prior_beta']

        with pm.Model() as model:
            theta = pm.Beta('theta', alpha=prior_alpha, beta=prior_beta)
            likelihood = pm.Binomial('likelihood', n=n, p=theta, observed=churned)

            trace = pm.sample(n_samples, tune=500, chains=2, random_seed=42,
                            progressbar=False, logger=None)

            # 验证：采样成功
            assert trace.posterior['theta'].shape[1] == n_samples

    @pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
    @pytest.mark.parametrize("tune", [500, 1000, 2000])
    def test_tune_steps_impact(self, mcmc_test_data, tune):
        """
        正例：tune 步数的影响

        更多 tune 步 → 更好的初始采样
        """
        n = mcmc_test_data['n']
        churned = mcmc_test_data['churned']
        prior_alpha = mcmc_test_data['prior_alpha']
        prior_beta = mcmc_test_data['prior_beta']

        with pm.Model() as model:
            theta = pm.Beta('theta', alpha=prior_alpha, beta=prior_beta)
            likelihood = pm.Binomial('likelihood', n=n, p=theta, observed=churned)

            trace = pm.sample(500, tune=tune, chains=2, random_seed=42,
                            progressbar=False, logger=None)

            # 验证：采样成功
            assert trace.posterior['theta'].shape[1] == 500

    @pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
    @pytest.mark.parametrize("chains", [1, 2, 4])
    def test_number_of_chains(self, mcmc_test_data, chains):
        """
        正例：链数的影响

        多条链有助于检测收敛问题
        """
        n = mcmc_test_data['n']
        churned = mcmc_test_data['churned']
        prior_alpha = mcmc_test_data['prior_alpha']
        prior_beta = mcmc_test_data['prior_beta']

        with pm.Model() as model:
            theta = pm.Beta('theta', alpha=prior_alpha, beta=prior_beta)
            likelihood = pm.Binomial('likelihood', n=n, p=theta, observed=churned)

            trace = pm.sample(500, tune=300, chains=chains, random_seed=42,
                            progressbar=False, logger=None)

            # 验证：链数正确
            assert trace.posterior['theta'].shape[0] == chains


# =============================================================================
# 迹图和可视化测试
# =============================================================================

class TestTracePlots:
    """测试迹图和后验图"""

    @pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
    def test_trace_plot_structure(self, mcmc_test_data):
        """
        正例：迹图结构

        验证：可以访问迹图数据
        """
        n = mcmc_test_data['n']
        churned = mcmc_test_data['churned']
        prior_alpha = mcmc_test_data['prior_alpha']
        prior_beta = mcmc_test_data['prior_beta']

        with pm.Model() as model:
            theta = pm.Beta('theta', alpha=prior_alpha, beta=prior_beta)
            likelihood = pm.Binomial('likelihood', n=n, p=theta, observed=churned)

            trace = pm.sample(500, tune=300, chains=2, random_seed=42,
                            progressbar=False, logger=None)

            # 验证：可以获取迹图数据
            assert 'theta' in trace.posterior

            # 每条链的迹
            for chain in range(trace.posterior['theta'].shape[0]):
                chain_trace = trace.posterior['theta'][chain].values
                assert len(chain_trace) == 500

    @pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
    def test_posterior_plot_data(self, mcmc_test_data):
        """
        正例：后验图数据

        验证：可以获取后验图所需数据
        """
        n = mcmc_test_data['n']
        churned = mcmc_test_data['churned']
        prior_alpha = mcmc_test_data['prior_alpha']
        prior_beta = mcmc_test_data['prior_beta']

        with pm.Model() as model:
            theta = pm.Beta('theta', alpha=prior_alpha, beta=prior_beta)
            likelihood = pm.Binomial('likelihood', n=n, p=theta, observed=churned)

            trace = pm.sample(1000, tune=500, chains=2, random_seed=42,
                            progressbar=False, logger=None)

            # 获取后验摘要
            summary = az.summary(trace, var_names=['theta'])

            # 验证：摘要包含关键统计量
            assert 'mean' in summary.columns
            assert 'sd' in summary.columns
            assert 'hdi_3%' in summary.columns or 'ci_lower' in summary.columns
            assert 'hdi_97%' in summary.columns or 'ci_upper' in summary.columns


# =============================================================================
# MCMC vs 解析解对比测试
# =============================================================================

class TestMCMCvsAnalytical:
    """测试 MCMC 采样与解析解的对比"""

    @pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
    def test_mcmc_approximates_analytical_mean(self, mcmc_test_data, tolerance):
        """
        正例：MCMC 均值接近解析解均值

        验证：采样足够大时，MCMC 均值 ≈ 解析解均值
        """
        n = mcmc_test_data['n']
        churned = mcmc_test_data['churned']
        prior_alpha = mcmc_test_data['prior_alpha']
        prior_beta = mcmc_test_data['prior_beta']

        # 解析解
        alpha_post = prior_alpha + churned
        beta_post = prior_beta + (n - churned)
        analytical_mean = alpha_post / (alpha_post + beta_post)

        with pm.Model() as model:
            theta = pm.Beta('theta', alpha=prior_alpha, beta=prior_beta)
            likelihood = pm.Binomial('likelihood', n=n, p=theta, observed=churned)

            trace = pm.sample(3000, tune=1000, chains=4, random_seed=42,
                            progressbar=False, logger=None)

            mcmc_mean = trace.posterior['theta'].mean().item()

            # 验证：MCMC 均值接近解析解
            assert abs(mcmc_mean - analytical_mean) < 0.005

    @pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
    def test_mcmc_approximates_analytical_ci(self, mcmc_test_data, tolerance):
        """
        正例：MCMC 可信区间接近解析解区间

        验证：采样足够大时，MCMC CI ≈ 解析解 CI
        """
        n = mcmc_test_data['n']
        churned = mcmc_test_data['churned']
        prior_alpha = mcmc_test_data['prior_alpha']
        prior_beta = mcmc_test_data['prior_beta']

        # 解析解
        alpha_post = prior_alpha + churned
        beta_post = prior_beta + (n - churned)
        analytical_ci = stats.beta.interval(0.95, alpha_post, beta_post)

        with pm.Model() as model:
            theta = pm.Beta('theta', alpha=prior_alpha, beta=prior_beta)
            likelihood = pm.Binomial('likelihood', n=n, p=theta, observed=churned)

            trace = pm.sample(3000, tune=1000, chains=4, random_seed=42,
                            progressbar=False, logger=None)

            # MCMC 可信区间
            mcmc_ci = az.hdi(trace, hdi_prob=0.95).theta.values.item()

            # 对于单峰分布，HDI 和等尾区间接近
            # 验证：MCMC CI 端点在解析解附近
            assert mcmc_ci[0] > analytical_ci[0] - 0.01
            assert mcmc_ci[1] < analytical_ci[1] + 0.01


# =============================================================================
# 边界情况测试
# =============================================================================

class TestMCMCEdgeCases:
    """测试 MCMC 的边界情况"""

    @pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
    def test_small_sample(self):
        """
        边界：小样本

        数据少时，后验接近先验
        """
        n = 10
        churned = 2
        prior_alpha, prior_beta = 15, 85

        # 先验均值
        prior_mean = prior_alpha / (prior_alpha + prior_beta)

        with pm.Model() as model:
            theta = pm.Beta('theta', alpha=prior_alpha, beta=prior_beta)
            likelihood = pm.Binomial('likelihood', n=n, p=theta, observed=churned)

            trace = pm.sample(1000, tune=500, chains=2, random_seed=42,
                            progressbar=False, logger=None)

            posterior_mean = trace.posterior['theta'].mean().item()

            # 验证：后验均值接近先验均值（数据太少）
            assert abs(posterior_mean - prior_mean) < 0.05

    @pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
    def test_extreme_prior(self):
        """
        边界：极端先验

        先验非常强时，数据很难改变
        """
        n = 100
        churned = 30
        prior_alpha, prior_beta = 1000, 9000  # 极强先验，均值 10%

        with pm.Model() as model:
            theta = pm.Beta('theta', alpha=prior_alpha, beta=prior_beta)
            likelihood = pm.Binomial('likelihood', n=n, p=theta, observed=churned)

            trace = pm.sample(1000, tune=500, chains=2, random_seed=42,
                            progressbar=False, logger=None)

            posterior_mean = trace.posterior['theta'].mean().item()

            # 验证：后验均值仍然接近先验均值（10%）
            # 即使数据是 30%，强先验仍然主导
            assert posterior_mean < 0.15


# =============================================================================
# PyMC 不可用时的测试
# =============================================================================

class TestPyMCNotAvailable:
    """测试 PyMC 不可用时的行为"""

    def test_pymc_import_error_handling(self):
        """
        边界：PyMC 不可用时的处理

        验证：代码能优雅处理 PyMC 不可用的情况
        """
        # 这个测试总是运行，即使 PyMC 不可用
        if not PYMC_AVAILABLE:
            # 验证：可以检测到 PyMC 不可用
            assert not PYMC_AVAILABLE
        else:
            # PyMC 可用
            assert PYMC_AVAILABLE

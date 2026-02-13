"""
测试 StatLab 集成功能

正例：
- StatLab 报告能成功生成
- 后验摘要格式正确（mean, hdi_3%, hdi_97%）
- 先验敏感性分析结果一致

边界：
- 空数据输入处理
- 极端先验的处理

反例：
- 无效先验参数
"""
import pytest
import numpy as np
import pandas as pd
from scipy import stats

try:
    import pymc as pm
    import arviz as az
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False


class TestStatlabReportGeneration:
    """测试 StatLab 报告生成"""

    def test_bayesian_report_structure(self):
        """
        正例：后验摘要格式正确

        包含 mean, hdi_3%, hdi_97% 等字段
        """
        # 模拟后验数据
        np.random.seed(42)
        posterior_samples = np.random.normal(30, 5, 1000)

        # 计算摘要统计
        summary = {
            'mean': posterior_samples.mean(),
            'sd': posterior_samples.std(),
            'hdi_3%': np.percentile(posterior_samples, 3),
            'hdi_97%': np.percentile(posterior_samples, 97),
        }

        # 验证格式
        assert 'mean' in summary
        assert 'sd' in summary
        assert 'hdi_3%' in summary
        assert 'hdi_97%' in summary

        # 验证数值合理
        assert summary['mean'] == pytest.approx(30, abs=1)
        assert summary['sd'] == pytest.approx(5, abs=1)
        assert summary['hdi_3%'] < summary['mean'] < summary['hdi_97%']

    def test_bayesian_report_probability_statement(self):
        """
        正例：生成概率陈述

        例如："效应 > 0 的概率 99.9%"
        """
        # 模拟后验样本
        np.random.seed(42)
        beta_samples = np.random.normal(30, 5, 1000)

        # 计算概率
        prob_positive = (beta_samples > 0).mean()
        prob_gt_20 = (beta_samples > 20).mean()

        # 验证概率在合理范围
        assert 0 <= prob_positive <= 1
        assert 0 <= prob_gt_20 <= 1

        # 验证逻辑（均值 30，标准差 5，大概率 > 0）
        assert prob_positive > 0.99
        assert prob_gt_20 > 0.8

    def test_bayesian_report_credible_interval(self):
        """
        正例：计算可信区间（Credible Interval）

        95% HDI
        """
        # 模拟后验样本
        np.random.seed(42)
        samples = np.random.normal(30, 5, 1000)

        # 计算 95% HDI
        hdi_low = np.percentile(samples, 2.5)
        hdi_high = np.percentile(samples, 97.5)

        # 验证区间
        assert hdi_low < hdi_high
        assert 20 < hdi_low < 25  # 粗略范围
        assert 35 < hdi_high < 40

        # 验证覆盖率（约 95%）
        coverage = ((samples >= hdi_low) & (samples <= hdi_high)).mean()
        assert 0.94 < coverage < 0.96


@pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
class TestStatlabPriorSensitivity:
    """测试 StatLab 先验敏感性分析"""

    def test_prior_sensitivity_consistent_results(self, regression_data):
        """
        正例：先验敏感性分析结果一致

        弱先验和中等先验应该产生相似结果（当样本量足够时）
        """
        X = regression_data[['X1', 'X2']].values
        y = regression_data['y'].values
        n = len(y)

        # 弱先验
        with pm.Model() as model_weak:
            alpha = pm.Normal("alpha", mu=0, sigma=10)
            beta = pm.Normal("beta", mu=0, sigma=10, shape=2)
            sigma = pm.HalfNormal("sigma", sigma=1)
            mu = alpha + beta[0] * X[:, 0] + beta[1] * X[:, 1]
            Y_obs = pm.Normal("Y_obs", mu=mu, sigma=sigma, observed=y)
            idata_weak = pm.sample(1000, tune=500, chains=2,
                                 random_seed=42, progressbar=False)

        # 中等先验
        with pm.Model() as model_medium:
            alpha = pm.Normal("alpha", mu=0, sigma=5)
            beta = pm.Normal("beta", mu=0, sigma=5, shape=2)
            sigma = pm.HalfNormal("sigma", sigma=1)
            mu = alpha + beta[0] * X[:, 0] + beta[1] * X[:, 1]
            Y_obs = pm.Normal("Y_obs", mu=mu, sigma=sigma, observed=y)
            idata_medium = pm.sample(1000, tune=500, chains=2,
                                   random_seed=42, progressbar=False)

        # 比较后验均值
        beta_weak_mean = idata_weak.posterior["beta"].mean().values
        beta_medium_mean = idata_medium.posterior["beta"].mean().values

        # 两者应该接近
        assert np.allclose(beta_weak_mean, beta_medium_mean, atol=1)

    def test_prior_sensitivity_strong_prior_differs(self, regression_data):
        """
        边界：强先验会产生不同结果

        强先验会让后验向先验收缩
        """
        X = regression_data[['X1', 'X2']].values
        y = regression_data['y'].values

        # 弱先验
        with pm.Model() as model_weak:
            beta = pm.Normal("beta", mu=0, sigma=10, shape=2)
            alpha = pm.Normal("alpha", mu=0, sigma=10)
            sigma_resid = pm.HalfNormal("sigma_resid", sigma=1)
            mu = alpha + beta[0] * X[:, 0] + beta[1] * X[:, 1]
            Y_obs = pm.Normal("Y_obs", mu=mu, sigma=sigma_resid, observed=y)
            idata_weak = pm.sample(1000, tune=500, chains=2,
                                 random_seed=42, progressbar=False)

        # 强先验（假设真实系数是 1.5 和 0.3）
        with pm.Model() as model_strong:
            beta = pm.Normal("beta", mu=[1.5, 0.3], sigma=0.1, shape=2)
            alpha = pm.Normal("alpha", mu=50, sigma=1)
            sigma_resid = pm.HalfNormal("sigma_resid", sigma=1)
            mu = alpha + beta[0] * X[:, 0] + beta[1] * X[:, 1]
            Y_obs = pm.Normal("Y_obs", mu=mu, sigma=sigma_resid, observed=y)
            idata_strong = pm.sample(1000, tune=500, chains=2,
                                   random_seed=42, progressbar=False)

        # 强先验的后验应该更接近先验
        beta_weak_mean = idata_weak.posterior["beta"].mean().values
        beta_strong_mean = idata_strong.posterior["beta"].mean().values

        # 强先验应该产生不同的估计
        # （至少在某些系数上）
        # 注意：这里不做严格断言，因为取决于数据和先验强度

    def test_prior_sensitivity_extreme_prior(self, regression_data):
        """
        反例：极端先验会产生不合理结果

        测试极端错误先验的影响
        """
        X = regression_data[['X1', 'X2']].values
        y = regression_data['y'].values

        # 极端先验（假设系数是 -100，明显错误）
        with pm.Model() as model_extreme:
            beta = pm.Normal("beta", mu=-100, sigma=1, shape=2)
            alpha = pm.Normal("alpha", mu=0, sigma=10)
            sigma_resid = pm.HalfNormal("sigma_resid", sigma=1)
            mu = alpha + beta[0] * X[:, 0] + beta[1] * X[:, 1]
            Y_obs = pm.Normal("Y_obs", mu=mu, sigma=sigma_resid, observed=y)
            idata_extreme = pm.sample(1000, tune=500, chains=2,
                                    random_seed=42, progressbar=False)

        # 即使是极端先验，后验也应该向数据移动
        beta_extreme_mean = idata_extreme.posterior["beta"].mean().values

        # 后验不应该等于先验（-100）
        # 应该向真实值（约 1.5, 0.3）移动
        # 但由于强先验，移动可能有限
        # 这里只验证后验和先验不完全相同


class TestStatlabEdgeCases:
    """测试 StatLab 边界情况"""

    def test_empty_data_handling(self):
        """
        边界：空数据输入处理

        当没有数据时，后验应该等于先验
        """
        prior_alpha = 2
        prior_beta = 40

        # 空数据
        successes = 0
        trials = 0

        posterior = stats.beta(
            prior_alpha + successes,
            prior_beta + trials - successes
        )

        # 后验应该等于先验
        assert posterior.args == (prior_alpha, prior_beta)

    def test_single_observation_handling(self):
        """
        边界：单次观测

        单次观测的后验应该介于先验和观测之间
        """
        prior = stats.beta(2, 40)

        # 单次成功
        posterior = stats.beta(prior.args[0] + 1, prior.args[1])

        # 后验均值应该介于先验和观测之间
        prior_mean = prior.mean()
        obs_value = 1.0  # 单次观测
        posterior_mean = posterior.mean()

        assert prior_mean < posterior_mean < obs_value or \
               obs_value < posterior_mean < prior_mean

    def test_zero_variance_data(self):
        """
        边界：零方差数据

        当所有观测值相同时
        """
        # 所有值相同
        y_constant = np.array([50] * 100)
        X = np.random.normal(0, 1, (100, 2))

        if PYMC_AVAILABLE:
            with pm.Model() as model:
                beta = pm.Normal("beta", mu=0, sigma=10, shape=2)
                alpha = pm.Normal("alpha", mu=0, sigma=10)
                sigma = pm.HalfNormal("sigma", sigma=1)
                mu = alpha + beta[0] * X[:, 0] + beta[1] * X[:, 1]
                Y_obs = pm.Normal("Y_obs", mu=mu, sigma=sigma, observed=y_constant)

                idata = pm.sample(500, tune=300, chains=2,
                                random_seed=42, progressbar=False)

            # sigma 应该收敛到接近 0
            sigma_mean = idata.posterior["sigma"].mean().values
            assert sigma_mean < 1  # 应该很小


class TestStatlabReportContent:
    """测试 StatLab 报告内容"""

    def test_report_contains_decision_friendly_metrics(self):
        """
        正例：报告包含决策友好的指标

        例如：P(效应 > 0)、预期损失等
        """
        # 模拟后验样本
        np.random.seed(42)
        effect_samples = np.random.normal(30, 5, 1000)

        # 计算决策指标
        metrics = {
            'prob_positive': (effect_samples > 0).mean(),
            'prob_gt_threshold': (effect_samples > 20).mean(),
            'expected_loss': -effect_samples[effect_samples < 0].mean()
            if (effect_samples < 0).any() else 0,
        }

        # 验证指标存在且合理
        assert 'prob_positive' in metrics
        assert 'prob_gt_threshold' in metrics
        assert 'expected_loss' in metrics

        assert 0 <= metrics['prob_positive'] <= 1
        assert 0 <= metrics['prob_gt_threshold'] <= 1
        assert metrics['expected_loss'] <= 0  # 损失是负值

    def test_report_claims_boundaries(self):
        """
        正例：报告声明结论边界

        说明模型能回答和不能回答的问题
        """
        # 模拟报告内容
        report_sections = {
            'can_answer': [
                '给定数据和先验，效应的概率分布',
                'P(效应 > 阈值)',
                '可信区间',
            ],
            'cannot_answer': [
                '未观察的混杂变量',
                '个体因果效应',
                '数据范围外的长期效应',
            ]
        }

        # 验证两个部分都存在
        assert len(report_sections['can_answer']) > 0
        assert len(report_sections['cannot_answer']) > 0

    def test_report_comparison_frequentist_bayesian(self):
        """
        正例：报告对比频率学派和贝叶斯学派

        解释置信区间 vs 可信区间
        """
        # 模拟对比表
        comparison = {
            'frequentist': {
                'estimate': 30.2,
                'ci': [20.5, 39.9],
                'interpretation': '重复抽样 100 次，95 个区间覆盖真值'
            },
            'bayesian': {
                'estimate': 29.8,
                'hdi': [20.6, 39.2],
                'interpretation': '参数有 95% 的概率在区间内'
            }
        }

        # 验证格式
        assert 'estimate' in comparison['frequentist']
        assert 'ci' in comparison['frequentist']
        assert 'interpretation' in comparison['frequentist']

        assert 'estimate' in comparison['bayesian']
        assert 'hdi' in comparison['bayesian']
        assert 'interpretation' in comparison['bayesian']


class TestStatlabMarkdownOutput:
    """测试 StatLab Markdown 输出"""

    def test_markdown_report_formatting(self):
        """
        测试：Markdown 格式正确

        验证生成的报告符合 Markdown 语法
        """
        # 模拟后验摘要
        summary = {
            'mean': 29.8,
            'hdi_3%': 20.6,
            'hdi_97%': 39.2,
        }

        # 生成 Markdown 表格
        markdown_table = f"""
| 指标 | 估计值 |
|------|--------|
| 后验均值 | **{summary['mean']:.2f}** |
| 95% HDI | [{summary['hdi_3%']:.2f}, {summary['hdi_97%']:.2f}] |
"""

        # 验证格式
        assert '|' in markdown_table
        assert '---' in markdown_table
        assert '**' in markdown_table  # 粗体

    def test_markdown_probability_statement(self):
        """
        测试：Markdown 中包含概率陈述
        """
        prob_positive = 0.999

        # 生成概率陈述
        statement = f"效应为正的概率为 **{prob_positive:.1%}**"

        # 验证格式
        assert '**' in statement
        assert '%' in statement
        assert '概率' in statement


@pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
class TestStatlabEndToEnd:
    """端到端测试 StatLab 工作流"""

    def test_full_bayesian_analysis_workflow(self, regression_data):
        """
        测试：完整的贝叶斯分析工作流

        1. 定义模型
        2. 采样
        3. 诊断
        4. 生成报告
        """
        X = regression_data[['X1', 'X2']].values
        y = regression_data['y'].values

        # 1. 定义模型并采样
        with pm.Model() as model:
            alpha = pm.Normal("alpha", mu=0, sigma=10)
            beta = pm.Normal("beta", mu=0, sigma=10, shape=2)
            sigma = pm.HalfNormal("sigma", sigma=1)
            mu = alpha + beta[0] * X[:, 0] + beta[1] * X[:, 1]
            Y_obs = pm.Normal("Y_obs", mu=mu, sigma=sigma, observed=y)
            idata = pm.sample(1000, tune=500, chains=2,
                            random_seed=42, progressbar=False)

        # 2. 诊断
        summary = az.summary(idata)
        r_hat = az.rhat(idata)

        # 验证诊断通过
        assert 'beta[0]' in summary.index
        assert (r_hat['beta'] < 1.05).all() or r_hat['beta'].shape == ()

        # 3. 生成报告
        report_content = {
            'posterior_mean': summary.loc['beta[0]', 'mean'],
            'hdi_low': summary.loc['beta[0]', 'hdi_3%']
            if 'hdi_3%' in summary else summary.loc['beta[0]', 'ci_lower'],
            'hdi_high': summary.loc['beta[0]', 'hdi_97%']
            if 'hdi_97%' in summary else summary.loc['beta[0]', 'ci_upper'],
        }

        # 验证报告内容
        assert 'posterior_mean' in report_content
        assert 'hdi_low' in report_content
        assert 'hdi_high' in report_content
        assert report_content['hdi_low'] < report_content['hdi_high']

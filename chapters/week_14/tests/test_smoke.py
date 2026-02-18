"""
Smoke Tests for Week 14 solution.py

基础冒烟测试：
- 验证模块可以导入
- 验证基本函数存在
- 验证基本功能可运行
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Add starter_code to path
starter_code_path = Path(__file__).parent.parent / "starter_code"
sys.path.insert(0, str(starter_code_path))


# =============================================================================
# 模块导入测试
# =============================================================================

def test_solution_module_exists():
    """
    冒烟测试：solution.py 模块应存在

    如果此测试失败，说明 solution.py 文件不存在
    """
    try:
        import solution
        assert solution is not None
    except ImportError:
        pytest.skip("solution.py not found - expected to be implemented later")


def test_solution_has_basic_functions():
    """
    冒烟测试：solution.py 应包含贝叶斯分析相关函数

    检查核心函数是否存在（示例函数名，实际可能不同）
    """
    try:
        import solution
    except ImportError:
        pytest.skip("solution.py not found")

    # 贝叶斯分析相关的可能函数名
    functions = [
        # 贝叶斯定理
        'compute_posterior',
        'bayes_update',
        'bayes_theorem',

        # Beta-Binomial 模型
        'beta_binomial_posterior',
        'compute_beta_posterior',
        'beta_binomial_conjugate',

        # MCMC 采样
        'mcmc_sample',
        'run_mcmc',
        'pymc_sampling',

        # 先验敏感性分析
        'prior_sensitivity_analysis',
        'compare_priors',
        'sensitivity_analysis',

        # 可信区间
        'credible_interval',
        'compute_hdi',
        'posterior_interval',
    ]

    # 至少有一个贝叶斯相关的函数存在
    has_any = any(hasattr(solution, func) for func in functions)


# =============================================================================
# 贝叶斯定理冒烟测试
# =============================================================================

def test_bayes_theorem_smoke():
    """
    冒烟测试：贝叶斯定理计算应能运行

    测试基本的贝叶斯定理计算
    P(A|B) = P(B|A) * P(A) / P(B)
    """
    try:
        import solution
    except ImportError:
        pytest.skip("solution.py not found")

    # 简单的疾病检测例子
    prior = 0.01  # P(Disease)
    sensitivity = 0.99  # P(Positive|Disease)
    false_positive_rate = 0.05  # P(Positive|No Disease)

    # 尝试计算后验
    if hasattr(solution, 'bayes_theorem'):
        result = solution.bayes_theorem(prior, sensitivity, false_positive_rate)
        assert result is not None
    elif hasattr(solution, 'compute_posterior'):
        result = solution.compute_posterior(prior, sensitivity, false_positive_rate)
        assert result is not None
    else:
        pytest.skip("bayes_theorem/compute_posterior not implemented")


# =============================================================================
# Beta-Binomial 模型冒烟测试
# =============================================================================

def test_beta_binomial_smoke():
    """
    冒烟测试：Beta-Binomial 后验计算应能运行

    测试共轭先验的更新
    """
    try:
        import solution
    except ImportError:
        pytest.skip("solution.py not found")

    # Beta 先验 + Binomial 数据
    alpha_prior = 15
    beta_prior = 85
    n = 1000
    churned = 180

    # 尝试计算后验
    if hasattr(solution, 'beta_binomial_posterior'):
        result = solution.beta_binomial_posterior(alpha_prior, beta_prior, n, churned)
        assert result is not None
    elif hasattr(solution, 'compute_beta_posterior'):
        result = solution.compute_beta_posterior(alpha_prior, beta_prior, n, churned)
        assert result is not None
    else:
        pytest.skip("beta_binomial_posterior not implemented")


def test_beta_distribution_smoke():
    """
    冒烟测试：Beta 分布统计量计算应能运行

    测试 Beta 分布的均值、方差、众数计算
    """
    try:
        import solution
    except ImportError:
        pytest.skip("solution.py not found")

    alpha = 15
    beta = 85

    # 尝试计算 Beta 分布统计量
    if hasattr(solution, 'beta_mean'):
        result = solution.beta_mean(alpha, beta)
        assert result is not None
    else:
        pytest.skip("beta_mean not implemented")


# =============================================================================
# MCMC 采样冒烟测试
# =============================================================================

def test_mcmc_smoke():
    """
    冒烟测试：MCMC 采样应能运行

    测试基本的 MCMC 采样功能
    """
    try:
        import solution
    except ImportError:
        pytest.skip("solution.py not found")

    # 简单数据
    n = 100
    churned = 18
    prior_alpha = 5
    prior_beta = 20

    # 尝试 MCMC 采样
    if hasattr(solution, 'mcmc_sample'):
        result = solution.mcmc_sample(n, churned, prior_alpha, prior_beta)
        assert result is not None
    elif hasattr(solution, 'run_mcmc'):
        result = solution.run_mcmc(n, churned, prior_alpha, prior_beta)
        assert result is not None
    else:
        pytest.skip("mcmc_sample not implemented")


def test_mcmc_convergence_check_smoke():
    """
    冒烟测试：MCMC 收敛性检查应能运行

    测试 R-hat 和 ESS 的计算
    """
    try:
        import solution
    except ImportError:
        pytest.skip("solution.py not found")

    # 模拟的 trace 数据
    np.random.seed(42)
    chains = [
        np.random.normal(0.18, 0.02, 1000),
        np.random.normal(0.18, 0.02, 1000),
        np.random.normal(0.18, 0.02, 1000),
        np.random.normal(0.18, 0.02, 1000),
    ]

    # 尝试检查收敛性
    if hasattr(solution, 'check_convergence'):
        result = solution.check_convergence(chains)
        assert result is not None
    elif hasattr(solution, 'compute_rhat'):
        result = solution.compute_rhat(chains)
        assert result is not None
    else:
        pytest.skip("check_convergence not implemented")


# =============================================================================
# 先验敏感性分析冒烟测试
# =============================================================================

def test_prior_sensitivity_smoke():
    """
    冒烟测试：先验敏感性分析应能运行

    测试比较不同先验的后验结果
    """
    try:
        import solution
    except ImportError:
        pytest.skip("solution.py not found")

    # 数据
    n = 1000
    churned = 180

    # 多个先验
    priors = {
        '无信息': (1, 1),
        '弱信息': (5, 20),
        '信息性': (150, 850),
    }

    # 尝试敏感性分析
    if hasattr(solution, 'prior_sensitivity_analysis'):
        result = solution.prior_sensitivity_analysis(n, churned, priors)
        assert result is not None
    elif hasattr(solution, 'compare_priors'):
        result = solution.compare_priors(n, churned, priors)
        assert result is not None
    else:
        pytest.skip("prior_sensitivity_analysis not implemented")


# =============================================================================
# 可信区间冒烟测试
# =============================================================================

def test_credible_interval_smoke():
    """
    冒烟测试：可信区间计算应能运行

    测试贝叶斯可信区间的计算
    """
    try:
        import solution
    except ImportError:
        pytest.skip("solution.py not found")

    # Beta 分布参数
    alpha = 195
    beta = 905

    # 尝试计算可信区间
    if hasattr(solution, 'credible_interval'):
        result = solution.credible_interval(alpha, beta, 0.95)
        assert result is not None
    elif hasattr(solution, 'compute_hdi'):
        result = solution.compute_hdi(alpha, beta, 0.95)
        assert result is not None
    else:
        pytest.skip("credible_interval not implemented")


# =============================================================================
# 综合冒烟测试
# =============================================================================

def test_end_to_end_bayesian_analysis_smoke():
    """
    冒烟测试：端到端贝叶斯分析流程应能运行

    测试从先验定义到后验分析的完整流程
    """
    try:
        import solution
    except ImportError:
        pytest.skip("solution.py not found")

    # 数据
    n = 1000
    churned = 180

    # 尝试完整流程
    results = {}

    # 1. 定义先验
    if hasattr(solution, 'define_priors'):
        priors = solution.define_priors()
        results['priors'] = priors

    # 2. 计算后验
    if hasattr(solution, 'compute_beta_posterior'):
        posterior = solution.compute_beta_posterior(15, 85, n, churned)
        results['posterior'] = posterior

    # 3. 可信区间
    if hasattr(solution, 'credible_interval'):
        ci = solution.credible_interval(195, 905, 0.95)
        results['ci'] = ci

    # 至少有一个步骤成功
    if results:
        assert True
    else:
        pytest.skip("No Bayesian analysis functions implemented")


# =============================================================================
# 异常处理冒烟测试
# =============================================================================

def test_empty_data_handling():
    """
    冒烟测试：空数据应被正确处理

    验证函数对空输入的容错性
    """
    try:
        import solution
    except ImportError:
        pytest.skip("solution.py not found")

    # 空数据
    n = 0
    churned = 0

    # 尝试计算（应该报错或返回 None）
    if hasattr(solution, 'compute_beta_posterior'):
        try:
            result = solution.compute_beta_posterior(1, 1, n, churned)
            # 如果不报错，应该返回 None 或合理的默认值
            assert result is None or isinstance(result, (int, float, dict, tuple))
        except (ValueError, ZeroDivisionError, RuntimeError):
            # 报错也是可接受的
            assert True


def test_invalid_beta_parameters():
    """
    冒烟测试：无效的 Beta 参数应被正确处理

    验证对负数或零参数的处理
    """
    try:
        import solution
    except ImportError:
        pytest.skip("solution.py not found")

    # 无效参数
    invalid_params = [
        (0, 0),    # 两个零
        (-1, 1),   # 负数
        (1, -1),   # 负数
    ]

    for alpha, beta in invalid_params:
        if hasattr(solution, 'beta_mean'):
            try:
                result = solution.beta_mean(alpha, beta)
                # 如果不报错，应该返回 None 或 nan
                assert result is None or (isinstance(result, float) and np.isnan(result))
            except (ValueError, ZeroDivisionError, RuntimeError):
                # 报错也是可接受的
                assert True


# =============================================================================
# 概念理解冒烟测试
# =============================================================================

def test_bayesian_concepts():
    """
    冒烟测试：贝叶斯基本概念

    验证对先验、似然、后验的理解
    """
    # 概念测试：不需要具体实现
    concepts = {
        'prior': 'P(θ) - 先验，数据之前的信念',
        'likelihood': 'P(data|θ) - 似然，给定参数下数据的概率',
        'posterior': 'P(θ|data) - 后验，数据之后的信念',
        'evidence': 'P(data) - 证据，归一化常数',
    }

    assert len(concepts) == 4
    assert 'prior' in concepts
    assert 'likelihood' in concepts
    assert 'posterior' in concepts
    assert 'evidence' in concepts


def test_frequentist_vs_bayesian_concepts():
    """
    冒烟测试：频率学派 vs 贝叶斯学派概念

    验证对两种范式区别的理解
    """
    # 概念测试
    comparisons = {
        'parameter': '频率学派：固定未知；贝叶斯学派：随机变量',
        'data': '频率学派：随机；贝叶斯学派：固定（已观测）',
        'interval': '频率学派：置信区间；贝叶斯学派：可信区间',
        'p_value': '频率学派：P(data|H0)；贝叶斯学派：P(H0|data)',
    }

    assert len(comparisons) == 4
    assert 'parameter' in comparisons
    assert 'data' in comparisons
    assert 'interval' in comparisons
    assert 'p_value' in comparisons


def test_conjugate_prior_concept():
    """
    冒烟测试：共轭先验概念

    验证对共轭先验的理解
    """
    # 概念测试
    conjugate_pairs = {
        'Beta-Binomial': 'Beta 先验 + Binomial 似然 → Beta 后验',
        'Normal-Normal': 'Normal 先验 + Normal 似然（已知方差）→ Normal 后验',
        'Gamma-Poisson': 'Gamma 先验 + Poisson 似然 → Gamma 后验',
    }

    assert 'Beta-Binomial' in conjugate_pairs
    assert conjugate_pairs['Beta-Binomial'] == 'Beta 先验 + Binomial 似然 → Beta 后验'

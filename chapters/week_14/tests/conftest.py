"""
Week 14 共享 Fixtures

提供测试用的共享数据和工具函数。

测试覆盖：
- 贝叶斯定理计算
- Beta-Binomial 共轭先验
- MCMC 采样和收敛性
- 先验敏感性分析
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import sys
from scipy import stats
from typing import Dict, List, Tuple, Optional

# 添加 starter_code 到 Python 路径
starter_code_path = Path(__file__).parent.parent / "starter_code"
if str(starter_code_path) not in sys.path:
    sys.path.insert(0, str(starter_code_path))


# =============================================================================
# 贝叶斯定理测试 Fixtures
# =============================================================================

@pytest.fixture
def basic_bayes_data():
    """
    基础贝叶斯定理测试数据

    场景：疾病检测（经典例子）
    - P(Disease) = 0.01 (先验概率)
    - P(Positive|Disease) = 0.99 (真阳性率)
    - P(Positive|No Disease) = 0.05 (假阳性率)
    """
    return {
        'prior_disease': 0.01,           # P(D) = 1%
        'sensitivity': 0.99,             # P(+|D) = 99%
        'false_positive_rate': 0.05,     # P(+|¬D) = 5%
        'expected_posterior': 0.166,     # P(D|+) ≈ 16.6%
    }


@pytest.fixture
def churn_bayes_data():
    """
    流失率贝叶斯估计数据

    场景：电商客户流失
    - 先验：Beta(15, 85) -> 均值 15%
    - 数据：1000 个客户中 180 个流失
    """
    return {
        'alpha_prior': 15,    # 先验 Beta 参数
        'beta_prior': 85,
        'n': 1000,            # 样本量
        'churned': 180,       # 流失数
        'expected_posterior_mean': 0.177,  # 后验均值约 17.7%
        'expected_alpha_post': 195,         # 15 + 180
        'expected_beta_post': 905,          # 85 + 820
    }


@pytest.fixture
def multiple_priors_data():
    """
    多先验对比数据

    用于先验敏感性分析
    """
    return {
        'n': 1000,
        'churned': 180,
        'priors': {
            '无信息': (1, 1),           # Beta(1, 1) -> 均匀分布
            '弱信息': (5, 20),          # Beta(5, 20) -> 均值 20%，方差大
            '信息性': (150, 850),       # Beta(150, 850) -> 均值 15%，方差小
            '市场部': (180, 820),       # 基于历史数据
            '产品部': (5, 15),          # 基于最近趋势
        }
    }


# =============================================================================
# Beta-Binomial 模型测试 Fixtures
# =============================================================================

@pytest.fixture
def beta_binomial_conjugate_data():
    """
    Beta-Binomial 共轭先验数据

    验证：Beta 先验 + Binomial 似然 = Beta 后验
    """
    return {
        'test_cases': [
            # (alpha_prior, beta_prior, n, churned, alpha_post, beta_post)
            (1, 1, 100, 20, 21, 81),      # 无信息先验
            (15, 85, 1000, 180, 195, 905),  # 信息性先验
            (5, 20, 50, 10, 15, 60),      # 弱信息先验
        ]
    }


@pytest.fixture
def beta_distribution_properties():
    """
    Beta 分布性质数据

    用于验证 Beta 分布的统计性质
    """
    return {
        'uniform': {
            'alpha': 1,
            'beta': 1,
            'mean': 0.5,
            'variance': 1/12,
            'mode': None,  # 均匀分布无众数
        },
        'informative_low': {
            'alpha': 15,
            'beta': 85,
            'mean': 15/100,
            'variance': (15 * 85) / (100**2 * 101),
            'mode': (15 - 1) / (100 - 2),  # (alpha-1)/(alpha+beta-2)
        },
        'informative_high': {
            'alpha': 85,
            'beta': 15,
            'mean': 85/100,
            'variance': (85 * 15) / (100**2 * 101),
            'mode': (85 - 1) / (100 - 2),
        },
        'symmetric': {
            'alpha': 10,
            'beta': 10,
            'mean': 0.5,
            'variance': (10 * 10) / (20**2 * 21),
            'mode': 0.5,
        }
    }


# =============================================================================
# MCMC 采样测试 Fixtures
# =============================================================================

@pytest.fixture
def mcmc_test_data():
    """
    MCMC 采样测试数据

    用于测试 PyMC 采样和收敛性
    """
    return {
        'n': 1000,
        'churned': 180,
        'prior_alpha': 15,
        'prior_beta': 85,
        'n_samples': 2000,
        'tune': 1000,
        'chains': 4,
        'expected_posterior_mean': 0.177,
        'rhat_threshold': 1.05,     # R-hat < 1.05 表示收敛
        'ess_threshold': 400,       # ESS > 400 表示样本量足够
    }


@pytest.fixture
def mcmc_convergence_data():
    """
    MCMC 收敛性测试数据

    包含收敛和未收敛的情况
    """
    return {
        'converged': {
            'rhat': 1.001,      # 接近 1，收敛
            'ess_bulk': 2000,   # 足够大
            'ess_tail': 1800,   # 足够大
            'is_converged': True,
        },
        'not_converged': {
            'rhat': 1.15,       # > 1.05，未收敛
            'ess_bulk': 100,    # 太小
            'ess_tail': 80,     # 太小
            'is_converged': False,
        },
        'borderline': {
            'rhat': 1.04,       # 接近阈值
            'ess_bulk': 450,    # 接近阈值
            'ess_tail': 380,    # 低于阈值
            'is_converged': False,  # ESS 不足
        }
    }


# =============================================================================
# 先验敏感性测试 Fixtures
# =============================================================================

@pytest.fixture
def prior_sensitivity_scenarios():
    """
    先验敏感性测试场景

    包含"敏感"和"不敏感"两种情况
    """
    return {
        'large_data': {
            'n': 1000,
            'churned': 180,
            'priors': {
                '无信息': (1, 1),
                '弱信息': (5, 20),
                '强信息': (150, 850),
            },
            'is_sensitive': False,  # 数据量大，不敏感
            'expected_mean_range': 0.02,  # 后验均值差异 < 2%
        },
        'small_data': {
            'n': 50,
            'churned': 10,
            'priors': {
                '无信息': (1, 1),
                '弱信息': (5, 20),
                '强信息': (150, 850),
            },
            'is_sensitive': True,   # 数据量小，敏感
            'expected_mean_range': 0.03,  # 后验均值差异 > 3%
        },
        'extreme_prior': {
            'n': 100,
            'churned': 20,
            'priors': {
                '极端悲观': (1, 99),    # 均值 1%
                '极端乐观': (99, 1),    # 均值 99%
                '中等': (10, 40),       # 均值 20%
            },
            'is_sensitive': True,
            'expected_mean_range': 0.10,  # 后验均值差异 > 10%
        }
    }


@pytest.fixture
def department_prior_disagreement():
    """
    部门间先验分歧数据

    场景：市场部 vs 产品部
    """
    return {
        'marketing': {
            'name': '市场部',
            'prior': (180, 820),    # 基于历史数据，均值 18%
            'rationale': '基于过去 1000 个客户的历史数据'
        },
        'product': {
            'name': '产品部',
            'prior': (5, 15),       # 基于最近投诉增加，均值 25%
            'rationale': '基于最近用户投诉增加的趋势'
        },
        'current_data': {
            'n': 100,
            'churned': 22,
        },
        'expected_divergence': True,  # 后验应该有差异
    }


# =============================================================================
# 边界情况 Fixtures
# =============================================================================

@pytest.fixture
def extreme_bayes_data():
    """
    极端贝叶斯计算数据

    包含边界和极端情况
    """
    return {
        'zero_probability': {
            'prior': 0.0,      # 先验概率为 0
            'likelihood': 0.5,
            'expected_posterior': 0.0,  # 后验仍为 0（零先验不会被更新）
        },
        'certainty_prior': {
            'prior': 1.0,      # 先验概率为 1
            'likelihood': 0.0,
            'expected_posterior': 1.0,  # 后验仍为 1
        },
        'perfect_test': {
            'sensitivity': 1.0,        # 完美敏感度
            'specificity': 1.0,        # 完美特异度
            'prior': 0.5,
            'positive_posterior': 1.0,  # 阳性 => 100% 患病
            'negative_posterior': 0.0,  # 阴性 => 0% 患病
        },
        'useless_test': {
            'sensitivity': 0.5,        # 无信息测试
            'specificity': 0.5,
            'prior': 0.5,
            'positive_posterior': 0.5,  # 后验 = 先验
        }
    }


@pytest.fixture
def edge_case_beta_parameters():
    """
    Beta 分布参数边界情况

    测试极端参数值
    """
    return {
        'uniform': (1, 1),           # 均匀
        'jeffreys': (0.5, 0.5),      # Jeffreys 先验（在 0 和 1 处发散）
        'very_strong': (1000, 9000), # 极强先验（均值 10%）
        'degenerate': (0.01, 0.01),  # 接近 0（数值不稳定）
        'one_sided': (1, 100),       # 强烈偏向 0
        'other_sided': (100, 1),     # 强烈偏向 1
    }


@pytest.fixture
def minimal_data():
    """
    最小数据集

    用于测试极端小样本情况
    """
    return {
        'single_observation': {
            'n': 1,
            'churned': 0,    # 唯一观测未流失
            'prior': (1, 1),
            'expected_posterior': (1, 2),  # Beta(1, 2)
        },
        'all_churned': {
            'n': 10,
            'churned': 10,   # 全部流失
            'prior': (1, 1),
            'expected_posterior': (11, 1),  # Beta(11, 1)
        },
        'none_churned': {
            'n': 10,
            'churned': 0,    # 全部未流失
            'prior': (1, 1),
            'expected_posterior': (1, 11),  # Beta(1, 11)
        }
    }


# =============================================================================
# 概念对比 Fixtures
# =============================================================================

@pytest.fixture
def frequentist_vs_bayesian_data():
    """
    频率学派 vs 贝叶斯学派对比数据

    展示两种方法的区别
    """
    return {
        'n': 1000,
        'churned': 180,
        'hypothesized_p': 0.15,
        'frequentist': {
            'point_estimate': 0.18,
            'confidence_interval': (0.156, 0.204),  # 95% CI
            'p_value': 0.03,  # p < 0.05，显著
        },
        'bayesian': {
            'posterior_mean': 0.177,
            'credible_interval': (0.154, 0.201),  # 95% CI
            'p_theta_gt_threshold': 0.98,  # P(θ > 15% | data)
        }
    }


# =============================================================================
# 验证工具 Fixtures
# =============================================================================

@pytest.fixture
def tolerance():
    """
    数值比较的容差
    """
    return {
        'rtol': 1e-3,
        'atol': 1e-5,
        'prob_tol': 0.01,         # 概率容差 1%
        'mean_tol': 0.005,        # 均值容差 0.5%
        'ci_tol': 0.01,           # 区间容差 1%
    }


@pytest.fixture
def bayesian_vocab():
    """
    贝叶斯术语字典
    """
    return {
        'prior': '先验 - P(θ)，在看到数据之前的信念',
        'likelihood': '似然 - P(data|θ)，如果参数是 θ，观察到数据的概率',
        'posterior': '后验 - P(θ|data)，看到数据后的信念',
        'evidence': '证据 - P(data)，数据的边际概率（归一化常数）',
        'conjugate_prior': '共轭先验 - 先验和后验属于同一分布族',
        'mcmc': 'MCMC - 马尔可夫链蒙特卡洛，用采样近似后验分布',
        'credible_interval': '可信区间 - 参数有 95% 的概率在这个区间里',
        'confidence_interval': '置信区间 - 长期频率下 95% 的区间包含真实参数',
        'prior_sensitivity': '先验敏感性 - 后验对先验选择的依赖程度',
        'rhat': 'R-hat - MCMC 收敛性诊断，< 1.05 表示收敛',
        'ess': 'ESS - 有效样本量，> 400 表示样本量足够',
        'hdi': 'HDI - 最高密度区间，包含 95% 后验概率的最窄区间',
    }


@pytest.fixture
def bayesian_formulas():
    """
    贝叶斯公式
    """
    return {
        'bayes_theorem': 'P(θ|data) = P(data|θ) × P(θ) / P(data)',
        'proportionality': 'P(θ|data) ∝ P(data|θ) × P(θ)',
        'beta_binomial_update': 'α_post = α_prior + successes, β_post = β_prior + failures',
        'beta_mean': 'E[θ] = α / (α + β)',
        'beta_mode': 'Mode(θ) = (α - 1) / (α + β - 2), for α, β > 1',
        'beta_variance': 'Var(θ) = αβ / [(α+β)²(α+β+1)]',
    }

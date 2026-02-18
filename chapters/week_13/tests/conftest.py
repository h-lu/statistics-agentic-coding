"""
Week 13 共享 Fixtures

提供测试用的共享数据和工具函数。

测试覆盖：
- 因果推断三层级（Association, Intervention, Counterfactual）
- DAG 基础（混杂、碰撞、链式结构）
- d-分离与后门准则
- RCT 模拟
- 观察研究中的因果推断（DID、IV、PSM）
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
# 因果推断三层级测试 Fixtures
# =============================================================================

@pytest.fixture
def association_vs_intervention_data():
    """
    关联 vs 干预测试数据

    模拟场景：优惠券（X）对流失率（Y）的影响
    - 存在混杂变量 U（高价值客户）
    - 关联 P(Y|X) != 干预 P(Y|do(X))
    """
    np.random.seed(42)
    n = 1000

    # 混杂变量：高价值客户（不可观测）
    U = np.random.binomial(1, 0.4, n)  # 40% 是高价值客户

    # Treatment：优惠券（受 U 影响）
    # 高价值客户更可能收到优惠券
    prob_X_given_U = np.where(U == 1, 0.8, 0.2)
    X = np.random.binomial(1, prob_X_given_U)

    # Outcome：流失率
    # 真实因果效应：优惠券降低流失率（-0.1）
    # 高价值客户流失率低（-0.3）
    Y_potential_0 = np.random.binomial(1, 0.3 - 0.2 * U, n)  # 不发优惠券时的流失率
    Y_potential_1 = np.random.binomial(1, 0.25 - 0.2 * U, n)  # 发优惠券时的流失率

    Y = np.where(X == 1, Y_potential_1, Y_potential_0)

    return {
        'X': X,  # Treatment: 优惠券
        'Y': Y,  # Outcome: 流失
        'U': U,  # Confounder: 高价值客户（不可观测）
        'true_ate': -0.05,  # 真实平均处理效应
        'n_samples': n
    }


@pytest.fixture
def counterfactual_data():
    """
    反事实测试数据

    包含潜在结果（potential outcomes）用于验证反事实推断
    """
    np.random.seed(42)
    n = 500

    # 特征
    age = np.random.randint(18, 70, n)
    income = np.random.gamma(20, 5000, n)

    # Treatment：培训项目（受年龄和收入影响）
    prob_T = 1 / (1 + np.exp(-(0.02 * age + 0.00005 * income - 2)))
    T = np.random.binomial(1, prob_T)

    # 潜在结果
    # 培训的因果效应：+5000 收入
    Y0 = 30000 + 500 * age + 2 * income + np.random.randn(n) * 10000
    Y1 = Y0 + 5000 + np.random.randn(n) * 5000

    # 实际结果
    Y = np.where(T == 1, Y1, Y0)

    return {
        'T': T,
        'Y': Y,
        'Y0': Y0,
        'Y1': Y1,
        'age': age,
        'income': income,
        'true_ate': 5000
    }


@pytest.fixture
def confounding_misleading_correlation_data():
    """
    混杂导致误导性相关的数据

    经典例子：冰淇淋销量和溺水人数的相关（由温度导致）
    """
    np.random.seed(42)
    n = 365

    # 混杂变量：温度
    temperature = np.random.normal(25, 10, n)

    # 变量1：冰淇淋销量（受温度影响）
    ice_cream_sales = 100 + 5 * temperature + np.random.randn(n) * 20

    # 变量2：溺水人数（受温度影响）
    drowning_deaths = np.maximum(0, 2 + 0.1 * temperature + np.random.randn(n) * 2)

    # 冰淇淋销量和溺水人数高度相关，但没有因果关系
    correlation = np.corrcoef(ice_cream_sales, drowning_deaths)[0, 1]

    return {
        'ice_cream_sales': ice_cream_sales,
        'drowning_deaths': drowning_deaths,
        'temperature': temperature,
        'misleading_correlation': correlation,  # 高相关但无因果
        'true_confounder': 'temperature'
    }


# =============================================================================
# DAG 基础测试 Fixtures
# =============================================================================

@pytest.fixture
def confounder_dag():
    """
    混杂结构的 DAG

    U -> X, U -> Y, X -> Y
    """
    return {
        'nodes': ['U', 'X', 'Y'],
        'edges': [('U', 'X'), ('U', 'Y'), ('X', 'Y')],
        'structure_type': 'confounder',
        'should_control': ['U'],  # 需要控制混杂
        'paths': {
            'backdoor': ['X <- U -> Y'],
            'frontdoor': ['X -> Y']
        }
    }


@pytest.fixture
def collider_dag():
    """
    碰撞结构的 DAG

    X -> Z, Y -> Z
    """
    return {
        'nodes': ['X', 'Y', 'Z'],
        'edges': [('X', 'Z'), ('Y', 'Z')],
        'structure_type': 'collider',
        'should_not_control': ['Z'],  # 不要控制碰撞变量
        'paths': {
            'open': None,  # X 和 Y 之间没有路径
            'conditioned': ['X -> Z <- Y']  # 控制 Z 后产生虚假相关
        }
    }


@pytest.fixture
def mediator_dag():
    """
    中介结构的 DAG

    X -> M -> Y
    """
    return {
        'nodes': ['X', 'M', 'Y'],
        'edges': [('X', 'M'), ('M', 'Y')],
        'structure_type': 'mediator',
        'control_for_total_effect': [],  # 估计总效应时不控制
        'control_for_direct_effect': ['M'],  # 估计直接效应时控制
        'paths': {
            'total': ['X -> M -> Y'],
            'direct': ['X -> Y'],  # 如果存在直接路径
            'indirect': ['X -> M -> Y']
        }
    }


@pytest.fixture
def complex_causal_dag():
    """
    复杂因果图示例

    场景：优惠券对流失率的影响
    """
    return {
        'nodes': [
            'high_value_customer',  # 高价值客户（混杂，可能不可观测）
            'coupon',               # 优惠券
            'purchase_count',       # 购买次数（中介）
            'customer_satisfaction', # 客户满意（碰撞变量）
            'vip_status',           # VIP 身份
            'churn'                 # 流失
        ],
        'edges': [
            ('high_value_customer', 'coupon'),
            ('high_value_customer', 'churn'),
            ('high_value_customer', 'vip_status'),
            ('coupon', 'purchase_count'),
            ('purchase_count', 'churn'),
            ('vip_status', 'churn'),
            ('coupon', 'customer_satisfaction'),  # 优惠券 -> 满意
            ('churn', 'customer_satisfaction'),   # 流失 -> 满意（反向，简化处理）
        ],
        'confounders': ['high_value_customer'],
        'colliders': ['customer_satisfaction'],
        'mediators': ['purchase_count'],
        'backdoor_paths': [
            'coupon <- high_value_customer -> churn'
        ],
        'adjustment_set': ['high_value_customer']  # 但可能不可观测
    }


@pytest.fixture
def simulated_dag_data():
    """
    模拟 DAG 数据用于测试

    包含：混杂、碰撞、中介三种结构
    """
    np.random.seed(42)
    n = 1000

    # 混杂变量 U
    U = np.random.randn(n)

    # Treatment X（受 U 影响）
    X = 0.5 * U + np.random.randn(n) * 0.5

    # 中介 M（受 X 影响）
    M = 0.7 * X + np.random.randn(n) * 0.3

    # 碰撞变量 C（受 X 和 Y 影响）
    # 先生成 Y
    Y = 0.3 * X + 0.4 * U + 0.5 * M + np.random.randn(n) * 0.3

    # 碰撞变量
    C = 0.6 * X - 0.6 * Y + np.random.randn(n) * 0.2

    return {
        'U': U,
        'X': X,
        'Y': Y,
        'M': M,
        'C': C,
        'true_direct_effect': 0.3,
        'true_indirect_effect': 0.7 * 0.5,  # X -> M -> Y
        'true_total_effect': 0.3 + 0.7 * 0.5
    }


# =============================================================================
# d-分离与后门准则测试 Fixtures
# =============================================================================

@pytest.fixture
def d_separation_examples():
    """
    d-分离示例集
    """
    return {
        'chain': {
            'dag': {'edges': [('A', 'B'), ('B', 'C')]},
            'd_separated': {  # 无条件时
                ('A', 'C'): False,  # A -> B -> C 路径开放
            },
            'conditioned': {  # 控制 B 后
                ('A', 'C'): True,  # 路径被阻断
            }
        },
        'confounder': {
            'dag': {'edges': [('A', 'B'), ('A', 'C')]},
            'd_separated': {
                ('B', 'C'): False,  # B <- A -> C 路径开放
            },
            'conditioned': {  # 控制 A 后
                ('B', 'C'): True,  # 路径被阻断
            }
        },
        'collider': {
            'dag': {'edges': [('A', 'B'), ('C', 'B')]},
            'd_separated': {
                ('A', 'C'): True,  # A 和 C 独立
            },
            'conditioned': {  # 控制 B 后
                ('A', 'C'): False,  # 产生虚假相关
            }
        }
    }


@pytest.fixture
def backdoor_criterion_examples():
    """
    后门准则示例
    """
    return {
        'simple_confounder': {
            'treatment': 'X',
            'outcome': 'Y',
            'edges': [('U', 'X'), ('U', 'Y'), ('X', 'Y')],
            'backdoor_paths': [['X', 'U', 'Y']],
            'adjustment_set': ['U'],
            'valid_sets': [['U']]
        },
        'multiple_confounders': {
            'treatment': 'X',
            'outcome': 'Y',
            'edges': [('U1', 'X'), ('U1', 'Y'), ('U2', 'X'), ('U2', 'Y'), ('X', 'Y')],
            'backdoor_paths': [['X', 'U1', 'Y'], ['X', 'U2', 'Y']],
            'adjustment_set': ['U1', 'U2'],
            'valid_sets': [['U1', 'U2']]
        },
        'mediator_present': {
            'treatment': 'X',
            'outcome': 'Y',
            'edges': [('U', 'X'), ('U', 'Y'), ('X', 'M'), ('M', 'Y')],
            'backdoor_paths': [['X', 'U', 'Y']],
            'adjustment_set': ['U'],
            'should_not_control': ['M'],  # 不要控制中介（除非估计直接效应）
        },
        'no_backdoor': {
            'treatment': 'X',
            'outcome': 'Y',
            'edges': [('X', 'Y')],
            'backdoor_paths': [],
            'adjustment_set': [],  # 无需控制
        }
    }


# =============================================================================
# RCT 模拟测试 Fixtures
# =============================================================================

@pytest.fixture
def rct_data():
    """
    随机对照试验数据

    随机化确保 Treatment 和 Outcome 之间无混杂
    """
    np.random.seed(42)
    n = 500

    # 随机分配 Treatment
    T = np.random.binomial(1, 0.5, n)

    # 基线特征（在两组间平衡）
    age = np.random.randint(18, 70, n)
    income = np.random.gamma(20, 5000, n)

    # 潜在结果
    # Treatment 的因果效应：+3000
    Y0 = 40000 + 500 * age + 0.5 * income + np.random.randn(n) * 8000
    Y1 = Y0 + 3000

    # 实际结果
    Y = np.where(T == 1, Y1, Y0)

    # 验证随机化：基线特征在两组间应该平衡
    balance_age = stats.ttest_ind(age[T == 0], age[T == 1]).pvalue
    balance_income = stats.ttest_ind(income[T == 0], income[T == 1]).pvalue

    return {
        'T': T,
        'Y': Y,
        'Y0': Y0,
        'Y1': Y1,
        'age': age,
        'income': income,
        'true_ate': 3000,
        'is_randomized': balance_age > 0.05 and balance_income > 0.05,
        'balance_p_values': {
            'age': balance_age,
            'income': balance_income
        }
    }


@pytest.fixture
def failed_randomization_data():
    """
    随机化失败的数据

    Treatment 和基线特征相关
    """
    np.random.seed(42)
    n = 500

    # Treatment 不是随机分配的
    # 年龄大的人更可能进入实验组
    age = np.random.randint(18, 70, n)
    prob_T = 1 / (1 + np.exp(-(0.05 * age - 2)))
    T = np.random.binomial(1, prob_T)

    income = np.random.gamma(20, 5000, n)

    # 潜在结果
    Y0 = 40000 + 500 * age + 0.5 * income + np.random.randn(n) * 8000
    Y1 = Y0 + 3000
    Y = np.where(T == 1, Y1, Y0)

    return {
        'T': T,
        'Y': Y,
        'age': age,
        'income': income,
        'true_ate': 3000,
        'is_randomized': False,  # 随机化失败
        'confounded': True
    }


# =============================================================================
# 观察研究测试 Fixtures
# =============================================================================

@pytest.fixture
def did_data():
    """
    双重差分（DID）数据

    场景：某城市试点政策，评估政策效果
    """
    np.random.seed(42)
    n_treated = 50
    n_control = 50
    n_pre = 5
    n_post = 5

    # 处理组（试点城市）
    treated_pre = np.random.normal(100, 10, (n_treated, n_pre))
    # 政策效应：+20
    treated_post = np.random.normal(120, 10, (n_treated, n_post))

    # 对照组（非试点城市）
    control_pre = np.random.normal(95, 10, (n_control, n_pre))
    # 平行趋势：对照组也增长
    control_post = np.random.normal(105, 10, (n_control, n_post))

    return {
        'treated_pre': treated_pre,
        'treated_post': treated_post,
        'control_pre': control_pre,
        'control_post': control_post,
        'true_effect': 15,  # DID 应该识别的效应
        'parallel_trend_holds': True
    }


@pytest.fixture
def did_violated_parallel_trend_data():
    """
    平行趋势假设被违反的 DID 数据

    处理组和对照组在政策前趋势不同
    """
    np.random.seed(42)
    n = 10  # 每组样本数

    # 处理组：趋势向上（5 期 -> 5 个观测）
    treated_pre = np.array([80, 85, 90, 95, 100]) + np.random.randn(5) * 3
    treated_post = np.array([105, 110, 115, 120, 125]) + np.random.randn(5) * 3

    # 对照组：趋势向下（5 期 -> 5 个观测）
    control_pre = np.array([100, 98, 96, 94, 92]) + np.random.randn(5) * 3
    control_post = np.array([90, 88, 86, 84, 82]) + np.random.randn(5) * 3

    return {
        'treated_pre': treated_pre,
        'treated_post': treated_post,
        'control_pre': control_pre,
        'control_post': control_post,
        'parallel_trend_holds': False
    }


@pytest.fixture
def instrumental_variable_data():
    """
    工具变量（IV）数据

    场景：教育（X）对收入（Y）的因果效应
    工具变量（Z）：义务教育法变化
    未观测混杂（U）：能力
    """
    np.random.seed(42)
    n = 1000

    # 未观测混杂：能力
    U = np.random.randn(n)

    # 工具变量：义务教育法变化
    # 假设：法律变化只通过教育影响收入
    Z = np.random.binomial(1, 0.5, n)

    # Treatment：教育年限（受 Z 和 U 影响）
    # Z 每增加 1 单位，教育增加约 2 年
    X = 12 + 2 * Z + 0.5 * U + np.random.randn(n) * 1

    # Outcome：收入（受 X 和 U 影响）
    # 教育的因果效应：每增加 1 年，收入增加 2000
    Y = 20000 + 2000 * X + 5000 * U + np.random.randn(n) * 10000

    return {
        'Z': Z,
        'X': X,
        'Y': Y,
        'U': U,  # 未观测混杂
        'true_effect': 2000,  # X 对 Y 的真实因果效应
        'instrument_valid': True,
        'instrument_relevance': np.corrcoef(Z, X)[0, 1],  # Z 和 X 相关
        'instrument_exogeneity': np.corrcoef(Z, U)[0, 1]  # Z 和 U 不相关（应该接近0）
    }


@pytest.fixture
def psm_data():
    """
    倾向得分匹配（PSM）数据

    处理组和对照组特征分布不同
    """
    np.random.seed(42)
    n = 1000

    # 特征
    age = np.random.randint(18, 70, n)
    income = np.random.gamma(20, 5000, n)
    education = np.random.randint(9, 20, n)

    # Treatment：培训项目
    # 年轻、高收入、高教育的人更可能参加
    prob_T = 1 / (1 + np.exp(-(-5 + 0.03 * age + 0.0001 * income + 0.2 * education)))
    T = np.random.binomial(1, prob_T)

    # Outcome：收入
    # 培训的因果效应：+5000
    Y0 = 30000 + 300 * age + 2 * income + 1000 * education + np.random.randn(n) * 10000
    Y1 = Y0 + 5000
    Y = np.where(T == 1, Y1, Y0)

    # 合并协变量
    covariates = pd.DataFrame({
        'age': age,
        'income': income,
        'education': education
    })

    return {
        'T': T,
        'Y': Y,
        'Y0': Y0,
        'Y1': Y1,
        'covariates': covariates,
        'true_ate': 5000,
        'has_selection_bias': True  # 存在选择偏差
    }


@pytest.fixture
def no_overlap_data():
    """
    无重叠区域的数据

    处理组和对照组的特征分布完全不重叠
    PSM 在此情况下失效
    """
    np.random.seed(42)
    n = 500

    # 处理组：高收入
    income_treated = np.random.gamma(30, 10000, n // 2)
    T_treated = np.ones(n // 2)

    # 对照组：低收入
    income_control = np.random.gamma(10, 2000, n // 2)
    T_control = np.zeros(n // 2)

    income = np.concatenate([income_treated, income_control])
    T = np.concatenate([T_treated, T_control])

    # Outcome
    Y = 30000 + 2 * income + T * 5000 + np.random.randn(n) * 5000

    return {
        'T': T,
        'Y': Y,
        'income': income,
        'has_overlap': False,  # 无重叠区域
        'psm_fails': True
    }


# =============================================================================
# 边界情况 Fixtures
# =============================================================================

@pytest.fixture
def minimal_causal_data():
    """
    最小因果推断数据
    """
    n = 20
    X = np.random.binomial(1, 0.5, n)
    Y = np.random.binomial(1, 0.3, n)
    U = np.random.binomial(1, 0.4, n)

    return {
        'X': X,
        'Y': Y,
        'U': U,
        'n': n
    }


@pytest.fixture
def all_treated_data():
    """
    全部处理的数据

    无法估计因果效应（缺少对照组）
    """
    n = 100
    X = np.ones(n, dtype=int)
    Y = np.random.binomial(1, 0.2, n)

    return {
        'X': X,
        'Y': Y,
        'no_control_group': True
    }


@pytest.fixture
def all_control_data():
    """
    全部对照的数据

    无法估计因果效应（缺少处理组）
    """
    n = 100
    X = np.zeros(n, dtype=int)
    Y = np.random.binomial(1, 0.2, n)

    return {
        'X': X,
        'Y': Y,
        'no_treatment_group': True
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
        'effect_tol': 0.1,      # 因果效应容差（10%）
        'correlation_tol': 0.05, # 相关性容差
        'balance_tol': 0.05,    # 平衡性检验容差
    }


@pytest.fixture
def causal_vocab():
    """
    因果推断术语字典
    """
    return {
        'association': '关联 - P(Y|X)',
        'intervention': '干预 - P(Y|do(X))',
        'counterfactual': '反事实 - P(Y_x|X\',Y\')',
        'confounder': '混杂变量 - 同时影响 Treatment 和 Outcome',
        'collider': '碰撞变量 - 受 Treatment 和 Outcome 共同影响',
        'mediator': '中介变量 - Treatment 通过它影响 Outcome',
        'backdoor_path': '后门路径 - Treatment <- ... -> Outcome',
        'frontdoor_path': '前门路径 - Treatment -> ... -> Outcome',
        'd_separation': 'd-分离 - 判断变量间独立性',
        'rct': '随机对照试验 - 因果推断金标准',
        'did': '双重差分 - Difference-in-Differences',
        'iv': '工具变量 - Instrumental Variable',
        'psm': '倾向得分匹配 - Propensity Score Matching',
        'ate': '平均处理效应 - Average Treatment Effect',
    }

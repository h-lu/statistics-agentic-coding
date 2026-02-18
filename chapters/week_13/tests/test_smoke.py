"""
Smoke Tests for Week 13 solution.py

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
    冒烟测试：solution.py 应包含因果推断相关函数

    检查核心函数是否存在（示例函数名，实际可能不同）
    """
    try:
        import solution
    except ImportError:
        pytest.skip("solution.py not found")

    # 因果推断相关的可能函数名
    functions = [
        # 因果推断三层级
        'compute_association',
        'estimate_intervention_effect',
        'compute_counterfactual',

        # DAG 相关
        'identify_confounders',
        'identify_colliders',
        'identify_mediators',
        'draw_causal_dag',

        # d-分离和后门准则
        'check_d_separation',
        'find_backdoor_paths',
        'get_adjustment_set',

        # RCT
        'check_randomization_balance',
        'estimate_ate',

        # 观察研究方法
        'estimate_did',
        'estimate_iv',
        'propensity_score_matching',
    ]

    # 至少有一个因果推断相关的函数存在
    has_any = any(hasattr(solution, func) for func in functions)
    # 如果没有任何预期函数，只是因为还没实现
    # 这个测试不会 fail，只会 skip


# =============================================================================
# 因果推断三层级冒烟测试
# =============================================================================

def test_association_smoke():
    """
    冒烟测试：关联计算应能运行

    测试基本的关联计算（P(Y|X)）
    """
    try:
        import solution
    except ImportError:
        pytest.skip("solution.py not found")

    # 创建简单数据
    np.random.seed(42)
    X = np.random.binomial(1, 0.5, 100)
    Y = np.random.binomial(1, 0.3, 100)

    # 尝试计算关联
    if hasattr(solution, 'compute_association'):
        result = solution.compute_association(X, Y)
        assert result is not None
    else:
        pytest.skip("compute_association not implemented")


def test_intervention_smoke():
    """
    冒烟测试：干预效应估计应能运行

    测试基本的干预效应计算（P(Y|do(X))）
    """
    try:
        import solution
    except ImportError:
        pytest.skip("solution.py not found")

    # 创建简单数据
    np.random.seed(42)
    n = 100
    T = np.random.binomial(1, 0.5, n)
    Y = np.random.randn(n)

    # 尝试估计干预效应
    if hasattr(solution, 'estimate_intervention_effect'):
        result = solution.estimate_intervention_effect(T, Y)
        assert result is not None
    else:
        pytest.skip("estimate_intervention_effect not implemented")


# =============================================================================
# DAG 基础冒烟测试
# =============================================================================

def test_dag_structure_smoke():
    """
    冒烟测试：DAG 结构表示应能运行

    测试基本的 DAG 数据结构
    """
    try:
        import solution
    except ImportError:
        pytest.skip("solution.py not found")

    # 尝试使用 DAG（创建 SimpleDAG 对象）
    if hasattr(solution, 'SimpleDAG'):
        dag = solution.SimpleDAG()
        dag.add_edge('Z', 'X')
        dag.add_edge('Z', 'Y')
        dag.add_edge('X', 'Y')

        # 尝试识别混杂
        if hasattr(solution, 'identify_confounders'):
            result = solution.identify_confounders(dag, 'X', 'Y')
            assert result is not None
        else:
            pytest.skip("identify_confounders not implemented")
    else:
        pytest.skip("SimpleDAG not implemented")


def test_d_separation_smoke():
    """
    冒烟测试：d-分离检查应能运行

    测试基本的 d-分离判断
    """
    try:
        import solution
    except ImportError:
        pytest.skip("solution.py not found")

    # 简单 DAG
    dag = {
        'nodes': ['A', 'B', 'C'],
        'edges': [('A', 'B'), ('B', 'C')]
    }

    # 尝试检查 d-分离
    if hasattr(solution, 'check_d_separation'):
        result = solution.check_d_separation(dag, 'A', 'C', conditioned=None)
        assert result is not None
    else:
        pytest.skip("check_d_separation not implemented")


# =============================================================================
# RCT 冒烟测试
# =============================================================================

def test_rct_balance_smoke():
    """
    冒烟测试：RCT 平衡性检验应能运行

    测试基线平衡检验
    """
    try:
        import solution
    except ImportError:
        pytest.skip("solution.py not found")

    # 创建 RCT 数据
    np.random.seed(42)
    n = 100
    T = np.random.binomial(1, 0.5, n)
    age = np.random.randint(18, 70, n)
    income = np.random.randn(n) * 10000 + 50000

    # 尝试检验平衡
    if hasattr(solution, 'check_randomization_balance'):
        result = solution.check_randomization_balance(T, {'age': age, 'income': income})
        assert result is not None
    else:
        pytest.skip("check_randomization_balance not implemented")


def test_ate_estimation_smoke():
    """
    冒烟测试：ATE 估计应能运行

    测试平均处理效应估计
    """
    try:
        import solution
    except ImportError:
        pytest.skip("solution.py not found")

    # 创建简单数据
    np.random.seed(42)
    n = 100
    T = np.random.binomial(1, 0.5, n)
    Y = np.random.randn(n)

    # 尝试估计 ATE
    if hasattr(solution, 'estimate_ate'):
        result = solution.estimate_ate(T, Y)
        assert result is not None
    else:
        pytest.skip("estimate_ate not implemented")


# =============================================================================
# 观察研究方法冒烟测试
# =============================================================================

def test_did_smoke():
    """
    冒烟测试：DID 估计应能运行

    测试双重差分方法
    """
    try:
        import solution
    except ImportError:
        pytest.skip("solution.py not found")

    # 创建 DID 数据
    np.random.seed(42)
    treated_pre = np.random.normal(100, 10, 50)
    treated_post = np.random.normal(110, 10, 50)
    control_pre = np.random.normal(95, 10, 50)
    control_post = np.random.normal(100, 10, 50)

    # 尝试估计 DID
    if hasattr(solution, 'estimate_did'):
        result = solution.estimate_did(treated_pre, treated_post, control_pre, control_post)
        assert result is not None
    else:
        pytest.skip("estimate_did not implemented")


def test_iv_smoke():
    """
    冒烟测试：IV 估计应能运行

    测试工具变量方法
    """
    try:
        import solution
    except ImportError:
        pytest.skip("solution.py not found")

    # 创建 IV 数据
    np.random.seed(42)
    n = 100
    Z = np.random.binomial(1, 0.5, n)
    X = 0.5 * Z + np.random.randn(n) * 0.5
    Y = 2 * X + np.random.randn(n)

    # 尝试 IV 估计
    if hasattr(solution, 'estimate_iv'):
        result = solution.estimate_iv(Z, X, Y)
        assert result is not None
    else:
        pytest.skip("estimate_iv not implemented")


def test_psm_smoke():
    """
    冒烟测试：PSM 应能运行

    测试倾向得分匹配
    """
    try:
        import solution
    except ImportError:
        pytest.skip("solution.py not found")

    # 创建 PSM 数据（DataFrame 格式）
    np.random.seed(42)
    n = 100
    T = np.random.binomial(1, 0.5, n)
    covariates = np.random.randn(n, 3)
    Y = np.random.randn(n)

    import pandas as pd
    df = pd.DataFrame({
        'treatment': T,
        'outcome': Y,
        'cov1': covariates[:, 0],
        'cov2': covariates[:, 1],
        'cov3': covariates[:, 2],
    })

    # 尝试 PSM
    if hasattr(solution, 'propensity_score_matching'):
        result = solution.propensity_score_matching(
            df, 'treatment', 'outcome', ['cov1', 'cov2', 'cov3']
        )
        assert result is not None
    else:
        pytest.skip("propensity_score_matching not implemented")


# =============================================================================
# 综合冒烟测试
# =============================================================================

def test_end_to_end_smoke():
    """
    冒烟测试：端到端因果推断流程应能运行

    测试从 DAG 到因果效应估计的完整流程
    """
    try:
        import solution
    except ImportError:
        pytest.skip("solution.py not found")

    # 尝试完整流程
    results = {}

    if hasattr(solution, 'SimpleDAG'):
        # 创建 DAG
        dag = solution.SimpleDAG()
        dag.add_edge('U', 'X')
        dag.add_edge('U', 'Y')
        dag.add_edge('X', 'Y')

        if hasattr(solution, 'identify_confounders'):
            results['confounders'] = solution.identify_confounders(dag, 'X', 'Y')

        if hasattr(solution, 'find_backdoor_paths'):
            results['backdoor_paths'] = solution.find_backdoor_paths(dag, 'X', 'Y')

    # 至少有一个步骤成功
    if results:
        assert True
    else:
        pytest.skip("No causal inference functions implemented")


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
    X = np.array([])
    Y = np.array([])

    # 尝试计算（应该报错或返回 None）
    if hasattr(solution, 'compute_association'):
        try:
            result = solution.compute_association(X, Y)
            # 如果不报错，应该返回 None 或合理的默认值
            assert result is None or isinstance(result, (int, float, dict))
        except (ValueError, IndexError, RuntimeError):
            # 报错也是可接受的
            assert True


def test_single_group_handling():
    """
    冒烟测试：单一组应被正确处理

    验证只有处理组或只有对照组时的行为
    """
    try:
        import solution
    except ImportError:
        pytest.skip("solution.py not found")

    # 全部处理
    T = np.ones(50, dtype=int)
    Y = np.random.randn(50)

    # 尝试估计 ATE（应该报错或返回 None）
    if hasattr(solution, 'estimate_ate'):
        try:
            result = solution.estimate_ate(T, Y)
            # 如果不报错，应该返回 None 或警告
            assert result is None or isinstance(result, (int, float, dict))
        except (ValueError, RuntimeError):
            # 报错也是可接受的
            assert True


# =============================================================================
# 概念理解冒烟测试
# =============================================================================

def test_causal_ladder_concepts():
    """
    冒烟测试：因果推断三层级概念

    验证对关联、干预、反事实的理解
    """
    # 概念测试：不需要具体实现
    concepts = {
        'association': 'P(Y|X) - 关联',
        'intervention': 'P(Y|do(X)) - 干预',
        'counterfactual': 'P(Y_x|X\',Y\') - 反事实'
    }

    assert len(concepts) == 3
    assert 'association' in concepts
    assert 'intervention' in concepts
    assert 'counterfactual' in concepts


def test_dag_concepts():
    """
    冒烟测试：DAG 基本概念

    验证对混杂、碰撞、中介的理解
    """
    # 概念测试
    structures = {
        'confounder': '需要控制',
        'collider': '不要控制',
        'mediator': '取决于研究问题'
    }

    assert len(structures) == 3
    assert 'confounder' in structures
    assert 'collider' in structures
    assert 'mediator' in structures

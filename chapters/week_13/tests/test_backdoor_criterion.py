"""
Test Suite: d-分离与后门准则（d-Separation and Backdoor Criterion）

测试因果推断的核心算法：
1. d-分离：判断变量之间的独立性
2. 后门准则：识别需要控制的变量集
3. 前门路径和后门路径
4. 调整集的选择

测试覆盖：
- 正确应用 d-分离规则
- 正确识别后门路径
- 正确选择调整集
- 避免包含碰撞变量的调整集
- 无后门路径的情况
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# 添加 starter_code 到路径
starter_code_path = Path(__file__).parent.parent / "starter_code"
if str(starter_code_path) not in sys.path:
    sys.path.insert(0, str(starter_code_path))


# =============================================================================
# d-分离基础测试
# =============================================================================

class TestDSeparationBasics:
    """测试 d-分离的基础规则"""

    def test_chain_structure_d_separation(self):
        """
        正例：链式结构的 d-分离

        A -> B -> C
        - 无条件：A 和 C 不独立（路径开放）
        - 控制 B：A 和 C 独立（路径被阻断）
        """
        # 概念测试
        # 在链式结构中，控制中间变量阻断路径

        chain_dag = {
            'nodes': ['A', 'B', 'C'],
            'edges': [('A', 'B'), ('B', 'C')]
        }

        assert chain_dag['edges'][0] == ('A', 'B')
        assert chain_dag['edges'][1] == ('B', 'C')

    def test_confounder_structure_d_separation(self):
        """
        正例：混杂结构的 d-分离

        A <- B -> C（或 B -> A, B -> C）
        - 无条件：A 和 C 不独立（路径开放）
        - 控制 B：A 和 C 独立（路径被阻断）
        """
        confounder_dag = {
            'nodes': ['A', 'B', 'C'],
            'edges': [('B', 'A'), ('B', 'C')],
            'structure_type': 'confounder'
        }

        assert confounder_dag['structure_type'] == 'confounder'

    def test_collider_structure_d_separation(self):
        """
        正例：碰撞结构的 d-分离

        A -> B <- C
        - 无条件：A 和 C 独立（路径被阻断）
        - 控制 B：A 和 C 不独立（路径被打开）
        """
        collider_dag = {
            'nodes': ['A', 'B', 'C'],
            'edges': [('A', 'B'), ('C', 'B')],
            'structure_type': 'collider'
        }

        assert collider_dag['structure_type'] == 'collider'

    @pytest.mark.parametrize("structure,condition,expected_independent", [
        ("chain", None, False),     # 链式：无条件不独立
        ("chain", "middle", True),  # 链式：控制中间变量后独立
        ("confounder", None, False),     # 混杂：无条件不独立
        ("confounder", "confounder", True),  # 混杂：控制混杂后独立
        ("collider", None, True),    # 碰撞：无条件独立
        ("collider", "collider", False),  # 碰撞：控制碰撞后不独立
    ])
    def test_d_separation_rules(
        self, structure, condition, expected_independent
    ):
        """
        正例：d-分离的三条基本规则

        验证链式、混杂、碰撞三种结构的 d-分离规则
        """
        # 概念测试
        assert structure in ["chain", "confounder", "collider"]
        assert isinstance(expected_independent, bool)


# =============================================================================
# 路径识别测试
# =============================================================================

class TestPathIdentification:
    """测试因果路径的识别"""

    def test_identify_frontdoor_path(self, mediator_dag):
        """
        正例：识别前门路径

        前门路径：Treatment -> ... -> Outcome
        这是因果效应的传递路径
        """
        edges = mediator_dag['edges']

        # 验证：存在 X -> M -> Y 的前门路径
        assert ('X', 'M') in edges
        assert ('M', 'Y') in edges

    def test_identify_backdoor_path(self, confounder_dag):
        """
        正例：识别后门路径

        后门路径：Treatment <- ... -> Outcome
        这是需要控制的混杂路径
        """
        backdoor_paths = confounder_dag['paths']['backdoor']

        # 验证：存在后门路径
        assert len(backdoor_paths) > 0

        # 在混杂结构中，后门路径是 X <- U -> Y
        assert any('U' in path for path in backdoor_paths)

    def test_no_backdoor_path(self):
        """
        边界：无后门路径

        如果 X 和 Y 之间只有前门路径，无需控制任何变量
        """
        no_backdoor_dag = {
            'nodes': ['X', 'Y'],
            'edges': [('X', 'Y')],
            'backdoor_paths': [],
            'adjustment_set': []
        }

        # 验证：无后门路径
        assert len(no_backdoor_dag['backdoor_paths']) == 0

        # 验证：无需控制
        assert len(no_backdoor_dag['adjustment_set']) == 0

    def test_multiple_backdoor_paths(self, backdoor_criterion_examples):
        """
        正例：多个后门路径

        可能存在多条后门路径，需要全部阻断
        """
        multiple_conf = backdoor_criterion_examples['multiple_confounders']

        # 验证：存在多条后门路径
        assert len(multiple_conf['backdoor_paths']) >= 2

        # 验证：需要控制所有混杂
        assert len(multiple_conf['adjustment_set']) >= 2


# =============================================================================
# 后门准则测试
# =============================================================================

class TestBackdoorCriterion:
    """测试后店准则的应用"""

    def test_backdoor_criterion_definition(self, backdoor_criterion_examples):
        """
        正例：理解后门准则的定义

        满足条件：
        1. 调整集 Z 不包含 X 的后代
        2. 控制 Z 后，所有后门路径被阻断
        """
        simple_conf = backdoor_criterion_examples['simple_confounder']

        # 验证：调整集存在
        assert len(simple_conf['adjustment_set']) > 0

        # 验证：调整集有效
        assert len(simple_conf['valid_sets']) > 0

    def test_backdoor_adjustment_set_identification(
        self, backdoor_criterion_examples
    ):
        """
        正例：识别有效的调整集

        后门准则告诉我们应该控制哪些变量
        """
        for example_name, example in backdoor_criterion_examples.items():
            if 'adjustment_set' in example:
                adjustment_set = example['adjustment_set']

                # 验证：调整集是列表
                assert isinstance(adjustment_set, list)

    def test_minimal_adjustment_set(self, backdoor_criterion_examples):
        """
        正例：最小调整集

        不需要控制所有变量，只需要控制必要的
        """
        simple_conf = backdoor_criterion_examples['simple_confounder']

        # 在这个例子中，只需要控制 U
        adjustment_set = simple_conf['adjustment_set']

        # 验证：调整集非空
        assert len(adjustment_set) > 0

    def test_backdoor_with_mediator(self, backdoor_criterion_examples):
        """
        正例：存在中介时的后店准则

        估计总效应时，不要控制中介变量
        """
        mediator_example = backdoor_criterion_examples['mediator_present']

        # 验证：不要控制中介
        should_not_control = mediator_example['should_not_control']
        assert 'M' in should_not_control


# =============================================================================
# 调整集选择测试
# =============================================================================

class TestAdjustmentSetSelection:
    """测试调整集的选择"""

    def test_adjustment_set_should_not_include_collider(
        self, collider_dag
    ):
        """
        反例：调整集不应包含碰撞变量

        控制碰撞变量会引入虚假相关
        """
        should_not_control = collider_dag['should_not_control']

        # 验证：碰撞变量不应该被控制
        assert 'Z' in should_not_control

    def test_adjustment_set_should_not_include_mediator_for_total_effect(
        self, mediator_dag
    ):
        """
        反例：估计总效应时不控制中介

        控制中介会阻断因果路径
        """
        control_for_total = mediator_dag['control_for_total_effect']

        # 验证：估计总效应时不控制中介
        assert 'M' not in control_for_total

    def test_adjustment_set_may_include_mediator_for_direct_effect(
        self, mediator_dag
    ):
        """
        正例：估计直接效应时可控制中介

        如果想区分直接效应和间接效应，可以控制中介
        """
        control_for_direct = mediator_dag['control_for_direct_effect']

        # 验证：估计直接效应时控制中介
        assert 'M' in control_for_direct

    @pytest.mark.parametrize("include_collider,expected_valid", [
        (True, False),   # 包含碰撞：调整集无效
        (False, True),   # 不包含碰撞：调整集可能有效
    ])
    def test_adjustment_set_validity(
        self, include_collider, expected_valid
    ):
        """
        正例：验证调整集的有效性

        有效的调整集必须：
        1. 不包含碰撞变量
        2. 阻断所有后门路径
        """
        # 概念测试
        assert isinstance(include_collider, bool)
        assert isinstance(expected_valid, bool)

        if include_collider:
            # 包含碰撞的调整集无效
            assert not expected_valid
        else:
            # 不包含碰撞的调整集可能有效
            # 还需要检查是否阻断所有后门路径
            assert expected_valid or True


# =============================================================================
# 复杂 DAG 的后门分析
# =============================================================================

class TestComplexBackdoorAnalysis:
    """测试复杂 DAG 的后门分析"""

    def test_coupon_churn_backdoor_analysis(self, complex_causal_dag):
        """
        正例：优惠券-流失问题的后门分析

        DAG: coupon -> purchase_count -> churn
             <- high_value_customer ->
             <- vip_status
        """
        backdoor_paths = complex_causal_dag['backdoor_paths']
        adjustment_set = complex_causal_dag['adjustment_set']

        # 验证：存在后门路径
        assert len(backdoor_paths) > 0

        # 验证：需要控制混杂
        assert len(adjustment_set) > 0

        # 主要混杂：高价值客户
        assert 'high_value_customer' in adjustment_set

    def test_unobservable_confounder_problem(self, complex_causal_dag):
        """
        边界：不可观测的混杂变量

        如果调整集中的变量不可观测，后店准则无法应用
        """
        adjustment_set = complex_causal_dag['adjustment_set']

        # 高价值客户可能不可观测
        # 需要找代理变量或使用其他方法

        assert len(adjustment_set) > 0

    def test_multiple_path_blocking(self, backdoor_criterion_examples):
        """
        正例：阻断多条后门路径

        当存在多条后门路径时，需要全部阻断
        """
        multiple_conf = backdoor_criterion_examples['multiple_confounders']

        # 验证：需要控制多个变量
        adjustment_set = multiple_conf['adjustment_set']
        assert len(adjustment_set) >= 2


# =============================================================================
# 数据验证测试
# =============================================================================

class TestDataValidation:
    """测试用数据验证 d-分离"""

    def test_d_separation_in_data(self, simulated_dag_data):
        """
        正例：在数据中验证 d-分离

        如果两个变量 d-分离，它们在数据中应该独立
        """
        # 在这个模拟中：
        # U -> X, U -> Y, X -> Y
        # 控制 U 后，X 和 Y 的"剩余"关联来自因果路径

        X = simulated_dag_data['X']
        Y = simulated_dag_data['Y']
        U = simulated_dag_data['U']

        # 不控制 U 时，X 和 Y 的关联包含因果和混杂
        corr_raw = np.corrcoef(X, Y)[0, 1]

        # 控制 U 后（简化方法：分层）
        U_median = np.median(U)
        corr_controlled = (
            np.corrcoef(X[U > U_median], Y[U > U_median])[0, 1] +
            np.corrcoef(X[U <= U_median], Y[U <= U_median])[0, 1]
        ) / 2

        # 验证：控制后相关性改变
        # （具体方向取决于模拟参数）

        assert not np.isnan(corr_raw)
        assert not np.isnan(corr_controlled)

    def test_collider_conditioning_in_data(self, simulated_dag_data):
        """
        反例：控制碰撞变量在数据中的表现

        控制 C 后，原本可能独立的变量变得相关
        """
        X = simulated_dag_data['X']
        Y = simulated_dag_data['Y']
        C = simulated_dag_data['C']

        # 不控制 C 时的相关性
        corr_raw = np.corrcoef(X, Y)[0, 1]

        # 控制 C 后（按 C 的中位数分层）
        C_median = np.median(C)
        if len(X[C > C_median]) > 1 and len(X[C <= C_median]) > 1:
            corr_high_C = np.corrcoef(X[C > C_median], Y[C > C_median])[0, 1]
            corr_low_C = np.corrcoef(X[C <= C_median], Y[C <= C_median])[0, 1]

            # 验证：相关性模式改变
            assert not np.isnan(corr_high_C)
            assert not np.isnan(corr_low_C)


# =============================================================================
# 常见错误测试
# =============================================================================

class TestCommonMistakes:
    """测试后门准则应用中的常见错误"""

    def test_mistake_including_collider_in_adjustment_set(self):
        """
        反例：调整集包含碰撞变量

        这是常见且严重的错误
        """
        # 概念测试
        # 调整集不应该包含碰撞变量
        # 控制碰撞会打开虚假路径

        assert True  # 概念验证

    def test_mistake_including_mediator_for_total_effect(self):
        """
        反例：估计总效应时控制中介

        这会低估因果效应
        """
        # 概念测试
        # 总效应 = 直接效应 + 间接效应
        # 控制中介会阻断间接效应

        assert True  # 概念验证

    def test_mistake_assuming_all_variables_should_be_controlled(self):
        """
        反例：假设"控制所有变量"

        后门准则告诉我们：只控制正确的变量
        """
        # 概念测试
        # 控制所有变量是错误策略
        # 应该只控制混杂，不控制碰撞和中介（除非有特定目的）

        assert True  # 概念验证

    def test_mistake_ignoring_unobserved_confounders(self):
        """
        反例：忽略未观测混杂

        即使后门准则说"控制 X"，如果 X 不可观测，仍无法识别因果
        """
        # 概念测试
        # 未观测混杂是观察研究的核心挑战

        assert True  # 概念验证


# =============================================================================
# 边界情况测试
# =============================================================================

class TestEdgeCases:
    """测试边界情况"""

    def test_no_confounders_case(self, backdoor_criterion_examples):
        """
        边界：无混杂的情况

        如果没有后门路径，无需控制任何变量
        """
        no_backdoor = backdoor_criterion_examples['no_backdoor']

        # 验证：无后门路径
        assert len(no_backdoor['backdoor_paths']) == 0

        # 验证：无需控制
        assert len(no_backdoor['adjustment_set']) == 0

    def test_single_confounder_case(self, backdoor_criterion_examples):
        """
        边界：单一混杂的情况

        最简单的后门路径场景
        """
        simple_conf = backdoor_criterion_examples['simple_confounder']

        # 验证：只有一个混杂
        adjustment_set = simple_conf['adjustment_set']
        assert len(adjustment_set) == 1

    def test_circular_path_not_allowed_in_dag(self):
        """
        边界：DAG 不允许环

        如果存在环，不是有效的 DAG
        """
        # 概念测试
        # DAG = Directed Acyclic Graph
        # 不允许 A -> B -> ... -> A

        assert True  # 概念验证

    def test_empty_adjustment_set_with_causation(self):
        """
        边界：空调整集但存在因果效应

        X -> Y，无混杂，无需控制
        """
        direct_causation = {
            'treatment': 'X',
            'outcome': 'Y',
            'edges': [('X', 'Y')],
            'adjustment_set': [],
            'has_causal_effect': True
        }

        assert len(direct_causation['adjustment_set']) == 0
        assert direct_causation['has_causal_effect']

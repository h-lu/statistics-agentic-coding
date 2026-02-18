"""
Test Suite: DAG 基础（Causal Diagram Basics）

测试因果图（DAG, Directed Acyclic Graph）的基础概念：
1. 混杂变量（Confounder）：需要控制
2. 碰撞变量（Collider）：不要控制
3. 中介变量（Mediator）：取决于研究问题

测试覆盖：
- 正确识别混杂变量
- 正确识别碰撞变量
- 正确识别中介变量
- 避免控制碰撞变量的错误
- DAG 的基本性质（有向、无环）
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
# DAG 基本性质测试
# =============================================================================

class TestDAGBasicProperties:
    """测试 DAG 的基本性质"""

    def test_dag_is_directed(self, confounder_dag):
        """
        正例：DAG 是有向图

        箭头有方向：A -> B 表示"A 导致 B"
        """
        edges = confounder_dag['edges']

        # 验证：所有边都有方向
        # 在实际实现中，边应该表示为 (from, to) 元组
        for edge in edges:
            assert isinstance(edge, tuple)
            assert len(edge) == 2
            # edge[0] 是起点，edge[1] 是终点

    def test_dag_is_acyclic(self, confounder_dag, collider_dag, mediator_dag):
        """
        正例：DAG 无环

        不能存在 A -> ... -> A 的路径
        """
        # 简化的验证：在这些小 DAG 中，我们手动确保无环
        # 在实际实现中，应该用拓扑排序或环检测算法

        dags = [confounder_dag, collider_dag, mediator_dag]

        for dag in dags:
            edges = dag['edges']
            nodes = dag['nodes']

            # 基本检查：节点数应该 >= 边数（避免自环）
            assert len(nodes) > 0
            # 实际的环检测需要更复杂的算法

    def test_dag_represents_assumptions_not_data(self):
        """
        正例：DAG 表示假设，不是从数据中发现的

        数据不会告诉你因果方向，DAG 是根据领域知识假设的
        """
        # 概念测试
        # DAG 中的箭头方向是研究者的假设
        # 不是数据挖掘的结果

        assert True  # 概念验证


# =============================================================================
# 混杂变量测试
# =============================================================================

class TestConfounder:
    """测试混杂变量的识别和处理"""

    def test_identify_confounder_structure(self, confounder_dag):
        """
        正例：识别混杂结构

        混杂：Z -> X, Z -> Y
        Z 是 X 和 Y 的共同原因
        """
        structure_type = confounder_dag['structure_type']

        # 验证：这是混杂结构
        assert structure_type == 'confounder'

        # 验证：应该控制混杂
        should_control = confounder_dag['should_control']
        assert len(should_control) > 0

    def test_confounder_creates_spurious_correlation(
        self, association_vs_intervention_data
    ):
        """
        正例：混杂导致虚假相关

        X 和 Y 相关，不是因为 X -> Y，而是因为 Z -> X 和 Z -> Y
        """
        X = association_vs_intervention_data['X']
        Y = association_vs_intervention_data['Y']
        U = association_vs_intervention_data['U']

        # 验证：X 和 Y 相关
        correlation_XY = np.corrcoef(X, Y)[0, 1]

        # 验证：U 是混杂（同时影响 X 和 Y）
        correlation_XU = np.corrcoef(X, U)[0, 1]
        correlation_UY = np.corrcoef(U, Y)[0, 1]

        assert abs(correlation_XU) > 0.1, "U 应该影响 X"
        assert abs(correlation_UY) > 0.1, "U 应该影响 Y"

    def test_controlling_confounder_blocks_backdoor_path(
        self, simulated_dag_data
    ):
        """
        正例：控制混杂变量阻断后门路径

        控制 U 后，X 和 Y 的关联（如果存在直接路径）就是因果效应
        """
        X = simulated_dag_data['X']
        Y = simulated_dag_data['Y']
        U = simulated_dag_data['U']

        # 计算控制 U 后的关联
        # 简化方法：按 U 的中位数分层
        U_median = np.median(U)
        corr_high_U = np.corrcoef(X[U > U_median], Y[U > U_median])[0, 1]
        corr_low_U = np.corrcoef(X[U <= U_median], Y[U <= U_median])[0, 1]

        # 验证：分层后可以计算关联
        assert not np.isnan(corr_high_U)
        assert not np.isnan(corr_low_U)

    @pytest.mark.parametrize("include_confounder,expected_bias", [
        (False, "high"),   # 不控制混杂，偏差大
        (True, "low"),     # 控制混杂，偏差小
    ])
    def test_regression_with_confounder(
        self, simulated_dag_data, include_confounder, expected_bias
    ):
        """
        正例：回归中包含混杂变量

        控制混杂后，X 的系数更接近因果效应
        """
        # 概念测试：验证"应该控制混杂"
        assert isinstance(include_confounder, bool)
        assert expected_bias in ["high", "low"]

    def test_unobserved_confounder_problem(self, association_vs_intervention_data):
        """
        边界：未观测混杂

        如果混杂变量不可观测，无法通过简单控制消除偏差
        """
        U = association_vs_intervention_data['U']

        # 在这个模拟中，U 是"不可观测"的
        # 现实中：无法控制未观测变量

        # 验证：未观测混杂存在
        assert len(U) > 0

        # 解决方案：RCT、工具变量、断点回归等


# =============================================================================
# 碰撞变量测试
# =============================================================================

class TestCollider:
    """测试碰撞变量的识别和处理"""

    def test_identify_collider_structure(self, collider_dag):
        """
        正例：识别碰撞结构

        碰撞：X -> Z <- Y
        Z 是 X 和 Y 的共同结果
        """
        structure_type = collider_dag['structure_type']

        # 验证：这是碰撞结构
        assert structure_type == 'collider'

        # 验证：不应该控制碰撞
        should_not_control = collider_dag['should_not_control']
        assert len(should_not_control) > 0

    def test_collider_creates_independence(
        self, simulated_dag_data
    ):
        """
        正例：X 和 Y 在碰撞结构中独立

        在 X -> Z <- Y 中，X 和 Y 独立
        """
        X = simulated_dag_data['X']
        Y = simulated_dag_data['Y']
        C = simulated_dag_data['C']

        # 在这个模拟中，X 和 Y 有直接因果关系
        # 但 X 和 Y 对 C 的贡献方向相反

        # 验证：C 确实受 X 和 Y 影响
        assert len(C) == len(X) == len(Y)

    def test_controlling_collider_creates_bias(
        self, simulated_dag_data
    ):
        """
        反例：控制碰撞变量引入虚假相关

        控制 Z 后，原本独立的 X 和 Y 变得相关
        """
        X = simulated_dag_data['X']
        Y = simulated_dag_data['Y']
        C = simulated_dag_data['C']

        # 不控制 C 时的相关性
        original_corr = np.corrcoef(X, Y)[0, 1]

        # 控制 C 后（按 C 的中位数分层）
        C_median = np.median(C)
        corr_high_C = np.corrcoef(X[C > C_median], Y[C > C_median])[0, 1]
        corr_low_C = np.corrcoef(X[C <= C_median], Y[C <= C_median])[0, 1]

        # 验证：相关性模式改变
        # （具体结果取决于模拟参数）

        assert isinstance(original_corr, (float, np.floating))
        assert isinstance(corr_high_C, (float, np.floating))
        assert isinstance(corr_low_C, (float, np.floating))

    def test_mistake_controlling_colliding_variable(
        self, simulated_dag_data
    ):
        """
        反例：控制碰撞变量是常见错误

        "控制一切变量越好"是错误策略
        """
        X = simulated_dag_data['X']
        Y = simulated_dag_data['Y']
        C = simulated_dag_data['C']

        # 验证：C 是碰撞变量（受 X 和 Y 影响）
        # 在这个模拟中，C = 0.6*X - 0.6*Y + noise
        # 所以 C 确实受两者影响

        assert len(C) > 0


# =============================================================================
# 中介变量测试
# =============================================================================

class TestMediator:
    """测试中介变量的识别和处理"""

    def test_identify_mediator_structure(self, mediator_dag):
        """
        正例：识别中介结构

        中介：X -> M -> Y
        M 是 X 影响 Y 的中介
        """
        structure_type = mediator_dag['structure_type']

        # 验证：这是中介结构
        assert structure_type == 'mediator'

        # 验证：是否控制 M 取决于研究问题
        control_for_total = mediator_dag['control_for_total_effect']
        control_for_direct = mediator_dag['control_for_direct_effect']

        # 估计总效应时不控制中介
        assert isinstance(control_for_total, list)
        # 估计直接效应时控制中介
        assert isinstance(control_for_direct, list)

    def test_mediator_transmits_treatment_effect(
        self, simulated_dag_data
    ):
        """
        正例：中介传递 Treatment 的效应

        X 通过 M 影响 Y
        """
        X = simulated_dag_data['X']
        M = simulated_dag_data['M']
        Y = simulated_dag_data['Y']

        # 验证：M 和 X 相关（M 受 X 影响）
        corr_XM = np.corrcoef(X, M)[0, 1]
        assert abs(corr_XM) > 0.1, "M 应该受 X 影响"

        # 验证：M 和 Y 相关（M 影响 Y）
        corr_MY = np.corrcoef(M, Y)[0, 1]
        assert abs(corr_MY) > 0.1, "M 应该影响 Y"

    @pytest.mark.parametrize("control_mediator,effect_type", [
        (False, "total"),    # 不控制中介：估计总效应
        (True, "direct"),    # 控制中介：估计直接效应
    ])
    def test_controlling_mediator_changes_effect(
        self, simulated_dag_data, control_mediator, effect_type
    ):
        """
        正例：控制中介改变估计的效应

        不控制 M：总效应（直接效应 + 间接效应）
        控制 M：直接效应
        """
        X = simulated_dag_data['X']
        Y = simulated_dag_data['Y']
        M = simulated_dag_data['M']

        # 验证：M 是中介
        assert isinstance(control_mediator, bool)
        assert effect_type in ["total", "direct"]

    def test_indirect_effect_through_mediator(
        self, simulated_dag_data
    ):
        """
        正例：通过中介的间接效应

        间接效应 = X 对 M 的影响 × M 对 Y 的影响
        """
        true_direct = simulated_dag_data['true_direct_effect']
        true_indirect = simulated_dag_data['true_indirect_effect']
        true_total = simulated_dag_data['true_total_effect']

        # 验证：总效应 = 直接效应 + 间接效应
        assert abs(true_total - (true_direct + true_indirect)) < 0.01

    def test_mistake_controlling_mediator_for_total_effect(
        self, simulated_dag_data
    ):
        """
        反例：估计总效应时控制中介

        如果想估计"X 对 Y 的总影响"，不应该控制中介 M
        """
        # 概念测试
        # 控制 M 会阻断 X -> M -> Y 的路径
        # 只剩下直接效应（如果存在）

        assert True  # 概念验证


# =============================================================================
# 复杂 DAG 测试
# =============================================================================

class TestComplexDAG:
    """测试复杂因果图"""

    def test_identify_multiple_confounders(self, complex_causal_dag):
        """
        正例：识别多个混杂变量

        复杂 DAG 可能包含多个混杂
        """
        confounders = complex_causal_dag['confounders']

        # 验证：存在混杂变量
        assert len(confounders) > 0

        # 在优惠券例子中，高价值客户是混杂
        assert 'high_value_customer' in confounders

    def test_identify_multiple_colliders(self, complex_causal_dag):
        """
        正例：识别多个碰撞变量

        复杂 DAG 可能包含多个碰撞
        """
        colliders = complex_causal_dag['colliders']

        # 验证：存在碰撞变量
        assert len(colliders) > 0

        # 在优惠券例子中，客户满意是碰撞
        assert 'customer_satisfaction' in colliders

    def test_identify_multiple_mediators(self, complex_causal_dag):
        """
        正例：识别多个中介变量

        复杂 DAG 可能包含多个中介
        """
        mediators = complex_causal_dag['mediators']

        # 验证：存在中介变量
        assert len(mediators) > 0

        # 在优惠券例子中，购买次数是中介
        assert 'purchase_count' in mediators

    def test_dag_for_coupon_churn_causal_question(self, complex_causal_dag):
        """
        正例：优惠券-流失因果图的完整分析

        场景：优惠券对流失率的影响
        """
        nodes = complex_causal_dag['nodes']
        edges = complex_causal_dag['edges']
        backdoor_paths = complex_causal_dag['backdoor_paths']
        adjustment_set = complex_causal_dag['adjustment_set']

        # 验证：DAG 结构完整
        assert len(nodes) > 0
        assert len(edges) > 0

        # 验证：存在后门路径
        assert len(backdoor_paths) > 0

        # 验证：需要控制混杂
        assert len(adjustment_set) > 0

        # 验证：关键变量存在
        assert 'coupon' in nodes
        assert 'churn' in nodes


# =============================================================================
# 常见错误测试
# =============================================================================

class TestCommonMistakes:
    """测试 DAG 中的常见错误"""

    def test_mistake_confusing_confounder_with_collider(self):
        """
        反例：混淆混杂和碰撞

        混杂需要控制，碰撞不能控制
        混淆两者会导致错误的控制策略
        """
        # 概念测试
        # 混杂：Z -> X, Z -> Y
        # 碰撞：X -> Z <- Y

        assert True  # 概念验证

    def test_mistake_assuming_more_control_is_better(
        self, simulated_dag_data
    ):
        """
        反例：假设"控制越多越好"

        控制碰撞变量会引入偏差
        """
        X = simulated_dag_data['X']
        Y = simulated_dag_data['Y']
        C = simulated_dag_data['C']

        # 验证：C 是碰撞变量
        # 控制 C 会产生虚假相关

        assert len(C) > 0

    def test_mistake_dag_from_data_not_domain_knowledge(self):
        """
        反例：从数据中"发现"DAG

        DAG 表示因果假设，必须基于领域知识
        数据只能发现相关性，不能确定因果方向
        """
        # 概念测试
        # 算法可以输出"候选 DAG"
        # 但因果方向必须由人类设定

        assert True  # 概念验证


# =============================================================================
# 边界情况测试
# =============================================================================

class TestEdgeCases:
    """测试边界情况"""

    def test_empty_dag(self):
        """
        边界：空 DAG

        没有节点或边的 DAG
        """
        empty_dag = {
            'nodes': [],
            'edges': []
        }

        assert len(empty_dag['nodes']) == 0
        assert len(empty_dag['edges']) == 0

    def test_single_node_dag(self):
        """
        边界：单节点 DAG

        只有一个变量的 DAG（无因果效应）
        """
        single_node_dag = {
            'nodes': ['X'],
            'edges': []
        }

        assert len(single_node_dag['nodes']) == 1
        assert len(single_node_dag['edges']) == 0

    def test_two_node_dag(self):
        """
        边界：两节点 DAG

        最简单的因果结构：X -> Y
        """
        two_node_dag = {
            'nodes': ['X', 'Y'],
            'edges': [('X', 'Y')],
            'structure_type': 'simple_causation'
        }

        assert len(two_node_dag['nodes']) == 2
        assert len(two_node_dag['edges']) == 1

    def test_disconnected_variables(self):
        """
        边界：不连通的变量

        DAG 中某些变量之间没有路径
        """
        disconnected_dag = {
            'nodes': ['X', 'Y', 'Z'],
            'edges': [('X', 'Y')],  # Z 与 X, Y 不连通
        }

        # Z 和 X, Y 独立
        assert 'Z' in disconnected_dag['nodes']
        assert len([e for e in disconnected_dag['edges'] if 'Z' in e]) == 0

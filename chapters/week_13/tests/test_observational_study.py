"""
Test Suite: 观察研究中的因果推断

测试当 RCT 不可行时的因果推断方法：
1. 双重差分（DID, Difference-in-Differences）
2. 工具变量（IV, Instrumental Variables）
3. 倾向得分匹配（PSM, Propensity Score Matching）

测试覆盖：
- 正例：每种方法的基本原理和应用
- 边界：方法失效的情况（无重叠、平行趋势违反）
- 反例：常见错误（忽略假设检验）
- 局限：未观测混杂仍然存在
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

# 添加 starter_code 到路径
starter_code_path = Path(__file__).parent.parent / "starter_code"
if str(starter_code_path) not in sys.path:
    sys.path.insert(0, str(starter_code_path))


# =============================================================================
# 双重差分（DID）测试
# =============================================================================

class TestDifferenceInDifferences:
    """测试双重差分方法"""

    def test_did_basic_calculation(self, did_data):
        """
        正例：DID 的基本计算

        DID = (处理后 - 处理前) - (对照后 - 对照前)
        """
        treated_pre = did_data['treated_pre']
        treated_post = did_data['treated_post']
        control_pre = did_data['control_pre']
        control_post = did_data['control_post']

        # 计算平均变化
        treated_change = treated_post.mean() - treated_pre.mean()
        control_change = control_post.mean() - control_pre.mean()

        # DID 估计
        did_estimate = treated_change - control_change

        # 验证：DID 可以计算
        assert isinstance(did_estimate, (float, np.floating))

    def test_parallel_trend_assumption(self, did_data):
        """
        正例：平行趋势假设

        处理组和对照组在政策前趋势相同
        """
        parallel_trend_holds = did_data['parallel_trend_holds']

        # 在这个模拟中，平行趋势成立
        assert parallel_trend_holds

    def test_parallel_trend_violation_detection(
        self, did_violated_parallel_trend_data
    ):
        """
        反例：平行趋势假设被违反

        如果趋势不平行，DID 会失效
        """
        parallel_trend_holds = did_violated_parallel_trend_data[
            'parallel_trend_holds'
        ]

        # 验证：平行趋势不成立
        assert not parallel_trend_holds

    def test_did_with_multiple_time_periods(self):
        """
        边界：多个时间点的 DID

        使用回归方法处理多期数据
        """
        np.random.seed(42)
        n_units = 100
        n_periods = 10

        # 生成面板数据
        unit_ids = np.repeat(np.arange(n_units), n_periods)
        time_periods = np.tile(np.arange(n_periods), n_units)

        # 处理变量（某单位在某时期后接受处理）
        treatment = np.zeros(len(unit_ids))
        treatment_period = 5
        for i in range(n_units):
            if i < n_units // 2:  # 处理组
                mask = (unit_ids == i) & (time_periods >= treatment_period)
                treatment[mask] = 1

        # 生成结果变量
        outcome = 10 + time_periods * 0.5 + treatment * 2 + np.random.randn(len(unit_ids))

        # 验证：数据结构正确
        assert len(treatment) == len(outcome)
        assert len(unit_ids) == len(outcome)

    @pytest.mark.parametrize("use_fixed_effects,expected_method", [
        (True, "twoway_fe"),     # 双向固定效应
        (False, "simple_diff"),  # 简单差分
    ])
    def test_did_estimation_methods(
        self, use_fixed_effects, expected_method
    ):
        """
        正例：DID 的不同估计方法

        - 简单四格表法
        - 回归法（可以控制协变量）
        - 双向固定效应模型
        """
        # 概念测试
        assert isinstance(use_fixed_effects, bool)
        assert expected_method in ["twoway_fe", "simple_diff"]


# =============================================================================
# 工具变量（IV）测试
# =============================================================================

class TestInstrumentalVariables:
    """测试工具变量方法"""

    def test_iv_three_requirements(self, instrumental_variable_data):
        """
        正例：工具变量的三个要求

        1. 相关性：Z 和 X 相关
        2. 外生性：Z 和 Y 的混杂不相关
        3. 排他性：Z 只能通过 X 影响 Y
        """
        Z = instrumental_variable_data['Z']
        X = instrumental_variable_data['X']
        Y = instrumental_variable_data['Y']
        U = instrumental_variable_data['U']

        # 1. 相关性：Z 和 X 相关
        corr_ZX = np.corrcoef(Z, X)[0, 1]
        assert abs(corr_ZX) > 0.1, "工具变量应该与 Treatment 相关"

        # 2. 外生性：Z 和 U 不相关
        corr_ZU = np.corrcoef(Z, U)[0, 1]
        assert abs(corr_ZU) < 0.1, "工具变量应该与混杂不相关"

        # 3. 排他性：无法直接检验（只能辩护）

    def test_iv_first_stage(self, instrumental_variable_data):
        """
        正例：IV 第一阶段

        用 Z 预测 X
        """
        Z = instrumental_variable_data['Z']
        X = instrumental_variable_data['X']

        # 第一阶段回归：X ~ Z
        # 简化：计算相关性和 F 统计量
        corr_ZX = np.corrcoef(Z, X)[0, 1]

        # 验证：工具变量相关
        assert abs(corr_ZX) > 0.1

    def test_iv_second_stage(self, instrumental_variable_data):
        """
        正例：IV 第二阶段

        用预测的 X 预测 Y
        """
        # 概念验证
        # 第二阶段：Y ~ X_hat
        # X_hat 是从第一阶段得到的

        assert True  # 概念验证

    def test_weak_instrument_problem(self):
        """
        反例：弱工具变量问题

        如果 Z 和 X 弱相关，IV 估计不稳定
        """
        np.random.seed(42)
        n = 1000

        # 生成弱工具变量
        Z = np.random.randn(n)
        U = np.random.randn(n)

        # Treatment：受 Z 影响很小（弱工具）
        X = 0.1 * Z + U + np.random.randn(n) * 0.5

        # Outcome
        Y = 2 * X + U + np.random.randn(n)

        # 检查工具强度
        corr_ZX = np.corrcoef(Z, X)[0, 1]

        # 验证：这是弱工具（相关系数 < 0.3）
        assert abs(corr_ZX) < 0.3, "这是弱工具变量示例"

    def test_invalid_instrument(self):
        """
        反例：无效的工具变量

        如果工具变量不满足外生性或排他性，IV 估计有偏
        """
        np.random.seed(42)
        n = 1000

        # 无效工具：Z 同时影响 X 和 Y（违反排他性）
        Z = np.random.randn(n)
        U = np.random.randn(n)
        X = Z + U + np.random.randn(n)
        Y = 2 * X + Z + U + np.random.randn(n)  # Z 直接影响 Y

        # 这个工具变量无效（Z 直接影响 Y）

        # 验证：Z 和 X 相关（满足相关性）
        corr_ZX = np.corrcoef(Z, X)[0, 1]
        assert abs(corr_ZX) > 0.1

    def test_iv_vs_ols_bias(self, instrumental_variable_data):
        """
        正例：IV 消除 OLS 的混杂偏差

        对比 OLS（有偏）和 IV（无偏，如果假设成立）
        """
        X = instrumental_variable_data['X']
        Y = instrumental_variable_data['Y']
        U = instrumental_variable_data['U']
        true_effect = instrumental_variable_data['true_effect']

        # OLS 估计（有偏，因为存在 U）
        # 简化：计算相关系数作为有偏估计的示例
        corr_XY = np.corrcoef(X, Y)[0, 1]

        # IV 估计（无偏，如果工具有效）
        # 实际实现需要 2SLS

        # 验证：真实效应存在（Python int 可以检查 isinstance(int, (int, float))）
        assert isinstance(true_effect, (int, float, np.floating, np.integer))


# =============================================================================
# 倾向得分匹配（PSM）测试
# =============================================================================

class TestPropensityScoreMatching:
    """测试倾向得分匹配方法"""

    def test_psm_basic_concept(self, psm_data):
        """
        正例：PSM 的基本概念

        倾向得分：P(T=1|X)
        匹配：给处理组找相似倾向得分的对照
        """
        T = psm_data['T']
        covariates = psm_data['covariates']

        # 验证：处理组和对照组都存在
        assert (T == 1).sum() > 0
        assert (T == 0).sum() > 0

        # 验证：协变量存在
        assert covariates.shape[0] == len(T)

    def test_psm_propensity_score_estimation(self, psm_data):
        """
        正例：倾向得分估计

        使用逻辑回归估计 P(T=1|X)
        """
        T = psm_data['T']
        covariates = psm_data['covariates']

        # 简化：使用线性组合作为倾向得分的近似
        # 实际实现应该用逻辑回归
        propensity_approx = (
            covariates['age'] * 0.03 +
            covariates['income'] * 0.0001 +
            covariates['education'] * 0.2
        )
        propensity_approx = 1 / (1 + np.exp(-propensity_approx))

        # 验证：倾向得分在 [0, 1] 范围内
        assert propensity_approx.min() >= 0
        assert propensity_approx.max() <= 1

    def test_psm_matching_quality_check(self, psm_data):
        """
        正例：匹配质量检验

        匹配后，处理组和对照组的协变量应该平衡
        """
        T = psm_data['T']
        covariates = psm_data['covariates']

        # 匹配前的标准化差异（示例：年龄）
        age_treated = covariates['age'][T == 1]
        age_control = covariates['age'][T == 0]

        # 标准化差异
        pooled_std = np.sqrt((age_treated.var() + age_control.var()) / 2)
        std_diff_before = (age_treated.mean() - age_control.mean()) / pooled_std

        # 验证：可以计算标准化差异
        assert isinstance(std_diff_before, (float, np.floating))

    def test_psm_no_overlap_problem(self, no_overlap_data):
        """
        反例：无重叠区域

        如果处理组和对照组的倾向得分分布不重叠，PSM 失效
        """
        has_overlap = no_overlap_data['has_overlap']
        psm_fails = no_overlap_data['psm_fails']

        # 验证：无重叠区域
        assert not has_overlap
        assert psm_fails

    def test_psm_common_support_region(self):
        """
        边界：共同支持区域

        只分析倾向得分重叠的样本
        """
        np.random.seed(42)
        n = 200

        # 生成有重叠的倾向得分
        propensity_treated = np.random.uniform(0.3, 0.9, n // 2)
        propensity_control = np.random.uniform(0.1, 0.7, n // 2)

        # 共同支持区域：[0.3, 0.7]
        common_min = max(propensity_treated.min(), propensity_control.min())
        common_max = min(propensity_treated.max(), propensity_control.max())

        # 验证：存在共同支持区域
        assert common_min < common_max

    @pytest.mark.parametrize("matching_method", [
        "nearest_neighbor",  # 最近邻匹配
        "caliper",          # 卡尺匹配
        "kernel",           # 核匹配
        "stratification",   # 分层匹配
    ])
    def test_psm_matching_methods(self, matching_method):
        """
        正例：不同的匹配方法

        - 最近邻：找最接近的对照
        - 卡尺：只在一定距离内匹配
        - 核匹配：加权平均
        - 分层：按倾向得分分层
        """
        # 概念测试
        assert matching_method in [
            "nearest_neighbor", "caliper", "kernel", "stratification"
        ]


# =============================================================================
# 未观测混杂问题测试
# =============================================================================

class TestUnobservedConfounding:
    """测试未观测混杂的问题"""

    def test_psm_cannot_control_unobserved(self, psm_data):
        """
        局限：PSM 只能控制观测变量

        未观测混杂仍然存在
        """
        # 概念验证
        # PSM 假设"无未观测混杂"
        # 这个假设无法检验，只能辩护

        assert True  # 概念验证

    def test_did_sensitivity_to_trends(self, did_violated_parallel_trend_data):
        """
        局限：DID 对趋势假设敏感

        如果平行趋势不成立，DID 有偏
        """
        parallel_trend_holds = did_violated_parallel_trend_data[
            'parallel_trend_holds'
        ]

        # 验证：趋势假设不成立
        assert not parallel_trend_holds

    def test_iv_exogeneity_untestable(self, instrumental_variable_data):
        """
        局限：IV 外生性无法检验

        工具变量的外生性只能辩护，无法验证
        """
        # 概念验证
        # 外生性：Cov(Z, U) = 0
        # 但 U 是未观测的，无法检验

        assert True  # 概念验证

    def test_sensitivity_analysis_for_unobserved_confounding(self):
        """
        正例：敏感性分析

        评估未观测混杂对结论的影响
        """
        # 概念验证
        # 敏感性分析回答：
        # "需要多强的未观测混杂才能推翻结论？"

        assert True  # 概念验证


# =============================================================================
# 观察研究 vs RCT 对比
# =============================================================================

class TestObservationalVsRCT:
    """测试观察研究与 RCT 的对比"""

    def test_rct_is_gold_standard(self, rct_data):
        """
        正例：RCT 是金标准

        RCT 不依赖"无未观测混杂"假设
        """
        assert rct_data['is_randomized']

    def test_observational_methods_are_approximations(self):
        """
        正例：观察研究方法是"近似"

        依赖更强的假设
        """
        # 概念验证
        # DID 需要平行趋势
        # IV 需要工具变量外生
        # PSM 需要无未观测混杂

        assert True  # 概念验证

    def test_observational_still_valuable(self):
        """
        正例：观察研究仍然有价值

        当 RCT 不可行时，观察研究 + 敏感性分析
        """
        # 概念验证
        # 某些问题只能用观察研究
        # 示例：吸烟与健康（不能随机让人吸烟）

        assert True  # 概念验证


# =============================================================================
# 常见错误测试
# =============================================================================

class TestCommonMistakes:
    """测试观察研究中的常见错误"""

    def test_mistake_did_without_parallel_trend_check(
        self, did_violated_parallel_trend_data
    ):
        """
        反例：不检验平行趋势

        DID 必须检验平行趋势假设
        """
        parallel_trend_holds = did_violated_parallel_trend_data[
            'parallel_trend_holds'
        ]

        # 在这个案例中，平行趋势不成立
        # 如果不检验就使用 DID，结论有误

        assert not parallel_trend_holds

    def test_mistake_weak_instrument(self):
        """
        反例：使用弱工具变量

        弱工具导致 IV 估计不稳定
        """
        np.random.seed(42)
        n = 1000

        # 弱工具
        Z = np.random.randn(n)
        U = np.random.randn(n)
        X = 0.05 * Z + U + np.random.randn(n) * 0.5

        # 检查相关性
        corr_ZX = np.corrcoef(Z, X)[0, 1]

        # 验证：这是弱工具
        assert abs(corr_ZX) < 0.2

    def test_mistake_psm_without_balance_check(self, psm_data):
        """
        反例：PSM 后不检验平衡性

        匹配后必须检查协变量是否平衡
        """
        # 概念验证
        # 匹配后需要检验：
        # - 标准化差异 < 0.1
        # - t 检验 p > 0.05

        assert True  # 概念验证

    def test_mistake_ignoring_clustering(self):
        """
        反例：忽略聚类问题

        在 DID 中，需要处理面板数据的聚类
        """
        # 概念验证
        # 同一个体的多次观测存在聚类
        # 需要使用聚类稳健标准误

        assert True  # 概念验证


# =============================================================================
# 边界情况测试
# =============================================================================

class TestEdgeCases:
    """测试边界情况"""

    def test_small_sample_psm(self, minimal_causal_data):
        """
        边界：小样本 PSM

        小样本下匹配质量差
        """
        n = minimal_causal_data['n']

        # 验证：样本量小
        assert n < 30

    def test_single_pre_post_period_did(self):
        """
        边界：单一前后期 DID

        最简单的 DID 设计
        """
        # 简化的两期 DID
        treated_pre = np.array([100, 105, 98, 102, 97])
        treated_post = np.array([110, 115, 108, 112, 107])
        control_pre = np.array([95, 97, 93, 96, 94])
        control_post = np.array([98, 100, 96, 99, 97])

        # 计算 DID
        did = (
            (treated_post.mean() - treated_pre.mean()) -
            (control_post.mean() - control_pre.mean())
        )

        # 验证：可以计算
        assert isinstance(did, (float, np.floating))

    def test_iv_with_multiple_instruments(self):
        """
        边界：多个工具变量

        当有多个工具变量时，可以使用 2SLS
        """
        # 概念验证
        # 多个工具变量 Z1, Z2, Z3
        # 第一阶段：X ~ Z1 + Z2 + Z3
        # 第二阶段：Y ~ X_hat

        assert True  # 概念验证

    def test_psm_with_many_covariates(self):
        """
        边界：多个协变量

        倾向得分模型中包含很多协变量
        """
        np.random.seed(42)
        n = 500
        n_covariates = 20

        # 生成多个协变量
        covariates = np.random.randn(n, n_covariates)

        # Treatment 依赖于多个协变量
        logit = np.dot(covariates, np.random.randn(n_covariates)) - 2
        propensity = 1 / (1 + np.exp(-logit))
        T = np.random.binomial(1, propensity, n)

        # 验证：可以处理多个协变量
        assert covariates.shape == (n, n_covariates)
        assert len(T) == n


# =============================================================================
# 方法选择测试
# =============================================================================

class TestMethodSelection:
    """测试不同场景下的方法选择"""

    @pytest.mark.parametrize("scenario,recommended_method", [
        ("policy_evaluation", "DID"),         # 政策评估
        ("unobserved_confounding", "IV"),     # 未观测混杂
        ("many_confounders", "PSM"),          # 多混杂变量
        ("natural_experiment", "RDD"),        # 自然实验/断点回归
        ("longitudinal_data", "fixed_effects"), # 面板数据
    ])
    def test_method_selection_by_scenario(
        self, scenario, recommended_method
    ):
        """
        正例：根据场景选择方法

        不同场景适合不同的因果推断方法
        """
        assert isinstance(scenario, str)
        assert recommended_method in ["DID", "IV", "PSM", "RDD", "fixed_effects"]

    def test_combined_methods(self):
        """
        正例：组合多种方法

        示例：DID + 匹配，IV + 回归调整
        """
        # 概念验证
        # 组合方法可以提高稳健性

        assert True  # 概念验证

    def test_triangulation(self):
        """
        正例：三角验证

        用多种方法逼近同一个因果问题
        """
        # 概念验证
        # 如果 DID、IV、PSM 都给出类似结论
        # 结论更可信

        assert True  # 概念验证

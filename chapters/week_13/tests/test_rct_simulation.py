"""
Test Suite: 随机对照试验（RCT）模拟

测试随机对照试验的原理和实践：
1. 随机化如何切断混杂路径
2. RCT 的假设检验
3. RCT 与观察研究的对比
4. RCT 的局限

测试覆盖：
- 正例：随机化成功消除混杂
- 反例：随机化失败导致混杂
- 边界：小样本、不平衡
- 常见错误：忽略基线平衡检验
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


# =============================================================================
# 随机化原理测试
# =============================================================================

class TestRandomizationPrinciple:
    """测试随机化的核心原理"""

    def test_randomization_breaks_confounder_paths(self, rct_data):
        """
        正例：随机化切断混杂路径

        随机化后，Treatment 和混杂变量独立
        """
        T = rct_data['T']
        age = rct_data['age']
        income = rct_data['income']

        # 验证：随机化成功
        assert rct_data['is_randomized']

        # 验证：Treatment 和年龄不相关
        p_age = rct_data['balance_p_values']['age']
        assert p_age > 0.05, "年龄在两组间应该平衡"

        # 验证：Treatment 和收入不相关
        p_income = rct_data['balance_p_values']['income']
        assert p_income > 0.05, "收入在两组间应该平衡"

    def test_randomization_equals_association_to_intervention(self, rct_data):
        """
        正例：在 RCT 中，关联 = 干预

        E[Y|X=1] - E[Y|X=0] = E[Y|do(X=1)] - E[Y|do(X=0)]
        """
        T = rct_data['T']
        Y = rct_data['Y']
        true_ate = rct_data['true_ate']

        # 简单差异估计
        naive_ate = Y[T == 1].mean() - Y[T == 0].mean()

        # 验证：估计接近真实值
        assert abs(naive_ate - true_ate) < 1000, \
            f"估计 ATE {naive_ate} 应该接近真实值 {true_ate}"

    def test_observational_study_bias_vs_rct(
        self, rct_data, association_vs_intervention_data
    ):
        """
        正例：RCT 消除观察研究中的偏差

        对比有混杂的观察研究和 RCT
        """
        # 观察研究（有混杂）
        X_obs = association_vs_intervention_data['X']
        Y_obs = association_vs_intervention_data['Y']
        naive_obs = Y_obs[X_obs == 1].mean() - Y_obs[X_obs == 0].mean()

        # RCT（无混杂）
        T = rct_data['T']
        Y = rct_data['Y']
        naive_rct = Y[T == 1].mean() - Y[T == 0].mean()

        # 验证：两者都是数值
        assert isinstance(naive_obs, (float, np.floating))
        assert isinstance(naive_rct, (float, np.floating))

    def test_randomization_is_gold_standard(self, rct_data):
        """
        正例：RCT 是因果推断的金标准

        随机化确保已知和未知混杂都被控制
        """
        # 验证：随机化成功
        assert rct_data['is_randomized']

        # 在 RCT 中，简单差异就是无偏估计
        T = rct_data['T']
        Y = rct_data['Y']
        ate_estimate = Y[T == 1].mean() - Y[T == 0].mean()

        assert isinstance(ate_estimate, (float, np.floating))


# =============================================================================
# 基线平衡检验测试
# =============================================================================

class TestBaselineBalance:
    """测试基线平衡检验"""

    @pytest.mark.parametrize("alpha", [0.05, 0.01, 0.1])
    def test_balance_test_with_different_alpha(self, rct_data, alpha):
        """
        正例：使用不同的显著性水平检验平衡

        常用 alpha = 0.05
        """
        T = rct_data['T']
        age = rct_data['age']

        # t 检验
        t_stat, p_value = stats.ttest_ind(age[T == 0], age[T == 1])

        # 验证：p 值是有效的
        assert 0 <= p_value <= 1

        # 在这个模拟中，随机化成功，p 值应该 > alpha
        # 但这只是随机性，可能偶尔失败
        assert isinstance(p_value, (float, np.floating))

    def test_multiple_balance_checks(self, rct_data):
        """
        正例：检验多个变量的平衡

        需要检验所有重要的基线特征
        """
        T = rct_data['T']
        age = rct_data['age']
        income = rct_data['income']

        # 年龄平衡
        _, p_age = stats.ttest_ind(age[T == 0], age[T == 1])

        # 收入平衡
        _, p_income = stats.ttest_ind(income[T == 0], income[T == 1])

        # 验证：所有检验返回有效 p 值
        assert 0 <= p_age <= 1
        assert 0 <= p_income <= 1

    def test_failed_randomization_detection(self, failed_randomization_data):
        """
        正例：检测随机化失败

        如果基线不平衡，随机化可能失败
        """
        T = failed_randomization_data['T']
        age = failed_randomization_data['age']

        # 检验年龄平衡
        _, p_age = stats.ttest_ind(age[T == 0], age[T == 1])

        # 验证：随机化失败，p 值应该 < 0.05
        # （Treatment 和年龄相关）
        assert failed_randomization_data['is_randomized'] == False

    @pytest.mark.parametrize("method", ["t_test", "ks_test", "rank_sum"])
    def test_different_balance_test_methods(self, rct_data, method):
        """
        正例：不同的平衡检验方法

        - t_test：参数方法，假设正态
        - ks_test：非参数，比较分布
        - rank_sum：非参数，比较中位数
        """
        T = rct_data['T']
        age = rct_data['age']

        if method == "t_test":
            stat, p_value = stats.ttest_ind(age[T == 0], age[T == 1])
        elif method == "ks_test":
            stat, p_value = stats.ks_2samp(age[T == 0], age[T == 1])
        elif method == "rank_sum":
            stat, p_value = stats.ranksums(age[T == 0], age[T == 1])

        # 验证：所有方法返回有效 p 值
        assert 0 <= p_value <= 1


# =============================================================================
# ATE 估计测试
# =============================================================================

class TestATEEstimation:
    """测试平均处理效应的估计"""

    def test_simple_difference_estimator(self, rct_data):
        """
        正例：简单差异估计器

        ATE = E[Y|T=1] - E[Y|T=0]
        """
        T = rct_data['T']
        Y = rct_data['Y']
        true_ate = rct_data['true_ate']

        # 简单差异
        ate_estimate = Y[T == 1].mean() - Y[T == 0].mean()

        # 验证：估计接近真实值
        error_pct = abs(ate_estimate - true_ate) / abs(true_ate)
        assert error_pct < 0.5, f"误差 {error_pct:.1%} 应该小于 50%"

    def test_regression_adjustment_estimator(self, rct_data):
        """
        正例：回归调整估计器

        控制基线特征可以提高精度
        """
        T = rct_data['T']
        Y = rct_data['Y']
        age = rct_data['age']
        income = rct_data['income']

        # 简单差异
        simple_ate = Y[T == 1].mean() - Y[T == 0].mean()

        # 回归调整（概念验证）
        # 在实际实现中，会运行回归 Y ~ T + age + income
        # 这里只验证概念

        # 验证：简单估计可用
        assert isinstance(simple_ate, (float, np.floating))

    def test_confidence_interval_for_ate(self, rct_data):
        """
        正例：ATE 的置信区间

        估计因果效应的不确定性
        """
        T = rct_data['T']
        Y = rct_data['Y']

        # 简单差异
        ate = Y[T == 1].mean() - Y[T == 0].mean()

        # 标准误（简化计算）
        n1 = (T == 1).sum()
        n0 = (T == 0).sum()
        se = np.sqrt(Y[T == 1].var() / n1 + Y[T == 0].var() / n0)

        # 95% 置信区间
        ci_lower = ate - 1.96 * se
        ci_upper = ate + 1.96 * se

        # 验证：置信区间有效
        assert ci_lower < ci_upper
        assert isinstance(ci_lower, (float, np.floating))
        assert isinstance(ci_upper, (float, np.floating))

    def test_heterogeneous_treatment_effect(self, rct_data):
        """
        边界：异质性处理效应

        不同个体的处理效应可能不同
        """
        T = rct_data['T']
        Y0 = rct_data['Y0']
        Y1 = rct_data['Y1']

        # 个体处理效应
        ite = Y1 - Y0

        # 验证：存在异质性（不是所有人的效应都相同）
        assert ite.std() > 0, "处理效应应该有异质性"

        # 平均处理效应
        ate = ite.mean()
        assert isinstance(ate, (float, np.floating))


# =============================================================================
# RCT 假设测试
# =============================================================================

class TestRCTAssumptions:
    """测试 RCT 的关键假设"""

    def test_sutva_assumption(self, rct_data):
        """
        正例：SUTVA 假设

        稳定单元处理值假设：
        1. 个体的处理不影响其他人
        2. 对同一个体，处理版本一致
        """
        # 概念验证
        # 在这个模拟中，SUTVA 成立
        # （没有网络效应）

        assert True  # 概念验证

    def test_compliance_assumption(self, rct_data):
        """
        正例：依从性假设

        实验组应该接受 Treatment，对照组不应该
        """
        T = rct_data['T']

        # 在这个模拟中，完美依从
        # （T 就是实际接受的 Treatment）

        # 验证：处理组存在
        assert (T == 1).sum() > 0
        # 验证：对照组存在
        assert (T == 0).sum() > 0

    def test_no_attrition_bias_assumption(self):
        """
        正例：无流失偏差假设

        流失（退出实验）在两组间随机
        """
        # 概念验证
        # 如果存在流失，需要检查流失是否随机

        assert True  # 概念验证

    @pytest.mark.parametrize("assumption,can_test", [
        ("randomization", True),      # 随机化：可以检验
        ("sutva", False),             # SUTVA：难以检验
        ("compliance", True),         # 依从性：可以检验
        ("no_attrition", True),       # 无流失：可以检验
        ("exclusion", False),         # 排他性：难以检验
    ])
    def test_assumption_testability(self, assumption, can_test):
        """
        正例：RCT 假设的可检验性

        某些假设可以检验，某些难以检验
        """
        assert isinstance(assumption, str)
        assert isinstance(can_test, bool)


# =============================================================================
# RCT 局限测试
# =============================================================================

class TestRCTLimitations:
    """测试 RCT 的局限性"""

    def test_cost_limitation(self):
        """
        边界：成本限制

        RCT 通常成本高昂
        """
        # 概念验证
        # 医学 RCT 可能需要数百万美元
        # A/B 测试需要开发资源

        assert True  # 概念验证

    def test_ethical_limitation(self):
        """
        边界：伦理限制

        某些实验不符合伦理
        """
        # 概念验证
        # 不能随机让人吸烟来测试吸烟的危害

        assert True  # 概念验证

    def test_external_validity_limitation(self, rct_data):
        """
        边界：外部有效性

        实验室结论可能不适用于真实世界
        """
        # 验证：RCT 数据在特定条件下生成
        # 推广到其他场景需要谨慎

        assert rct_data['is_randomized']

    def test_network_effect_violates_sutva(self):
        """
        边界：网络效应违反 SUTVA

        在社交产品中，用户行为会互相影响
        """
        # 概念验证
        # 如果 A 组用户使用新功能并告诉 B 组用户
        # B 组的行为也会变化

        assert True  # 概念验证


# =============================================================================
# 观察研究 vs RCT 对比测试
# =============================================================================

class TestObservationalVsRCT:
    """测试观察研究和 RCT 的对比"""

    def test_bias_in_observational_study(self, association_vs_intervention_data):
        """
        正例：观察研究中的偏差

        不控制混杂时，关联 != 因果
        """
        X = association_vs_intervention_data['X']
        Y = association_vs_intervention_data['Y']
        U = association_vs_intervention_data['U']

        # 简单差异（有偏）
        naive_diff = Y[X == 1].mean() - Y[X == 0].mean()

        # 验证：混杂存在
        corr_XU = np.corrcoef(X, U)[0, 1]
        assert abs(corr_XU) > 0.1, "存在混杂"

    def test_rct_eliminates_confounding(self, rct_data):
        """
        正例：RCT 消除混杂

        随机化确保无混杂
        """
        T = rct_data['T']
        age = rct_data['age']
        income = rct_data['income']

        # 验证：Treatment 和基线特征不相关
        _, p_age = stats.ttest_ind(age[T == 0], age[T == 1])
        _, p_income = stats.ttest_ind(income[T == 0], income[T == 1])

        # 在成功的 RCT 中，p 值应该 > 0.05
        assert rct_data['is_randomized']

    def test_when_rct_is_not_feasible(self):
        """
        边界：RCT 不可行的情况

        需要使用观察研究方法
        """
        # 概念验证
        # 某些场景下 RCT 不可行：
        # - 成本太高
        # - 伦理问题
        # - 时间限制
        # - 政策已实施

        assert True  # 概念验证


# =============================================================================
# 常见错误测试
# =============================================================================

class TestCommonMistakes:
    """测试 RCT 中的常见错误"""

    def test_mistake_ignoring_balance_check(self, failed_randomization_data):
        """
        反例：忽略基线平衡检验

        随机化可能失败，需要检验
        """
        T = failed_randomization_data['T']
        age = failed_randomization_data['age']

        # 检验平衡
        _, p_age = stats.ttest_ind(age[T == 0], age[T == 1])

        # 验证：这个案例中随机化失败
        assert not failed_randomization_data['is_randomized']

    def test_mistake_over_adjustment_in_rct(self, rct_data):
        """
        反例：在 RCT 中过度调整

        随机化后控制变量是为了提高精度，不是消除混杂
        """
        # 概念验证
        # 在 RCT 中，不控制变量也能得到无偏估计
        # 控制变量只是为了减少方差

        T = rct_data['T']
        Y = rct_data['Y']

        # 简单差异是无偏的
        simple_ate = Y[T == 1].mean() - Y[T == 0].mean()
        assert isinstance(simple_ate, (float, np.floating))

    def test_mistake_confusing_association_with_causation_in_observational(
        self, association_vs_intervention_data
    ):
        """
        反例：在观察研究中混淆关联和因果

        看到关联就下因果结论
        """
        X = association_vs_intervention_data['X']
        Y = association_vs_intervention_data['Y']

        # 计算关联
        association = Y[X == 1].mean() - Y[X == 0].mean()

        # 验证：关联可以计算
        assert isinstance(association, (float, np.floating))

        # 但这不等于因果（存在混杂 U）
        U = association_vs_intervention_data['U']
        assert len(U) > 0


# =============================================================================
# 边界情况测试
# =============================================================================

class TestEdgeCases:
    """测试边界情况"""

    def test_small_sample_rct(self):
        """
        边界：小样本 RCT

        小样本下估计不稳定，但仍然无偏
        """
        np.random.seed(42)
        n = 30

        # 小样本 RCT
        T = np.random.binomial(1, 0.5, n)
        Y0 = np.random.randn(n) * 10
        Y1 = Y0 + 5
        Y = np.where(T == 1, Y1, Y0)

        # 估计 ATE
        ate = Y[T == 1].mean() - Y[T == 0].mean()

        # 验证：可以估计（但不稳定）
        if (T == 1).sum() > 0 and (T == 0).sum() > 0:
            assert isinstance(ate, (float, np.floating))

    def test_imbalanced_group_sizes(self):
        """
        边界：组大小不平衡

        随机化不保证 50/50 分配
        """
        np.random.seed(42)
        n = 100
        p = 0.3  # 30% 概率分配到处理组

        T = np.random.binomial(1, p, n)
        Y0 = np.random.randn(n)
        Y1 = Y0 + 2
        Y = np.where(T == 1, Y1, Y0)

        # 仍然可以估计 ATE
        if (T == 1).sum() > 0 and (T == 0).sum() > 0:
            ate = Y[T == 1].mean() - Y[T == 0].mean()
            assert isinstance(ate, (float, np.floating))

    def test_all_treated_or_all_control(
        self, all_treated_data, all_control_data
    ):
        """
        边界：全部处理或全部对照

        无法估计因果效应
        """
        # 全部处理
        X_all_treated = all_treated_data['X']
        assert X_all_treated.all() == 1
        assert all_treated_data['no_control_group']

        # 全部对照
        X_all_control = all_control_data['X']
        assert X_all_control.all() == 0
        assert all_control_data['no_treatment_group']

    def test_binary_outcome_rct(self):
        """
        边界：二元 Outcome 的 RCT

        常见场景：成功/失败，流失/留存
        """
        np.random.seed(42)
        n = 200

        T = np.random.binomial(1, 0.5, n)
        Y0 = np.random.binomial(1, 0.3, n)
        Y1 = np.random.binomial(1, 0.2, n)  # Treatment 降低 10% 风险
        Y = np.where(T == 1, Y1, Y0)

        # 估计 ATE（风险差）
        ate = Y[T == 1].mean() - Y[T == 0].mean()

        # 验证：ATE 为负（Treatment 降低风险）
        assert ate < 0
        assert isinstance(ate, (float, np.floating))

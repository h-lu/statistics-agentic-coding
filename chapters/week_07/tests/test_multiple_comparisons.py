"""
Test suite for multiple comparisons problem (Week 07)

Focus: Family-wise Error Rate (FWER) calculation and understanding
"""
from __future__ import annotations

import numpy as np
import pytest

import sys
from pathlib import Path

# Add starter_code to path
sys.path.insert(0, str(Path(__file__).parent.parent / "starter_code"))


class TestFamilyWiseErrorRateCalculation:
    """测试 FWER 计算公式"""

    def test_fwer_formula_m_equals_one(self):
        """
        正例：m=1 时 FWER = α

        单个检验时，假阳性率就是显著性水平
        """
        # TODO: Uncomment when solution.py is created
        # result = calculate_family_wise_error_rate(n_tests=1, alpha=0.05)
        # expected = 0.05
        # assert result == pytest.approx(expected, rel=1e-6)
        pytest.skip("solution.py not yet created")

    def test_fwer_formula_m_equals_five(self):
        """
        正例：m=5 时 FWER = 1 - (1-α)^5

        验证公式：FWER = 1 - (1 - α)^m
        FWER = 1 - 0.95^5 ≈ 0.226
        """
        # TODO: Uncomment when solution.py is created
        # alpha = 0.05
        # m = 5
        # result = calculate_family_wise_error_rate(n_tests=m, alpha=alpha)
        # expected = 1 - (1 - alpha) ** m
        # assert result == pytest.approx(expected, rel=0.01)
        pytest.skip("solution.py not yet created")

    def test_fwer_formula_m_equals_ten(self):
        """
        正例：m=10 时 FWER ≈ 0.401

        验证公式对更大的 m
        """
        # TODO: Uncomment when solution.py is created
        # alpha = 0.05
        # m = 10
        # result = calculate_family_wise_error_rate(n_tests=m, alpha=alpha)
        # expected = 1 - (1 - alpha) ** m
        # assert result == pytest.approx(expected, rel=0.01)
        pytest.skip("solution.py not yet created")

    def test_fwer_formula_m_equals_twenty(self):
        """
        正例：m=20 时 FWER ≈ 0.642

        验证公式对更大的 m，假阳性率快速增长
        """
        # TODO: Uncomment when solution.py is created
        # alpha = 0.05
        # m = 20
        # result = calculate_family_wise_error_rate(n_tests=m, alpha=alpha)
        # expected = 1 - (1 - alpha) ** m
        # assert result == pytest.approx(expected, rel=0.01)
        pytest.skip("solution.py not yet created")

    def test_fwer_formula_m_equals_fifty(self):
        """
        正例：m=50 时 FWER ≈ 0.923

        当检验次数很大时，几乎肯定会有假阳性
        """
        # TODO: Uncomment when solution.py is created
        # alpha = 0.05
        # m = 50
        # result = calculate_family_wise_error_rate(n_tests=m, alpha=alpha)
        # expected = 1 - (1 - alpha) ** m
        # assert result == pytest.approx(expected, rel=0.01)
        # assert result > 0.90  # 超过 90%
        pytest.skip("solution.py not yet created")

    def test_fwer_different_alpha(self):
        """
        正例：不同 alpha 下的 FWER

        验证公式对不同 alpha 都成立
        """
        # TODO: Uncomment when solution.py is created
        # m = 10
        # for alpha in [0.01, 0.05, 0.10]:
        #     result = calculate_family_wise_error_rate(n_tests=m, alpha=alpha)
        #     expected = 1 - (1 - alpha) ** m
        #     assert result == pytest.approx(expected, rel=0.01)
        pytest.skip("solution.py not yet created")


class TestFWERInterpretation:
    """测试 FWER 的解释"""

    def test_interpret_fwer_below_percent(self):
        """
        正例：FWER < 5% 时风险低

        当 FWER < 0.05 时，可以接受
        """
        # TODO: Uncomment when solution.py is created
        # interpretation = interpret_fwer(0.03)
        # assert '低' in interpretation or '可接受' in interpretation
        pytest.skip("solution.py not yet created")

    def test_interpret_fwer_above_fifty_percent(self):
        """
        正例：FWER > 50% 时风险高

        当 FWER > 0.5 时，说明假阳性风险很高
        """
        # TODO: Uncomment when solution.py is created
        # interpretation = interpret_fwer(0.65)
        # assert '高' in interpretation or '风险' in interpretation
        pytest.skip("solution.py not yet created")

    def test_interpret_fwer_near_one(self):
        """
        正例：FWER ≈ 1% 时几乎肯定有假阳性

        """
        # TODO: Uncomment when solution.py is created
        # interpretation = interpret_fwer(0.98)
        # assert '几乎肯定' in interpretation or '必定' in interpretation
        pytest.skip("solution.py not yet created")


class TestMultipleComparisonsSimulation:
    """测试多重比较的模拟实验"""

    def test_simulate_single_test_false_positive_rate(self):
        """
        正例：单次检验的假阳性率应约为 α

        当原假设成立时，p < α 的概率应约为 α
        """
        # TODO: Uncomment when solution.py is created
        # np.random.seed(42)
        # alpha = 0.05
        # n_simulations = 10000
        #
        # # 模拟 n_simulations 次，每次做 1 个检验
        # p_values = np.random.uniform(0, 1, n_simulations)
        # false_positive_rate = (p_values < alpha).mean()
        #
        # # 应接近 alpha（允许抽样误差）
        # assert 0.04 <= false_positive_rate <= 0.06
        pytest.skip("solution.py not yet created")

    def test_simulate_multiple_tests_false_positive_accumulation(self):
        """
        正例：多次检验的假阳性累积

        模拟验证理论 FWER
        """
        # TODO: Uncomment when solution.py is created
        # np.random.seed(42)
        # alpha = 0.05
        # n_tests = 10
        # n_simulations = 5000
        #
        # # 每次模拟做 n_tests 个检验，检查是否有至少 1 个假阳性
        # at_least_one_fp = 0
        # for _ in range(n_simulations):
        #     p_values = np.random.uniform(0, 1, n_tests)
        #     if (p_values < alpha).any():
        #         at_least_one_fp += 1
        #
        # empirical_fwer = at_least_one_fp / n_simulations
        # expected_fwer = 1 - (1 - alpha) ** n_tests
        #
        # # 经验 FWER 应接近理论 FWER
        # assert 0.35 <= empirical_fwer <= 0.45  # 接近 0.401
        pytest.skip("solution.py not yet created")

    def test_simulate_expected_false_positive_count(self):
        """
        正例：假阳性数量期望

        m 个检验的假阳性数期望 = m × α
        """
        # TODO: Uncomment when solution.py is created
        # np.random.seed(42)
        # alpha = 0.05
        # n_tests = 50
        # n_simulations = 1000
        #
        # fp_counts = []
        # for _ in range(n_simulations):
        #     p_values = np.random.uniform(0, 1, n_tests)
        #     fp_count = (p_values < alpha).sum()
        #     fp_counts.append(fp_count)
        #
        # mean_fp_count = np.mean(fp_counts)
        # expected = n_tests * alpha  # 50 * 0.05 = 2.5
        #
        # # 平均假阳性数应接近期望
        # assert 2.0 <= mean_fp_count <= 3.0
        pytest.skip("solution.py not yet created")


class TestMultipleComparisonsCorrectionNeed:
    """测试何时需要多重比较校正"""

    def test_no_correction_needed_single_test(self):
        """
        正例：单个检验不需要校正

        m=1 时，FWER = α，无需校正
        """
        # TODO: Uncomment when solution.py is created
        # decision = needs_correction(n_tests=1)
        # assert decision is False
        pytest.skip("solution.py not yet created")

    def test_correction_needed_multiple_tests(self):
        """
        正例：多个检验需要校正

        m>1 时，需要考虑校正
        """
        # TODO: Uncomment when solution.py is created
        # decision = needs_correction(n_tests=5)
        # assert decision is True
        pytest.skip("solution.py not yet created")

    def test_correction_urgency_increases_with_m(self):
        """
        正例：m 越大，校正越重要

        """
        # TODO: Uncomment when solution.py is created
        # urgency_5 = correction_urgency(n_tests=5)
        # urgency_20 = correction_urgency(n_tests=20)
        # urgency_50 = correction_urgency(n_tests=50)
        #
        # # m 越大，紧急度越高
        # assert urgency_5 < urgency_20 < urgency_50
        pytest.skip("solution.py not yet created")


class TestDeadSalmonProblem:
    """测试"死鲑鱼实验"类型的问题"""

    def test_salmon_scenario_detection(self):
        """
        正例：检测类似"死鲑鱼实验"的问题

        当检验次数很大但 p 值都很小时，需要警惕
        """
        # TODO: Uncomment when solution.py is created
        # # 模拟：做了 10000 个检验，发现 5 个"显著"
        # # 但这些可能是假阳性
        # n_tests = 10000
        # n_significant = 5
        #
        # # 预期假阳性数
        # expected_fp = n_tests * 0.05  # 500
        #
        # # 实际显著数远少于预期假阳性数？
        # # 或者：这些"显著"结果可能只是运气
        # analysis = analyze_salmon_scenario(
        #     n_tests=n_tests,
        #     n_significant=n_significant,
        #     alpha=0.05
        # )
        #
        # # 应警告可能是假阳性
        # assert '假阳性' in analysis or '运气' in analysis
        pytest.skip("solution.py not yet created")

    def test_p_hacking_detection(self):
        """
        正例：检测 p-hacking 行为

        """
        # TODO: Uncomment when solution.py is created
        # # 模拟：只报告显著结果，不报告不显著的
        # reported_p_values = [0.03, 0.04, 0.01]  # 只报告显著的
        # n_total_tests = 50  # 实际做了 50 个
        #
        # analysis = detect_p_hacking(
        #     reported_p_values=reported_p_values,
        #     n_total_tests=n_total_tests,
        #     alpha=0.05
        # )
        #
        # # 应警告可能的选择性报告
        # assert '选择性' in analysis or 'p-hacking' in analysis
        pytest.skip("solution.py not yet created")


class TestFWERTable:
    """测试 FWER 表格生成"""

    def test_generate_fwer_table(self):
        """
        正例：生成 FWER 表格

        常见检验次数的 FWER
        """
        # TODO: Uncomment when solution.py is created
        # table = generate_fwer_table(alpha=0.05)
        #
        # # 应包含常见检验次数
        # assert '1' in table
        # assert '5' in table
        # assert '10' in table
        # assert '20' in table
        # assert '50' in table
        #
        # # FWER 应递增
        # assert table['1'] < table['5'] < table['10'] < table['20'] < table['50']
        pytest.skip("solution.py not yet created")

    def test_fwer_table_custom_alpha(self):
        """
        正例：自定义 alpha 的 FWER 表格
        """
        # TODO: Uncomment when solution.py is created
        # table_01 = generate_fwer_table(alpha=0.01)
        # table_10 = generate_fwer_table(alpha=0.10)
        #
        # # alpha=0.01 的 FWER 应小于 alpha=0.10 的
        # for m in [5, 10, 20]:
        #     assert table_01[str(m)] < table_10[str(m)]
        pytest.skip("solution.py not yet created")


class TestEducationalExamples:
    """教育性的测试示例"""

    def test_student_understanding_scenario(self):
        """
        场景：学生想检验 30 个假设

        帮助学生理解风险
        """
        # TODO: Uncomment when solution.py is created
        # scenario = "学生想检验 30 个假设，每个 α=0.05"
        # fwer = calculate_family_wise_error_rate(n_tests=30, alpha=0.05)
        # explanation = explain_risk_to_student(scenario, fwer)
        #
        # # 应解释清楚
        # assert '78%' in explanation or '0.78' in explanation
        # assert '假阳性' in explanation
        pytest.skip("solution.py not yet created")

    def test_researcher_scenario(self):
        """
        场景：研究者做了 20 个检验，发现 2 个显著

        帮助判断是真实效应还是运气
        """
        # TODO: Uncomment when solution.py is created
        # n_tests = 20
        # n_significant = 2
        # fwer = calculate_family_wise_error_rate(n_tests=n_tests, alpha=0.05)
        #
        # # 预期假阳性数 = 20 * 0.05 = 1
        # # 发现 2 个显著，可能一个是假阳性
        # advice = advise_researcher(n_tests, n_significant, fwer)
        #
        # # 应建议谨慎
        # assert '谨慎' in advice or '校正' in advice or '验证' in advice
        pytest.skip("solution.py not yet created")

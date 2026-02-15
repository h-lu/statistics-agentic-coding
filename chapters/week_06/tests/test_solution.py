"""
Test suite for Week 06 solution.py

Tests cover:
1. p 值理解与计算
2. t 检验（单样本、双样本、配对）
3. 卡方检验
4. 效应量计算（Cohen's d 等）
5. 前提假设检查（正态性、方差齐性）
6. AI 结论审查

注意：由于 solution.py 尚未创建，这些测试定义了预期的接口规范。
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import stats

# Import functions from solution module
import sys
from pathlib import Path

# Add starter_code to path
sys.path.insert(0, str(Path(__file__).parent.parent / "starter_code"))

# Solution module will be imported once created
# For now, we define the tests with expected function signatures


# =============================================================================
# 1. p 值理解测试
# =============================================================================

class TestPValueInterpretation:
    """Tests for p value interpretation functions."""

    def test_interpret_p_value_significant(self):
        """
        测试 p < 0.05 时正确拒绝原假设

        当 p 值小于显著性水平时，应返回拒绝原假设的结论
        """
        # TODO: Uncomment when solution.py is created
        # result = interpret_p_value(p_value=0.02, alpha=0.05)
        # assert result['reject_null'] is True
        # assert '拒绝原假设' in result['conclusion']
        pytest.skip("solution.py not yet created")

    def test_interpret_p_value_not_significant(self):
        """
        测试 p >= 0.05 时无法拒绝原假设

        当 p 值大于等于显著性水平时，应返回无法拒绝原假设的结论
        注意：不是说"接受原假设"
        """
        # TODO: Uncomment when solution.py is created
        # result = interpret_p_value(p_value=0.15, alpha=0.05)
        # assert result['reject_null'] is False
        # assert '无法拒绝' in result['conclusion']
        # assert '接受' not in result['conclusion']
        pytest.skip("solution.py not yet created")

    def test_interpret_p_value_boundary(self):
        """
        测试 p ≈ 0.05 时的边界判断

        p = 0.05 正好在边界上，应该说明这是任意设定
        """
        # TODO: Uncomment when solution.py is created
        # result = interpret_p_value(p_value=0.05, alpha=0.05)
        # assert result['reject_null'] is False
        # assert result['on_boundary'] is True
        pytest.skip("solution.py not yet created")

    def test_interpret_p_value_multiple_alpha(self):
        """
        测试不同显著性水平下的解释

        同一个 p 值在不同 alpha 下可能有不同的结论
        """
        # TODO: Uncomment when solution.py is created
        # p_value = 0.03
        # result_05 = interpret_p_value(p_value, alpha=0.05)
        # result_01 = interpret_p_value(p_value, alpha=0.01)
        # assert result_05['reject_null'] is True
        # assert result_01['reject_null'] is False
        pytest.skip("solution.py not yet created")


# =============================================================================
# 2. t 检验测试
# =============================================================================

class TestTwoSampleTTest:
    """Tests for two sample t test functions."""

    def test_t_test_significant_difference(self, normal_two_groups):
        """
        测试有显著差异的两组数据

        两组均值有真实差异时，t 检验应检测到
        """
        group_a, group_b = normal_two_groups
        # TODO: Uncomment when solution.py is created
        # result = two_sample_t_test(group_a, group_b)
        # assert result['p_value'] < 0.05
        # assert result['reject_null'] is True
        # assert 't_statistic' in result
        pytest.skip("solution.py not yet created")

    def test_t_test_no_difference(self, null_hypothesis_data):
        """
        测试无差异的两组数据

        两组来自同一分布时，p 值应均匀分布在 [0, 1]
        """
        group_a, group_b = null_hypothesis_data
        # TODO: Uncomment when solution.py is created
        # result = two_sample_t_test(group_a, group_b)
        # assert result['p_value'] > 0.05
        # assert result['reject_null'] is False
        pytest.skip("solution.py not yet created")

    def test_t_test_small_sample(self, small_sample_groups):
        """
        测试小样本情况

        小样本时 t 检验仍应工作，但功效较低
        """
        group_a, group_b = small_sample_groups
        # TODO: Uncomment when solution.py is created
        # result = two_sample_t_test(group_a, group_b)
        # assert 'p_value' in result
        # assert 'degrees_of_freedom' in result
        # assert result['degrees_of_freedom'] == len(group_a) + len(group_b) - 2
        pytest.skip("solution.py not yet created")

    def test_t_test_empty_data(self, empty_data):
        """
        测试空数据应报错

        空输入应引发 ValueError
        """
        # TODO: Uncomment when solution.py is created
        # with pytest.raises(ValueError, match="empty"):
        #     two_sample_t_test(empty_data, empty_data)
        pytest.skip("solution.py not yet created")

    def test_t_test_single_value(self, single_value_data):
        """
        测试单值数据

        单个值无法计算标准差，应返回 None 或报错
        """
        # TODO: Uncomment when solution.py is created
        # result = two_sample_t_test(single_value_data, single_value_data)
        # assert result is None or 'error' in result
        pytest.skip("solution.py not yet created")


class TestProportionTest:
    """Tests for proportion test functions (z-test for proportions)."""

    def test_proportion_test_significant(self, binary_conversion_data):
        """
        测试比例检验检测显著差异

        A 渠道 12% vs B 渠道 9% 应该有显著差异
        """
        conversions_a, conversions_b = binary_conversion_data
        # TODO: Uncomment when solution.py is created
        # result = proportion_test(conversions_a, conversions_b)
        # assert result['p_value'] < 0.05
        # assert result['rate_a'] == pytest.approx(0.12, rel=0.01)
        # assert result['rate_b'] == pytest.approx(0.09, rel=0.01)
        pytest.skip("solution.py not yet created")

    def test_proportion_test_same_rate(self):
        """
        测试相同比例

        相同比例时 p 值应很大
        """
        np.random.seed(42)
        both = np.array([1] * 100 + [0] * 900)
        # TODO: Uncomment when solution.py is created
        # result = proportion_test(both, both)
        # assert result['p_value'] == pytest.approx(1.0, abs=0.01)
        # assert result['difference'] == 0
        pytest.skip("solution.py not yet created")

    def test_proportion_test_empty_data(self, empty_data):
        """
        测试空数据应报错
        """
        # TODO: Uncomment when solution.py is created
        # with pytest.raises(ValueError, match="empty"):
        #     proportion_test(empty_data, empty_data)
        pytest.skip("solution.py not yet created")


class TestPairedTTest:
    """Tests for paired t test functions."""

    def test_paired_t_test_significant(self, paired_data):
        """
        测试配对 t 检验检测显著变化

        配对数据中"治疗后"比"治疗前"显著提高
        """
        before, after = paired_data
        # TODO: Uncomment when solution.py is created
        # result = paired_t_test(before, after)
        # assert result['p_value'] < 0.05
        # assert result['mean_difference'] > 0
        # assert result['reject_null'] is True
        pytest.skip("solution.py not yet created")

    def test_paired_vs_independent(self, normal_two_groups):
        """
        测试配对检验与独立检验的区别

        配对检验应考虑配对内的相关性
        """
        group_a, group_b = normal_two_groups
        # TODO: Uncomment when solution.py is created
        # independent_result = two_sample_t_test(group_a, group_b)
        # paired_result = paired_t_test(group_a, group_b)
        # 结果应该不同（虽然在这个未配对的数据上不应使用配对检验）
        pytest.skip("solution.py not yet created")

    def test_paired_t_test_mismatched_lengths(self):
        """
        测试长度不匹配的配对数据应报错
        """
        # TODO: Uncomment when solution.py is created
        # with pytest.raises(ValueError, match="same length"):
        #     paired_t_test(np.array([1, 2, 3]), np.array([1, 2]))
        pytest.skip("solution.py not yet created")


# =============================================================================
# 3. 卡方检验测试
# =============================================================================

class TestChiSquareTest:
    """Tests for chi-square test functions."""

    def test_chi_square_independent(self, contingency_table_independent):
        """
        测试独立变量的卡方检验

        独立变量应得到较大 p 值
        """
        # TODO: Uncomment when solution.py is created
        # result = chi_square_test(contingency_table_independent)
        # assert result['p_value'] > 0.05
        # assert result['reject_null'] is False
        # assert result['conclusion'] contains '独立'
        pytest.skip("solution.py not yet created")

    def test_chi_square_dependent(self, contingency_table_dependent):
        """
        测试有关联变量的卡方检验

        有关联的变量应得到较小 p 值
        """
        # TODO: Uncomment when solution.py is created
        # result = chi_square_test(contingency_table_dependent)
        # assert result['p_value'] < 0.05
        # assert result['reject_null'] is True
        pytest.skip("solution.py not yet created")

    def test_chi_square_expected_frequencies(self, contingency_table_independent):
        """
        测试期望频数的计算

        期望频数应基于独立假设
        """
        # TODO: Uncomment when solution.py is created
        # result = chi_square_test(contingency_table_independent)
        # assert 'expected_frequencies' in result
        # assert result['expected_frequencies'].shape == contingency_table_independent.shape
        pytest.skip("solution.py not yet created")


# =============================================================================
# 4. 效应量测试
# =============================================================================

class TestCohensD:
    """Tests for Cohen's d effect size calculation."""

    def test_cohens_d_large_effect(self, large_effect_data):
        """
        测试大效应量计算

        Cohen's d > 0.8 应被解释为大效应
        """
        group_a, group_b = large_effect_data
        # TODO: Uncomment when solution.py is created
        # d = cohens_d(group_a, group_b)
        # assert d > 0.8
        # interpretation = interpret_cohens_d(d)
        # assert interpretation == 'large'
        pytest.skip("solution.py not yet created")

    def test_cohens_d_medium_effect(self, medium_effect_data):
        """
        测试中等效应量计算

        0.2 <= Cohen's d < 0.8 应被解释为中等效应
        """
        group_a, group_b = medium_effect_data
        # TODO: Uncomment when solution.py is created
        # d = cohens_d(group_a, group_b)
        # assert 0.2 <= d < 0.8
        # interpretation = interpret_cohens_d(d)
        # assert interpretation == 'medium'
        pytest.skip("solution.py not yet created")

    def test_cohens_d_small_effect(self, small_effect_data):
        """
        测试小效应量计算

        Cohen's d < 0.2 应被解释为小效应
        """
        group_a, group_b = small_effect_data
        # TODO: Uncomment when solution.py is created
        # d = cohens_d(group_a, group_b)
        # assert d < 0.2
        # interpretation = interpret_cohens_d(d)
        # assert interpretation == 'small'
        pytest.skip("solution.py not yet created")

    def test_cohens_d_no_effect(self, no_effect_data):
        """
        测试无效应情况

        相同分布的两组 Cohen's d 应接近 0
        """
        group_a, group_b = no_effect_data
        # TODO: Uncomment when solution.py is created
        # d = cohens_d(group_a, group_b)
        # assert d == pytest.approx(0, abs=0.1)
        pytest.skip("solution.py not yet created")

    def test_cohens_d_direction(self, normal_two_groups):
        """
        测试效应量的方向

        group_a > group_b 时 d 应为正
        """
        group_a, group_b = normal_two_groups
        # TODO: Uncomment when solution.py is created
        # d = cohens_d(group_a, group_b)
        # d_reversed = cohens_d(group_b, group_a)
        # assert d > 0 > d_reversed
        # assert abs(d) == abs(d_reversed)
        pytest.skip("solution.py not yet created")

    def test_cohens_d_empty_data(self, empty_data):
        """
        测试空数据应报错
        """
        # TODO: Uncomment when solution.py is created
        # with pytest.raises(ValueError, match="empty"):
        #     cohens_d(empty_data, empty_data)
        pytest.skip("solution.py not yet created")


class TestRiskDifference:
    """Tests for risk difference and related effect sizes."""

    def test_risk_difference_calculation(self, binary_conversion_data):
        """
        测试风险差计算

        对于转化率数据，风险差 = 12% - 9% = 3%
        """
        conversions_a, conversions_b = binary_conversion_data
        # TODO: Uncomment when solution.py is created
        # rd = risk_difference(conversions_a, conversions_b)
        # assert rd == pytest.approx(0.03, rel=0.1)
        pytest.skip("solution.py not yet created")

    def test_risk_ratio_calculation(self, binary_conversion_data):
        """
        测试风险比计算

        RR = 12% / 9% ≈ 1.33
        """
        conversions_a, conversions_b = binary_conversion_data
        # TODO: Uncomment when solution.py is created
        # rr = risk_ratio(conversions_a, conversions_b)
        # assert rr == pytest.approx(1.33, rel=0.1)
        pytest.skip("solution.py not yet created")


# =============================================================================
# 5. 前提假设检查测试
# =============================================================================

class TestNormalityCheck:
    """Tests for normality assumption checking."""

    def test_shapiro_wilk_normal_data(self, normal_data):
        """
        测试正态数据的 Shapiro-Wilk 检验

        正态数据应得到 p > 0.05
        """
        # TODO: Uncomment when solution.py is created
        # result = check_normality(normal_data)
        # assert result['test'] == 'Shapiro-Wilk'
        # assert result['p_value'] > 0.05
        # assert result['is_normal'] is True
        pytest.skip("solution.py not yet created")

    def test_shapiro_wilk_skewed_data(self, skewed_data):
        """
        测试偏态数据的 Shapiro-Wilk 检验

        偏态数据应得到 p < 0.05
        """
        # TODO: Uncomment when solution.py is created
        # result = check_normality(skewed_data)
        # assert result['p_value'] < 0.05
        # assert result['is_normal'] is False
        pytest.skip("solution.py not yet created")

    def test_shapiro_wilk_constant_data(self, constant_data):
        """
        测试常数数据的正态性检验

        常数数据 technically 是正态的（方差为0）
        """
        # TODO: Uncomment when solution.py is created
        # result = check_normality(constant_data)
        # assert result['is_normal'] is True  # 或者返回特殊标记
        pytest.skip("solution.py not yet created")

    def test_shapiro_wilk_small_sample(self, two_values_data):
        """
        测试极小样本的正态性检验

        Shapiro-Wilk 最少需要 3 个样本
        """
        # TODO: Uncomment when solution.py is created
        # result = check_normality(two_values_data)
        # 应返回警告或使用其他方法
        pytest.skip("solution.py not yet created")

    def test_shapiro_wilk_empty_data(self, empty_data):
        """
        测试空数据应报错
        """
        # TODO: Uncomment when solution.py is created
        # with pytest.raises(ValueError, match="empty"):
        #     check_normality(empty_data)
        pytest.skip("solution.py not yet created")


class TestVarianceHomogeneityCheck:
    """Tests for variance homogeneity checking."""

    def test_levene_equal_variance(self, equal_variance_groups):
        """
        测试方差相等数据的 Levene 检验

        方差相等应得到 p > 0.05
        """
        group_a, group_b = equal_variance_groups
        # TODO: Uncomment when solution.py is created
        # result = check_variance_homogeneity(group_a, group_b)
        # assert result['test'] == 'Levene'
        # assert result['p_value'] > 0.05
        # assert result['equal_variance'] is True
        pytest.skip("solution.py not yet created")

    def test_levene_unequal_variance(self, unequal_variance_groups):
        """
        测试方差不等数据的 Levene 检验

        方差不等应得到 p < 0.05
        """
        group_a, group_b = unequal_variance_groups
        # TODO: Uncomment when solution.py is created
        # result = check_variance_homogeneity(group_a, group_b)
        # assert result['p_value'] < 0.05
        # assert result['equal_variance'] is False
        pytest.skip("solution.py not yet created")

    def test_levene_constant_data(self, constant_data):
        """
        测试常数数据的方差齐性检验

        常数数据的方差为 0，应返回特殊结果
        """
        # TODO: Uncomment when solution.py is created
        # result = check_variance_homogeneity(constant_data, constant_data)
        # assert result['equal_variance'] is True
        pytest.skip("solution.py not yet created")


class TestAssumptionCheckingIntegration:
    """Tests for integrated assumption checking."""

    def test_choose_test_based_on_assumptions(self, normal_data):
        """
        测试基于假设选择合适的检验

        正态 + 方差齐 -> Student's t
        正态 + 方差不齐 -> Welch's t
        非正态 -> Mann-Whitney U
        """
        # TODO: Uncomment when solution.py is created
        # group_a = normal_data[:100]
        # group_b = normal_data[100:]
        # result = choose_test_auto(group_a, group_b)
        # assert result['test_used'] in ["Student's t", "Welch's t", "Mann-Whitney U"]
        # assert result['assumptions_checked'] is True
        pytest.skip("solution.py not yet created")

    def test_assumption_report_includes_warnings(self, skewed_data):
        """
        测试假设检查报告应包含警告

        当假设不满足时，报告应明确说明
        """
        # TODO: Uncomment when solution.py is created
        # group_a = skewed_data[:100]
        # group_b = skewed_data[100:]
        # result = choose_test_auto(group_a, group_b)
        # assert 'warning' in result or 'assumptions' in result
        pytest.skip("solution.py not yet created")


# =============================================================================
# 6. AI 结论审查测试
# =============================================================================

class TestAIReportReview:
    """Tests for AI-generated statistical report review."""

    def test_review_good_report(self, ai_good_report):
        """
        测试审查合格的 AI 报告

        合格报告应通过所有检查项
        """
        # TODO: Uncomment when solution.py is created
        # review = review_ai_report(ai_good_report)
        # assert review['passed'] is True
        # assert len(review['issues']) == 0
        # assert review['overall_rating'] == 'good'
        pytest.skip("solution.py not yet created")

    def test_review_bad_report_missing_ci(self, ai_bad_report):
        """
        测试审查缺少置信区间的报告

        应标记为问题
        """
        # TODO: Uncomment when solution.py is created
        # review = review_ai_report(ai_bad_report)
        # assert review['passed'] is False
        # assert any('置信区间' in issue or 'CI' in issue for issue in review['issues'])
        pytest.skip("solution.py not yet created")

    def test_review_bad_report_missing_effect_size(self, ai_bad_report):
        """
        测试审查缺少效应量的报告

        应标记为问题
        """
        # TODO: Uncomment when solution.py is created
        # review = review_ai_report(ai_bad_report)
        # assert review['passed'] is False
        # assert any('效应量' in issue or 'effect size' in issue for issue in review['issues'])
        pytest.skip("solution.py not yet created")

    def test_review_bad_report_no_assumption_check(self, ai_bad_report):
        """
        测试审查未检查假设的报告

        应标记为严重问题
        """
        # TODO: Uncomment when solution.py is created
        # review = review_ai_report(ai_bad_report)
        # assert review['passed'] is False
        # assert any('假设' in issue or 'assumption' in issue for issue in review['issues'])
        pytest.skip("solution.py not yet created")

    def test_review_bad_report_overinterpretation(self, ai_bad_report):
        """
        测试审查过度解读的报告

        "建议全面切换" 是过度解读，应标记
        """
        # TODO: Uncomment when solution.py is created
        # review = review_ai_report(ai_bad_report)
        # assert any('过度' in issue or '解读' in issue for issue in review['issues'])
        pytest.skip("solution.py not yet created")

    def test_review_wrong_test_for_data_type(self, ai_bad_report):
        """
        测试审查对二元数据使用 t 检验

        应建议使用比例检验
        """
        # TODO: Uncomment when solution.py is created
        # review = review_ai_report(ai_bad_report)
        # assert any('比例' in issue or 'proportion' in issue for issue in review['issues'])
        pytest.skip("solution.py not yet created")


class TestMultipleComparisonCorrection:
    """Tests for multiple comparison correction."""

    def test_bonferroni_correction(self, multiple_hypotheses_results):
        """
        测试 Bonferroni 校正

        5 个假设，alpha = 0.05 -> 校正后 alpha = 0.01
        """
        # TODO: Uncomment when solution.py is created
        # corrected = bonferroni_correction(multiple_hypotheses_results, alpha=0.05)
        # assert corrected['corrected_alpha'] == pytest.approx(0.01, abs=0.001)
        # 原本显著的 p=0.023 在校正后不再显著
        # assert corrected['results'][0]['still_significant'] is False
        pytest.skip("solution.py not yet created")

    def test_false_discovery_rate(self, multiple_hypotheses_results):
        """
        测试 FDR (Benjamini-Hochberg) 校正

        FDR 比 Bonferroni 更保守
        """
        # TODO: Uncomment when solution.py is created
        # corrected = fdr_correction(multiple_hypotheses_results, q=0.05)
        # assert 'adjusted_p_values' in corrected
        pytest.skip("solution.py not yet created")

    def test_calculate_family_wise_error_rate(self):
        """
        测试计算家族错误率

        P(at least one FP) = 1 - (1 - alpha)^n
        """
        # TODO: Uncomment when solution.py is created
        # fwer = calculate_family_wise_error_rate(n_hypotheses=5, alpha=0.05)
        # expected = 1 - (1 - 0.05) ** 5
        # assert fwer == pytest.approx(expected, rel=0.01)
        pytest.skip("solution.py not yet created")


# =============================================================================
# 7. 综合测试：完整检验流程
# =============================================================================

class TestCompleteHypothesisTestWorkflow:
    """Tests for complete hypothesis testing workflow."""

    def test_complete_two_group_test(self, normal_two_groups):
        """
        测试完整的两组比较流程

        应包含：
        1. 描述统计
        2. 假设检查
        3. 检验执行
        4. 效应量计算
        5. 置信区间
        6. 结论解释
        """
        group_a, group_b = normal_two_groups
        # TODO: Uncomment when solution.py is created
        # result = complete_two_group_test(group_a, group_b, alpha=0.05)
        # assert 'descriptive_stats' in result
        # assert 'assumption_checks' in result
        # assert 'test_result' in result
        # assert 'effect_size' in result
        # assert 'confidence_interval' in result
        # assert 'conclusion' in result
        pytest.skip("solution.py not yet created")

    def test_generate_test_report(self, normal_two_groups):
        """
        测试生成 Markdown 格式的检验报告

        报告应可读且包含所有关键信息
        """
        group_a, group_b = normal_two_groups
        # TODO: Uncomment when solution.py is created
        # report = generate_hypothesis_test_report(group_a, group_b, group_names=['A', 'B'], value_name='score')
        # assert '##' in report  # Markdown headers
        # assert 'p 值' in report or 'p-value' in report
        # assert '效应量' in report or 'effect size' in report
        # assert '置信区间' in report or 'CI' in report
        pytest.skip("solution.py not yet created")

"""
Week 07 测试：多组比较与多重比较校正

测试功能：
- ANOVA 的 F 统计量计算
- η²（eta-squared）效应量计算
- Bonferroni 校正的阈值计算
- FWER 计算
- Tukey HSD 结果解释
- 卡方检验的 Cramér's V 计算
- AI 生成报告的审查
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

# 导入需要测试的函数（这些函数需要在 solution.py 中实现）
# 注意：如果 solution.py 还未实现，这些测试会失败
try:
    from solution import (
        calculate_f_statistic,
        calculate_eta_squared,
        calculate_fwer,
        bonferroni_correction,
        interpret_tukey_hsd,
        calculate_cramers_v,
        chi_square_test,
        review_anova_report,
        anova_test,
    )
except ImportError:
    pytest.skip("solution.py not implemented yet", allow_module_level=True)


# =============================================================================
# 测试 F 统计量计算
# =============================================================================

class TestFStatistic:
    """测试 F 统计量计算"""

    def test_calculate_f_statistic_basic(self, sample_three_groups):
        """测试基础 F 统计量计算"""
        groups = list(sample_three_groups.values())
        result = calculate_f_statistic(groups)

        assert 'f_statistic' in result
        assert 'df_between' in result
        assert 'df_within' in result
        assert 'p_value' in result
        assert isinstance(result['f_statistic'], (int, float))

    def test_calculate_f_statistic_no_difference(self, sample_groups_no_difference):
        """测试无差异时 F 统计量接近 1"""
        groups = list(sample_groups_no_difference.values())
        result = calculate_f_statistic(groups)

        # 无差异时，F 应该接近 1
        assert result['f_statistic'] > 0
        # p 值应该较大（不显著）
        assert result['p_value'] > 0.05

    def test_calculate_f_statistic_large_effect(self, sample_groups_large_effect):
        """测试大效应时 F 统计量较大"""
        groups = list(sample_groups_large_effect.values())
        result = calculate_f_statistic(groups)

        # 大效应时，F 应该较大
        assert result['f_statistic'] > 1
        # p 值应该很小（显著）
        assert result['p_value'] < 0.05

    def test_calculate_f_statistic_five_groups(self, sample_five_groups_with_effect):
        """测试五组数据的 F 统计量"""
        groups = list(sample_five_groups_with_effect.values())
        result = calculate_f_statistic(groups)

        assert result['df_between'] == 4  # k - 1 = 5 - 1
        assert result['df_within'] == 495  # N - k = 500 - 5
        assert result['f_statistic'] > 0

    def test_calculate_f_statistic_equal_to_scipy(self, sample_three_groups):
        """测试 F 统计量与 scipy 结果一致"""
        groups = list(sample_three_groups.values())
        result = calculate_f_statistic(groups)

        # 与 scipy 结果比较
        f_scipy, p_scipy = stats.f_oneway(*groups)

        assert abs(result['f_statistic'] - f_scipy) < 0.01
        assert abs(result['p_value'] - p_scipy) < 0.01


# =============================================================================
# 测试 η²（eta-squared）效应量
# =============================================================================

class TestEtaSquared:
    """测试 η² 效应量计算"""

    def test_calculate_eta_squared_basic(self, sample_three_groups):
        """测试基础 η² 计算"""
        groups = list(sample_three_groups.values())
        result = calculate_eta_squared(groups)

        assert 'eta_squared' in result
        assert 'ss_between' in result
        assert 'ss_within' in result
        assert 'ss_total' in result
        assert 0 <= result['eta_squared'] <= 1

    def test_calculate_eta_squared_verification(self, sample_three_groups):
        """测试 η² = SSB / SST"""
        groups = list(sample_three_groups.values())
        result = calculate_eta_squared(groups)

        # 验证 η² = SSB / SST
        expected_eta2 = result['ss_between'] / result['ss_total']
        assert abs(result['eta_squared'] - expected_eta2) < 0.001

    def test_calculate_eta_squared_small_effect(self, sample_groups_no_difference):
        """测试小效应（η² < 0.01）"""
        groups = list(sample_groups_no_difference.values())
        result = calculate_eta_squared(groups)

        # 无差异时，效应量应该很小
        assert result['eta_squared'] < 0.1

    def test_calculate_eta_squared_interpretation(self, sample_five_groups_with_effect):
        """测试效应量解释"""
        groups = list(sample_five_groups_with_effect.values())
        result = calculate_eta_squared(groups)

        # 应该包含解释字段
        assert 'interpretation' in result or 'category' in result

        if 'category' in result:
            # 检查分类是否合理
            valid_categories = ['negligible', 'small', 'medium', 'large']
            assert result['category'] in valid_categories


# =============================================================================
# 测试 FWER（Family-wise Error Rate）
# =============================================================================

class TestFWER:
    """测试 FWER 计算"""

    def test_calculate_fwer_single_test(self):
        """测试单次检验的 FWER = α"""
        result = calculate_fwer(alpha=0.05, n_tests=1)
        assert abs(result - 0.05) < 0.001

    def test_calculate_fwer_ten_tests(self):
        """测试 10 次检验的 FWER ≈ 0.401"""
        result = calculate_fwer(alpha=0.05, n_tests=10)
        # FWER = 1 - (1 - 0.05)^10 ≈ 0.401
        expected = 1 - (1 - 0.05) ** 10
        assert abs(result - expected) < 0.001

    def test_calculate_fwer_twenty_tests(self):
        """测试 20 次检验的 FWER > 0.64"""
        result = calculate_fwer(alpha=0.05, n_tests=20)
        # FWER = 1 - (1 - 0.05)^20 ≈ 0.641
        expected = 1 - (1 - 0.05) ** 20
        assert abs(result - expected) < 0.001

    def test_calculate_fwer_formula(self):
        """测试 FWER 公式：FWER = 1 - (1-α)^m"""
        for n_tests in [1, 5, 10, 20, 50]:
            result = calculate_fwer(alpha=0.05, n_tests=n_tests)
            expected = 1 - (1 - 0.05) ** n_tests
            assert abs(result - expected) < 0.001

    def test_calculate_fwer_increases_with_tests(self):
        """测试 FWER 随检验次数增加"""
        fwer_5 = calculate_fwer(alpha=0.05, n_tests=5)
        fwer_10 = calculate_fwer(alpha=0.05, n_tests=10)
        fwer_20 = calculate_fwer(alpha=0.05, n_tests=20)

        assert fwer_5 < fwer_10 < fwer_20


# =============================================================================
# 测试 Bonferroni 校正
# =============================================================================

class TestBonferroniCorrection:
    """测试 Bonferroni 校正"""

    def test_bonferroni_correction_basic(self):
        """测试基础 Bonferroni 校正"""
        alpha = 0.05
        n_tests = 10
        result = bonferroni_correction(alpha=alpha, n_tests=n_tests)

        assert 'corrected_alpha' in result
        assert result['corrected_alpha'] == alpha / n_tests
        assert abs(result['corrected_alpha'] - 0.005) < 0.001

    def test_bonferroni_correction_single_test(self):
        """测试单次检验时 α' = α"""
        result = bonferroni_correction(alpha=0.05, n_tests=1)
        assert abs(result['corrected_alpha'] - 0.05) < 0.001

    def test_bonferroni_correction_many_tests(self):
        """测试多次检验时阈值降低"""
        result_10 = bonferroni_correction(alpha=0.05, n_tests=10)
        result_100 = bonferroni_correction(alpha=0.05, n_tests=100)

        assert result_10['corrected_alpha'] == 0.005
        assert result_100['corrected_alpha'] == 0.0005

    def test_bonferroni_correction_p_values(self, multiple_test_p_values):
        """测试对 p 值列表进行 Bonferroni 校正"""
        result = bonferroni_correction(
            alpha=0.05,
            n_tests=len(multiple_test_p_values),
            p_values=multiple_test_p_values
        )

        assert 'corrected_p_values' in result
        assert 'significant_after_correction' in result

        # 校正后的 p 值应该 >= 原始 p 值
        for orig, corrected in zip(multiple_test_p_values, result['corrected_p_values']):
            assert corrected >= orig

    def test_bonferroni_reduces_significant(self, multiple_test_p_values):
        """测试 Bonferroni 校正后显著结果减少"""
        orig_sig_count = sum(1 for p in multiple_test_p_values if p < 0.05)

        result = bonferroni_correction(
            alpha=0.05,
            n_tests=len(multiple_test_p_values),
            p_values=multiple_test_p_values
        )

        corrected_sig_count = result['significant_after_correction']

        # 校正后显著数量应该 <= 原始显著数量
        assert corrected_sig_count <= orig_sig_count


# =============================================================================
# 测试 Tukey HSD 结果解释
# =============================================================================

class TestTukeyHSD:
    """测试 Tukey HSD 结果解释"""

    def test_interpret_tukey_hsd_basic(self, tukey_hsd_results):
        """测试基础 Tukey HSD 结果解释"""
        result = interpret_tukey_hsd(tukey_hsd_results)

        assert 'n_comparisons' in result
        assert 'n_significant' in result
        assert 'significant_pairs' in result
        assert result['n_comparisons'] == len(tukey_hsd_results)

    def test_interpret_tukey_hsd_identify_significant(self, tukey_hsd_results):
        """测试识别显著的城市对"""
        result = interpret_tukey_hsd(tukey_hsd_results)

        # 根据模拟数据，应该有 4 对显著
        assert result['n_significant'] == 4
        assert len(result['significant_pairs']) == 4

    def test_interpret_tukey_hsd_pair_list(self, tukey_hsd_results):
        """测试显著对列表"""
        result = interpret_tukey_hsd(tukey_hsd_results)

        for pair in result['significant_pairs']:
            assert 'group1' in pair or 'pair' in pair
            assert 'meandiff' in pair
            assert 'p_adj' in pair

    def test_interpret_tukey_hsd_summary(self, tukey_hsd_results):
        """测试生成摘要"""
        result = interpret_tukey_hsd(tukey_hsd_results)

        assert 'summary' in result or 'interpretation' in result


# =============================================================================
# 测试卡方检验与 Cramér's V
# =============================================================================

class TestChiSquare:
    """测试卡方检验"""

    def test_chi_square_test_basic(self, contingency_table_associated):
        """测试基础卡方检验"""
        result = chi_square_test(contingency_table_associated)

        assert 'chi2' in result
        assert 'p_value' in result
        assert 'dof' in result
        assert 'cramers_v' in result
        assert isinstance(result['chi2'], (int, float))
        assert result['chi2'] >= 0

    def test_chi_square_test_independent(self, contingency_table_independent):
        """测试独立变量的卡方检验（p 值应较大）"""
        result = chi_square_test(contingency_table_independent)

        # 独立变量应该 p 值较大
        assert result['p_value'] > 0.05
        # Cramér's V 应该很小
        assert result['cramers_v'] < 0.3

    def test_chi_square_test_associated(self, contingency_table_associated):
        """测试有关联变量的卡方检验"""
        result = chi_square_test(contingency_table_associated)

        assert 'cramers_v' in result
        assert 0 <= result['cramers_v'] <= 1

    def test_calculate_cramers_v_basic(self, contingency_table_associated):
        """测试基础 Cramér's V 计算"""
        observed = contingency_table_associated.values
        chi2, p, dof, _ = stats.chi2_contingency(observed)
        n = observed.sum()
        min_dim = min(observed.shape)

        result = calculate_cramers_v(chi2, n, min_dim)

        assert 0 <= result['cramers_v'] <= 1
        assert 'interpretation' in result or 'category' in result

    def test_calculate_cramers_v_formula(self):
        """测试 Cramér's V 公式"""
        # V = sqrt(χ² / (n * (min_dim - 1)))
        chi2 = 10.0
        n = 100
        min_dim = 3

        result = calculate_cramers_v(chi2, n, min_dim)
        expected = np.sqrt(chi2 / (n * (min_dim - 1)))

        assert abs(result['cramers_v'] - expected) < 0.001

    def test_calculate_cramers_v_interpretation(self):
        """测试 Cramér's V 解释"""
        # 小效应
        result_small = calculate_cramers_v(chi2=5, n=100, min_dim=3)
        assert result_small['cramers_v'] < 0.3

        # 中等效应
        result_medium = calculate_cramers_v(chi2=20, n=100, min_dim=3)
        assert 'interpretation' in result_medium or 'category' in result_medium


# =============================================================================
# 测试 AI 报告审查
# =============================================================================

class TestAnovaReportReview:
    """测试 ANOVA 报告审查"""

    def test_review_good_report(self, good_anova_report):
        """测试审查合格的报告"""
        result = review_anova_report(good_anova_report)

        assert 'has_issues' in result
        assert 'issues' in result
        assert 'issue_count' in result
        # 合格报告应该问题较少或没有
        assert result['issue_count'] <= 2

    def test_review_report_overinterpretation(self, bad_anova_report_overinterpretation):
        """测试识别 ANOVA 过度解释"""
        result = review_anova_report(bad_anova_report_overinterpretation)

        assert result['has_issues'] is True
        assert result['issue_count'] >= 2

        # 应该识别出"ANOVA 过度解释"问题
        issue_descriptions = [issue.get('问题', '') for issue in result['issues']]
        assert any('过度解释' in desc or '具体哪几对' in desc
                   for desc in issue_descriptions)

    def test_review_report_no_correction(self, bad_anova_report_no_correction):
        """测试识别未校正多重比较"""
        result = review_anova_report(bad_anova_report_no_correction)

        assert result['has_issues'] is True

        # 应该识别出"未校正多重比较"问题
        issue_descriptions = [issue.get('问题', '') for issue in result['issues']]
        assert any('校正' in desc or '多重比较' in desc
                   for desc in issue_descriptions)

    def test_review_report_causation_error(self, bad_chisquare_report_causation):
        """测试识别相关误写成因果"""
        result = review_anova_report(bad_chisquare_report_causation)

        assert result['has_issues'] is True

        # 应该识别出"相关误写成因果"问题
        issue_descriptions = [issue.get('问题', '') for issue in result['issues']]
        assert any('因果' in desc or '相关' in desc
                   for desc in issue_descriptions)

    def test_review_report_missing_effect_size(self):
        """测试识别缺少效应量"""
        bad_report = """
        ANOVA 结果：
        F=8.52, p=0.002，拒绝 H0。
        """
        result = review_anova_report(bad_report)

        # 应该识别出"缺少效应量"问题
        issue_descriptions = [issue.get('问题', '') for issue in result['issues']]
        assert any('效应量' in desc or 'η²' in desc
                   for desc in issue_descriptions)

    def test_review_report_missing_assumptions(self):
        """测试识别未验证前提假设"""
        bad_report = """
        ANOVA 结果：
        F=8.52, p=0.002，拒绝 H0。
        上海和深圳显著高于其他城市。
        """
        result = review_anova_report(bad_report)

        # 应该识别出"未验证前提假设"问题
        issue_descriptions = [issue.get('问题', '') for issue in result['issues']]
        assert any('正态性' in desc or '方差齐性' in desc or '假设' in desc
                   for desc in issue_descriptions)


# =============================================================================
# 测试 ANOVA 完整流程
# =============================================================================

class TestAnovaComplete:
    """测试完整的 ANOVA 流程"""

    def test_anova_test_basic(self, sample_three_groups):
        """测试基础 ANOVA 流程"""
        groups = list(sample_three_groups.values())
        result = anova_test(groups, check_assumptions=False)

        assert 'f_statistic' in result
        assert 'p_value' in result
        assert 'eta_squared' in result
        assert 'decision' in result

    def test_anova_test_with_assumptions(self, sample_three_groups):
        """测试带前提假设检查的 ANOVA"""
        groups = list(sample_three_groups.values())
        result = anova_test(groups, check_assumptions=True)

        assert 'assumptions' in result
        assert 'normality' in result['assumptions']
        assert 'variance_homogeneity' in result['assumptions']

    def test_anova_test_decision(self, sample_groups_large_effect):
        """测试 ANOVA 决策（大效应应拒绝 H0）"""
        groups = list(sample_groups_large_effect.values())
        result = anova_test(groups, check_assumptions=False)

        # 大效应应该拒绝 H0
        assert result['decision'] == 'reject_H0' or result['p_value'] < 0.05

    def test_anova_test_no_difference(self, sample_groups_no_difference):
        """测试 ANOVA 决策（无差异应保留 H0）"""
        groups = list(sample_groups_no_difference.values())
        result = anova_test(groups, check_assumptions=False)

        # 无差异应该保留 H0
        assert result['decision'] == 'retain_H0' or result['p_value'] >= 0.05


# =============================================================================
# 边界情况测试
# =============================================================================

class TestEdgeCases:
    """测试边界情况"""

    def test_f_statistic_with_constant_groups(self, constant_groups):
        """测试常数组数据的 F 统计量"""
        groups = list(constant_groups.values())
        result = calculate_f_statistic(groups)

        # 常数组应该 F = 0、接近 0 或 NaN（方差为 0 导致除零）
        # 这是预期行为：没有方差就无法计算 F 统计量
        assert 'f_statistic' in result
        # 可能是 NaN、inf 或 0
        try:
            # 如果是有限数值，应该 >= 0
            if np.isfinite(result['f_statistic']):
                assert result['f_statistic'] >= 0
        except (TypeError, ValueError):
            # NaN 或 inf 也是可接受的结果
            assert True

    def test_f_statistic_with_tiny_groups(self, tiny_groups):
        """测试极小样本的 F 统计量"""
        groups = list(tiny_groups.values())
        result = calculate_f_statistic(groups)

        # 应该能返回结果
        assert 'f_statistic' in result
        assert 'p_value' in result

    def test_eta_squared_boundary(self, constant_groups):
        """测试 η² 边界值（0 <= η² <= 1）"""
        groups = list(constant_groups.values())
        result = calculate_eta_squared(groups)

        assert 0 <= result['eta_squared'] <= 1

    def test_bonferroni_with_zero_tests(self):
        """测试 0 次检验的 Bonferroni 校正"""
        # 0 次检验应该返回 None 或 1（不进行任何检验）
        result = bonferroni_correction(alpha=0.05, n_tests=0)
        # 可能是 None 或 alpha 本身
        assert result.get('corrected_alpha') is None or result.get('corrected_alpha') > 0

    def test_fwer_with_zero_tests(self):
        """测试 0 次检验的 FWER"""
        result = calculate_fwer(alpha=0.05, n_tests=0)
        # 0 次检验不应该有假阳性
        assert result == 0


# =============================================================================
# 测试与 scipy 一致性
# =============================================================================

class TestScipyConsistency:
    """测试与 scipy.stats 结果一致性"""

    def test_f_statistic_matches_scipy(self, sample_five_groups_with_effect):
        """测试 F 统计量与 scipy.f_oneway 一致"""
        groups = list(sample_five_groups_with_effect.values())
        result = calculate_f_statistic(groups)

        f_scipy, p_scipy = stats.f_oneway(*groups)

        assert abs(result['f_statistic'] - f_scipy) < 0.01
        assert abs(result['p_value'] - p_scipy) < 0.01

    def test_chi_square_matches_scipy(self, contingency_table_associated):
        """测试卡方检验与 scipy.chi2_contingency 一致"""
        observed = contingency_table_associated.values
        result = chi_square_test(contingency_table_associated)

        chi2_scipy, p_scipy, dof_scipy, _ = stats.chi2_contingency(observed)

        assert abs(result['chi2'] - chi2_scipy) < 0.01
        assert abs(result['p_value'] - p_scipy) < 0.01
        assert result['dof'] == dof_scipy

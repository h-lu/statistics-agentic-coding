"""
Edge case tests for Week 07 solution.py

Tests covering:
- Empty inputs
- Single values
- Extreme values
- Boundary conditions
- Degenerate cases
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

# Import functions from solution module
import sys
from pathlib import Path

# Add starter_code to path
sys.path.insert(0, str(Path(__file__).parent.parent / "starter_code"))


# =============================================================================
# 多重比较问题 - 边界情况
# =============================================================================

class TestFWEREdgeCases:
    """FWER 计算的边界情况"""

    def test_fwer_zero_tests(self):
        """边界：m=0 时 FWER=0"""
        # TODO: Uncomment when solution.py is created
        # result = calculate_family_wise_error_rate(0, alpha=0.05)
        # assert result == 0.0
        pytest.skip("solution.py not yet created")

    def test_fwer_negative_alpha(self):
        """反例：alpha<0 无意义"""
        # TODO: Uncomment when solution.py is created
        # with pytest.raises(ValueError, match="alpha|positive"):
        #     calculate_family_wise_error_rate(5, alpha=-0.01)
        pytest.skip("solution.py not yet created")

    def test_fwer_alpha_greater_than_one(self):
        """反例：alpha>1 无意义"""
        # TODO: Uncomment when solution.py is created
        # with pytest.raises(ValueError, match="alpha|1.0"):
        #     calculate_family_wise_error_rate(5, alpha=1.5)
        pytest.skip("solution.py not yet created")

    def test_fwer_zero_alpha(self):
        """边界：alpha=0 时 FWER=0"""
        # TODO: Uncomment when solution.py is created
        # result = calculate_family_wise_error_rate(5, alpha=0.0)
        # assert result == 0.0
        pytest.skip("solution.py not yet created")

    def test_fwer_alpha_one(self):
        """边界：alpha=1 时 FWER=1"""
        # TODO: Uncomment when solution.py is created
        # result = calculate_family_wise_error_rate(5, alpha=1.0)
        # assert result == 1.0
        pytest.skip("solution.py not yet created")

    def test_fwer_very_large_m(self):
        """边界：m 极大时 FWER 接近 1"""
        # TODO: Uncomment when solution.py is created
        # result = calculate_family_wise_error_rate(1000, alpha=0.05)
        # # 1 - 0.95^1000 ≈ 1.0
        # assert result > 0.999
        pytest.skip("solution.py not yet created")


# =============================================================================
# ANOVA - 边界情况
# =============================================================================

class TestANOVAEdgeCases:
    """ANOVA 的边界情况"""

    def test_anova_empty_groups(self, empty_groups):
        """反例：空组应报错"""
        # TODO: Uncomment when solution.py is created
        # groups = list(empty_groups.values())
        # with pytest.raises(ValueError, match="empty|数据不足"):
        #     one_way_anova(*groups)
        pytest.skip("solution.py not yet created")

    def test_anova_single_value_per_group(self, single_value_groups):
        """边界：每组一个值时组内方差为0"""
        # TODO: Uncomment when solution.py is created
        # groups = list(single_value_groups.values())
        # # 可能返回 F=inf 或报错
        # try:
        #     result = one_way_anova(*groups)
        #     # 如果成功，组内方差为0可能导致问题
        #     assert result['f_statistic'] == float('inf') or result['p_value'] == 0
        # except (ValueError, ZeroDivisionError):
        #     # 报错也是合理的
        #     pass
        pytest.skip("solution.py not yet created")

    def test_anova_two_groups_only(self, two_groups):
        """边界：只有两组时 ANOVA = t 检验"""
        # TODO: Uncomment when solution.py is created
        # groups = list(two_groups.values())
        # result = one_way_anova(*groups)
        # # F 统计量应有定义
        # assert 'f_statistic' in result
        # assert 'p_value' in result
        pytest.skip("solution.py not yet created")

    def test_anova_identical_data(self, identical_groups):
        """边界：所有数据完全相同时"""
        # TODO: Uncomment when solution.py is created
        # groups = list(identical_groups.values())
        # result = one_way_anova(*groups)
        # # 组间方差为0，F=0
        # assert result['f_statistic'] < 1e-10
        # assert result['p_value'] == pytest.approx(1.0, abs=0.01)
        pytest.skip("solution.py not yet created")

    def test_anova_constant_values_within_groups(self):
        """边界：组内常数，组间不同"""
        # TODO: Uncomment when solution.py is created
        # group_a = np.array([50, 50, 50, 50, 50])
        # group_b = np.array([60, 60, 60, 60, 60])
        # group_c = np.array([70, 70, 70, 70, 70])
        # # 组内方差=0，组间方差>0
        # # F = inf 或非常大
        # result = one_way_anova(group_a, group_b, group_c)
        # assert result['f_statistic'] > 1000 or result['f_statistic'] == float('inf')
        # assert result['p_value'] < 0.001
        pytest.skip("solution.py not yet created")

    def test_anova_very_large_sample(self):
        """边界：极大样本量"""
        # TODO: Uncomment when solution.py is created
        # np.random.seed(42)
        # group_a = np.random.normal(100, 15, 10000)
        # group_b = np.random.normal(100, 15, 10000)
        # group_c = np.random.normal(100, 15, 10000)
        # result = one_way_anova(group_a, group_b, group_c)
        # # 应能正常处理
        # assert 'p_value' in result
        pytest.skip("solution.py not yet created")

    def test_anova_very_small_sample(self):
        """边界：极小样本量"""
        # TODO: Uncomment when solution.py is created
        # np.random.seed(42)
        # group_a = np.random.normal(100, 15, 3)
        # group_b = np.random.normal(100, 15, 3)
        # group_c = np.random.normal(100, 15, 3)
        # result = one_way_anova(group_a, group_b, group_c)
        # # 小样本功效低，但应能运行
        # assert 'p_value' in result
        pytest.skip("solution.py not yet created")

    def test_anova_extreme_outliers(self, extreme_outlier_groups):
        """边界：极端离群点"""
        # TODO: Uncomment when solution.py is created
        # groups = list(extreme_outlier_groups.values())
        # result = one_way_anova(*groups)
        # # 离群点可能导致显著
        # # 不强制要求结果，只检查能运行
        # assert 'p_value' in result
        pytest.skip("solution.py not yet created")

    def test_anova_nan_values(self):
        """反例：包含 NaN 值"""
        # TODO: Uncomment when solution.py is created
        # group_a = np.array([1, 2, np.nan, 4, 5])
        # group_b = np.array([2, 3, 4, 5, 6])
        # # 应处理 NaN 或报错
        # try:
        #     result = one_way_anova(group_a, group_b)
        #     # 如果成功，可能自动删除了 NaN
        #     assert 'p_value' in result
        # except ValueError:
        #     # 报错也是合理的
        #     pass
        pytest.skip("solution.py not yet created")

    def test_anova_inf_values(self):
        """反例：包含 Inf 值"""
        # TODO: Uncomment when solution.py is created
        # group_a = np.array([1, 2, 3, 4, 5])
        # group_b = np.array([2, 3, 4, 5, np.inf])
        # # 应报错
        # with pytest.raises(ValueError, match="inf|invalid"):
        #     one_way_anova(group_a, group_b)
        pytest.skip("solution.py not yet created")

    def test_anova_unequal_sample_sizes(self):
        """边界：各组样本量不等"""
        # TODO: Uncomment when solution.py is created
        # np.random.seed(42)
        # group_a = np.random.normal(100, 15, 30)
        # group_b = np.random.normal(100, 15, 50)
        # group_c = np.random.normal(100, 15, 100)
        # result = one_way_anova(group_a, group_b, group_c)
        # # 应能处理不等样本量
        # assert 'p_value' in result
        pytest.skip("solution.py not yet created")

    def test_anova_heteroscedasticity_large(self, unequal_variance_groups):
        """边界：方差不齐严重"""
        # TODO: Uncomment when solution.py is created
        # groups = list(unequal_variance_groups.values())
        # result = one_way_anova(*groups)
        # # ANOVA 对方差不齐敏感
        # # 应检查假设并可能建议替代方法
        # assert 'assumptions_checked' in result or 'warning' in result
        pytest.skip("solution.py not yet created")


# =============================================================================
# 事后比较 - 边界情况
# =============================================================================

class TestTukeyHSDEdgeCases:
    """Tukey HSD 的边界情况"""

    def test_tukey_two_groups_only(self, two_groups):
        """边界：只有两组时"""
        # TODO: Uncomment when solution.py is created
        # groups_dict = two_groups
        # result = tukey_hsd_test(groups_dict)
        # # 2 组只有 1 对
        # assert len(result) == 1
        pytest.skip("solution.py not yet created")

    def test_tukey_single_group(self):
        """反例：只有一组无法做比较"""
        # TODO: Uncomment when solution.py is created
        # groups_dict = {'A': np.array([1, 2, 3, 4, 5])}
        # with pytest.raises(ValueError, match="至少.*组|2.*groups"):
        #     tukey_hsd_test(groups_dict)
        pytest.skip("solution.py not yet created")

    def test_tukey_identical_groups(self, identical_groups):
        """边界：完全相同的组"""
        # TODO: Uncomment when solution.py is created
        # groups_dict = identical_groups
        # result = tukey_hsd_test(groups_dict)
        # # 所有均值差为0，都不显著
        # for pair in result:
        #     assert pair['meandiff'] == 0
        #     assert pair['reject'] is False
        pytest.skip("solution.py not yet created")

    def test_tukey_binary_data(self, binary_conversion_groups):
        """边界：二元数据"""
        # TODO: Uncomment when solution.py is created
        # groups_dict = binary_conversion_groups
        # result = tukey_hsd_test(groups_dict)
        # # Tukey HSD 可用于二元数据（虽然可能不是最佳）
        # assert len(result) == 6  # 4 组 = 6 对
        pytest.skip("solution.py not yet created")


# =============================================================================
# 校正方法 - 边界情况
# =============================================================================

class TestCorrectionEdgeCases:
    """校正方法的边界情况"""

    def test_bonferroni_empty_p_values(self):
        """反例：空 p 值列表"""
        # TODO: Uncomment when solution.py is created
        # with pytest.raises(ValueError, match="empty"):
        #     bonferroni_correction([])
        pytest.skip("solution.py not yet created")

    def test_bonferroni_single_p_value(self, single_p_value):
        """边界：单个 p 值"""
        # TODO: Uncomment when solution.py is created
        # result = bonferroni_correction(single_p_value, alpha=0.05)
        # # m=1 时不变
        # assert result['adjusted_alpha'] == 0.05
        pytest.skip("solution.py not yet created")

    def test_bonferroni_p_value_exactly_alpha(self):
        """边界：p = alpha 时"""
        # TODO: Uncomment when solution.py is created
        # result = bonferroni_correction([0.05], alpha=0.05)
        # # p = alpha 不显著（严格不等式）
        # assert result['rejected'][0] is False
        pytest.skip("solution.py not yet created")

    def test_bonferroni_p_value_zero(self):
        """边界：p=0 时"""
        # TODO: Uncomment when solution.py is created
        # result = bonferroni_correction([0.0, 0.1], alpha=0.05)
        # # p=0 无论怎么校正都显著
        # assert result['rejected'][0] is True
        pytest.skip("solution.py not yet created")

    def test_bonferroni_p_value_one(self):
        """边界：p=1 时"""
        # TODO: Uncomment when solution.py is created
        # result = bonferroni_correction([1.0, 0.01], alpha=0.05)
        # # p=1 无论怎么校正都不显著
        # assert result['rejected'][0] is False
        pytest.skip("solution.py not yet created")

    def test_bonferroni_very_large_m(self):
        """边界：m 极大时 adjusted_alpha 极小"""
        # TODO: Uncomment when solution.py is created
        # m = 10000
        # p_values = np.array([0.0001] * 10)  # 都很小
        # result = bonferroni_correction(p_values, alpha=0.05)
        # # adjusted_alpha = 0.05 / 10000 = 5e-6
        # assert result['adjusted_alpha'] < 1e-5
        pytest.skip("solution.py not yet created")

    def test_fdr_empty_p_values(self):
        """反例：空 p 值列表"""
        # TODO: Uncomment when solution.py is created
        # with pytest.raises(ValueError, match="empty"):
        #     fdr_correction([])
        pytest.skip("solution.py not yet created")

    def test_fdr_all_zeros(self):
        """边界：所有 p=0"""
        # TODO: Uncomment when solution.py is created
        # result = fdr_correction([0.0, 0.0, 0.0], alpha=0.05)
        # # 都应显著
        # assert result['rejected'].all()
        pytest.skip("solution.py not yet created")

    def test_fdr_all_ones(self):
        """边界：所有 p=1"""
        # TODO: Uncomment when solution.py is created
        # result = fdr_correction([1.0, 1.0, 1.0], alpha=0.05)
        # # 都不显著
        # assert not result['rejected'].any()
        pytest.skip("solution.py not yet created")

    def test_fdr_unsorted_input(self):
        """边界：输入未排序"""
        # TODO: Uncomment when solution.py is created
        # p_values = np.array([0.05, 0.01, 0.10, 0.001])
        # result = fdr_correction(p_values, alpha=0.05)
        # # FDR 应内部排序
        # # 不强制特定结果，只检查能运行
        # assert 'rejected' in result
        pytest.skip("solution.py not yet created")

    def test_fdr_duplicate_p_values(self):
        """边界：重复的 p 值"""
        # TODO: Uncomment when solution.py is created
        # p_values = np.array([0.01, 0.01, 0.01, 0.10])
        # result = fdr_correction(p_values, alpha=0.05)
        # # 应能处理重复值
        # assert 'rejected' in result
        pytest.skip("solution.py not yet created")


# =============================================================================
# 效应量 - 边界情况
# =============================================================================

class TestEffectSizeEdgeCases:
    """效应量的边界情况"""

    def test_eta_squared_no_variation(self, identical_groups):
        """边界：无变异时 η²=0"""
        # TODO: Uncomment when solution.py is created
        # groups = list(identical_groups.values())
        # eta_sq = calculate_eta_squared(groups)
        # # 组间无变异
        # assert eta_sq == 0.0
        pytest.skip("solution.py not yet created")

    def test_eta_squared_perfect_separation(self):
        """边界：完全分离时 η²=1"""
        # TODO: Uncomment when solution.py is created
        # group_a = np.array([0, 0, 0, 0, 0])
        # group_b = np.array([100, 100, 100, 100, 100])
        # # 组间变异大，组内变异小
        # # η² 应接近 1
        # eta_sq = calculate_eta_squared([group_a, group_b])
        # assert eta_sq > 0.9
        pytest.skip("solution.py not yet created")

    def test_interpret_eta_squared_boundary(self):
        """边界：η² 边界值的解释"""
        # TODO: Uncomment when solution.py is created
        # assert interpret_eta_squared(0.009) == "小效应或可忽略"
        # assert interpret_eta_squared(0.01) == "小效应"
        # assert interpret_eta_squared(0.059) == "小效应"
        # assert interpret_eta_squared(0.06) == "中等效应"
        # assert interpret_eta_squared(0.139) == "中等效应"
        # assert interpret_eta_squared(0.14) == "大效应"
        pytest.skip("solution.py not yet created")


# =============================================================================
# 非参数检验 - 边界情况
# =============================================================================

class TestKruskalWallisEdgeCases:
    """Kruskal-Wallis 的边界情况"""

    def test_kruskal_wallis_empty_groups(self, empty_groups):
        """反例：空组应报错"""
        # TODO: Uncomment when solution.py is created
        # groups = list(empty_groups.values())
        # with pytest.raises(ValueError, match="empty"):
        #     kruskal_wallis_test(*groups)
        pytest.skip("solution.py not yet created")

    def test_kruskal_wallis_tied_values(self):
        """边界：有很多相同值（结）"""
        # TODO: Uncomment when solution.py is created
        # # 离散数据有很多结
        # group_a = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
        # group_b = np.array([1, 1, 2, 2, 3, 3, 4, 4, 4])
        # result = kruskal_wallis_test(group_a, group_b)
        # # 应能处理结
        # assert 'p_value' in result
        pytest.skip("solution.py not yet created")

    def test_kruskal_wallis_small_sample(self):
        """边界：小样本"""
        # TODO: Uncomment when solution.py is created
        # group_a = np.array([1, 2, 3])
        # group_b = np.array([4, 5, 6])
        # group_c = np.array([7, 8, 9])
        # result = kruskal_wallis_test(group_a, group_b, group_c)
        # # 小样本可能需要精确检验
        # assert 'p_value' in result
        pytest.skip("solution.py not yet created")


# =============================================================================
# AI 审查 - 边界情况
# =============================================================================

class TestAIReviewEdgeCases:
    """AI 审查的边界情况"""

    def test_review_empty_report(self):
        """反例：空报告"""
        # TODO: Uncomment when solution.py is created
        # with pytest.raises(ValueError, match="empty|report"):
        #     review_ai_anova_report({})
        pytest.skip("solution.py not yet created")

    def test_report_missing_p_value(self):
        """边界：缺少 p 值"""
        # TODO: Uncomment when solution.py is created
        # report = {'test_method': 'anova', 'f_statistic': 3.2}
        # review = review_ai_anova_report(report)
        # assert '缺少 p 值' in review['issues']
        pytest.skip("solution.py not yet created")

    def test_report_negative_p_value(self):
        """反例：p 值为负"""
        # TODO: Uncomment when solution.py is created
        # report = {'test_method': 'anova', 'p_value': -0.01}
        # review = review_ai_anova_report(report)
        # assert 'p 值无效' in review['issues'] or 'negative' in review['issues']
        pytest.skip("solution.py not yet created")

    def test_report_p_value_greater_than_one(self):
        """反例：p 值 > 1"""
        # TODO: Uncomment when solution.py is created
        # report = {'test_method': 'anova', 'p_value': 1.5}
        # review = review_ai_anova_report(report)
        # assert 'p 值无效' in review['issues'] or '> 1' in review['issues']
        pytest.skip("solution.py not yet created")

    def test_report_conflicting_conclusions(self):
        """边界：结论与 p 值矛盾"""
        # TODO: Uncomment when solution.py is created
        # report = {
        #     'test_method': 'anova',
        #     'p_value': 0.50,  # 不显著
        #     'conclusion': '各组存在显著差异'  # 但说显著
        # }
        # review = review_ai_anova_report(report)
        # assert '结论矛盾' in review['issues'] or '不一致' in review['issues']
        pytest.skip("solution.py not yet created")

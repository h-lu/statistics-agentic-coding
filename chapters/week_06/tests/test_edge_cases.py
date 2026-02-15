"""
Edge case tests for Week 06 solution.py

Tests focus on boundary conditions, invalid inputs, and edge cases.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

# Add starter_code to path
sys.path.insert(0, str(Path(__file__).parent.parent / "starter_code"))


# =============================================================================
# 空输入和极小样本测试
# =============================================================================

class TestEmptyInput:
    """Tests for empty input handling."""

    def test_p_value_interpretation_empty_input(self):
        """空 p 值应报错或返回 None"""
        # TODO: Uncomment when solution.py is created
        # with pytest.raises(ValueError):
        #     interpret_p_value(p_value=None)
        # with pytest.raises(ValueError):
        #     interpret_p_value(p_value=np.nan)
        pytest.skip("solution.py not yet created")

    def test_t_test_empty_group(self):
        """空组应报错"""
        # TODO: Uncomment when solution.py is created
        # with pytest.raises(ValueError, match="empty"):
        #     two_sample_t_test(np.array([1, 2, 3]), np.array([]))
        pytest.skip("solution.py not yet created")

    def test_cohens_d_empty_data(self):
        """空数据应报错"""
        # TODO: Uncomment when solution.py is created
        # with pytest.raises(ValueError, match="empty"):
        #     cohens_d(np.array([]), np.array([1, 2, 3]))
        pytest.skip("solution.py not yet created")

    def test_normality_test_empty_data(self):
        """空数据应报错"""
        # TODO: Uncomment when solution.py is created
        # with pytest.raises(ValueError, match="empty"):
        #     check_normality(np.array([]))
        pytest.skip("solution.py not yet created")


class TestMinimalSampleSize:
    """Tests for minimal sample sizes."""

    def test_t_test_n_equals_2(self):
        """每组 2 个样本（最小可计算标准差）"""
        group_a = np.array([1.0, 2.0])
        group_b = np.array([3.0, 4.0])
        # TODO: Uncomment when solution.py is created
        # result = two_sample_t_test(group_a, group_b)
        # assert result['degrees_of_freedom'] == 2
        # assert result['p_value'] is not None
        pytest.skip("solution.py not yet created")

    def test_t_test_n_equals_1(self):
        """每组 1 个样本（无法计算标准差）"""
        group_a = np.array([5.0])
        group_b = np.array([6.0])
        # TODO: Uncomment when solution.py is created
        # result = two_sample_t_test(group_a, group_b)
        # 应返回 None 或报错
        # assert result is None or 'error' in result
        pytest.skip("solution.py not yet created")

    def test_shapiro_wilk_min_sample(self):
        """Shapiro-Wilk 最小需要 3 个样本"""
        # TODO: Uncomment when solution.py is created
        # result = check_normality(np.array([1, 2]))
        # 应返回警告或 None
        # assert result['warning'] is not None or result is None
        pytest.skip("solution.py not yet created")

    def test_bootstrap_n_equals_2(self):
        """Bootstrap 用 2 个样本（理论上可行但估计很差）"""
        data = np.array([1.0, 2.0])
        # TODO: Uncomment when solution.py is created
        # ci = bootstrap_confidence_interval(data, np.mean, n_bootstrap=100)
        # 应该能运行但区间很宽
        # assert ci[1] - ci[0] > 1  # 区间很宽
        pytest.skip("solution.py not yet created")


# =============================================================================
# 极端值和异常值测试
# =============================================================================

class TestExtremeValues:
    """Tests for extreme and outlier values."""

    def test_t_test_with_extreme_outlier(self):
        """单个极端异常值对 t 检验的影响"""
        np.random.seed(42)
        group_a = np.random.normal(100, 10, 50)
        group_b = np.random.normal(100, 10, 50)
        # 添加一个极端异常值
        group_b = np.append(group_b, 1000)
        # TODO: Uncomment when solution.py is created
        # result = two_sample_t_test(group_a, group_b)
        # 异常值可能导致显著结果（假阳性）
        # 考虑添加稳健检验选项
        pytest.skip("solution.py not yet created")

    def test_cohens_d_with_outlier(self):
        """异常值对效应量的影响"""
        np.random.seed(42)
        group_a = np.random.normal(100, 10, 50)
        group_b = np.random.normal(105, 10, 50)
        # 添加异常值
        group_a_outlier = np.append(group_a, 1000)
        # TODO: Uncomment when solution.py is created
        # d_normal = cohens_d(group_a, group_b)
        # d_outlier = cohens_d(group_a_outlier, group_b)
        # 异常值应显著改变效应量
        # assert abs(d_outlier) < abs(d_normal)  # 异常值在较大的组，实际效应被稀释
        pytest.skip("solution.py not yet created")

    def test_proportion_test_all_success(self):
        """全部成功的极端情况（100% 转化率）"""
        all_converted = np.array([1] * 100)
        none_converted = np.array([0] * 100)
        # TODO: Uncomment when solution.py is created
        # result = proportion_test(all_converted, none_converted)
        # assert result['p_value'] < 0.001  # 极端显著
        # assert result['rate_a'] == 1.0
        # assert result['rate_b'] == 0.0
        pytest.skip("solution.py not yet created")

    def test_proportion_test_all_failure(self):
        """全部失败的极端情况（0% 转化率）"""
        # TODO: Uncomment when solution.py is created
        # all_failed_a = np.array([0] * 100)
        # all_failed_b = np.array([0] * 100)
        # result = proportion_test(all_failed_a, all_failed_b)
        # assert result['p_value'] == pytest.approx(1.0, abs=0.01)
        # assert result['difference'] == 0
        pytest.skip("solution.py not yet created")


# =============================================================================
# 特殊数据类型测试
# =============================================================================

class TestSpecialDataTypes:
    """Tests for special data types."""

    def test_all_same_values(self):
        """全部相同值（方差为0）"""
        constant_a = np.array([50] * 50)
        constant_b = np.array([55] * 50)
        # TODO: Uncomment when solution.py is created
        # result = two_sample_t_test(constant_a, constant_b)
        # 方差为 0，标准 t 检验可能失败
        # 应使用替代方法或返回特殊结果
        pytest.skip("solution.py not yet created")

    def test_binary_with_rare_event(self):
        """稀有事件（转化率 < 1%）"""
        np.random.seed(42)
        rare_a = np.array([1] * 5 + [0] * 995)  # 0.5% 转化率
        rare_b = np.array([1] * 2 + [0] * 998)  # 0.2% 转化率
        # TODO: Uncomment when solution.py is created
        # result = proportion_test(rare_a, rare_b)
        # 稀有事件需要大样本
        # 应警告统计功效不足
        pytest.skip("solution.py not yet created")

    def test_very_large_sample(self):
        """极大样本（n > 100000）"""
        np.random.seed(42)
        large_a = np.random.normal(100, 10, 100000)
        large_b = np.random.normal(100.1, 10, 100000)  # 极小差异
        # TODO: Uncomment when solution.py is created
        # result = two_sample_t_test(large_a, large_b)
        # 大样本可能检测到微小的、无实际意义的差异
        # 应报告效应量而不仅仅是 p 值
        # assert result['p_value'] < 0.05  # 可能显著
        # assert abs(result['effect_size']) < 0.01  # 但效应量极小
        pytest.skip("solution.py not yet created")

    def test_nan_values(self):
        """包含 NaN 值的数据"""
        data_with_nan = np.array([1, 2, np.nan, 4, 5])
        # TODO: Uncomment when solution.py is created
        # result = check_normality(data_with_nan)
        # 应自动处理 NaN（删除或报错）
        # assert 'warning' in result or result is not None
        pytest.skip("solution.py not yet created")

    def test_inf_values(self):
        """包含 Inf 值的数据"""
        data_with_inf = np.array([1, 2, np.inf, 4, 5])
        # TODO: Uncomment when solution.py is created
        # result = two_sample_t_test(data_with_inf, data_with_inf)
        # 应报错或处理 Inf
        # assert 'error' in result or raises ValueError
        pytest.skip("solution.py not yet created")


# =============================================================================
# 参数边界测试
# =============================================================================

class TestParameterBoundaries:
    """Tests for parameter boundary conditions."""

    def test_alpha_boundary_zero(self):
        """alpha = 0（无假阳性容忍）"""
        # TODO: Uncomment when solution.py is created
        # result = interpret_p_value(p_value=0.001, alpha=0.0)
        # alpha=0 不合理，应报错
        # with pytest.raises(ValueError, match="alpha"):
        #     interpret_p_value(p_value=0.05, alpha=0.0)
        pytest.skip("solution.py not yet created")

    def test_alpha_boundary_one(self):
        """alpha = 1（总是拒绝原假设）"""
        # TODO: Uncomment when solution.py is created
        # with pytest.raises(ValueError, match="alpha"):
        #     interpret_p_value(p_value=0.5, alpha=1.0)
        pytest.skip("solution.py not yet created")

    def test_p_value_boundary_zero(self):
        """p 值 = 0（极端罕见）"""
        # TODO: Uncomment when solution.py is created
        # result = interpret_p_value(p_value=0.0, alpha=0.05)
        # assert result['reject_null'] is True
        # assert 'extremely_significant' in result
        pytest.skip("solution.py not yet created")

    def test_p_value_boundary_one(self):
        """p 值 = 1（完全符合原假设）"""
        # TODO: Uncomment when solution.py is created
        # result = interpret_p_value(p_value=1.0, alpha=0.05)
        # assert result['reject_null'] is False
        # assert 'no_evidence' in result
        pytest.skip("solution.py not yet created")

    def test_negative_p_value(self):
        """负 p 值（无效输入）"""
        # TODO: Uncomment when solution.py is created
        # with pytest.raises(ValueError, match="p.value"):
        #     interpret_p_value(p_value=-0.01)
        pytest.skip("solution.py not yet created")

    def test_p_value_greater_than_one(self):
        """p 值 > 1（无效输入）"""
        # TODO: Uncomment when solution.py is created
        # with pytest.raises(ValueError, match="p.value"):
        #     interpret_p_value(p_value=1.5)
        pytest.skip("solution.py not yet created")


# =============================================================================
# 多重比较边界测试
# =============================================================================

class TestMultipleComparisonEdges:
    """Tests for multiple comparison edge cases."""

    def test_single_hypothesis(self):
        """单个假设（无需多重比较校正）"""
        # TODO: Uncomment when solution.py is created
        # results = [{'p_value': 0.03}]
        # corrected = bonferroni_correction(results, alpha=0.05)
        # assert corrected['corrected_alpha'] == 0.05  # 无变化
        # assert corrected['results'][0]['still_significant'] is True
        pytest.skip("solution.py not yet created")

    def test_zero_hypotheses(self):
        """零个假设（无效输入）"""
        # TODO: Uncomment when solution.py is created
        # with pytest.raises(ValueError, match="hypotheses"):
        #     bonferroni_correction([], alpha=0.05)
        pytest.skip("solution.py not yet created")

    def test_all_significant_hypotheses(self):
        """所有假设都显著"""
        # TODO: Uncomment when solution.py is created
        # results = [
        #     {'p_value': 0.001},
        #     {'p_value': 0.002},
        #     {'p_value': 0.003},
        #     {'p_value': 0.004},
        #     {'p_value': 0.005}
        # ]
        # corrected = bonferroni_correction(results, alpha=0.05)
        # 校正后 alpha = 0.01
        # 所有 p 值仍应显著
        # assert all(r['still_significant'] for r in corrected['results'])
        pytest.skip("solution.py not yet created")

    def test_all_non_significant_hypotheses(self):
        """所有假设都不显著"""
        # TODO: Uncomment when solution.py is created
        # results = [
        #     {'p_value': 0.10},
        #     {'p_value': 0.20},
        #     {'p_value': 0.30},
        #     {'p_value': 0.40},
        #     {'p_value': 0.50}
        # ]
        # corrected = bonferroni_correction(results, alpha=0.05)
        # assert not any(r['still_significant'] for r in corrected['results'])
        pytest.skip("solution.py not yet created")

    def test_very_many_hypotheses(self):
        """极多假设（alpha 变得很小）"""
        # TODO: Uncomment when solution.py is created
        # n = 1000
        # corrected = bonferroni_correction([{'p_value': 0.0001}] * n, alpha=0.05)
        # corrected_alpha = 0.05 / n
        # 极小的 alpha 使得几乎所有结果都不显著
        # assert corrected['corrected_alpha'] == pytest.approx(0.00005, rel=0.1)
        pytest.skip("solution.py not yet created")


# =============================================================================
# 数值精度和稳定性测试
# =============================================================================

class TestNumericalStability:
    """Tests for numerical precision and stability."""

    def test_very_small_numbers(self):
        """极小数值（接近浮点精度极限）"""
        tiny_a = np.array([1e-300, 2e-300, 3e-300])
        tiny_b = np.array([2e-300, 3e-300, 4e-300])
        # TODO: Uncomment when solution.py is created
        # result = two_sample_t_test(tiny_a, tiny_b)
        # 应能处理而不溢出
        # assert result is not None
        pytest.skip("solution.py not yet created")

    def test_very_large_numbers(self):
        """极大数值（接近浮点精度极限）"""
        huge_a = np.array([1e300, 2e300, 3e300])
        huge_b = np.array([2e300, 3e300, 4e300])
        # TODO: Uncomment when solution.py is created
        # result = two_sample_t_test(huge_a, huge_b)
        # 应能处理而不溢出
        # assert result is not None
        pytest.skip("solution.py not yet created")

    def test_mixed_scale_numbers(self):
        """混合数量级的数值"""
        mixed = np.array([1e-10, 1e-5, 1, 1e5, 1e10])
        # TODO: Uncomment when solution.py is created
        # result = check_normality(mixed)
        # 应能处理（但可能警告分布极度偏态）
        # assert result is not None
        pytest.skip("solution.py not yet created")


# =============================================================================
# 列联表边界测试
# =============================================================================

class TestContingencyTableEdges:
    """Tests for contingency table edge cases."""

    def test_2x2_table_with_zero(self):
        """包含 0 的 2x2 列联表"""
        table_with_zero = pd.DataFrame({
            'A': [10, 0],
            'B': [5, 5]
        }, index=['Success', 'Fail'])
        # TODO: Uncomment when solution.py is created
        # result = chi_square_test(table_with_zero)
        # 包含 0 但期望频数可能 > 5，应能处理
        # assert result is not None
        pytest.skip("solution.py not yet created")

    def test_very_small_counts(self):
        """极小计数的列联表"""
        tiny_table = pd.DataFrame({
            'A': [1, 2],
            'B': [2, 1]
        }, index=['X', 'Y'])
        # TODO: Uncomment when solution.py is created
        # result = chi_square_test(tiny_table)
        # 可能需要 Fisher 精确检验而不是卡方检验
        # 应警告或自动切换方法
        pytest.skip("solution.py not yet created")

    def test_single_row_table(self):
        """单行列联表（无效输入）"""
        # TODO: Uncomment when solution.py is created
        # single_row = pd.DataFrame({'A': [10], 'B': [20]})
        # with pytest.raises(ValueError, match="at least 2 rows"):
        #     chi_square_test(single_row)
        pytest.skip("solution.py not yet created")

    def test_single_column_table(self):
        """单列列联表（无效输入）"""
        # TODO: Uncomment when solution.py is created
        # single_col = pd.DataFrame({'Count': [10, 20]}, index=['A', 'B'])
        # with pytest.raises(ValueError, match="at least 2 columns"):
        #     chi_square_test(single_col)
        pytest.skip("solution.py not yet created")

"""
Test suite for Week 07 solution.py

Tests cover:
1. 多重比较问题（FWER 计算）
2. ANOVA（方差分析）
3. F 统计量
4. 事后比较（Tukey HSD）
5. 多重比较校正（Bonferroni、FDR）

注意：由于 solution.py 尚未创建，这些测试定义了预期的接口规范。
测试分为以下类别：
- 正例（happy path）：正常输入时的正确行为
- 边界：空输入、极值、特殊情况
- 反例：错误输入或应拒绝的情况
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
# 1. 多重比较问题测试
# =============================================================================

class TestFamilyWiseErrorRate:
    """测试家族错误率（FWER）计算"""

    def test_fwer_single_test_returns_alpha(self, single_test_result):
        """
        正例：m=1 时 FWER 应等于 alpha

        单个检验时，FWER = alpha（无累积）
        """
        # TODO: Uncomment when solution.py is created
        # result = calculate_family_wise_error_rate(n_tests=1, alpha=0.05)
        # assert result == pytest.approx(0.05)
        pytest.skip("solution.py not yet created")

    def test_fwer_five_tests(self):
        """
        正例：m=5 时 FWER 应约为 0.226

        FWER = 1 - (1 - α)^m = 1 - (1 - 0.05)^5 ≈ 0.226
        """
        # TODO: Uncomment when solution.py is created
        # result = calculate_family_wise_error_rate(n_tests=5, alpha=0.05)
        # expected = 1 - (1 - 0.05) ** 5
        # assert result == pytest.approx(expected, rel=0.01)
        pytest.skip("solution.py not yet created")

    def test_fwer_twenty_tests(self):
        """
        正例：m=20 时 FWER 应约为 0.642

        验证 FWER 随检验次数指数增长
        """
        # TODO: Uncomment when solution.py is created
        # result = calculate_family_wise_error_rate(n_tests=20, alpha=0.05)
        # expected = 1 - (1 - 0.05) ** 20
        # assert result == pytest.approx(expected, rel=0.01)
        pytest.skip("solution.py not yet created")

    def test_fwer_zero_tests(self):
        """
        边界：m=0 时 FWER 应为 0

        无检验时，假阳性率为 0
        """
        # TODO: Uncomment when solution.py is created
        # result = calculate_family_wise_error_rate(n_tests=0, alpha=0.05)
        # assert result == 0.0
        pytest.skip("solution.py not yet created")

    def test_fwer_negative_tests(self):
        """
        反例：m<0 应返回 0 或报错

        负数的检验次数无意义
        """
        # TODO: Uncomment when solution.py is created
        # with pytest.raises(ValueError, match="positive"):
        #     calculate_family_wise_error_rate(n_tests=-1, alpha=0.05)
        pytest.skip("solution.py not yet created")

    def test_fwer_different_alpha(self):
        """
        正例：不同 alpha 下的 FWER 计算

        验证公式对不同的 alpha 都成立
        """
        # TODO: Uncomment when solution.py is created
        # result_01 = calculate_family_wise_error_rate(n_tests=10, alpha=0.01)
        # result_10 = calculate_family_wise_error_rate(n_tests=10, alpha=0.10)
        # expected_01 = 1 - (1 - 0.01) ** 10
        # expected_10 = 1 - (1 - 0.10) ** 10
        # assert result_01 == pytest.approx(expected_01, rel=0.01)
        # assert result_10 == pytest.approx(expected_10, rel=0.01)
        pytest.skip("solution.py not yet created")


class TestMultipleComparisonsSimulation:
    """测试多重比较的模拟实验"""

    def test_false_positive_rate_single_test(self):
        """
        正例：单次检验的假阳性率应约为 alpha

        当原假设成立时，p 值均匀分布，p < alpha 的概率应约为 alpha
        """
        # TODO: Uncomment when solution.py is created
        # np.random.seed(42)
        # p_values = np.random.uniform(0, 1, 1000)
        # false_positive_rate = (p_values < 0.05).mean()
        # assert 0.03 <= false_positive_rate <= 0.07  # 允许抽样误差
        pytest.skip("solution.py not yet created")

    def test_false_positive_accumulation(self):
        """
        正例：多次检验的假阳性累积

        模拟 1000 次，每次做 10 个检验，计算至少一个假阳性的概率
        应接近理论 FWER
        """
        # TODO: Uncomment when solution.py is created
        # np.random.seed(42)
        # n_simulations = 1000
        # n_tests = 10
        # alpha = 0.05
        #
        # at_least_one_fp = 0
        # for _ in range(n_simulations):
        #     p_values = np.random.uniform(0, 1, n_tests)
        #     if (p_values < alpha).any():
        #         at_least_one_fp += 1
        #
        # empirical_fwer = at_least_one_fp / n_simulations
        # expected_fwer = 1 - (1 - alpha) ** n_tests
        # assert 0.35 <= empirical_fwer <= 0.45  # 接近 0.401
        pytest.skip("solution.py not yet created")


# =============================================================================
# 2. ANOVA 测试
# =============================================================================

class TestOneWayANOVA:
    """测试单因素 ANOVA"""

    def test_anova_no_difference(self, four_groups_no_difference):
        """
        正例：无差异时 p 值应均匀分布

        四组来自同一分布时，ANOVA p 值应 > 0.05（大部分情况下）
        注意：由于随机性，约 5% 的情况会 p < 0.05
        """
        # TODO: Uncomment when solution.py is created
        # groups = list(four_groups_no_difference.values())
        # result = one_way_anova(*groups)
        # # 大部分情况下不应显著
        # # 这里我们只检查结果结构，不强制 p > 0.05
        # assert 'p_value' in result
        # assert 'f_statistic' in result
        pytest.skip("solution.py not yet created")

    def test_anova_with_difference(self, four_groups_with_difference):
        """
        正例：有差异时应检测到显著性

        D 组均值明显不同时，ANOVA 应有较小 p 值
        """
        # TODO: Uncomment when solution.py is created
        # groups = list(four_groups_with_difference.values())
        # result = one_way_anova(*groups)
        # assert result['p_value'] < 0.05
        # assert result['f_statistic'] > 1
        # assert result['reject_null'] is True
        pytest.skip("solution.py not yet created")

    def test_anova_small_difference(self, four_groups_small_difference):
        """
        边界：小差异时 p 值可能不显著

        效应量小时，ANOVA 可能不显著（取决于样本量和方差）
        """
        # TODO: Uncomment when solution.py is created
        # groups = list(four_groups_small_difference.values())
        # result = one_way_anova(*groups)
        # # 小差异可能不显著
        # # 不强制要求，只检查结构
        # assert 'p_value' in result
        # assert 'effect_size' in result or 'eta_squared' in result
        pytest.skip("solution.py not yet created")

    def test_anova_binary_data(self, binary_conversion_groups):
        """
        正例：二元数据上的 ANOVA

        转化率是 0/1 数据，ANOVA 仍可运行（但可能不是最佳方法）
        """
        # TODO: Uncomment when solution.py is created
        # groups = list(binary_conversion_groups.values())
        # result = one_way_anova(*groups)
        # assert 'p_value' in result
        # assert 'f_statistic' in result
        pytest.skip("solution.py not yet created")

    def test_anova_two_groups(self, two_groups):
        """
        边界：只有两组时 ANOVA 退化为 t 检验

        F 统计量应等于 t 统计量的平方
        """
        # TODO: Uncomment when solution.py is created
        # groups = list(two_groups.values())
        # result_anova = one_way_anova(*groups)
        # result_t = two_sample_t_test(groups[0], groups[1])
        # # F ≈ t²
        # assert result_anova['f_statistic'] == pytest.approx(result_t['t_statistic']**2, rel=0.01)
        pytest.skip("solution.py not yet created")

    def test_anova_empty_group(self, empty_groups):
        """
        反例：空组应报错

        ANOVA 需要每组至少有一个观测值
        """
        # TODO: Uncomment when solution.py is created
        # groups = list(empty_groups.values())
        # with pytest.raises(ValueError, match="empty|insufficient"):
        #     one_way_anova(*groups)
        pytest.skip("solution.py not yet created")

    def test_anova_single_value(self, single_value_groups):
        """
        边界：单值组

        每组只有一个值时，组内方差为 0，可能导致问题
        """
        # TODO: Uncomment when solution.py is created
        # groups = list(single_value_groups.values())
        # # 可能返回特殊结果或报错
        # try:
        #     result = one_way_anova(*groups)
        #     # 如果成功，检查结果
        #     assert 'p_value' in result
        # except ValueError:
        #     # 如果报错，也是合理的
        #     pass
        pytest.skip("solution.py not yet created")

    def test_anova_identical_groups(self, identical_groups):
        """
        边界：完全相同的组

        所有组完全相同时，F 统计量应为 0 或接近 0
        """
        # TODO: Uncomment when solution.py is created
        # groups = list(identical_groups.values())
        # result = one_way_anova(*groups)
        # # 组间方差为 0，F 应接近 0
        # assert result['f_statistic'] < 1e-10
        # assert result['p_value'] == pytest.approx(1.0, abs=0.01)
        pytest.skip("solution.py not yet created")

    def test_anova_extreme_outlier(self, extreme_outlier_groups):
        """
        边界：极端离群点

        离群点可能严重影响 ANOVA 结果
        """
        # TODO: Uncomment when solution.py is created
        # groups = list(extreme_outlier_groups.values())
        # result = one_way_anova(*groups)
        # # 离群点导致显著，但可能不是真实的组间差异
        # assert 'p_value' in result
        # # 实际应用中应先处理离群点
        pytest.skip("solution.py not yet created")

    def test_anova_with_dataframe(self, anova_dataframe):
        """
        正例：使用 DataFrame 格式的输入

        实际应用中，数据常以 DataFrame 格式存储
        """
        # TODO: Uncomment when solution.py is created
        # df = anova_dataframe
        # result = one_way_anova_from_df(df, group_col='group', value_col='value')
        # assert 'p_value' in result
        # assert 'f_statistic' in result
        pytest.skip("solution.py not yet created")


class TestANOVAAssumptions:
    """测试 ANOVA 前提假设检查"""

    def test_check_normality_all_groups(self, four_groups_with_difference):
        """
        正例：检查所有组的正态性

        对每组执行 Shapiro-Wilk 检验
        """
        # TODO: Uncomment when solution.py is created
        # groups_dict = four_groups_with_difference
        # results = check_anova_normality(groups_dict)
        # for group_name, result in results.items():
        #     assert 'p_value' in result
        #     assert 'is_normal' in result
        pytest.skip("solution.py not yet created")

    def test_check_homogeneity_variance(self, four_groups_with_difference):
        """
        正例：检查方差齐性

        使用 Levene 检验检查各组方差是否相等
        """
        # TODO: Uncomment when solution.py is created
        # groups = list(four_groups_with_difference.values())
        # result = check_homogeneity_variance(*groups)
        # assert 'p_value' in result
        # assert 'equal_variance' in result
        pytest.skip("solution.py not yet created")

    def test_unequal_variance_detection(self, unequal_variance_groups):
        """
        正例：检测方差不齐

        当方差不齐时，Levene 检验应显著
        """
        # TODO: Uncomment when solution.py is created
        # groups = list(unequal_variance_groups.values())
        # result = check_homogeneity_variance(*groups)
        # # 方差不齐时，Levene p 值应较小
        # assert result['p_value'] < 0.05
        # assert result['equal_variance'] is False
        pytest.skip("solution.py not yet created")

    def test_skewed_data_detection(self, skewed_groups):
        """
        正例：检测偏态数据

        偏态数据在 Shapiro-Wilk 检验中应被检测到
        """
        # TODO: Uncomment when solution.py is created
        # groups_dict = skewed_groups
        # results = check_anova_normality(groups_dict)
        # # 偏态数据应被检测为非正态
        # for group_name, result in results.items():
        #     # 指数分布数据通常不通过正态性检验
        #     assert result['is_normal'] is False
        pytest.skip("solution.py not yet created")


class TestKruskalWallis:
    """测试 Kruskal-Wallis 非参数检验"""

    def test_kruskal_wallis_no_difference(self, four_groups_no_difference):
        """
        正例：无差异时 p 值应较大

        Kruskal-Wallis 在原假设成立时 p 值应均匀分布
        """
        # TODO: Uncomment when solution.py is created
        # groups = list(four_groups_no_difference.values())
        # result = kruskal_wallis_test(*groups)
        # assert 'p_value' in result
        # assert 'h_statistic' in result
        pytest.skip("solution.py not yet created")

    def test_kruskal_wallis_with_difference(self, four_groups_with_difference):
        """
        正例：有差异时应检测到

        Kruskal-Wallis 应检测到组间差异
        """
        # TODO: Uncomment when solution.py is created
        # groups = list(four_groups_with_difference.values())
        # result = kruskal_wallis_test(*groups)
        # # Kruskal-Wallis 功效可能低于 ANOVA
        # # 但仍应检测到明显差异
        # assert result['p_value'] < 0.05
        pytest.skip("solution.py not yet created")

    def test_kruskal_wallis_skewed_data(self, skewed_groups):
        """
        正例：偏态数据上使用 Kruskal-Wallis

        偏态数据应使用非参数方法
        """
        # TODO: Uncomment when solution.py is created
        # groups = list(skewed_groups.values())
        # result = kruskal_wallis_test(*groups)
        # assert 'p_value' in result
        # # 不要求显著，只检查能正常运行
        pytest.skip("solution.py not yet created")


class TestEffectSizeANOVA:
    """测试 ANOVA 效应量（η²）"""

    def test_eta_squared_no_difference(self, four_groups_no_difference):
        """
        正例：无差异时 η² 应接近 0

        组间变异小时，η² 应接近 0
        """
        # TODO: Uncomment when solution.py is created
        # groups = list(four_groups_no_difference.values())
        # result = one_way_anova(*groups)
        # eta_sq = calculate_eta_squared(groups)
        # # 无差异时，η² 应很小
        # assert eta_sq < 0.1
        pytest.skip("solution.py not yet created")

    def test_eta_squared_with_difference(self, four_groups_with_difference):
        """
        正例：有差异时 η² 应较大

        组间变异大时，η² 应较大
        """
        # TODO: Uncomment when solution.py is created
        # groups = list(four_groups_with_difference.values())
        # eta_sq = calculate_eta_squared(groups)
        # # D 组均值高 15，应有中等效应
        # assert eta_sq > 0.05
        pytest.skip("solution.py not yet created")

    def test_interpret_eta_squared(self):
        """
        正例：解释 η² 的大小

        小效应：η² ≈ 0.01
        中等效应：η² ≈ 0.06
        大效应：η² ≈ 0.14
        """
        # TODO: Uncomment when solution.py is created
        # assert interpret_eta_squared(0.01) == "小效应"
        # assert interpret_eta_squared(0.06) == "中等效应"
        # assert interpret_eta_squared(0.14) == "大效应"
        pytest.skip("solution.py not yet created")


# =============================================================================
# 3. 事后比较测试
# =============================================================================

class TestTukeyHSD:
    """测试 Tukey HSD 事后比较"""

    def test_tukey_significant_pairs(self, tukey_significant_pairs):
        """
        正例：正确识别显著差异对

        D 组与其他组应被识别为显著差异
        """
        # TODO: Uncomment when solution.py is created
        # groups_dict = tukey_significant_pairs
        # result = tukey_hsd_test(groups_dict)
        # # D vs A, B, C 应显著
        # significant_pairs = [p for p in result if p['reject']]
        # # 至少应有 3 对显著（D vs A/B/C）
        # assert len(significant_pairs) >= 3
        pytest.skip("solution.py not yet created")

    def test_tukey_no_significant_pairs(self, tukey_no_significant_pairs):
        """
        正例：无差异时不拒绝任何对

        所有组来自同一分布时，Tukey HSD 应不拒绝
        """
        # TODO: Uncomment when solution.py is created
        # groups_dict = tukey_no_significant_pairs
        # result = tukey_hsd_test(groups_dict)
        # significant_pairs = [p for p in result if p['reject']]
        # # 大部分情况下不应有显著对
        # # 允许约 5% 的假阳性
        # assert len(significant_pairs) <= 1  # 最多一对假阳性
        pytest.skip("solution.py not yet created")

    def test_tukey_returns_all_pairs(self):
        """
        正例：返回所有两两比较

        4 组应有 6 对比较
        """
        # TODO: Uncomment when solution.py is created
        # groups_dict = {'A': np.array([1, 2, 3]), 'B': np.array([2, 3, 4]),
        #                'C': np.array([3, 4, 5]), 'D': np.array([4, 5, 6])}
        # result = tukey_hsd_test(groups_dict)
        # # 4 组应有 C(4,2) = 6 对
        # assert len(result) == 6
        pytest.skip("solution.py not yet created")

    def test_tukey_with_dataframe(self, anova_dataframe):
        """
        正例：使用 DataFrame 格式

        实际应用中常使用 DataFrame
        """
        # TODO: Uncomment when solution.py is created
        # df = anova_dataframe
        # result = tukey_hsd_from_df(df, group_col='group', value_col='value')
        # assert len(result) == 6  # 4 组 = 6 对
        # # 每对应有特定字段
        # for pair in result:
        #     assert 'group1' in pair
        #     assert 'group2' in pair
        #     assert 'meandiff' in pair
        #     assert 'p_adj' in pair
        #     assert 'reject' in pair
        pytest.skip("solution.py not yet created")

    def test_tukey_confidence_interval(self, tukey_significant_pairs):
        """
        正例：提供置信区间

        Tukey HSD 应提供 95% 置信区间
        """
        # TODO: Uncomment when solution.py is created
        # groups_dict = tukey_significant_pairs
        # result = tukey_hsd_test(groups_dict, alpha=0.05)
        # for pair in result:
        #     assert 'ci_lower' in pair or 'lower' in pair
        #     assert 'ci_upper' in pair or 'upper' in pair
        pytest.skip("solution.py not yet created")

    def test_tukey_vs_uncorrected_ttest(self, tukey_significant_pairs):
        """
        反例：未校正的 t 检验假阳性率高

        Tukey HSD 的 p 值应大于未校正的 t 检验 p 值
        """
        # TODO: Uncomment when solution.py is created
        # groups_dict = tukey_significant_pairs
        # result_tukey = tukey_hsd_test(groups_dict)
        #
        # # 做未校正的 t 检验
        # groups = list(groups_dict.values())
        # uncorrected_p = []
        # for i in range(len(groups)):
        #     for j in range(i+1, len(groups)):
        #         _, p = stats.ttest_ind(groups[i], groups[j])
        #         uncorrected_p.append(p)
        #
        # # Tukey HSD 的 p 值应大于或等于未校正的 p 值
        # tukey_p = [p['p_adj'] for p in result_tukey]
        # # 校正后 p 值通常更大（更保守）
        # # 不强制每个都更大，但总体趋势如此
        pytest.skip("solution.py not yet created")


class TestPostHocDecisionTree:
    """测试事后比较的决策流程"""

    def test_anova_significant_then_posthoc(self, four_groups_with_difference):
        """
        正例：ANOVA 显著后做事后比较

        标准 ANOVA + 事后比较流程
        """
        # TODO: Uncomment when solution.py is created
        # groups = list(four_groups_with_difference.values())
        # anova_result = one_way_anova(*groups)
        #
        # if anova_result['reject_null']:
        #     # ANOVA 显著，做 Tukey HSD
        #     posthoc_result = tukey_hsd_test({'A': groups[0], 'B': groups[1],
        #                                       'C': groups[2], 'D': groups[3]})
        #     assert len(posthoc_result) == 6
        # else:
        #     # ANOVA 不显著，不做事后比较
        #     pass
        pytest.skip("solution.py not yet created")

    def test_anova_not_significant_no_posthoc(self, four_groups_no_difference):
        """
        边界：ANOVA 不显著时不做事后比较

        传统派观点：先 ANOVA，显著再做事后比较
        """
        # TODO: Uncomment when solution.py is created
        # groups = list(four_groups_no_difference.values())
        # anova_result = one_way_anova(*groups)
        #
        # # 如果 ANOVA 不显著，传统观点是不做事后比较
        # # 但现代观点也可以直接做 Tukey HSD
        # # 这里只检查决策函数的行为
        # decision = should_do_posthoc(anova_result)
        # # 传统派：ANOVA 不显著时不做事后比较
        # assert decision is False
        pytest.skip("solution.py not yet created")

    def test_explore_all_pairs_directly(self, tukey_no_significant_pairs):
        """
        正例：直接比较所有对（现代派观点）

        如果研究问题就是"比较所有对"，可以直接做 Tukey HSD
        """
        # TODO: Uncomment when solution.py is created
        # groups_dict = tukey_no_significant_pairs
        # result = tukey_hsd_test(groups_dict)
        # # 现代派：不管 ANOVA 如何，直接做 Tukey HSD
        # assert len(result) == 6
        pytest.skip("solution.py not yet created")


# =============================================================================
# 4. 校正方法测试
# =============================================================================

class TestBonferroniCorrection:
    """测试 Bonferroni 校正"""

    def test_bonferroni_single_test(self, single_p_value):
        """
        边界：单个检验时校正无效

        m=1 时，校正后 alpha = alpha，p 值不变
        """
        # TODO: Uncomment when solution.py is created
        # result = bonferroni_correction([0.03], alpha=0.05)
        # # 单个检验，校正后阈值不变
        # assert result['adjusted_alpha'] == 0.05
        # assert result['rejected'][0] == (0.03 < 0.05)
        pytest.skip("solution.py not yet created")

    def test_bonferroni_multiple_tests(self, p_values_for_correction):
        """
        正例：多个检验时正确校正

        校正后 alpha = alpha / m
        """
        # TODO: Uncomment when solution.py is created
        # p_values = p_values_for_correction
        # result = bonferroni_correction(p_values, alpha=0.05)
        # m = len(p_values)
        # expected_alpha = 0.05 / m
        # assert result['adjusted_alpha'] == pytest.approx(expected_alpha)
        pytest.skip("solution.py not yet created")

    def test_bonferroni_adjusted_p_values(self, p_values_for_correction):
        """
        正例：校正后的 p 值

        p_adjusted = p_original * m（ capped at 1）
        """
        # TODO: Uncomment when solution.py is created
        # p_values = p_values_for_correction
        # result = bonferroni_correction(p_values, alpha=0.05)
        # m = len(p_values)
        # for i, p_orig in enumerate(p_values):
        #     expected_p = min(p_orig * m, 1.0)
        #     assert result['p_adjusted'][i] == pytest.approx(expected_p)
        pytest.skip("solution.py not yet created")

    def test_bonferroni_conservative(self, many_p_values):
        """
        正例：Bonferroni 过于保守

        当 m 很大时，Bonferroni 会拒绝很少假设
        """
        # TODO: Uncomment when solution.py is created
        # p_values = many_p_values
        # result_bonf = bonferroni_correction(p_values, alpha=0.05)
        # result_uncorrected = {'rejected': p_values < 0.05}
        #
        # # Bonferroni 拒绝的假设数应 <= 未校正拒绝的假设数
        # assert result_bonf['rejected'].sum() <= result_uncorrected['rejected'].sum()
        pytest.skip("solution.py not yet created")

    def test_bonferroni_all_significant(self, all_significant_p_values):
        """
        正例：全部显著时仍应通过

        即使校正后，非常小的 p 值仍应显著
        """
        # TODO: Uncomment when solution.py is created
        # p_values = all_significant_p_values
        # result = bonferroni_correction(p_values, alpha=0.05)
        # # 这些 p 值很小，即使乘以 8 仍 < 0.05
        # assert result['rejected'].all()
        pytest.skip("solution.py not yet created")

    def test_bonferroni_all_nonsignificant(self, all_nonsignificant_p_values):
        """
        正例：全部不显著时仍不显著

        校正后仍不应显著
        """
        # TODO: Uncomment when solution.py is created
        # p_values = all_nonsignificant_p_values
        # result = bonferroni_correction(p_values, alpha=0.05)
        # assert not result['rejected'].any()
        pytest.skip("solution.py not yet created")

    def test_bonferroni_boundary(self, boundary_p_values):
        """
        边界：边界附近的 p 值

        p ≈ 0.05 时，校正后可能不显著
        """
        # TODO: Uncomment when solution.py is created
        # p_values = boundary_p_values
        # result = bonferroni_correction(p_values, alpha=0.05)
        # # m=5, alpha_adj = 0.01
        # # 只有 p < 0.01 的才显著
        # # 第一个 p=0.04 可能不显著了
        # assert not result['rejected'][0]  # p=0.04 * 5 = 0.20 > 0.05
        pytest.skip("solution.py not yet created")


class TestFDRCorrection:
    """测试 FDR (Benjamini-Hochberg) 校正"""

    def test_fdr_single_test(self, single_p_value):
        """
        边界：单个检验时 FDR ≈ Bonferroni

        m=1 时，FDR 和 Bonferroni 结果相同
        """
        # TODO: Uncomment when solution.py is created
        # p_values = single_p_value
        # result_fdr = fdr_correction(p_values, alpha=0.05)
        # result_bonf = bonferroni_correction(p_values, alpha=0.05)
        # # 单个检验时，结果应相同
        # assert result_fdr['rejected'][0] == result_bonf['rejected'][0]
        pytest.skip("solution.py not yet created")

    def test_fdr_less_conservative(self, many_p_values):
        """
        正例：FDR 比 Bonferroni 更不保守

        FDR 应拒绝更多假设
        """
        # TODO: Uncomment when solution.py is created
        # p_values = many_p_values
        # result_fdr = fdr_correction(p_values, alpha=0.05)
        # result_bonf = bonferroni_correction(p_values, alpha=0.05)
        #
        # # FDR 拒绝的假设数应 >= Bonferroni 拒绝的假设数
        # assert result_fdr['rejected'].sum() >= result_bonf['rejected'].sum()
        pytest.skip("solution.py not yet created")

    def test_fdr_sorted_order(self, p_values_for_correction):
        """
        正例：FDR 考虑 p 值排序

        BH 方法按 p 值从小到大排序后决定拒绝
        """
        # TODO: Uncomment when solution.py is created
        # p_values = p_values_for_correction
        # result = fdr_correction(p_values, alpha=0.05)
        # # BH 方法的特性：如果第 k 个被拒绝，
        # # 所有 p 值更小的也应被拒绝
        # rejected = result['rejected']
        # if rejected.any():
        #     # 找到第一个不拒绝的位置
        #     first_not_rejected = np.where(~rejected)[0]
        #     if len(first_not_rejected) > 0:
        #         # 在第一个不拒绝之后，不应再有拒绝
        #         assert not rejected[first_not_rejected[0]:].any()
        pytest.skip("solution.py not yet created")

    def test_fdr_all_significant(self, all_significant_p_values):
        """
        正例：全部显著时应全部通过

        """
        # TODO: Uncomment when solution.py is created
        # p_values = all_significant_p_values
        # result = fdr_correction(p_values, alpha=0.05)
        # assert result['rejected'].all()
        pytest.skip("solution.py not yet created")

    def test_fdr_all_nonsignificant(self, all_nonsignificant_p_values):
        """
        正例：全部不显著时应全部不通过

        """
        # TODO: Uncomment when solution.py is created
        # p_values = all_nonsignificant_p_values
        # result = fdr_correction(p_values, alpha=0.05)
        # assert not result['rejected'].any()
        pytest.skip("solution.py not yet created")


class TestCorrectionMethodComparison:
    """比较不同校正方法"""

    def test_methods_ranking(self, many_p_values):
        """
        正例：不同方法的保守程度排序

        未校正 > FDR > Bonferroni（拒绝假设数）
        """
        # TODO: Uncomment when solution.py is created
        # p_values = many_p_values
        #
        # # 未校正
        # uncorrected_rejected = (p_values < 0.05).sum()
        #
        # # Bonferroni
        # result_bonf = bonferroni_correction(p_values, alpha=0.05)
        # bonf_rejected = result_bonf['rejected'].sum()
        #
        # # FDR
        # result_fdr = fdr_correction(p_values, alpha=0.05)
        # fdr_rejected = result_fdr['rejected'].sum()
        #
        # # 未校正 >= FDR >= Bonferroni
        # assert uncorrected_rejected >= fdr_rejected >= bonf_rejected
        pytest.skip("solution.py not yet created")

    def test_choose_correction_small_m(self, multiple_test_results):
        """
        正例：检验次数少时用 Bonferroni

        m < 10 时推荐 Bonferroni
        """
        # TODO: Uncomment when solution.py is created
        # m = len(multiple_test_results)
        # recommendation = choose_correction_method(m, study_type='confirmatory')
        # assert recommendation == 'bonferroni' or recommendation == 'tukey_hsd'
        pytest.skip("solution.py not yet created")

    def test_choose_correction_large_m(self, many_test_results):
        """
        正例：检验次数多时用 FDR

        m > 50 时推荐 FDR
        """
        # TODO: Uncomment when solution.py is created
        # m = len(many_test_results)
        # recommendation = choose_correction_method(m, study_type='exploratory')
        # assert recommendation == 'fdr'
        pytest.skip("solution.py not yet created")


# =============================================================================
# 5. AI 审查测试
# =============================================================================

class TestAIReportReview:
    """测试 AI 生成报告的审查"""

    def test_review_good_report(self, ai_anova_good_report):
        """
        正例：合格报告应通过审查

        """
        # TODO: Uncomment when solution.py is created
        # report = ai_anova_good_report
        # review = review_ai_anova_report(report)
        # assert review['passes'] is True
        # assert len(review['issues']) == 0
        pytest.skip("solution.py not yet created")

    def test_review_bad_report(self, ai_anova_bad_report):
        """
        正例：有问题的报告应被发现

        """
        # TODO: Uncomment when solution.py is created
        # report = ai_anova_bad_report
        # review = review_ai_anova_report(report)
        # assert review['passes'] is False
        # assert len(review['issues']) > 0
        # # 应发现问题：未校正、缺少效应量、过度解读
        # assert any('校正' in issue or 'correction' in issue.lower()
        #            for issue in review['issues'])
        pytest.skip("solution.py not yet created")

    def test_detect_missing_correction(self, ai_posthoc_missing_correction):
        """
        正例：检测缺少多重比较校正

        """
        # TODO: Uncomment when solution.py is created
        # report = ai_posthoc_missing_correction
        # review = review_ai_anova_report(report)
        # assert '缺少校正' in review['issues'] or '未校正' in review['issues']
        pytest.skip("solution.py not yet created")

    def test_detect_missing_effect_size(self, ai_anova_bad_report):
        """
        正例：检测缺少效应量

        """
        # TODO: Uncomment when solution.py is created
        # report = ai_anova_bad_report
        # review = review_ai_anova_report(report)
        # assert '效应量' in review['issues'] or 'effect size' in review['issues']
        pytest.skip("solution.py not yet created")

    def test_detect_overinterpretation(self, ai_anova_bad_report):
        """
        正例：检测过度解读

        """
        # TODO: Uncomment when solution.py is created
        # report = ai_anova_bad_report
        # review = review_ai_anova_report(report)
        # assert '过度解读' in review['issues'] or '全面切换' in review['issues']
        pytest.skip("solution.py not yet created")

    def test_detect_skipped_anova(self, ai_anova_bad_report):
        """
        正例：检测跳过 ANOVA 直接做两两比较

        """
        # TODO: Uncomment when solution.py is created
        # report = ai_anova_bad_report
        # review = review_ai_anova_report(report)
        # assert '跳过 ANOVA' in review['issues'] or 'multiple_t_tests' in review['issues']
        pytest.skip("solution.py not yet created")


# =============================================================================
# 6. 综合流程测试
# =============================================================================

class TestCompleteANOVAWorkflow:
    """测试完整的 ANOVA 工作流"""

    def test_complete_workflow(self, four_groups_with_difference):
        """
        正例：完整流程

        描述统计 → 前提假设检查 → ANOVA → 效应量 → 事后比较
        """
        # TODO: Uncomment when solution.py is created
        # groups_dict = four_groups_with_difference
        # result = complete_anova_workflow(groups_dict, alpha=0.05)
        #
        # assert 'descriptive' in result
        # assert 'assumptions' in result
        # assert 'anova' in result
        # assert 'effect_size' in result
        # assert 'posthoc' in result if result['anova']['reject_null'] else True
        pytest.skip("solution.py not yet created")

    def test_workflow_with_assumption_violation(self, unequal_variance_groups):
        """
        正例：前提假设违反时使用非参数方法

        """
        # TODO: Uncomment when solution.py is created
        # groups_dict = unequal_variance_groups
        # result = complete_anova_workflow(groups_dict, alpha=0.05)
        #
        # # 方差不齐时应使用 Kruskal-Wallis
        # assert result['test_method'] == 'Kruskal-Wallis'
        pytest.skip("solution.py not yet created")

    def test_workflow_generate_report(self, four_groups_with_difference):
        """
        正例：生成 Markdown 报告

        """
        # TODO: Uncomment when solution.py is created
        # groups_dict = four_groups_with_difference
        # report = generate_anova_report(groups_dict, alpha=0.05)
        #
        # assert '# ANOVA 报告' in report or '## ' in report
        # assert 'F 统计量' in report or 'F-statistic' in report
        # assert 'p 值' in report or 'p-value' in report
        pytest.skip("solution.py not yet created")


# =============================================================================
# 7. StatLab 项目测试
# =============================================================================

class TestStatLabIntegration:
    """测试 StatLab 项目集成"""

    def test_statlab_channel_comparison(self, statlab_anova_data):
        """
        正例：多渠道转化率比较

        """
        # TODO: Uncomment when solution.py is created
        # df = statlab_anova_data
        # result = statlab_compare_channels(df, channel_col='channel',
        #                                     converted_col='converted')
        #
        # assert 'anova_result' in result
        # assert 'posthoc_result' in result
        # assert 'recommendation' in result
        pytest.skip("solution.py not yet created")

    def test_statlab_user_segment_comparison(self, statlab_user_segment_data):
        """
        正例：用户群组消费金额比较

        """
        # TODO: Uncomment when solution.py is created
        # df = statlab_user_segment_data
        # result = statlab_compare_segments(df, segment_col='segment',
        #                                    amount_col='amount')
        #
        # assert 'anova_result' in result
        # assert 'effect_size' in result
        pytest.skip("solution.py not yet created")

    def test_statlab_generate_section(self, statlab_anova_data):
        """
        正例：生成 StatLab 报告的多组比较章节

        """
        # TODO: Uncomment when solution.py is created
        # df = statlab_anova_data
        # section = statlab_generate_anova_section(df, tests=[
        #     ('channel', 'converted')
        # ])
        #
        # assert '## 多组比较' in section
        # assert 'ANOVA' in section
        # assert 'Tukey HSD' in section or '校正' in section
        pytest.skip("solution.py not yet created")

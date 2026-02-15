"""
Test suite for ANOVA (Week 07)

Focus: One-way ANOVA, F-statistic, effect size, and assumptions
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

import sys
from pathlib import Path

# Add starter_code to path
sys.path.insert(0, str(Path(__file__).parent.parent / "starter_code"))


class TestOneWayANOVABasic:
    """测试单因素 ANOVA 基本功能"""

    def test_anova_returns_required_fields(self, four_groups_with_difference):
        """
        正例：ANOVA 应返回必要字段

        至少包含：f_statistic, p_value, reject_null
        """
        # TODO: Uncomment when solution.py is created
        # groups = list(four_groups_with_difference.values())
        # result = one_way_anova(*groups)
        #
        # assert 'f_statistic' in result
        # assert 'p_value' in result
        # assert 'reject_null' in result
        pytest.skip("solution.py not yet created")

    def test_anova_null_hypothesis_true(self, four_groups_no_difference):
        """
        正例：原假设成立时 p 值应均匀分布

        四组来自同一分布，p 值应 > 0.05（大部分情况下）
        注意：由于随机性，约 5% 的情况会 p < 0.05
        """
        # TODO: Uncomment when solution.py is created
        # groups = list(four_groups_no_difference.values())
        # result = one_way_anova(*groups)
        #
        # # 不强制 p > 0.05，但检查结果合理
        # assert 0 <= result['p_value'] <= 1
        # assert result['f_statistic'] >= 0
        pytest.skip("solution.py not yet created")

    def test_anova_alternative_hypothesis_true(self, four_groups_with_difference):
        """
        正例：备择假设成立时应检测到显著

        D 组均值明显不同，ANOVA p 值应较小
        """
        # TODO: Uncomment when solution.py is created
        # groups = list(four_groups_with_difference.values())
        # result = one_way_anova(*groups, alpha=0.05)
        #
        # # D 组均值高 15，应有统计显著性
        # assert result['p_value'] < 0.05
        # assert result['reject_null'] is True
        # assert result['f_statistic'] > 1
        pytest.skip("solution.py not yet created")

    def test_anova_small_difference_may_not_detect(self, four_groups_small_difference):
        """
        边界：小差异可能检测不到

        效应量小时，ANOVA 可能不显著
        """
        # TODO: Uncomment when solution.py is created
        # groups = list(four_groups_small_difference.values())
        # result = one_way_anova(*groups, alpha=0.05)
        #
        # # 小差异可能不显著（取决于样本量和方差）
        # # 不强制要求，只检查结构
        # assert 'p_value' in result
        # assert 'f_statistic' in result
        pytest.skip("solution.py not yet created")


class TestFStatistic:
    """测试 F 统计量"""

    def test_f_statistic_formula(self):
        """
        正例：验证 F 统计量计算公式

        F = 组间方差 / 组内方差
        """
        # TODO: Uncomment when solution.py is created
        # # 构造简单数据验证
        # group_a = np.array([1, 2, 3, 4, 5])
        # group_b = np.array([2, 3, 4, 5, 6])
        # group_c = np.array([3, 4, 5, 6, 7])
        #
        # result = one_way_anova(group_a, group_b, group_c)
        #
        # # F 应 > 0
        # assert result['f_statistic'] > 0
        #
        # # 可以验证 F = t²（两组时）
        # result_2group = one_way_anova(group_a, group_b)
        # t_stat, _ = stats.ttest_ind(group_a, group_b)
        # assert result_2group['f_statistic'] == pytest.approx(t_stat**2, rel=0.01)
        pytest.skip("solution.py not yet created")

    def test_f_statistic_no_difference(self, four_groups_no_difference):
        """
        正例：无差异时 F ≈ 1

        组间方差 ≈ 组内方差时，F 接近 1
        """
        # TODO: Uncomment when solution.py is created
        # groups = list(four_groups_no_difference.values())
        # result = one_way_anova(*groups)
        #
        # # 无差异时，F 应接近 1
        # # 允许一定的抽样误差
        # assert 0.5 <= result['f_statistic'] <= 2.0
        pytest.skip("solution.py not yet created")

    def test_f_statistic_large_difference(self, four_groups_with_difference):
        """
        正例：大差异时 F >> 1

        组间方差 >> 组内方差时，F 远大于 1
        """
        # TODO: Uncomment when solution.py is created
        # groups = list(four_groups_with_difference.values())
        # result = one_way_anova(*groups)
        #
        # # 有差异时，F 应明显大于 1
        # assert result['f_statistic'] > 2
        pytest.skip("solution.py not yet created")

    def test_f_statistic_identical_groups(self, identical_groups):
        """
        边界：完全相同的组，F = 0 或接近 0

        组间方差为 0 时，F = 0
        """
        # TODO: Uncomment when solution.py is created
        # groups = list(identical_groups.values())
        # result = one_way_anova(*groups)
        #
        # # 组间方差为 0
        # assert result['f_statistic'] < 1e-10
        pytest.skip("solution.py not yet created")


class TestANOVAEffectSize:
    """测试 ANOVA 效应量（η²）"""

    def test_eta_squared_formula(self, four_groups_with_difference):
        """
        正例：验证 η² 计算公式

        η² = 组间平方和 / 总平方和
        """
        # TODO: Uncomment when solution.py is created
        # groups = list(four_groups_with_difference.values())
        # result = one_way_anova(*groups)
        #
        # # η² 应在 [0, 1] 范围内
        # eta_sq = result.get('eta_squared') or calculate_eta_squared(groups)
        # assert 0 <= eta_sq <= 1
        pytest.skip("solution.py not yet created")

    def test_eta_squared_no_variation(self, identical_groups):
        """
        边界：无变异时 η² = 0

        """
        # TODO: Uncomment when solution.py is created
        # groups = list(identical_groups.values())
        # eta_sq = calculate_eta_squared(groups)
        # assert eta_sq == 0.0
        pytest.skip("solution.py not yet created")

    def test_eta_squared_large_effect(self):
        """
        正例：大效应时 η² > 0.14

        """
        # TODO: Uncomment when solution.py is created
        # group_a = np.array([0, 0, 0, 0, 0])
        # group_b = np.array([100, 100, 100, 100, 100])
        # group_c = np.array([200, 200, 200, 200, 200])
        #
        # eta_sq = calculate_eta_squared([group_a, group_b, group_c])
        # # 几乎所有变异都来自组间
        # assert eta_sq > 0.9
        pytest.skip("solution.py not yet created")

    def test_interpret_eta_squared_thresholds(self):
        """
        正例：η² 的解释阈值

        小效应：η² ≈ 0.01
        中等效应：η² ≈ 0.06
        大效应：η² ≈ 0.14
        """
        # TODO: Uncomment when solution.py is created
        # assert interpret_eta_squared(0.01) == "小效应"
        # assert interpret_eta_squared(0.06) == "中等效应"
        # assert interpret_eta_squared(0.14) == "大效应"
        pytest.skip("solution.py not yet created")


class TestANOVAAssumptions:
    """测试 ANOVA 前提假设检查"""

    def test_check_normality_each_group(self, four_groups_with_difference):
        """
        正例：检查每组的正态性

        使用 Shapiro-Wilk 检验
        """
        # TODO: Uncomment when solution.py is created
        # groups_dict = four_groups_with_difference
        # results = check_anova_normality(groups_dict, alpha=0.05)
        #
        # # 应返回每组的结果
        # assert len(results) == 4
        # for group_name, result in results.items():
        #     assert 'p_value' in result
        #     assert 'is_normal' in result
        pytest.skip("solution.py not yet created")

    def test_check_homogeneity_variance(self, four_groups_with_difference):
        """
        正例：检查方差齐性

        使用 Levene 检验
        """
        # TODO: Uncomment when solution.py is created
        # groups = list(four_groups_with_difference.values())
        # result = check_homogeneity_variance(*groups, alpha=0.05)
        #
        # assert 'p_value' in result
        # assert 'equal_variance' in result
        # assert 'test' in result
        # assert result['test'] == 'Levene'
        pytest.skip("solution.py not yet created")

    def test_detect_unequal_variance(self, unequal_variance_groups):
        """
        正例：检测方差不齐

        """
        # TODO: Uncomment when solution.py is created
        # groups = list(unequal_variance_groups.values())
        # result = check_homogeneity_variance(*groups, alpha=0.05)
        #
        # # 方差显著不同
        # assert result['equal_variance'] is False
        # assert result['p_value'] < 0.05
        pytest.skip("solution.py not yet created")

    def test_assumption_violation_recommendation(self, unequal_variance_groups):
        """
        正例：假设违反时的建议

        """
        # TODO: Uncomment when solution.py is created
        # groups = list(unequal_variance_groups.values())
        # result = complete_anova_workflow({'A': groups[0], 'B': groups[1],
        #                                    'C': groups[2], 'D': groups[3]})
        #
        # # 方差不齐时，应建议使用 Welch ANOVA 或 Kruskal-Wallis
        # if not result['assumptions']['homogeneity']['equal_variance']:
        #     assert 'Welch' in result['recommendation'] or \
        #            'Kruskal' in result['recommendation'] or \
        #            '非参数' in result['recommendation']
        pytest.skip("solution.py not yet created")

    def test_skewed_data_normality_test(self, skewed_groups):
        """
        正例：偏态数据无法通过正态性检验

        """
        # TODO: Uncomment when solution.py is created
        # groups_dict = skewed_groups
        # results = check_anova_normality(groups_dict, alpha=0.05)
        #
        # # 指数分布数据通常不通过正态性检验
        # for group_name, result in results.items():
        #     # 偏态数据 p 值应很小
        #     assert result['p_value'] < 0.05
        #     assert result['is_normal'] is False
        pytest.skip("solution.py not yet created")


class TestKruskalWallisAlternative:
    """测试 Kruskal-Wallis 非参数替代"""

    def test_kruskal_wallis_signature(self, four_groups_with_difference):
        """
        正例：Kruskal-Wallis 应返回类似 ANOVA 的字段

        """
        # TODO: Uncomment when solution.py is created
        # groups = list(four_groups_with_difference.values())
        # result = kruskal_wallis_test(*groups)
        #
        # assert 'h_statistic' in result or 'statistic' in result
        # assert 'p_value' in result
        # assert 'reject_null' in result
        pytest.skip("solution.py not yet created")

    def test_kruskal_wallis_detects_difference(self, four_groups_with_difference):
        """
        正例：Kruskal-Wallis 应检测到差异

        """
        # TODO: Uncomment when solution.py is created
        # groups = list(four_groups_with_difference.values())
        # result = kruskal_wallis_test(*groups, alpha=0.05)
        #
        # # 功效可能低于 ANOVA，但大差异仍应检测到
        # assert result['p_value'] < 0.05
        pytest.skip("solution.py not yet created")

    def test_kruskal_wallis_handles_skewed_data(self, skewed_groups):
        """
        正例：Kruskal-Wallis 适用于偏态数据

        """
        # TODO: Uncomment when solution.py is created
        # groups = list(skewed_groups.values())
        # result = kruskal_wallis_test(*groups)
        #
        # # 应能正常运行
        # assert 'p_value' in result
        # assert 'h_statistic' in result or 'statistic' in result
        pytest.skip("solution.py not yet created")

    def test_choose_test_auto(self):
        """
        正例：根据假设自动选择检验方法

        """
        # TODO: Uncomment when solution.py is created
        # # 正态 + 方差齐 → ANOVA
        # normal_groups = {
        #     'A': np.random.normal(100, 10, 50),
        #     'B': np.random.normal(100, 10, 50),
        # }
        # result1 = choose_anova_test_auto(normal_groups, alpha=0.05)
        # assert result1['test_method'] == 'ANOVA' or 'one_way_anova' in result1['test_method'].lower()
        #
        # # 偏态 → Kruskal-Wallis
        # skewed_groups = {
        #     'A': np.random.exponential(30, 50),
        #     'B': np.random.exponential(35, 50),
        # }
        # result2 = choose_anova_test_auto(skewed_groups, alpha=0.05)
        # assert 'Kruskal' in result2['test_method']
        pytest.skip("solution.py not yet created")


class TestANOVAWithBinaryData:
    """测试二元数据的 ANOVA"""

    def test_anova_binary_data_works(self, binary_conversion_groups):
        """
        正例：ANOVA 可用于二元数据（虽然不是最佳）

        转化率是 0/1 数据
        """
        # TODO: Uncomment when solution.py is created
        # groups = list(binary_conversion_groups.values())
        # result = one_way_anova(*groups)
        #
        # assert 'p_value' in result
        # assert 'f_statistic' in result
        pytest.skip("solution.py not yet created")

    def test_binary_data_recommends_alternative(self, binary_conversion_groups):
        """
        正例：二元数据应建议使用比例检验

        """
        # TODO: Uncomment when solution.py is created
        # groups = list(binary_conversion_groups.values())
        # result = complete_anova_workflow({'A': groups[0], 'B': groups[1],
        #                                    'C': groups[2], 'D': groups[3]})
        #
        # # 二元数据应建议使用卡方检验或比例检验
        # assert 'warning' in result or 'recommendation' in result
        # if 'warning' in result:
        #     assert '二元' in result['warning'] or '比例' in result['warning']
        pytest.skip("solution.py not yet created")


class TestANOVAFromDataFrame:
    """测试从 DataFrame 执行 ANOVA"""

    def test_anova_from_dataframe(self, anova_dataframe):
        """
        正例：从 DataFrame 执行 ANOVA

        实际应用中数据常以 DataFrame 格式存储
        """
        # TODO: Uncomment when solution.py is created
        # df = anova_dataframe
        # result = one_way_anova_from_df(df, group_col='group', value_col='value')
        #
        # assert 'p_value' in result
        # assert 'f_statistic' in result
        # assert 'groups' in result or 'n_groups' in result
        pytest.skip("solution.py not yet created")

    def test_anova_dataframe_invalid_column(self, anova_dataframe):
        """
        反例：列名不存在时应报错

        """
        # TODO: Uncomment when solution.py is created
        # df = anova_dataframe
        # with pytest.raises(KeyError, match="column|列"):
        #     one_way_anova_from_df(df, group_col='invalid', value_col='value')
        pytest.skip("solution.py not yet created")


class TestANOVAReporting:
    """测试 ANOVA 报告生成"""

    def test_generate_anova_report(self, four_groups_with_difference):
        """
        正例：生成 ANOVA 报告

        """
        # TODO: Uncomment when solution.py is created
        # groups_dict = four_groups_with_difference
        # report = generate_anova_report(groups_dict, alpha=0.05)
        #
        # assert 'ANOVA' in report
        # assert 'F 统计量' in report or 'F-statistic' in report
        # assert 'p 值' in report or 'p-value' in report
        # assert 'η²' in report or 'eta' in report.lower() or '效应量' in report
        pytest.skip("solution.py not yet created")

    def test_report_includes_assumptions(self, four_groups_with_difference):
        """
        正例：报告应包含假设检查结果

        """
        # TODO: Uncomment when solution.py is created
        # groups_dict = four_groups_with_difference
        # report = generate_anova_report(groups_dict, include_assumptions=True)
        #
        # assert '正态性' in report or 'normality' in report.lower()
        # assert '方差齐性' in report or 'homogeneity' in report.lower() or 'Levene' in report
        pytest.skip("solution.py not yet created")

    def test_report_includes_posthoc_if_significant(self, four_groups_with_difference):
        """
        正例：ANOVA 显著时报告应包含事后比较

        """
        # TODO: Uncomment when solution.py is created
        # groups_dict = four_groups_with_difference
        # report = generate_anova_report(groups_dict, include_posthoc=True)
        #
        # # 如果 ANOVA 显著，应包含事后比较
        # # 不强制，因为可能 p 值恰好 > 0.05
        # # 但如果包含 posthoc 部分
        # if 'Tukey' in report or '事后' in report:
        #     assert 'HSD' in report or 'post-hoc' in report.lower()
        pytest.skip("solution.py not yet created")


class TestEducationalScenarios:
    """教育性场景测试"""

    def test_explain_f_statistic_to_student(self):
        """
        场景：向学生解释 F 统计量的含义

        """
        # TODO: Uncomment when solution.py is created
        # explanation = explain_f_statistic(f_stat=5.2)
        #
        # assert '组间' in explanation
        # assert '组内' in explanation
        # assert '方差' in explanation
        pytest.skip("solution.py not yet created")

    def test_explain_anova_vs_ttests(self):
        """
        场景：解释为什么用 ANOVA 而不是多次 t 检验

        """
        # TODO: Uncomment when solution.py is created
        # explanation = explain_anova_advantage(n_groups=5)
        #
        # assert '假阳性' in explanation or 'FWER' in explanation
        # assert '10 次' in explanation or 'C(5,2)' in explanation
        pytest.skip("solution.py not yet created")

    def test_interpret_anova_result_scenario(self):
        """
        场景：解释具体的 ANOVA 结果

        """
        # TODO: Uncomment when solution.py is created
        # result = {
        #     'f_statistic': 4.5,
        #     'p_value': 0.01,
        #     'eta_squared': 0.08,
        #     'reject_null': True
        # }
        #
        # interpretation = interpret_anova_result(result)
        #
        # assert '显著' in interpretation
        # assert '效应量' in interpretation or 'η²' in interpretation
        pytest.skip("solution.py not yet created")

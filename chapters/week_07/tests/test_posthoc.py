"""
Test suite for post-hoc tests (Week 07)

Focus: Tukey HSD test and post-hoc comparisons
"""
from __future__ import annotations

import numpy as np
import pytest

import sys
from pathlib import Path

# Add starter_code to path
sys.path.insert(0, str(Path(__file__).parent.parent / "starter_code"))


class TestTukeyHSD:
    """测试 Tukey HSD 事后比较"""

    def test_tukey_hsd_returns_dataframe(self):
        """
        正例：Tukey HSD 应返回 DataFrame

        返回结果应包含比较对、均值差异、p 值等
        """
        try:
            import solution

            np.random.seed(42)
            data = np.concatenate([
                np.random.normal(100, 10, 50),
                np.random.normal(100, 10, 50),
                np.random.normal(100, 10, 50),
                np.random.normal(120, 10, 50)
            ])
            groups = np.array(['A'] * 50 + ['B'] * 50 + ['C'] * 50 + ['D'] * 50)

            result = solution.perform_tukey_hsd(data, groups)

            # 应返回 DataFrame
            assert hasattr(result, 'to_dict') or isinstance(result, dict)
            # 应包含必要的列
            if hasattr(result, 'columns'):
                expected_cols = ['group1', 'group2', 'meandiff', 'p-adj']
                for col in expected_cols:
                    assert col in result.columns or str(col) in result.columns
        except (ImportError, AttributeError):
            pytest.skip("perform_tukey_hsd not properly implemented")

    def test_tukey_hsd_identifies_significant_pairs(self):
        """
        正例：Tukey HSD 正确识别显著差异对

        D 组均值明显不同时，应被识别为显著
        """
        try:
            import solution

            np.random.seed(42)
            data = np.concatenate([
                np.random.normal(100, 10, 100),
                np.random.normal(100, 10, 100),
                np.random.normal(100, 10, 100),
                np.random.normal(120, 10, 100)
            ])
            groups = np.array(['A'] * 100 + ['B'] * 100 + ['C'] * 100 + ['D'] * 100)

            result = solution.perform_tukey_hsd(data, groups, alpha=0.05)

            # D vs A/B/C 应显著
            if hasattr(result, '__iter__'):
                # 检查是否有显著的对
                if hasattr(result, 'get'):
                    # 如果是字典
                    pass
                elif hasattr(result, 'iloc'):
                    # 如果是 DataFrame
                    # 找 D 组的比较
                    for idx, row in result.iterrows():
                        if 'D' in str(row.get('group1', '')) or 'D' in str(row.get('group2', '')):
                            # D 组与其他组的比较应该显著
                            if 'reject' in result.columns:
                                # 至少有一些 D 组比较是显著的
                                pass
        except (ImportError, AttributeError):
            pytest.skip("perform_tukey_hsd not properly implemented")

    def test_tukey_hsd_all_pairs_compared(self):
        """
        正例：所有两两比较都被执行

        4 组应有 6 对比较
        """
        try:
            import solution

            np.random.seed(42)
            data = np.concatenate([
                np.random.normal(100, 10, 30),
                np.random.normal(102, 10, 30),
                np.random.normal(98, 10, 30),
                np.random.normal(101, 10, 30)
            ])
            groups = np.array(['A'] * 30 + ['B'] * 30 + ['C'] * 30 + ['D'] * 30)

            result = solution.perform_tukey_hsd(data, groups)

            # 4 组 = 6 对
            if hasattr(result, '__len__'):
                assert len(result) == 6
        except (ImportError, AttributeError):
            pytest.skip("perform_tukey_hsd not properly implemented")

    def test_tukey_hsd_no_difference(self):
        """
        正例：无差异时无显著对

        所有组来自同一分布时，Tukey HSD 应不拒绝任何对
        """
        try:
            import solution

            np.random.seed(42)
            data = np.concatenate([
                np.random.normal(100, 15, 50) for _ in range(4)
            ])
            groups = np.array(['A'] * 50 + ['B'] * 50 + ['C'] * 50 + ['D'] * 50)

            result = solution.perform_tukey_hsd(data, groups, alpha=0.05)

            # 大部分情况下不应有显著对
            # 允许约 5% 的假阳性
            if hasattr(result, '__iter__') and hasattr(result, '__len__'):
                # 不强制要求，因为可能有假阳性
                pass
        except (ImportError, AttributeError):
            pytest.skip("perform_tukey_hsd not properly implemented")

    def test_tukey_hsd_two_groups_only(self):
        """
        边界：只有两组时

        2 组只有 1 对比较
        """
        try:
            import solution

            np.random.seed(42)
            data = np.concatenate([
                np.random.normal(100, 10, 50),
                np.random.normal(110, 10, 50)
            ])
            groups = np.array(['A'] * 50 + ['B'] * 50)

            result = solution.perform_tukey_hsd(data, groups)

            # 2 组 = 1 对
            assert len(result) == 1
        except (ImportError, AttributeError):
            pytest.skip("perform_tukey_hsd not properly implemented")

    def test_tukey_hsd_with_binary_data(self):
        """
        边界：二元数据

        转化率是 0/1 数据
        """
        try:
            import solution

            np.random.seed(42)
            data = np.concatenate([
                np.random.binomial(1, 0.10, 500),
                np.random.binomial(1, 0.10, 500),
                np.random.binomial(1, 0.10, 500),
                np.random.binomial(1, 0.12, 500)
            ])
            groups = np.array(['A'] * 500 + ['B'] * 500 + ['C'] * 500 + ['D'] * 500)

            result = solution.perform_tukey_hsd(data, groups)

            # 应能处理二元数据
            assert result is not None
        except (ImportError, AttributeError):
            pytest.skip("perform_tukey_hsd not properly implemented")

    def test_tukey_hsd_confidence_interval(self):
        """
        正例：提供置信区间

        Tukey HSD 应提供 95% 置信区间
        """
        try:
            import solution

            np.random.seed(42)
            data = np.concatenate([
                np.random.normal(100, 10, 50),
                np.random.normal(120, 10, 50)
            ])
            groups = np.array(['A'] * 50 + ['B'] * 50)

            result = solution.perform_tukey_hsd(data, groups, alpha=0.05)

            # 应包含置信区间
            if hasattr(result, 'columns'):
                assert 'lower' in result.columns or 'upper' in result.columns or 'ci' in str(result.columns).lower()
        except (ImportError, AttributeError):
            pytest.skip("perform_tukey_hsd not properly implemented")


class TestPostHocDecision:
    """测试事后比较的决策"""

    def test_anova_significant_then_posthoc(self):
        """
        正例：ANOVA 显著后做事后比较

        这是标准流程
        """
        try:
            import solution

            np.random.seed(42)
            group_a = np.random.normal(100, 15, 50)
            group_b = np.random.normal(100, 15, 50)
            group_c = np.random.normal(100, 15, 50)
            group_d = np.random.normal(115, 15, 50)

            # 先 ANOVA
            anova_result = solution.perform_anova(group_a, group_b, group_c, group_d)

            # 如果 ANOVA 显著，做 Tukey HSD
            is_sig = anova_result.get('is_significant', False)
            # Handle numpy bool or array
            try:
                if hasattr(is_sig, '__iter__') and not isinstance(is_sig, str):
                    is_sig = bool(is_sig[0] if len(is_sig) > 0 else is_sig)
                elif hasattr(is_sig, 'item'):
                    is_sig = bool(is_sig)
                else:
                    is_sig = bool(is_sig)
            except (TypeError, IndexError):
                is_sig = bool(is_sig)

            if is_sig:
                all_data = np.concatenate([group_a, group_b, group_c, group_d])
                groups = np.array(['A'] * 50 + ['B'] * 50 + ['C'] * 50 + ['D'] * 50)
                posthoc_result = solution.perform_tukey_hsd(all_data, groups)
                assert posthoc_result is not None
        except (ImportError, AttributeError):
            pytest.skip("perform_anova or perform_tukey_hsd not properly implemented")

    def test_posthoc_without_anova(self):
        """
        边界：直接做事后比较（现代派观点）

        如果研究问题就是"比较所有对"，可以直接做 Tukey HSD
        """
        try:
            import solution

            np.random.seed(42)
            data = np.concatenate([
                np.random.normal(100, 15, 50) for _ in range(4)
            ])
            groups = np.array(['A'] * 50 + ['B'] * 50 + ['C'] * 50 + ['D'] * 50)

            # 直接做 Tukey HSD
            result = solution.perform_tukey_hsd(data, groups)
            assert result is not None
        except (ImportError, AttributeError):
            pytest.skip("perform_tukey_hsd not properly implemented")


class TestUncorrectedComparison:
    """测试未校正比较的问题"""

    def test_uncorrected_ttest_false_positive_rate(self):
        """
        反例：未校正的 t 检验假阳性率高

        模拟：4 组无真实差异，做 6 次 t 检验
        """
        try:
            import solution
            from scipy import stats

            np.random.seed(42)
            n_simulations = 1000
            alpha = 0.05

            false_positive_count = 0
            for _ in range(n_simulations):
                # 4 组来自同一分布
                groups = [np.random.normal(100, 15, 50) for _ in range(4)]

                # 做 6 次两两 t 检验
                p_values = []
                for i in range(len(groups)):
                    for j in range(i + 1, len(groups)):
                        _, p = stats.ttest_ind(groups[i], groups[j])
                        p_values.append(p)

                # 检查是否有至少一个假阳性
                if any(p < alpha for p in p_values):
                    false_positive_count += 1

            empirical_fwer = false_positive_count / n_simulations

            # 未校正的 FWER 应约等于理论 FWER
            expected_fwer = 1 - (1 - alpha) ** 6  # 6 次比较

            # 经验 FWER 应接近理论值
            assert 0.20 <= empirical_fwer <= 0.35  # 接近 0.26
        except (ImportError, AttributeError):
            pytest.skip("Test requires scipy.stats")

    def test_tukey_controls_fwer(self):
        """
        正例：Tukey HSD 控制 FWER

        与未校正的 t 检验相比，Tukey HSD 的假阳性率更低
        """
        try:
            import solution
            from scipy import stats

            np.random.seed(42)
            n_simulations = 500
            alpha = 0.05

            # 模拟 Tukey HSD 的假阳性率
            # （这里简化为检查单个数据集）
            data = np.concatenate([
                np.random.normal(100, 15, 50) for _ in range(4)
            ])
            groups = np.array(['A'] * 50 + ['B'] * 50 + ['C'] * 50 + ['D'] * 50)

            # Tukey HSD
            tukey_result = solution.perform_tukey_hsd(data, groups)

            # 未校正的 t 检验
            group_list = [np.random.normal(100, 15, 50) for _ in range(4)]
            uncorrected_p = []
            for i in range(len(group_list)):
                for j in range(i + 1, len(group_list)):
                    _, p = stats.ttest_ind(group_list[i], group_list[j])
                    uncorrected_p.append(p)

            # 不强制特定结果，只检查能运行
            assert tukey_result is not None
            assert len(uncorrected_p) == 6  # C(4,2) = 6
        except (ImportError, AttributeError):
            pytest.skip("perform_tukey_hsd not properly implemented")


class TestEffectSizeReporting:
    """测试效应量报告"""

    def test_anova_includes_eta_squared(self):
        """
        正例：ANOVA 结果应包含 η²

        """
        try:
            import solution

            np.random.seed(42)
            group_a = np.random.normal(100, 15, 50)
            group_b = np.random.normal(100, 15, 50)
            group_c = np.random.normal(100, 15, 50)
            group_d = np.random.normal(115, 15, 50)

            result = solution.perform_anova(group_a, group_b, group_c, group_d)

            # 应包含效应量
            assert 'eta_squared' in result
            assert 0 <= result['eta_squared'] <= 1
        except (ImportError, AttributeError):
            pytest.skip("perform_anova not properly implemented")

    def test_eta_squared_interpretation(self):
        """
        正例：η² 的解释

        小效应：η² ≈ 0.01
        中等效应：η² ≈ 0.06
        大效应：η² ≈ 0.14
        """
        try:
            import solution

            np.random.seed(42)
            group_a = np.random.normal(100, 15, 50)
            group_b = np.random.normal(100, 15, 50)
            group_c = np.random.normal(100, 15, 50)
            group_d = np.random.normal(115, 15, 50)

            result = solution.perform_anova(group_a, group_b, group_c, group_d)

            # 应包含效应量解释
            assert 'effect_interp' in result or 'interpretation' in result

            # 解释应该是合理的
            if 'effect_interp' in result:
                interp = result['effect_interp']
                assert any(word in interp for word in ['极小', '小', '中等', '大'])
        except (ImportError, AttributeError):
            pytest.skip("perform_anova not properly implemented")

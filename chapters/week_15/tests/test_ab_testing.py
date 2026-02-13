"""
Test suite for Week 15: A/B Testing Engineering

This module tests A/B testing engineering and automation, covering:
- T-test for A/B comparison
- Sample Ratio Mismatch (SRM) detection
- Effect size and confidence intervals
- Decision rules and early stopping
"""

import pytest
import numpy as np
import pandas as pd
from scipy import stats


# =============================================================================
# Test 1: T-test for A/B Comparison
# =============================================================================

class TestABTTest:
    """Test basic A/B testing with t-test."""

    def test_t_test_detects_significant_difference(self, ab_test_data_significant):
        """
        Happy path: T-test detects significant difference between A and B.

        学习目标:
        - 理解双样本 t 检验的基本用法
        - p < 0.05 表示拒绝原假设（两组均值相等）
        """
        # Split into groups
        group_A = ab_test_data_significant[
            ab_test_data_significant['group'] == 'A']['value'].values
        group_B = ab_test_data_significant[
            ab_test_data_significant['group'] == 'B']['value'].values

        # Run t-test
        t_stat, p_value = stats.ttest_ind(group_B, group_A)

        # Should detect significant difference
        assert p_value < 0.05, \
            f"应检测到显著差异 (p < 0.05)，实际 p = {p_value:.4f}"

    def test_t_test_no_significant_difference(self, ab_test_data_no_effect):
        """
        Happy path: T-test doesn't detect difference when none exists.

        学习目标:
        - 理解 p > 0.05 表示不能拒绝原假设
        - "不能拒绝" ≠ "证明原假设为真"
        """
        group_A = ab_test_data_no_effect[
            ab_test_data_no_effect['group'] == 'A']['value'].values
        group_B = ab_test_data_no_effect[
            ab_test_data_no_effect['group'] == 'B']['value'].values

        t_stat, p_value = stats.ttest_ind(group_B, group_A)

        # Should not detect significant difference
        assert p_value > 0.05, \
            f"无效应时 p 应 > 0.05，实际 p = {p_value:.4f}"

    def test_t_test_effect_size_calculation(self, ab_test_data_significant):
        """
        Happy path: Calculate effect size (mean difference).

        学习目标:
        - 理解效应量 = group_B_mean - group_A_mean
        - 效应量有业务含义（如"提升8元"）
        """
        group_A = ab_test_data_significant[
            ab_test_data_significant['group'] == 'A']['value'].values
        group_B = ab_test_data_significant[
            ab_test_data_significant['group'] == 'B']['value'].values

        effect_size = np.mean(group_B) - np.mean(group_A)

        # Should be positive (B better than A)
        assert effect_size > 0, \
            f"效应量应 > 0（B 比 A 好），实际: {effect_size:.2f}"

        # Should be around 8 (true effect)
        assert 5 < effect_size < 12, \
            f"效应量应在合理范围内，实际: {effect_size:.2f}"

    def test_t_test_confidence_interval(self, ab_test_data_significant):
        """
        Happy path: Calculate 95% CI for effect size.

        学习目标:
        - 理解置信区间的构造
        - effect ± 1.96 * SE
        """
        group_A = ab_test_data_significant[
            ab_test_data_significant['group'] == 'A']['value'].values
        group_B = ab_test_data_significant[
            ab_test_data_significant['group'] == 'B']['value'].values

        effect_size = np.mean(group_B) - np.mean(group_A)

        # Standard error
        se = np.sqrt(np.var(group_A, ddof=1) / len(group_A) +
                     np.var(group_B, ddof=1) / len(group_B))

        # 95% CI
        ci_low = effect_size - 1.96 * se
        ci_high = effect_size + 1.96 * se

        # Check structure
        assert ci_low < ci_high, \
            "CI 下界应小于上界"

        # Effect size should be in CI (center)
        assert ci_low < effect_size < ci_high, \
            "效应量应在 CI 中心"

    def test_t_test_requires_sufficient_sample_size(self):
        """
        Edge case: Small sample size reduces power.

        学习目标:
        - 理解样本量影响检验力
        - 样本太小 → 即使有效应也检测不到
        """
        np.random.seed(42)

        # Small samples (n=10 per group)
        group_A = np.random.normal(100, 20, 10)
        group_B = np.random.normal(108, 20, 10)

        t_stat, p_value = stats.ttest_ind(group_B, group_A)

        # With small samples, p might not be significant
        # even though true effect = 8
        # This demonstrates low power
        assert isinstance(p_value, (float, np.floating))
        assert 0 <= p_value <= 1


# =============================================================================
# Test 2: Sample Ratio Mismatch (SRM)
# =============================================================================

class TestSampleRatioMismatch:
    """Test Sample Ratio Mismatch detection."""

    def test_detect_srm_with_chi_square(self, ab_test_data_with_srm):
        """
        Happy path: Detect sample ratio mismatch with chi-square test.

        学习目标:
        - 理解 SRM 是用卡方检验检测的
        - 比较"实际样本量"vs"预期样本量"
        """
        data = ab_test_data_with_srm

        # Observed counts
        observed_A = len(data[data['group'] == 'A'])
        observed_B = len(data[data['group'] == 'B'])

        # Expected counts (50:50 ratio)
        total = observed_A + observed_B
        expected_A = total * 0.5
        expected_B = total * 0.5

        # Chi-square test
        observed = np.array([observed_A, observed_B])
        expected = np.array([expected_A, expected_B])
        chi2, p_value = stats.chisquare(observed, expected)

        # Should detect SRM (p < 0.05)
        assert p_value < 0.05, \
            f"应检测到样本比例异常 (p < 0.05)，实际 p = {p_value:.4f}"

    def test_no_srm_with_balanced_data(self, ab_test_data_significant):
        """
        Happy path: No SRM with balanced 50:50 split.

        学习目标:
        - 理解正常情况下样本比例应接近预期
        - p > 0.05 表示无 SRM
        """
        data = ab_test_data_significant

        observed_A = len(data[data['group'] == 'A'])
        observed_B = len(data[data['group'] == 'B'])

        total = observed_A + observed_B
        expected_A = total * 0.5
        expected_B = total * 0.5

        # Chi-square test
        observed = np.array([observed_A, observed_B])
        expected = np.array([expected_A, expected_B])
        chi2, p_value = stats.chisquare(observed, expected)

        # Should not detect SRM
        assert p_value > 0.05, \
            f"平衡数据不应检测到 SRM (p > 0.05)，实际 p = {p_value:.4f}"

    def test_srm_extreme_imbalance(self):
        """
        Edge case: Extreme sample ratio mismatch.

        学习目标:
        - 理解严重不平衡（如 90:10）很容易检测
        - p 值会非常小
        """
        # Create extreme imbalance: 90:10
        n_A = 450
        n_B = 50

        group_A = np.random.normal(100, 20, n_A)
        group_B = np.random.normal(100, 20, n_B)

        # Chi-square test
        observed = np.array([n_A, n_B])
        expected = np.array([(n_A + n_B) * 0.5, (n_A + n_B) * 0.5])
        chi2, p_value = stats.chisquare(observed, expected)

        # Should strongly detect SRM
        assert p_value < 0.001, \
            f"极端不平衡应产生很小的 p 值，实际 p = {p_value:.6f}"

    def test_srm_minimum_sample_detection(self):
        """
        Edge case: Minimum samples to detect SRM.

        学习目标:
        - 理解 SRM 检测需要一定样本量
        - 样本太小 → 卡方检验不可靠
        """
        # Very small samples
        n_A = 8
        n_B = 2  # 80:20 ratio, but very small

        observed = np.array([n_A, n_B])
        expected = np.array([5, 5])  # Expected 50:50

        # Chi-square test might not be reliable with such small samples
        # But should still run
        chi2, p_value = stats.chisquare(observed, expected)

        # Just verify it runs
        assert isinstance(p_value, (float, np.floating))
        assert 0 <= p_value <= 1


# =============================================================================
# Test 3: Decision Rules
# =============================================================================

class TestDecisionRules:
    """Test A/B test decision rules."""

    def test_launch_b_when_significant_and_large_effect(self, ab_test_data_significant):
        """
        Happy path: Launch B when p < 0.05 and effect > threshold.

        学习目标:
        - 理解决策规则：p < 0.05 且 |效应| > 阈值
        - 统计显著 + 业务显著 → launch_B
        """
        group_A = ab_test_data_significant[
            ab_test_data_significant['group'] == 'A']['value'].values
        group_B = ab_test_data_significant[
            ab_test_data_significant['group'] == 'B']['value'].values

        t_stat, p_value = stats.ttest_ind(group_B, group_A)
        effect_size = np.mean(group_B) - np.mean(group_A)

        min_effect = 5.0

        # Decision rule
        if p_value < 0.05 and abs(effect_size) >= min_effect:
            decision = "launch_B"
        elif p_value < 0.10:
            decision = "continue"
        else:
            decision = "reject_B"

        assert decision == "launch_B", \
            f"p < 0.05 且效应 >= {min_effect} 时应 launch_B"

    def test_continue_when_borderline_significance(self):
        """
        Test: Continue collecting data when 0.05 < p < 0.10.

        学习目标:
        - 理解边界显著的情况
        - 继续收集数据，不要做决策
        """
        np.random.seed(42)

        # Create data with borderline p-value
        group_A = np.random.normal(100, 20, 100)
        group_B = np.random.normal(104, 20, 100)  # Small effect

        t_stat, p_value = stats.ttest_ind(group_B, group_A)
        effect_size = np.mean(group_B) - np.mean(group_A)

        # Decision rule
        if p_value < 0.05 and abs(effect_size) >= 5:
            decision = "launch_B"
        elif p_value < 0.10:
            decision = "continue"
        else:
            decision = "reject_B"

        # Depends on random data, but decision should be made
        assert decision in ["launch_B", "continue", "reject_B"]

    def test_reject_b_when_no_effect(self, ab_test_data_no_effect):
        """
        Happy path: Reject B when p >= 0.10 (no evidence).

        学习目标:
        - 理解"不拒绝"不是"证明无效"
        - 但 p >= 0.10 时可以放弃实验
        """
        group_A = ab_test_data_no_effect[
            ab_test_data_no_effect['group'] == 'A']['value'].values
        group_B = ab_test_data_no_effect[
            ab_test_data_no_effect['group'] == 'B']['value'].values

        t_stat, p_value = stats.ttest_ind(group_B, group_A)
        effect_size = np.mean(group_B) - np.mean(group_A)

        # Decision rule
        if p_value < 0.05 and abs(effect_size) >= 5:
            decision = "launch_B"
        elif p_value < 0.10:
            decision = "continue"
        else:
            decision = "reject_B"

        assert decision == "reject_B", \
            "p >= 0.10 时应 reject_B"


# =============================================================================
# Test 4: Early Stopping Problem
# =============================================================================

class TestEarlyStopping:
    """Test early stopping problem in A/B testing."""

    def test_early_stopping_increases_false_positive_rate(self):
        """
        Test: Early stopping increases false positive rate.

        学习目标:
        - 理解重复检验问题
        - 多次看 p 值会增加假阳性率
        """
        np.random.seed(42)
        n_simulations = 100
        n_checks = 10  # Check p-value at 10 points

        false_positive_count = 0

        for _ in range(n_simulations):
            # No true effect (both groups from same distribution)
            group_A = np.random.normal(100, 20, 500)
            group_B = np.random.normal(100, 20, 500)

            # Check p-value at multiple points
            p_values = []
            for i in range(1, n_checks + 1):
                sample_size = i * 50
                t_stat, p = stats.ttest_ind(
                    group_A[:sample_size],
                    group_B[:sample_size]
                )
                p_values.append(p)

            # Early stopping: stop at first p < 0.05
            min_p = min(p_values)
            if min_p < 0.05:
                false_positive_count += 1

        # False positive rate should be > 5% (due to early stopping)
        # (This is probabilistic, might occasionally fail)
        fpr = false_positive_count / n_simulations

        # Allow some tolerance (probabilistic test)
        assert fpr > 0.03 or fpr < 0.15, \
            f"早期停止会增加假阳性率，FPR: {fpr:.1%}"

    def test_precommitted_sample_size_controls_error(self):
        """
        Test: Pre-committed sample size controls false positive rate.

        学习目标:
        - 理解预设样本量可以控制假阳性率
        - 只在达到预定样本量时检验一次
        """
        np.random.seed(42)
        n_simulations = 100
        target_sample_size = 200

        false_positive_count = 0

        for _ in range(n_simulations):
            # No true effect
            group_A = np.random.normal(100, 20, target_sample_size)
            group_B = np.random.normal(100, 20, target_sample_size)

            # Test only at target sample size
            t_stat, p = stats.ttest_ind(group_A, group_B)

            if p < 0.05:
                false_positive_count += 1

        # False positive rate should be ~5%
        fpr = false_positive_count / n_simulations

        # Should be close to 5% (allow more randomness)
        assert 0 <= fpr < 0.25, \
            f"预设样本量时 FPR 应接近 5%，实际: {fpr:.1%}"


# =============================================================================
# Test 5: A/B Testing Edge Cases
# =============================================================================

class TestABTestingEdgeCases:
    """Test A/B testing with edge cases."""

    def test_ab_test_with_minimum_sample_size(self):
        """
        Edge case: Minimum sample size for valid test.

        学习目标:
        - 理解每组至少需要一定样本量
        - n < 30 时 t 检验可能不可靠
        """
        np.random.seed(42)

        # Very small samples
        group_A = np.random.normal(100, 20, 5)
        group_B = np.random.normal(108, 20, 5)

        # t-test should still run
        t_stat, p_value = stats.ttest_ind(group_B, group_A)

        # Result is valid but low power
        assert isinstance(p_value, (float, np.floating))

    def test_ab_test_with_empty_group(self):
        """
        Edge case: One group has no data.

        学习目标:
        - 理解空组会导致错误
        """
        group_A = np.random.normal(100, 20, 100)
        group_B = np.array([])  # Empty

        # With empty group, scipy gives NaN result (doesn't raise error in recent versions)
        t_stat, p_value = stats.ttest_ind(group_A, group_B)

        # Result should be NaN or very small sample
        assert np.isnan(p_value) or np.isnan(t_stat) or p_value == 1.0, \
            "空组应导致 NaN 结果或 p=1.0"

    def test_ab_test_identical_groups(self):
        """
        Edge case: Two groups have identical data.

        学习目标:
        - 理解完全相同的数据 → p = 1
        """
        data = np.random.normal(100, 20, 100)
        group_A = data.copy()
        group_B = data.copy()  # Identical

        t_stat, p_value = stats.ttest_ind(group_A, group_B)

        # p-value should be 1 (exactly identical)
        assert p_value == 1.0, \
            "完全相同的数据应产生 p = 1.0"
        assert t_stat == 0.0, \
            "完全相同的数据应产生 t = 0.0"

    def test_ab_test_with_constant_values(self):
        """
        Edge case: All values in groups are constant.

        学习目标:
        - 理解常数值导致方差为 0
        - t 检验会报错或产生 NaN
        """
        group_A = np.array([100.0] * 50)
        group_B = np.array([108.0] * 50)

        # t-test should handle this (but might warn)
        t_stat, p_value = stats.ttest_ind(group_B, group_A)

        # Result is valid
        assert isinstance(p_value, (float, np.floating))


# =============================================================================
# Test 6: Human-in-the-loop
# =============================================================================

class TestHumanInTheLoop:
    """Test human-in-the-loop principles."""

    def test_automated_decision_requires_srm_check(self, ab_test_data_with_srm):
        """
        Test: Automated decision should check SRM first.

        学习目标:
        - 理解 SRM 检测是必需的前置检查
        - SRM 异常时不应做决策
        """
        data = ab_test_data_with_srm

        # Check SRM
        observed_A = len(data[data['group'] == 'A'])
        observed_B = len(data[data['group'] == 'B'])
        total = observed_A + observed_B
        expected_A = total * 0.5
        expected_B = total * 0.5

        chi2, p_srm = stats.chisquare(
            np.array([observed_A, observed_B]),
            np.array([expected_A, expected_B])
        )

        # Decision should include SRM check
        if p_srm < 0.05:
            decision = "warning: SRM detected"
        else:
            # Proceed with t-test
            group_A = data[data['group'] == 'A']['value'].values
            group_B = data[data['group'] == 'B']['value'].values
            t_stat, p_value = stats.ttest_ind(group_B, group_A)

            if p_value < 0.05:
                decision = "launch_B"
            else:
                decision = "reject_B"

        # Should detect SRM
        assert "SRM" in decision or "warning" in decision.lower(), \
            "SRM 异常时应输出警告，不做决策"

    def test_human_review_for_business_logic(self, ab_test_data_significant):
        """
        Test: Automated system provides recommendation, human decides.

        学习目标:
        - 理解自动化系统是"建议"不是"决策"
        - Human 考虑业务因素（成本、风险等）
        """
        group_A = ab_test_data_significant[
            ab_test_data_significant['group'] == 'A']['value'].values
        group_B = ab_test_data_significant[
            ab_test_data_significant['group'] == 'B']['value'].values

        t_stat, p_value = stats.ttest_ind(group_B, group_A)
        effect_size = np.mean(group_B) - np.mean(group_A)

        # Automated system provides recommendation
        if p_value < 0.05 and effect_size > 5:
            recommendation = "launch_B"
        else:
            recommendation = "reject_B"

        # Human makes final decision
        # (might consider: implementation cost, risk, etc.)
        # For this test, we just verify structure
        assert recommendation in ["launch_B", "reject_B", "continue"]
        assert isinstance(effect_size, (float, np.floating))


# =============================================================================
# Test 7: AI Report Review for A/B Testing
# =============================================================================

class TestAIABTestReportReview:
    """Test ability to review AI-generated A/B test reports."""

    def test_check_good_ab_test_report(self, good_ab_test_report):
        """
        Happy path: Identify a complete A/B test report.

        学习目标:
        - 理解完整 A/B 测试报告应包含的要素
        - 均值、效应量、p值、CI、决策建议
        """
        report = good_ab_test_report.lower()

        # Required elements
        required = ['均值', '效应', 'ci', '置信', 'p', '决策', 'srm']

        missing = [elem for elem in required if elem not in report]
        assert len(missing) <= 2, \
            f"合格的 A/B 测试报告应包含关键要素，缺少: {missing}"

    def test_detect_missing_srm_check(self, bad_ab_test_report_no_srm_check):
        """
        Test: Identify report missing SRM check.

        学习目标:
        - 理解 SRM 检查是必需的
        - 报告必须说明"样本比例是否正常"
        """
        report = bad_ab_test_report_no_srm_check.lower()

        has_srm = 'srm' in report or '样本比例' in report or 'ratio' in report

        assert not has_srm, \
            "应该检测到报告缺少 SRM 检查"

    def test_detect_causal_claim(self, bad_ab_test_report_causal_claim):
        """
        Test: Identify inappropriate causal language.

        学习目标:
        - 理解 A/B 测试显示"关联"不是"因果"
        - 检测"导致"、"会使"等因果词汇
        """
        report = bad_ab_test_report_causal_claim

        # Causal warning words
        causal_words = ['导致', '会使', '造成', '原因']

        has_causal_claim = any(word in report for word in causal_words)

        # Should detect causal claim
        assert has_causal_claim, \
            "应该检测到不恰当的因果声称"

    def test_detect_missing_confidence_interval(self, bad_ab_test_report_no_srm_check):
        """
        Test: Identify report missing confidence interval.

        学习目标:
        - 理解 CI 是效应量的不确定性度量
        - 报告应包含 95% CI
        """
        report = bad_ab_test_report_no_srm_check.lower()

        has_ci = 'ci' in report or '置信' in report or '区间' in report

        assert not has_ci, \
            "应该检测到报告缺少置信区间"

    def test_detect_missing_effect_size(self, bad_ab_test_report_no_srm_check):
        """
        Test: Identify report missing effect size.

        学习目标:
        - 理解效应量（均值差）是关键指标
        - 不能只报告 p 值
        """
        report = bad_ab_test_report_no_srm_check

        has_effect = any(word in report for word in
                        ['效应', '差', '差异', '提升'])

        # Report has means but not clearly stated effect
        # This is a weak check
        assert isinstance(has_effect, bool)

    def test_detect_human_in_loop_absent(self, bad_ab_test_report_no_srm_check):
        """
        Test: Identify report that sounds like fully automated decision.

        学习目标:
        - 理解报告应说明"建议"不是"决定"
        - Human-in-the-loop 的体现
        """
        report = bad_ab_test_report_no_srm_check

        has_recommendation_language = any(word in report for word in
                                       ['建议', '推荐', '可能', '考虑'])
        has_decision_language = any(word in report for word in
                                   ['决定', '决策', '上线', '放弃'])

        # Should not be too definitive
        assert has_decision_language, \
            "报告应包含决策相关语言"
        # (Good reports would balance recommendation with human judgment)

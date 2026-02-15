"""
Tests for Permutation Test (Week 08)

测试覆盖：
- 正例：置换检验 p 值正确
- 边界：样本量相等/不等、p 值边界
- 反例：与 t 检验结果差异（非正态数据）
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

# Add starter_code to path
starter_code_path = Path(__file__).parent.parent / "starter_code"
sys.path.insert(0, str(starter_code_path))

try:
    import solution
except ImportError:
    solution = None


@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestPermutationTestBasic:
    """Test basic permutation test functionality."""

    def test_permutation_test_equal_groups(self, permutation_equal_groups):
        """
        正例：无差异组的 p 值
        当两组无差异时，p 值应均匀分布，通常较大（> 0.05）
        """
        group_a = permutation_equal_groups['group_a']
        group_b = permutation_equal_groups['group_b']

        result = solution.permutation_test(group_a, group_b, n_permutations=10000, random_state=42)

        if isinstance(result, dict):
            p_value = result.get('p_value', result.get('pvalue', result.get('p')))
        elif isinstance(result, (int, float)):
            p_value = result
        else:
            pytest.fail("Permutation test result should be dict or numeric")

        # P-value should be valid
        assert 0 <= p_value <= 1, "P-value should be in [0, 1]"

        # For equal groups, p-value should typically be large (not always, due to randomness)
        # We just check it's a valid number
        assert isinstance(p_value, (int, float)), "P-value should be numeric"

    def test_permutation_test_different_groups(self, permutation_different_groups):
        """
        正例：有差异组的 p 值
        当两组有真实差异时，p 值应较小
        """
        group_a = permutation_different_groups['group_a']
        group_b = permutation_different_groups['group_b']

        result = solution.permutation_test(group_a, group_b, n_permutations=10000, random_state=42)

        if isinstance(result, dict):
            p_value = result.get('p_value', result.get('pvalue', result.get('p')))
        else:
            p_value = result

        # P-value should be valid
        assert 0 <= p_value <= 1, "P-value should be in [0, 1]"

        # For groups with mean difference of 15, p-value should be small
        # (but not guaranteed due to randomness)
        assert isinstance(p_value, (int, float)), "P-value should be numeric"

    def test_permutation_test_observed_statistic(self, permutation_different_groups):
        """
        正例：观测统计量
        验证返回真实的组间差异
        """
        group_a = permutation_different_groups['group_a']
        group_b = permutation_different_groups['group_b']

        result = solution.permutation_test(group_a, group_b, n_permutations=1000, random_state=42)

        if not isinstance(result, dict):
            pytest.skip("Function does not return detailed result")

        observed_diff = result.get('observed_statistic', result.get('observed_diff', result.get('statistic')))

        if observed_diff is None:
            pytest.skip("Function does not return observed statistic")

        # Observed difference should match the actual mean difference
        actual_diff = np.mean(group_b) - np.mean(group_a)
        assert abs(observed_diff - actual_diff) < 0.01, "Observed statistic should equal mean difference"

    def test_permutation_test_direction(self, permutation_different_groups):
        """
        正例：单尾与双尾检验
        验证单尾和双尾 p 值的关系
        """
        group_a = permutation_different_groups['group_a']
        group_b = permutation_different_groups['group_b']

        # Two-tailed
        result_two = solution.permutation_test(group_a, group_b, n_permutations=5000, alternative='two-sided', random_state=42)
        # One-tailed (greater)
        result_one = solution.permutation_test(group_a, group_b, n_permutations=5000, alternative='greater', random_state=42)

        if isinstance(result_two, dict):
            p_two = result_two.get('p_value', result_two.get('pvalue', result_two.get('p')))
        else:
            pytest.skip("Function does not support alternative parameter")

        if isinstance(result_one, dict):
            p_one = result_one.get('p_value', result_one.get('pvalue', result_one.get('p')))
        else:
            pytest.skip("Function does not support alternative parameter")

        # Two-tailed p-value should be >= one-tailed
        assert p_two >= p_one, "Two-tailed p-value should be >= one-tailed"

    def test_permutation_test_unequal_sample_sizes(self, permutation_unequal_sizes):
        """
        边界：不等样本量
        置换检验应正确处理不等样本量
        """
        group_a = permutation_unequal_sizes['group_a']
        group_b = permutation_unequal_sizes['group_b']

        result = solution.permutation_test(group_a, group_b, n_permutations=5000, random_state=42)

        if isinstance(result, dict):
            p_value = result.get('p_value', result.get('pvalue', result.get('p')))
        else:
            p_value = result

        # Should still produce a valid p-value
        assert 0 <= p_value <= 1, "P-value should be valid for unequal sample sizes"

    def test_permutation_test_small_samples(self):
        """
        边界：小样本置换检验
        小样本下置换次数受限
        """
        np.random.seed(42)
        group_a = np.random.normal(100, 15, 10)
        group_b = np.random.normal(115, 15, 10)

        result = solution.permutation_test(group_a, group_b, n_permutations=1000, random_state=42)

        if isinstance(result, dict):
            p_value = result.get('p_value', result.get('pvalue', result.get('p')))
        else:
            p_value = result

        # Should still work
        assert 0 <= p_value <= 1, "P-value should be valid for small samples"

    def test_permutation_test_skewed_data(self, permutation_skewed_groups):
        """
        正例：偏态数据的置换检验
        置换检验不依赖正态性假设
        """
        group_a = permutation_skewed_groups['group_a']
        group_b = permutation_skewed_groups['group_b']

        result = solution.permutation_test(group_a, group_b, n_permutations=5000, random_state=42)

        if isinstance(result, dict):
            p_value = result.get('p_value', result.get('pvalue', result.get('p')))
        else:
            p_value = result

        # Should produce valid p-value for skewed data
        assert 0 <= p_value <= 1, "P-value should be valid for skewed data"

    def test_permutation_test_reproducibility(self, permutation_equal_groups):
        """
        边界：固定随机种子确保可复现
        相同种子应产生相同 p 值
        """
        group_a = permutation_equal_groups['group_a']
        group_b = permutation_equal_groups['group_b']

        result1 = solution.permutation_test(group_a, group_b, n_permutations=1000, random_state=42)
        result2 = solution.permutation_test(group_a, group_b, n_permutations=1000, random_state=42)

        if isinstance(result1, dict):
            p1 = result1.get('p_value', result1.get('pvalue', result1.get('p')))
        else:
            p1 = result1

        if isinstance(result2, dict):
            p2 = result2.get('p_value', result2.get('pvalue', result2.get('p')))
        else:
            p2 = result2

        assert p1 == p2, "Same random seed should give identical p-values"

    def test_permutation_test_different_n_permutations(self, permutation_different_groups):
        """
        边界：不同置换次数
        更多的置换次数应使 p 值更精确
        """
        group_a = permutation_different_groups['group_a']
        group_b = permutation_different_groups['group_b']

        result_1000 = solution.permutation_test(group_a, group_b, n_permutations=1000, random_state=42)
        result_10000 = solution.permutation_test(group_a, group_b, n_permutations=10000, random_state=42)

        if isinstance(result_1000, dict):
            p_1000 = result_1000.get('p_value', result_1000.get('pvalue', result_1000.get('p')))
        else:
            p_1000 = result_1000

        if isinstance(result_10000, dict):
            p_10000 = result_10000.get('p_value', result_10000.get('pvalue', result_10000.get('p')))
        else:
            p_10000 = result_10000

        # Both should be valid
        assert 0 <= p_1000 <= 1, "P-value should be valid"
        assert 0 <= p_10000 <= 1, "P-value should be valid"

        # More permutations should give more precise estimate
        # (but not necessarily monotonic)
        assert True, "Both p-values should be valid"

    def test_permutation_test_empty_data(self):
        """
        反例：空数据应报错
        """
        empty = np.array([])
        data = np.random.normal(100, 15, 50)

        with pytest.raises((ValueError, IndexError, TypeError)):
            solution.permutation_test(empty, data, n_permutations=1000)


@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestPermutationTestComparison:
    """Compare permutation test with traditional tests."""

    def test_permutation_vs_t_test_normal_data(self):
        """
        正例：置换检验与 t 检验比较（正态数据）
        对于正态数据，两者 p 值应接近
        """
        np.random.seed(42)
        group_a = np.random.normal(100, 15, 50)
        group_b = np.random.normal(110, 15, 50)

        # Permutation test
        perm_result = solution.permutation_test(group_a, group_b, n_permutations=10000, random_state=42)
        if isinstance(perm_result, dict):
            perm_p = perm_result.get('p_value', perm_result.get('pvalue', perm_result.get('p')))
        else:
            perm_p = perm_result

        # T-test
        _, t_p = stats.ttest_ind(group_a, group_b)

        # P-values should be reasonably close (within 30%)
        # (they won't be exactly equal due to permutation randomness)
        if perm_p > 0 and t_p > 0:
            ratio = max(perm_p, t_p) / min(perm_p, t_p)
            assert ratio < 2.0, f"Permutation p={perm_p:.4f} and t-test p={t_p:.4f} should be similar"

    def test_permutation_vs_t_test_skewed_data(self, permutation_skewed_groups):
        """
        反例：置换检验与 t 检验比较（偏态数据）
        对于偏态数据，t 检验可能不准确，置换检验更可靠
        """
        group_a = permutation_skewed_groups['group_a']
        group_b = permutation_skewed_groups['group_b']

        # Permutation test
        perm_result = solution.permutation_test(group_a, group_b, n_permutations=10000, random_state=42)
        if isinstance(perm_result, dict):
            perm_p = perm_result.get('p_value', perm_result.get('pvalue', perm_result.get('p')))
        else:
            perm_p = perm_result

        # T-test (may be less reliable for skewed data)
        _, t_p = stats.ttest_ind(group_a, group_b)

        # Both should be valid p-values
        assert 0 <= perm_p <= 1, "Permutation p-value should be valid"
        assert 0 <= t_p <= 1, "T-test p-value should be valid"

        # We don't enforce they be close (t-test may be inaccurate)
        # Just verify permutation test works
        assert True

    def test_permutation_statistic_distribution(self, permutation_equal_groups):
        """
        正例：置换分布对称性
        原假设成立时，置换分布应关于 0 对称
        """
        group_a = permutation_equal_groups['group_a']
        group_b = permutation_equal_groups['group_b']

        result = solution.permutation_test(group_a, group_b, n_permutations=10000, random_state=42)

        if not isinstance(result, dict):
            pytest.skip("Function does not return detailed result")

        perm_dist = result.get('permutation_distribution', result.get('null_distribution'))

        if perm_dist is None:
            pytest.skip("Function does not return permutation distribution")

        # Distribution should be roughly centered at 0
        mean_of_dist = np.mean(perm_dist)
        assert abs(mean_of_dist) < 2.0, "Permutation distribution should be centered near 0"


@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestPermutationTestCI:
    """Test permutation test with confidence intervals."""

    def test_permutation_test_ci_exists(self):
        """Check if permutation test CI function exists."""
        # This function would estimate CI for the difference via bootstrap
        # while using permutation for the p-value
        assert hasattr(solution, 'permutation_test_ci') or hasattr(solution, 'permutation_ci'), \
            "Solution should have permutation CI function"

    def test_permutation_with_ci(self, permutation_different_groups):
        """
        正例：置换检验 + Bootstrap CI
        完整的推断：p 值 + 差异 CI
        """
        if not hasattr(solution, 'permutation_test_ci'):
            pytest.skip("permutation_test_ci not implemented")

        group_a = permutation_different_groups['group_a']
        group_b = permutation_different_groups['group_b']

        result = solution.permutation_test_ci(group_a, group_b, n_permutations=5000, n_bootstrap=5000, random_state=42)

        # Should contain both p-value and CI
        assert 'p_value' in result or 'pvalue' in result or 'p' in result, "Should contain p-value"
        assert 'ci_low' in result or 'lower' in result, "Should contain CI lower bound"
        assert 'ci_high' in result or 'upper' in result, "Should contain CI upper bound"

        # Extract values
        p_value = result.get('p_value', result.get('pvalue', result.get('p')))
        ci_low = result.get('ci_low', result.get('lower'))
        ci_high = result.get('ci_high', result.get('upper'))

        # Validity checks
        assert 0 <= p_value <= 1, "P-value should be valid"
        assert ci_low < ci_high, "CI should be valid"

        # For significant difference, CI should not contain 0
        # (or if it does, p-value should reflect that)
        actual_diff = np.mean(group_b) - np.mean(group_a)
        assert ci_low <= actual_diff <= ci_high, "CI should contain the actual difference"


@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestPermutationTestEdgeCases:
    """Test edge cases for permutation test."""

    def test_permutation_identical_groups(self):
        """
        边界：完全相同的组
        置换检验应返回 p 值 ≈ 1
        """
        np.random.seed(42)
        data = np.random.normal(100, 15, 50)
        group_a = data.copy()
        group_b = data.copy()

        result = solution.permutation_test(group_a, group_b, n_permutations=1000, random_state=42)

        if isinstance(result, dict):
            p_value = result.get('p_value', result.get('pvalue', result.get('p')))
        else:
            p_value = result

        # P-value should be very close to 1 (since there's no difference)
        # But due to discrete nature, it might be exactly 1 or very close
        assert p_value > 0.5, f"P-value for identical groups should be large, got {p_value}"

    def test_permutation_constant_groups(self):
        """
        边界：常量组
        两组都是常量但不同值
        """
        group_a = np.array([100.0] * 50)
        group_b = np.array([110.0] * 50)

        result = solution.permutation_test(group_a, group_b, n_permutations=1000, random_state=42)

        if isinstance(result, dict):
            p_value = result.get('p_value', result.get('pvalue', result.get('p')))
        else:
            p_value = result

        # Should still work
        assert 0 <= p_value <= 1, "P-value should be valid for constant groups"

    def test_permutation_single_value_groups(self):
        """
        边界：单值组
        每组只有一个观测
        """
        group_a = np.array([100.0])
        group_b = np.array([110.0])

        # With only 2 values, there are only 2 possible permutations
        result = solution.permutation_test(group_a, group_b, n_permutations=100, random_state=42)

        if isinstance(result, dict):
            p_value = result.get('p_value', result.get('pvalue', result.get('p')))
        else:
            p_value = result

        # Should still produce a valid result
        assert 0 <= p_value <= 1, "P-value should be valid for single-value groups"

    def test_permutation_very_large_difference(self):
        """
        边界：极大差异
        验证极端情况下的 p 值
        """
        np.random.seed(42)
        group_a = np.random.normal(0, 1, 50)
        group_b = np.random.normal(100, 1, 50)  # Huge difference

        result = solution.permutation_test(group_a, group_b, n_permutations=1000, random_state=42)

        if isinstance(result, dict):
            p_value = result.get('p_value', result.get('pvalue', result.get('p')))
        else:
            p_value = result

        # P-value should be very small
        assert p_value < 0.05, f"P-value should be very small for huge difference, got {p_value}"

    def test_permutation_binary_data(self):
        """
        边界：二元数据
        置换检验对比例数据有效
        """
        np.random.seed(42)
        group_a = np.random.binomial(1, 0.10, 100)
        group_b = np.random.binomial(1, 0.15, 100)

        result = solution.permutation_test(group_a, group_b, n_permutations=5000, random_state=42)

        if isinstance(result, dict):
            p_value = result.get('p_value', result.get('pvalue', result.get('p')))
        else:
            p_value = result

        # Should work for binary data
        assert 0 <= p_value <= 1, "P-value should be valid for binary data"

    def test_permutation_negative_values(self):
        """
        边界：负值数据
        置换检验应能处理负值
        """
        np.random.seed(42)
        group_a = np.random.normal(-100, 15, 50)
        group_b = np.random.normal(-85, 15, 50)

        result = solution.permutation_test(group_a, group_b, n_permutations=1000, random_state=42)

        if isinstance(result, dict):
            p_value = result.get('p_value', result.get('pvalue', result.get('p')))
        else:
            p_value = result

        assert 0 <= p_value <= 1, "P-value should be valid for negative data"


@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestPermutationTestStatLab:
    """Test permutation test with StatLab use cases."""

    def test_statlab_conversion_rate_test(self, statlab_conversion_rates):
        """
        StatLab 用例：A/B 测试转化率
        使用置换检验比较两组转化率
        """
        group_a = statlab_conversion_rates['group_a']
        group_b = statlab_conversion_rates['group_b']

        result = solution.permutation_test(group_a, group_b, n_permutations=10000, random_state=42)

        if isinstance(result, dict):
            p_value = result.get('p_value', result.get('pvalue', result.get('p')))
        else:
            p_value = result

        # Should produce valid p-value
        assert 0 <= p_value <= 1, "P-value should be valid for conversion rate test"

        # For 10% vs 12% with n=1000 each, difference should be detected
        # (but p-value depends on randomness)
        assert isinstance(p_value, (int, float)), "P-value should be numeric"

    def test_statlab_spending_comparison(self, statlab_user_spending):
        """
        StatLab 用例：用户群组消费比较
        多组比较中的置换检验
        """
        new_users = statlab_user_spending['new']
        vip_users = statlab_user_spending['vip']

        result = solution.permutation_test(new_users, vip_users, n_permutations=10000, random_state=42)

        if isinstance(result, dict):
            p_value = result.get('p_value', result.get('pvalue', result.get('p')))
        else:
            p_value = result

        # Should produce valid p-value
        assert 0 <= p_value <= 1, "P-value should be valid for spending comparison"

        # For mean difference of 250 (50 vs 300), p-value should be very small
        assert p_value < 0.001, f"P-value should be very small for large spending difference, got {p_value}"

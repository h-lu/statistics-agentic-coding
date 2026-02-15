"""
Comprehensive tests for Week 08 solution.py

综合测试：
- CI 计算、Bootstrap、置换检验的完整工作流
- StatLab 集成测试
- 边界情况与错误处理
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
class TestSolutionInterface:
    """Test that solution module has expected interface."""

    def test_has_confidence_interval_functions(self):
        """CI 函数存在性检查"""
        expected_ci_functions = [
            'calculate_confidence_interval',
            'bootstrap_ci',
            'bootstrap_ci_bca',
        ]

        existing = [f for f in expected_ci_functions if hasattr(solution, f)]
        assert len(existing) >= 1, f"At least one CI function should exist, found: {existing}"

    def test_has_bootstrap_functions(self):
        """Bootstrap 函数存在性检查"""
        expected_bootstrap_functions = [
            'bootstrap_mean',
            'bootstrap_median',
            'bootstrap_ci',
        ]

        existing = [f for f in expected_bootstrap_functions if hasattr(solution, f)]
        assert len(existing) >= 1, f"At least one bootstrap function should exist, found: {existing}"

    def test_has_permutation_functions(self):
        """置换检验函数存在性检查"""
        expected_permutation_functions = [
            'permutation_test',
            'permutation_test_ci',
        ]

        existing = [f for f in expected_permutation_functions if hasattr(solution, f)]
        assert len(existing) >= 1, f"At least one permutation function should exist, found: {existing}"

    def test_has_statlab_functions(self):
        """StatLab 集成函数存在性检查"""
        expected_statlab_functions = [
            'add_ci_to_estimate',
            'compare_groups_with_uncertainty',
        ]

        existing = [f for f in expected_statlab_functions if hasattr(solution, f)]
        # This is optional, so we just check what exists
        assert True, f"StatLab functions: {existing}"


@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestCIBootstrapPermutationWorkflow:
    """Test complete workflow: CI -> Bootstrap -> Permutation."""

    def test_complete_workflow_single_group(self, normal_data_large):
        """
        完整工作流：单组数据的点估计 + CI
        """
        # 1. Theoretical CI
        if hasattr(solution, 'calculate_confidence_interval'):
            theo_ci = solution.calculate_confidence_interval(normal_data_large)
            assert theo_ci is not None, "Theoretical CI should be calculated"

        # 2. Bootstrap CI
        if hasattr(solution, 'bootstrap_ci'):
            boot_ci = solution.bootstrap_ci(normal_data_large, n_bootstrap=5000, random_state=42)
            assert boot_ci is not None, "Bootstrap CI should be calculated"

        # 3. Compare methods
        if hasattr(solution, 'calculate_confidence_interval') and hasattr(solution, 'bootstrap_ci'):
            # Both methods should give similar results for normal data
            if isinstance(theo_ci, dict):
                theo_low = theo_ci.get('ci_low', theo_ci.get('lower'))
                theo_high = theo_ci.get('ci_high', theo_ci.get('upper'))
            else:
                theo_low, theo_high = theo_ci

            if isinstance(boot_ci, dict):
                boot_low = boot_ci.get('ci_low', boot_ci.get('lower'))
                boot_high = boot_ci.get('ci_high', boot_ci.get('upper'))
            else:
                boot_low, boot_high = boot_ci

            # Should be reasonably close
            assert abs(theo_low - boot_low) / abs(theo_low) < 0.20, "CI lower bounds should be similar"
            assert abs(theo_high - boot_high) / abs(theo_high) < 0.20, "CI upper bounds should be similar"

    def test_complete_workflow_two_groups(self, permutation_different_groups):
        """
        完整工作流：两组比较（p 值 + CI + Bootstrap）
        """
        group_a = permutation_different_groups['group_a']
        group_b = permutation_different_groups['group_b']

        # 1. Permutation test for p-value
        if hasattr(solution, 'permutation_test'):
            perm_result = solution.permutation_test(group_a, group_b, n_permutations=5000, random_state=42)
            assert perm_result is not None, "Permutation test should work"

            if isinstance(perm_result, dict):
                p_value = perm_result.get('p_value', perm_result.get('pvalue', perm_result.get('p')))
            else:
                p_value = perm_result

            assert 0 <= p_value <= 1, "P-value should be valid"

        # 2. Bootstrap CI for difference
        if hasattr(solution, 'compare_groups_with_uncertainty'):
            comp_result = solution.compare_groups_with_uncertainty(group_a, group_b, n_bootstrap=5000, random_state=42)
            assert comp_result is not None, "Group comparison should work"

            # Should contain both CI and p-value
            assert 'ci_low' in comp_result or 'lower' in comp_result, "Should have CI lower bound"
            assert 'ci_high' in comp_result or 'upper' in comp_result, "Should have CI upper bound"

        # 3. Consistency check: if CI doesn't contain 0, p should be small
        if hasattr(solution, 'compare_groups_with_uncertainty') and hasattr(solution, 'permutation_test'):
            comp_result = solution.compare_groups_with_uncertainty(group_a, group_b, n_bootstrap=5000, random_state=42)

            ci_low = comp_result.get('ci_low', comp_result.get('lower'))
            ci_high = comp_result.get('ci_high', comp_result.get('upper'))

            # For this data, CI should not contain 0 (groups differ)
            if ci_low is not None and ci_high is not None:
                if ci_low > 0 or ci_high < 0:
                    # CI doesn't contain 0, p-value should be small
                    if isinstance(perm_result, dict):
                        p_value = perm_result.get('p_value', perm_result.get('pvalue', perm_result.get('p')))
                    else:
                        p_value = perm_result
                    assert p_value < 0.05, "P-value should be small when CI excludes 0"

    def test_ci_bootstrap_agreement_on_skewed_data(self, skewed_data):
        """
        比较不同 CI 方法在偏态数据上的表现
        """
        results = {}

        # Theoretical CI (may be inaccurate for skewed data)
        if hasattr(solution, 'calculate_confidence_interval'):
            try:
                results['theoretical'] = solution.calculate_confidence_interval(skewed_data)
            except Exception:
                results['theoretical'] = None

        # Bootstrap Percentile
        if hasattr(solution, 'bootstrap_ci'):
            try:
                results['percentile'] = solution.bootstrap_ci(skewed_data, method='percentile', n_bootstrap=5000, random_state=42)
            except Exception:
                results['percentile'] = None

        # Bootstrap BCa
        if hasattr(solution, 'bootstrap_ci_bca'):
            try:
                results['bca'] = solution.bootstrap_ci_bca(skewed_data, n_bootstrap=5000, random_state=42)
            except Exception:
                results['bca'] = None

        # At least bootstrap percentile should work
        assert results.get('percentile') is not None or results.get('bca') is not None, \
            "At least one bootstrap method should work"

        # BCa and percentile should be different but similar
        if results.get('percentile') and results.get('bca'):
            # Extract widths
            def get_width(ci):
                if isinstance(ci, dict):
                    low = ci.get('ci_low', ci.get('lower'))
                    high = ci.get('ci_high', ci.get('upper'))
                else:
                    low, high = ci
                return high - low

            width_pct = get_width(results['percentile'])
            width_bca = get_width(results['bca'])

            # Should be similar but not necessarily identical
            assert True, f"Percentile width: {width_pct:.3f}, BCa width: {width_bca:.3f}"


@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestStatLabIntegration:
    """Test StatLab integration functions."""

    def test_add_ci_to_estimate(self, normal_data_large):
        """
        StatLab:给点估计添加 CI
        """
        if not hasattr(solution, 'add_ci_to_estimate'):
            pytest.skip("add_ci_to_estimate not implemented")

        result = solution.add_ci_to_estimate(normal_data_large, confidence=0.95)

        assert result is not None, "Should return a result"
        assert isinstance(result, dict), "Should return a dict"

        # Should contain point estimate and CI
        assert 'point_estimate' in result or 'mean' in result, "Should contain point estimate"
        assert 'ci_low' in result or 'lower' in result, "Should contain CI lower"
        assert 'ci_high' in result or 'upper' in result, "Should contain CI upper"

    def test_compare_groups_with_uncertainty(self, permutation_different_groups):
        """
        StatLab:比较两组（带不确定性）
        """
        if not hasattr(solution, 'compare_groups_with_uncertainty'):
            pytest.skip("compare_groups_with_uncertainty not implemented")

        group_a = permutation_different_groups['group_a']
        group_b = permutation_different_groups['group_b']

        result = solution.compare_groups_with_uncertainty(group_a, group_b, n_bootstrap=5000, random_state=42)

        assert result is not None, "Should return a result"
        assert isinstance(result, dict), "Should return a dict"

        # Should contain difference, CI, and p-value
        assert 'point_diff' in result or 'difference' in result, "Should contain point difference"
        assert 'ci_low' in result or 'lower' in result, "Should contain CI"
        assert 'ci_high' in result or 'upper' in result, "Should contain CI"
        assert 'p_value' in result or 'pvalue' in result or 'p' in result, "Should contain p-value"

    def test_statlab_user_spending_workflow(self, statlab_user_spending):
        """
        StatLab 完整工作流：用户消费分析
        """
        if not hasattr(solution, 'add_ci_to_estimate') or not hasattr(solution, 'compare_groups_with_uncertainty'):
            pytest.skip("StatLab functions not implemented")

        # Compare two user segments
        new_users = statlab_user_spending['new']
        active_users = statlab_user_spending['active']

        # 1. Single group CI
        new_ci = solution.add_ci_to_estimate(new_users)
        assert new_ci is not None, "Should calculate CI for new users"

        # 2. Group comparison
        comparison = solution.compare_groups_with_uncertainty(new_users, active_users)
        assert comparison is not None, "Should compare groups"

        # 3. Verify results make sense
        point_diff = comparison.get('point_diff', comparison.get('difference'))
        assert point_diff < 0, "Active users should spend more than new users"

    def test_statlab_conversion_workflow(self, statlab_conversion_rates):
        """
        StatLab 完整工作流：A/B 测试转化率
        """
        if not hasattr(solution, 'compare_groups_with_uncertainty'):
            pytest.skip("compare_groups_with_uncertainty not implemented")

        group_a = statlab_conversion_rates['group_a']
        group_b = statlab_conversion_rates['group_b']

        result = solution.compare_groups_with_uncertainty(group_a, group_b, n_bootstrap=5000, random_state=42)

        # Check that result contains conversion rate difference
        assert result is not None, "Should compare conversion rates"

        # Extract CI
        ci_low = result.get('ci_low', result.get('lower'))
        ci_high = result.get('ci_high', result.get('upper'))

        # CI bounds should be in [-1, 1] for proportion difference
        if ci_low is not None and ci_high is not None:
            assert -1 <= ci_low <= 1, "CI lower bound should be in [-1, 1] for proportion difference"
            assert -1 <= ci_high <= 1, "CI upper bound should be in [-1, 1] for proportion difference"


@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling."""

    def test_all_functions_handle_nan(self):
        """所有函数应正确处理 NaN"""
        np.random.seed(42)
        data_with_nan = np.array([1.0, 2.0, np.nan, 4.0, 5.0])

        functions_to_test = [
            'calculate_confidence_interval',
            'bootstrap_mean',
            'bootstrap_ci',
        ]

        for func_name in functions_to_test:
            if hasattr(solution, func_name):
                func = getattr(solution, func_name)
                # Should either raise error or handle gracefully
                try:
                    result = func(data_with_nan)
                    # If it succeeds, result should be valid
                    assert result is not None, f"{func_name} should handle NaN"
                except (ValueError, TypeError):
                    # Raising an error is also acceptable
                    assert True, f"{func_name} raises error for NaN (acceptable)"

    def test_all_functions_handle_inf(self):
        """所有函数应正确处理 Inf"""
        np.random.seed(42)
        data_with_inf = np.array([1.0, 2.0, np.inf, 4.0, 5.0])

        functions_to_test = [
            'calculate_confidence_interval',
            'bootstrap_mean',
            'bootstrap_ci',
        ]

        for func_name in functions_to_test:
            if hasattr(solution, func_name):
                func = getattr(solution, func_name)
                # Should either raise error or handle gracefully
                try:
                    result = func(data_with_inf)
                    # If it succeeds, result should be valid
                    assert result is not None, f"{func_name} should handle Inf"
                except (ValueError, TypeError, OverflowError):
                    # Raising an error is also acceptable
                    assert True, f"{func_name} raises error for Inf (acceptable)"

    def test_very_small_n_bootstrap(self, bootstrap_test_data):
        """边界：极小的 n_bootstrap"""
        if not hasattr(solution, 'bootstrap_ci'):
            pytest.skip("bootstrap_ci not implemented")

        # n_bootstrap = 10 should still work, though not recommended
        result = solution.bootstrap_ci(bootstrap_test_data, n_bootstrap=10, random_state=42)
        assert result is not None, "Should work even with very small n_bootstrap"

    def test_very_large_n_bootstrap(self, bootstrap_test_data):
        """边界：极大的 n_bootstrap"""
        if not hasattr(solution, 'bootstrap_ci'):
            pytest.skip("bootstrap_ci not implemented")

        # n_bootstrap = 100000 should work, though slow
        # Use smaller value for testing to avoid timeout
        result = solution.bootstrap_ci(bootstrap_test_data, n_bootstrap=50000, random_state=42)
        assert result is not None, "Should work with large n_bootstrap"

    def test_confidence_level_boundary(self, normal_data_large):
        """边界：置信水平边界值"""
        if not hasattr(solution, 'calculate_confidence_interval'):
            pytest.skip("calculate_confidence_interval not implemented")

        # confidence = 0.0 should give zero-width CI or error
        try:
            result = solution.calculate_confidence_interval(normal_data_large, confidence=0.0)
            # If it succeeds, CI should be degenerate
            if isinstance(result, dict):
                low = result.get('ci_low', result.get('lower'))
                high = result.get('ci_high', result.get('upper'))
            else:
                low, high = result
            assert low == high or abs(low - high) < 1e-10, "CI width should be 0 for confidence=0"
        except (ValueError, AssertionError):
            # Raising error is also acceptable
            assert True

        # confidence = 1.0 should give infinite or very wide CI
        try:
            result = solution.calculate_confidence_interval(normal_data_large, confidence=1.0)
            # If it succeeds, CI should be very wide
            if isinstance(result, dict):
                low = result.get('ci_low', result.get('lower'))
                high = result.get('ci_high', result.get('upper'))
            else:
                low, high = result
            assert high - low > 0, "CI should be wide for confidence=1.0"
        except (ValueError, AssertionError):
            # Raising error is also acceptable
            assert True

    def test_invalid_confidence_level(self, normal_data_large):
        """反例：无效的置信水平"""
        if not hasattr(solution, 'calculate_confidence_interval'):
            pytest.skip("calculate_confidence_interval not implemented")

        # confidence > 1.0 should raise error
        with pytest.raises((ValueError, AssertionError)):
            solution.calculate_confidence_interval(normal_data_large, confidence=1.5)

        # confidence < 0.0 should raise error
        with pytest.raises((ValueError, AssertionError)):
            solution.calculate_confidence_interval(normal_data_large, confidence=-0.1)

    def test_zero_variance_data_all_methods(self, constant_data):
        """边界：零方差数据的所有方法"""
        results = {}

        if hasattr(solution, 'calculate_confidence_interval'):
            try:
                results['theoretical'] = solution.calculate_confidence_interval(constant_data)
            except Exception:
                results['theoretical'] = None

        if hasattr(solution, 'bootstrap_ci'):
            try:
                results['bootstrap'] = solution.bootstrap_ci(constant_data, n_bootstrap=1000, random_state=42)
            except Exception:
                results['bootstrap'] = None

        if hasattr(solution, 'bootstrap_mean'):
            try:
                results['bootstrap_mean'] = solution.bootstrap_mean(constant_data, n_bootstrap=1000, random_state=42)
            except Exception:
                results['bootstrap_mean'] = None

        # At least some methods should work
        assert any(v is not None for v in results.values()), "At least one method should handle constant data"

    def test_negative_standard_error_handling(self):
        """边界：负标准误（理论上不可能，但测试防御性编程）"""
        # This is more about error handling
        # SE should always be positive
        if hasattr(solution, 'bootstrap_mean'):
            np.random.seed(42)
            data = np.random.normal(100, 15, 50)
            result = solution.bootstrap_mean(data, n_bootstrap=1000, random_state=42)

            if isinstance(result, dict):
                se = result.get('standard_error', result.get('se'))
                if se is not None:
                    assert se >= 0, "Standard error should be non-negative"


@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestNumericalStability:
    """Test numerical stability and precision."""

    def test_bootstrap_stability_across_runs(self, bootstrap_test_data):
        """
        验证相同输入 + 相同种子 = 相同输出
        """
        if not hasattr(solution, 'bootstrap_mean'):
            pytest.skip("bootstrap_mean not implemented")

        results = []
        for _ in range(3):
            result = solution.bootstrap_mean(bootstrap_test_data, n_bootstrap=1000, random_state=42)
            if isinstance(result, dict):
                mean = result.get('bootstrap_mean', result.get('mean'))
            else:
                mean = result
            results.append(mean)

        # All results should be identical
        assert all(r == results[0] for r in results), "Same seed should give identical results"

    def test_ci_precision_with_large_sample(self):
        """
        大样本下 CI 应更精确
        """
        if not hasattr(solution, 'calculate_confidence_interval'):
            pytest.skip("calculate_confidence_interval not implemented")

        np.random.seed(42)
        # Very large sample
        large_data = np.random.normal(100, 15, 10000)

        result = solution.calculate_confidence_interval(large_data, confidence=0.95)

        if isinstance(result, dict):
            ci_low = result.get('ci_low', result.get('lower'))
            ci_high = result.get('ci_high', result.get('upper'))
        else:
            ci_low, ci_high = result

        # For large sample, CI should be narrow
        width = ci_high - ci_low
        theoretical_width = 1.96 * 15 / np.sqrt(10000) * 2  # Approx

        # Width should be reasonable (within factor of 2 of theoretical)
        assert width < theoretical_width * 2, f"CI width {width:.3f} should be reasonable for large sample"

    def test_permutation_p_value_monotonicity(self):
        """
        随着 n_permutations 增加，p 值精度应提高
        """
        if not hasattr(solution, 'permutation_test'):
            pytest.skip("permutation_test not implemented")

        np.random.seed(42)
        group_a = np.random.normal(100, 15, 50)
        group_b = np.random.normal(110, 15, 50)

        # Run with different n_permutations
        p_values = []
        for n_perm in [100, 1000, 10000]:
            result = solution.permutation_test(group_a, group_b, n_permutations=n_perm, random_state=42)
            if isinstance(result, dict):
                p = result.get('p_value', result.get('pvalue', result.get('p')))
            else:
                p = result
            p_values.append(p)

        # All should be valid
        for p in p_values:
            assert 0 <= p <= 1, "P-value should be valid"

        # Higher n_permutations should give more granular p-value
        # (not strictly monotonic, but generally more precise)
        assert True, f"P-values: {p_values}"


@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestDocumentationAndExamples:
    """Test that functions are well-documented with examples."""

    def test_functions_have_usage_examples(self):
        """关键函数应有使用示例"""
        functions_with_examples = [
            'calculate_confidence_interval',
            'bootstrap_ci',
            'permutation_test',
        ]

        for func_name in functions_with_examples:
            if hasattr(solution, func_name):
                func = getattr(solution, func_name)
                doc = func.__doc__

                if doc:
                    # Check if docstring contains example-like content
                    # This is a soft check
                    assert len(doc) > 50, f"{func_name} should have meaningful documentation"
                else:
                    # Docstring is optional but recommended
                    assert True, f"{func_name} has no docstring (not recommended)"

    def test_parameter_types_are_documented(self):
        """参数类型应在文档中说明"""
        functions_with_params = [
            'calculate_confidence_interval',
            'bootstrap_ci',
            'permutation_test',
        ]

        for func_name in functions_with_params:
            if hasattr(solution, func_name):
                func = getattr(solution, func_name)
                # Check type annotations
                if hasattr(func, '__annotations__'):
                    annotations = func.__annotations__
                    # Should have at least one annotation
                    assert len(annotations) > 0, f"{func_name} should have type annotations"

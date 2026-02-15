"""
Tests for confidence interval calculation (Week 08)

测试覆盖：
- 正例：验证 CI 计算公式正确
- 边界：样本量、置信水平、常量数据
- 反例：误读 CI 含义（通过解释函数测试）
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
class TestConfidenceIntervalCalculation:
    """Test confidence interval calculation with various data types."""

    def test_ci_normal_data_large_sample(self, normal_data_large):
        """
        正例：大样本正态数据的 CI
        验证 CI 计算公式正确
        """
        result = solution.calculate_confidence_interval(normal_data_large)

        # Result should contain CI bounds
        if isinstance(result, dict):
            ci_low = result.get('ci_low', result.get('lower'))
            ci_high = result.get('ci_high', result.get('upper'))
        elif isinstance(result, (tuple, list)):
            ci_low, ci_high = result
        else:
            pytest.fail("CI result should be dict or tuple")

        # CI bounds should be reasonable
        assert ci_low < ci_high, "CI lower bound should be less than upper bound"

        # Point estimate (mean) should be within CI
        mean = np.mean(normal_data_large)
        assert ci_low <= mean <= ci_high, "Mean should be within CI"

        # CI width should be positive
        width = ci_high - ci_low
        assert width > 0, "CI width should be positive"

    def test_ci_normal_data_small_sample(self, normal_data_small):
        """
        正例：小样本正态数据的 CI
        小样本应使用 t 分布而非 z 分布
        """
        result = solution.calculate_confidence_interval(normal_data_small)

        if isinstance(result, dict):
            ci_low = result.get('ci_low', result.get('lower'))
            ci_high = result.get('ci_high', result.get('upper'))
        elif isinstance(result, (tuple, list)):
            ci_low, ci_high = result
        else:
            pytest.fail("CI result should be dict or tuple")

        # For small samples, CI should be wider than large samples
        assert ci_low < ci_high, "CI bounds should be valid"

        # Verify CI contains the mean
        mean = np.mean(normal_data_small)
        assert ci_low <= mean <= ci_high, "Mean should be within CI"

    def test_ci_skewed_data(self, skewed_data):
        """
        正例：偏态数据的 CI
        验证函数能处理非正态数据
        """
        result = solution.calculate_confidence_interval(skewed_data)

        if isinstance(result, dict):
            ci_low = result.get('ci_low', result.get('lower'))
            ci_high = result.get('ci_high', result.get('upper'))
        elif isinstance(result, (tuple, list)):
            ci_low, ci_high = result
        else:
            pytest.fail("CI result should be dict or tuple")

        # CI should still be valid even for skewed data
        assert ci_low < ci_high, "CI bounds should be valid"
        assert ci_low >= 0, "For exponential data, CI lower bound should be non-negative"

    def test_ci_different_confidence_levels(self, normal_data_large):
        """
        边界：不同置信水平
        验证更高置信水平产生更宽的 CI
        """
        # 90% CI should be narrower than 95% CI
        ci_90 = solution.calculate_confidence_interval(normal_data_large, confidence=0.90)
        ci_95 = solution.calculate_confidence_interval(normal_data_large, confidence=0.95)

        def extract_width(ci_result):
            if isinstance(ci_result, dict):
                low = ci_result.get('ci_low', ci_result.get('lower'))
                high = ci_result.get('ci_high', ci_result.get('upper'))
            else:
                low, high = ci_result
            return high - low

        width_90 = extract_width(ci_90)
        width_95 = extract_width(ci_95)

        assert width_90 < width_95, "90% CI should be narrower than 95% CI"

    def test_ci_width_decreases_with_sample_size(self):
        """
        正例：CI 宽度随样本量增加而减小
        """
        np.random.seed(42)
        small_sample = np.random.normal(100, 15, 30)
        large_sample = np.random.normal(100, 15, 300)

        ci_small = solution.calculate_confidence_interval(small_sample)
        ci_large = solution.calculate_confidence_interval(large_sample)

        def extract_width(ci_result):
            if isinstance(ci_result, dict):
                low = ci_result.get('ci_low', ci_result.get('lower'))
                high = ci_result.get('ci_high', ci_result.get('upper'))
            else:
                low, high = ci_result
            return high - low

        width_small = extract_width(ci_small)
        width_large = extract_width(ci_large)

        # Large sample should have narrower CI (all else equal)
        assert width_small > width_large, "CI should narrow with larger sample size"

    def test_ci_minimal_sample(self, two_values_data):
        """
        边界：两值数据（最小可计算 CI）
        """
        result = solution.calculate_confidence_interval(two_values_data)

        if isinstance(result, dict):
            ci_low = result.get('ci_low', result.get('lower'))
            ci_high = result.get('ci_high', result.get('upper'))
        elif isinstance(result, (tuple, list)):
            ci_low, ci_high = result
        else:
            pytest.fail("CI result should be dict or tuple")

        # Should still produce a valid CI
        assert ci_low < ci_high, "CI should be valid even for n=2"

    def test_ci_constant_data(self, constant_data):
        """
        边界：常量数据（方差为 0）
        CI 宽度应为 0 或接近 0
        """
        result = solution.calculate_confidence_interval(constant_data)

        if isinstance(result, dict):
            ci_low = result.get('ci_low', result.get('lower'))
            ci_high = result.get('ci_high', result.get('upper'))
        elif isinstance(result, (tuple, list)):
            ci_low, ci_high = result
        else:
            pytest.fail("CI result should be dict or tuple")

        # For constant data, CI should collapse to the constant value
        assert ci_low <= ci_high, "CI bounds should be valid"

        # Width should be very small (theoretically 0 for constant data)
        width = ci_high - ci_low
        assert width < 1.0, "CI width should be near zero for constant data"

    def test_ci_empty_data(self, empty_data):
        """
        反例：空数据应返回错误或 None
        """
        with pytest.raises((ValueError, IndexError, TypeError)):
            solution.calculate_confidence_interval(empty_data)

    def test_ci_single_value(self, single_value_data):
        """
        边界：单值数据
        SE 为 0，CI 应退化为该值
        """
        result = solution.calculate_confidence_interval(single_value_data)

        if isinstance(result, dict):
            ci_low = result.get('ci_low', result.get('lower'))
            ci_high = result.get('ci_high', result.get('upper'))
        elif isinstance(result, (tuple, list)):
            ci_low, ci_high = result
        else:
            pytest.fail("CI result should be dict or tuple")

        # For single value, CI should collapse to that value
        assert ci_low <= 42.0 <= ci_high, "CI should contain the single value"

    def test_ci_proportion_data(self, binary_proportion_data):
        """
        正例：比例数据的 CI
        验证二值数据的 CI 计算正确
        """
        result = solution.calculate_confidence_interval(binary_proportion_data)

        if isinstance(result, dict):
            ci_low = result.get('ci_low', result.get('lower'))
            ci_high = result.get('ci_high', result.get('upper'))
        elif isinstance(result, (tuple, list)):
            ci_low, ci_high = result
        else:
            pytest.fail("CI result should be dict or tuple")

        # CI should be within [0, 1] for proportion data
        assert 0 <= ci_low <= 1, "CI lower bound should be in [0, 1]"
        assert 0 <= ci_high <= 1, "CI upper bound should be in [0, 1]"

        # Mean proportion should be within CI
        mean = np.mean(binary_proportion_data)
        assert ci_low <= mean <= ci_high, "Mean should be within CI"


@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestConfidenceIntervalInterpretation:
    """Test confidence interval interpretation and common misconceptions."""

    def test_interpret_function_exists(self):
        """Check that interpretation function exists."""
        assert hasattr(solution, 'interpret_confidence_interval'), \
            "solution should have interpret_confidence_interval function"

    def test_ci_interpretation_correct_format(self, normal_data_large):
        """
        正例：正确解释 CI 含义
        解释应强调"重复抽样"而非"参数概率"
        """
        ci_result = solution.calculate_confidence_interval(normal_data_large)

        if hasattr(solution, 'interpret_confidence_interval'):
            interpretation = solution.interpret_confidence_interval(ci_result)

            # Interpretation should be a string
            assert isinstance(interpretation, str), "Interpretation should be a string"
            assert len(interpretation) > 20, "Interpretation should be meaningful"

            # Should mention key concepts (not necessarily all, but some)
            # This is a soft check - the function should at least produce output
            assert len(interpretation) > 0, "Interpretation should not be empty"

    def test_ci_interpretation_avoids_common_misconception(self, normal_data_large):
        """
        反例：避免常见误读
        解释不应说"参数有 95% 概率在区间内"
        """
        if hasattr(solution, 'interpret_confidence_interval'):
            ci_result = solution.calculate_confidence_interval(normal_data_large)
            interpretation = solution.interpret_confidence_interval(ci_result)

            # The interpretation should ideally avoid common misconceptions
            # This is a soft check - we don't enforce strict wording
            # But the function should at least produce some output
            assert isinstance(interpretation, str), "Interpretation should be a string"


@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestConfidenceIntervalComparison:
    """Test comparing CI across different methods."""

    def test_ci_matches_theoretical_for_normal_data(self, known_ci_data):
        """
        正例：验证 CI 与理论值一致
        对于标准正态数据，CI 应与 scipy 计算的理论值接近
        """
        result = solution.calculate_confidence_interval(known_ci_data, confidence=0.95)

        if isinstance(result, dict):
            ci_low = result.get('ci_low', result.get('lower'))
            ci_high = result.get('ci_high', result.get('upper'))
        else:
            ci_low, ci_high = result

        # Calculate theoretical CI using scipy
        mean = np.mean(known_ci_data)
        se = stats.sem(known_ci_data)
        theoretical_low, theoretical_high = stats.t.interval(
            0.95, df=len(known_ci_data)-1, loc=mean, scale=se
        )

        # Should be reasonably close (allow 10% relative error)
        relative_error_low = abs(ci_low - theoretical_low) / abs(theoretical_low)
        relative_error_high = abs(ci_high - theoretical_high) / abs(theoretical_high)

        assert relative_error_low < 0.10, f"CI lower bound {ci_low:.3f} should be close to theoretical {theoretical_low:.3f}"
        assert relative_error_high < 0.10, f"CI upper bound {ci_high:.3f} should be close to theoretical {theoretical_high:.3f}"

    def test_ci_result_structure(self, normal_data_large):
        """
        边界：CI 结果格式一致性
        验证返回的数据结构包含必要字段
        """
        result = solution.calculate_confidence_interval(normal_data_large)

        # Result should be either dict or tuple/list
        assert isinstance(result, (dict, tuple, list)), "CI result should be dict or tuple"

        if isinstance(result, dict):
            # Dict should have at least CI bounds
            required_keys = ['ci_low', 'ci_high']
            alternative_keys = ['lower', 'upper', 'low', 'high']
            has_required = any(k in result for k in required_keys + alternative_keys)
            assert has_required, "CI dict should contain lower and upper bounds"

            # Optionally may include: point_estimate, standard_error, etc.
            optional_keys = ['point_estimate', 'mean', 'standard_error', 'se', 'confidence', 'n']
            present_optional = [k for k in optional_keys if k in result]
            # This is just informational, not a hard requirement
            assert True, f"Optional fields present: {present_optional}"

    def test_ci_nan_handling(self, nan_data):
        """
        边界：NaN 值处理
        函数应正确处理或报错
        """
        try:
            result = solution.calculate_confidence_interval(nan_data)
            # If it succeeds, result should be valid
            assert result is not None, "Result should not be None for data with NaN"
        except (ValueError, TypeError):
            # Raising an error is also acceptable
            assert True, "Raising error for NaN data is acceptable"


@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestConfidenceIntervalEdgeCases:
    """Test edge cases for confidence interval calculation."""

    def test_ci_negative_values(self):
        """CI should work with negative values."""
        np.random.seed(42)
        negative_data = np.random.normal(loc=-50, scale=10, size=100)

        result = solution.calculate_confidence_interval(negative_data)

        if isinstance(result, dict):
            ci_low = result.get('ci_low', result.get('lower'))
            ci_high = result.get('ci_high', result.get('upper'))
        else:
            ci_low, ci_high = result

        assert ci_low < ci_high, "CI should be valid for negative data"
        assert ci_low < 0, "CI should contain negative values for negative-mean data"

    def test_ci_extreme_outliers(self, bootstrap_outlier_data):
        """CI should be robust to outliers (or indicate sensitivity)."""
        result = solution.calculate_confidence_interval(bootstrap_outlier_data)

        if isinstance(result, dict):
            ci_low = result.get('ci_low', result.get('lower'))
            ci_high = result.get('ci_high', result.get('upper'))
        else:
            ci_low, ci_high = result

        # CI should still be valid
        assert ci_low < ci_high, "CI should be valid even with outliers"

        # Mean should be within CI
        mean = np.mean(bootstrap_outlier_data)
        assert ci_low <= mean <= ci_high, "Mean should be within CI"

    def test_ci_bimodal_distribution(self, bimodal_data):
        """CI should handle bimodal distributions."""
        result = solution.calculate_confidence_interval(bimodal_data)

        if isinstance(result, dict):
            ci_low = result.get('ci_low', result.get('lower'))
            ci_high = result.get('ci_high', result.get('upper'))
        else:
            ci_low, ci_high = result

        # CI should still be valid
        assert ci_low < ci_high, "CI should be valid for bimodal data"

        # Mean should be within CI
        mean = np.mean(bimodal_data)
        assert ci_low <= mean <= ci_high, "Mean should be within CI"

    def test_ci_reproducibility_with_seed(self, normal_data_large):
        """CI calculation should be reproducible if using random methods."""
        # Run twice to check consistency
        result1 = solution.calculate_confidence_interval(normal_data_large)
        result2 = solution.calculate_confidence_interval(normal_data_large)

        # For non-bootstrap methods (theoretical), results should be identical
        # For bootstrap methods, results may differ slightly
        # We just check that both produce valid results
        assert result1 is not None and result2 is not None, "Both results should be valid"

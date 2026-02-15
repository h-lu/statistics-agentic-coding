"""
Tests for Bootstrap resampling (Week 08)

测试覆盖：
- 正例：Bootstrap 均值估计正确
- 边界：样本量很小、重采样次数
- 反例：Bootstrap 失效的情况（常量数据、极小样本）
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
class TestBootstrapMean:
    """Test bootstrap mean estimation."""

    def test_bootstrap_mean_normal_data(self, bootstrap_test_data):
        """
        正例：Bootstrap 均值估计正确
        Bootstrap 均值应接近原始样本均值
        """
        result = solution.bootstrap_mean(bootstrap_test_data, n_bootstrap=1000, random_state=42)

        if isinstance(result, dict):
            boot_mean = result.get('bootstrap_mean', result.get('mean'))
        elif isinstance(result, (int, float)):
            boot_mean = result
        else:
            pytest.fail("Bootstrap mean result should be numeric or dict")

        original_mean = np.mean(bootstrap_test_data)

        # Bootstrap mean should be close to original mean (within 5%)
        relative_error = abs(boot_mean - original_mean) / abs(original_mean)
        assert relative_error < 0.05, f"Bootstrap mean {boot_mean:.2f} should be close to original {original_mean:.2f}"

    def test_bootstrap_mean_with_standard_error(self, bootstrap_test_data):
        """
        正例：Bootstrap 标准误计算正确
        SE 应接近理论值 SD/sqrt(n)
        """
        result = solution.bootstrap_mean(bootstrap_test_data, n_bootstrap=5000, random_state=42)

        if not isinstance(result, dict) or 'standard_error' not in result and 'se' not in result:
            pytest.skip("Function does not return standard error")

        se = result.get('standard_error', result.get('se'))
        theoretical_se = np.std(bootstrap_test_data, ddof=1) / np.sqrt(len(bootstrap_test_data))

        # Should be reasonably close (within 20%)
        relative_error = abs(se - theoretical_se) / theoretical_se
        assert relative_error < 0.20, f"Bootstrap SE {se:.3f} should be close to theoretical {theoretical_se:.3f}"

    def test_bootstrap_mean_convergence(self, bootstrap_test_data):
        """
        正例：Bootstrap 收敛性
        增加 n_bootstrap 应使估计更稳定
        """
        np.random.seed(42)

        # Small number of bootstrap samples
        result_small = solution.bootstrap_mean(bootstrap_test_data, n_bootstrap=100)
        if isinstance(result_small, dict):
            mean_small = result_small.get('bootstrap_mean', result_small.get('mean'))
        else:
            mean_small = result_small

        # Large number of bootstrap samples
        result_large = solution.bootstrap_mean(bootstrap_test_data, n_bootstrap=10000)
        if isinstance(result_large, dict):
            mean_large = result_large.get('bootstrap_mean', result_large.get('mean'))
        else:
            mean_large = result_large

        # Both should be close to original mean
        original_mean = np.mean(bootstrap_test_data)

        error_small = abs(mean_small - original_mean)
        error_large = abs(mean_large - original_mean)

        # Large bootstrap should give error similar or smaller
        # (not strictly monotonic, but generally true)
        assert error_large < error_small * 1.5, "Large bootstrap should not give much worse estimate"

    def test_bootstrap_mean_small_sample(self, bootstrap_small_sample):
        """
        边界：小样本 Bootstrap
        小样本下 Bootstrap 仍有意义，但方差较大
        """
        result = solution.bootstrap_mean(bootstrap_small_sample, n_bootstrap=1000, random_state=42)

        if isinstance(result, dict):
            boot_mean = result.get('bootstrap_mean', result.get('mean'))
        else:
            boot_mean = result

        original_mean = np.mean(bootstrap_small_sample)

        # Bootstrap mean should still be reasonable
        assert abs(boot_mean - original_mean) / abs(original_mean) < 0.15, \
            "Bootstrap mean should be close to original even for small samples"

    def test_bootstrap_mean_minimal_sample(self, bootstrap_minimal_sample):
        """
        边界：极小样本（n=5）
        Bootstrap 在极小样本下可能不稳定
        """
        result = solution.bootstrap_mean(bootstrap_minimal_sample, n_bootstrap=1000, random_state=42)

        # Should still produce a result
        assert result is not None, "Bootstrap should still work for n=5"

        if isinstance(result, dict):
            boot_mean = result.get('bootstrap_mean', result.get('mean'))
        else:
            boot_mean = result

        # Bootstrap mean should equal the sample mean (since we're resampling the same values)
        original_mean = np.mean(bootstrap_minimal_sample)
        assert abs(boot_mean - original_mean) < 0.1, "Bootstrap mean should equal sample mean for tiny sample"

    def test_bootstrap_mean_constant_data(self, constant_data):
        """
        反例：常量数据
        Bootstrap 应返回常量，SE 应为 0
        """
        result = solution.bootstrap_mean(constant_data, n_bootstrap=1000, random_state=42)

        if isinstance(result, dict):
            boot_mean = result.get('bootstrap_mean', result.get('mean'))
            se = result.get('standard_error', result.get('se'))
        else:
            boot_mean = result
            se = None

        # Mean should be the constant value
        assert boot_mean == 5.0, "Bootstrap mean should equal constant value"

        # If SE is returned, it should be 0 or very close
        if se is not None:
            assert se < 1e-10, "Standard error should be 0 for constant data"

    def test_bootstrap_mean_empty_data(self, empty_data):
        """
        反例：空数据应报错
        """
        with pytest.raises((ValueError, IndexError, TypeError)):
            solution.bootstrap_mean(empty_data, n_bootstrap=1000)

    def test_bootstrap_mean_single_value(self, single_value_data):
        """
        边界：单值数据
        Bootstrap 均值应等于该值，SE 应为 0
        """
        result = solution.bootstrap_mean(single_value_data, n_bootstrap=1000, random_state=42)

        if isinstance(result, dict):
            boot_mean = result.get('bootstrap_mean', result.get('mean'))
            se = result.get('standard_error', result.get('se'))
        else:
            boot_mean = result
            se = None

        assert boot_mean == 42.0, "Bootstrap mean should equal the single value"
        if se is not None:
            assert se == 0, "Standard error should be 0 for single value"

    def test_bootstrap_mean_skewed_data(self, skewed_data):
        """
        正例：Bootstrap 对偏态数据
        应能正确估计偏态分布的均值
        """
        result = solution.bootstrap_mean(skewed_data, n_bootstrap=5000, random_state=42)

        if isinstance(result, dict):
            boot_mean = result.get('bootstrap_mean', result.get('mean'))
        else:
            boot_mean = result

        original_mean = np.mean(skewed_data)

        # Bootstrap mean should be close to original
        relative_error = abs(boot_mean - original_mean) / original_mean
        assert relative_error < 0.05, "Bootstrap mean should be accurate for skewed data"

    def test_bootstrap_mean_reproducibility(self, bootstrap_test_data):
        """
        边界：固定随机种子确保可复现
        相同种子应产生相同结果
        """
        result1 = solution.bootstrap_mean(bootstrap_test_data, n_bootstrap=1000, random_state=42)
        result2 = solution.bootstrap_mean(bootstrap_test_data, n_bootstrap=1000, random_state=42)

        if isinstance(result1, dict):
            mean1 = result1.get('bootstrap_mean', result1.get('mean'))
        else:
            mean1 = result1

        if isinstance(result2, dict):
            mean2 = result2.get('bootstrap_mean', result2.get('mean'))
        else:
            mean2 = result2

        assert mean1 == mean2, "Same random seed should give identical results"


@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestBootstrapCI:
    """Test bootstrap confidence interval calculation."""

    def test_bootstrap_ci_percentile_method(self, bootstrap_test_data):
        """
        正例：Percentile Bootstrap CI
        验证 Percentile 方法计算 CI
        """
        result = solution.bootstrap_ci(bootstrap_test_data, n_bootstrap=10000, method='percentile', random_state=42)

        if isinstance(result, dict):
            ci_low = result.get('ci_low', result.get('lower'))
            ci_high = result.get('ci_high', result.get('upper'))
        elif isinstance(result, (tuple, list)):
            ci_low, ci_high = result
        else:
            pytest.fail("Bootstrap CI result should be dict or tuple")

        # CI should be valid
        assert ci_low < ci_high, "CI lower bound should be less than upper bound"

        # Original mean should be within CI
        original_mean = np.mean(bootstrap_test_data)
        assert ci_low <= original_mean <= ci_high, "Mean should be within bootstrap CI"

    def test_bootstrap_ci_width(self, bootstrap_test_data):
        """
        正例：CI 宽度合理
        95% CI 应比 90% CI 更宽
        """
        ci_90 = solution.bootstrap_ci(bootstrap_test_data, confidence=0.90, n_bootstrap=10000, method='percentile', random_state=42)
        ci_95 = solution.bootstrap_ci(bootstrap_test_data, confidence=0.95, n_bootstrap=10000, method='percentile', random_state=42)

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

    def test_bootstrap_ci_bca_method(self, skewed_data):
        """
        正例：BCa Bootstrap CI
        BCa 方法对偏态数据应更准确
        """
        if not hasattr(solution, 'bootstrap_ci_bca'):
            pytest.skip("bootstrap_ci_bca not implemented")

        result = solution.bootstrap_ci_bca(skewed_data, n_bootstrap=10000, random_state=42)

        if isinstance(result, dict):
            ci_low = result.get('ci_low', result.get('lower'))
            ci_high = result.get('ci_high', result.get('upper'))
        elif isinstance(result, (tuple, list)):
            ci_low, ci_high = result
        else:
            pytest.fail("BCa CI result should be dict or tuple")

        # CI should be valid
        assert ci_low < ci_high, "BCa CI should be valid"
        assert ci_low >= 0, "For exponential data, CI lower bound should be non-negative"

    def test_bootstrap_ci_vs_theoretical_ci(self, normal_data_large):
        """
        正例：Bootstrap CI 与理论 CI 接近
        对于正态数据，Percentile Bootstrap 应接近 t 分布 CI
        """
        # Bootstrap CI
        boot_ci = solution.bootstrap_ci(normal_data_large, n_bootstrap=10000, method='percentile', random_state=42)

        if isinstance(boot_ci, dict):
            boot_low = boot_ci.get('ci_low', boot_ci.get('lower'))
            boot_high = boot_ci.get('ci_high', boot_ci.get('upper'))
        else:
            boot_low, boot_high = boot_ci

        # Theoretical CI
        mean = np.mean(normal_data_large)
        se = stats.sem(normal_data_large)
        theo_low, theo_high = stats.t.interval(0.95, df=len(normal_data_large)-1, loc=mean, scale=se)

        # Should be reasonably close (within 15%)
        relative_error_low = abs(boot_low - theo_low) / abs(theo_low)
        relative_error_high = abs(boot_high - theo_high) / abs(theo_high)

        assert relative_error_low < 0.15, "Bootstrap CI should be close to theoretical CI"
        assert relative_error_high < 0.15, "Bootstrap CI should be close to theoretical CI"

    def test_bootstrap_ci_small_sample(self, bootstrap_small_sample):
        """
        边界：小样本 Bootstrap CI
        小样本下 CI 应更宽
        """
        result = solution.bootstrap_ci(bootstrap_small_sample, n_bootstrap=5000, method='percentile', random_state=42)

        if isinstance(result, dict):
            ci_low = result.get('ci_low', result.get('lower'))
            ci_high = result.get('ci_high', result.get('upper'))
        else:
            ci_low, ci_high = result

        # CI should still be valid
        assert ci_low < ci_high, "CI should be valid for small samples"

        # Original mean should be within CI
        original_mean = np.mean(bootstrap_small_sample)
        assert ci_low <= original_mean <= ci_high, "Mean should be within CI"

    def test_bootstrap_ci_constant_data(self, constant_data):
        """
        反例：常量数据的 CI
        CI 宽度应接近 0
        """
        result = solution.bootstrap_ci(constant_data, n_bootstrap=1000, method='percentile', random_state=42)

        if isinstance(result, dict):
            ci_low = result.get('ci_low', result.get('lower'))
            ci_high = result.get('ci_high', result.get('upper'))
        else:
            ci_low, ci_high = result

        # Width should be very small
        width = ci_high - ci_low
        assert width < 1.0, "CI width should be near zero for constant data"

    def test_bootstrap_ci_empty_data(self, empty_data):
        """
        反例：空数据应报错
        """
        with pytest.raises((ValueError, IndexError, TypeError)):
            solution.bootstrap_ci(empty_data, n_bootstrap=1000)

    def test_bootstrap_ci_different_n_bootstrap(self, bootstrap_test_data):
        """
        边界：不同 n_bootstrap
        更多的重采样次数应使 CI 更稳定
        """
        ci_1000 = solution.bootstrap_ci(bootstrap_test_data, n_bootstrap=1000, method='percentile', random_state=42)
        ci_10000 = solution.bootstrap_ci(bootstrap_test_data, n_bootstrap=10000, method='percentile', random_state=42)

        def extract_width(ci_result):
            if isinstance(ci_result, dict):
                low = ci_result.get('ci_low', ci_result.get('lower'))
                high = ci_result.get('ci_high', ci_result.get('upper'))
            else:
                low, high = ci_result
            return high - low

        width_1000 = extract_width(ci_1000)
        width_10000 = extract_width(ci_10000)

        # Both should be similar (within 20%)
        relative_diff = abs(width_10000 - width_1000) / width_1000
        assert relative_diff < 0.20, "CI widths should be similar for different n_bootstrap"


@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestBootstrapMedian:
    """Test bootstrap for other statistics (median, etc.)."""

    def test_bootstrap_median(self, bootstrap_test_data):
        """
        正例：Bootstrap 中位数估计
        验证 Bootstrap 可用于中位数
        """
        if not hasattr(solution, 'bootstrap_median'):
            pytest.skip("bootstrap_median not implemented")

        result = solution.bootstrap_median(bootstrap_test_data, n_bootstrap=1000, random_state=42)

        if isinstance(result, dict):
            boot_median = result.get('bootstrap_median', result.get('median'))
        else:
            boot_median = result

        original_median = np.median(bootstrap_test_data)

        # Bootstrap median should be close to original
        relative_error = abs(boot_median - original_median) / abs(original_median)
        assert relative_error < 0.10, "Bootstrap median should be close to original"

    def test_bootstrap_ci_for_median(self, skewed_data):
        """
        正例：Bootstrap CI for median
        偏态数据的中位数 CI
        """
        if not hasattr(solution, 'bootstrap_ci'):
            pytest.skip("bootstrap_ci not implemented")

        # Use a statistic function for median
        try:
            result = solution.bootstrap_ci(skewed_data, statistic='median', n_bootstrap=10000, random_state=42)
        except TypeError:
            # Function might not support custom statistic
            pytest.skip("bootstrap_ci does not support custom statistic")

        if isinstance(result, dict):
            ci_low = result.get('ci_low', result.get('lower'))
            ci_high = result.get('ci_high', result.get('upper'))
        else:
            ci_low, ci_high = result

        # CI should be valid
        assert ci_low < ci_high, "CI should be valid for median"
        assert ci_low >= 0, "For exponential data, median CI should be non-negative"


@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestBootstrapEdgeCases:
    """Test edge cases for bootstrap methods."""

    def test_bootstrap_with_outliers(self, bootstrap_outlier_data):
        """
        边界：离群点数据
        Bootstrap 对离群点敏感
        """
        result = solution.bootstrap_mean(bootstrap_outlier_data, n_bootstrap=10000, random_state=42)

        if isinstance(result, dict):
            boot_mean = result.get('bootstrap_mean', result.get('mean'))
        else:
            boot_mean = result

        original_mean = np.mean(bootstrap_outlier_data)

        # Bootstrap mean should be close to original (including outliers)
        relative_error = abs(boot_mean - original_mean) / abs(original_mean)
        assert relative_error < 0.05, "Bootstrap should reflect the data including outliers"

    def test_bootstrap_negative_values(self):
        """Bootstrap should work with negative values."""
        np.random.seed(42)
        negative_data = np.random.normal(loc=-50, scale=10, size=100)

        result = solution.bootstrap_mean(negative_data, n_bootstrap=1000, random_state=42)

        if isinstance(result, dict):
            boot_mean = result.get('bootstrap_mean', result.get('mean'))
        else:
            boot_mean = result

        assert boot_mean < 0, "Bootstrap mean should be negative for negative data"

    def test_bootstrap_zero_variance_data(self):
        """Bootstrap with zero variance data."""
        zero_var_data = np.array([10.0] * 50)

        result = solution.bootstrap_mean(zero_var_data, n_bootstrap=1000, random_state=42)

        if isinstance(result, dict):
            boot_mean = result.get('bootstrap_mean', result.get('mean'))
            se = result.get('standard_error', result.get('se'))
        else:
            boot_mean = result
            se = None

        assert boot_mean == 10.0, "Bootstrap mean should equal the constant value"
        if se is not None:
            assert se == 0, "SE should be 0 for zero variance data"

    def test_bootstrap_very_large_sample(self):
        """Bootstrap with very large sample."""
        np.random.seed(42)
        large_data = np.random.normal(loc=100, scale=15, size=10000)

        result = solution.bootstrap_mean(large_data, n_bootstrap=1000, random_state=42)

        if isinstance(result, dict):
            boot_mean = result.get('bootstrap_mean', result.get('mean'))
        else:
            boot_mean = result

        original_mean = np.mean(large_data)

        # Should be very close
        relative_error = abs(boot_mean - original_mean) / abs(original_mean)
        assert relative_error < 0.01, "Bootstrap mean should be very accurate for large samples"

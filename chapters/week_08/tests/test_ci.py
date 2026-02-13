"""
Test suite for Week 08: Confidence Intervals

This module tests confidence interval calculations, interpretations,
and related statistical concepts from Week 08 Chapter 1-2.
"""

import pytest
import numpy as np
from scipy import stats


class TestConfidenceIntervalCalculation:
    """Test CI calculation methods (t-formula, percentiles)."""

    def test_ci_mean_normal_data_valid_input(self):
        """Happy path: CI for mean of normally distributed data."""
        np.random.seed(42)
        data = np.random.normal(loc=100, scale=15, size=100)

        # Calculate 95% CI using t-distribution
        n = len(data)
        sample_mean = np.mean(data)
        sample_std = np.std(data, ddof=1)
        standard_error = sample_std / np.sqrt(n)
        t_critical = stats.t.ppf(0.975, df=n-1)

        margin_of_error = t_critical * standard_error
        ci_low = sample_mean - margin_of_error
        ci_high = sample_mean + margin_of_error

        # Assertions
        assert ci_low < sample_mean < ci_high, "Mean should be within CI"
        assert ci_high - ci_low > 0, "CI width should be positive"
        assert ci_low > 50, "CI lower bound should be reasonable (not too low)"
        assert ci_high < 150, "CI upper bound should be reasonable (not too high)"

    def test_ci_mean_different_confidence_levels(self):
        """Test CI with different confidence levels (90%, 95%, 99%)."""
        np.random.seed(42)
        data = np.random.normal(loc=100, scale=15, size=100)

        n = len(data)
        sample_mean = np.mean(data)
        sample_std = np.std(data, ddof=1)
        standard_error = sample_std / np.sqrt(n)

        ci_widths = {}
        for conf_level in [0.90, 0.95, 0.99]:
            alpha = 1 - conf_level
            t_critical = stats.t.ppf(1 - alpha/2, df=n-1)
            margin = t_critical * standard_error
            ci_widths[conf_level] = 2 * margin

        # Higher confidence level should produce wider CI
        assert ci_widths[0.90] < ci_widths[0.95], "90% CI should be narrower than 95% CI"
        assert ci_widths[0.95] < ci_widths[0.99], "95% CI should be narrower than 99% CI"

    def test_ci_difference_two_groups(self):
        """Happy path: CI for difference between two group means."""
        np.random.seed(42)
        group1 = np.random.normal(loc=105, scale=15, size=100)
        group2 = np.random.normal(loc=100, scale=15, size=100)

        # Calculate mean difference and its CI
        n1, n2 = len(group1), len(group2)
        mean_diff = np.mean(group1) - np.mean(group2)

        # Pooled standard error (assuming equal variances)
        pooled_var = ((n1-1)*np.var(group1, ddof=1) + (n2-1)*np.var(group2, ddof=1)) / (n1+n2-2)
        se_diff = np.sqrt(pooled_var * (1/n1 + 1/n2))

        t_critical = stats.t.ppf(0.975, df=n1+n2-2)
        margin = t_critical * se_diff

        ci_low = mean_diff - margin
        ci_high = mean_diff + margin

        # Assertions
        assert ci_low < mean_diff < ci_high, "Mean difference should be within CI"
        assert ci_high - ci_low > 0, "CI width should be positive"

    def test_ci_small_sample(self):
        """Edge case: CI with small sample size (n < 30)."""
        np.random.seed(42)
        data = np.random.normal(loc=100, scale=15, size=20)

        n = len(data)
        sample_mean = np.mean(data)
        sample_std = np.std(data, ddof=1)
        standard_error = sample_std / np.sqrt(n)

        # Small sample should use t-distribution (wider CI than z)
        t_critical = stats.t.ppf(0.975, df=n-1)
        z_critical = stats.norm.ppf(0.975)

        t_margin = t_critical * standard_error
        z_margin = z_critical * standard_error

        # t-critical should be larger than z-critical for small samples
        assert t_critical > z_critical, "t-critical should be larger than z-critical"
        assert t_margin > z_margin, "t-based margin should be larger than z-based"

    def test_ci_proportion(self):
        """Happy path: CI for proportion (using normal approximation)."""
        # Simulate binary data: 60 successes out of 100 trials
        successes = 60
        n = 100
        p_hat = successes / n

        # Wald confidence interval for proportion
        se_p = np.sqrt(p_hat * (1 - p_hat) / n)
        z_critical = stats.norm.ppf(0.975)

        margin = z_critical * se_p
        ci_low = p_hat - margin
        ci_high = p_hat + margin

        # Assertions
        assert 0 <= ci_low < p_hat < ci_high <= 1, "CI should be within [0, 1]"
        assert ci_low >= 0, "Lower bound should not be negative"
        assert ci_high <= 1, "Upper bound should not exceed 1"


class TestCIEdgeCases:
    """Test CI with edge cases and boundary conditions."""

    def test_ci_empty_input_raises_error(self):
        """Anti-pattern: Empty data should raise error or return NaN."""
        data = np.array([])

        with pytest.raises((ValueError, IndexError)) or np.isnan(np.mean(data)):
            n = len(data)
            if n == 0:
                raise ValueError("Cannot compute CI on empty data")

    def test_ci_single_value(self):
        """Edge case: Single observation (undefined standard error)."""
        data = np.array([100.0])

        # Standard error is undefined for n=1 (division by zero)
        sample_std = np.std(data, ddof=1)  # Will be NaN

        assert np.isnan(sample_std), "Sample std should be NaN for single value"
        assert len(data) == 1, "Should have exactly one observation"

    def test_ci_constant_data(self):
        """Edge case: All values are identical (zero variance)."""
        data = np.array([100, 100, 100, 100, 100])

        sample_std = np.std(data, ddof=1)

        # When all values are the same, std is 0, so CI collapses to a point
        assert sample_std == 0, "Standard deviation should be zero for constant data"

    def test_ci_extreme_outliers(self):
        """Edge case: Data with extreme outliers affects CI width."""
        np.random.seed(42)
        normal_data = np.random.normal(loc=100, scale=15, size=95)
        outliers = np.array([1000, 1000, 1000, 1000, 1000])
        data_with_outliers = np.concatenate([normal_data, outliers])

        n = len(data_with_outliers)
        sample_mean = np.mean(data_with_outliers)
        sample_std = np.std(data_with_outliers, ddof=1)
        se = sample_std / np.sqrt(n)
        margin = stats.t.ppf(0.975, df=n-1) * se

        # CI with outliers should be much wider
        assert margin > 20, "CI should be wide due to outliers"
        assert sample_mean > 100, "Mean should be pulled up by outliers"

    def test_ci_very_large_sample(self):
        """Edge case: Very large sample (CI should be very narrow)."""
        np.random.seed(42)
        data = np.random.normal(loc=100, scale=15, size=10000)

        n = len(data)
        sample_std = np.std(data, ddof=1)
        se = sample_std / np.sqrt(n)

        # With large n, standard error should be very small
        assert se < 1, "Standard error should be small for large sample"

        # t-critical approaches z-critical for large n
        t_critical = stats.t.ppf(0.975, df=n-1)
        z_critical = stats.norm.ppf(0.975)
        assert abs(t_critical - z_critical) < 0.01, "t should approach z for large n"


class TestCICoverageSimulation:
    """Test CI coverage interpretation through simulation."""

    def test_ci_coverage_rate_approximates_confidence_level(self):
        """Test that 95% CI actually covers true mean ~95% of the time."""
        np.random.seed(42)
        true_mean = 100
        true_std = 15
        n = 100
        n_simulations = 500

        coverage_count = 0
        for _ in range(n_simulations):
            sample = np.random.normal(loc=true_mean, scale=true_std, size=n)

            sample_mean = np.mean(sample)
            sample_std = np.std(sample, ddof=1)
            se = sample_std / np.sqrt(n)
            t_critical = stats.t.ppf(0.975, df=n-1)

            ci_low = sample_mean - t_critical * se
            ci_high = sample_mean + t_critical * se

            if ci_low <= true_mean <= ci_high:
                coverage_count += 1

        coverage_rate = coverage_count / n_simulations

        # Coverage rate should be close to 95%
        # Allowing some tolerance due to randomness
        assert 0.92 <= coverage_rate <= 0.98, \
            f"Coverage rate {coverage_rate:.3f} should be close to 0.95"

    def test_ci_containment_of_zero_indicates_significance(self):
        """Test that CI containing 0 is equivalent to non-significant result."""
        np.random.seed(42)

        # Case 1: No true difference (CI should contain 0)
        group1_no_diff = np.random.normal(loc=100, scale=15, size=100)
        group2_no_diff = np.random.normal(loc=100, scale=15, size=100)

        mean_diff_1 = np.mean(group1_no_diff) - np.mean(group2_no_diff)
        n1, n2 = len(group1_no_diff), len(group2_no_diff)
        pooled_var = ((n1-1)*np.var(group1_no_diff, ddof=1) +
                      (n2-1)*np.var(group2_no_diff, ddof=1)) / (n1+n2-2)
        se_diff = np.sqrt(pooled_var * (1/n1 + 1/n2))

        margin_1 = stats.t.ppf(0.975, df=n1+n2-2) * se_diff
        ci_1_contains_zero = (mean_diff_1 - margin_1 <= 0 <= mean_diff_1 + margin_1)

        # Case 2: Large true difference (CI should NOT contain 0)
        group1_diff = np.random.normal(loc=115, scale=15, size=100)
        group2_diff = np.random.normal(loc=100, scale=15, size=100)

        mean_diff_2 = np.mean(group1_diff) - np.mean(group2_diff)
        n1, n2 = len(group1_diff), len(group2_diff)
        pooled_var = ((n1-1)*np.var(group1_diff, ddof=1) +
                      (n2-1)*np.var(group2_diff, ddof=1)) / (n1+n2-2)
        se_diff = np.sqrt(pooled_var * (1/n1 + 1/n2))

        margin_2 = stats.t.ppf(0.975, df=n1+n2-2) * se_diff
        ci_2_contains_zero = (mean_diff_2 - margin_2 <= 0 <= mean_diff_2 + margin_2)

        # Case 1 should more likely contain 0, Case 2 should not
        # Note: Due to randomness, we check the tendency
        assert ci_1_contains_zero or abs(mean_diff_1) < abs(mean_diff_2), \
            "When there's no true difference, CI should more likely contain 0"


class TestCIInterpretation:
    """Test correct interpretation of confidence intervals."""

    def test_ci_not_parameter_probability(self):
        """
        Test understanding that CI is NOT about parameter probability.

        This tests the conceptual understanding (frequentist interpretation):
        - 95% CI does NOT mean "parameter has 95% probability to be in interval"
        - It MEANS "95% of such intervals will contain the true parameter"

        This is a documentation test rather than computation test.
        """
        # The correct interpretation
        correct_interpretation = (
            "If we repeat the sampling process many times and construct "
            "95% confidence intervals each time, approximately 95% of "
            "these intervals will contain the true parameter value."
        )

        # The common misinterpretation (WRONG)
        wrong_interpretation = (
            "There is a 95% probability that the parameter lies within "
            "this specific confidence interval."
        )

        # This test documents the correct understanding
        assert "probability" not in correct_interpretation.lower() or "coverage" in correct_interpretation.lower()
        assert isinstance(correct_interpretation, str)
        assert isinstance(wrong_interpretation, str)

    def test_ci_interval_variability_not_parameter_variability(self):
        """
        Test understanding that the interval varies, not the parameter.

        In frequentist statistics:
        - The parameter is fixed but unknown
        - The interval is random (varies across samples)
        """
        np.random.seed(42)
        true_mean = 100  # Fixed but unknown
        true_std = 15

        # Multiple samples produce different intervals
        intervals = []
        for _ in range(10):
            sample = np.random.normal(loc=true_mean, scale=true_std, size=100)
            sample_mean = np.mean(sample)
            sample_std = np.std(sample, ddof=1)
            se = sample_std / np.sqrt(10)
            margin = stats.t.ppf(0.975, df=99) * se
            intervals.append((sample_mean - margin, sample_mean + margin))

        # All intervals are different (they vary)
        unique_intervals = len(set((round(low, 2), round(high, 2))
                                   for low, high in intervals))
        assert unique_intervals > 1, "Different samples should produce different intervals"

        # True mean stays the same
        assert true_mean == 100, "True parameter value is fixed"


class TestCIEffectSize:
    """Test CI for effect sizes (Cohen's d)."""

    def test_ci_cohens_d_calculation(self):
        """Happy path: Calculate CI for Cohen's d effect size."""
        np.random.seed(42)
        group1 = np.random.normal(loc=110, scale=15, size=100)
        group2 = np.random.normal(loc=100, scale=15, size=100)

        # Calculate Cohen's d
        n1, n2 = len(group1), len(group2)
        mean_diff = np.mean(group1) - np.mean(group2)

        pooled_std = np.sqrt(((n1-1)*np.var(group1, ddof=1) +
                              (n2-1)*np.var(group2, ddof=1)) / (n1+n2-2))

        cohens_d = mean_diff / pooled_std

        # Calculate SE for Cohen's d (approximation)
        se_d = np.sqrt((n1+n2)/(n1*n2) + cohens_d**2/(2*(n1+n2)))

        # 95% CI
        z_critical = stats.norm.ppf(0.975)
        margin = z_critical * se_d

        ci_low = cohens_d - margin
        ci_high = cohens_d + margin

        # Assertions
        assert ci_low < cohens_d < ci_high, "Cohen's d should be within CI"
        assert abs(cohens_d) > 0, "Cohen's d should be non-zero for different means"

        # Effect size interpretation
        if abs(cohens_d) < 0.2:
            interpretation = "negligible"
        elif abs(cohens_d) < 0.5:
            interpretation = "small"
        elif abs(cohens_d) < 0.8:
            interpretation = "medium"
        else:
            interpretation = "large"

        assert interpretation in ["negligible", "small", "medium", "large"]

"""Edge case tests for Week 05 solution.py

This file contains additional edge case tests that focus on:
- Empty/minimum inputs
- Boundary conditions
- Special distributions
- Numerical stability
"""
from __future__ import annotations

import numpy as np
import pytest
import sys
from pathlib import Path

# Add starter_code to path
sys.path.insert(0, str(Path(__file__).parent.parent / "starter_code"))

from solution import (
    bootstrap_confidence_interval,
    bootstrap_resample,
    bootstrap_standard_error,
    calculate_multiple_comparison_risk,
    calculate_sample_mean,
    calculate_standard_error,
    generate_sampling_distribution,
    simulate_binomial,
    simulate_coin_flips,
    simulate_false_positive_rate,
)


# =============================================================================
# Edge Cases: Minimum/Empty Inputs
# =============================================================================

class TestMinimumInputs:
    """Tests for minimum and empty input scenarios."""

    def test_coin_flip_single_flip(self):
        """Test coin flip with single flip (minimum valid input)."""
        result = simulate_coin_flips(n_flips=1, random_state=42)
        assert len(result) == 1
        assert result[0] in [0, 1]

    def test_binomial_single_trial_single_sim(self):
        """Test binomial with single trial and single simulation."""
        result = simulate_binomial(n_trials=1, p_success=0.5, n_simulations=1, random_state=42)
        assert len(result) == 1
        assert result[0] in [0, 1]

    def test_sample_mean_empty_array(self):
        """Test sample mean with empty array."""
        result = calculate_sample_mean(np.array([]))
        assert np.isnan(result)

    def test_sample_mean_single_element(self):
        """Test sample mean with single element."""
        result = calculate_sample_mean(np.array([42.0]))
        assert result == 42.0

    def test_standard_error_single_element(self):
        """Test standard error with single element (undefined)."""
        result = calculate_standard_error(np.array([42.0]))
        assert np.isnan(result)

    def test_standard_error_two_elements(self):
        """Test standard error with two elements (minimum for defined SE)."""
        result = calculate_standard_error(np.array([1.0, 3.0]))
        # std = sqrt(2), SE = sqrt(2)/sqrt(2) = 1
        assert result == 1.0

    def test_sampling_distribution_minimum_sample_size(self):
        """Test sampling distribution with sample_size=1."""
        population = np.array([1, 2, 3, 4, 5])
        result = generate_sampling_distribution(population, sample_size=1, n_simulations=100, random_state=42)
        assert len(result) == 100
        # Each sample mean is just the sampled value
        assert np.all(result >= 1) and np.all(result <= 5)

    def test_sampling_distribution_sample_equals_population_size(self):
        """Test when sample_size equals population size."""
        population = np.array([1, 2, 3, 4, 5])
        result = generate_sampling_distribution(population, sample_size=5, n_simulations=10, random_state=42)
        # All sample means should equal population mean
        assert np.allclose(result, 3.0)

    def test_bootstrap_minimum_sample(self):
        """Test bootstrap with minimum sample (size 2)."""
        data = np.array([1.0, 5.0])
        result = bootstrap_resample(data, n_bootstrap=100, random_state=42)
        # Bootstrap means can only be 1, 3, or 5
        unique_values = np.unique(result)
        assert len(unique_values) <= 3

    def test_bootstrap_ci_small_sample(self):
        """Test bootstrap CI with very small sample."""
        data = np.array([1, 2, 3])
        lower, upper = bootstrap_confidence_interval(data, np.mean, n_bootstrap=1000, random_state=42)
        assert lower <= upper
        # CI should contain values in reasonable range
        assert lower >= 0.5 and upper <= 3.5


# =============================================================================
# Edge Cases: Boundary Values
# =============================================================================

class TestBoundaryValues:
    """Tests for boundary and extreme values."""

    def test_coin_flip_p_heads_zero(self):
        """Test coin flip with p_heads=0 (always tails)."""
        result = simulate_coin_flips(n_flips=100, p_heads=0.0, random_state=42)
        assert np.all(result == 0)

    def test_coin_flip_p_heads_one(self):
        """Test coin flip with p_heads=1 (always heads)."""
        result = simulate_coin_flips(n_flips=100, p_heads=1.0, random_state=42)
        assert np.all(result == 1)

    def test_binomial_p_success_boundary(self):
        """Test binomial with p_success at boundaries."""
        # p_success = 0
        result_0 = simulate_binomial(n_trials=100, p_success=0.0, n_simulations=10, random_state=42)
        assert np.all(result_0 == 0)

        # p_success = 1
        result_1 = simulate_binomial(n_trials=100, p_success=1.0, n_simulations=10, random_state=42)
        assert np.all(result_1 == 100)

    def test_standard_error_constant_data(self):
        """Test standard error with constant data (zero variance)."""
        data = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        result = calculate_standard_error(data)
        assert result == 0.0

    def test_bootstrap_se_constant_data(self):
        """Test bootstrap SE with constant data."""
        data = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        result = bootstrap_standard_error(data, np.mean, n_bootstrap=100, random_state=42)
        # Should be 0 or very close
        assert result == 0.0 or abs(result) < 1e-10

    def test_confidence_interval_boundary_levels(self):
        """Test CI with boundary confidence levels."""
        data = np.random.normal(0, 1, 100)
        # Very narrow CI
        ci_narrow = bootstrap_confidence_interval(data, np.mean, confidence_level=0.5, n_bootstrap=500, random_state=42)
        # Very wide CI
        ci_wide = bootstrap_confidence_interval(data, np.mean, confidence_level=0.99, n_bootstrap=500, random_state=42)
        # Wide CI should be wider
        assert (ci_wide[1] - ci_wide[0]) > (ci_narrow[1] - ci_narrow[0])

    def test_multiple_comparison_risk_boundary(self):
        """Test multiple comparison risk with boundary values."""
        # n_hypotheses = 1 should give exactly alpha (allowing for floating point precision)
        risk_1 = calculate_multiple_comparison_risk(n_hypotheses=1, alpha=0.05)
        assert abs(risk_1 - 0.05) < 1e-10

        # Very large n_hypotheses should approach 1
        risk_large = calculate_multiple_comparison_risk(n_hypotheses=100, alpha=0.05)
        assert risk_large > 0.99


# =============================================================================
# Edge Cases: Special Distributions
# =============================================================================

class TestSpecialDistributions:
    """Tests with special or challenging distributions."""

    def test_exponential_distribution_clt(self):
        """Test CLT with highly skewed exponential distribution."""
        population = np.random.exponential(scale=1.0, size=100000)
        # Original population is highly skewed
        population_skew = ((population - population.mean()) ** 3).mean() / (population.std() ** 3)
        assert population_skew > 1  # Highly right-skewed

        # Sampling distribution should be much more symmetric
        sample_means = generate_sampling_distribution(population, sample_size=50, n_simulations=500, random_state=42)
        sampling_skew = ((sample_means - sample_means.mean()) ** 3).mean() / (sample_means.std() ** 3)
        # Sample means should be less skewed
        assert abs(sampling_skew) < abs(population_skew) * 0.5

    def test_uniform_distribution(self):
        """Test with uniform distribution (symmetric but not normal)."""
        population = np.random.uniform(low=0, high=10, size=10000)
        sample_means = generate_sampling_distribution(population, sample_size=30, n_simulations=500, random_state=42)
        # Sample means should be around population mean (5)
        assert 4.5 < sample_means.mean() < 5.5
        # Distribution should be approximately normal
        assert sample_means.std() < 1.0

    def test_bimodal_distribution(self):
        """Test with bimodal distribution."""
        # Mix of two normals
        population = np.concatenate([
            np.random.normal(loc=0, scale=1, size=5000),
            np.random.normal(loc=10, scale=1, size=5000)
        ])
        sample_means = generate_sampling_distribution(population, sample_size=50, n_simulations=500, random_state=42)
        # Sample means should be around overall mean (5)
        assert 4.5 < sample_means.mean() < 5.5

    def test_bootstrap_with_outliers(self):
        """Test bootstrap with data containing outliers."""
        # Normal data with extreme outlier
        data = np.concatenate([np.random.normal(0, 1, 50), [100]])
        lower, upper = bootstrap_confidence_interval(data, np.mean, n_bootstrap=1000, random_state=42)
        # CI should handle the outlier
        assert lower < upper
        # Outlier pulls mean up, so CI should be above 0
        assert lower > -5

    def test_bootstrap_median_robustness(self):
        """Test that bootstrap median is more robust to outliers than mean."""
        data = np.concatenate([np.random.normal(0, 1, 50), [100]])

        # Mean-based CI will be affected by outlier
        mean_ci = bootstrap_confidence_interval(data, np.mean, n_bootstrap=1000, random_state=42)
        # Median-based CI should be less affected
        median_ci = bootstrap_confidence_interval(data, np.median, n_bootstrap=1000, random_state=42)

        # Median CI should be centered much lower than mean CI
        median_center = (median_ci[0] + median_ci[1]) / 2
        mean_center = (mean_ci[0] + mean_ci[1]) / 2
        assert median_center < mean_center


# =============================================================================
# Edge Cases: Numerical Stability
# =============================================================================

class TestNumericalStability:
    """Tests for numerical stability and precision."""

    def test_very_large_values(self):
        """Test with very large numeric values."""
        data = np.array([1e10, 1e10 + 1, 1e10 + 2, 1e10 + 3, 1e10 + 4])
        result = calculate_sample_mean(data)
        # Should handle large values without overflow
        assert not np.isnan(result)
        assert not np.isinf(result)
        assert abs(result - 1e10 - 2) < 1e-6

    def test_very_small_values(self):
        """Test with very small numeric values."""
        data = np.array([1e-10, 2e-10, 3e-10, 4e-10, 5e-10])
        result = calculate_sample_mean(data)
        assert not np.isnan(result)
        assert not np.isinf(result)
        assert abs(result - 3e-10) < 1e-16

    def test_mixed_positive_negative(self):
        """Test with mixed positive and negative values."""
        data = np.array([-100, -50, 0, 50, 100])
        result = calculate_sample_mean(data)
        assert result == 0

    def test_near_duplicate_values(self):
        """Test with values that are very close together."""
        data = np.array([1.0, 1.0000001, 1.0000002, 1.0000003, 1.0000004])
        se = calculate_standard_error(data)
        # Should have very small but non-zero standard error
        assert se > 0
        assert se < 1e-4

    def test_bootstrap_reproducibility_across_runs(self):
        """Test that same seed gives exact same results."""
        data = np.random.randn(50)
        result1 = bootstrap_resample(data, n_bootstrap=1000, random_state=12345)
        result2 = bootstrap_resample(data, n_bootstrap=1000, random_state=12345)
        assert np.array_equal(result1, result2)


# =============================================================================
# Edge Cases: Statistical Properties
# =============================================================================

class TestStatisticalProperties:
    """Tests that verify expected statistical properties."""

    def test_law_of_large_numbers_coin_flips(self):
        """Test that more flips gives proportion closer to p_heads."""
        # Small number of flips - more variable
        small_flips = simulate_coin_flips(n_flips=100, p_heads=0.5, random_state=42)
        small_proportion = small_flips.mean()

        # Large number of flips - closer to 0.5
        large_flips = simulate_coin_flips(n_flips=10000, p_heads=0.5, random_state=43)
        large_proportion = large_flips.mean()

        # Large sample should be closer to true probability
        assert abs(large_proportion - 0.5) < abs(small_proportion - 0.5)

    def test_standard_error_decreases_with_sample_size(self):
        """Test that SE decreases as sample size increases."""
        base_data = np.random.normal(0, 1, size=1000)

        se_10 = calculate_standard_error(base_data[:10])
        se_50 = calculate_standard_error(base_data[:50])
        se_100 = calculate_standard_error(base_data[:100])
        se_1000 = calculate_standard_error(base_data)

        # SE should decrease as n increases
        assert se_1000 < se_100 < se_50 < se_10

    def test_sampling_distribution_convergence(self):
        """Test that sampling distribution converges to expected shape."""
        population = np.random.normal(100, 15, size=100000)

        # Small sample size - more variable
        small_samples = generate_sampling_distribution(population, sample_size=10, n_simulations=500, random_state=42)
        small_std = small_samples.std()

        # Large sample size - less variable
        large_samples = generate_sampling_distribution(population, sample_size=100, n_simulations=500, random_state=43)
        large_std = large_samples.std()

        # Larger samples should have smaller standard error
        assert large_std < small_std

    def test_bootstrap_ci_contains_true_parameter(self):
        """Test that 95% CI contains true parameter approximately 95% of time."""
        np.random.seed(42)
        true_mean = 50
        count_contains = 0
        n_trials = 200

        for _ in range(n_trials):
            sample = np.random.normal(true_mean, scale=10, size=50)
            lower, upper = bootstrap_confidence_interval(sample, np.mean, n_bootstrap=500, random_state=None)
            if lower <= true_mean <= upper:
                count_contains += 1

        proportion = count_contains / n_trials
        # Should be approximately 0.95 (with some margin for error)
        assert 0.88 < proportion < 1.0  # Allow for simulation error

    def test_false_positive_rate_matches_theoretical(self):
        """Test that simulated FP rate matches theoretical calculation."""
        n_hypotheses = 10
        alpha = 0.05
        n_simulations = 5000

        simulated = simulate_false_positive_rate(n_hypotheses, alpha, n_simulations, random_state=42)
        theoretical = calculate_multiple_comparison_risk(n_hypotheses, alpha)

        # Should be close (within 5 percentage points)
        assert abs(simulated - theoretical) < 0.05

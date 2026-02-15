"""Test suite for Week 05 solution.py

Tests cover:
1. Simulation Functions (coin flips, binomial)
2. Sampling Distribution Functions (sample mean, standard error, CLT)
3. Bootstrap Functions (resampling, confidence intervals, standard error)
4. False Positive Simulation (multiple comparisons)
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pytest

# Import functions from solution module
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
# 1. Simulation Functions Tests
# =============================================================================

class TestSimulateCoinFlips:
    """Tests for simulate_coin_flips function."""

    def test_simulate_coin_flips_normal_input(self):
        """Test coin flip simulation with normal input."""
        result = simulate_coin_flips(n_flips=100, p_heads=0.5, random_state=42)
        assert len(result) == 100
        assert np.all((result == 0) | (result == 1))
        # With fair coin, should be around 50 heads
        assert 35 <= result.sum() <= 65

    def test_simulate_coin_flips_reproducibility(self):
        """Test that random seed produces reproducible results."""
        result1 = simulate_coin_flips(n_flips=10, p_heads=0.5, random_state=42)
        result2 = simulate_coin_flips(n_flips=10, p_heads=0.5, random_state=42)
        assert np.array_equal(result1, result2)

    def test_simulate_coin_flips_different_seeds(self):
        """Test that different seeds produce different results."""
        result1 = simulate_coin_flips(n_flips=100, p_heads=0.5, random_state=42)
        result2 = simulate_coin_flips(n_flips=100, p_heads=0.5, random_state=43)
        assert not np.array_equal(result1, result2)

    def test_simulate_coin_flips_biased_coin(self):
        """Test simulation with biased coin."""
        result = simulate_coin_flips(n_flips=1000, p_heads=0.7, random_state=42)
        # Should get around 700 heads
        assert 650 <= result.sum() <= 750

    def test_simulate_coin_flips_boundary_one_flip(self):
        """Test simulation with single flip."""
        result = simulate_coin_flips(n_flips=1, p_heads=0.5, random_state=42)
        assert len(result) == 1
        assert result[0] in [0, 1]

    def test_simulate_coin_flips_invalid_n_flips_zero(self):
        """Test that zero flips raises ValueError."""
        with pytest.raises(ValueError, match="n_flips must be positive"):
            simulate_coin_flips(n_flips=0, p_heads=0.5)

    def test_simulate_coin_flips_invalid_n_flips_negative(self):
        """Test that negative flips raises ValueError."""
        with pytest.raises(ValueError, match="n_flips must be positive"):
            simulate_coin_flips(n_flips=-1, p_heads=0.5)

    def test_simulate_coin_flips_invalid_probability(self):
        """Test that invalid probability raises ValueError."""
        with pytest.raises(ValueError, match="p_heads must be between 0 and 1"):
            simulate_coin_flips(n_flips=10, p_heads=1.5)
        with pytest.raises(ValueError, match="p_heads must be between 0 and 1"):
            simulate_coin_flips(n_flips=10, p_heads=-0.1)

    def test_simulate_coin_flips_always_heads(self):
        """Test with p_heads=1 (always heads)."""
        result = simulate_coin_flips(n_flips=100, p_heads=1.0, random_state=42)
        assert np.all(result == 1)

    def test_simulate_coin_flips_always_tails(self):
        """Test with p_heads=0 (always tails)."""
        result = simulate_coin_flips(n_flips=100, p_heads=0.0, random_state=42)
        assert np.all(result == 0)


class TestSimulateBinomial:
    """Tests for simulate_binomial function."""

    def test_simulate_binomial_normal_input(self):
        """Test binomial simulation with normal input."""
        result = simulate_binomial(n_trials=100, p_success=0.5, n_simulations=1000, random_state=42)
        assert len(result) == 1000
        assert np.all(result >= 0) and np.all(result <= 100)
        # Mean should be around 50
        assert 45 <= result.mean() <= 55

    def test_simulate_binomial_reproducibility(self):
        """Test that random seed produces reproducible results."""
        result1 = simulate_binomial(n_trials=10, p_success=0.5, n_simulations=100, random_state=42)
        result2 = simulate_binomial(n_trials=10, p_success=0.5, n_simulations=100, random_state=42)
        assert np.array_equal(result1, result2)

    def test_simulate_binomial_single_simulation(self):
        """Test with single simulation."""
        result = simulate_binomial(n_trials=100, p_success=0.5, n_simulations=1, random_state=42)
        assert len(result) == 1
        assert 0 <= result[0] <= 100

    def test_simulate_binomial_single_trial(self):
        """Test with single trial per simulation."""
        result = simulate_binomial(n_trials=1, p_success=0.5, n_simulations=100, random_state=42)
        assert len(result) == 100
        assert np.all((result == 0) | (result == 1))

    def test_simulate_binomial_invalid_n_trials(self):
        """Test that invalid n_trials raises ValueError."""
        with pytest.raises(ValueError, match="n_trials must be positive"):
            simulate_binomial(n_trials=0, p_success=0.5)
        with pytest.raises(ValueError, match="n_trials must be positive"):
            simulate_binomial(n_trials=-1, p_success=0.5)

    def test_simulate_binomial_invalid_n_simulations(self):
        """Test that invalid n_simulations raises ValueError."""
        with pytest.raises(ValueError, match="n_simulations must be positive"):
            simulate_binomial(n_trials=10, p_success=0.5, n_simulations=0)

    def test_simulate_binomial_invalid_probability(self):
        """Test that invalid probability raises ValueError."""
        with pytest.raises(ValueError, match="p_success must be between 0 and 1"):
            simulate_binomial(n_trials=10, p_success=1.5)

    def test_simulate_binomial_low_success_probability(self):
        """Test with low success probability."""
        result = simulate_binomial(n_trials=100, p_success=0.1, n_simulations=1000, random_state=42)
        # Mean should be around 10
        assert 5 <= result.mean() <= 15

    def test_simulate_binomial_high_success_probability(self):
        """Test with high success probability."""
        result = simulate_binomial(n_trials=100, p_success=0.9, n_simulations=1000, random_state=42)
        # Mean should be around 90
        assert 85 <= result.mean() <= 95


# =============================================================================
# 2. Sampling Distribution Functions Tests
# =============================================================================

class TestCalculateSampleMean:
    """Tests for calculate_sample_mean function."""

    def test_calculate_sample_mean_normal_input(self):
        """Test sample mean calculation with normal input."""
        data = np.array([1, 2, 3, 4, 5])
        result = calculate_sample_mean(data)
        assert result == 3.0

    def test_calculate_sample_mean_with_floats(self):
        """Test sample mean with floating point values."""
        data = np.array([1.5, 2.5, 3.5])
        result = calculate_sample_mean(data)
        assert result == 2.5

    def test_calculate_sample_mean_single_value(self):
        """Test sample mean with single value."""
        data = np.array([42])
        result = calculate_sample_mean(data)
        assert result == 42

    def test_calculate_sample_mean_empty_array(self):
        """Test sample mean with empty array returns NaN."""
        data = np.array([])
        result = calculate_sample_mean(data)
        assert np.isnan(result)

    def test_calculate_sample_mean_negative_values(self):
        """Test sample mean with negative values."""
        data = np.array([-5, -10, -15])
        result = calculate_sample_mean(data)
        assert result == -10

    def test_calculate_sample_mean_large_numbers(self):
        """Test sample mean with large numbers."""
        data = np.array([1000000, 2000000, 3000000])
        result = calculate_sample_mean(data)
        assert result == 2000000


class TestCalculateStandardError:
    """Tests for calculate_standard_error function."""

    def test_calculate_standard_error_normal_input(self):
        """Test standard error calculation with normal input."""
        data = np.array([1, 2, 3, 4, 5])
        result = calculate_standard_error(data)
        # For this data: std ≈ 1.58, SE ≈ 1.58 / sqrt(5) ≈ 0.71
        assert 0.6 < result < 0.8

    def test_calculate_standard_error_constant_data(self):
        """Test standard error with constant data (zero variance)."""
        data = np.array([5, 5, 5, 5, 5])
        result = calculate_standard_error(data)
        # std = 0, so SE should be 0 or very close
        assert result == 0 or abs(result) < 1e-10

    def test_calculate_standard_error_empty_array(self):
        """Test standard error with empty array returns NaN."""
        data = np.array([])
        result = calculate_standard_error(data)
        assert np.isnan(result)

    def test_calculate_standard_error_single_value(self):
        """Test standard error with single value returns NaN."""
        data = np.array([42])
        result = calculate_standard_error(data)
        assert np.isnan(result)

    def test_calculate_standard_error_two_values(self):
        """Test standard error with two values."""
        data = np.array([1, 3])
        result = calculate_standard_error(data)
        # std = sqrt(2), SE = sqrt(2) / sqrt(2) = 1
        assert result == 1.0

    def test_calculate_standard_error_larger_sample(self):
        """Test that larger sample gives smaller SE (all else equal)."""
        data_small = np.array([1, 2, 3, 4, 5])
        data_large = np.concatenate([data_small] * 4)  # 20 values, same mean and similar std
        se_small = calculate_standard_error(data_small)
        se_large = calculate_standard_error(data_large)
        assert se_large < se_small


class TestGenerateSamplingDistribution:
    """Tests for generate_sampling_distribution function."""

    def test_generate_sampling_distribution_normal_input(self):
        """Test sampling distribution generation with normal input."""
        population = np.random.normal(loc=100, scale=15, size=10000)
        result = generate_sampling_distribution(population, sample_size=100, n_simulations=100, random_state=42)
        assert len(result) == 100
        # Sample means should be around population mean
        assert 95 < result.mean() < 105

    def test_generate_sampling_distribution_reproducibility(self):
        """Test that random seed produces reproducible results."""
        population = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result1 = generate_sampling_distribution(population, sample_size=5, n_simulations=10, random_state=42)
        result2 = generate_sampling_distribution(population, sample_size=5, n_simulations=10, random_state=42)
        assert np.array_equal(result1, result2)

    def test_generate_sampling_distribution_clt_shape(self):
        """Test that sampling distribution is approximately normal (CLT)."""
        # Exponential distribution (highly skewed)
        population = np.random.exponential(scale=1.0, size=100000)
        result = generate_sampling_distribution(population, sample_size=50, n_simulations=500, random_state=42)
        # Check if distribution is roughly symmetric
        skewness = ((result - result.mean()) ** 3).mean() / (result.std() ** 3)
        # Sample mean distribution should be much less skewed than population
        assert abs(skewness) < 1  # Much less skewed than exponential

    def test_generate_sampling_distribution_invalid_sample_size(self):
        """Test that invalid sample_size raises ValueError."""
        population = np.array([1, 2, 3, 4, 5])
        with pytest.raises(ValueError, match="sample_size must be positive"):
            generate_sampling_distribution(population, sample_size=0)
        with pytest.raises(ValueError, match="sample_size must be positive"):
            generate_sampling_distribution(population, sample_size=-1)

    def test_generate_sampling_distribution_invalid_n_simulations(self):
        """Test that invalid n_simulations raises ValueError."""
        population = np.array([1, 2, 3, 4, 5])
        with pytest.raises(ValueError, match="n_simulations must be positive"):
            generate_sampling_distribution(population, sample_size=3, n_simulations=0)

    def test_generate_sampling_distribution_sample_exceeds_population(self):
        """Test that sample_size exceeding population raises ValueError."""
        population = np.array([1, 2, 3, 4, 5])
        with pytest.raises(ValueError, match="sample_size cannot exceed population size"):
            generate_sampling_distribution(population, sample_size=10)

    def test_generate_sampling_distribution_empty_population(self):
        """Test that empty population raises ValueError."""
        population = np.array([])
        with pytest.raises(ValueError, match="population cannot be empty"):
            generate_sampling_distribution(population, sample_size=1)

    def test_generate_sampling_distribution_sample_equals_population(self):
        """Test with sample_size equal to population size."""
        population = np.array([1, 2, 3, 4, 5])
        result = generate_sampling_distribution(population, sample_size=5, n_simulations=10, random_state=42)
        assert len(result) == 10
        # All sample means should equal population mean when sampling entire population
        assert np.allclose(result, population.mean())


# =============================================================================
# 3. Bootstrap Functions Tests
# =============================================================================

class TestBootstrapResample:
    """Tests for bootstrap_resample function."""

    def test_bootstrap_resample_normal_input(self):
        """Test bootstrap resampling with normal input."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = bootstrap_resample(data, n_bootstrap=1000, random_state=42)
        assert len(result) == 1000
        # Bootstrap means should be around original mean
        assert 4.5 < result.mean() < 6.5

    def test_bootstrap_resample_reproducibility(self):
        """Test that random seed produces reproducible results."""
        data = np.array([1, 2, 3, 4, 5])
        result1 = bootstrap_resample(data, n_bootstrap=100, random_state=42)
        result2 = bootstrap_resample(data, n_bootstrap=100, random_state=42)
        assert np.array_equal(result1, result2)

    def test_bootstrap_resample_small_sample(self):
        """Test bootstrap with small sample."""
        data = np.array([1, 2, 3])
        result = bootstrap_resample(data, n_bootstrap=100, random_state=42)
        assert len(result) == 100
        # Bootstrap means should be in range [1, 3]
        assert np.all(result >= 1) and np.all(result <= 3)

    def test_bootstrap_resample_large_sample(self):
        """Test bootstrap with large sample."""
        data = np.random.normal(loc=50, scale=10, size=1000)
        result = bootstrap_resample(data, n_bootstrap=100, random_state=42)
        assert len(result) == 100
        # Bootstrap means should be around original mean
        assert 45 < result.mean() < 55

    def test_bootstrap_resample_invalid_n_bootstrap(self):
        """Test that invalid n_bootstrap raises ValueError."""
        data = np.array([1, 2, 3, 4, 5])
        with pytest.raises(ValueError, match="n_bootstrap must be positive"):
            bootstrap_resample(data, n_bootstrap=0)

    def test_bootstrap_resample_empty_data(self):
        """Test that empty data raises ValueError."""
        data = np.array([])
        with pytest.raises(ValueError, match="data cannot be empty"):
            bootstrap_resample(data)

    def test_bootstrap_resample_skewed_distribution(self):
        """Test bootstrap with skewed distribution."""
        data = np.random.exponential(scale=1.0, size=100)
        result = bootstrap_resample(data, n_bootstrap=500, random_state=42)
        assert len(result) == 500
        # Bootstrap means should be positive
        assert np.all(result > 0)


class TestBootstrapConfidenceInterval:
    """Tests for bootstrap_confidence_interval function."""

    def test_bootstrap_ci_mean_normal_input(self):
        """Test bootstrap CI for mean with normal input."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        lower, upper = bootstrap_confidence_interval(data, np.mean, n_bootstrap=1000, random_state=42)
        assert lower < upper
        # Original mean (5.5) should be within CI
        assert lower < 5.5 < upper

    def test_bootstrap_ci_reproducibility(self):
        """Test that random seed produces reproducible results."""
        data = np.array([1, 2, 3, 4, 5])
        ci1 = bootstrap_confidence_interval(data, np.mean, n_bootstrap=100, random_state=42)
        ci2 = bootstrap_confidence_interval(data, np.mean, n_bootstrap=100, random_state=42)
        assert ci1 == ci2

    def test_bootstrap_ci_different_confidence_levels(self):
        """Test with different confidence levels."""
        data = np.random.normal(loc=0, scale=1, size=100)
        ci_90 = bootstrap_confidence_interval(data, np.mean, n_bootstrap=1000, confidence_level=0.90, random_state=42)
        ci_95 = bootstrap_confidence_interval(data, np.mean, n_bootstrap=1000, confidence_level=0.95, random_state=42)
        ci_99 = bootstrap_confidence_interval(data, np.mean, n_bootstrap=1000, confidence_level=0.99, random_state=42)
        # Higher confidence level = wider interval
        assert ci_99[1] - ci_99[0] > ci_95[1] - ci_95[0] > ci_90[1] - ci_90[0]

    def test_bootstrap_ci_median(self):
        """Test bootstrap CI for median."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        lower, upper = bootstrap_confidence_interval(data, np.median, n_bootstrap=1000, random_state=42)
        assert lower < upper
        # Original median (5.5) should be within CI
        assert lower < 5.5 < upper

    def test_bootstrap_ci_invalid_confidence_level(self):
        """Test that invalid confidence_level raises ValueError."""
        data = np.array([1, 2, 3, 4, 5])
        with pytest.raises(ValueError, match="confidence_level must be between 0 and 1"):
            bootstrap_confidence_interval(data, np.mean, confidence_level=0)
        with pytest.raises(ValueError, match="confidence_level must be between 0 and 1"):
            bootstrap_confidence_interval(data, np.mean, confidence_level=1)

    def test_bootstrap_ci_empty_data(self):
        """Test that empty data raises ValueError."""
        data = np.array([])
        with pytest.raises(ValueError, match="data cannot be empty"):
            bootstrap_confidence_interval(data, np.mean)

    def test_bootstrap_ci_invalid_n_bootstrap(self):
        """Test that invalid n_bootstrap raises ValueError."""
        data = np.array([1, 2, 3, 4, 5])
        with pytest.raises(ValueError, match="n_bootstrap must be positive"):
            bootstrap_confidence_interval(data, np.mean, n_bootstrap=0)


class TestBootstrapStandardError:
    """Tests for bootstrap_standard_error function."""

    def test_bootstrap_se_mean_normal_input(self):
        """Test bootstrap SE for mean with normal input."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        se = bootstrap_standard_error(data, np.mean, n_bootstrap=1000, random_state=42)
        assert se > 0
        # SE should be reasonably small for this data
        assert se < 2

    def test_bootstrap_se_reproducibility(self):
        """Test that random seed produces reproducible results."""
        data = np.array([1, 2, 3, 4, 5])
        se1 = bootstrap_standard_error(data, np.mean, n_bootstrap=100, random_state=42)
        se2 = bootstrap_standard_error(data, np.mean, n_bootstrap=100, random_state=42)
        assert se1 == se2

    def test_bootstrap_se_median(self):
        """Test bootstrap SE for median."""
        data = np.random.normal(loc=0, scale=1, size=100)
        se = bootstrap_standard_error(data, np.median, n_bootstrap=500, random_state=42)
        assert se > 0

    def test_bootstrap_se_constant_data(self):
        """Test bootstrap SE with constant data."""
        data = np.array([5, 5, 5, 5, 5])
        se = bootstrap_standard_error(data, np.mean, n_bootstrap=100, random_state=42)
        # SE should be 0 or very close for constant data
        assert se == 0 or abs(se) < 1e-10

    def test_bootstrap_se_empty_data(self):
        """Test that empty data raises ValueError."""
        data = np.array([])
        with pytest.raises(ValueError, match="data cannot be empty"):
            bootstrap_standard_error(data, np.mean)

    def test_bootstrap_se_invalid_n_bootstrap(self):
        """Test that invalid n_bootstrap raises ValueError."""
        data = np.array([1, 2, 3, 4, 5])
        with pytest.raises(ValueError, match="n_bootstrap must be positive"):
            bootstrap_standard_error(data, np.mean, n_bootstrap=0)

    def test_bootstrap_se_comparison_larger_sample(self):
        """Test that larger sample gives smaller SE."""
        small_data = np.random.normal(loc=0, scale=1, size=30)
        large_data = np.random.normal(loc=0, scale=1, size=300)
        se_small = bootstrap_standard_error(small_data, np.mean, n_bootstrap=500, random_state=42)
        se_large = bootstrap_standard_error(large_data, np.mean, n_bootstrap=500, random_state=43)
        # SE should be smaller for larger sample
        assert se_large < se_small


# =============================================================================
# 4. False Positive Simulation Tests
# =============================================================================

class TestSimulateFalsePositiveRate:
    """Tests for simulate_false_positive_rate function."""

    def test_simulate_false_positive_rate_single_hypothesis(self):
        """Test false positive rate with single hypothesis."""
        rate = simulate_false_positive_rate(n_hypotheses=1, alpha=0.05, n_simulations=10000, random_state=42)
        # Should be approximately alpha (0.05)
        assert 0.04 < rate < 0.06

    def test_simulate_false_positive_rate_reproducibility(self):
        """Test that random seed produces reproducible results."""
        rate1 = simulate_false_positive_rate(n_hypotheses=5, alpha=0.05, n_simulations=1000, random_state=42)
        rate2 = simulate_false_positive_rate(n_hypotheses=5, alpha=0.05, n_simulations=1000, random_state=42)
        assert rate1 == rate2

    def test_simulate_false_positive_rate_multiple_hypotheses(self):
        """Test false positive rate with multiple hypotheses."""
        rate = simulate_false_positive_rate(n_hypotheses=5, alpha=0.05, n_simulations=5000, random_state=42)
        # Theoretical rate: 1 - (1 - 0.05)^5 ≈ 0.226
        # Allow for simulation error
        assert 0.18 < rate < 0.28

    def test_simulate_false_positive_rate_high_alpha(self):
        """Test with high alpha value."""
        rate = simulate_false_positive_rate(n_hypotheses=1, alpha=0.10, n_simulations=10000, random_state=42)
        # Should be approximately 0.10
        assert 0.08 < rate < 0.12

    def test_simulate_false_positive_rate_invalid_n_hypotheses(self):
        """Test that invalid n_hypotheses raises ValueError."""
        with pytest.raises(ValueError, match="n_hypotheses must be positive"):
            simulate_false_positive_rate(n_hypotheses=0)
        with pytest.raises(ValueError, match="n_hypotheses must be positive"):
            simulate_false_positive_rate(n_hypotheses=-1)

    def test_simulate_false_positive_rate_invalid_alpha(self):
        """Test that invalid alpha raises ValueError."""
        with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
            simulate_false_positive_rate(n_hypotheses=5, alpha=0)
        with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
            simulate_false_positive_rate(n_hypotheses=5, alpha=1)

    def test_simulate_false_positive_rate_invalid_n_simulations(self):
        """Test that invalid n_simulations raises ValueError."""
        with pytest.raises(ValueError, match="n_simulations must be positive"):
            simulate_false_positive_rate(n_hypotheses=5, n_simulations=0)

    def test_simulate_false_positive_rate_many_hypotheses(self):
        """Test with many hypotheses (high false positive risk)."""
        rate = simulate_false_positive_rate(n_hypotheses=20, alpha=0.05, n_simulations=2000, random_state=42)
        # Theoretical rate: 1 - (1 - 0.05)^20 ≈ 0.64
        # Should be very high
        assert 0.55 < rate < 0.75


class TestCalculateMultipleComparisonRisk:
    """Tests for calculate_multiple_comparison_risk function."""

    def test_calculate_multiple_comparison_risk_single_hypothesis(self):
        """Test risk calculation with single hypothesis."""
        risk = calculate_multiple_comparison_risk(n_hypotheses=1, alpha=0.05)
        # P(FP) = 1 - (1 - 0.05)^1 = 0.05 (allowing for floating point precision)
        assert abs(risk - 0.05) < 1e-10

    def test_calculate_multiple_comparison_risk_multiple_hypotheses(self):
        """Test risk calculation with multiple hypotheses."""
        risk = calculate_multiple_comparison_risk(n_hypotheses=5, alpha=0.05)
        # P(FP) = 1 - (1 - 0.05)^5 ≈ 0.226
        expected = 1 - (1 - 0.05) ** 5
        assert abs(risk - expected) < 1e-10

    def test_calculate_multiple_comparison_risk_different_alpha(self):
        """Test with different alpha values."""
        risk_01 = calculate_multiple_comparison_risk(n_hypotheses=5, alpha=0.01)
        risk_05 = calculate_multiple_comparison_risk(n_hypotheses=5, alpha=0.05)
        risk_10 = calculate_multiple_comparison_risk(n_hypotheses=5, alpha=0.10)
        # Higher alpha = higher risk
        assert risk_10 > risk_05 > risk_01

    def test_calculate_multiple_comparison_risk_many_hypotheses(self):
        """Test with many hypotheses."""
        risk = calculate_multiple_comparison_risk(n_hypotheses=20, alpha=0.05)
        # P(FP) = 1 - (1 - 0.05)^20 ≈ 0.64
        expected = 1 - (1 - 0.05) ** 20
        assert abs(risk - expected) < 1e-10
        # Risk should be very high
        assert risk > 0.6

    def test_calculate_multiple_comparison_risk_invalid_n_hypotheses(self):
        """Test that invalid n_hypotheses raises ValueError."""
        with pytest.raises(ValueError, match="n_hypotheses must be positive"):
            calculate_multiple_comparison_risk(n_hypotheses=0)
        with pytest.raises(ValueError, match="n_hypotheses must be positive"):
            calculate_multiple_comparison_risk(n_hypotheses=-1)

    def test_calculate_multiple_comparison_risk_invalid_alpha(self):
        """Test that invalid alpha raises ValueError."""
        with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
            calculate_multiple_comparison_risk(n_hypotheses=5, alpha=0)
        with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
            calculate_multiple_comparison_risk(n_hypotheses=5, alpha=1)

    def test_calculate_multiple_comparison_risk_consistency_with_simulation(self):
        """Test that theoretical calculation matches simulation (approximately)."""
        n_hypotheses = 5
        alpha = 0.05
        theoretical_risk = calculate_multiple_comparison_risk(n_hypotheses, alpha)
        simulated_risk = simulate_false_positive_rate(n_hypotheses, alpha, n_simulations=10000, random_state=42)
        # Should be close (allow for simulation error)
        assert abs(simulated_risk - theoretical_risk) < 0.05

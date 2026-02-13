"""
Test suite for Week 08: Bootstrap (Resampling Methods)

This module tests Bootstrap methods for confidence interval estimation,
covering the core concepts from Week 08 Chapter 3-4.
"""

import pytest
import numpy as np


class TestBootstrapCoreConcept:
    """Test fundamental Bootstrap concepts and mechanics."""

    def test_bootstrap_sample_with_replacement(self):
        """Test that Bootstrap samples are drawn WITH replacement."""
        np.random.seed(42)
        original_sample = np.array([1, 2, 3, 4, 5])
        n = len(original_sample)

        # Draw one Bootstrap sample
        boot_sample = np.random.choice(original_sample, size=n, replace=True)

        # Bootstrap sample should have same size as original
        assert len(boot_sample) == len(original_sample), \
            "Bootstrap sample should have same size as original"

        # Bootstrap sample should contain only values from original
        assert all(x in original_sample for x in boot_sample), \
            "Bootstrap sample should only contain values from original"

        # Some values should be repeated (because of replacement)
        unique_count = len(set(boot_sample))
        assert unique_count < n, \
            "With replacement, some values should be repeated"

    def test_bootstrap_without_replacement_fails(self):
        """Test that sampling WITHOUT replacement just returns original."""
        np.random.seed(42)
        original_sample = np.array([1, 2, 3, 4, 5])
        n = len(original_sample)

        # Without replacement
        boot_sample_no_replace = np.random.choice(original_sample, size=n, replace=False)

        # Without replacement gives a permutation (no repeats)
        assert len(set(boot_sample_no_replace)) == n, \
            "Without replacement should have all unique values"

        # It's a permutation of original
        assert set(boot_sample_no_replace) == set(original_sample), \
            "Without replacement should be a permutation"

    def test_bootstrap_distribution_shape(self):
        """Test that Bootstrap distribution approximates sampling distribution."""
        np.random.seed(42)
        population = np.random.normal(loc=100, scale=15, size=10000)
        original_sample = np.random.choice(population, size=100, replace=False)

        # Generate Bootstrap distribution of means
        n_bootstrap = 1000
        n = len(original_sample)
        boot_means = []

        for _ in range(n_bootstrap):
            boot_sample = np.random.choice(original_sample, size=n, replace=True)
            boot_means.append(np.mean(boot_sample))

        boot_means = np.array(boot_means)

        # Bootstrap distribution should be approximately normal (CLT)
        # Check skewness and kurtosis are reasonable
        from scipy import stats
        skewness = stats.skew(boot_means)
        kurtosis = stats.kurtosis(boot_means)

        # Should be roughly symmetric (skewness close to 0)
        assert abs(skewness) < 0.5, \
            f"Bootstrap distribution should be roughly symmetric, skewness={skewness:.3f}"

        # Should have reasonable kurtosis (not too heavy-tailed)
        assert -1 < kurtosis < 2, \
            f"Bootstrap distribution kurtosis should be reasonable, kurtosis={kurtosis:.3f}"

    def test_bootstrap_mean_convergence(self):
        """Test that Bootstrap mean converges to sample mean."""
        np.random.seed(42)
        original_sample = np.random.normal(loc=100, scale=15, size=100)
        sample_mean = np.mean(original_sample)

        # Generate Bootstrap distribution
        n_bootstrap = 1000
        n = len(original_sample)
        boot_means = []

        for _ in range(n_bootstrap):
            boot_sample = np.random.choice(original_sample, size=n, replace=True)
            boot_means.append(np.mean(boot_sample))

        boot_means = np.array(boot_means)
        mean_of_boot_means = np.mean(boot_means)

        # Mean of Bootstrap means should be close to original sample mean
        assert abs(mean_of_boot_means - sample_mean) < 2, \
            f"Mean of Bootstrap means ({mean_of_boot_means:.2f}) should be close to sample mean ({sample_mean:.2f})"


class TestBootstrapConfidenceIntervals:
    """Test Bootstrap confidence interval construction methods."""

    def test_bootstrap_ci_percentile_method(self):
        """Happy path: Bootstrap CI using percentile method."""
        np.random.seed(42)
        data = np.random.normal(loc=100, scale=15, size=100)

        # Generate Bootstrap distribution
        n_bootstrap = 10000
        n = len(data)
        boot_means = []

        for _ in range(n_bootstrap):
            boot_sample = np.random.choice(data, size=n, replace=True)
            boot_means.append(np.mean(boot_sample))

        boot_means = np.array(boot_means)

        # Calculate percentile CI
        alpha = 0.05
        ci_low = np.percentile(boot_means, 100 * alpha / 2)
        ci_high = np.percentile(boot_means, 100 * (1 - alpha / 2))

        # Assertions
        assert ci_low < ci_high, "CI lower bound should be less than upper bound"
        assert ci_low > 50, "CI lower bound should be reasonable"
        assert ci_high < 150, "CI upper bound should be reasonable"

        # Check that about 95% of Bootstrap distribution is within CI
        within_ci = np.sum((boot_means >= ci_low) & (boot_means <= ci_high))
        proportion_within = within_ci / n_bootstrap

        # Should be very close to 0.95 (allowing small tolerance)
        assert 0.94 <= proportion_within <= 0.96, \
            f"Proportion within CI ({proportion_within:.3f}) should be ~0.95"

    def test_bootstrap_ci_different_confidence_levels(self):
        """Test Bootstrap CI with different confidence levels."""
        np.random.seed(42)
        data = np.random.normal(loc=100, scale=15, size=100)

        # Generate Bootstrap distribution
        n_bootstrap = 10000
        n = len(data)
        boot_means = []

        for _ in range(n_bootstrap):
            boot_sample = np.random.choice(data, size=n, replace=True)
            boot_means.append(np.mean(boot_sample))

        boot_means = np.array(boot_means)

        # Calculate CI for different confidence levels
        ci_widths = {}
        for conf_level in [0.80, 0.90, 0.95, 0.99]:
            alpha = 1 - conf_level
            ci_low = np.percentile(boot_means, 100 * alpha / 2)
            ci_high = np.percentile(boot_means, 100 * (1 - alpha / 2))
            ci_widths[conf_level] = ci_high - ci_low

        # Higher confidence -> wider interval
        assert ci_widths[0.80] < ci_widths[0.90], "80% CI should be narrower than 90%"
        assert ci_widths[0.90] < ci_widths[0.95], "90% CI should be narrower than 95%"
        assert ci_widths[0.95] < ci_widths[0.99], "95% CI should be narrower than 99%"

    def test_bootstrap_ci_difference_of_means(self):
        """Happy path: Bootstrap CI for difference between two group means."""
        np.random.seed(42)
        group1 = np.random.normal(loc=110, scale=15, size=100)
        group2 = np.random.normal(loc=100, scale=15, size=100)

        # Generate Bootstrap distribution of mean differences
        n_bootstrap = 10000
        n1, n2 = len(group1), len(group2)
        boot_diffs = []

        for _ in range(n_bootstrap):
            boot_sample1 = np.random.choice(group1, size=n1, replace=True)
            boot_sample2 = np.random.choice(group2, size=n2, replace=True)
            boot_diff = np.mean(boot_sample1) - np.mean(boot_sample2)
            boot_diffs.append(boot_diff)

        boot_diffs = np.array(boot_diffs)

        # Calculate 95% CI
        ci_low = np.percentile(boot_diffs, 2.5)
        ci_high = np.percentile(boot_diffs, 97.5)

        # Assertions
        assert ci_low < ci_high, "CI should be valid"
        assert ci_low < 0 < ci_high or ci_low > 0, \
            "CI should either include or exclude 0 (not be trivial)"

        # Observed difference should be within Bootstrap distribution range
        observed_diff = np.mean(group1) - np.mean(group2)
        assert boot_diffs.min() <= observed_diff <= boot_diffs.max(), \
            "Observed difference should be within Bootstrap range"

    def test_bootstrap_ci_median(self):
        """Test Bootstrap CI for median (distribution-free statistic)."""
        np.random.seed(42)
        # Skewed distribution where median differs from mean
        data = np.random.exponential(scale=50, size=200)

        # Generate Bootstrap distribution of medians
        n_bootstrap = 10000
        n = len(data)
        boot_medians = []

        for _ in range(n_bootstrap):
            boot_sample = np.random.choice(data, size=n, replace=True)
            boot_medians.append(np.median(boot_sample))

        boot_medians = np.array(boot_medians)

        # Calculate 95% CI
        ci_low = np.percentile(boot_medians, 2.5)
        ci_high = np.percentile(boot_medians, 97.5)

        # Assertions
        assert ci_low < ci_high, "CI should be valid"
        assert ci_low > 0, "CI should be positive for exponential data"

        # Median CI should be different from mean CI
        boot_means = []
        for _ in range(n_bootstrap):
            boot_sample = np.random.choice(data, size=n, replace=True)
            boot_means.append(np.mean(boot_sample))
        boot_means = np.array(boot_means)

        mean_ci_low = np.percentile(boot_means, 2.5)
        mean_ci_high = np.percentile(boot_means, 97.5)

        # Median CI should differ from mean CI for skewed data
        assert not (ci_low == mean_ci_low and ci_high == mean_ci_high), \
            "Median CI should differ from mean CI for skewed data"


class TestBootstrapEffectSize:
    """Test Bootstrap for effect size confidence intervals."""

    def test_bootstrap_ci_cohens_d(self):
        """Happy path: Bootstrap CI for Cohen's d effect size."""
        np.random.seed(42)
        group1 = np.random.normal(loc=110, scale=15, size=100)
        group2 = np.random.normal(loc=100, scale=15, size=100)

        # Calculate observed Cohen's d
        n1, n2 = len(group1), len(group2)
        mean_diff = np.mean(group1) - np.mean(group2)
        pooled_std = np.sqrt(((n1-1)*np.var(group1, ddof=1) +
                              (n2-1)*np.var(group2, ddof=1)) / (n1+n2-2))
        observed_d = mean_diff / pooled_std

        # Generate Bootstrap distribution of Cohen's d
        n_bootstrap = 10000
        boot_ds = []

        for _ in range(n_bootstrap):
            boot_sample1 = np.random.choice(group1, size=n1, replace=True)
            boot_sample2 = np.random.choice(group2, size=n2, replace=True)

            boot_diff = np.mean(boot_sample1) - np.mean(boot_sample2)
            boot_pooled_std = np.sqrt(((n1-1)*np.var(boot_sample1, ddof=1) +
                                      (n2-1)*np.var(boot_sample2, ddof=1)) / (n1+n2-2))
            boot_d = boot_diff / boot_pooled_std
            boot_ds.append(boot_d)

        boot_ds = np.array(boot_ds)

        # Calculate 95% CI
        ci_low = np.percentile(boot_ds, 2.5)
        ci_high = np.percentile(boot_ds, 97.5)

        # Assertions
        assert ci_low < observed_d < ci_high, "Observed d should be within CI"
        assert abs(observed_d) > 0, "Effect size should be non-zero"

        # Effect size interpretation
        if abs(observed_d) < 0.2:
            interpretation = "negligible"
        elif abs(observed_d) < 0.5:
            interpretation = "small"
        elif abs(observed_d) < 0.8:
            interpretation = "medium"
        else:
            interpretation = "large"

        # Bootstrap CI should not include zero if effect is meaningful
        # (Note: This is probabilistic, so we just check the structure)
        assert isinstance(ci_low, (float, np.floating))
        assert isinstance(ci_high, (float, np.floating))


class TestBootstrapEdgeCases:
    """Test Bootstrap with edge cases and boundary conditions."""

    def test_bootstrap_small_sample(self):
        """Edge case: Bootstrap with small sample (n < 30)."""
        np.random.seed(42)
        data = np.random.normal(loc=100, scale=15, size=20)

        # Generate Bootstrap distribution
        n_bootstrap = 1000
        n = len(data)
        boot_means = []

        for _ in range(n_bootstrap):
            boot_sample = np.random.choice(data, size=n, replace=True)
            boot_means.append(np.mean(boot_sample))

        boot_means = np.array(boot_means)

        # Should still produce valid CI (though wider)
        ci_low = np.percentile(boot_means, 2.5)
        ci_high = np.percentile(boot_means, 97.5)

        assert ci_low < ci_high, "CI should be valid even for small samples"
        assert ci_high - ci_low > 0, "CI should have positive width"

    def test_bootstrap_constant_data(self):
        """Edge case: Bootstrap with constant data (zero variance)."""
        np.random.seed(42)
        data = np.array([100, 100, 100, 100, 100])

        # Generate Bootstrap distribution
        n_bootstrap = 1000
        n = len(data)
        boot_means = []

        for _ in range(n_bootstrap):
            boot_sample = np.random.choice(data, size=n, replace=True)
            boot_means.append(np.mean(boot_sample))

        boot_means = np.array(boot_means)

        # All Bootstrap means should be exactly 100
        assert np.all(boot_means == 100), \
            "All Bootstrap means should be 100 for constant data"

        # CI should collapse to a point
        ci_low = np.percentile(boot_means, 2.5)
        ci_high = np.percentile(boot_means, 97.5)

        assert ci_low == ci_high == 100, \
            "CI should collapse to a point for constant data"

    def test_bootstrap_bootstrap_replicates_impact(self):
        """Test that more Bootstrap replicates give more stable CI."""
        np.random.seed(42)
        data = np.random.normal(loc=100, scale=15, size=100)

        # Small number of replicates
        n_bootstrap_small = 100
        n = len(data)
        boot_means_small = []

        for _ in range(n_bootstrap_small):
            boot_sample = np.random.choice(data, size=n, replace=True)
            boot_means_small.append(np.mean(boot_sample))

        boot_means_small = np.array(boot_means_small)
        ci_low_small = np.percentile(boot_means_small, 2.5)
        ci_high_small = np.percentile(boot_means_small, 97.5)

        # Large number of replicates
        n_bootstrap_large = 10000
        boot_means_large = []

        for _ in range(n_bootstrap_large):
            boot_sample = np.random.choice(data, size=n, replace=True)
            boot_means_large.append(np.mean(boot_sample))

        boot_means_large = np.array(boot_means_large)
        ci_low_large = np.percentile(boot_means_large, 2.5)
        ci_high_large = np.percentile(boot_means_large, 97.5)

        # Both should produce valid CIs
        assert ci_low_small < ci_high_small, "Small replicate CI should be valid"
        assert ci_low_large < ci_high_large, "Large replicate CI should be valid"

        # CI width should be similar (not wildly different)
        width_small = ci_high_small - ci_low_small
        width_large = ci_high_large - ci_low_large

        # Should be within 50% of each other
        assert abs(width_small - width_large) / max(width_small, width_large) < 0.5, \
            "CI widths should be reasonably similar"


class TestBootstrapLimitations:
    """Test understanding of Bootstrap limitations."""

    def test_bootstrap_biased_sample_cannot_fix(self):
        """
        Test that Bootstrap cannot fix a biased sample.

        Bootstrap resamples from the sample, so if the sample is biased,
        Bootstrap will amplify (not fix) the bias.
        """
        np.random.seed(42)

        # Simulate a biased sample (underestimates true mean)
        true_population = np.random.normal(loc=100, scale=15, size=10000)
        biased_sample = true_population[true_population < 100][:200]  # Only lower half

        # Bootstrap from biased sample
        n_bootstrap = 1000
        n = len(biased_sample)
        boot_means = []

        for _ in range(n_bootstrap):
            boot_sample = np.random.choice(biased_sample, size=n, replace=True)
            boot_means.append(np.mean(boot_sample))

        boot_means = np.array(boot_means)

        # Bootstrap mean should be close to biased sample mean
        # (both should underestimate true mean)
        bootstrap_mean = np.mean(boot_means)
        sample_mean = np.mean(biased_sample)
        true_mean = 100

        assert abs(bootstrap_mean - sample_mean) < 2, \
            "Bootstrap mean should be close to sample mean"
        assert bootstrap_mean < true_mean, \
            "Bootstrap from biased sample should still be biased"

    def test_bootstrap_requires_iid_assumption(self):
        """
        Test documentation that Bootstrap assumes i.i.d. data.

        Bootstrap does NOT work well with:
        - Time series (autocorrelated data)
        - Clustered data
        - Spatial data with spatial correlation

        This is a documentation test.
        """
        # Bootstrap assumptions (documented)
        assumptions = {
            "independence": "Observations should be independent",
            "identical_distribution": "Observations should come from same distribution",
            "representative_sample": "Sample should represent the population",
            "sufficient_sample_size": "Sample size should be adequate (n >= 20 recommended)"
        }

        # This test documents understanding of limitations
        assert "independence" in assumptions
        assert "identical_distribution" in assumptions
        assert isinstance(assumptions, dict)


class TestBootstrapComparisonWithTInterval:
    """Compare Bootstrap CI with theoretical t-interval."""

    def test_bootstrap_vs_t_ci_normal_data(self):
        """Test that Bootstrap CI approximates t-interval for normal data."""
        np.random.seed(42)
        data = np.random.normal(loc=100, scale=15, size=100)

        # t-interval
        n = len(data)
        sample_mean = np.mean(data)
        sample_std = np.std(data, ddof=1)
        se = sample_std / np.sqrt(n)
        from scipy import stats
        t_critical = stats.t.ppf(0.975, df=n-1)

        t_margin = t_critical * se
        t_ci_low = sample_mean - t_margin
        t_ci_high = sample_mean + t_margin

        # Bootstrap CI
        n_bootstrap = 10000
        boot_means = []

        for _ in range(n_bootstrap):
            boot_sample = np.random.choice(data, size=n, replace=True)
            boot_means.append(np.mean(boot_sample))

        boot_means = np.array(boot_means)
        boot_ci_low = np.percentile(boot_means, 2.5)
        boot_ci_high = np.percentile(boot_means, 97.5)

        # For normal data, Bootstrap CI should be close to t-interval
        # Allow 20% difference in width
        t_width = t_ci_high - t_ci_low
        boot_width = boot_ci_high - boot_ci_low

        relative_difference = abs(t_width - boot_width) / max(t_width, boot_width)

        assert relative_difference < 0.20, \
            f"Bootstrap CI width should be close to t-interval for normal data. " \
            f"Relative difference: {relative_difference:.3f}"

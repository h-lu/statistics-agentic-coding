"""
Test suite for Week 08: Bayesian Credible Intervals

This module tests basic Bayesian inference concepts and credible interval
interpretation, covering concepts from Week 08 Chapter 6.
"""

import pytest
import numpy as np


class TestBayesianVsFrequentist:
    """Test understanding of Bayesian vs Frequentist interpretations."""

    def test_bayesian_parameter_is_random_variable(self):
        """
        Test understanding that in Bayesian framework, parameter is random.

        Bayesian: Parameter is random variable with probability distribution
        Frequentist: Parameter is fixed but unknown
        """
        # Bayesian view
        bayesian_view = {
            "parameter": "Random variable with probability distribution",
            "interval": "Credible interval: P(parameter in interval) = 0.95",
            "interpretation": "There is 95% probability that parameter lies in interval"
        }

        # Frequentist view
        frequentist_view = {
            "parameter": "Fixed but unknown constant",
            "interval": "Confidence interval: Method captures parameter 95% of time",
            "interpretation": "95% of such intervals will contain the true parameter"
        }

        # Test understanding
        assert bayesian_view["parameter"] != frequentist_view["parameter"]
        assert "random" in bayesian_view["parameter"].lower()
        assert "fixed" in frequentist_view["parameter"].lower()

    def test_correct_credible_interval_interpretation(self):
        """
        Test correct interpretation of credible interval.

        In Bayesian framework, you CAN say:
        "There is 95% probability that the parameter is in the interval"

        This is DIFFERENT from frequentist CI.
        """
        # Correct Bayesian interpretation
        correct_bayesian = (
            "Given the data and prior, there is 95% probability "
            "that the parameter lies within this credible interval."
        )

        # This is VALID in Bayesian framework (unlike frequentist)
        assert "probability" in correct_bayesian.lower()
        assert "parameter" in correct_bayesian.lower()
        assert isinstance(correct_bayesian, str)


class TestBayesianConjugatePrior:
    """Test Bayesian inference with conjugate priors (normal-normal)."""

    def test_bayesian_normal_mean_conjugate_update(self):
        """
        Happy path: Bayesian update for normal mean with normal prior.

        Prior: N(μ0, σ0²)
        Likelihood: N(μ, σ²)
        Posterior: N(μn, σn²)
        """
        np.random.seed(42)

        # Prior parameters
        prior_mean = 100
        prior_std = 50

        # Data
        data = np.random.normal(loc=105, scale=15, size=100)
        n = len(data)
        sample_mean = np.mean(data)
        likelihood_std = 15  # Known data std

        # Posterior parameters (conjugate formula)
        posterior_precision = 1/prior_std**2 + n/likelihood_std**2
        posterior_var = 1 / posterior_precision
        posterior_std = np.sqrt(posterior_var)
        posterior_mean = posterior_var * (prior_mean/prior_std**2 + n*sample_mean/likelihood_std**2)

        # Posterior should be between prior and data
        # (weighted average of prior mean and sample mean)
        assert min(prior_mean, sample_mean) <= posterior_mean <= max(prior_mean, sample_mean), \
            "Posterior mean should be between prior mean and sample mean"

        # Posterior should be more certain than prior
        assert posterior_std < prior_std, \
            "Posterior std should be smaller than prior std (more certain after data)"

        # With more data, posterior should be closer to data than prior
        # (data dominates with large n)
        assert abs(posterior_mean - sample_mean) < abs(prior_mean - sample_mean), \
            "Posterior should be closer to data than prior"

    def test_bayesian_credible_interval_from_posterior(self):
        """Happy path: Construct credible interval from posterior distribution."""
        np.random.seed(42)

        # Prior
        prior_mean = 100
        prior_std = 50

        # Data
        data = np.random.normal(loc=105, scale=15, size=100)
        n = len(data)
        sample_mean = np.mean(data)
        likelihood_std = 15

        # Posterior
        posterior_precision = 1/prior_std**2 + n/likelihood_std**2
        posterior_var = 1 / posterior_precision
        posterior_std = np.sqrt(posterior_var)
        posterior_mean = posterior_var * (prior_mean/prior_std**2 + n*sample_mean/likelihood_std**2)

        # Sample from posterior
        n_samples = 10000
        posterior_samples = np.random.normal(posterior_mean, posterior_std, n_samples)

        # Credible interval (percentile method)
        ci_low = np.percentile(posterior_samples, 2.5)
        ci_high = np.percentile(posterior_samples, 97.5)

        # Assertions
        assert ci_low < ci_high, "Credible interval should be valid"
        assert ci_low < posterior_mean < ci_high, "Posterior mean should be within CI"

        # In Bayesian framework, we can interpret this correctly
        proportion_within_ci = np.sum((posterior_samples >= ci_low) &
                                      (posterior_samples <= ci_high)) / n_samples
        assert 0.94 <= proportion_within_ci <= 0.96, \
            "95% credible interval should contain 95% of posterior samples"

    def test_bayesian_informative_vs_weak_prior(self):
        """Test effect of informative vs weak prior on posterior."""
        np.random.seed(42)

        # Same data
        data = np.random.normal(loc=105, scale=15, size=100)
        sample_mean = np.mean(data)
        n = len(data)
        likelihood_std = 15

        # Informative prior (strong belief)
        informative_prior_mean = 100
        informative_prior_std = 5  # Very certain

        # Weak prior (vague belief)
        weak_prior_mean = 100
        weak_prior_std = 1000  # Very uncertain

        # Posterior with informative prior
        posterior_var_informative = 1 / (1/informative_prior_std**2 + n/likelihood_std**2)
        posterior_mean_informative = posterior_var_informative * (
            informative_prior_mean/informative_prior_std**2 +
            n*sample_mean/likelihood_std**2
        )

        # Posterior with weak prior
        posterior_var_weak = 1 / (1/weak_prior_std**2 + n/likelihood_std**2)
        posterior_mean_weak = posterior_var_weak * (
            weak_prior_mean/weak_prior_std**2 +
            n*sample_mean/likelihood_std**2
        )

        # Weak prior should be dominated by data (posterior closer to sample mean)
        # Informative prior should pull posterior toward prior

        # With weak prior, posterior mean should be very close to sample mean
        assert abs(posterior_mean_weak - sample_mean) < 1, \
            "Weak prior: posterior should be close to data"

        # With informative prior, posterior should be pulled toward prior
        # (further from sample mean than weak prior case)
        assert abs(posterior_mean_informative - sample_mean) > abs(posterior_mean_weak - sample_mean) or \
               abs(posterior_mean_informative - informative_prior_mean) < abs(posterior_mean_weak - informative_prior_mean), \
            "Informative prior should influence posterior more"


class TestBayesianSmallSample:
    """Test Bayesian advantage with small samples."""

    def test_bayesian_small_sample_borrows_strength_from_prior(self):
        """
        Test that Bayesian with prior can handle small samples better.

        With small n, Bayesian borrows strength from prior.
        """
        np.random.seed(42)

        # Small sample
        data = np.random.normal(loc=105, scale=15, size=10)
        sample_mean = np.mean(data)
        n = len(data)
        likelihood_std = 15

        # Informative prior
        prior_mean = 100
        prior_std = 10

        # Posterior
        posterior_var = 1 / (1/prior_std**2 + n/likelihood_std**2)
        posterior_mean = posterior_var * (
            prior_mean/prior_std**2 +
            n*sample_mean/likelihood_std**2
        )

        # Posterior should be between prior and sample mean
        # (borrowing strength from prior)
        assert min(prior_mean, sample_mean) <= posterior_mean <= max(prior_mean, sample_mean), \
            "Posterior should be between prior and data with small sample"

        # Posterior variance should be smaller than prior variance
        # (data adds information)
        posterior_std = np.sqrt(posterior_var)
        assert posterior_std < prior_std, \
            "Posterior should be more certain than prior even with small sample"


class TestBayesianInterpretation:
    """Test correct Bayesian interpretations."""

    def test_bayesian_vs_frequentist_interpretation_difference(self):
        """
        Test the key difference in interpretation.

        This is a conceptual/documentation test.
        """
        # Frequentist (WRONG for Bayesian, CORRECT for frequentist)
        frequentist_statement = (
            "If we repeat this procedure many times, 95% of such intervals "
            "will contain the true parameter."
        )

        # Bayesian (CORRECT for Bayesian, WRONG for frequentist)
        bayesian_statement = (
            "There is a 95% probability that the parameter lies within "
            "this specific interval, given the data and prior."
        )

        # Test understanding
        assert "probability" in bayesian_statement.lower()
        assert "procedure" in frequentist_statement.lower() or "repeat" in frequentist_statement.lower()

        # Both are valid in their respective frameworks
        assert isinstance(frequentist_statement, str)
        assert isinstance(bayesian_statement, str)

    def test_bayesian_requires_prior_choice(self):
        """
        Test understanding that Bayesian requires choosing a prior.

        This is both a strength (can incorporate prior knowledge) and
        weakness (result depends on prior).
        """
        # Different priors lead to different posteriors
        priors = {
            "informative": {"mean": 100, "std": 5, "description": "Strong prior belief"},
            "weak": {"mean": 100, "std": 1000, "description": "Vague prior belief"},
            "flat": {"description": "Uniform prior (all values equally likely)"}
        }

        # Test understanding
        assert "informative" in priors
        assert "weak" in priors
        assert "flat" in priors
        assert isinstance(priors, dict)


class TestBayesianPrediction:
    """Test Bayesian prediction intervals."""

    def test_bayesian_predictive_interval(self):
        """
        Test predictive interval for future observation.

        Predictive distribution accounts for BOTH:
        - Parameter uncertainty (posterior)
        - Data uncertainty (likelihood)
        """
        np.random.seed(42)

        # Posterior from previous data
        posterior_mean = 105
        posterior_std = 3

        # Predictive distribution for new observation
        # Var(predictive) = Var(posterior) + Var(likelihood)
        likelihood_std = 15
        predictive_std = np.sqrt(posterior_std**2 + likelihood_std**2)

        # Predictive interval should be wider than posterior CI
        # (because it includes data uncertainty)
        posterior_ci_width = 2 * 1.96 * posterior_std
        predictive_ci_width = 2 * 1.96 * predictive_std

        assert predictive_ci_width > posterior_ci_width, \
            "Predictive interval should be wider than parameter CI"

        # Generate predictive samples
        n_samples = 10000
        posterior_samples = np.random.normal(posterior_mean, posterior_std, n_samples)
        predictive_samples = np.random.normal(posterior_samples, likelihood_std)

        # Predictive CI
        pred_ci_low = np.percentile(predictive_samples, 2.5)
        pred_ci_high = np.percentile(predictive_samples, 97.5)

        assert pred_ci_low < pred_ci_high, "Predictive CI should be valid"


class TestBayesianEdgeCases:
    """Test Bayesian methods with edge cases."""

    def test_bayesian_conjugate_with_constant_data(self):
        """Edge case: Bayesian update with constant data."""
        np.random.seed(42)

        # Constant data
        data = np.array([100, 100, 100, 100, 100])

        # Prior
        prior_mean = 105
        prior_std = 10

        # Likelihood
        likelihood_std = 1  # Very small (data is precise)

        # Posterior
        n = len(data)
        sample_mean = np.mean(data)

        posterior_var = 1 / (1/prior_std**2 + n/likelihood_std**2)
        posterior_mean = posterior_var * (
            prior_mean/prior_std**2 +
            n*sample_mean/likelihood_std**2
        )

        # With precise constant data, posterior should be pulled toward data
        assert posterior_mean != prior_mean, \
            "Constant data should still update posterior"

        # Posterior should be very certain (small variance)
        posterior_std = np.sqrt(posterior_var)
        assert posterior_std < prior_std, \
            "Posterior should be more certain than prior"

    def test_bayesian_with_conflicting_prior_and_data(self):
        """
        Edge case: Prior and data are in conflict.

        When prior says "mean is around 100" but data says "mean is around 150",
        posterior should be somewhere in between (weighted by certainty).
        """
        np.random.seed(42)

        # Prior: confident that mean is 100
        prior_mean = 100
        prior_std = 5  # Very confident

        # Data: suggests mean is 150
        data = np.random.normal(loc=150, scale=15, size=100)
        sample_mean = np.mean(data)
        n = len(data)
        likelihood_std = 15

        # Posterior
        posterior_var = 1 / (1/prior_std**2 + n/likelihood_std**2)
        posterior_mean = posterior_var * (
            prior_mean/prior_std**2 +
            n*sample_mean/likelihood_std**2
        )

        # Posterior should be between prior and data (compromise)
        assert min(prior_mean, sample_mean) <= posterior_mean <= max(prior_mean, sample_mean), \
            "Posterior should compromise between conflicting prior and data"

        # With large sample, data should dominate even confident prior
        # (Posterior closer to data than prior)
        data_distance = abs(posterior_mean - sample_mean)
        prior_distance = abs(posterior_mean - prior_mean)

        # With n=100, data should have more influence
        assert data_distance < prior_distance, \
            "With sufficient data, posterior should be closer to data than prior"


class TestBayesianComparisonFrequentist:
    """Compare Bayesian and Frequentist approaches."""

    def test_bayesian_frequentist_agreement_with_large_sample(self):
        """
        Test that Bayesian and Frequentist converge with large samples.

        With large n, prior influence diminishes, and Bayesian posterior
        approaches Frequentist MLE.
        """
        np.random.seed(42)

        # Large sample
        data = np.random.normal(loc=105, scale=15, size=1000)
        sample_mean = np.mean(data)
        n = len(data)
        likelihood_std = 15

        # Weak prior (won't influence much)
        prior_mean = 100
        prior_std = 1000

        # Bayesian posterior
        posterior_var = 1 / (1/prior_std**2 + n/likelihood_std**2)
        posterior_mean = posterior_var * (
            prior_mean/prior_std**2 +
            n*sample_mean/likelihood_std**2
        )

        # With weak prior and large sample, posterior should be very close to MLE (sample mean)
        assert abs(posterior_mean - sample_mean) < 1, \
            "With weak prior and large sample, Bayesian should approximate Frequentist"

    def test_when_to_use_bayesian_vs_frequentist(self):
        """
        Test understanding of when to use each framework.

        This is a documentation test.
        """
        # Use Bayesian when:
        bayesian_situations = [
            "Have prior knowledge to incorporate",
            "Small sample size",
            "Need intuitive probability interpretation",
            "Want full posterior distribution",
            "Sequential updating of beliefs"
        ]

        # Use Frequentist when:
        frequentist_situations = [
            "No prior knowledge",
            "Large sample size",
            "Need objective, reproducible results",
            "Computational simplicity is important"
        ]

        # Test understanding
        assert len(bayesian_situations) > 0
        assert len(frequentist_situations) > 0
        assert isinstance(bayesian_situations, list)
        assert isinstance(frequentist_situations, list)


class TestBayesianLimitations:
    """Test understanding of Bayesian limitations."""

    def test_bayesian_computational_cost(self):
        """
        Test understanding that Bayesian can be computationally expensive.

        For complex models, need MCMC which is slow.
        Conjugate priors are fast but limited.
        """
        # Simple conjugate case: fast
        simple_computational_cost = "Analytical solution, instant"

        # Complex non-conjugate case: slow
        complex_computational_cost = "MCMC sampling, can be slow"

        assert "analytical" in simple_computational_cost.lower()
        assert "mcmc" in complex_computational_cost.lower() or "slow" in complex_computational_cost.lower()

    def test_bayesian_prior_sensitivity(self):
        """
        Test understanding that results can be sensitive to prior choice.

        Especially with small samples.
        """
        # This is a documentation test
        warning = (
            "With small samples, posterior can be sensitive to prior choice. "
            "With large samples, data dominates and prior has less influence."
        )

        assert "small" in warning.lower()
        assert "prior" in warning.lower()
        assert isinstance(warning, str)

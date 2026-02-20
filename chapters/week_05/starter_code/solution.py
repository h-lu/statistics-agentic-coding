"""
Week 05 Starter Code - Simulation and Sampling Distribution

Student implementation template. Students need to complete the following functions:

1. Simulation Functions
   - simulate_coin_flips(n_flips, p_heads=0.5, random_state=None)
   - simulate_binomial(n_trials, p_success, n_simulations=1000, random_state=None)

2. Sampling Distribution Functions
   - calculate_sample_mean(data)
   - calculate_standard_error(data)
   - generate_sampling_distribution(population, sample_size, n_simulations=1000, random_state=None)

3. Bootstrap Functions
   - bootstrap_resample(data, n_bootstrap=1000, random_state=None)
   - bootstrap_confidence_interval(data, statistic_func, n_bootstrap=1000, confidence_level=0.95, random_state=None)
   - bootstrap_standard_error(data, statistic_func, n_bootstrap=1000, random_state=None)

4. False Positive Simulation
   - simulate_false_positive_rate(n_hypotheses, alpha=0.05, n_simulations=1000, random_state=None)
   - calculate_multiple_comparison_risk(n_hypotheses, alpha=0.05)
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np


def simulate_coin_flips(n_flips: int, p_heads: float = 0.5, random_state: int | None = None) -> np.ndarray:
    """
    Simulate coin flips.

    Args:
        n_flips: Number of coin flips to simulate
        p_heads: Probability of heads (default 0.5 for fair coin)
        random_state: Random seed for reproducibility

    Returns:
        Array of 1s (heads) and 0s (tails) representing the flips

    Notes:
        - Returns 1 for heads, 0 for tails
        - Uses np.random.binomial for simulation
    """
    # TODO: Student implementation
    if random_state is not None:
        np.random.seed(random_state)
    if n_flips <= 0:
        raise ValueError("n_flips must be positive")
    if not 0 <= p_heads <= 1:
        raise ValueError("p_heads must be between 0 and 1")
    return np.random.binomial(n=1, p=p_heads, size=n_flips)


def simulate_binomial(
    n_trials: int,
    p_success: float,
    n_simulations: int = 1000,
    random_state: int | None = None
) -> np.ndarray:
    """
    Simulate binomial experiment multiple times.

    Args:
        n_trials: Number of trials per simulation
        p_success: Probability of success in each trial
        n_simulations: Number of simulations to run
        random_state: Random seed for reproducibility

    Returns:
        Array of success counts for each simulation

    Notes:
        - Each simulation represents n_trials independent Bernoulli trials
        - Returns the count of successes for each simulation
    """
    # TODO: Student implementation
    if random_state is not None:
        np.random.seed(random_state)
    if n_trials <= 0:
        raise ValueError("n_trials must be positive")
    if n_simulations <= 0:
        raise ValueError("n_simulations must be positive")
    if not 0 <= p_success <= 1:
        raise ValueError("p_success must be between 0 and 1")
    return np.random.binomial(n=n_trials, p=p_success, size=n_simulations)


def calculate_sample_mean(data: np.ndarray) -> float:
    """
    Calculate the sample mean.

    Args:
        data: Input data array

    Returns:
        Sample mean

    Notes:
        - Returns NaN for empty array
    """
    # TODO: Student implementation
    if len(data) == 0:
        return np.nan
    return float(np.mean(data))


def calculate_standard_error(data: np.ndarray) -> float:
    """
    Calculate the standard error of the mean.

    Args:
        data: Input data array

    Returns:
        Standard error of the mean (std / sqrt(n))

    Notes:
        - SE = s / sqrt(n) (s is sample standard deviation)
        - Returns NaN for empty array or single element
    """
    # TODO: Student implementation
    if len(data) <= 1:
        return np.nan
    return float(np.std(data, ddof=1) / np.sqrt(len(data)))


def generate_sampling_distribution(
    population: np.ndarray,
    sample_size: int,
    n_simulations: int = 1000,
    random_state: int | None = None
) -> np.ndarray:
    """
    Generate sampling distribution by repeated sampling from population.

    Args:
        population: Population data to sample from
        sample_size: Size of each sample
        n_simulations: Number of samples to draw
        random_state: Random seed for reproducibility

    Returns:
        Array of sample means from each simulation

    Notes:
        - Samples without replacement from population
        - Assumes sample_size <= len(population)
        - 注意：此函数使用无放回抽样（replace=False），适用于从有限总体抽样的场景。如果模拟从理论分布抽样，应使用 replace=True。
    """
    # TODO: Student implementation
    if random_state is not None:
        np.random.seed(random_state)
    if len(population) == 0:
        raise ValueError("population cannot be empty")
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if n_simulations <= 0:
        raise ValueError("n_simulations must be positive")
    if sample_size > len(population):
        raise ValueError("sample_size cannot exceed population size")

    sample_means = []
    for _ in range(n_simulations):
        sample = np.random.choice(population, size=sample_size, replace=False)
        sample_means.append(np.mean(sample))
    return np.array(sample_means)


def bootstrap_resample(
    data: np.ndarray,
    n_bootstrap: int = 1000,
    random_state: int | None = None
) -> np.ndarray:
    """
    Perform bootstrap resampling.

    Args:
        data: Original sample data
        n_bootstrap: Number of bootstrap samples to generate
        random_state: Random seed for reproducibility

    Returns:
        Array of bootstrap sample means

    Notes:
        - Each bootstrap sample is drawn WITH replacement from original data
        - Bootstrap sample size equals original data size
    """
    # TODO: Student implementation
    if random_state is not None:
        np.random.seed(random_state)
    if len(data) == 0:
        raise ValueError("data cannot be empty")
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")

    n = len(data)
    bootstrap_means = []
    for _ in range(n_bootstrap):
        boot_sample = np.random.choice(data, size=n, replace=True)
        bootstrap_means.append(np.mean(boot_sample))
    return np.array(bootstrap_means)


def bootstrap_confidence_interval(
    data: np.ndarray,
    statistic_func: Callable[[np.ndarray], float],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int | None = None
) -> tuple[float, float]:
    """
    Calculate bootstrap confidence interval for a statistic.

    Args:
        data: Original sample data
        statistic_func: Function to calculate statistic (e.g., np.mean, np.median)
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (lower_bound, upper_bound) for the confidence interval

    Notes:
        - Uses percentile method for CI calculation
    """
    # TODO: Student implementation
    if random_state is not None:
        np.random.seed(random_state)
    if len(data) == 0:
        raise ValueError("data cannot be empty")
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")

    n = len(data)
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        boot_sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats.append(statistic_func(boot_sample))

    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    ci_lower = float(np.percentile(bootstrap_stats, lower_percentile))
    ci_upper = float(np.percentile(bootstrap_stats, upper_percentile))

    return (ci_lower, ci_upper)


def bootstrap_standard_error(
    data: np.ndarray,
    statistic_func: Callable[[np.ndarray], float],
    n_bootstrap: int = 1000,
    random_state: int | None = None
) -> float:
    """
    Calculate bootstrap standard error for a statistic.

    Args:
        data: Original sample data
        statistic_func: Function to calculate statistic (e.g., np.mean, np.median)
        n_bootstrap: Number of bootstrap samples
        random_state: Random seed for reproducibility

    Returns:
        Standard error of the bootstrap statistic distribution

    Notes:
        - SE = std(bootstrap_statistics)
    """
    # TODO: Student implementation
    if random_state is not None:
        np.random.seed(random_state)
    if len(data) == 0:
        raise ValueError("data cannot be empty")
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")

    n = len(data)
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        boot_sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats.append(statistic_func(boot_sample))

    return float(np.std(bootstrap_stats, ddof=1))


def simulate_false_positive_rate(
    n_hypotheses: int,
    alpha: float = 0.05,
    n_simulations: int = 1000,
    random_state: int | None = None
) -> float:
    """
    Simulate false positive rate when testing multiple hypotheses.

    Args:
        n_hypotheses: Number of hypotheses being tested
        alpha: Significance level for each test
        n_simulations: Number of simulations to run
        random_state: Random seed for reproducibility

    Returns:
        Proportion of simulations with at least one false positive

    Notes:
        - Assumes null hypothesis is true for all tests
        - False positive = p < alpha when null is true
    """
    # TODO: Student implementation
    if random_state is not None:
        np.random.seed(random_state)
    if n_hypotheses <= 0:
        raise ValueError("n_hypotheses must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    if n_simulations <= 0:
        raise ValueError("n_simulations must be positive")

    false_positive_count = 0
    for _ in range(n_simulations):
        # Generate p-values under null hypothesis (uniform distribution)
        p_values = np.random.uniform(0, 1, size=n_hypotheses)
        # Check if any p-value is below alpha (false positive)
        if np.any(p_values < alpha):
            false_positive_count += 1

    return false_positive_count / n_simulations


def calculate_multiple_comparison_risk(n_hypotheses: int, alpha: float = 0.05) -> float:
    """
    Calculate theoretical risk of at least one false positive in multiple comparisons.

    Args:
        n_hypotheses: Number of independent hypotheses being tested
        alpha: Significance level for each test

    Returns:
        Probability of at least one false positive

    Notes:
        - P(at least one FP) = 1 - (1 - alpha)^n
        - Assumes independent tests
    """
    # TODO: Student implementation
    if n_hypotheses <= 0:
        raise ValueError("n_hypotheses must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")

    return 1 - (1 - alpha) ** n_hypotheses


# =============================================================================
# Reference Solutions and Examples
# =============================================================================

def example_coin_flip_simulation():
    """
    Example: Coin flip simulation demonstrating long-run frequency.

    This demonstrates:
    1. Single experiment (10 flips) - high variability
    2. Large number of flips (1000) - frequency approaches probability
    3. Repeated experiments (1000 x 100 flips) - sampling distribution

    Key insight: The standard deviation of the sampling distribution (~5 for 100 flips)
    quantifies "how much randomness to expect".
    """
    print("\n" + "=" * 60)
    print("Example: Coin Flip Simulation")
    print("=" * 60)

    rng = np.random.default_rng(seed=42)

    # Experiment 1: Flip 10 times
    flips_10 = rng.choice(["H", "T"], size=10)
    heads_10 = (flips_10 == "H").sum()
    print(f"\nFlip 10 times: {heads_10} heads, {10 - heads_10} tails")
    print(f"Frequency of heads: {heads_10 / 10:.1%}")

    # Experiment 2: Flip 1000 times
    flips_1000 = rng.choice(["H", "T"], size=1000)
    heads_1000 = (flips_1000 == "H").sum()
    print(f"\nFlip 1000 times: {heads_1000} heads, {1000 - heads_1000} tails")
    print(f"Frequency of heads: {heads_1000 / 1000:.1%}")
    print("Notice: Frequency is closer to 50% with more flips (Law of Large Numbers)")

    # Experiment 3: Repeat 1000 times (each time flip 100 coins)
    n_simulations = 1000
    n_flips_per_sim = 100
    heads_counts = []

    for _ in range(n_simulations):
        flips = rng.choice([0, 1], size=n_flips_per_sim)
        heads_counts.append(flips.sum())

    heads_counts = np.array(heads_counts)

    print(f"\nRepeat {n_simulations} times (each time flip {n_flips_per_sim} coins):")
    print(f"  Mean: {heads_counts.mean():.1f}")
    print(f"  Std Dev: {heads_counts.std():.1f}")
    print(f"  Range: [{heads_counts.min()}, {heads_counts.max()}]")
    print(f"  % in [40, 60]: {((heads_counts >= 40) & (heads_counts <= 60)).mean():.1%}")


def example_sampling_distribution():
    """
    Example: Sampling distribution when true difference is zero.

    This demonstrates:
    1. When true conversion rate is 10% for both groups
    2. Repeated sampling produces different "false differences"
    3. The sampling distribution centers at 0 with spread = standard error

    Key insight: Your observed difference (e.g., 3%) can be positioned
    on this sampling distribution to estimate its "unusualness".
    """
    print("\n" + "=" * 60)
    print("Example: Sampling Distribution (Null Scenario)")
    print("=" * 60)

    rng = np.random.default_rng(seed=42)

    # Both groups have true rate of 10% (no difference)
    true_rate = 0.10
    sample_size = 1000
    n_simulations = 1000

    print(f"\nSetup:")
    print(f"  True rate (both groups): {true_rate:.0%}")
    print(f"  Sample size per group: {sample_size}")
    print(f"  True difference: {true_rate - true_rate:.0%}")

    differences = []
    for _ in range(n_simulations):
        sample_a = rng.binomial(n=1, p=true_rate, size=sample_size)
        sample_b = rng.binomial(n=1, p=true_rate, size=sample_size)
        differences.append(sample_a.mean() - sample_b.mean())

    differences = np.array(differences)

    std_error = differences.std()
    prob_ge_3pc = (differences >= 0.03).mean()

    print(f"\nResults from {n_simulations} simulations:")
    print(f"  Standard Error: {std_error:.2%}")
    print(f"  P(difference >= 3%): {prob_ge_3pc:.2%}")

    print(f"\nInterpretation:")
    print(f"  If you observed a 3% difference, it would occur by chance")
    print(f"  approximately {prob_ge_3pc:.1%} of the time when true diff = 0%")
    print(f"  This is the intuition behind p-values!")


def example_bootstrap():
    """
    Example: Bootstrap method for uncertainty quantification.

    This demonstrates:
    1. Single sample with observed difference (12% vs 9% = 3%)
    2. Bootstrap resampling to estimate sampling distribution
    3. Standard error and 95% confidence interval

    Key insight: Bootstrap uses a single sample to estimate uncertainty
    without assuming a specific distribution.
    """
    print("\n" + "=" * 60)
    print("Example: Bootstrap Uncertainty Quantification")
    print("=" * 60)

    rng = np.random.default_rng(seed=42)

    # Observed data: A = 12%, B = 9%, difference = 3%
    conversions_a = np.array([1] * 120 + [0] * 880)
    conversions_b = np.array([1] * 90 + [0] * 910)

    observed_diff = conversions_a.mean() - conversions_b.mean()

    print(f"\nObserved data:")
    print(f"  Group A: {conversions_a.sum()}/{len(conversions_a)} ({conversions_a.mean():.1%})")
    print(f"  Group B: {conversions_b.sum()}/{len(conversions_b)} ({conversions_b.mean():.1%})")
    print(f"  Difference: {observed_diff:.2%}")

    # Bootstrap
    n_bootstrap = 1000
    n_a, n_b = len(conversions_a), len(conversions_b)
    bootstrap_diffs = []

    for _ in range(n_bootstrap):
        boot_a = rng.choice(conversions_a, size=n_a, replace=True)
        boot_b = rng.choice(conversions_b, size=n_b, replace=True)
        bootstrap_diffs.append(boot_a.mean() - boot_b.mean())

    bootstrap_diffs = np.array(bootstrap_diffs)

    se = bootstrap_diffs.std()
    ci_low, ci_high = np.percentile(bootstrap_diffs, [2.5, 97.5])

    print(f"\nBootstrap results ({n_bootstrap} resamples):")
    print(f"  Standard Error: {se:.2%}")
    print(f"  95% CI: [{ci_low:.2%}, {ci_high:.2%}]")

    if ci_low > 0:
        print(f"  Conclusion: CI does not include 0, suggests A > B")
    elif ci_high < 0:
        print(f"  Conclusion: CI does not include 0, suggests B > A")
    else:
        print(f"  Conclusion: CI includes 0, difference may not be significant")


def example_false_positive():
    """
    Example: False positive rate in multiple testing.

    This demonstrates:
    1. Single test: 5% false positive rate (alpha = 0.05)
    2. Multiple tests: much higher probability of at least one false positive
    3. Formula: P(at least one FP) = 1 - (1 - alpha)^k

    Key insight: Testing many hypotheses increases the chance of
    finding at least one "significant" result by pure luck.
    """
    print("\n" + "=" * 60)
    print("Example: False Positive Rate in Multiple Testing")
    print("=" * 60)

    alpha = 0.05
    n_simulations = 1000
    rng = np.random.default_rng(seed=42)

    for n_hypotheses in [1, 5, 10]:
        # Simulate
        fp_count = 0
        for _ in range(n_simulations):
            p_values = rng.uniform(0, 1, size=n_hypotheses)
            if (p_values < alpha).any():
                fp_count += 1

        simulated_rate = fp_count / n_simulations
        theoretical_rate = 1 - (1 - alpha) ** n_hypotheses

        print(f"\nTest {n_hypotheses} hypotheses:")
        print(f"  Simulated FP rate: {simulated_rate:.2%}")
        print(f"  Theoretical FP rate: {theoretical_rate:.2%}")
        print(f"  Formula: 1 - (1 - {alpha})^{n_hypotheses} = {theoretical_rate:.2%}")

    print(f"\nInterpretation:")
    print(f"  If you test 5 hypotheses with alpha = 0.05:")
    print(f"  - Each test has 5% false positive rate")
    print(f"  - But there's ~23% chance of at least one false positive!")
    print(f"  - This is why multiple comparison correction is needed")


# =============================================================================
# Run Examples
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Week 05: Reference Solutions and Examples")
    print("=" * 60)

    example_coin_flip_simulation()
    example_sampling_distribution()
    example_bootstrap()
    example_false_positive()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("1. Simulation builds intuition for randomness")
    print("2. Sampling distribution describes statistic variability")
    print("3. Bootstrap estimates uncertainty from a single sample")
    print("4. Multiple testing increases false positive risk")
    print("\nNext week: Hypothesis testing (p-values, significance, etc.)")

"""Pytest configuration for week_05 tests."""
import pytest
import numpy as np


@pytest.fixture
def rng():
    """Fixture providing a seeded random number generator."""
    return np.random.default_rng(seed=42)


@pytest.fixture
def sample_data():
    """Fixture providing sample data for testing."""
    return np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


@pytest.fixture
def conversion_data():
    """Fixture providing conversion data for testing."""
    # A: 12% conversion, B: 9% conversion
    a = np.array([1] * 12 + [0] * 88)
    b = np.array([1] * 9 + [0] * 91)
    return a, b

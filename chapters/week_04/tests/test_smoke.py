"""Smoke tests for Week 04 - EDA 叙事与假设清单

These tests verify that basic infrastructure is in place and functions
can be imported and called with minimal inputs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


def test_can_import_pandas():
    """Test that pandas can be imported."""
    import pandas as pd
    assert pd is not None


def test_can_import_numpy():
    """Test that numpy can be imported."""
    import numpy as np
    assert np is not None


def test_can_import_seaborn():
    """Test that seaborn can be imported."""
    if not HAS_SEABORN:
        pytest.skip("seaborn not available")
    import seaborn as sns
    assert sns is not None


def test_can_load_penguins_dataset():
    """Test that Palmer Penguins dataset can be loaded from seaborn."""
    if not HAS_SEABORN:
        pytest.skip("seaborn not available")

    penguins = sns.load_dataset("penguins")
    assert isinstance(penguins, pd.DataFrame)
    assert len(penguins) > 0
    assert len(penguins.columns) > 0


def test_can_calculate_correlation():
    """Test basic correlation calculation."""
    np.random.seed(42)
    df = pd.DataFrame({
        'x': [1, 2, 3, 4, 5],
        'y': [2, 4, 6, 8, 10]
    })

    corr = df['x'].corr(df['y'])
    assert corr > 0.9  # Should be highly correlated


def test_can_use_groupby():
    """Test basic groupby functionality."""
    df = pd.DataFrame({
        'group': ['A', 'A', 'B', 'B'],
        'value': [1, 2, 3, 4]
    })

    grouped = df.groupby('group')['value'].mean()
    assert len(grouped) == 2
    assert grouped['A'] == 1.5
    assert grouped['B'] == 3.5


def test_can_create_pivot_table():
    """Test basic pivot table creation."""
    df = pd.DataFrame({
        'row': ['A', 'A', 'B', 'B'],
        'col': ['X', 'Y', 'X', 'Y'],
        'value': [1, 2, 3, 4]
    })

    pivot = df.pivot_table(values='value', index='row', columns='col', aggfunc='mean')
    assert pivot.shape[0] == 2  # 2 rows
    assert pivot.shape[1] == 2  # 2 columns


def test_can_compare_correlation_methods():
    """Test comparing different correlation methods."""
    np.random.seed(42)
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 6, 8, 10])

    # Pearson
    df_pearson = pd.DataFrame({'x': x, 'y': y})
    pearson = df_pearson.corr(method='pearson').loc['x', 'y']

    # Spearman
    spearman = df_pearson.corr(method='spearman').loc['x', 'y']

    # Kendall
    kendall = df_pearson.corr(method='kendall').loc['x', 'y']

    # All should be high for linear relationship
    assert pearson > 0.9
    assert spearman > 0.9
    assert kendall > 0.8


def test_time_series_index():
    """Test time series date range creation."""
    dates = pd.date_range("2025-01-01", "2025-01-31", freq="D")
    assert len(dates) == 31
    assert dates[0].day == 1
    assert dates[-1].day == 31


def test_dataframe_corr_matrix():
    """Test correlation matrix for DataFrame."""
    np.random.seed(42)
    df = pd.DataFrame({
        'a': np.random.randn(100),
        'b': np.random.randn(100),
        'c': np.random.randn(100)
    })

    corr_matrix = df.corr()
    assert corr_matrix.shape == (3, 3)
    assert np.allclose(np.diag(corr_matrix.values), 1.0)  # Diagonals should be 1


def test_groupby_multiple_aggregations():
    """Test groupby with multiple aggregation functions."""
    df = pd.DataFrame({
        'group': ['A', 'A', 'B', 'B', 'A', 'B'],
        'value': [1, 2, 3, 4, 5, 6]
    })

    result = df.groupby('group')['value'].agg(['mean', 'median', 'std', 'count'])

    assert 'mean' in result.columns
    assert 'median' in result.columns
    assert 'std' in result.columns
    assert 'count' in result.columns
    assert len(result) == 2


def test_pivot_with_missing_values():
    """Test pivot table handling of missing combinations."""
    df = pd.DataFrame({
        'row': ['A', 'A', 'B'],
        'col': ['X', 'Y', 'X'],
        'value': [1, 2, 3]
    })

    pivot = df.pivot_table(values='value', index='row', columns='col', aggfunc='mean')

    # B, Y combination is missing
    assert pd.isna(pivot.loc['B', 'Y'])


def test_can_create_dataframe_from_dict():
    """Test creating DataFrame from dictionary of lists."""
    data = {
        'observation': ['obs1', 'obs2'],
        'explanation': ['exp1', 'exp2'],
        'test_method': ['test1', 'test2'],
        'priority': ['high', 'medium']
    }

    df = pd.DataFrame(data)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert 'observation' in df.columns
    assert 'priority' in df.columns


def test_penguins_groupby_works():
    """Test groupby works on Penguins dataset."""
    if not HAS_SEABORN:
        pytest.skip("seaborn not available")

    penguins = sns.load_dataset("penguins")

    grouped = penguins.groupby("species")["body_mass_g"].mean()

    assert len(grouped) == 3  # Three species
    assert all(grouped > 0)  # All means should be positive
    assert grouped["Gentoo"] > grouped["Adelie"]  # Gentoo is heaviest


def test_correlation_with_outliers():
    """Test correlation calculation with potential outliers."""
    np.random.seed(42)
    # Base data with strong positive correlation
    x = np.random.normal(50, 10, 50)
    y = x * 0.8 + np.random.normal(0, 5, 50)

    # Add an outlier
    x_with_outlier = np.append(x, 100)
    y_with_outlier = np.append(y, 0)

    df = pd.DataFrame({'x': x_with_outlier, 'y': y_with_outlier})

    # Calculate correlation
    corr = df['x'].corr(df['y'])

    # Outlier should reduce correlation
    assert isinstance(corr, float)
    assert -1 <= corr <= 1


def test_empty_dataframe_groupby():
    """Test groupby on empty DataFrame."""
    df = pd.DataFrame({'group': [], 'value': []})

    result = df.groupby('group')['value'].mean()

    assert len(result) == 0


def test_single_value_correlation():
    """Test correlation with single value returns NaN."""
    df = pd.DataFrame({'x': [1], 'y': [2]})

    corr = df['x'].corr(df['y'])

    # Single value correlation is undefined
    assert pd.isna(corr)

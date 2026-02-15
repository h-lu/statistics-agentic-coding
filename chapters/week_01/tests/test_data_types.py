"""Tests for data type identification and validation."""

from __future__ import annotations

import pandas as pd
import pytest


def test_identify_numeric_continuous():
    """Test identification of continuous numeric data."""
    # Continuous: can take any value within a range
    continuous_examples = [
        (pd.Series([1.5, 2.3, 3.7, 4.2]), True),
        (pd.Series([39.1, 46.5, 48.1]), True),
        (pd.Series([180.5, 192.3, 210.7]), True),
    ]
    # TODO: Implement after solution.py has is_continuous function
    # for series, expected in continuous_examples:
    #     result = is_continuous(series)
    #     assert result is expected


def test_identify_numeric_discrete():
    """Test identification of discrete numeric data."""
    # Discrete: can only take integer values
    discrete_examples = [
        (pd.Series([1, 2, 3, 4, 5]), True),
        (pd.Series([0, 1, 2, 3]), True),
        (pd.Series([10, 20, 30]), True),
    ]
    # TODO: Implement after solution.py has is_discrete function
    # for series, expected in discrete_examples:
    #     result = is_discrete(series)
    #     assert result is expected


def test_identify_categorical_nominal():
    """Test identification of nominal categorical data."""
    # Nominal: no inherent order
    nominal_examples = [
        (pd.Series(['Adelie', 'Chinstrap', 'Gentoo']), True),
        (pd.Series(['Biscoe', 'Dream', 'Torgersen']), True),
        (pd.Series(['male', 'female']), True),
    ]
    # TODO: Implement after solution.py has is_nominal function
    # for series, expected in nominal_examples:
    #     result = is_nominal(series)
    #     assert result is expected


def test_identify_categorical_ordinal():
    """Test identification of ordinal categorical data."""
    # Ordinal: has order but unequal intervals
    ordinal_examples = [
        (pd.Series(['low', 'medium', 'high']), True),
        (pd.Series(['small', 'medium', 'large', 'xlarge']), True),
        (pd.Series([1, 2, 3, 4, 5]), True),  # Could be ordinal if representing ratings
    ]
    # TODO: Implement after solution.py has is_ordinal function
    # for series, expected in ordinal_examples:
    #     result = is_ordinal(series)
    #     assert result is expected


def test_classify_column_type():
    """Test comprehensive column type classification."""
    test_cases = [
        # (series, expected_type)
        (pd.Series([1.5, 2.3, 3.7]), "numeric_continuous"),
        (pd.Series([1, 2, 3, 4]), "numeric_discrete"),
        (pd.Series(['a', 'b', 'c']), "categorical_nominal"),
        (pd.Series(['low', 'medium', 'high']), "categorical_ordinal"),
    ]
    # TODO: Implement after solution.py has classify_column function
    # for series, expected in test_cases:
    #     result = classify_column(series)
    #     assert result == expected


def test_zipcode_treated_as_categorical():
    """Test that zipcode is correctly identified as categorical, not numeric."""
    zipcodes = pd.Series([90210, 10001, 60601, 33101])
    # Zipcodes look numeric but are categorical
    # TODO: Implement after solution.py has classify_column function
    # result = classify_column(zipcodes)
    # assert result == "categorical_nominal"
    # assert "numeric" not in result


def test_mixed_type_column_detection():
    """Test detection of columns with mixed types."""
    mixed_series = pd.Series(['1', '2', 3, 4.5, 'text'])
    # TODO: Implement after solution.py has detect_mixed_types function
    # result = detect_mixed_types(mixed_series)
    # assert result is True


def test_edge_case_empty_series():
    """Test type identification with empty series."""
    empty_series = pd.Series([], dtype=float)
    # TODO: Implement after solution.py has classify_column function
    # with pytest.raises(ValueError, match="empty"):
    #     classify_column(empty_series)


def test_edge_case_all_na_series():
    """Test type identification with all NA series."""
    na_series = pd.Series([None, None, None])
    # TODO: Implement after solution.py has classify_column function
    # with pytest.raises(ValueError, match="all.*NA"):
    #     classify_column(na_series)


def test_edge_case_single_unique_value():
    """Test type identification with constant series."""
    constant_series = pd.Series([5, 5, 5, 5])
    # TODO: Implement after solution.py has classify_column function
    # result = classify_column(constant_series)
    # # Should identify as numeric but warn about low variance
    # assert "numeric" in result


def test_incorrect_type_usage_error():
    """Test that treating categorical as numeric produces warning."""
    species = pd.Series(['Adelie', 'Chinstrap', 'Gentoo'])
    # TODO: Implement after solution.py has validate_numeric_operation function
    # with pytest.warns(UserWarning, match="categorical.*numeric"):
    #     validate_numeric_operation(species, "mean")

"""Smoke tests for Week 01 - Basic functionality checks.

These tests verify that the basic infrastructure is in place and functions
can be imported and called with minimal inputs.
"""

from __future__ import annotations

import pytest
import pandas as pd
import seaborn as sns
from pathlib import Path


def test_can_import_pandas():
    """Test that pandas can be imported."""
    import pandas as pd
    assert pd is not None


def test_can_import_seaborn():
    """Test that seaborn can be imported."""
    import seaborn as sns
    assert sns is not None


def test_can_load_penguins_dataset():
    """Test that Palmer Penguins dataset can be loaded from seaborn."""
    penguins = sns.load_dataset("penguins")
    assert isinstance(penguins, pd.DataFrame)
    assert len(penguins) > 0
    assert len(penguins.columns) > 0


def test_penguins_dataset_has_expected_columns():
    """Test that penguins dataset has expected columns."""
    penguins = sns.load_dataset("penguins")
    expected_cols = ['species', 'island', 'bill_length_mm', 'bill_depth_mm',
                     'flipper_length_mm', 'body_mass_g', 'sex']
    assert list(penguins.columns) == expected_cols


def test_penguins_dataset_shape():
    """Test that penguins dataset has expected shape."""
    penguins = sns.load_dataset("penguins")
    assert penguins.shape[0] > 300  # Should have > 300 rows
    assert penguins.shape[1] == 7   # Should have 7 columns


def test_penguins_dataset_has_missing_values():
    """Test that penguins dataset has some missing values (as expected)."""
    penguins = sns.load_dataset("penguins")
    missing_counts = penguins.isna().sum()
    assert missing_counts.sum() > 0  # Should have at least some missing values


def test_can_create_simple_dataframe():
    """Test basic DataFrame creation."""
    df = pd.DataFrame({
        'a': [1, 2, 3],
        'b': ['x', 'y', 'z']
    })
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert list(df.columns) == ['a', 'b']


def test_can_access_dataframe_dtypes():
    """Test accessing DataFrame dtypes."""
    df = pd.DataFrame({
        'int_col': [1, 2, 3],
        'float_col': [1.1, 2.2, 3.3],
        'str_col': ['a', 'b', 'c']
    })
    dtypes = df.dtypes
    assert len(dtypes) == 3
    assert 'int' in str(dtypes['int_col'])
    assert 'float' in str(dtypes['float_col'])
    # Newer pandas versions use 'str' instead of 'object' for string columns
    str_dtype = str(dtypes['str_col'])
    assert 'object' in str_dtype or 'str' in str_dtype


def test_can_calculate_missing_values():
    """Test calculating missing values."""
    df = pd.DataFrame({
        'a': [1, 2, None, 4],
        'b': ['x', None, 'z', 'w']
    })
    missing_a = df['a'].isna().sum()
    missing_b = df['b'].isna().sum()
    assert missing_a == 1
    assert missing_b == 1


def test_can_groupby_and_aggregate():
    """Test basic groupby and aggregation."""
    penguins = sns.load_dataset("penguins")
    grouped = penguins.groupby("species")["bill_length_mm"].mean()
    assert len(grouped) == 3  # Three species
    assert all(grouped > 0)  # All means should be positive


def test_can_write_markdown_file(tmp_path):
    """Test writing a simple markdown file."""
    output_path = tmp_path / "test.md"
    content = "# Test Header\n\nSome content here."
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    assert output_path.exists()
    assert output_path.read_text(encoding='utf-8') == content


def test_can_create_directory_if_not_exists(tmp_path):
    """Test creating nested directories."""
    nested_dir = tmp_path / "nested" / "dir"
    nested_dir.mkdir(parents=True, exist_ok=True)
    assert nested_dir.exists()
    assert nested_dir.is_dir()


def test_pandas_category_conversion():
    """Test converting string columns to category dtype."""
    df = pd.DataFrame({
        'species': ['Adelie', 'Chinstrap', 'Gentoo', 'Adelie']
    })
    df_converted = df.astype({'species': 'category'})
    assert df_converted['species'].dtype.name == 'category'


def test_basic_string_operations():
    """Test basic string operations for data card generation."""
    lines = ["# Header\n", "Line 1\n", "Line 2\n"]
    content = "".join(lines)
    assert content.startswith("#")
    assert "Line 1" in content
    assert len(lines) == 3


def test_f_string_formatting():
    """Test f-string formatting for report generation."""
    name = "Test Dataset"
    rows = 100
    cols = 5
    formatted = f"Dataset: {name}, Rows: {rows}, Columns: {cols}"
    assert "Test Dataset" in formatted
    assert "100" in formatted
    assert "5" in formatted


def test_markdown_table_formatting():
    """Test Markdown table formatting."""
    header = "| Col1 | Col2 | Col3 |\n"
    separator = "|------|------|------|\n"
    row1 = "| A    | B    | C    |\n"
    table = header + separator + row1

    assert "|" in table
    assert "---" in table
    assert table.count("|") >= 12  # Each row should have 4 pipes (3 cols + 1)

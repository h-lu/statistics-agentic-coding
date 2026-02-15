"""Tests for pandas basic operations."""

from __future__ import annotations

import pandas as pd
import pytest
from pathlib import Path


def test_read_csv_success(sample_csv_file):
    """Test successful CSV reading."""
    # TODO: Implement after solution.py has read_data function
    # df = read_data(sample_csv_file)
    # assert isinstance(df, pd.DataFrame)
    # assert len(df) == 3
    # assert list(df.columns) == ['a', 'b']
    pass


def test_read_csv_file_not_found():
    """Test reading non-existent file raises appropriate error."""
    nonexistent = Path("nonexistent_file.csv")
    # TODO: Implement after solution.py has read_data function
    # with pytest.raises(FileNotFoundError):
    #     read_data(nonexistent)
    pass


def test_read_csv_with_encoding_issue(tmp_path):
    """Test reading CSV with different encodings."""
    # Create a file with UTF-8 BOM or special characters
    csv_path = tmp_path / "encoding_test.csv"
    with open(csv_path, 'w', encoding='utf-8-sig') as f:
        f.write('a,b\n1,测试\n2,数据\n')

    # TODO: Implement after solution.py has read_data function with encoding parameter
    # df = read_data(csv_path, encoding='utf-8-sig')
    # assert len(df) == 2
    pass


def test_dataframe_shape(sample_dataframe):
    """Test getting DataFrame shape."""
    # TODO: Implement after solution.py has get_dataframe_info function
    # result = get_dataframe_info(sample_dataframe)
    # assert result['rows'] == 5
    # assert result['columns'] == 7
    pass


def test_dataframe_dtypes(sample_dataframe):
    """Test getting DataFrame dtypes."""
    # TODO: Implement after solution.py has get_dataframe_info function
    # result = get_dataframe_info(sample_dataframe)
    # assert 'dtypes' in result
    # assert result['dtypes']['species'] == 'object'
    # assert result['dtypes']['bill_length_mm'] == 'float64'
    pass


def test_dataframe_missing_values(sample_dataframe):
    """Test counting missing values."""
    # TODO: Implement after solution.py has get_missing_info function
    # result = get_missing_info(sample_dataframe)
    # assert 'missing_counts' in result
    # assert 'missing_rates' in result
    pass


def test_dataframe_missing_values_with_na():
    """Test counting missing values when NA present."""
    df_with_na = pd.DataFrame({
        'a': [1, 2, None, 4],
        'b': ['x', None, 'z', 'w']
    })
    # TODO: Implement after solution.py has get_missing_info function
    # result = get_missing_info(df_with_na)
    # assert result['missing_counts']['a'] == 1
    # assert result['missing_counts']['b'] == 1
    pass


def test_dataframe_empty_shape(empty_dataframe):
    """Test getting shape of empty DataFrame."""
    # TODO: Implement after solution.py has get_dataframe_info function
    # result = get_dataframe_info(empty_dataframe)
    # assert result['rows'] == 0
    # assert result['columns'] == 0
    pass


def test_type_conversion_to_category(sample_dataframe):
    """Test converting string columns to category type."""
    # TODO: Implement after solution.py has convert_to_category function
    # df_converted = convert_to_category(sample_dataframe, ['species', 'island'])
    # assert df_converted['species'].dtype.name == 'category'
    # assert df_converted['island'].dtype.name == 'category'
    pass


def test_head_and_tail(sample_dataframe):
    """Test getting head and tail of DataFrame."""
    # TODO: Implement after solution.py has preview_data function
    # head_result = preview_data(sample_dataframe, n=3, method='head')
    # assert len(head_result) == 3
    #
    # tail_result = preview_data(sample_dataframe, n=2, method='tail')
    # assert len(tail_result) == 2
    pass


def test_unique_values_for_categorical(sample_dataframe):
    """Test getting unique values for categorical columns."""
    # TODO: Implement after solution.py has get_unique_values function
    # result = get_unique_values(sample_dataframe, 'species')
    # expected = {'Adelie', 'Chinstrap', 'Gentoo'}
    # assert set(result) == expected
    pass


def test_pandas_integration_with_seaborn_penguins(penguins_dataset):
    """Test working with seaborn's penguins dataset."""
    # TODO: Implement after solution.py has validate_dataset function
    # result = validate_dataset(penguins_dataset)
    # assert result['rows'] > 0
    # assert result['columns'] > 0
    # assert 'species' in result['columns_list']
    pass


def test_data_path_resolution(tmp_path):
    """Test resolving relative and absolute paths."""
    # Create test file in subdirectory
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    test_file = data_dir / "test.csv"
    test_file.write_text("a,b\n1,2\n")

    # TODO: Implement after solution.py has resolve_path function
    # Test relative path
    # relative_path = resolve_path("data/test.csv", base_dir=tmp_path)
    # assert relative_path.exists()
    #
    # # Test absolute path
    # absolute_path = resolve_path(str(test_file))
    # assert absolute_path.exists()
    pass


def test_edge_case_path_with_spaces(tmp_path):
    """Test reading file with spaces in path."""
    # Create file with spaces in name
    file_with_spaces = tmp_path / "file with spaces.csv"
    file_with_spaces.write_text("a,b\n1,2\n")

    # TODO: Implement after solution.py has read_data function
    # df = read_data(file_with_spaces)
    # assert len(df) == 1
    pass

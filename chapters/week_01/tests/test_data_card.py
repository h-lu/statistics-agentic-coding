"""Tests for data card generation functionality."""

from __future__ import annotations

import pandas as pd
import pytest
from pathlib import Path


def test_generate_data_card_basic(sample_dataframe, sample_metadata):
    """Test basic data card generation."""
    # TODO: Implement after solution.py has generate_data_card function
    # data_card = generate_data_card(sample_dataframe, sample_metadata)
    #
    # assert isinstance(data_card, str)
    # assert "数据卡" in data_card
    # assert "数据来源" in data_card
    # assert "字段字典" in data_card
    # assert "规模概览" in data_card
    # assert "缺失概览" in data_card
    pass


def test_data_card_includes_metadata(sample_dataframe, sample_metadata):
    """Test that data card includes all metadata fields."""
    # TODO: Implement after solution.py has generate_data_card function
    # data_card = generate_data_card(sample_dataframe, sample_metadata)
    #
    # for key in sample_metadata.keys():
    #     assert key in data_card
    #     assert sample_metadata[key] in data_card
    pass


def test_data_card_field_dictionary(sample_dataframe, sample_metadata):
    """Test that field dictionary includes all columns."""
    # TODO: Implement after solution.py has generate_data_card function
    # data_card = generate_data_card(sample_dataframe, sample_metadata)
    #
    # for col in sample_dataframe.columns:
    #     assert col in data_card
    #     assert str(sample_dataframe[col].dtype) in data_card
    pass


def test_data_card_missing_rates(sample_dataframe, sample_metadata):
    """Test that missing rates are calculated correctly."""
    # TODO: Implement after solution.py has generate_data_card function
    # data_card = generate_data_card(sample_dataframe, sample_metadata)
    #
    # # Sample dataframe has no missing values
    # assert "0.0%" in data_card or "无缺失值" in data_card
    pass


def test_data_card_with_missing_values(sample_metadata, temp_output_dir):
    """Test data card with missing values."""
    df_with_na = pd.DataFrame({
        'a': [1, 2, None, 4],
        'b': ['x', None, 'z', 'w'],
        'c': [1.1, 2.2, 3.3, 4.4]
    })

    # TODO: Implement after solution.py has generate_data_card function
    # data_card = generate_data_card(df_with_na, sample_metadata)
    #
    # # Check that missing values are reported
    # assert 'a' in data_card
    # assert 'b' in data_card
    # # Missing rate should be 25% for both a and b
    # assert "25.0%" in data_card
    pass


def test_data_card_scale_overview(sample_dataframe, sample_metadata):
    """Test that scale overview is correct."""
    # TODO: Implement after solution.py has generate_data_card function
    # data_card = generate_data_card(sample_dataframe, sample_metadata)
    #
    # # Check row and column counts
    # assert "5" in data_card  # 5 rows
    # assert "7" in data_card  # 7 columns
    # assert "行数" in data_card
    # assert "列数" in data_card
    pass


def test_data_card_empty_dataframe(empty_dataframe, sample_metadata):
    """Test data card generation with empty DataFrame."""
    # TODO: Implement after solution.py has generate_data_card function
    # data_card = generate_data_card(empty_dataframe, sample_metadata)
    #
    # assert "0" in data_card  # 0 rows
    # assert "0" in data_card  # 0 columns (or close to it)
    pass


def test_data_card_all_na_dataframe(all_na_dataframe, sample_metadata):
    """Test data card with all NA values."""
    # TODO: Implement after solution.py has generate_data_card function
    # data_card = generate_data_card(all_na_dataframe, sample_metadata)
    #
    # # Should report high missing rates
    # assert "100.0%" in data_card
    pass


def test_data_card_markdown_format(sample_dataframe, sample_metadata):
    """Test that data card is valid Markdown."""
    # TODO: Implement after solution.py has generate_data_card function
    # data_card = generate_data_card(sample_dataframe, sample_metadata)
    #
    # # Check for Markdown headers
    # assert data_card.strip().startswith("#")
    # assert "##" in data_card  # Level 2 headers
    #
    # # Check for Markdown tables
    # assert "|" in data_card  # Table syntax
    # assert "---" in data_card  # Table separator
    pass


def test_data_card_write_to_file(sample_dataframe, sample_metadata, temp_output_dir):
    """Test writing data card to file."""
    output_path = temp_output_dir / "data_card.md"

    # TODO: Implement after solution.py has write_data_card function
    # write_data_card(sample_dataframe, sample_metadata, output_path)
    #
    # assert output_path.exists()
    # content = output_path.read_text(encoding='utf-8')
    # assert "数据卡" in content
    pass


def test_data_card_overwrite_existing(sample_dataframe, sample_metadata, temp_output_dir):
    """Test that writing overwrites existing file."""
    output_path = temp_output_dir / "data_card.md"

    # Create existing file with different content
    output_path.write_text("Old content", encoding='utf-8')

    # TODO: Implement after solution.py has write_data_card function
    # write_data_card(sample_dataframe, sample_metadata, output_path)
    #
    # content = output_path.read_text(encoding='utf-8')
    # assert "Old content" not in content
    # assert "数据卡" in content
    pass


def test_data_card_with_penguins_dataset(penguins_dataset, penguins_metadata):
    """Test data card generation with real Palmer Penguins dataset."""
    # TODO: Implement after solution.py has generate_data_card function
    # data_card = generate_data_card(penguins_dataset, penguins_metadata)
    #
    # # Check that all expected columns are present
    # expected_columns = ['species', 'island', 'bill_length_mm', 'bill_depth_mm',
    #                     'flipper_length_mm', 'body_mass_g', 'sex']
    # for col in expected_columns:
    #     assert col in data_card
    #
    # # Check that metadata is included
    # assert "Palmer Penguins" in data_card
    # assert "Palmer Station" in data_card
    #
    # # Penguins dataset has some missing values
    # # Check that missing overview is present and has some entries
    # assert "缺失概览" in data_card
    pass


def test_data_card_invalid_input_not_dataframe(sample_metadata):
    """Test that non-DataFrame input raises appropriate error."""
    # TODO: Implement after solution.py has generate_data_card function
    # with pytest.raises(TypeError, match="DataFrame"):
    #     generate_data_card([1, 2, 3], sample_metadata)
    #
    # with pytest.raises(TypeError, match="DataFrame"):
    #     generate_data_card("not a dataframe", sample_metadata)
    #
    # with pytest.raises(TypeError, match="DataFrame"):
    #     generate_data_card({'a': [1, 2, 3]}, sample_metadata)
    pass


def test_data_card_missing_metadata(sample_dataframe):
    """Test data card generation with minimal metadata."""
    minimal_metadata = {"source": "test"}

    # TODO: Implement after solution.py has generate_data_card function
    # data_card = generate_data_card(sample_dataframe, minimal_metadata)
    #
    # # Should still generate, just with minimal metadata
    # assert isinstance(data_card, str)
    # assert "source" in data_card
    # assert "test" in data_card
    pass


def test_data_chinese_encoding(sample_dataframe, sample_metadata, temp_output_dir):
    """Test that data card handles Chinese characters correctly."""
    output_path = temp_output_dir / "data_card_中文.md"

    # TODO: Implement after solution.py has write_data_card function
    # write_data_card(sample_dataframe, sample_metadata, output_path)
    #
    # content = output_path.read_text(encoding='utf-8')
    # assert "数据卡" in content
    # assert "数据来源" in content
    # assert "字段字典" in content
    pass


def test_data_card_mixed_dtypes(mixed_types_dataframe, sample_metadata):
    """Test data card with mixed data types."""
    # TODO: Implement after solution.py has generate_data_card function
    # data_card = generate_data_card(mixed_types_dataframe, sample_metadata)
    #
    # # Check that all types are represented
    # assert "int" in data_card.lower()
    # assert "float" in data_card.lower()
    # assert "object" in data_card.lower() or "str" in data_card.lower()
    # assert "bool" in data_card.lower()
    pass

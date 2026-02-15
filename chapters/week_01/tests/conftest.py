"""Shared fixtures for Week 01 tests."""

from __future__ import annotations

import pandas as pd
import pytest
import seaborn as sns
from pathlib import Path
import tempfile


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'species': ['Adelie', 'Chinstrap', 'Gentoo', 'Adelie', 'Gentoo'],
        'island': ['Torgersen', 'Dream', 'Biscoe', 'Biscoe', 'Biscoe'],
        'bill_length_mm': [39.1, 46.5, 48.1, 37.8, 46.5],
        'bill_depth_mm': [18.7, 17.9, 15.1, 18.3, 15.0],
        'flipper_length_mm': [181, 192, 210, 174, 212],
        'body_mass_g': [3750, 3800, 5500, 3400, 5200],
        'sex': ['male', 'female', 'male', 'female', 'male']
    })


@pytest.fixture
def empty_dataframe():
    """Create an empty DataFrame for testing."""
    return pd.DataFrame()


@pytest.fixture
def all_na_dataframe():
    """Create a DataFrame with all NA values."""
    return pd.DataFrame({
        'col1': [None, None, None],
        'col2': [None, None, None]
    })


@pytest.fixture
def mixed_types_dataframe():
    """Create a DataFrame with mixed types."""
    return pd.DataFrame({
        'int_col': [1, 2, 3],
        'float_col': [1.1, 2.2, 3.3],
        'str_col': ['a', 'b', 'c'],
        'bool_col': [True, False, True]
    })


@pytest.fixture
def penguins_dataset():
    """Load Palmer Penguins dataset from seaborn."""
    return sns.load_dataset("penguins")


@pytest.fixture
def sample_metadata():
    """Create sample metadata dictionary."""
    return {
        "数据集名称": "Test Dataset",
        "来源": "Test Source",
        "描述": "Test description",
        "收集时间": "2024"
    }


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary directory for test outputs."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def sample_csv_file(tmp_path):
    """Create a sample CSV file for testing."""
    df = pd.DataFrame({
        'a': [1, 2, 3],
        'b': ['x', 'y', 'z']
    })
    csv_path = tmp_path / "sample.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def penguins_metadata():
    """Create metadata for Palmer Penguins dataset."""
    return {
        "数据集名称": "Palmer Penguins",
        "来源": "seaborn 内置数据集",
        "原始来源": "Palmer Station, Antarctica LTER",
        "描述": "南极 Palmer Station 的三种企鹅（Adelie, Chinstrap, Gentoo）的形态测量数据",
        "收集时间": "2007-2009 年",
        "单位说明": "长度单位为毫米（mm），重量单位为克（g）"
    }

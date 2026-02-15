"""
Week 03 共享 Fixtures

提供测试用的共享数据和工具函数。

测试覆盖：
- 缺失值处理（MCAR/MAR/MNAR）
- 异常值检测（IQR/Z-score）
- 数据转换（标准化/归一化/对数变换）
- 特征编码（One-hot/Label encoding）
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import sys

# 添加 starter_code 到 Python 路径
starter_code_path = Path(__file__).parent.parent / "starter_code"
if str(starter_code_path) not in sys.path:
    sys.path.insert(0, str(starter_code_path))


# =============================================================================
# Missing Values Fixtures
# =============================================================================

@pytest.fixture
def dataframe_with_missing_values() -> pd.DataFrame:
    """
    创建包含缺失值的 DataFrame
    用于测试缺失值检测和处理

    缺失模式：
    - age: 15% 缺失（MCAR 风格）
    - income: 20% 缺失（MAR 风格：与 age 相关）
    - city: 10% 缺失
    """
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        'user_id': range(1, n + 1),
        'age': np.random.randint(18, 70, size=n),
        'income': np.random.randint(20000, 100000, size=n),
        'city': np.random.choice(['北京', '上海', '深圳', '广州'], size=n),
    })

    # MCAR: 随机缺失
    age_missing_mask = np.random.random(n) < 0.15
    df.loc[age_missing_mask, 'age'] = np.nan

    # MAR: 与 age 相关的缺失（年龄大的更可能不填收入）
    income_missing_prob = (df['age'] / 70).fillna(0.5) * 0.3
    income_missing_mask = np.random.random(n) < income_missing_prob
    df.loc[income_missing_mask, 'income'] = np.nan

    # 随机城市缺失
    city_missing_mask = np.random.random(n) < 0.10
    df.loc[city_missing_mask, 'city'] = np.nan

    return df


@pytest.fixture
def dataframe_all_missing_column() -> pd.DataFrame:
    """
    创建包含全缺失列的 DataFrame
    用于测试极端情况
    """
    return pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'all_missing': [np.nan, np.nan, np.nan, np.nan, np.nan],
        'partial_missing': [1, np.nan, 3, np.nan, 5],
        'no_missing': [10, 20, 30, 40, 50],
    })


@pytest.fixture
def dataframe_no_missing() -> pd.DataFrame:
    """创建无缺失值的 DataFrame"""
    np.random.seed(42)
    return pd.DataFrame({
        'x': np.random.randn(50),
        'y': np.random.randn(50) + 2,
    })


# =============================================================================
# Outlier Detection Fixtures
# =============================================================================

@pytest.fixture
def series_with_outliers() -> pd.Series:
    """
    创建包含异常值的 Series
    用于测试 IQR 和 Z-score 检测
    """
    # 基础正态分布数据
    np.random.seed(42)
    base_data = list(np.random.normal(100, 15, 50))
    # 添加明显的异常值
    base_data.extend([10, 15, 190, 200])
    return pd.Series(base_data)


@pytest.fixture
def series_no_outliers() -> pd.Series:
    """
    创建无异常值的正态分布 Series
    用于测试在干净数据上的行为
    """
    np.random.seed(42)
    return pd.Series(np.random.normal(100, 15, 100))


@pytest.fixture
def series_all_outliers() -> pd.Series:
    """
    创建大部分是异常值的 Series
    用于测试极端情况
    """
    return pd.Series([1, 2, 3, 1000, 2000, 3000, -500, -1000])


@pytest.fixture
def skewed_series() -> pd.Series:
    """
    创建右偏的 Series
    用于测试 Z-score 在非正态数据上的问题
    """
    np.random.seed(42)
    # 对数正态分布产生右偏
    return pd.Series(np.random.lognormal(mean=3, sigma=0.8, size=200))


# =============================================================================
# Data Transformation Fixtures
# =============================================================================

@pytest.fixture
def multi_scale_dataframe() -> pd.DataFrame:
    """
    创建包含不同尺度变量的 DataFrame
    用于测试标准化和归一化
    """
    np.random.seed(42)
    return pd.DataFrame({
        'small_scale': np.random.uniform(0, 1, 100),      # 0-1 范围
        'medium_scale': np.random.uniform(10, 100, 100),  # 10-100 范围
        'large_scale': np.random.uniform(1000, 10000, 100),  # 1000-10000 范围
    })


@pytest.fixture
def constant_column_dataframe() -> pd.DataFrame:
    """
    创建包含常量列的 DataFrame
    用于测试边界情况（标准差为0）
    """
    return pd.DataFrame({
        'constant': [42, 42, 42, 42, 42],
        'varying': [1, 2, 3, 4, 5],
    })


@pytest.fixture
def single_row_dataframe() -> pd.DataFrame:
    """
    创建单行 DataFrame
    用于测试边界情况
    """
    return pd.DataFrame({
        'a': [10],
        'b': [20],
        'c': [30],
    })


@pytest.fixture
def right_skewed_series() -> pd.Series:
    """
    创建右偏的 Series（模拟收入数据）
    用于测试对数变换
    """
    np.random.seed(42)
    return pd.Series(np.random.lognormal(mean=10, sigma=0.5, size=1000))


@pytest.fixture
def series_with_zeros() -> pd.Series:
    """
    创建包含零值的 Series
    用于测试对数变换的限制（log(0) 无定义）
    """
    return pd.Series([0, 1, 2, 3, 10, 100, 1000])


@pytest.fixture
def series_with_negatives() -> pd.Series:
    """
    创建包含负值的 Series
    用于测试对数变换的限制
    """
    return pd.Series([-10, -5, -1, 0, 1, 5, 10])


# =============================================================================
# Feature Encoding Fixtures
# =============================================================================

@pytest.fixture
def nominal_dataframe() -> pd.DataFrame:
    """
    创建包含名义变量的 DataFrame
    用于测试 One-hot 编码
    """
    return pd.DataFrame({
        'id': [1, 2, 3, 4, 5, 6],
        'species': ['Adelie', 'Chinstrap', 'Gentoo', 'Adelie', 'Chinstrap', 'Gentoo'],
        'island': ['Biscoe', 'Dream', 'Biscoe', 'Torgersen', 'Dream', 'Biscoe'],
    })


@pytest.fixture
def ordinal_dataframe() -> pd.DataFrame:
    """
    创建包含有序变量的 DataFrame
    用于测试 Label 编码
    """
    return pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'education': ['高中', '本科', '硕士', '博士', '本科'],
        'satisfaction': ['低', '中', '高', '中', '低'],
    })


# =============================================================================
# Path Fixtures
# =============================================================================

@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """
    创建临时输出目录
    用于测试图表和文件生成功能
    """
    output_dir = tmp_path / "output"
    output_dir.mkdir(exist_ok=True)
    return output_dir


# =============================================================================
# Helper Fixtures
# =============================================================================

@pytest.fixture
def sample_cleaning_log() -> list[dict[str, Any]]:
    """
    创建示例清洗日志
    用于测试清洗日志生成功能
    """
    return [
        {
            'variable': 'age',
            'issue': 'missing_values',
            'action': 'median_fill',
            'reason': '缺失率 15%，使用中位数填充保持分布稳定性',
            'n_affected': 15,
        },
        {
            'variable': 'income',
            'issue': 'missing_values',
            'action': 'keep_as_is',
            'reason': 'MAR 缺失机制，填充可能引入偏差',
            'n_affected': 20,
        },
        {
            'variable': 'body_mass_g',
            'issue': 'outliers',
            'action': 'keep_as_is',
            'reason': 'IQR 检测到的值在合理范围内',
            'n_affected': 2,
        },
    ]

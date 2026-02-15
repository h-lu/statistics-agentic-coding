"""
Week 02 共享 Fixtures

提供测试用的共享数据和工具函数。
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
# Data Fixtures
# =============================================================================

@pytest.fixture
def sample_numeric_data() -> pd.Series:
    """
    创建示例数值型数据（正态分布，无极端值）
    用于测试集中趋势和离散程度的基础功能
    """
    np.random.seed(42)
    return pd.Series([10, 12, 15, 18, 20, 22, 25, 28, 30, 32])


@pytest.fixture
def sample_data_with_outliers() -> pd.Series:
    """
    创建包含极端值的数值型数据
    用于测试均值 vs 中位数的差异
    """
    return pd.Series([10, 12, 15, 18, 20, 22, 25, 28, 30, 32, 100, 150])


@pytest.fixture
def sample_skewed_data() -> pd.Series:
    """
    创建右偏数据（模拟收入等长尾分布）
    用于测试偏态分布下的集中趋势选择
    """
    return pd.Series([100, 150, 180, 200, 220, 250, 280, 300, 350, 5000])


@pytest.fixture
def sample_categorical_series() -> pd.Series:
    """创建分类型数据 Series"""
    return pd.Series(['北京', '上海', '深圳', '北京', '上海', '北京', '深圳', '上海'])


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """
    创建用于测试的完整 DataFrame
    包含数值型和分类型列
    """
    np.random.seed(42)
    return pd.DataFrame({
        'user_id': range(1, 11),
        'age': [22, 25, 28, 30, 32, 35, 38, 40, 42, 45],
        'monthly_spend': [100, 150, 180, 200, 220, 250, 280, 300, 350, 5000],
        'city': ['北京', '上海', '深圳', '北京', '上海', '深圳', '北京', '上海', '深圳', '北京'],
        'is_vip': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    })


@pytest.fixture
def dataframe_with_missing() -> pd.DataFrame:
    """创建包含缺失值的 DataFrame"""
    return pd.DataFrame({
        'age': [22, 25, None, 30, 32, 35, 38, 40, 42, 45],
        'salary': [8000, 12000, 15000, None, 20000, 25000, 28000, 30000, 35000, None],
        'city': ['北京', '上海', None, '北京', '上海', '深圳', '北京', None, '深圳', '北京'],
    })


@pytest.fixture
def empty_series() -> pd.Series:
    """创建空 Series"""
    return pd.Series([], dtype=float)


@pytest.fixture
def single_value_series() -> pd.Series:
    """创建只包含一个值的 Series"""
    return pd.Series([42.0])


# =============================================================================
# Path Fixtures
# =============================================================================

@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """
    创建临时输出目录
    用于测试图表生成功能
    """
    output_dir = tmp_path / "figures"
    output_dir.mkdir(exist_ok=True)
    return output_dir


@pytest.fixture
def sample_report_path(tmp_path: Path) -> Path:
    """创建临时报告文件路径"""
    return tmp_path / "report.md"


# =============================================================================
# Helper Fixtures
# =============================================================================

@pytest.fixture
def sample_summary_dict() -> dict[str, Any]:
    """
    创建示例摘要字典
    用于测试报告生成功能
    """
    return {
        'monthly_spend': {
            'mean': 683.0,
            'median': 235.0,
            'std': 1523.45,
            'q1': 187.5,
            'q3': 315.0,
            'iqr': 127.5,
        },
        'age': {
            'mean': 33.7,
            'median': 33.5,
            'std': 7.8,
            'q1': 27.5,
            'q3': 40.5,
            'iqr': 13.0,
        }
    }


# =============================================================================
# Statistical test data fixtures
# =============================================================================

@pytest.fixture
def two_groups_data() -> dict[str, pd.Series]:
    """
    创建两组数据用于比较
    模拟 A/B 测试场景
    """
    np.random.seed(42)
    return {
        'group_a': pd.Series(np.random.normal(100, 15, 50)),
        'group_b': pd.Series(np.random.normal(105, 15, 50)),
    }


@pytest.fixture
def multi_category_data() -> pd.DataFrame:
    """
    创建多分类数据
    用于测试分组箱线图等功能
    """
    np.random.seed(42)
    categories = ['A', 'B', 'C', 'D']
    data = []
    for cat in categories:
        if cat == 'A':
            data.extend([(cat, x) for x in np.random.normal(100, 10, 20)])
        elif cat == 'B':
            data.extend([(cat, x) for x in np.random.normal(105, 15, 20)])
        elif cat == 'C':
            data.extend([(cat, x) for x in np.random.normal(95, 12, 20)])
        else:
            data.extend([(cat, x) for x in np.random.normal(110, 20, 20)])

    return pd.DataFrame(data, columns=['category', 'value'])


# =============================================================================
# Visualization test fixtures
# =============================================================================

@pytest.fixture
def sample_plot_config() -> dict[str, Any]:
    """
    创建示例图表配置
    用于测试可视化功能
    """
    return {
        'title': '测试图表',
        'xlabel': 'X轴',
        'ylabel': 'Y轴',
        'figsize': (8, 4),
        'dpi': 150,
    }

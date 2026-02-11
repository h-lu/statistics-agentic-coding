"""
Week 03 共享 Fixtures

提供测试用的共享数据和工具函数。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest


# =============================================================================
# Data Fixtures - 基础测试数据
# =============================================================================

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """
    创建包含缺失值、异常值的测试数据

    包含以下列：
    - age: 年龄，有正常值和异常值
    - income: 收入，有缺失值和极端值
    - city: 城市，分类型数据
    - user_level: 用户等级，ordinal 类别
    - score: 评分，用于缩放测试
    """
    np.random.seed(42)
    return pd.DataFrame({
        'age': [22, 25, 28, 30, 32, 35, 38, 40, 42, 150, -5, None],
        'income': [5000, 6000, 7500, None, 10000, 12000, None, 15000, 20000, 500000, 8000, 9000],
        'city': ['北京', '上海', '深圳', '北京', None, '上海', '深圳', '北京', '上海', '深圳', '北京', '上海'],
        'user_level': ['bronze', 'silver', 'gold', 'bronze', 'silver', 'diamond', 'gold', 'silver', 'diamond', 'gold', 'bronze', 'silver'],
        'score': [65, 72, 88, 91, 55, 78, 82, 95, 60, 85, 70, 75],
    })


@pytest.fixture
def df_with_missing() -> pd.DataFrame:
    """创建包含各种缺失值情况的 DataFrame"""
    return pd.DataFrame({
        'numeric_col': [1.0, 2.0, np.nan, 4.0, np.nan, 6.0],
        'string_col': ['a', None, 'c', 'd', None, 'f'],
        'all_missing': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        'no_missing': [1, 2, 3, 4, 5, 6],
    })


@pytest.fixture
def df_with_outliers() -> pd.DataFrame:
    """创建包含异常值的 DataFrame"""
    return pd.DataFrame({
        'normal_with_outliers': [10, 12, 15, 18, 20, 22, 25, 28, 30, 100, 150],
        'all_normal': [10, 12, 15, 18, 20, 22, 25, 28, 30, 32, 35],
        'all_outliers': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100],
        'constant': [50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50],
    })


@pytest.fixture
def df_for_scaling() -> pd.DataFrame:
    """创建用于特征缩放测试的 DataFrame"""
    np.random.seed(42)
    return pd.DataFrame({
        'feature_small': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'feature_large': [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000],
        'feature_normal': np.random.normal(50, 10, 10),
        'feature_constant': [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    })


@pytest.fixture
def df_for_encoding() -> pd.DataFrame:
    """创建用于特征编码测试的 DataFrame"""
    return pd.DataFrame({
        'nominal_cat': ['北京', '上海', '深圳', '广州', '北京', '上海', '深圳', '广州'],
        'ordinal_cat': ['low', 'medium', 'high', 'low', 'medium', 'high', 'low', 'medium'],
        'numeric': [1, 2, 3, 4, 5, 6, 7, 8],
    })


@pytest.fixture
def series_with_missing() -> pd.Series:
    """创建包含缺失值的 Series"""
    return pd.Series([1.0, 2.0, np.nan, 4.0, np.nan, 6.0, 7.0])


@pytest.fixture
def series_all_missing() -> pd.Series:
    """创建全缺失的 Series"""
    return pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan])


@pytest.fixture
def series_no_missing() -> pd.Series:
    """创建无缺失值的 Series"""
    return pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])


# =============================================================================
# Path Fixtures
# =============================================================================

@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """创建临时输出目录"""
    output_dir = tmp_path / "output"
    output_dir.mkdir(exist_ok=True)
    return output_dir


# =============================================================================
# Helper Fixtures - 清洗决策记录
# =============================================================================

@pytest.fixture
def sample_cleaning_decisions() -> list[dict[str, Any]]:
    """创建示例清洗决策列表"""
    return [
        {
            'field': 'monthly_spend',
            'problem': '缺失率 12.3%，缺失率随用户等级升高而增加',
            'strategy': '按用户等级分组，用组内中位数填充',
            'rationale': '中位数对极端值稳健；分组填充利用 MAR 信息',
            'alternatives': '删除（会损失 12% 样本）；全局均值填充（会压缩方差）',
            'impact': '填充后均值从 245 元变为 267 元，标准差从 180 变为 165'
        },
        {
            'field': 'monthly_spend',
            'problem': '检测到 23 个 IQR 异常点',
            'strategy': '分类处理：删除疑似刷单（高消费+新用户），保留 VIP 用户',
            'rationale': '异常点不一定是错误，需要结合业务规则判断',
            'alternatives': '全部删除（会损失真实 VIP）；全部保留（会污染模型）',
            'impact': '删除 3 条可疑记录，保留 20 条 VIP 记录'
        },
        {
            'field': 'age, monthly_income',
            'problem': '特征尺度差异大（18-80 vs 3000-50000）',
            'strategy': 'StandardScaler 标准化',
            'rationale': '数据近似正态分布，标准化后均值为 0、标准差为 1',
            'alternatives': 'MinMaxScaler（适合有界数据）；RobustScaler（如果有异常值）',
            'impact': '两特征现在处于相同尺度，可公平比较'
        }
    ]


@pytest.fixture
def sample_decision_record() -> dict[str, Any]:
    """创建单个清洗决策记录"""
    return {
        'field': 'income',
        'problem': '缺失率 15%，高收入用户缺失率更高',
        'strategy': '按用户等级分组，用组内中位数填充',
        'rationale': '中位数对极端值稳健；分组填充利用 MAR 信息',
        'alternatives': '删除（会损失样本）；全局均值填充',
        'impact': '填充后均值变化在 5% 以内'
    }


# =============================================================================
# Edge Case Fixtures
# =============================================================================

@pytest.fixture
def empty_dataframe() -> pd.DataFrame:
    """创建空 DataFrame"""
    return pd.DataFrame()


@pytest.fixture
def single_row_df() -> pd.DataFrame:
    """创建单行 DataFrame"""
    return pd.DataFrame({
        'a': [1],
        'b': ['test'],
    })


@pytest.fixture
def df_with_special_chars() -> pd.DataFrame:
    """创建包含特殊字符的 DataFrame"""
    return pd.DataFrame({
        'text': ['hello', 'world', 'test@#$', '123', '', None],
        'numeric': [1, 2, 3, 4, 5, 6],
    })

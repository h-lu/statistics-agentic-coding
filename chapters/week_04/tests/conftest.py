"""
Week 04 共享 Fixtures

提供测试用的共享数据和工具函数，用于 EDA 相关测试。
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
    创建用于 EDA 测试的基础数据

    包含以下列：
    - user_id: 用户ID
    - age: 年龄
    - monthly_income: 月收入
    - monthly_spend: 月消费
    - gender: 性别
    - city_tier: 城市级别
    - user_level: 用户等级
    """
    np.random.seed(42)
    n = 100

    df = pd.DataFrame({
        'user_id': range(1, n + 1),
        'age': np.random.normal(35, 10, n).astype(int).clip(18, 65),
        'monthly_income': np.random.lognormal(8.5, 0.5, n).astype(int),
        'gender': np.random.choice(['男', '女'], size=n),
        'city_tier': np.random.choice(['一线', '二线', '三线'], size=n),
    })

    # 让消费和收入相关（模拟真实业务逻辑）
    df['monthly_spend'] = (
        df['monthly_income'] * 0.3 * np.random.uniform(0.5, 1.5, n) +
        np.random.normal(0, 200, n)
    ).astype(int).clip(100, None)

    # 用户等级基于消费
    df['user_level'] = pd.cut(
        df['monthly_spend'],
        bins=[0, 500, 1500, 5000, float('inf')],
        labels=['普通', '银卡', '金卡', '钻石']
    )

    return df


@pytest.fixture
def df_for_correlation() -> pd.DataFrame:
    """
    创建用于相关性测试的数据

    包含强相关、弱相关、无相关的变量对
    """
    np.random.seed(42)
    n = 50

    # 基础变量
    x = np.random.normal(50, 10, n)

    # 创建单调相关：基于 x 的排序位置
    y_monotonic = np.argsort(np.argsort(x)) + np.random.normal(0, 3, n)

    return pd.DataFrame({
        'x': x,
        'y_strong': x * 2 + np.random.normal(0, 5, n),      # 强线性相关
        'y_weak': x * 0.1 + np.random.normal(0, 20, n),     # 弱线性相关
        'y_none': np.random.normal(50, 10, n),              # 无相关
        'y_monotonic': y_monotonic,                         # 单调相关
    })


@pytest.fixture
def df_with_outliers() -> pd.DataFrame:
    """
    创建包含异常值的数据，用于测试 Pearson vs Spearman
    """
    np.random.seed(42)
    n = 50

    x = np.random.normal(50, 10, n)
    y = x * 2 + np.random.normal(0, 5, n)

    # 添加极端异常值 - 确保长度一致
    x_with_outlier = np.append(x, [500])
    y_with_outlier = np.append(y, [1000])

    # 创建 DataFrame 时确保所有列长度相同
    return pd.DataFrame({
        'x_normal': np.append(x, [np.nan]),  # 添加 nan 使长度一致
        'y_normal': np.append(y, [np.nan]),
        'x_with_outlier': x_with_outlier,
        'y_with_outlier': y_with_outlier,
    })


@pytest.fixture
def df_for_groupby() -> pd.DataFrame:
    """
    创建用于分组比较测试的数据
    """
    np.random.seed(42)

    return pd.DataFrame({
        'group': ['A'] * 30 + ['B'] * 30 + ['C'] * 30,
        'subgroup': (['X'] * 15 + ['Y'] * 15) * 3,
        'value': (
            list(np.random.normal(100, 10, 30)) +
            list(np.random.normal(120, 15, 30)) +
            list(np.random.normal(90, 8, 30))
        ),
        'category': np.random.choice(['cat1', 'cat2', 'cat3'], 90),
    })


@pytest.fixture
def df_with_missing_for_corr() -> pd.DataFrame:
    """
    创建包含缺失值的相关性测试数据
    """
    np.random.seed(42)
    n = 50

    df = pd.DataFrame({
        'x': np.random.normal(50, 10, n),
        'y': np.random.normal(50, 10, n),
        'z': np.random.normal(50, 10, n),
    })

    # 添加缺失值
    df.loc[0:4, 'x'] = np.nan
    df.loc[10:14, 'y'] = np.nan
    df.loc[20:24, 'z'] = np.nan

    return df


@pytest.fixture
def df_for_stratified() -> pd.DataFrame:
    """
    创建用于分层分析测试的数据

    模拟混杂变量场景：性别对消费的影响被收入混杂
    """
    np.random.seed(42)
    n = 100

    # 女性收入略高
    income_male = np.random.lognormal(8.4, 0.5, n // 2)
    income_female = np.random.lognormal(8.6, 0.5, n // 2)

    # 消费主要由收入决定，性别直接影响较小
    spend_male = income_male * 0.3 + np.random.normal(0, 100, n // 2)
    spend_female = income_female * 0.3 + np.random.normal(0, 100, n // 2)

    return pd.DataFrame({
        'gender': ['男'] * (n // 2) + ['女'] * (n // 2),
        'income': np.concatenate([income_male, income_female]).astype(int),
        'spend': np.concatenate([spend_male, spend_female]).astype(int),
    })


@pytest.fixture
def sample_hypotheses() -> list[dict[str, Any]]:
    """
    创建示例假设清单
    """
    return [
        {
            'id': 'H1',
            'description': '用户收入与月消费金额存在正相关关系',
            'H0': '收入与消费的 Pearson 相关系数 = 0',
            'H1': '收入与消费的 Pearson 相关系数 > 0',
            'data_support': 'Pearson r = 0.52, p < 0.001',
            'proposed_test': 'Pearson 相关性检验',
            'confounders': '年龄、城市级别',
            'priority': '高'
        },
        {
            'id': 'H2',
            'description': '不同城市级别用户的平均消费存在差异',
            'H0': '一线 = 二线 = 三线城市的平均消费',
            'H1': '至少有一组城市的平均消费不同',
            'data_support': '透视表显示一线城市均值 2850，三线 1920',
            'proposed_test': '单因素方差分析 (ANOVA)',
            'confounders': '收入分布、用户等级构成',
            'priority': '中'
        },
    ]


@pytest.fixture
def invalid_hypotheses() -> list[dict[str, Any]]:
    """
    创建无效假设示例（用于反例测试）
    """
    return [
        {
            'id': 'H_BAD_1',
            'description': '',  # 空描述
            'H0': '相关系数 = 0',
            'H1': '相关系数 > 0',
        },
        {
            'id': 'H_BAD_2',
            'description': '测试描述',
            'H0': '',  # 空 H0
            'H1': '有差异',
        },
        {
            'id': 'H_BAD_3',
            'description': '测试描述',
            'H0': '无差异',
            'H1': '',  # 空 H1
        },
    ]


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
        'x': [1],
        'y': [2],
        'group': ['A'],
    })


@pytest.fixture
def df_with_empty_groups() -> pd.DataFrame:
    """
    创建包含空分组的数据
    """
    return pd.DataFrame({
        'group': ['A', 'A', 'B', 'B', 'C'],  # C 只有一个值
        'value': [1, 2, 3, 4, 5],
    })


@pytest.fixture
def constant_column_df() -> pd.DataFrame:
    """
    创建包含常数列的数据（相关性测试边界情况）
    """
    np.random.seed(42)
    return pd.DataFrame({
        'x': np.random.normal(50, 10, 20),
        'constant': [5] * 20,
    })


@pytest.fixture
def non_numeric_df() -> pd.DataFrame:
    """
    创建非数值型数据（用于反例测试）
    """
    return pd.DataFrame({
        'text_col': ['a', 'b', 'c', 'd'],
        'another_text': ['x', 'y', 'z', 'w'],
    })


# =============================================================================
# Path Fixtures
# =============================================================================

@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """创建临时输出目录"""
    output_dir = tmp_path / "output"
    output_dir.mkdir(exist_ok=True)
    return output_dir

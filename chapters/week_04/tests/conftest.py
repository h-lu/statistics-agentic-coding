"""
Week 04 共享 Fixtures

提供测试用的共享数据和工具函数。

测试覆盖：
- 相关分析（Pearson/Spearman/Kendall）
- 分组比较（groupby/透视表）
- 多变量可视化（散点图矩阵/热力图）
- 时间序列初步（趋势/季节性）
- 假设清单生成
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
# Correlation Analysis Fixtures
# =============================================================================

@pytest.fixture
def positive_correlation_data() -> pd.DataFrame:
    """
    创建正相关的数据
    用于测试 Pearson 相关系数 > 0 的情况
    """
    np.random.seed(42)
    n = 100
    x = np.linspace(0, 100, n)
    y = 2 * x + 10 + np.random.normal(0, 5, n)  # 强正相关
    return pd.DataFrame({'x': x, 'y': y})


@pytest.fixture
def negative_correlation_data() -> pd.DataFrame:
    """
    创建负相关的数据
    用于测试 Pearson 相关系数 < 0 的情况
    """
    np.random.seed(42)
    n = 100
    x = np.linspace(0, 100, n)
    y = -1.5 * x + 150 + np.random.normal(0, 8, n)  # 强负相关
    return pd.DataFrame({'x': x, 'y': y})


@pytest.fixture
def no_correlation_data() -> pd.DataFrame:
    """
    创建无相关的数据
    用于测试相关系数 ≈ 0 的情况
    """
    np.random.seed(42)
    n = 100
    x = np.random.normal(50, 15, n)
    y = np.random.normal(50, 15, n)  # 完全独立
    return pd.DataFrame({'x': x, 'y': y})


@pytest.fixture
def monotonic_nonlinear_data() -> pd.DataFrame:
    """
    创建单调但非线性的关系
    用于测试 Spearman 相关系数（对单调关系有效）
    """
    np.random.seed(42)
    n = 100
    x = np.linspace(0, 10, n)
    y = x**2 + np.random.normal(0, 2, n)  # 二次关系，单调递增
    return pd.DataFrame({'x': x, 'y': y})


@pytest.fixture
def correlation_with_outliers() -> pd.DataFrame:
    """
    创建包含异常值的相关数据
    用于测试异常值对相关系数的影响
    """
    np.random.seed(42)
    # 基础正相关数据
    x = np.random.normal(50, 10, 50)
    y = x * 0.8 + np.random.normal(0, 5, 50)

    df = pd.DataFrame({'x': x, 'y': y})

    # 添加异常值：一个点会显著改变相关系数
    df_outlier = pd.DataFrame({'x': [100], 'y': [0]})
    df_with_outlier = pd.concat([df, df_outlier], ignore_index=True)

    return df_with_outlier


@pytest.fixture
def small_sample_data() -> pd.DataFrame:
    """
    创建小样本数据（< 20）
    用于测试 Kendall 相关系数（对小样本稳定）
    """
    np.random.seed(42)
    n = 15
    x = np.random.normal(50, 10, n)
    y = x * 0.7 + np.random.normal(0, 5, n)
    return pd.DataFrame({'x': x, 'y': y})


@pytest.fixture
def multivariate_data() -> pd.DataFrame:
    """
    创建多变量数据
    用于测试相关矩阵和散点图矩阵
    """
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        'age': np.random.randint(18, 70, n),
        'income': np.random.lognormal(10, 0.5, n),
        'purchase_amount': 50 + np.random.randint(18, 70, n) * 2 + np.random.normal(0, 20, n),
        'time_on_site': np.random.exponential(180, n)
    })


# =============================================================================
# Group Comparison Fixtures
# =============================================================================

@pytest.fixture
def grouped_purchase_data() -> pd.DataFrame:
    """
    创建按渠道分组的购买数据
    用于测试 groupby 功能
    """
    np.random.seed(42)
    n = 300

    # 三个来源渠道：direct, search, social
    source_list = []
    purchase_list = []

    for source in ['direct', 'search', 'social']:
        # 不同渠道有不同的均值
        if source == 'direct':
            mean_purchase = 100
            std_purchase = 20
        elif source == 'search':
            mean_purchase = 120
            std_purchase = 25
        else:  # social
            mean_purchase = 80
            std_purchase = 15

        purchases = np.random.normal(mean_purchase, std_purchase, n)
        source_list.extend([source] * n)
        purchase_list.extend(purchases)

    return pd.DataFrame({
        'source': source_list[:len(purchase_list)],
        'purchase_amount': purchase_list
    })


@pytest.fixture
def multi_group_data() -> pd.DataFrame:
    """
    创建多维度分组数据
    用于测试透视表功能
    """
    np.random.seed(42)
    n = 200

    data = {
        'region': np.random.choice(['North', 'South', 'East', 'West'], n),
        'product': np.random.choice(['A', 'B', 'C'], n),
        'sales': np.random.randint(50, 500, n),
        'quantity': np.random.randint(1, 20, n)
    }

    return pd.DataFrame(data)


@pytest.fixture
def time_series_data() -> pd.DataFrame:
    """
    创建时间序列数据
    用于测试趋势和季节性识别
    """
    np.random.seed(42)
    dates = pd.date_range("2025-01-01", "2025-12-31", freq="D")

    # 基础趋势：线性上升
    trend = np.linspace(100, 200, len(dates))

    # 季节性：周末高、工作日低
    seasonality = np.where(dates.dayofweek >= 5, 30, 0)

    # 随机噪声
    noise = np.random.normal(0, 15, len(dates))

    daily_sales = trend + seasonality + noise

    df = pd.DataFrame({
        'date': dates,
        'sales': daily_sales
    })
    df['week'] = df['date'].dt.isocalendar().week

    return df


# =============================================================================
# Hypothesis List Fixtures
# =============================================================================

@pytest.fixture
def sample_observations() -> list[dict[str, Any]]:
    """
    创建示例观察列表
    用于测试假设清单生成
    """
    return [
        {
            'observation': '搜索渠道的平均购买金额（$120）显著高于社交渠道（$80）',
            'explanation': '搜索渠道的用户有明确的购买意图',
            'test_method': '双样本 t 检验（Week 06）',
            'priority': 'high'
        },
        {
            'observation': '年龄与购买金额呈正相关（Pearson r = 0.65）',
            'explanation': '年龄大的用户购买力更强',
            'test_method': '回归分析（Week 09）',
            'priority': 'medium'
        }
    ]


@pytest.fixture
def invalid_hypothesis() -> dict[str, Any]:
    """
    创建缺少必填字段的假设
    用于测试假设验证功能
    """
    return {
        'observation': '搜索渠道购买金额更高',
        # 缺少 explanation 和 test_method
    }


# =============================================================================
# Edge Case Fixtures
# =============================================================================

@pytest.fixture
def empty_dataframe() -> pd.DataFrame:
    """创建空 DataFrame"""
    return pd.DataFrame()


@pytest.fixture
def single_column_dataframe() -> pd.DataFrame:
    """创建单列 DataFrame"""
    return pd.DataFrame({'value': [1, 2, 3, 4, 5]})


@pytest.fixture
def constant_column_dataframe() -> pd.DataFrame:
    """
    创建包含常量列的 DataFrame
    用于测试边界情况（标准差为0，相关系数无定义）
    """
    return pd.DataFrame({
        'constant': [42, 42, 42, 42, 42],
        'varying': [1, 2, 3, 4, 5],
    })


@pytest.fixture
def all_nan_dataframe() -> pd.DataFrame:
    """创建全 NaN 的 DataFrame"""
    return pd.DataFrame({
        'x': [np.nan, np.nan, np.nan],
        'y': [np.nan, np.nan, np.nan],
    })


# =============================================================================
# Path Fixtures
# =============================================================================

@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """
    创建临时输出目录
    用于测试图表生成功能
    """
    output_dir = tmp_path / "output"
    output_dir.mkdir(exist_ok=True)
    return output_dir

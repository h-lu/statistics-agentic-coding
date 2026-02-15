"""
Week 08 共享 Fixtures

提供测试用的共享数据和工具函数。

测试覆盖：
- 置信区间（t 分布、Percentile Bootstrap、BCa Bootstrap）
- Bootstrap 重采样原理
- 置换检验
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
# 置信区间测试 Fixtures
# =============================================================================

@pytest.fixture
def normal_data_small():
    """
    小样本正态数据（n=30）
    用于测试 t 分布 CI
    """
    np.random.seed(42)
    return np.random.normal(loc=3.2, scale=1.5, size=30)


@pytest.fixture
def normal_data_large():
    """
    大样本正态数据（n=200）
    用于测试大样本下的 CI 收敛
    """
    np.random.seed(42)
    return np.random.normal(loc=3.2, scale=1.5, size=200)


@pytest.fixture
def skewed_data():
    """
    偏态数据（指数分布）
    用于测试 Bootstrap 在非正态数据上的表现
    """
    np.random.seed(42)
    return np.random.exponential(scale=2.0, size=100)


@pytest.fixture
def bimodal_data():
    """
    双峰分布数据
    用于测试 CI 在复杂分布上的表现
    """
    np.random.seed(42)
    # 两个正态分布混合
    group1 = np.random.normal(loc=0, scale=1, size=50)
    group2 = np.random.normal(loc=10, scale=1, size=50)
    return np.concatenate([group1, group2])


@pytest.fixture
def binary_proportion_data():
    """
    二元比例数据（0/1）
    用于测试比例的 CI
    """
    np.random.seed(42)
    return np.random.binomial(1, 0.15, 200)


# =============================================================================
# Bootstrap 测试 Fixtures
# =============================================================================

@pytest.fixture
def bootstrap_test_data():
    """
    用于 Bootstrap 测试的标准数据集
    """
    np.random.seed(42)
    return np.random.normal(loc=100, scale=15, size=100)


@pytest.fixture
def bootstrap_small_sample():
    """
    小样本数据（n=20）
    用于测试 Bootstrap 在小样本下的局限性
    """
    np.random.seed(42)
    return np.random.normal(loc=50, scale=10, size=20)


@pytest.fixture
def bootstrap_minimal_sample():
    """
    极小样本数据（n=5）
    用于测试 Bootstrap 的边界情况
    """
    np.random.seed(42)
    return np.array([3.2, 2.8, 3.5, 3.1, 2.9])


@pytest.fixture
def bootstrap_outlier_data():
    """
    包含离群点的数据
    用于测试 Bootstrap 对离群点的敏感性
    """
    np.random.seed(42)
    base = np.random.normal(loc=50, scale=10, size=95)
    outliers = np.array([150, 160, 170, 180, 190])  # 极端离群点
    return np.concatenate([base, outliers])


# =============================================================================
# 置换检验测试 Fixtures
# =============================================================================

@pytest.fixture
def permutation_equal_groups():
    """
    无显著差异的两组数据
    用于测试置换检验在原假设成立时的行为
    """
    np.random.seed(42)
    group_a = np.random.normal(loc=100, scale=15, size=50)
    group_b = np.random.normal(loc=100, scale=15, size=50)
    return {'group_a': group_a, 'group_b': group_b}


@pytest.fixture
def permutation_different_groups():
    """
    有显著差异的两组数据
    用于测试置换检验在备择假设成立时的行为
    """
    np.random.seed(42)
    group_a = np.random.normal(loc=100, scale=15, size=50)
    group_b = np.random.normal(loc=115, scale=15, size=50)  # 均值高 15
    return {'group_a': group_a, 'group_b': group_b}


@pytest.fixture
def permutation_small_difference():
    """
    小差异的两组数据
    用于测试置换检验对小效应的敏感性
    """
    np.random.seed(42)
    group_a = np.random.normal(loc=100, scale=15, size=50)
    group_b = np.random.normal(loc=105, scale=15, size=50)  # 均值高 5
    return {'group_a': group_a, 'group_b': group_b}


@pytest.fixture
def permutation_unequal_sizes():
    """
    样本量不等的两组数据
    用于测试置换检验对不等样本量的处理
    """
    np.random.seed(42)
    group_a = np.random.normal(loc=100, scale=15, size=30)
    group_b = np.random.normal(loc=115, scale=15, size=70)
    return {'group_a': group_a, 'group_b': group_b}


@pytest.fixture
def permutation_skewed_groups():
    """
    偏态分布的两组数据
    用于测试置换检验在非正态数据上的表现
    """
    np.random.seed(42)
    group_a = np.random.exponential(scale=30, size=50)
    group_b = np.random.exponential(scale=40, size=50)
    return {'group_a': group_a, 'group_b': group_b}


# =============================================================================
# 边界情况 Fixtures
# =============================================================================

@pytest.fixture
def empty_data():
    """空数据"""
    return np.array([])


@pytest.fixture
def single_value_data():
    """单值数据"""
    return np.array([42.0])


@pytest.fixture
def two_values_data():
    """两值数据（最小可计算 CI）"""
    return np.array([3.0, 5.0])


@pytest.fixture
def constant_data():
    """常量数据（方差为0）"""
    return np.array([5.0, 5.0, 5.0, 5.0, 5.0])


@pytest.fixture
def nan_data():
    """包含 NaN 的数据"""
    np.random.seed(42)
    data = np.random.normal(loc=100, scale=15, size=50)
    data[10] = np.nan
    data[25] = np.nan
    return data


# =============================================================================
# StatLab 测试 Fixtures
# =============================================================================

@pytest.fixture
def statlab_user_spending():
    """StatLab 项目用例：用户消费金额数据"""
    np.random.seed(42)
    # 模拟不同用户群组的消费数据
    new_users = np.random.normal(loc=50, scale=20, size=200)
    active_users = np.random.normal(loc=120, scale=40, size=200)
    vip_users = np.random.normal(loc=300, scale=80, size=200)
    return {
        'new': new_users,
        'active': active_users,
        'vip': vip_users
    }


@pytest.fixture
def statlab_conversion_rates():
    """StatLab 项目用例：A/B 测试转化率数据"""
    np.random.seed(42)
    # A 组：10% 转化率
    conversions_a = np.random.binomial(1, 0.10, 1000)
    # B 组：12% 转化率
    conversions_b = np.random.binomial(1, 0.12, 1000)
    return {'group_a': conversions_a, 'group_b': conversions_b}


# =============================================================================
# 验证工具 Fixtures
# =============================================================================

@pytest.fixture
def known_ci_data():
    """
    已知理论 CI 的数据
    用于验证 CI 计算的正确性
    """
    np.random.seed(42)
    # 标准正态分布，n=100
    # 均值 ≈ 0, SD ≈ 1, SE ≈ 0.1
    # 95% CI ≈ mean ± 1.984 * SE (t 分布, df=99)
    return np.random.normal(loc=0, scale=1, size=100)


@pytest.fixture
def ci_comparison_tolerance():
    """
    CI 比较的容差
    用于允许不同方法之间的合理差异
    """
    return {
        'relative': 0.10,  # 10% 相对误差
        'absolute': 0.50   # 0.5 绝对误差
    }

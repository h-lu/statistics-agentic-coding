"""
Week 05 共享 Fixtures

提供测试用的共享数据和工具函数，用于概率与模拟相关测试。
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
def rng() -> np.random.Generator:
    """创建可复现的随机数生成器"""
    return np.random.default_rng(42)


@pytest.fixture
def disease_test_data() -> dict[str, Any]:
    """
    创建医疗检测问题的测试数据

    参数：
        population: 总人口
        prevalence: 发病率
        sensitivity: 敏感性 P(阳性|患病)
        specificity: 特异性 P(阴性|健康)
    """
    return {
        'population': 100000,
        'prevalence': 0.01,
        'sensitivity': 0.99,
        'specificity': 0.99
    }


@pytest.fixture
def normal_sample() -> np.ndarray:
    """创建正态分布样本"""
    np.random.seed(42)
    return np.random.normal(100, 15, 1000)


@pytest.fixture
def skewed_sample() -> np.ndarray:
    """创建右偏样本（对数正态分布）"""
    np.random.seed(42)
    return np.random.lognormal(4, 0.5, 1000)


@pytest.fixture
def binomial_sample() -> np.ndarray:
    """创建二项分布样本"""
    np.random.seed(42)
    return np.random.binomial(n=100, p=0.05, size=1000)


@pytest.fixture
def poisson_sample() -> np.ndarray:
    """创建泊松分布样本"""
    np.random.seed(42)
    return np.random.poisson(lam=3, size=1000)


@pytest.fixture
def two_groups() -> dict[str, np.ndarray]:
    """创建两组数据用于均值差异测试"""
    np.random.seed(42)
    group1 = np.random.lognormal(8, 0.6, 50)
    group2 = np.random.lognormal(6.5, 0.5, 200)
    return {'group1': group1, 'group2': group2}


@pytest.fixture
def correlated_data() -> dict[str, np.ndarray]:
    """创建相关数据"""
    np.random.seed(42)
    x = np.random.normal(50, 10, 100)
    # y 与 x 相关
    y = x * 0.8 + np.random.normal(0, 5, 100)
    return {'x': x, 'y': y}


# =============================================================================
# Bootstrap Fixtures
# =============================================================================

@pytest.fixture
def bootstrap_sample() -> np.ndarray:
    """创建用于 Bootstrap 测试的样本"""
    np.random.seed(42)
    return np.random.lognormal(7, 0.8, 100)


@pytest.fixture
def bootstrap_results() -> dict[str, Any]:
    """创建预设的 Bootstrap 结果（用于测试 CI 计算）"""
    return {
        'bootstrap_means': np.array([100, 102, 98, 101, 99, 100, 101, 97, 103, 99]),
        'observed_mean': 100.0,
        'ci_low': 97.0,
        'ci_high': 103.0,
        'se': 2.0
    }


# =============================================================================
# CLT Fixtures
# =============================================================================

@pytest.fixture
def exponential_population() -> np.ndarray:
    """创建指数分布总体（用于 CLT 测试）"""
    np.random.seed(42)
    return np.random.exponential(scale=10, size=100000)


@pytest.fixture
def clt_sample_sizes() -> list[int]:
    """CLT 测试的样本量序列"""
    return [5, 10, 30, 100]


# =============================================================================
# Edge Case Fixtures
# =============================================================================

@pytest.fixture
def empty_array() -> np.ndarray:
    """空数组"""
    return np.array([])


@pytest.fixture
def single_value_array() -> np.ndarray:
    """单值数组"""
    return np.array([5.0])


@pytest.fixture
def constant_array() -> np.ndarray:
    """常量数组（所有值相同）"""
    return np.array([5.0, 5.0, 5.0, 5.0, 5.0])


@pytest.fixture
def extreme_prevalence_data() -> dict[str, Any]:
    """极端发病率数据（用于边界测试）"""
    return {
        'ultra_rare': 0.0001,    # 极罕见病
        'rare': 0.01,            # 罕见病
        'common': 0.1,            # 常见病
        'very_common': 0.5          # 非常常见
    }


# =============================================================================
# Test Result Fixtures
# =============================================================================

@pytest.fixture
def tolerance() -> float:
    """浮点数比较容差"""
    return 0.01


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """创建临时输出目录"""
    output_dir = tmp_path / "output"
    output_dir.mkdir(exist_ok=True)
    return output_dir


# =============================================================================
# Utility Functions
# =============================================================================

def assert_ci_contains(
    ci_low: float,
    ci_high: float,
    value: float,
    msg: str | None = None
) -> None:
    """断言值在置信区间内"""
    assert ci_low <= value <= ci_high, msg or f"{value} 不在 CI [{ci_low}, {ci_high}] 内"


def assert_ci_excludes(
    ci_low: float,
    ci_high: float,
    value: float,
    msg: str | None = None
) -> None:
    """断言值不在置信区间内"""
    assert not (ci_low <= value <= ci_high), msg or f"{value} 在 CI [{ci_low}, {ci_high}] 内"


def simulate_clt(
    population: np.ndarray,
    n: int,
    n_simulations: int = 1000,
    seed: int = 42
) -> np.ndarray:
    """
    CLT 模拟辅助函数

    参数：
        population: 总体数据
        n: 样本量
        n_simulations: 模拟次数
        seed: 随机种子

    返回：
        样本均值数组
    """
    rng = np.random.default_rng(seed)
    sample_means = np.array([
        np.mean(rng.choice(population, n, replace=False))
        for _ in range(n_simulations)
    ])
    return sample_means

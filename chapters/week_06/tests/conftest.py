"""
Week 06 共享 Fixtures

提供测试用的共享数据和工具函数。

测试覆盖：
- p 值理解与计算
- t 检验（单样本、双样本、配对）
- 卡方检验
- 效应量（Cohen's d, 风险差等）
- 前提假设检查（正态性、方差齐性）
- AI 结论审查
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
# p 值理解测试 Fixtures
# =============================================================================

@pytest.fixture
def null_hypothesis_data():
    """
    创建符合原假设的数据（无真实差异）
    用于测试 p 值在原假设成立时的分布
    """
    np.random.seed(42)
    # 两组数据来自同一分布（无真实差异）
    group_a = np.random.normal(loc=100, scale=15, size=100)
    group_b = np.random.normal(loc=100, scale=15, size=100)
    return group_a, group_b


@pytest.fixture
def alternative_hypothesis_data():
    """
    创建符合备择假设的数据（有真实差异）
    用于测试 p 值在有真实差异时的表现
    """
    np.random.seed(42)
    # A 组均值 100，B 组均值 110（有 10 的差异）
    group_a = np.random.normal(loc=100, scale=15, size=100)
    group_b = np.random.normal(loc=110, scale=15, size=100)
    return group_a, group_b


# =============================================================================
# t 检验测试 Fixtures
# =============================================================================

@pytest.fixture
def normal_two_groups():
    """
    创建符合正态分布的两组数据
    用于测试标准 t 检验
    """
    np.random.seed(42)
    group_a = np.random.normal(loc=50, scale=10, size=120)
    group_b = np.random.normal(loc=53, scale=10, size=120)
    return group_a, group_b


@pytest.fixture
def small_sample_groups():
    """
    创建小样本数据
    用于测试 t 检验在小样本情况下的表现
    """
    np.random.seed(42)
    group_a = np.array([45, 48, 52, 50, 47])
    group_b = np.array([51, 55, 53, 58, 52])
    return group_a, group_b


@pytest.fixture
def binary_conversion_data():
    """
    创建二元转化数据（0/1）
    用于测试比例检验
    """
    np.random.seed(42)
    # A 渠道：12% 转化率
    conversions_a = np.array([1] * 144 + [0] * (1200 - 144))
    # B 渠道：9% 转化率
    conversions_b = np.array([1] * 108 + [0] * (1200 - 108))
    return conversions_a, conversions_b


@pytest.fixture
def paired_data():
    """
    创建配对数据
    用于测试配对 t 检验
    """
    np.random.seed(42)
    before = np.random.normal(loc=100, scale=10, size=50)
    # 每个对象治疗后提高约 5 分
    after = before + np.random.normal(loc=5, scale=3, size=50)
    return before, after


# =============================================================================
# 卡方检验测试 Fixtures
# =============================================================================

@pytest.fixture
def contingency_table_independent():
    """
    创建独立的列联表数据
    用于测试卡方检验（原假设成立）
    """
    return pd.DataFrame({
        'A渠道': [300, 200],
        'B渠道': [250, 250]
    }, index=['新用户', '老用户'])


@pytest.fixture
def contingency_table_dependent():
    """
    创建有关联的列联表数据
    用于测试卡方检验（备择假设成立）
    """
    return pd.DataFrame({
        'A渠道': [400, 100],
        'B渠道': [150, 350]
    }, index=['新用户', '老用户'])


# =============================================================================
# 效应量测试 Fixtures
# =============================================================================

@pytest.fixture
def large_effect_data():
    """
    创建大效应量的数据
    Cohen's d > 0.8
    """
    np.random.seed(42)
    group_a = np.random.normal(loc=100, scale=10, size=100)
    group_b = np.random.normal(loc=115, scale=10, size=100)  # 差异 1.5 个标准差
    return group_a, group_b


@pytest.fixture
def medium_effect_data():
    """
    创建中等效应量的数据
    0.2 <= Cohen's d < 0.8
    """
    np.random.seed(42)
    group_a = np.random.normal(loc=100, scale=10, size=100)
    group_b = np.random.normal(loc=105, scale=10, size=100)  # 差异 0.5 个标准差
    return group_a, group_b


@pytest.fixture
def small_effect_data():
    """
    创建小效应量的数据
    Cohen's d < 0.2
    """
    np.random.seed(42)
    group_a = np.random.normal(loc=100, scale=10, size=100)
    group_b = np.random.normal(loc=101.5, scale=10, size=100)  # 差异 0.15 个标准差
    return group_a, group_b


@pytest.fixture
def no_effect_data():
    """
    创建无效应的数据
    Cohen's d ≈ 0
    """
    np.random.seed(42)
    group_a = np.random.normal(loc=100, scale=10, size=100)
    group_b = np.random.normal(loc=100, scale=10, size=100)
    return group_a, group_b


# =============================================================================
# 前提假设检查测试 Fixtures
# =============================================================================

@pytest.fixture
def normal_data():
    """创建正态分布数据"""
    np.random.seed(42)
    return np.random.normal(loc=100, scale=15, size=200)


@pytest.fixture
def skewed_data():
    """创建偏态分布数据（指数分布）"""
    np.random.seed(42)
    return np.random.exponential(scale=30, size=200)


@pytest.fixture
def equal_variance_groups():
    """创建方差相等的数据"""
    np.random.seed(42)
    group_a = np.random.normal(loc=100, scale=10, size=100)
    group_b = np.random.normal(loc=105, scale=10, size=100)
    return group_a, group_b


@pytest.fixture
def unequal_variance_groups():
    """创建方差不等的数据"""
    np.random.seed(42)
    group_a = np.random.normal(loc=100, scale=5, size=100)
    group_b = np.random.normal(loc=105, scale=20, size=100)
    return group_a, group_b


@pytest.fixture
def constant_data():
    """创建常数数据（用于测试边界情况）"""
    np.random.seed(42)
    return np.array([50, 50, 50, 50, 50, 50, 50, 50, 50, 50])


# =============================================================================
# 边界情况 Fixtures
# =============================================================================

@pytest.fixture
def empty_data():
    """创建空数组"""
    return np.array([])


@pytest.fixture
def single_value_data():
    """创建单值数组"""
    return np.array([42])


@pytest.fixture
def two_values_data():
    """创建两值数组（最小可计算标准差）"""
    return np.array([40, 50])


# =============================================================================
# AI 审查测试 Fixtures
# =============================================================================

@pytest.fixture
def ai_good_report():
    """创建一份合格的 AI 生成的检验报告"""
    return {
        'test_method': 'proportions_ztest',
        'group_a_rate': 0.12,
        'group_b_rate': 0.09,
        'difference': 0.03,
        'p_value': 0.023,
        'confidence_interval': (0.004, 0.056),
        'effect_size': {'cohens_h': 0.09, 'interpretation': 'small'},
        'assumptions_checked': True,
        'normality_tested': False,  # 二元数据不需要
        'independence': '独立随机分配',
        'conclusion': '有证据表明 A 渠道转化率高于 B 渠道，但效应量较小'
    }


@pytest.fixture
def ai_bad_report():
    """创建一份有问题的 AI 生成的检验报告"""
    return {
        'test_method': 'ttest_ind',  # 对二元数据不合适
        'group_a_rate': 0.12,
        'group_b_rate': 0.09,
        'difference': 0.03,
        'p_value': 0.023,
        # 缺少置信区间
        # 缺少效应量
        'assumptions_checked': False,
        'conclusion': 'A 渠道显著优于 B 渠道，建议全面切换'  # 过度解读
    }


@pytest.fixture
def multiple_hypotheses_results():
    """创建多重检验结果"""
    return [
        {'hypothesis': 'A vs B 转化率', 'p_value': 0.023, 'significant': True},
        {'hypothesis': 'A vs C 转化率', 'p_value': 0.15, 'significant': False},
        {'hypothesis': 'B vs C 转化率', 'p_value': 0.08, 'significant': False},
        {'hypothesis': '新 vs 老 用户金额', 'p_value': 0.031, 'significant': True},
        {'hypothesis': '不同地区活跃度', 'p_value': 0.12, 'significant': False}
    ]

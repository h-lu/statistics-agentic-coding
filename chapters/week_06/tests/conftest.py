"""
Week 06 共享 Fixtures

提供测试用的共享数据和工具函数，用于假设检验相关测试。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

# 添加 starter_code 到导入路径
sys.path.insert(0, str(Path(__file__).parent.parent / "starter_code"))


# =============================================================================
# Data Fixtures - 基础测试数据
# =============================================================================

@pytest.fixture
def sample_data_normal():
    """创建正态分布的样本数据"""
    np.random.seed(42)
    return np.random.normal(loc=100, scale=15, size=100)


@pytest.fixture
def sample_data_two_groups():
    """创建两组独立样本数据"""
    np.random.seed(42)
    group1 = np.random.normal(loc=100, scale=15, size=50)
    group2 = np.random.normal(loc=105, scale=15, size=50)
    return {'group1': group1, 'group2': group2}


@pytest.fixture
def sample_data_paired():
    """创建配对样本数据（处理前/后）"""
    np.random.seed(42)
    before = np.random.normal(loc=100, scale=10, size=30)
    # 处理后均值增加 5，相关
    after = before + np.random.normal(loc=5, scale=5, size=30)
    return {'before': before, 'after': after}


@pytest.fixture
def sample_data_non_normal():
    """创建非正态分布数据（指数分布）"""
    np.random.seed(42)
    return np.random.exponential(scale=50, size=100)


@pytest.fixture
def sample_data_unequal_variance():
    """创建方差不等的两组数据"""
    np.random.seed(42)
    group1 = np.random.normal(loc=100, scale=10, size=50)   # 小方差
    group2 = np.random.normal(loc=100, scale=30, size=50)  # 大方差
    return {'group1': group1, 'group2': group2}


# =============================================================================
# A/B 测试场景数据
# =============================================================================

@pytest.fixture
def ab_test_data():
    """模拟 A/B 测试数据（对照组 vs 实验组）"""
    np.random.seed(42)
    control = np.random.normal(loc=100, scale=15, size=500)
    treatment = np.random.normal(loc=105, scale=15, size=500)
    return {
        'control': control,
        'treatment': treatment,
        'n_per_group': 500,
        'true_effect': 5
    }


@pytest.fixture
def ab_test_no_effect():
    """模拟无效应的 A/B 测试（H0 为真）"""
    np.random.seed(42)
    control = np.random.normal(loc=100, scale=15, size=500)
    treatment = np.random.normal(loc=100, scale=15, size=500)
    return {
        'control': control,
        'treatment': treatment,
        'n_per_group': 500,
        'true_effect': 0
    }


# =============================================================================
# Edge Case Fixtures
# =============================================================================

@pytest.fixture
def empty_array():
    """创建空数组"""
    return np.array([])


@pytest.fixture
def single_value():
    """创建单值数组"""
    return np.array([100])


@pytest.fixture
def tiny_sample():
    """创建极小样本（n=2）"""
    np.random.seed(42)
    return np.random.normal(loc=100, scale=15, size=2)


@pytest.fixture
def large_sample():
    """创建大样本（n=10000）"""
    np.random.seed(42)
    return np.random.normal(loc=100, scale=15, size=10000)


@pytest.fixture
def data_with_outliers():
    """包含极端异常值的数据"""
    np.random.seed(42)
    data = np.random.normal(loc=100, scale=15, size=50)
    # 添加极端异常值
    data_with_outliers = np.append(data, [500, 600, 700])
    return data_with_outliers


@pytest.fixture
def data_with_nan():
    """包含缺失值的数据"""
    np.random.seed(42)
    data = np.random.normal(loc=100, scale=15, size=100)
    data[10:15] = np.nan
    return data


@pytest.fixture
def constant_data():
    """常数列数据（所有值相同）"""
    return np.array([100] * 50)


# =============================================================================
# AI 报告示例
# =============================================================================

@pytest.fixture
def good_ai_report():
    """一份合格的 AI 生成报告"""
    return """
统计检验报告：

## 假设设定
- H0（原假设）：实验组与对照组的活跃度均值相等（μ_exp = μ_ctrl）
- H1（备择假设）：实验组活跃度高于对照组（μ_exp > μ_ctrl，单尾检验）

## 前提假设检查
- 正态性：Shapiro-Wilk 检验 p_ctrl=0.12, p_exp=0.08（> 0.05，正态性假设满足）
- 方差齐性：Levene 检验 p=0.21（> 0.05，方差齐性假设满足）

## 检验结果
- t 统计量：t(998) = 2.15
- p 值（单尾）：p = 0.016
- 均值差异：5.2 分，95% CI [1.8, 8.6]
- Cohen's d 效应量：d = 0.21（小效应）
"""


@pytest.fixture
def bad_ai_report():
    """一份有问题的 AI 生成报告（包含多个错误）"""
    return """
统计检验报告：

我们对实验组和对照组进行了 t 检验，结果 t=2.15, p=0.03。

结论：
1. 新功能显著提升了用户活跃度（H0 为真的概率是 3%）。
2. 两组均值差异为 5.2 分。

建议：
- 上线新功能，因为效果显著。
"""


@pytest.fixture
def bad_ai_report_p_hacking():
    """展示 p-hacking 痕迹的报告"""
    return """
统计检验报告：

我们尝试了多种分组方式和检验方法：
- 按年龄分组：不显著
- 按性别分组：不显著
- 按城市分组：不显著
- 按年龄和性别交叉分组：显著（p=0.03）

结论：
- 年龄和性别交互效应显著影响用户活跃度。
"""


# =============================================================================
# 假设示例
# =============================================================================

@pytest.fixture
def valid_hypothesis():
    """有效的假设陈述"""
    return {
        'H0': '两组均值相等（μ1 = μ2）',
        'H1': '第一组均值大于第二组（μ1 > μ2）',
        'test_type': 'one_tailed'
    }


@pytest.fixture
def invalid_hypothesis_missing_h1():
    """缺少 H1 的无效假设"""
    return {
        'H0': '两组均值相等',
        'test_type': 'two_tailed'
    }


@pytest.fixture
def invalid_hypothesis_empty_h0():
    """H0 为空的无效假设"""
    return {
        'H0': '',
        'H1': '有差异'
    }


# =============================================================================
# Path Fixtures
# =============================================================================

@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """创建临时输出目录"""
    output_dir = tmp_path / "output"
    output_dir.mkdir(exist_ok=True)
    return output_dir

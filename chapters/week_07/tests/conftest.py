"""
Week 07 共享 Fixtures

提供测试用的共享数据和工具函数，用于多组比较相关测试。
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
# ANOVA 数据 Fixtures
# =============================================================================

@pytest.fixture
def sample_three_groups():
    """创建三组数据（小差异）"""
    np.random.seed(42)
    group1 = np.random.normal(loc=100, scale=15, size=50)
    group2 = np.random.normal(loc=102, scale=15, size=50)
    group3 = np.random.normal(loc=98, scale=15, size=50)
    return {'group1': group1, 'group2': group2, 'group3': group3}


@pytest.fixture
def sample_five_groups_with_effect():
    """创建五组数据（中等效应）"""
    np.random.seed(42)
    return {
        'group1': np.random.normal(loc=280, scale=50, size=100),
        'group2': np.random.normal(loc=310, scale=50, size=100),
        'group3': np.random.normal(loc=270, scale=50, size=100),
        'group4': np.random.normal(loc=320, scale=50, size=100),
        'group5': np.random.normal(loc=290, scale=50, size=100),
    }


@pytest.fixture
def sample_groups_no_difference():
    """创建无差异的各组数据（H0 为真）"""
    np.random.seed(42)
    return {
        'group1': np.random.normal(loc=100, scale=15, size=50),
        'group2': np.random.normal(loc=100, scale=15, size=50),
        'group3': np.random.normal(loc=100, scale=15, size=50),
    }


@pytest.fixture
def sample_groups_large_effect():
    """创建大效应的各组数据"""
    np.random.seed(42)
    return {
        'group1': np.random.normal(loc=100, scale=15, size=50),
        'group2': np.random.normal(loc=130, scale=15, size=50),
        'group3': np.random.normal(loc=70, scale=15, size=50),
    }


@pytest.fixture
def sample_groups_unequal_variance():
    """创建方差不等的各组数据"""
    np.random.seed(42)
    return {
        'group1': np.random.normal(loc=100, scale=10, size=50),
        'group2': np.random.normal(loc=105, scale=30, size=50),
        'group3': np.random.normal(loc=98, scale=10, size=50),
    }


# =============================================================================
# 卡方检验数据 Fixtures
# =============================================================================

@pytest.fixture
def contingency_table_independent():
    """创建独立的列联表（H0 为真）"""
    np.random.seed(42)
    return pd.DataFrame(
        np.array([
            [25, 25, 25, 25],
            [25, 25, 25, 25],
            [25, 25, 25, 25],
        ]),
        index=['A', 'B', 'C'],
        columns=['X', 'Y', 'Z', 'W']
    )


@pytest.fixture
def contingency_table_associated():
    """创建有关联的列联表（H1 为真）"""
    return pd.DataFrame(
        np.array([
            [45, 30, 18, 7],
            [38, 32, 22, 8],
            [52, 28, 15, 5],
            [35, 35, 20, 10],
            [40, 30, 20, 10]
        ]),
        index=['北京', '上海', '广州', '深圳', '杭州'],
        columns=['普通', '银卡', '金卡', '钻石']
    )


@pytest.fixture
def contingency_table_2x2():
    """创建 2x2 列联表"""
    return pd.DataFrame(
        np.array([
            [50, 30],
            [20, 60]
        ]),
        index=['Control', 'Treatment'],
        columns=['Success', 'Failure']
    )


# =============================================================================
# 多重比较相关 Fixtures
# =============================================================================

@pytest.fixture
def multiple_test_p_values():
    """创建多次检验的 p 值列表"""
    return [0.001, 0.015, 0.032, 0.045, 0.089, 0.12, 0.21, 0.35, 0.48, 0.62]


@pytest.fixture
def tukey_hsd_results():
    """模拟 Tukey HSD 结果"""
    return pd.DataFrame({
        'group1': ['A', 'A', 'A', 'B', 'B', 'C'],
        'group2': ['B', 'C', 'D', 'C', 'D', 'D'],
        'meandiff': [5.2, -3.1, 8.5, -8.3, 3.3, 11.6],
        'p-adj': [0.042, 0.231, 0.003, 0.001, 0.156, 0.0001],
        'lower': [0.1, -8.5, 2.3, -14.2, -1.5, 5.8],
        'upper': [10.3, 2.3, 14.7, -2.4, 8.1, 17.4],
        'reject': [True, False, True, True, False, True]
    })


# =============================================================================
# AI 报告示例 Fixtures
# =============================================================================

@pytest.fixture
def good_anova_report():
    """一份合格的 ANOVA 报告"""
    return """
## ANOVA 结果

**假设设定**：
- H0：所有城市的平均消费相等（μ_北京 = μ_上海 = μ_广州 = μ_深圳 = μ_杭州）
- H1：至少有一对城市的平均消费不等

**前提假设检查**：
- 正态性：Shapiro-Wilk 检验各城市 p 值均 > 0.05（正态性假设满足）
- 方差齐性：Levene 检验 p=0.21（> 0.05，方差齐性假设满足）
- 独立性：用户随机抽样，各城市互不干扰

**检验结果**：
- F 统计量：F(4, 495) = 8.52
- p 值：p = 0.002
- η² 效应量：η² = 0.064（中等效应，6.4% 的变异由城市解释）
- 决策：拒绝 H0，至少有一对城市均值不同

**事后检验（Tukey HSD）**：
使用 Tukey HSD 方法进行配对比较，校正后的 p 值显示上海 vs 广州、深圳 vs 广州 等城市对显著不同。
"""


@pytest.fixture
def bad_anova_report_overinterpretation():
    """存在 ANOVA 过度解释问题的报告"""
    return """
多组比较报告：

我们对 5 个城市的用户消费进行了 ANOVA 分析，结果 F=8.52, p=0.002。

结论：
1. 上海和深圳的用户消费显著高于其他城市。
2. ANOVA 显示城市对消费有显著影响。
3. 建议在深圳和上海加大营销投入。
"""


@pytest.fixture
def bad_anova_report_no_correction():
    """未校正多重比较的报告"""
    return """
多组比较报告：

我们对 5 个城市进行了两两 t 检验，共进行 10 次比较。

结果：
- 北京 vs 上海：p=0.03（显著）
- 上海 vs 广州：p=0.01（显著）
- 深圳 vs 杭州：p=0.04（显著）

结论：多个城市对之间存在显著差异。
"""


@pytest.fixture
def bad_chisquare_report_causation():
    """混淆相关与因果的卡方检验报告"""
    return """
卡方检验结果：

我们对城市与用户等级进行了卡方检验，结果显示 χ²=12.34, p=0.002。

结论：城市影响用户等级（V=0.18），深圳的用户等级更高。
建议：在深圳加大营销投入以提升用户等级。
"""


# =============================================================================
# 边界情况 Fixtures
# =============================================================================

@pytest.fixture
def empty_group():
    """空组数据"""
    return np.array([])


@pytest.fixture
def single_value_groups():
    """单值各组数据"""
    return {
        'group1': np.array([100]),
        'group2': np.array([105]),
        'group3': np.array([98]),
    }


@pytest.fixture
def tiny_groups():
    """极小样本各组（n=2）"""
    np.random.seed(42)
    return {
        'group1': np.array([1, 2]),
        'group2': np.array([3, 4]),
        'group3': np.array([2, 3]),
    }


@pytest.fixture
def constant_groups():
    """常数各组数据"""
    return {
        'group1': np.array([100] * 50),
        'group2': np.array([100] * 50),
        'group3': np.array([100] * 50),
    }


@pytest.fixture
def groups_with_outliers():
    """包含异常值的各组数据"""
    np.random.seed(42)
    base_data = np.random.normal(loc=100, scale=15, size=50)
    data_with_outliers = np.concatenate([base_data, [500, 600]])
    return {
        'group1': data_with_outliers,
        'group2': np.random.normal(loc=105, scale=15, size=50),
        'group3': np.random.normal(loc=98, scale=15, size=50),
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

"""
Week 07 共享 Fixtures

提供测试用的共享数据和工具函数。

测试覆盖：
- 多重比较问题（FWER）
- ANOVA（方差分析）
- F 统计量
- 事后比较（Tukey HSD）
- 多重比较校正（Bonferroni、FDR）
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
# 多重比较问题测试 Fixtures
# =============================================================================

@pytest.fixture
def single_test_result():
    """单个检验结果（用于 m=1 时 FWER=α）"""
    return {'p_value': 0.03, 'test': 'A vs B'}


@pytest.fixture
def multiple_test_results():
    """多个检验结果（用于测试 FWER 计算）"""
    return [
        {'p_value': 0.023, 'test': 'A vs B'},
        {'p_value': 0.15, 'test': 'A vs C'},
        {'p_value': 0.08, 'test': 'B vs C'},
        {'p_value': 0.031, 'test': 'A vs D'},
        {'p_value': 0.12, 'test': 'B vs D'}
    ]


@pytest.fixture
def many_test_results():
    """大量检验结果（用于测试假阳性累积）"""
    np.random.seed(42)
    # 20 个 p 值，均匀分布（原假设成立时）
    return [
        {'p_value': float(np.random.uniform(0, 1)), 'test': f'Test_{i}'}
        for i in range(20)
    ]


@pytest.fixture
def mixed_significance_results():
    """混合显著性的检验结果（部分显著，部分不显著）"""
    return [
        {'p_value': 0.001, 'test': 'A vs B'},
        {'p_value': 0.003, 'test': 'A vs C'},
        {'p_value': 0.005, 'test': 'A vs D'},
        {'p_value': 0.008, 'test': 'A vs E'},
        {'p_value': 0.010, 'test': 'B vs C'},
        {'p_value': 0.15, 'test': 'B vs D'},
        {'p_value': 0.25, 'test': 'B vs E'},
        {'p_value': 0.45, 'test': 'C vs D'}
    ]


# =============================================================================
# ANOVA 测试 Fixtures
# =============================================================================

@pytest.fixture
def four_groups_no_difference():
    """
    创建四组无差异的数据（原假设成立）
    用于测试 ANOVA 在原假设成立时的行为
    """
    np.random.seed(42)
    group_a = np.random.normal(loc=100, scale=15, size=50)
    group_b = np.random.normal(loc=100, scale=15, size=50)
    group_c = np.random.normal(loc=100, scale=15, size=50)
    group_d = np.random.normal(loc=100, scale=15, size=50)
    return {
        'A': group_a,
        'B': group_b,
        'C': group_c,
        'D': group_d
    }


@pytest.fixture
def four_groups_with_difference():
    """
    创建四组有差异的数据（备择假设成立）
    用于测试 ANOVA 在有真实差异时的行为
    D 组均值比其他组高
    """
    np.random.seed(42)
    group_a = np.random.normal(loc=100, scale=15, size=50)
    group_b = np.random.normal(loc=100, scale=15, size=50)
    group_c = np.random.normal(loc=100, scale=15, size=50)
    group_d = np.random.normal(loc=115, scale=15, size=50)  # 均值高 15
    return {
        'A': group_a,
        'B': group_b,
        'C': group_c,
        'D': group_d
    }


@pytest.fixture
def four_groups_small_difference():
    """
    创建四组小差异的数据（效应量小）
    用于测试 ANOVA 对小效应的敏感性
    """
    np.random.seed(42)
    group_a = np.random.normal(loc=100, scale=15, size=50)
    group_b = np.random.normal(loc=102, scale=15, size=50)
    group_c = np.random.normal(loc=101, scale=15, size=50)
    group_d = np.random.normal(loc=103, scale=15, size=50)
    return {
        'A': group_a,
        'B': group_b,
        'C': group_c,
        'D': group_d
    }


@pytest.fixture
def binary_conversion_groups():
    """
    创建二元转化数据（0/1）
    用于测试 ANOVA 在比例数据上的表现
    """
    np.random.seed(42)
    conversions_a = np.random.binomial(1, 0.10, 500)
    conversions_b = np.random.binomial(1, 0.10, 500)
    conversions_c = np.random.binomial(1, 0.10, 500)
    conversions_d = np.random.binomial(1, 0.12, 500)
    return {
        'A': conversions_a,
        'B': conversions_b,
        'C': conversions_c,
        'D': conversions_d
    }


@pytest.fixture
def unequal_variance_groups():
    """
    创建方差不齐的数据
    用于测试 ANOVA 前提假设检查
    """
    np.random.seed(42)
    group_a = np.random.normal(loc=100, scale=5, size=50)
    group_b = np.random.normal(loc=100, scale=5, size=50)
    group_c = np.random.normal(loc=100, scale=20, size=50)  # 方差大
    group_d = np.random.normal(loc=100, scale=20, size=50)  # 方差大
    return {
        'A': group_a,
        'B': group_b,
        'C': group_c,
        'D': group_d
    }


@pytest.fixture
def skewed_groups():
    """
    创建偏态分布的数据
    用于测试 ANOVA 前提假设检查
    """
    np.random.seed(42)
    group_a = np.random.exponential(scale=30, size=50)
    group_b = np.random.exponential(scale=35, size=50)
    group_c = np.random.exponential(scale=32, size=50)
    group_d = np.random.exponential(scale=40, size=50)
    return {
        'A': group_a,
        'B': group_b,
        'C': group_c,
        'D': group_d
    }


@pytest.fixture
def anova_dataframe():
    """创建适合 ANOVA 的 DataFrame 格式数据"""
    np.random.seed(42)
    data = []
    for i, group in enumerate(['A', 'B', 'C', 'D']):
        if group == 'D':
            values = np.random.normal(loc=115, scale=15, size=50)
        else:
            values = np.random.normal(loc=100, scale=15, size=50)
        for v in values:
            data.append({'group': group, 'value': v})
    return pd.DataFrame(data)


# =============================================================================
# 事后比较测试 Fixtures
# =============================================================================

@pytest.fixture
def tukey_significant_pairs():
    """
    创建有显著差异对的数据
    用于测试 Tukey HSD 能正确识别差异
    """
    np.random.seed(42)
    group_a = np.random.normal(loc=100, scale=10, size=100)
    group_b = np.random.normal(loc=100, scale=10, size=100)
    group_c = np.random.normal(loc=100, scale=10, size=100)
    group_d = np.random.normal(loc=120, scale=10, size=100)  # 明显差异
    return {
        'A': group_a,
        'B': group_b,
        'C': group_c,
        'D': group_d
    }


@pytest.fixture
def tukey_no_significant_pairs():
    """
    创建无显著差异对的数据
    用于测试 Tukey HSD 的正确不拒绝
    """
    np.random.seed(42)
    groups = {}
    for letter in ['A', 'B', 'C', 'D']:
        groups[letter] = np.random.normal(loc=100, scale=15, size=50)
    return groups


@pytest.fixture
def anova_significant_tukey_not():
    """
    创建 ANOVA 显著但 Tukey HSD 无显著对的数据
    用于测试这种边界情况
    """
    np.random.seed(42)
    # 多组小差异，联合显著但两两不显著
    group_a = np.random.normal(loc=100, scale=15, size=50)
    group_b = np.random.normal(loc=102, scale=15, size=50)
    group_c = np.random.normal(loc=104, scale=15, size=50)
    group_d = np.random.normal(loc=106, scale=15, size=50)
    return {
        'A': group_a,
        'B': group_b,
        'C': group_c,
        'D': group_d
    }


# =============================================================================
# 校正方法测试 Fixtures
# =============================================================================

@pytest.fixture
def p_values_for_correction():
    """用于校正的 p 值列表"""
    return np.array([0.001, 0.003, 0.005, 0.008, 0.010, 0.15, 0.25, 0.45])


@pytest.fixture
def single_p_value():
    """单个 p 值（用于测试 m=1 时校正无效）"""
    return np.array([0.03])


@pytest.fixture
def many_p_values():
    """大量 p 值（用于测试 Bonferroni 过于保守）"""
    np.random.seed(42)
    # 50 个 p 值，前 5 个是真实显著
    true_sig = np.array([0.001, 0.003, 0.005, 0.008, 0.010])
    null_dist = np.random.uniform(0.05, 0.95, 45)
    combined = np.concatenate([true_sig, null_dist])
    np.random.shuffle(combined)
    return combined


@pytest.fixture
def all_significant_p_values():
    """全部显著的 p 值"""
    return np.array([0.001, 0.002, 0.003, 0.004, 0.005])


@pytest.fixture
def all_nonsignificant_p_values():
    """全部不显著的 p 值"""
    return np.array([0.15, 0.25, 0.35, 0.45, 0.55])


@pytest.fixture
def boundary_p_values():
    """边界附近的 p 值（≈0.05）"""
    return np.array([0.04, 0.05, 0.06, 0.07, 0.08])


# =============================================================================
# AI 审查测试 Fixtures
# =============================================================================

@pytest.fixture
def ai_anova_good_report():
    """合格的 AI 生成的 ANOVA 报告"""
    return {
        'test_method': 'one_way_anova',
        'groups': ['A', 'B', 'C', 'D', 'E'],
        'n_per_group': 500,
        'f_statistic': 2.45,
        'p_value': 0.045,
        'eta_squared': 0.02,
        'posthoc_method': 'Tukey HSD',
        'posthoc_corrected': True,
        'assumptions_checked': True,
        'normality': {'A': 0.23, 'B': 0.31, 'C': 0.18, 'D': 0.25, 'E': 0.29},
        'homogeneity': 0.15,
        'conclusion': 'ANOVA 发现显著差异，但效应量小，建议谨慎解释'
    }


@pytest.fixture
def ai_anova_bad_report():
    """有问题的 AI 生成的 ANOVA 报告"""
    return {
        'test_method': 'multiple_t_tests',  # 错误：应该用 ANOVA
        'groups': ['A', 'B', 'C', 'D', 'E'],
        'n_per_group': 500,
        'comparisons': 10,  # 做了 10 次两两比较
        'significant_pairs': [
            {'pair': 'A vs E', 'p_value': 0.03, 'significant': True},
            {'pair': 'B vs E', 'p_value': 0.04, 'significant': True}
        ],
        # 缺少校正说明
        # 缺少效应量
        # 缺少前提假设检查
        'conclusion': 'E 渠道显著优于 A 和 B，建议全面切换'  # 过度解读
    }


@pytest.fixture
def ai_posthoc_missing_correction():
    """缺少多重比较校正的 AI 报告"""
    return {
        'test_method': 'anova',
        'f_statistic': 3.2,
        'p_value': 0.01,
        'posthoc_pairs': [
            {'pair': 'A vs B', 'p_value': 0.03},  # 未校正
            {'pair': 'A vs C', 'p_value': 0.04},  # 未校正
            {'pair': 'A vs D', 'p_value': 0.08},
            {'pair': 'A vs E', 'p_value': 0.02},
        ],
        # 缺少校正说明
        'conclusion': 'A 与多个组存在显著差异'
    }


# =============================================================================
# 边界情况 Fixtures
# =============================================================================

@pytest.fixture
def empty_groups():
    """空组数据"""
    return {
        'A': np.array([]),
        'B': np.array([]),
        'C': np.array([])
    }


@pytest.fixture
def single_value_groups():
    """单值组数据"""
    return {
        'A': np.array([42]),
        'B': np.array([45]),
        'C': np.array([40])
    }


@pytest.fixture
def two_groups():
    """只有两组的数据（ANOVA 退化为 t 检验）"""
    np.random.seed(42)
    return {
        'A': np.random.normal(loc=100, scale=15, size=50),
        'B': np.random.normal(loc=110, scale=15, size=50)
    }


@pytest.fixture
def identical_groups():
    """完全相同的组数据（组内方差为0）"""
    values = np.array([50, 50, 50, 50, 50])
    return {
        'A': values.copy(),
        'B': values.copy(),
        'C': values.copy()
    }


@pytest.fixture
def extreme_outlier_groups():
    """包含极端离群点的组数据"""
    np.random.seed(42)
    base = np.random.normal(loc=100, scale=15, size=49)
    group_a = np.concatenate([base, np.array([500])])  # 极端离群点
    group_b = np.random.normal(loc=100, scale=15, size=50)
    group_c = np.random.normal(loc=100, scale=15, size=50)
    return {
        'A': group_a,
        'B': group_b,
        'C': group_c
    }


# =============================================================================
# StatLab 测试 Fixtures
# =============================================================================

@pytest.fixture
def statlab_anova_data():
    """StatLab 项目用例：多渠道转化率数据"""
    np.random.seed(42)
    data = []
    for channel in ['Email', 'SMS', 'Push', 'In-App']:
        if channel == 'Push':
            # Push 通知转化率略高
            conversions = np.random.binomial(1, 0.12, 1000)
        else:
            conversions = np.random.binomial(1, 0.10, 1000)
        for conv in conversions:
            data.append({'channel': channel, 'converted': conv})
    return pd.DataFrame(data)


@pytest.fixture
def statlab_user_segment_data():
    """StatLab 项目用例：用户群组消费金额数据"""
    np.random.seed(42)
    data = []
    for segment in ['New', 'Active', 'Lapsed', 'VIP']:
        if segment == 'VIP':
            # VIP 用户消费更高
            amounts = np.random.normal(loc=500, scale=100, size=200)
        elif segment == 'Active':
            amounts = np.random.normal(loc=300, scale=80, size=200)
        elif segment == 'New':
            amounts = np.random.normal(loc=150, scale=50, size=200)
        else:  # Lapsed
            amounts = np.random.normal(loc=100, scale  =40, size=200)
        for amount in amounts:
            data.append({'segment': segment, 'amount': max(0, amount)})
    return pd.DataFrame(data)

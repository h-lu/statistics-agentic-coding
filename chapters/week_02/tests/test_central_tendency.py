"""
Week 02 测试：集中趋势（Central Tendency）

测试覆盖：
1. calculate_central_tendency() - 计算均值/中位数/众数

测试用例类型：
- 正例：正常数据下的正确计算
- 边界：空数据、单值、极端值
- 反例：错误的数据类型
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

# 导入被测试的模块（路径已在 conftest.py 中设置）
solution = pytest.importorskip("solution")

# 获取可能存在的函数
calculate_central_tendency = getattr(solution, 'calculate_central_tendency', None)

if not calculate_central_tendency:
    pytest.skip("calculate_central_tendency 函数不存在", allow_module_level=True)


# =============================================================================
# Test: calculate_central_tendency()
# =============================================================================

class TestCalculateCentralTendency:
    """测试集中趋势计算函数"""

    # --------------------
    # 正例（happy path）
    # --------------------

    def test_calculate_normal_distribution(self, sample_numeric_data: pd.Series):
        """
        测试正态分布数据的集中趋势计算

        期望：均值、中位数、众数都存在且接近
        """
        result = calculate_central_tendency(sample_numeric_data)

        assert isinstance(result, dict), "返回值应该是字典"
        assert 'mean' in result, "结果应包含 'mean'"
        assert 'median' in result, "结果应包含 'median'"

        # 验证计算值正确性
        expected_mean = sample_numeric_data.mean()
        expected_median = sample_numeric_data.median()

        assert result['mean'] == pytest.approx(expected_mean, rel=1e-5)
        assert result['median'] == pytest.approx(expected_median, rel=1e-5)

    def test_calculate_with_outliers(self, sample_data_with_outliers: pd.Series):
        """
        测试包含极端值的数据

        期望：均值会被极端值拉高，中位数相对稳定
        """
        result = calculate_central_tendency(sample_data_with_outliers)

        # 验证均值 > 中位数（右偏）
        assert result['mean'] > result['median'], "极端值会拉高均值"

    # --------------------
    # 边界情况
    # --------------------

    def test_calculate_with_missing_values(self, dataframe_with_missing: pd.DataFrame):
        """
        测试包含缺失值的数据

        期望：应自动忽略缺失值进行计算
        """
        series = dataframe_with_missing['age'].dropna()
        result = calculate_central_tendency(series)

        # 验证计算时排除了 NaN
        assert not np.isnan(result['mean'])
        assert not np.isnan(result['median'])

    # --------------------
    # 反例（错误输入）
    # --------------------

    def test_calculate_all_nan_series(self):
        """
        测试全 NaN 的 Series

        期望：应返回 NaN 或 None
        """
        nan_series = pd.Series([np.nan, np.nan, np.nan])
        result = calculate_central_tendency(nan_series)

        # 结果应包含 NaN 值
        for val in result.values():
            assert val is None or (isinstance(val, float) and np.isnan(val))


# =============================================================================
# Test: 使用 Penguins 数据集
# =============================================================================

class TestWithPenguinsData:
    """使用 seaborn Penguins 数据集的测试"""

    def test_calculate_penguins_body_mass(self):
        """
        测试使用 Penguins 数据集计算体重集中趋势

        期望：应能正确处理真实数据集
        """
        try:
            import seaborn as sns
            penguins = sns.load_dataset("penguins")
        except ImportError:
            pytest.skip("seaborn 不可用")

        result = calculate_central_tendency(penguins['body_mass_g'].dropna())

        assert isinstance(result, dict)
        assert 'mean' in result
        assert 'median' in result

        # Penguins 体重均值应该大约在 4200g 左右
        assert 4000 < result['mean'] < 4500

    def test_calculate_penguins_by_species(self):
        """
        测试按物种分组计算集中趋势

        期望：Gentoo 企鹅应该最重
        """
        try:
            import seaborn as sns
            penguins = sns.load_dataset("penguins")
        except ImportError:
            pytest.skip("seaborn 不可用")

        species_stats = {}
        for species in penguins['species'].unique():
            data = penguins[penguins['species'] == species]['body_mass_g'].dropna()
            species_stats[species] = calculate_central_tendency(data)

        # Gentoo 应该是最重的
        assert species_stats['Gentoo']['mean'] > species_stats['Adelie']['mean']
        assert species_stats['Gentoo']['mean'] > species_stats['Chinstrap']['mean']

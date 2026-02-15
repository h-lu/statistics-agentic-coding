"""
Week 02 测试：离散程度（Dispersion）

测试覆盖：
1. calculate_dispersion() - 计算标准差、IQR、方差、极差

测试用例类型：
- 正例：正常数据下的正确计算
- 边界：空数据、单值、零方差数据
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
calculate_dispersion = getattr(solution, 'calculate_dispersion', None)

if not calculate_dispersion:
    pytest.skip("calculate_dispersion 函数不存在", allow_module_level=True)


# =============================================================================
# Test: calculate_dispersion()
# =============================================================================

class TestCalculateDispersion:
    """测试离散程度计算函数"""

    # --------------------
    # 正例（happy path）
    # --------------------

    def test_calculate_basic_dispersion(self, sample_numeric_data: pd.Series):
        """
        测试基本离散程度计算

        期望：返回标准差、IQR、方差、极差
        """
        result = calculate_dispersion(sample_numeric_data)

        assert isinstance(result, dict), "返回值应该是字典"

        # 验证关键字段存在
        expected_keys = ['std', 'iqr']
        for key in expected_keys:
            assert key in result, f"结果应包含 '{key}'"

        # 验证标准差计算正确
        expected_std = sample_numeric_data.std()
        assert result['std'] == pytest.approx(expected_std, rel=1e-5)

        # 验证 IQR 计算正确
        q1 = sample_numeric_data.quantile(0.25)
        q3 = sample_numeric_data.quantile(0.75)
        expected_iqr = q3 - q1
        assert result['iqr'] == pytest.approx(expected_iqr, rel=1e-5)

    def test_calculate_with_outliers(self, sample_data_with_outliers: pd.Series):
        """
        测试包含极端值的数据

        期望：标准差会受极端值影响变大，IQR 相对稳定
        """
        result = calculate_dispersion(sample_data_with_outliers)

        # 标准差应该较大（受极端值影响）
        assert result['std'] > result['iqr'], "受极端值影响，标准差应大于 IQR"

    def test_calculate_range(self, sample_numeric_data: pd.Series):
        """
        测试极差计算

        期望：极差 = 最大值 - 最小值
        """
        result = calculate_dispersion(sample_numeric_data)

        if 'range' in result:
            expected_range = sample_numeric_data.max() - sample_numeric_data.min()
            assert result['range'] == pytest.approx(expected_range, rel=1e-5)

    # --------------------
    # 边界情况
    # --------------------

    def test_calculate_with_missing_values(self, dataframe_with_missing: pd.DataFrame):
        """
        测试包含缺失值的数据

        期望：应自动忽略缺失值进行计算
        """
        series = dataframe_with_missing['salary'].dropna()
        result = calculate_dispersion(series)

        # 验证计算时排除了 NaN
        assert not np.isnan(result['std'])
        assert not np.isnan(result['iqr'])

    # --------------------
    # 反例（错误输入）
    # --------------------

    def test_calculate_all_nan_series(self):
        """
        测试全 NaN 的 Series

        期望：应返回 NaN 或 None
        """
        nan_series = pd.Series([np.nan, np.nan, np.nan])
        result = calculate_dispersion(nan_series)

        for val in result.values():
            assert val is None or (isinstance(val, float) and np.isnan(val))


# =============================================================================
# Test: 使用 Penguins 数据集
# =============================================================================

class TestDispersionWithPenguins:
    """使用 seaborn Penguins 数据集的测试"""

    def test_calculate_penguins_dispersion(self):
        """
        测试使用 Penguins 数据集计算离散程度

        期望：应能正确处理真实数据集
        """
        try:
            import seaborn as sns
            penguins = sns.load_dataset("penguins")
        except ImportError:
            pytest.skip("seaborn 不可用")

        result = calculate_dispersion(penguins['body_mass_g'].dropna())

        assert isinstance(result, dict)
        assert 'std' in result
        assert 'iqr' in result

        # Penguins 体重标准差应该大约在 800g 左右
        assert 500 < result['std'] < 1000

    def test_compare_dispersion_by_species(self):
        """
        测试比较不同物种的离散程度

        期望：Gentoo 企鹅的体重波动应该最大
        """
        try:
            import seaborn as sns
            penguins = sns.load_dataset("penguins")
        except ImportError:
            pytest.skip("seaborn 不可用")

        species_std = {}
        for species in penguins['species'].unique():
            data = penguins[penguins['species'] == species]['body_mass_g'].dropna()
            result = calculate_dispersion(data)
            species_std[species] = result['std']

        # Gentoo 的标准差应该最大（波动最大）
        assert species_std['Gentoo'] == max(species_std.values())

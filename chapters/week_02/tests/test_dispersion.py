"""
Week 02 测试：离散程度（Dispersion）

测试覆盖：
1. calculate_dispersion() - 计算标准差、IQR、方差、极差
2. compare_dispersion_measures() - 比较不同离散程度指标
3. detect_outliers_using_iqr() - 使用 IQR 检测异常值

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

# 导入被测试的模块
import sys
starter_code_path = Path(__file__).parent.parent / "starter_code"
sys.path.insert(0, str(starter_code_path))

# 尝试导入 solution 模块
try:
    from solution import (
        calculate_dispersion,
        compare_dispersion_measures,
        detect_outliers_using_iqr,
    )
except ImportError:
    pytest.skip("solution.py not yet created", allow_module_level=True)


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
        expected_keys = ['std', 'iqr', 'variance', 'range']
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

    def test_calculate_quartiles_details(self, sample_numeric_data: pd.Series):
        """
        测试四分位数详细计算

        期望：应包含 Q1, Q3 等中间结果
        """
        result = calculate_dispersion(sample_numeric_data)

        # 如果函数提供了详细的四分位数
        if 'q1' in result and 'q3' in result:
            assert result['q1'] < result['q3']
            # 验证 IQR = Q3 - Q1
            assert result['iqr'] == pytest.approx(result['q3'] - result['q1'], rel=1e-5)

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

    def test_calculate_zero_variance(self):
        """
        测试零方差数据（所有值相同）

        期望：标准差和方差应为 0
        """
        constant_series = pd.Series([5, 5, 5, 5, 5])
        result = calculate_dispersion(constant_series)

        assert result['std'] == pytest.approx(0, abs=1e-10)
        assert result['variance'] == pytest.approx(0, abs=1e-10)
        assert result['iqr'] == pytest.approx(0, abs=1e-10)
        assert result['range'] == pytest.approx(0, abs=1e-10)

    def test_calculate_empty_series(self, empty_series: pd.Series):
        """
        测试空 Series

        期望：应返回空字典或包含 NaN 的字典
        """
        result = calculate_dispersion(empty_series)

        assert isinstance(result, dict)
        assert len(result) == 0 or all(np.isnan(v) if isinstance(v, float) else v is None for v in result.values())

    def test_calculate_single_value(self, single_value_series: pd.Series):
        """
        测试单个值的数据

        期望：标准差和方差应为 0（或 NaN，取决于实现）
        """
        result = calculate_dispersion(single_value_series)

        # pandas 的 std() 在单个值时返回 NaN
        assert result.get('std', 0) in [pytest.approx(0, abs=1e-10), np.nan]
        assert result.get('variance', 0) in [pytest.approx(0, abs=1e-10), np.nan]
        assert result.get('iqr', 0) == pytest.approx(0, abs=1e-10)
        assert result.get('range', 0) == pytest.approx(0, abs=1e-10)

    def test_calculate_with_missing_values(self, dataframe_with_missing: pd.DataFrame):
        """
        测试包含缺失值的数据

        期望：应自动忽略缺失值进行计算
        """
        series = dataframe_with_missing['salary']
        result = calculate_dispersion(series)

        # 验证计算时排除了 NaN
        assert not np.isnan(result['std'])
        assert not np.isnan(result['iqr'])

    # --------------------
    # 反例（错误输入）
    # --------------------

    def test_calculate_non_series_input(self):
        """
        测试非 Series 输入

        期望：应抛出 TypeError 或 ValueError
        """
        with pytest.raises((TypeError, ValueError)):
            calculate_dispersion([1, 2, 3])

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
# Test: compare_dispersion_measures()
# =============================================================================

class TestCompareDispersionMeasures:
    """测试比较离散程度指标函数"""

    # --------------------
    # 正例（happy path）
    # --------------------

    def test_compare_stable_vs_volatile(self):
        """
        测试稳定数据 vs 波动数据的比较

        期望：波动数据的标准差和 IQR 都更大
        """
        stable_data = pd.Series([48, 49, 50, 51, 52])
        volatile_data = pd.Series([0, 25, 50, 75, 100])

        stable_result = calculate_dispersion(stable_data)
        volatile_result = calculate_dispersion(volatile_data)

        assert volatile_result['std'] > stable_result['std']
        assert volatile_result['iqr'] > stable_result['iqr']

    def test_compare_std_vs_iqr_sensitivity(self, sample_data_with_outliers: pd.Series):
        """
        测试标准差 vs IQR 对极端值的敏感度

        期望：标准差受极端值影响更大
        """
        dispersion = calculate_dispersion(sample_data_with_outliers)

        # 计算敏感度比率：std/iqr
        sensitivity_ratio = dispersion['std'] / dispersion['iqr'] if dispersion['iqr'] > 0 else float('inf')

        # 如果比率大于 2，说明标准差受极端值影响较大
        # 这是一个启发式判断
        comparison = compare_dispersion_measures(sample_data_with_outliers)

        if isinstance(comparison, dict):
            assert 'std_iqr_ratio' in comparison or 'interpretation' in comparison
        elif isinstance(comparison, str):
            # 应该在解释中提到极端值的影响
            has_outlier_warning = any(kw in comparison.lower() for kw in ['outlier', 'extreme', '极端', '异常'])
            # 这是一个可选的期望，取决于实现

    # --------------------
    # 边界情况
    # --------------------

    def test_compare_identical_values(self):
        """
        测试所有值相同的情况

        期望：所有离散程度指标都为 0
        """
        identical_data = pd.Series([42] * 10)
        comparison = compare_dispersion_measures(identical_data)

        if isinstance(comparison, dict):
            assert comparison.get('std', 0) == pytest.approx(0, abs=1e-10)
            assert comparison.get('iqr', 0) == pytest.approx(0, abs=1e-10)
        elif isinstance(comparison, str):
            assert 'zero' in comparison.lower() or '无' in comparison or '相同' in comparison

    # --------------------
    # 反例（错误输入）
    # --------------------

    def test_compare_non_series_input(self):
        """
        测试非 Series 输入

        期望：应抛出异常
        """
        with pytest.raises((TypeError, ValueError)):
            compare_dispersion_measures("not a series")


# =============================================================================
# Test: detect_outliers_using_iqr()
# =============================================================================

class TestDetectOutliersUsingIQR:
    """测试使用 IQR 检测异常值函数"""

    # --------------------
    # 正例（happy path）
    # --------------------

    def test_detect_no_outliers(self, sample_numeric_data: pd.Series):
        """
        测试无异常值的数据

        期望：应返回空列表或空的异常值集合
        """
        outliers = detect_outliers_using_iqr(sample_numeric_data)

        assert isinstance(outliers, (list, pd.Series, np.ndarray))
        assert len(outliers) == 0, "正态分布数据应无异常值"

    def test_detect_with_outliers(self, sample_data_with_outliers: pd.Series):
        """
        测试包含异常值的数据

        期望：应检测到极端值（100 和 150）
        """
        outliers = detect_outliers_using_iqr(sample_data_with_outliers)

        assert isinstance(outliers, (list, pd.Series, np.ndarray))
        assert len(outliers) > 0, "应检测到异常值"

        # 验证检测到的异常值是大的那些值
        if isinstance(outliers, (pd.Series, np.ndarray)):
            outlier_values = outliers.values if isinstance(outliers, pd.Series) else outliers
        else:
            outlier_values = outliers

        # 所有异常值都应该大于 Q3 + 1.5*IQR
        q1 = sample_data_with_outliers.quantile(0.25)
        q3 = sample_data_with_outliers.quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr

        for val in outlier_values:
            assert val > upper_bound, f"异常值 {val} 应大于上界 {upper_bound}"

    def test_detect_outliers_with_bounds(self, sample_data_with_outliers: pd.Series):
        """
        测试返回异常值边界信息

        期望：应返回上下界信息
        """
        result = detect_outliers_using_iqr(sample_data_with_outliers)

        # 如果返回字典，包含边界信息
        if isinstance(result, dict):
            assert 'outliers' in result or 'outlier_values' in result
            assert 'lower_bound' in result or 'upper_bound' in result

            outliers = result.get('outliers', result.get('outlier_values', []))
            upper_bound = result.get('upper_bound', 0)

            if len(outliers) > 0:
                # 验证所有异常值都在界外
                for val in outliers:
                    assert val > upper_bound

    # --------------------
    # 边界情况
    # --------------------

    def test_detect_empty_series(self, empty_series: pd.Series):
        """
        测试空 Series

        期望：应返回空列表或空结果
        """
        outliers = detect_outliers_using_iqr(empty_series)

        assert isinstance(outliers, (list, dict, pd.Series))
        if isinstance(outliers, list):
            assert len(outliers) == 0
        elif isinstance(outliers, dict):
            assert len(outliers.get('outliers', [])) == 0

    def test_detect_single_value(self, single_value_series: pd.Series):
        """
        测试单个值

        期望：单个值不应被判定为异常值
        """
        outliers = detect_outliers_using_iqr(single_value_series)

        if isinstance(outliers, (list, pd.Series, np.ndarray)):
            assert len(outliers) == 0
        elif isinstance(outliers, dict):
            assert len(outliers.get('outliers', [])) == 0

    def test_detect_with_missing_values(self, dataframe_with_missing: pd.DataFrame):
        """
        测试包含缺失值的数据

        期望：应忽略缺失值进行异常值检测
        """
        series = dataframe_with_missing['salary']
        result = detect_outliers_using_iqr(series)

        # 应该能正常处理，不抛出异常
        assert isinstance(result, (list, dict, pd.Series))

    # --------------------
    # 反例（错误输入）
    # --------------------

    def test_detect_non_series_input(self):
        """
        测试非 Series 输入

        期望：应抛出异常
        """
        with pytest.raises((TypeError, ValueError)):
            detect_outliers_using_iqr([1, 2, 3])


# =============================================================================
# Test: 辅助函数
# =============================================================================

class TestDispersionHelperFunctions:
    """测试离散程度相关的辅助函数"""

    def test_coefficient_of_variation(self, sample_numeric_data: pd.Series):
        """
        测试变异系数计算（如果实现）

        CV = std / mean，用于比较不同量级数据的波动
        """
        # 这可能不是一个独立的函数，而是在 calculate_dispersion 中
        result = calculate_dispersion(sample_numeric_data)

        # 如果包含 CV
        if 'cv' in result or 'coefficient_of_variation' in result:
            cv = result.get('cv', result.get('coefficient_of_variation'))
            expected_cv = sample_numeric_data.std() / sample_numeric_data.mean()
            assert cv == pytest.approx(expected_cv, rel=1e-5)

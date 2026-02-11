"""
Week 02 测试：集中趋势（Central Tendency）

测试覆盖：
1. calculate_central_tendency() - 计算均值/中位数/众数
2. recommend_central_measure() - 推荐合适的集中趋势指标
3. compare_mean_median() - 比较均值和中位数，判断偏态

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

# 导入被测试的模块
import sys
starter_code_path = Path(__file__).parent.parent / "starter_code"
sys.path.insert(0, str(starter_code_path))

# 尝试导入 solution 模块（如果尚未创建，测试会跳过）
try:
    from solution import (
        calculate_central_tendency,
        recommend_central_measure,
        compare_mean_median,
    )
except ImportError:
    pytest.skip("solution.py not yet created", allow_module_level=True)


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
        assert 'mode' in result, "结果应包含 'mode'"

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

    def test_calculate_categorical_data(self, sample_categorical_series: pd.Series):
        """
        测试分类型数据的集中趋势

        期望：众数应返回出现最多的类别
        """
        result = calculate_central_tendency(sample_categorical_series)

        assert 'mode' in result, "分类数据应返回众数"
        # 北京出现次数最多
        assert result['mode'] == '北京' or result['mode'] in ['北京', '上海', '深圳']

    # --------------------
    # 边界情况
    # --------------------

    def test_calculate_empty_series(self, empty_series: pd.Series):
        """
        测试空 Series

        期望：应返回空字典或 None，不应抛出异常
        """
        result = calculate_central_tendency(empty_series)

        # 期望返回空字典或包含 NaN 的字典
        assert isinstance(result, dict), "空数据应返回字典"
        assert len(result) == 0 or all(np.isnan(v) if isinstance(v, float) else v is None for v in result.values())

    def test_calculate_single_value(self, single_value_series: pd.Series):
        """
        测试单个值的数据

        期望：均值=中位数=众数=该值
        """
        result = calculate_central_tendency(single_value_series)

        assert result['mean'] == pytest.approx(42.0)
        assert result['median'] == pytest.approx(42.0)
        # 众数也可能是该值（取决于实现）

    def test_calculate_with_missing_values(self, dataframe_with_missing: pd.DataFrame):
        """
        测试包含缺失值的数据

        期望：应自动忽略缺失值进行计算
        """
        series = dataframe_with_missing['salary']
        result = calculate_central_tendency(series)

        # 验证计算时排除了 NaN
        assert not np.isnan(result['mean'])
        assert not np.isnan(result['median'])

    # --------------------
    # 反例（错误输入）
    # --------------------

    def test_calculate_non_series_input(self):
        """
        测试非 Series 输入

        期望：应抛出 TypeError 或返回错误提示
        """
        with pytest.raises((TypeError, ValueError)):
            calculate_central_tendency([1, 2, 3])

    def test_calculate_all_nan_series(self):
        """
        测试全 NaN 的 Series

        期望：应返回 NaN 或 None
        """
        nan_series = pd.Series([np.nan, np.nan, np.nan])
        result = calculate_central_tendency(nan_series)

        # 所有结果都应该是 NaN
        for val in result.values():
            assert val is None or (isinstance(val, float) and np.isnan(val))


# =============================================================================
# Test: recommend_central_measure()
# =============================================================================

class TestRecommendCentralMeasure:
    """测试推荐集中趋势指标函数"""

    # --------------------
    # 正例（happy path）
    # --------------------

    def test_recommend_for_symmetric_data(self, sample_numeric_data: pd.Series):
        """
        测试对称分布的推荐

        期望：推荐使用均值
        """
        recommendation = recommend_central_measure(sample_numeric_data)

        assert isinstance(recommendation, str) or isinstance(recommendation, dict)
        if isinstance(recommendation, str):
            assert 'mean' in recommendation.lower() or '均值' in recommendation
        else:
            assert 'recommended' in recommendation

    def test_recommend_for_skewed_data(self, sample_skewed_data: pd.Series):
        """
        测试偏态分布的推荐

        期望：推荐使用中位数（因为有极端值）
        """
        recommendation = recommend_central_measure(sample_skewed_data)

        if isinstance(recommendation, str):
            assert 'median' in recommendation.lower() or '中位数' in recommendation
        else:
            assert 'recommended' in recommendation

    def test_recommend_for_categorical_data(self, sample_categorical_series: pd.Series):
        """
        测试分类型数据的推荐

        期望：推荐使用众数
        """
        recommendation = recommend_central_measure(sample_categorical_series)

        if isinstance(recommendation, str):
            assert 'mode' in recommendation.lower() or '众数' in recommendation
        else:
            assert 'recommended' in recommendation

    # --------------------
    # 边界情况
    # --------------------

    def test_recommend_empty_series(self, empty_series: pd.Series):
        """
        测试空数据的推荐

        期望：应返回"数据不足"之类的提示
        """
        recommendation = recommend_central_measure(empty_series)

        assert isinstance(recommendation, str)
        assert any(kw in recommendation.lower() for kw in ['empty', '空', 'insufficient', '不足'])

    # --------------------
    # 反例（错误输入）
    # --------------------

    def test_recommend_non_series_input(self):
        """
        测试非 Series 输入

        期望：应抛出异常或返回错误提示
        """
        with pytest.raises((TypeError, ValueError)):
            recommend_central_measure("not a series")


# =============================================================================
# Test: compare_mean_median()
# =============================================================================

class TestCompareMeanMedian:
    """测试均值中位数比较函数"""

    # --------------------
    # 正例（happy path）
    # --------------------

    def test_compare_symmetric_distribution(self, sample_numeric_data: pd.Series):
        """
        测试对称分布的比较

        期望：均值和中位数接近，判定为"对称分布"
        """
        comparison = compare_mean_median(sample_numeric_data)

        assert isinstance(comparison, dict) or isinstance(comparison, str)

        if isinstance(comparison, dict):
            assert 'mean' in comparison
            assert 'median' in comparison
            assert 'skewness' in comparison or 'interpretation' in comparison

            # 验证判断结果
            interpretation = comparison.get('skewness', comparison.get('interpretation', ''))
            assert any(kw in interpretation.lower() for kw in ['symmetric', '对称', 'normal', 'normal-like'])
        elif isinstance(comparison, str):
            assert '对称' in comparison or 'symmetric' in comparison.lower()

    def test_compare_right_skewed(self, sample_skewed_data: pd.Series):
        """
        测试右偏分布的比较

        期望：均值 > 中位数，判定为"右偏"
        """
        comparison = compare_mean_median(sample_skewed_data)

        if isinstance(comparison, dict):
            assert comparison['mean'] > comparison['median']
            interpretation = comparison.get('skewness', comparison.get('interpretation', ''))
            assert any(kw in interpretation.lower() for kw in ['right', '右偏', 'skewed'])
        elif isinstance(comparison, str):
            assert '右偏' in comparison or 'right' in comparison.lower()

    def test_compare_left_skewed(self):
        """
        测试左偏分布的比较

        期望：均值 < 中位数，判定为"左偏"
        """
        # 创建左偏数据（大部分值较高，少数低值）
        left_skewed = pd.Series([1, 2, 80, 85, 90, 95, 100, 105, 110, 120])
        comparison = compare_mean_median(left_skewed)

        if isinstance(comparison, dict):
            assert comparison['mean'] < comparison['median']
            interpretation = comparison.get('skewness', comparison.get('interpretation', ''))
            assert any(kw in interpretation.lower() for kw in ['left', '左偏'])
        elif isinstance(comparison, str):
            assert '左偏' in comparison or 'left' in comparison.lower()

    # --------------------
    # 边界情况
    # --------------------

    def test_compare_single_value(self, single_value_series: pd.Series):
        """
        测试单个值的比较

        期望：均值 = 中位数，判定为"无偏态"
        """
        comparison = compare_mean_median(single_value_series)

        if isinstance(comparison, dict):
            assert comparison['mean'] == comparison['median']
        elif isinstance(comparison, str):
            assert '无' in comparison or 'equal' in comparison.lower()

    def test_compare_with_missing_values(self, dataframe_with_missing: pd.DataFrame):
        """
        测试包含缺失值的比较

        期望：应忽略缺失值进行计算
        """
        series = dataframe_with_missing['age']
        comparison = compare_mean_median(series)

        assert isinstance(comparison, (dict, str))
        if isinstance(comparison, dict):
            assert not np.isnan(comparison['mean'])
            assert not np.isnan(comparison['median'])

    # --------------------
    # 反例（错误输入）
    # --------------------

    def test_compare_empty_series(self, empty_series: pd.Series):
        """
        测试空 Series

        期望：应返回错误或 None
        """
        comparison = compare_mean_median(empty_series)

        # 可能返回空字典、None 或错误字符串
        assert comparison is None or comparison == {} or isinstance(comparison, str)

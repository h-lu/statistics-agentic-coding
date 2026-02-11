"""
Week 03 测试：特征缩放（Feature Scaling）

测试覆盖：
1. standardize_features() - StandardScaler 标准化
2. normalize_features() - MinMaxScaler 归一化

测试用例类型：
- 正例：StandardScaler 输出均值为0、标准差为1
- 正例：MinMaxScaler 输出范围在[0,1]
- 边界：常数列的缩放（标准差为0）
- 反例：未 fit 就 transform
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
        standardize_features,
        normalize_features,
    )
except ImportError:
    pytest.skip("solution.py not yet created", allow_module_level=True)


# =============================================================================
# Test: standardize_features() - StandardScaler
# =============================================================================

class TestStandardizeFeatures:
    """测试 StandardScaler 标准化"""

    # --------------------
    # 正例（happy path）
    # --------------------

    def test_standardize_returns_tuple(self, df_for_scaling: pd.DataFrame):
        """
        测试标准化返回元组

        期望：返回 (DataFrame, scaler)
        """
        result = standardize_features(df_for_scaling, columns=['feature_small', 'feature_large'])

        assert isinstance(result, tuple), "返回值应该是元组"
        assert len(result) == 2, "元组应该包含两个元素"

        df_result, scaler = result
        assert isinstance(df_result, pd.DataFrame), "第一个元素应该是 DataFrame"
        assert scaler is not None, "第二个元素应该是 scaler"

    def test_standardize_creates_new_columns(self, df_for_scaling: pd.DataFrame):
        """
        测试标准化创建新列

        期望：创建带有 _std 后缀的新列
        """
        result_df, _ = standardize_features(df_for_scaling, columns=['feature_small'])

        assert 'feature_small_std' in result_df.columns, "应该创建 feature_small_std 列"

    def test_standardized_mean_zero(self, df_for_scaling: pd.DataFrame):
        """
        测试标准化后均值为 0

        期望：标准化后的数据均值为 0（允许浮点误差）
        """
        result_df, _ = standardize_features(df_for_scaling, columns=['feature_small'])

        # 验证均值为 0
        assert result_df['feature_small_std'].mean() == pytest.approx(0.0, abs=1e-10)

    def test_standardized_std_one(self, df_for_scaling: pd.DataFrame):
        """
        测试标准化后标准差为 1

        期望：标准化后的数据标准差为 1（使用总体标准差 ddof=0）
        """
        result_df, _ = standardize_features(df_for_scaling, columns=['feature_small'])

        # 验证标准差为 1（使用总体标准差，与 StandardScaler 一致）
        assert result_df['feature_small_std'].std(ddof=0) == pytest.approx(1.0, rel=1e-5)

    def test_standardize_preserves_original(self, df_for_scaling: pd.DataFrame):
        """
        测试保留原始列

        期望：原始列保持不变
        """
        original = df_for_scaling['feature_small'].copy()
        result_df, _ = standardize_features(df_for_scaling, columns=['feature_small'])

        # 原始列应该保持不变
        pd.testing.assert_series_equal(result_df['feature_small'], original)

    # --------------------
    # 边界情况
    # --------------------

    def test_standardize_constant_column(self, df_for_scaling: pd.DataFrame):
        """
        测试常数列的标准化

        期望：抛出 ValueError 或返回 NaN
        """
        # 常数列的标准差为 0，标准化会导致除零错误
        with pytest.raises((ValueError, ZeroDivisionError)):
            standardize_features(df_for_scaling, columns=['feature_constant'])

    def test_standardize_single_column(self, df_for_scaling: pd.DataFrame):
        """
        测试单列标准化

        期望：正确处理单列
        """
        result_df, scaler = standardize_features(df_for_scaling, columns=['feature_small'])

        assert isinstance(result_df, pd.DataFrame)
        assert 'feature_small_std' in result_df.columns

    # --------------------
    # 反例（错误输入）
    # --------------------

    def test_standardize_nonexistent_column(self, df_for_scaling: pd.DataFrame):
        """
        测试不存在的列

        期望：抛出 KeyError
        """
        with pytest.raises(KeyError):
            standardize_features(df_for_scaling, columns=['nonexistent'])


# =============================================================================
# Test: normalize_features() - MinMaxScaler
# =============================================================================

class TestNormalizeFeatures:
    """测试 MinMaxScaler 归一化"""

    # --------------------
    # 正例（happy path）
    # --------------------

    def test_normalize_returns_tuple(self, df_for_scaling: pd.DataFrame):
        """
        测试归一化返回元组

        期望：返回 (DataFrame, scaler)
        """
        result = normalize_features(df_for_scaling, columns=['feature_small'])

        assert isinstance(result, tuple), "返回值应该是元组"
        assert len(result) == 2

    def test_normalize_creates_new_columns(self, df_for_scaling: pd.DataFrame):
        """
        测试归一化创建新列

        期望：创建带有 _norm 后缀的新列
        """
        result_df, _ = normalize_features(df_for_scaling, columns=['feature_small'])

        assert 'feature_small_norm' in result_df.columns, "应该创建 feature_small_norm 列"

    def test_normalize_range_zero_one(self, df_for_scaling: pd.DataFrame):
        """
        测试归一化后范围在 [0, 1]

        期望：所有值都在 0 和 1 之间
        """
        result_df, _ = normalize_features(df_for_scaling, columns=['feature_small', 'feature_large'])

        # 验证范围在 [0, 1]
        for col in ['feature_small_norm', 'feature_large_norm']:
            assert result_df[col].min() >= 0.0, f"{col} 最小值应 >= 0"
            assert result_df[col].max() <= 1.0, f"{col} 最大值应 <= 1"

    def test_normalize_min_zero_max_one(self, df_for_scaling: pd.DataFrame):
        """
        测试归一化后最小值为 0，最大值为 1

        期望：每列的最小值为 0，最大值为 1
        """
        result_df, _ = normalize_features(df_for_scaling, columns=['feature_small'])

        assert result_df['feature_small_norm'].min() == pytest.approx(0.0, abs=1e-10)
        assert result_df['feature_small_norm'].max() == pytest.approx(1.0, abs=1e-10)

    # --------------------
    # 边界情况
    # --------------------

    def test_normalize_constant_column(self, df_for_scaling: pd.DataFrame):
        """
        测试常数列的归一化

        期望：返回全 0 或全 1
        """
        result_df, _ = normalize_features(df_for_scaling, columns=['feature_constant'])

        # 常数列归一化后应该全为 0（或某个常数）
        unique_values = result_df['feature_constant_norm'].unique()
        assert len(unique_values) == 1, "常数列归一化后应该只有一个唯一值"

    # --------------------
    # 反例（错误输入）
    # --------------------

    def test_normalize_nonexistent_column(self, df_for_scaling: pd.DataFrame):
        """
        测试不存在的列

        期望：抛出 KeyError
        """
        with pytest.raises(KeyError):
            normalize_features(df_for_scaling, columns=['nonexistent'])


# =============================================================================
# Comparison Tests - 不同缩放方法比较
# =============================================================================

def test_standard_vs_normalize_output_range(df_for_scaling: pd.DataFrame):
    """
    比较 StandardScaler 和 MinMaxScaler 的输出范围

    期望：StandardScaler 输出范围不限于 [0,1]，MinMaxScaler 限于 [0,1]
    """
    std_df, _ = standardize_features(df_for_scaling, columns=['feature_large'])
    norm_df, _ = normalize_features(df_for_scaling, columns=['feature_large'])

    # MinMaxScaler 输出在 [0, 1]
    assert norm_df['feature_large_norm'].min() >= 0.0
    assert norm_df['feature_large_norm'].max() <= 1.0

    # StandardScaler 可能超出 [0, 1]
    # 对于 feature_large（范围 1000-10000），标准化后会有较大的值
    assert std_df['feature_large_std'].min() < 0 or std_df['feature_large_std'].max() > 1


def test_scaling_preserves_ranking(df_for_scaling: pd.DataFrame):
    """
    测试缩放是否保持数据的排序

    期望：缩放后数据的相对顺序保持不变
    """
    original = df_for_scaling['feature_small']

    std_df, _ = standardize_features(df_for_scaling, columns=['feature_small'])
    norm_df, _ = normalize_features(df_for_scaling, columns=['feature_small'])

    # 验证排序保持不变
    original_rank = original.rank()
    std_rank = std_df['feature_small_std'].rank()
    norm_rank = norm_df['feature_small_norm'].rank()

    pd.testing.assert_series_equal(original_rank, std_rank, check_names=False)
    pd.testing.assert_series_equal(original_rank, norm_rank, check_names=False)

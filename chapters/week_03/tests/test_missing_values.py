"""
Week 03 测试：缺失值处理（Missing Values）

测试覆盖：
1. analyze_missing() - 缺失值检测与分析
2. impute_missing() - 缺失值填充

测试用例类型：
- 正例：正常数据下的正确检测和填充
- 边界：全缺失列、空输入、单值
- 反例：错误策略填充、类型不匹配
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
        analyze_missing,
        impute_missing,
    )
except ImportError:
    pytest.skip("solution.py not yet created", allow_module_level=True)


# =============================================================================
# Test: analyze_missing() - 缺失值检测
# =============================================================================

class TestAnalyzeMissing:
    """测试缺失值检测函数"""

    # --------------------
    # 正例（happy path）
    # --------------------

    def test_analyze_missing_basic(self, df_with_missing: pd.DataFrame):
        """
        测试基本的缺失值检测

        期望：正确返回各列的缺失数量和缺失率
        """
        result = analyze_missing(df_with_missing)

        assert isinstance(result, pd.DataFrame), "返回值应该是 DataFrame"
        assert 'missing_count' in result.columns, "结果应包含 'missing_count'"
        assert 'missing_rate_%' in result.columns, "结果应包含 'missing_rate_%'"

        # 验证缺失数量计算正确
        assert result.loc['numeric_col', 'missing_count'] == 2
        assert result.loc['all_missing', 'missing_count'] == 6

    def test_analyze_missing_sorted(self, df_with_missing: pd.DataFrame):
        """
        测试缺失率排序

        期望：按缺失率降序排列
        """
        result = analyze_missing(df_with_missing)

        # all_missing 应该排在最前面（缺失率 100%）
        assert result.index[0] == 'all_missing', "全缺失列应排在首位"

    def test_analyze_missing_excludes_complete(self, df_with_missing: pd.DataFrame):
        """
        测试不包含无缺失的列

        期望：结果中只包含有缺失的列
        """
        result = analyze_missing(df_with_missing)

        # no_missing 列没有缺失值，不应该出现在结果中
        assert 'no_missing' not in result.index

    # --------------------
    # 边界情况
    # --------------------

    def test_analyze_no_missing(self):
        """
        测试无缺失值的情况

        期望：返回空 DataFrame
        """
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        result = analyze_missing(df)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0, "无缺失时应返回空 DataFrame"

    def test_analyze_all_missing(self):
        """
        测试全缺失的情况

        期望：正确返回全缺失列
        """
        df = pd.DataFrame({'all_nan': [np.nan, np.nan, np.nan]})
        result = analyze_missing(df)

        assert len(result) == 1
        assert result.loc['all_nan', 'missing_count'] == 3
        assert result.loc['all_nan', 'missing_rate_%'] == 100.0

    def test_analyze_empty_dataframe(self):
        """
        测试空 DataFrame

        期望：返回空 DataFrame
        """
        result = analyze_missing(pd.DataFrame())

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


# =============================================================================
# Test: impute_missing() - 缺失值填充
# =============================================================================

class TestImputeMissing:
    """测试缺失值填充函数"""

    # --------------------
    # 正例（happy path）
    # --------------------

    def test_impute_mean_strategy(self, df_with_missing: pd.DataFrame):
        """
        测试均值填充策略

        期望：缺失值被均值填充，填充后无缺失
        """
        result = impute_missing(df_with_missing, column='numeric_col', strategy='mean')

        # 验证无缺失值
        assert not result['numeric_col'].isna().any(), "填充后不应有缺失值"

        # 验证返回的是 DataFrame
        assert isinstance(result, pd.DataFrame)

    def test_impute_median_strategy(self, df_with_missing: pd.DataFrame):
        """
        测试中位数填充策略

        期望：缺失值被中位数填充，填充后无缺失
        """
        result = impute_missing(df_with_missing, column='numeric_col', strategy='median')

        # 验证无缺失值
        assert not result['numeric_col'].isna().any(), "填充后不应有缺失值"

    def test_impute_groupby_strategy(self, sample_df: pd.DataFrame):
        """
        测试分组填充策略

        期望：按指定列分组后填充
        """
        # 确保有缺失值
        df = sample_df.copy()
        df.loc[0, 'income'] = np.nan

        result = impute_missing(df, column='income', strategy='median', group_by='user_level')

        # 验证无缺失值
        assert not result['income'].isna().any(), "分组填充后不应有缺失值"

    def test_impute_preserves_other_columns(self, df_with_missing: pd.DataFrame):
        """
        测试填充时保留其他列

        期望：未指定的列保持不变
        """
        original_other = df_with_missing['string_col'].copy()
        result = impute_missing(df_with_missing, column='numeric_col', strategy='mean')

        # 其他列应该保持不变
        pd.testing.assert_series_equal(result['string_col'], original_other)

    # --------------------
    # 边界情况
    # --------------------

    def test_impute_no_missing(self, series_no_missing: pd.Series):
        """
        测试无缺失值的填充

        期望：数据保持不变
        """
        df = series_no_missing.to_frame('col')
        result = impute_missing(df, column='col', strategy='mean')

        # 验证数据未改变
        pd.testing.assert_series_equal(result['col'], df['col'])

    def test_impute_all_missing_groupby(self):
        """
        测试分组填充时某组全缺失

        期望：正确处理或保持缺失
        """
        df = pd.DataFrame({
            'value': [1.0, 2.0, np.nan, np.nan],
            'group': ['A', 'A', 'B', 'B']
        })

        # B 组全缺失，填充后应该还是缺失或保持原样
        result = impute_missing(df, column='value', strategy='mean', group_by='group')

        # 验证 A 组已填充
        assert not result.loc[result['group'] == 'A', 'value'].isna().any()

    # --------------------
    # 反例（错误输入）
    # --------------------

    def test_impute_invalid_strategy(self, df_with_missing: pd.DataFrame):
        """
        测试无效策略

        期望：抛出 ValueError 或 KeyError
        """
        with pytest.raises((ValueError, KeyError)):
            impute_missing(df_with_missing, column='numeric_col', strategy='invalid_strategy')

    def test_impute_nonexistent_column(self, df_with_missing: pd.DataFrame):
        """
        测试不存在的列

        期望：抛出 KeyError
        """
        with pytest.raises(KeyError):
            impute_missing(df_with_missing, column='nonexistent', strategy='mean')


# =============================================================================
# Parametrized Tests - 参数化测试
# =============================================================================

@pytest.mark.parametrize("strategy", ['mean', 'median', 'mode'])
def test_impute_strategies_parametrized(strategy: str, df_with_missing: pd.DataFrame):
    """
    参数化测试：各种填充策略

    验证不同策略都能正确填充缺失值
    """
    result = impute_missing(df_with_missing, column='numeric_col', strategy=strategy)

    # 所有策略都应该消除缺失值
    assert not result['numeric_col'].isna().any(), f"策略 {strategy} 应该消除所有缺失值"

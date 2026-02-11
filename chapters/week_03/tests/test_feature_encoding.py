"""
Week 03 测试：特征编码（Feature Encoding）

测试覆盖：
1. encode_categorical() - OneHotEncoder 编码 nominal 类别

测试用例类型：
- 正例：OneHotEncoder 正确编码 nominal 类别
- 边界：训练集未见过的类别
- 反例：对数值列误用编码
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
        encode_categorical,
    )
except ImportError:
    pytest.skip("solution.py not yet created", allow_module_level=True)


# =============================================================================
# Test: encode_categorical() - OneHotEncoder
# =============================================================================

class TestEncodeCategorical:
    """测试 OneHotEncoder 编码"""

    # --------------------
    # 正例（happy path）
    # --------------------

    def test_encode_categorical_basic(self, df_for_encoding: pd.DataFrame):
        """
        测试基本的 one-hot 编码

        期望：nominal 类别被正确编码为二元列
        """
        result = encode_categorical(df_for_encoding, column='nominal_cat')

        assert isinstance(result, tuple), "返回值应该是元组"
        assert len(result) == 2, "元组应该包含两个元素"

        result_df, encoder = result
        assert isinstance(result_df, pd.DataFrame), "第一个元素应该是 DataFrame"
        assert encoder is not None, "第二个元素应该是 encoder"

    def test_encode_categorical_creates_columns(self, df_for_encoding: pd.DataFrame):
        """
        测试编码创建新列

        期望：为每个类别创建新列
        """
        result_df, _ = encode_categorical(df_for_encoding, column='nominal_cat')

        # 验证创建了新的编码列
        expected_cols = ['nominal_cat_北京', 'nominal_cat_上海', 'nominal_cat_深圳', 'nominal_cat_广州']
        for col in expected_cols:
            assert col in result_df.columns, f"应该创建列: {col}"

    def test_encode_categorical_binary_values(self, df_for_encoding: pd.DataFrame):
        """
        测试编码后的值为二元

        期望：所有值都是 0 或 1
        """
        result_df, _ = encode_categorical(df_for_encoding, column='nominal_cat')

        # 获取编码列
        encode_cols = [c for c in result_df.columns if c.startswith('nominal_cat_')]

        # 验证所有值都是 0 或 1
        for col in encode_cols:
            unique_vals = set(result_df[col].unique())
            assert unique_vals.issubset({0, 1}), f"{col} 的值应该是 0 或 1"

    def test_encode_categorical_one_per_row(self, df_for_encoding: pd.DataFrame):
        """
        测试每行只有一个 1

        期望：每行的编码列之和为 1
        """
        result_df, _ = encode_categorical(df_for_encoding, column='nominal_cat')

        # 获取编码列
        encode_cols = [c for c in result_df.columns if c.startswith('nominal_cat_')]

        # 验证每行只有一个 1
        row_sums = result_df[encode_cols].sum(axis=1)
        assert (row_sums == 1).all(), "每行应该只有一个 1"

    def test_encode_categorical_preserves_original(self, df_for_encoding: pd.DataFrame):
        """
        测试保留原始列

        期望：原始列保持不变
        """
        original = df_for_encoding['nominal_cat'].copy()
        result_df, _ = encode_categorical(df_for_encoding, column='nominal_cat')

        # 原始列应该保持不变
        pd.testing.assert_series_equal(result_df['nominal_cat'], original)

    # --------------------
    # 边界情况
    # --------------------

    def test_encode_single_category(self):
        """
        测试单类别的编码

        期望：返回单列
        """
        df = pd.DataFrame({'cat': ['A', 'A', 'A', 'A']})
        result_df, _ = encode_categorical(df, column='cat')

        # 单类别应该只有 1 列
        assert 'cat_A' in result_df.columns

    # --------------------
    # 反例（错误输入）
    # --------------------

    def test_encode_nonexistent_column(self, df_for_encoding: pd.DataFrame):
        """
        测试不存在的列

        期望：抛出 KeyError
        """
        with pytest.raises(KeyError):
            encode_categorical(df_for_encoding, column='nonexistent')

    def test_encode_numeric_column(self, df_for_encoding: pd.DataFrame):
        """
        测试对数值列进行编码

        期望：可以执行，但可能无意义
        """
        # 数值列也可以被编码，每个唯一值成为一个类别
        result_df, _ = encode_categorical(df_for_encoding, column='numeric')

        # 验证创建了编码列
        encode_cols = [c for c in result_df.columns if c.startswith('numeric_')]
        assert len(encode_cols) > 0


# =============================================================================
# Integration Tests - 集成测试
# =============================================================================

def test_encoding_workflow(df_for_encoding: pd.DataFrame):
    """
    测试编码的完整流程

    从编码到验证
    """
    # 1. 编码
    result_df, encoder = encode_categorical(df_for_encoding, column='nominal_cat')

    # 2. 验证结果
    assert isinstance(result_df, pd.DataFrame)
    assert encoder is not None

    # 3. 验证编码列
    encode_cols = [c for c in result_df.columns if c.startswith('nominal_cat_')]
    assert len(encode_cols) == df_for_encoding['nominal_cat'].nunique()

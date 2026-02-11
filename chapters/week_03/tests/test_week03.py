"""
Week 03 综合测试：数据清洗与准备

本文件包含 Week 03 的综合测试，覆盖以下主题：
1. 缺失值处理（MCAR/MAR/MNAR 机制、填充策略）
2. 异常值检测与处理（IQR/Z-score、业务规则）
3. 特征缩放（StandardScaler/MinMaxScaler）
4. 特征编码（OneHotEncoder）
5. 清洗决策日志（记录与报告）

测试用例类型：
- 正例（happy path）：正常数据下的预期行为
- 边界：空输入、极值、特殊情况
- 反例：错误输入、无效参数
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
        # 缺失值处理
        analyze_missing,
        impute_missing,
        # 异常值检测
        detect_outliers_iqr,
        classify_outlier,
        handle_outliers,
        # 特征缩放
        standardize_features,
        normalize_features,
        # 特征编码
        encode_categorical,
        # 清洗日志
        create_cleaning_log,
        cleaning_pipeline,
    )
except ImportError as e:
    pytest.skip(f"solution.py not yet created or incomplete: {e}", allow_module_level=True)


# =============================================================================
# Smoke Tests - 快速冒烟测试
# =============================================================================

class TestSmoke:
    """冒烟测试：快速验证主要功能是否可用"""

    def test_missing_value_analysis(self, sample_df: pd.DataFrame):
        """测试缺失值分析功能"""
        result = analyze_missing(sample_df)
        assert isinstance(result, pd.DataFrame)

    def test_outlier_detection(self, df_with_outliers: pd.DataFrame):
        """测试异常值检测功能"""
        result = detect_outliers_iqr(df_with_outliers, column='normal_with_outliers')
        assert isinstance(result, pd.Series)

    def test_standardization(self, df_for_scaling: pd.DataFrame):
        """测试标准化功能"""
        result_df, scaler = standardize_features(df_for_scaling, columns=['feature_small'])
        assert isinstance(result_df, pd.DataFrame)

    def test_normalization(self, df_for_scaling: pd.DataFrame):
        """测试归一化功能"""
        result_df, scaler = normalize_features(df_for_scaling, columns=['feature_small'])
        assert isinstance(result_df, pd.DataFrame)

    def test_encoding(self, df_for_encoding: pd.DataFrame):
        """测试编码功能"""
        result_df, encoder = encode_categorical(df_for_encoding, column='nominal_cat')
        assert isinstance(result_df, pd.DataFrame)

    def test_cleaning_pipeline(self, sample_df: pd.DataFrame):
        """测试清洗流程"""
        result_df, log = cleaning_pipeline(sample_df)
        assert isinstance(result_df, pd.DataFrame)
        assert isinstance(log, str)


# =============================================================================
# Integration Tests - 集成测试
# =============================================================================

class TestIntegration:
    """集成测试：验证多个组件协同工作"""

    def test_full_preprocessing_workflow(self, sample_df: pd.DataFrame):
        """
        测试完整的预处理流程

        缺失值填充 -> 异常值处理 -> 特征缩放 -> 编码
        """
        df = sample_df.copy()

        # 1. 缺失值填充
        df = impute_missing(df, column='income', strategy='median')
        assert not df['income'].isna().any()

        # 2. 异常值处理
        df = handle_outliers(df, column='age')
        assert 'outlier_category' in df.columns

        # 3. 特征缩放
        df, _ = standardize_features(df, columns=['score'])
        assert 'score_std' in df.columns

    def test_pipeline_end_to_end(self):
        """
        端到端测试：从原始数据到清洗后数据
        """
        # 创建包含各种问题的测试数据
        df = pd.DataFrame({
            'age': [25, 30, np.nan, 35, 40, 150],  # 有缺失和异常值
            'income': [5000, 6000, 7000, np.nan, 9000, 10000],
            'city': ['北京', '上海', '北京', '上海', '深圳', '北京'],
            'total_spend': [100, 200, 300, 400, 500, 50000],
        })

        # 执行完整流程
        result_df, log = cleaning_pipeline(df)

        # 验证结果
        assert isinstance(result_df, pd.DataFrame)
        assert isinstance(log, str)
        assert len(log) > 0


# =============================================================================
# Edge Case Tests - 边界情况测试
# =============================================================================

class TestEdgeCases:
    """边界情况测试"""

    def test_empty_dataframe(self):
        """测试空 DataFrame"""
        df = pd.DataFrame()

        # analyze_missing 应该返回空 DataFrame
        result = analyze_missing(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_single_row(self):
        """测试单行数据"""
        df = pd.DataFrame({'a': [1], 'b': ['test']})

        # 各种操作应该能处理单行数据
        result = analyze_missing(df)
        assert isinstance(result, pd.DataFrame)

    def test_all_missing_column(self):
        """测试全缺失列"""
        df = pd.DataFrame({'all_nan': [np.nan, np.nan, np.nan]})

        result = analyze_missing(df)
        assert len(result) == 1
        assert result.loc['all_nan', 'missing_rate_%'] == 100.0

    def test_constant_column(self, df_for_scaling: pd.DataFrame):
        """测试常数列"""
        # 常数列的标准化应该抛出错误
        with pytest.raises((ValueError, ZeroDivisionError)):
            standardize_features(df_for_scaling, columns=['feature_constant'])

        # 常数列的归一化应该可以执行
        result_df, _ = normalize_features(df_for_scaling, columns=['feature_constant'])
        assert 'feature_constant_norm' in result_df.columns


# =============================================================================
# Data Quality Tests - 数据质量测试
# =============================================================================

class TestDataQuality:
    """数据质量测试"""

    def test_standardization_properties(self, df_for_scaling: pd.DataFrame):
        """
        测试标准化的数学性质

        标准化后：均值为 0，标准差为 1（使用总体标准差 ddof=0）
        """
        result_df, _ = standardize_features(df_for_scaling, columns=['feature_small'])

        std_col = result_df['feature_small_std']
        assert std_col.mean() == pytest.approx(0.0, abs=1e-10)
        # 使用总体标准差（ddof=0）来验证，因为 StandardScaler 使用总体标准差
        assert std_col.std(ddof=0) == pytest.approx(1.0, rel=1e-5)

    def test_normalization_properties(self, df_for_scaling: pd.DataFrame):
        """
        测试归一化的数学性质

        归一化后：范围在 [0, 1]
        """
        result_df, _ = normalize_features(df_for_scaling, columns=['feature_small'])

        norm_col = result_df['feature_small_norm']
        assert norm_col.min() == pytest.approx(0.0, abs=1e-10)
        assert norm_col.max() == pytest.approx(1.0, abs=1e-10)

    def test_encoding_properties(self, df_for_encoding: pd.DataFrame):
        """
        测试编码的性质

        One-hot 编码：每行只有一个 1
        """
        result_df, _ = encode_categorical(df_for_encoding, column='nominal_cat')

        # 获取编码列
        encode_cols = [c for c in result_df.columns if c.startswith('nominal_cat_')]

        # 每行只有一个 1
        row_sums = result_df[encode_cols].sum(axis=1)
        assert (row_sums == 1).all()

    def test_imputation_eliminates_missing(self, df_with_missing: pd.DataFrame):
        """
        测试填充确实消除了缺失值
        """
        result = impute_missing(df_with_missing, column='numeric_col', strategy='mean')
        assert not result['numeric_col'].isna().any()


# =============================================================================
# Parametrized Tests - 参数化测试
# =============================================================================

@pytest.mark.parametrize("strategy", ['mean', 'median', 'mode'])
def test_all_impute_strategies(strategy: str, df_with_missing: pd.DataFrame):
    """
    参数化测试：所有填充策略
    """
    result = impute_missing(df_with_missing, column='numeric_col', strategy=strategy)
    assert not result['numeric_col'].isna().any()


@pytest.mark.parametrize("multiplier", [1.5, 2.0, 3.0])
def test_iqr_multipliers(multiplier: float, df_with_outliers: pd.DataFrame):
    """
    参数化测试：不同 IQR 倍数
    """
    result = detect_outliers_iqr(df_with_outliers, column='normal_with_outliers', multiplier=multiplier)
    assert isinstance(result, pd.Series)
    assert result.dtype == bool

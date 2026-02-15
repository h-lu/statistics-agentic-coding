"""
Week 02 测试：一页分布报告（One Page Distribution Report）

测试覆盖：
1. generate_descriptive_summary() - 生成描述统计摘要

由于 solution.py 主要是参考答案实现，这里的测试主要测试
实际存在的函数。
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
generate_descriptive_summary = getattr(solution, 'generate_descriptive_summary', None)


# =============================================================================
# Test: generate_descriptive_summary()
# =============================================================================

class TestGenerateDescriptiveSummary:
    """测试描述统计摘要生成函数"""

    def test_function_exists(self):
        """检查函数存在"""
        assert generate_descriptive_summary is not None, "generate_descriptive_summary 函数应存在"

    # --------------------
    # 正例（happy path）
    # --------------------

    def test_generate_basic_summary(self, sample_dataframe: pd.DataFrame):
        """
        测试基本摘要生成

        期望：应返回包含均值、中位数、标准差、IQR 的字典
        """
        result = generate_descriptive_summary(sample_dataframe)

        assert isinstance(result, dict)

        # 应该包含所有数值列的摘要
        numeric_cols = sample_dataframe.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            assert col in result, f"应包含列 '{col}' 的摘要"

            col_summary = result[col]
            assert isinstance(col_summary, dict)

            # 验证关键字段存在
            required_fields = ['mean', 'median', 'std', 'iqr']
            for field in required_fields:
                assert field in col_summary, f"列 '{col}' 应包含 '{field}'"

            # 验证数据类型正确
            assert isinstance(col_summary['mean'], (int, float))
            assert isinstance(col_summary['median'], (int, float))
            assert isinstance(col_summary['std'], (int, float))
            assert isinstance(col_summary['iqr'], (int, float))

    def test_generate_summary_correctness(self, sample_dataframe: pd.DataFrame):
        """
        测试摘要值的正确性

        期望：计算值应与 pandas 直接计算一致
        """
        result = generate_descriptive_summary(sample_dataframe)

        # 选择第一个数值列验证
        numeric_col = sample_dataframe.select_dtypes(include=['number']).columns[0]
        series = sample_dataframe[numeric_col]
        summary = result[numeric_col]

        # 验证均值
        expected_mean = series.mean()
        assert summary['mean'] == pytest.approx(expected_mean, rel=1e-5)

        # 验证中位数
        expected_median = series.median()
        assert summary['median'] == pytest.approx(expected_median, rel=1e-5)

        # 验证标准差
        expected_std = series.std()
        assert summary['std'] == pytest.approx(expected_std, rel=1e-5)

        # 验证 IQR
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        expected_iqr = q3 - q1
        assert summary['iqr'] == pytest.approx(expected_iqr, rel=1e-5)

    # --------------------
    # 边界情况
    # --------------------

    def test_generate_empty_dataframe(self):
        """
        测试空 DataFrame

        期望：应返回空字典或提示
        """
        empty_df = pd.DataFrame()
        result = generate_descriptive_summary(empty_df)

        assert isinstance(result, dict)
        assert len(result) == 0

    def test_generate_no_numeric_columns(self):
        """
        测试无数值列的 DataFrame

        期望：应返回空字典或警告
        """
        categorical_df = pd.DataFrame({
            'city': ['北京', '上海', '深圳'],
            'category': ['A', 'B', 'C'],
        })

        result = generate_descriptive_summary(categorical_df)

        assert isinstance(result, dict)
        assert len(result) == 0

    def test_generate_with_missing_values(self, dataframe_with_missing: pd.DataFrame):
        """
        测试包含缺失值的数据

        期望：应自动忽略缺失值
        """
        result = generate_descriptive_summary(dataframe_with_missing)

        # 应该能计算，不抛出异常
        assert isinstance(result, dict)

        # 验证结果不包含 NaN
        for col_summary in result.values():
            assert not np.isnan(col_summary['mean'])
            assert not np.isnan(col_summary['median'])


# =============================================================================
# Test: 使用 Penguins 数据集
# =============================================================================

class TestDescriptiveSummaryWithPenguins:
    """使用 seaborn Penguins 数据集的测试"""

    def test_generate_penguins_summary(self):
        """
        测试使用 Penguins 数据集生成摘要

        期望：应能正确处理真实数据集
        """
        try:
            import seaborn as sns
            penguins = sns.load_dataset("penguins")
        except ImportError:
            pytest.skip("seaborn 不可用")

        result = generate_descriptive_summary(penguins)

        assert isinstance(result, dict)

        # 应该包含数值列
        numeric_cols = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
        for col in numeric_cols:
            if col in result:
                assert 'mean' in result[col]
                assert 'median' in result[col]

    def test_penguins_body_mass_summary(self):
        """
        测试 Penguins 体重摘要

        期望：应返回正确的统计值
        """
        try:
            import seaborn as sns
            penguins = sns.load_dataset("penguins")
        except ImportError:
            pytest.skip("seaborn 不可用")

        result = generate_descriptive_summary(penguins)

        if 'body_mass_g' in result:
            summary = result['body_mass_g']

            # Penguins 体重均值应该大约在 4200g 左右
            assert 4000 < summary['mean'] < 4500

            # 标准差应该在 800g 左右
            assert 500 < summary['std'] < 1000

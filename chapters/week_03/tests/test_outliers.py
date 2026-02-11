"""
Week 03 测试：异常值检测与处理（Outliers）

测试覆盖：
1. detect_outliers_iqr() - IQR 方法检测异常值
2. classify_outlier() - 异常值分类（错误 vs VIP）
3. handle_outliers() - 异常值处理

测试用例类型：
- 正例：正常数据下正确识别异常值
- 边界：无异常值、全为异常值、常数列
- 反例：无效参数、空数据
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
        detect_outliers_iqr,
        classify_outlier,
        handle_outliers,
    )
except ImportError:
    pytest.skip("solution.py not yet created", allow_module_level=True)


# =============================================================================
# Test: detect_outliers_iqr() - IQR 方法
# =============================================================================

class TestDetectOutliersIQR:
    """测试 IQR 方法异常值检测"""

    # --------------------
    # 正例（happy path）
    # --------------------

    def test_detect_outliers_iqr_basic(self, df_with_outliers: pd.DataFrame):
        """
        测试 IQR 方法基本功能

        期望：正确识别超出 1.5*IQR 范围的异常值
        """
        result = detect_outliers_iqr(df_with_outliers, column='normal_with_outliers')

        assert isinstance(result, pd.Series), "返回值应该是 Series"
        assert result.dtype == bool, "结果应该是布尔类型"
        assert len(result) == len(df_with_outliers), "结果长度应与输入相同"

    def test_detect_outliers_iqr_finds_extremes(self, df_with_outliers: pd.DataFrame):
        """
        测试 IQR 方法识别极端值

        期望：100 和 150 应该被识别为异常值
        """
        result = detect_outliers_iqr(df_with_outliers, column='normal_with_outliers')

        # 应该检测到异常值
        assert result.sum() > 0, "应该检测到异常值"

        # 极端值位置（100 和 150 在索引 9 和 10）
        assert result.iloc[9] or result.iloc[10], "极端值应该被识别为异常值"

    def test_detect_outliers_iqr_multiplier(self, df_with_outliers: pd.DataFrame):
        """
        测试不同的 IQR 倍数

        期望：倍数越大，检测到的异常值越少
        """
        result_1_5 = detect_outliers_iqr(df_with_outliers, column='normal_with_outliers', multiplier=1.5)
        result_3_0 = detect_outliers_iqr(df_with_outliers, column='normal_with_outliers', multiplier=3.0)

        # 3.0 倍 IQR 应该比 1.5 倍检测到的异常值少或相等
        assert result_3_0.sum() <= result_1_5.sum()

    # --------------------
    # 边界情况
    # --------------------

    def test_detect_no_outliers(self, df_with_outliers: pd.DataFrame):
        """
        测试无异常值的情况

        期望：返回全 False
        """
        result = detect_outliers_iqr(df_with_outliers, column='all_normal')

        # 正常分布数据应该没有异常值
        assert not result.any(), "正常分布数据不应有异常值"

    def test_detect_constant_column(self, df_with_outliers: pd.DataFrame):
        """
        测试常数列的异常值检测

        期望：IQR 为 0，无异常值
        """
        result = detect_outliers_iqr(df_with_outliers, column='constant')

        # 常数列不应该有异常值
        assert not result.any(), "常数列不应有异常值"

    # --------------------
    # 反例（错误输入）
    # --------------------

    def test_detect_nonexistent_column(self, df_with_outliers: pd.DataFrame):
        """
        测试不存在的列

        期望：抛出 KeyError
        """
        with pytest.raises(KeyError):
            detect_outliers_iqr(df_with_outliers, column='nonexistent')


# =============================================================================
# Test: classify_outlier() - 异常值分类
# =============================================================================

class TestClassifyOutlier:
    """测试异常值分类函数"""

    # --------------------
    # 正例（happy path）
    # --------------------

    def test_classify_suspicious_negative(self):
        """
        测试负数分类为 suspicious

        期望：负数返回 'suspicious'
        """
        result = classify_outlier(-100.0)
        assert result == 'suspicious'

    def test_classify_vip_high_value(self):
        """
        测试高值分类为 VIP

        期望：大于 50000 返回 'VIP'
        """
        result = classify_outlier(60000.0)
        assert result == 'VIP'

    def test_classify_normal(self):
        """
        测试正常值分类

        期望：正常值返回 'normal'
        """
        result = classify_outlier(1000.0)
        assert result == 'normal'

    def test_classify_boundary_values(self):
        """
        测试边界值

        期望：0 为 normal，50000 为 VIP
        """
        assert classify_outlier(0.0) == 'normal'
        assert classify_outlier(50000.0) == 'VIP'


# =============================================================================
# Test: handle_outliers() - 异常值处理
# =============================================================================

class TestHandleOutliers:
    """测试异常值处理函数"""

    # --------------------
    # 正例（happy path）
    # --------------------

    def test_handle_outliers_removes_suspicious(self, df_with_outliers: pd.DataFrame):
        """
        测试删除可疑异常值

        期望：suspicious 类别的行被删除
        """
        # 添加一些负数异常值
        df = df_with_outliers.copy()
        df.loc[0, 'normal_with_outliers'] = -999

        original_len = len(df)
        result = handle_outliers(df, column='normal_with_outliers')

        assert isinstance(result, pd.DataFrame), "返回值应该是 DataFrame"
        assert len(result) <= original_len, "删除后行数应减少或不变"

        # 验证添加了 outlier_category 列
        assert 'outlier_category' in result.columns

    def test_handle_outliers_preserves_vip(self, df_with_outliers: pd.DataFrame):
        """
        测试保留 VIP 异常值

        期望：VIP 类别的行被保留
        """
        # 添加 VIP 异常值
        df = df_with_outliers.copy()
        df.loc[0, 'normal_with_outliers'] = 60000

        result = handle_outliers(df, column='normal_with_outliers')

        # VIP 应该被保留
        assert 'VIP' in result['outlier_category'].values

    # --------------------
    # 边界情况
    # --------------------

    def test_handle_no_outliers(self, df_with_outliers: pd.DataFrame):
        """
        测试无异常值时的处理

        期望：数据保持不变（除了添加分类列）
        """
        result = handle_outliers(df_with_outliers, column='all_normal')

        # 应该保留所有行（all_normal 列没有异常值）
        assert len(result) == len(df_with_outliers)

    def test_handle_all_normal(self):
        """
        测试全正常值

        期望：所有行标记为 normal
        """
        df = pd.DataFrame({'value': [100, 200, 300, 400, 500]})
        result = handle_outliers(df, column='value')

        assert (result['outlier_category'] == 'normal').all()


# =============================================================================
# Integration Tests - 集成测试
# =============================================================================

def test_outlier_detection_and_handling_workflow(df_with_outliers: pd.DataFrame):
    """
    测试异常值检测和处理的完整流程

    从检测到分类再到处理
    """
    # 1. 检测异常值
    outlier_mask = detect_outliers_iqr(df_with_outliers, column='normal_with_outliers')
    assert isinstance(outlier_mask, pd.Series)

    # 2. 处理异常值
    result = handle_outliers(df_with_outliers, column='normal_with_outliers')
    assert isinstance(result, pd.DataFrame)

    # 3. 验证结果
    assert 'outlier_category' in result.columns

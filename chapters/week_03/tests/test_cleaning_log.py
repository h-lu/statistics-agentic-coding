"""
Week 03 测试：清洗决策日志（Cleaning Log）

测试覆盖：
1. create_cleaning_log() - 生成清洗决策日志
2. cleaning_pipeline() - 完整清洗流程

测试用例类型：
- 正例：决策记录包含所有必需字段
- 正例：生成的 Markdown 格式正确
- 边界：空决策列表、单条决策
- 反例：缺少必需字段的决策记录
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
        create_cleaning_log,
        cleaning_pipeline,
    )
except ImportError:
    pytest.skip("solution.py not yet created", allow_module_level=True)


# =============================================================================
# Test: create_cleaning_log() - 生成清洗日志
# =============================================================================

class TestCreateCleaningLog:
    """测试清洗日志生成函数"""

    # --------------------
    # 正例（happy path）
    # --------------------

    def test_create_log_basic(self, sample_cleaning_decisions: list[dict[str, Any]]):
        """
        测试基本的日志生成

        期望：返回格式化的 Markdown 字符串
        """
        result = create_cleaning_log(
            sample_cleaning_decisions,
            initial_shape=(100, 10),
            final_shape=(95, 12)
        )

        assert isinstance(result, str), "返回值应该是字符串"
        assert len(result) > 0, "日志内容不应为空"

        # 验证包含标题
        assert "#" in result or "清洗" in result, "应包含标题"

    def test_create_log_contains_operations(self, sample_cleaning_decisions: list[dict[str, Any]]):
        """
        测试日志包含所有操作

        期望：每条操作都在日志中
        """
        result = create_cleaning_log(
            sample_cleaning_decisions,
            initial_shape=(100, 10),
            final_shape=(95, 12)
        )

        # 验证包含操作信息（使用 'field' 作为操作标识）
        for decision in sample_cleaning_decisions:
            assert decision['field'] in result, f"应包含字段: {decision['field']}"

    def test_create_log_contains_shape_info(self, sample_cleaning_decisions: list[dict[str, Any]]):
        """
        测试日志包含形状信息

        期望：包含初始和最终形状
        """
        result = create_cleaning_log(
            sample_cleaning_decisions,
            initial_shape=(100, 10),
            final_shape=(95, 12)
        )

        # 验证包含形状信息
        assert "100" in result or "95" in result, "应包含形状信息"

    def test_create_log_markdown_formatting(self, sample_cleaning_decisions: list[dict[str, Any]]):
        """
        测试 Markdown 格式

        期望：正确使用 Markdown 语法
        """
        result = create_cleaning_log(
            sample_cleaning_decisions,
            initial_shape=(100, 10),
            final_shape=(95, 12)
        )

        # 验证使用 Markdown 标题
        assert "#" in result, "应使用 Markdown 标题"

        # 验证使用列表格式
        assert "- " in result or "* " in result or "**" in result, "应使用列表或加粗格式"

    # --------------------
    # 边界情况
    # --------------------

    def test_create_log_empty_operations(self):
        """
        测试空操作列表

        期望：返回基本信息的日志
        """
        result = create_cleaning_log(
            [],
            initial_shape=(100, 10),
            final_shape=(100, 10)
        )

        assert isinstance(result, str)
        # 应该包含基本信息
        assert "100" in result


# =============================================================================
# Test: cleaning_pipeline() - 完整清洗流程
# =============================================================================

class TestCleaningPipeline:
    """测试完整清洗流程"""

    # --------------------
    # 正例（happy path）
    # --------------------

    def test_pipeline_returns_tuple(self, sample_df: pd.DataFrame):
        """
        测试流程返回元组

        期望：返回 (DataFrame, log)
        """
        result = cleaning_pipeline(sample_df)

        assert isinstance(result, tuple), "返回值应该是元组"
        assert len(result) == 2, "元组应该包含两个元素"

        df_result, log = result
        assert isinstance(df_result, pd.DataFrame), "第一个元素应该是 DataFrame"
        assert isinstance(log, str), "第二个元素应该是字符串（日志）"

    def test_pipeline_cleans_data(self, sample_df: pd.DataFrame):
        """
        测试流程清洗数据

        期望：数据被清洗（缺失值填充、异常值处理等）
        """
        # 确保有缺失值和异常值
        df = sample_df.copy()
        df.loc[0, 'income'] = np.nan
        df.loc[1, 'income'] = -999  # 异常值

        result_df, _ = cleaning_pipeline(df)

        # 验证返回了 DataFrame
        assert isinstance(result_df, pd.DataFrame)

        # 验证添加了新列
        assert 'outlier_category' in result_df.columns

    def test_pipeline_generates_log(self, sample_df: pd.DataFrame):
        """
        测试流程生成日志

        期望：生成非空的 Markdown 日志
        """
        _, log = cleaning_pipeline(sample_df)

        assert isinstance(log, str)
        assert len(log) > 0
        assert "#" in log, "日志应使用 Markdown 格式"

    # --------------------
    # 边界情况
    # --------------------

    def test_pipeline_empty_dataframe(self):
        """
        测试空 DataFrame

        期望：正确处理或抛出错误
        """
        df = pd.DataFrame()

        # 可能抛出错误或返回空结果
        try:
            result_df, log = cleaning_pipeline(df)
            assert len(result_df) == 0
        except (ValueError, IndexError):
            pass  # 也接受抛出错误


# =============================================================================
# Integration Tests - 集成测试
# =============================================================================

def test_full_cleaning_workflow():
    """
    测试完整的清洗工作流程

    从原始数据到清洗后数据 + 日志
    """
    # 创建测试数据
    df = pd.DataFrame({
        'age': [25, 30, np.nan, 35, 40],
        'income': [5000, 6000, 7000, -999, 9000],
        'city': ['北京', '上海', '北京', '上海', '深圳'],
        'total_spend': [100, 200, 300, 400, 50000],
    })

    # 执行清洗流程
    result_df, log = cleaning_pipeline(df)

    # 验证结果
    assert isinstance(result_df, pd.DataFrame)
    assert isinstance(log, str)

    # 验证日志包含关键信息
    assert "#" in log
    assert len(log) > 0

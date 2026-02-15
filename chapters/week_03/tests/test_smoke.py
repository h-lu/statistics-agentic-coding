"""
Week 03 烟雾测试（Smoke Tests）

这些测试验证基本的函数存在性和签名正确性。
如果这些测试失败，说明 solution.py 的基本结构有问题。
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path
from typing import Any, List

import numpy as np
import pandas as pd
import pytest

# 添加 starter_code 到 Python 路径
starter_code_path = Path(__file__).parent.parent / "starter_code"
if str(starter_code_path) not in sys.path:
    sys.path.insert(0, str(starter_code_path))

solution = pytest.importorskip("solution")


# =============================================================================
# 模块导入测试
# =============================================================================

class TestModuleImport:
    """测试模块是否可以正确导入"""

    def test_module_exists(self):
        """测试 solution 模块是否存在"""
        assert solution is not None, "solution 模块应该存在"

    def test_module_has_docstring(self):
        """测试模块是否有文档字符串"""
        assert solution.__doc__ is not None, "solution 模块应该有文档字符串"


# =============================================================================
# 缺失值处理函数签名测试
# =============================================================================

class TestMissingValueFunctionSignatures:
    """测试缺失值处理函数的签名"""

    def test_detect_missing_pattern_exists(self):
        """测试 detect_missing_pattern 函数是否存在"""
        assert hasattr(solution, 'detect_missing_pattern'), \
            "solution 应该有 detect_missing_pattern 函数"

    def test_detect_missing_pattern_signature(self):
        """测试 detect_missing_pattern 函数签名"""
        func = getattr(solution, 'detect_missing_pattern', None)
        if func is None:
            pytest.skip("detect_missing_pattern 函数不存在")

        sig = inspect.signature(func)
        params = list(sig.parameters.keys())

        assert 'df' in params, "detect_missing_pattern 应该有 df 参数"

    def test_handle_missing_strategy_exists(self):
        """测试 handle_missing_strategy 函数是否存在"""
        assert hasattr(solution, 'handle_missing_strategy'), \
            "solution 应该有 handle_missing_strategy 函数"

    def test_handle_missing_strategy_signature(self):
        """测试 handle_missing_strategy 函数签名"""
        func = getattr(solution, 'handle_missing_strategy', None)
        if func is None:
            pytest.skip("handle_missing_strategy 函数不存在")

        sig = inspect.signature(func)
        params = list(sig.parameters.keys())

        assert 'df' in params, "handle_missing_strategy 应该有 df 参数"
        assert 'column' in params, "handle_missing_strategy 应该有 column 参数"
        assert 'strategy' in params, "handle_missing_strategy 应该有 strategy 参数"

    def test_missing_summary_exists(self):
        """测试 missing_summary 函数是否存在"""
        assert hasattr(solution, 'missing_summary'), \
            "solution 应该有 missing_summary 函数"


# =============================================================================
# 异常值检测函数签名测试
# =============================================================================

class TestOutlierDetectionFunctionSignatures:
    """测试异常值检测函数的签名"""

    def test_detect_outliers_iqr_exists(self):
        """测试 detect_outliers_iqr 函数是否存在"""
        assert hasattr(solution, 'detect_outliers_iqr'), \
            "solution 应该有 detect_outliers_iqr 函数"

    def test_detect_outliers_iqr_signature(self):
        """测试 detect_outliers_iqr 函数签名"""
        func = getattr(solution, 'detect_outliers_iqr', None)
        if func is None:
            pytest.skip("detect_outliers_iqr 函数不存在")

        sig = inspect.signature(func)
        params = list(sig.parameters.keys())

        assert 'series' in params, "detect_outliers_iqr 应该有 series 参数"

    def test_detect_outliers_zscore_exists(self):
        """测试 detect_outliers_zscore 函数是否存在"""
        assert hasattr(solution, 'detect_outliers_zscore'), \
            "solution 应该有 detect_outliers_zscore 函数"

    def test_detect_outliers_zscore_signature(self):
        """测试 detect_outliers_zscore 函数签名"""
        func = getattr(solution, 'detect_outliers_zscore', None)
        if func is None:
            pytest.skip("detect_outliers_zscore 函数不存在")

        sig = inspect.signature(func)
        params = list(sig.parameters.keys())

        assert 'series' in params, "detect_outliers_zscore 应该有 series 参数"


# =============================================================================
# 数据转换函数签名测试
# =============================================================================

class TestDataTransformationFunctionSignatures:
    """测试数据转换函数的签名"""

    def test_standardize_data_exists(self):
        """测试 standardize_data 函数是否存在"""
        assert hasattr(solution, 'standardize_data'), \
            "solution 应该有 standardize_data 函数"

    def test_standardize_data_signature(self):
        """测试 standardize_data 函数签名"""
        func = getattr(solution, 'standardize_data', None)
        if func is None:
            pytest.skip("standardize_data 函数不存在")

        sig = inspect.signature(func)
        params = list(sig.parameters.keys())

        assert 'df' in params, "standardize_data 应该有 df 参数"

    def test_normalize_data_exists(self):
        """测试 normalize_data 函数是否存在"""
        assert hasattr(solution, 'normalize_data'), \
            "solution 应该有 normalize_data 函数"

    def test_normalize_data_signature(self):
        """测试 normalize_data 函数签名"""
        func = getattr(solution, 'normalize_data', None)
        if func is None:
            pytest.skip("normalize_data 函数不存在")

        sig = inspect.signature(func)
        params = list(sig.parameters.keys())

        assert 'df' in params, "normalize_data 应该有 df 参数"

    def test_log_transform_exists(self):
        """测试 log_transform 函数是否存在"""
        assert hasattr(solution, 'log_transform'), \
            "solution 应该有 log_transform 函数"

    def test_log_transform_signature(self):
        """测试 log_transform 函数签名"""
        func = getattr(solution, 'log_transform', None)
        if func is None:
            pytest.skip("log_transform 函数不存在")

        sig = inspect.signature(func)
        params = list(sig.parameters.keys())

        assert 'series' in params, "log_transform 应该有 series 参数"


# =============================================================================
# 特征编码函数签名测试
# =============================================================================

class TestFeatureEncodingFunctionSignatures:
    """测试特征编码函数的签名"""

    def test_one_hot_encode_exists(self):
        """测试 one_hot_encode 函数是否存在"""
        assert hasattr(solution, 'one_hot_encode'), \
            "solution 应该有 one_hot_encode 函数"

    def test_one_hot_encode_signature(self):
        """测试 one_hot_encode 函数签名"""
        func = getattr(solution, 'one_hot_encode', None)
        if func is None:
            pytest.skip("one_hot_encode 函数不存在")

        sig = inspect.signature(func)
        params = list(sig.parameters.keys())

        assert 'df' in params, "one_hot_encode 应该有 df 参数"
        assert 'column' in params, "one_hot_encode 应该有 column 参数"

    def test_label_encode_exists(self):
        """测试 label_encode 函数是否存在"""
        assert hasattr(solution, 'label_encode'), \
            "solution 应该有 label_encode 函数"

    def test_label_encode_signature(self):
        """测试 label_encode 函数签名"""
        func = getattr(solution, 'label_encode', None)
        if func is None:
            pytest.skip("label_encode 函数不存在")

        sig = inspect.signature(func)
        params = list(sig.parameters.keys())

        assert 'df' in params, "label_encode 应该有 df 参数"
        assert 'column' in params, "label_encode 应该有 column 参数"


# =============================================================================
# 示例函数签名测试
# =============================================================================

class TestExampleFunctionSignatures:
    """测试示例函数的签名"""

    def test_exercise_1_missing_mechanism_exists(self):
        """测试 exercise_1_missing_mechanism 函数是否存在"""
        assert hasattr(solution, 'exercise_1_missing_mechanism'), \
            "solution 应该有 exercise_1_missing_mechanism 函数"

    def test_exercise_2_missing_strategies_exists(self):
        """测试 exercise_2_missing_strategies 函数是否存在"""
        assert hasattr(solution, 'exercise_2_missing_strategies'), \
            "solution 应该有 exercise_2_missing_strategies 函数"

    def test_exercise_3_outlier_detection_exists(self):
        """测试 exercise_3_outlier_detection 函数是否存在"""
        assert hasattr(solution, 'exercise_3_outlier_detection'), \
            "solution 应该有 exercise_3_outlier_detection 函数"

    def test_exercise_4_data_transformation_exists(self):
        """测试 exercise_4_data_transformation 函数是否存在"""
        assert hasattr(solution, 'exercise_4_data_transformation'), \
            "solution 应该有 exercise_4_data_transformation 函数"

    def test_exercise_5_feature_encoding_exists(self):
        """测试 exercise_5_feature_encoding 函数是否存在"""
        assert hasattr(solution, 'exercise_5_feature_encoding'), \
            "solution 应该有 exercise_5_feature_encoding 函数"

    def test_exercise_6_cleaning_log_exists(self):
        """测试 exercise_6_cleaning_log 函数是否存在"""
        assert hasattr(solution, 'exercise_6_cleaning_log'), \
            "solution 应该有 exercise_6_cleaning_log 函数"

    def test_main_exists(self):
        """测试 main 函数是否存在"""
        assert hasattr(solution, 'main'), \
            "solution 应该有 main 函数"


# =============================================================================
# 函数可调用性测试
# =============================================================================

class TestFunctionCallability:
    """测试函数是否可以被正确调用"""

    def test_detect_missing_pattern_callable(self):
        """测试 detect_missing_pattern 是否可调用"""
        func = getattr(solution, 'detect_missing_pattern', None)
        if func is None:
            pytest.skip("detect_missing_pattern 函数不存在")

        assert callable(func), "detect_missing_pattern 应该是可调用的"

    def test_handle_missing_strategy_callable(self):
        """测试 handle_missing_strategy 是否可调用"""
        func = getattr(solution, 'handle_missing_strategy', None)
        if func is None:
            pytest.skip("handle_missing_strategy 函数不存在")

        assert callable(func), "handle_missing_strategy 应该是可调用的"

    def test_detect_outliers_iqr_callable(self):
        """测试 detect_outliers_iqr 是否可调用"""
        func = getattr(solution, 'detect_outliers_iqr', None)
        if func is None:
            pytest.skip("detect_outliers_iqr 函数不存在")

        assert callable(func), "detect_outliers_iqr 应该是可调用的"

    def test_detect_outliers_zscore_callable(self):
        """测试 detect_outliers_zscore 是否可调用"""
        func = getattr(solution, 'detect_outliers_zscore', None)
        if func is None:
            pytest.skip("detect_outliers_zscore 函数不存在")

        assert callable(func), "detect_outliers_zscore 应该是可调用的"

    def test_standardize_data_callable(self):
        """测试 standardize_data 是否可调用"""
        func = getattr(solution, 'standardize_data', None)
        if func is None:
            pytest.skip("standardize_data 函数不存在")

        assert callable(func), "standardize_data 应该是可调用的"

    def test_normalize_data_callable(self):
        """测试 normalize_data 是否可调用"""
        func = getattr(solution, 'normalize_data', None)
        if func is None:
            pytest.skip("normalize_data 函数不存在")

        assert callable(func), "normalize_data 应该是可调用的"

    def test_log_transform_callable(self):
        """测试 log_transform 是否可调用"""
        func = getattr(solution, 'log_transform', None)
        if func is None:
            pytest.skip("log_transform 函数不存在")

        assert callable(func), "log_transform 应该是可调用的"

    def test_one_hot_encode_callable(self):
        """测试 one_hot_encode 是否可调用"""
        func = getattr(solution, 'one_hot_encode', None)
        if func is None:
            pytest.skip("one_hot_encode 函数不存在")

        assert callable(func), "one_hot_encode 应该是可调用的"

    def test_label_encode_callable(self):
        """测试 label_encode 是否可调用"""
        func = getattr(solution, 'label_encode', None)
        if func is None:
            pytest.skip("label_encode 函数不存在")

        assert callable(func), "label_encode 应该是可调用的"

    def test_main_callable(self):
        """测试 main 是否可调用"""
        func = getattr(solution, 'main', None)
        if func is None:
            pytest.skip("main 函数不存在")

        assert callable(func), "main 应该是可调用的"


# =============================================================================
# 基本功能测试
# =============================================================================

class TestBasicFunctionality:
    """测试基本功能是否正常工作"""

    def test_detect_missing_pattern_returns_dict(self):
        """测试 detect_missing_pattern 返回字典"""
        func = getattr(solution, 'detect_missing_pattern', None)
        if func is None:
            pytest.skip("detect_missing_pattern 函数不存在")

        test_df = pd.DataFrame({
            'a': [1, 2, np.nan, 4],
            'b': [1, 2, 3, 4],
        })
        result = func(test_df)

        assert isinstance(result, dict), "detect_missing_pattern 应该返回字典"

    def test_detect_outliers_iqr_returns_series(self):
        """测试 detect_outliers_iqr 返回 Series"""
        func = getattr(solution, 'detect_outliers_iqr', None)
        if func is None:
            pytest.skip("detect_outliers_iqr 函数不存在")

        test_series = pd.Series([1, 2, 3, 4, 5, 100])
        result = func(test_series)

        assert isinstance(result, pd.Series), "detect_outliers_iqr 应该返回 Series"
        assert result.dtype == bool, "detect_outliers_iqr 应该返回布尔 Series"

    def test_standardize_data_returns_dataframe(self):
        """测试 standardize_data 返回 DataFrame"""
        func = getattr(solution, 'standardize_data', None)
        if func is None:
            pytest.skip("standardize_data 函数不存在")

        test_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50],
        })
        result = func(test_df)

        assert isinstance(result, pd.DataFrame), "standardize_data 应该返回 DataFrame"
        assert result.shape == test_df.shape, "输出形状应与输入相同"

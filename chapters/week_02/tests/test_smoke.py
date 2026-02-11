"""
Week 02 Smoke Test (脚手架基线测试)

这是 solution.py 的基础导入测试，验证模块结构是否正确。
不测试具体功能实现，只验证：
1. 模块可以导入
2. 核心函数存在
3. 函数签名正确

不要修改此文件。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

# 添加 starter_code 到路径
starter_code_path = Path(__file__).parent.parent / "starter_code"
import sys
sys.path.insert(0, str(starter_code_path))


# =============================================================================
# Test: Module Import
# =============================================================================

class TestModuleImport:
    """测试模块导入"""

    def test_solution_module_exists(self):
        """
        测试 solution 模块存在

        期望：应能导入 solution 模块
        """
        try:
            import solution
            assert solution is not None
        except ImportError as e:
            pytest.skip(f"solution.py not yet created: {e}")

    def test_solution_has_required_functions(self):
        """
        测试 solution 包含必需的函数

        期望：应包含以下函数：
        - calculate_central_tendency
        - calculate_dispersion
        - plot_histogram / plot_distribution_summary
        - generate_descriptive_summary
        """
        try:
            import solution

            # 检查核心函数是否存在（允许部分未实现）
            required_functions = [
                'calculate_central_tendency',
                'calculate_dispersion',
            ]

            optional_functions = [
                'plot_histogram',
                'plot_boxplot',
                'plot_density',
                'plot_distribution_summary',
                'check_y_axis_baseline',
                'detect_misleading_truncation',
                'generate_descriptive_summary',
                'create_distribution_plots',
                'append_to_report',
                'generate_one_page_report',
                'update_report_with_descriptive_stats',
                'create_week2_checkpoint',
                'validate_report_completeness',
            ]

            # 验证必需函数存在
            for func_name in required_functions:
                assert hasattr(solution, func_name), f"solution 应包含 {func_name} 函数"

            # 验证可选函数（如果实现）
            for func_name in optional_functions:
                if hasattr(solution, func_name):
                    func = getattr(solution, func_name)
                    assert callable(func), f"{func_name} 应是可调用函数"

        except ImportError:
            pytest.skip("solution.py not yet created")


# =============================================================================
# Test: Function Signatures
# =============================================================================

class TestFunctionSignatures:
    """测试函数签名"""

    def test_calculate_central_tendency_signature(self):
        """
        测试 calculate_central_tendency 函数签名

        期望：应接受 Series 并返回字典
        """
        try:
            import solution

            if hasattr(solution, 'calculate_central_tendency'):
                func = solution.calculate_central_tendency

                # 验证函数可调用
                assert callable(func)

                # 验证基本签名（不调用函数，只检查可调用性）
                # 具体参数测试在其他测试文件中
        except ImportError:
            pytest.skip("solution.py not yet created")

    def test_calculate_dispersion_signature(self):
        """
        测试 calculate_dispersion 函数签名

        期望：应接受 Series 并返回字典
        """
        try:
            import solution

            if hasattr(solution, 'calculate_dispersion'):
                func = solution.calculate_dispersion
                assert callable(func)

        except ImportError:
            pytest.skip("solution.py not yet created")

    def test_generate_descriptive_summary_signature(self):
        """
        测试 generate_descriptive_summary 函数签名

        期望：应接受 DataFrame 并返回字典
        """
        try:
            import solution

            if hasattr(solution, 'generate_descriptive_summary'):
                func = solution.generate_descriptive_summary
                assert callable(func)

        except ImportError:
            pytest.skip("solution.py not yet created")


# =============================================================================
# Test: Basic Functionality Check
# =============================================================================

class TestBasicFunctionality:
    """测试基本功能（最小调用）"""

    def test_calculate_central_tendency_callable(self):
        """
        测试 calculate_central_tendency 可被调用

        期望：至少不应抛出 ImportError 或 NameError
        """
        try:
            import solution

            if hasattr(solution, 'calculate_central_tendency'):
                func = solution.calculate_central_tendency

                # 尝试用最简单的输入调用
                test_series = pd.Series([1, 2, 3, 4, 5])

                try:
                    result = func(test_series)
                    # 如果函数已实现，应返回字典
                    if result is not None:
                        assert isinstance(result, dict), "应返回字典"
                except NotImplementedError:
                    # 如果函数只是占位符，这是可以接受的
                    pytest.skip("calculate_central_tendency not yet implemented")
                except Exception as e:
                    # 其他错误也要报告
                    pytest.fail(f"调用失败: {e}")

        except ImportError:
            pytest.skip("solution.py not yet created")

    def test_calculate_dispersion_callable(self):
        """
        测试 calculate_dispersion 可被调用

        期望：至少不应抛出 ImportError 或 NameError
        """
        try:
            import solution

            if hasattr(solution, 'calculate_dispersion'):
                func = solution.calculate_dispersion

                test_series = pd.Series([1, 2, 3, 4, 5])

                try:
                    result = func(test_series)
                    if result is not None:
                        assert isinstance(result, dict), "应返回字典"
                except NotImplementedError:
                    pytest.skip("calculate_dispersion not yet implemented")
                except Exception as e:
                    pytest.fail(f"调用失败: {e}")

        except ImportError:
            pytest.skip("solution.py not yet created")

    def test_generate_descriptive_summary_callable(self):
        """
        测试 generate_descriptive_summary 可被调用

        期望：至少不应抛出 ImportError 或 NameError
        """
        try:
            import solution

            if hasattr(solution, 'generate_descriptive_summary'):
                func = solution.generate_descriptive_summary

                test_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

                try:
                    result = func(test_df)
                    if result is not None:
                        assert isinstance(result, dict), "应返回字典"
                except NotImplementedError:
                    pytest.skip("generate_descriptive_summary not yet implemented")
                except Exception as e:
                    pytest.fail(f"调用失败: {e}")

        except ImportError:
            pytest.skip("solution.py not yet created")


# =============================================================================
# Test: Module Structure
# =============================================================================

class TestModuleStructure:
    """测试模块结构"""

    def test_solution_file_exists(self):
        """
        测试 solution.py 文件存在

        期望：文件应存在于 starter_code 目录
        """
        solution_path = Path(__file__).parent.parent / "starter_code" / "solution.py"
        assert solution_path.exists(), "starter_code/solution.py 应存在"

    def test_solution_file_readable(self):
        """
        测试 solution.py 可读

        期望：应能读取文件内容
        """
        solution_path = Path(__file__).parent.parent / "starter_code" / "solution.py"

        if solution_path.exists():
            content = solution_path.read_text(encoding='utf-8')
            assert len(content) > 0, "solution.py 不应为空"


# =============================================================================
# Test: Dependencies
# =============================================================================

class TestDependencies:
    """测试依赖项"""

    def test_pandas_available(self):
        """测试 pandas 可用"""
        import pandas as pd
        assert pd is not None

    def test_numpy_available(self):
        """测试 numpy 可用"""
        import numpy as np
        assert np is not None

    def test_matplotlib_available(self):
        """测试 matplotlib 可用（可选）"""
        try:
            import matplotlib.pyplot as plt
            assert plt is not None
        except ImportError:
            pytest.skip("matplotlib not installed (optional for week 02)")

    def test_seaborn_available(self):
        """测试 seaborn 可用（可选）"""
        try:
            import seaborn as sns
            assert sns is not None
        except ImportError:
            pytest.skip("seaborn not installed (optional for week 02)")

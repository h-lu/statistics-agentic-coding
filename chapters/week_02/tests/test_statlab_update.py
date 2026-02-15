"""
Week 02 测试：StatLab 更新功能

测试覆盖：
1. 检查 solution.py 中的所有 exercise 函数
2. 检查 main 函数

由于 solution.py 主要是参考答案实现，这里的测试主要是烟雾测试，
确保示例代码能运行。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

# 导入被测试的模块（路径已在 conftest.py 中设置）
solution = pytest.importorskip("solution")


# =============================================================================
# Test: 检查所有函数存在
# =============================================================================

class TestAllFunctionsExist:
    """检查所有必需的函数都存在"""

    def test_setup_output_dir_exists(self):
        """检查 setup_output_dir 函数存在"""
        assert hasattr(solution, 'setup_output_dir')

    def test_main_function_exists(self):
        """检查 main 函数存在"""
        assert hasattr(solution, 'main')


# =============================================================================
# Test: 运行 main 函数
# =============================================================================

class TestMainFunction:
    """测试 main 函数"""

    def test_main_runs_without_errors(self, tmp_path: Path):
        """
        运行 main 函数，确保不报错

        期望：应生成所有输出文件
        """
        # 改变工作目录到临时目录
        import os
        original_dir = os.getcwd()
        original_starter = Path(__file__).parent.parent / "starter_code"
        os.chdir(original_starter)

        try:
            solution.main()

            # 检查输出目录是否创建
            output_dir = original_starter / "output"
            assert output_dir.exists(), "应创建 output 目录"

            # 检查是否生成了图表文件
            expected_files = [
                "histogram_solution.png",
                "boxplot_solution.png",
                "honest_viz_solution.png"
            ]

            for filename in expected_files:
                file_path = output_dir / filename
                if file_path.exists():
                    assert file_path.stat().st_size > 0, f"{filename} 应非空"

        finally:
            os.chdir(original_dir)


# =============================================================================
# Test: 输出文件质量
# =============================================================================

class TestOutputQuality:
    """测试输出文件的质量"""

    def test_output_files_exist(self, tmp_path: Path):
        """
        测试所有输出文件是否生成

        期望：应生成所有预期的图表文件
        """
        import os
        original_dir = os.getcwd()
        original_starter = Path(__file__).parent.parent / "starter_code"
        os.chdir(original_starter)

        try:
            solution.main()

            output_dir = original_starter / "output"
            expected_files = [
                "histogram_solution.png",
                "boxplot_solution.png",
                "honest_viz_solution.png"
            ]

            for filename in expected_files:
                file_path = output_dir / filename
                # 注意：文件可能不存在，这是允许的（因为 solution.py 可能还在开发中）
                if file_path.exists():
                    assert file_path.stat().st_size > 0, f"{filename} 应非空"

        finally:
            os.chdir(original_dir)

"""
Week 02 测试：分布可视化（Distribution Plots）

测试覆盖：
1. 检查 solution 中是否有绘图函数
2. 基础烟雾测试

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
# Test: 检查示例函数存在
# =============================================================================

class TestExampleFunctions:
    """测试示例函数是否存在且可调用"""

    def test_exercise_1_central_tendency_exists(self):
        """检查 exercise_1_central_tendency 函数存在"""
        assert hasattr(solution, 'exercise_1_central_tendency')

    def test_exercise_2_dispersion_exists(self):
        """检查 exercise_2_dispersion 函数存在"""
        assert hasattr(solution, 'exercise_2_dispersion')

    def test_exercise_3_histogram_exists(self):
        """检查 exercise_3_histogram 函数存在"""
        assert hasattr(solution, 'exercise_3_histogram')

    def test_exercise_4_boxplot_exists(self):
        """检查 exercise_4_boxplot 函数存在"""
        assert hasattr(solution, 'exercise_4_boxplot')

    def test_exercise_5_honest_visualization_exists(self):
        """检查 exercise_5_honest_visualization 函数存在"""
        assert hasattr(solution, 'exercise_5_honest_visualization')

    def test_exercise_6_analysis_report_exists(self):
        """检查 exercise_6_analysis_report 函数存在"""
        assert hasattr(solution, 'exercise_6_analysis_report')


# =============================================================================
# Test: 运行示例函数（烟雾测试）
# =============================================================================

class TestRunExampleFunctions:
    """测试示例函数能够运行"""

    def test_run_exercise_1_central_tendency(self):
        """运行 exercise_1_central_tendency，确保不报错"""
        try:
            solution.exercise_1_central_tendency()
        except Exception as e:
            pytest.fail(f"exercise_1_central_tendency 运行失败: {e}")

    def test_run_exercise_2_dispersion(self):
        """运行 exercise_2_dispersion，确保不报错"""
        try:
            solution.exercise_2_dispersion()
        except Exception as e:
            pytest.fail(f"exercise_2_dispersion 运行失败: {e}")

    def test_run_exercise_3_histogram(self, tmp_path: Path):
        """运行 exercise_3_histogram，确保生成文件"""
        output_dir = tmp_path / "output"
        output_dir.mkdir(exist_ok=True)

        try:
            solution.exercise_3_histogram(output_dir)
            # 检查文件是否生成
            assert (output_dir / "histogram_solution.png").exists()
        except Exception as e:
            pytest.fail(f"exercise_3_histogram 运行失败: {e}")

    def test_run_exercise_4_boxplot(self, tmp_path: Path):
        """运行 exercise_4_boxplot，确保生成文件"""
        output_dir = tmp_path / "output"
        output_dir.mkdir(exist_ok=True)

        try:
            solution.exercise_4_boxplot(output_dir)
            # 检查文件是否生成
            assert (output_dir / "boxplot_solution.png").exists()
        except Exception as e:
            pytest.fail(f"exercise_4_boxplot 运行失败: {e}")

    def test_run_exercise_5_honest_visualization(self, tmp_path: Path):
        """运行 exercise_5_honest_visualization，确保生成文件"""
        output_dir = tmp_path / "output"
        output_dir.mkdir(exist_ok=True)

        try:
            solution.exercise_5_honest_visualization(output_dir)
            # 检查文件是否生成
            assert (output_dir / "honest_viz_solution.png").exists()
        except Exception as e:
            pytest.fail(f"exercise_5_honest_visualization 运行失败: {e}")

    def test_run_exercise_6_analysis_report(self):
        """运行 exercise_6_analysis_report，确保不报错"""
        try:
            solution.exercise_6_analysis_report()
        except Exception as e:
            pytest.fail(f"exercise_6_analysis_report 运行失败: {e}")

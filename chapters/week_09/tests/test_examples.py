"""
Week 09 示例代码测试

测试所有示例代码是否能正常运行。
"""
import subprocess
import sys
from pathlib import Path


def run_example(script_name: str) -> bool:
    """运行示例脚本并检查是否成功"""
    script_path = Path(__file__).parent.parent / 'examples' / script_name

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(script_path.parent)
        )

        return result.returncode == 0
    except Exception:
        return False


def test_01_correlation():
    """测试示例 01"""
    assert run_example("01_correlation_vs_regression.py")


def test_02_interpret_coefficients():
    """测试示例 02"""
    assert run_example("02_interpret_coefficients.py")


def test_03_regression_assumptions():
    """测试示例 03"""
    assert run_example("03_regression_assumptions.py")


def test_04_model_diagnostics():
    """测试示例 04"""
    assert run_example("04_model_diagnostics.py")


def test_05_multiple_regression():
    """测试示例 05"""
    assert run_example("05_multiple_regression.py")


def test_99_statlab():
    """测试 StatLab 示例"""
    assert run_example("99_statlab_regression.py")


def test_solution():
    """测试 solution.py"""
    solution_path = Path(__file__).parent.parent / 'starter_code' / 'solution.py'

    result = subprocess.run(
        [sys.executable, str(solution_path)],
        capture_output=True,
        text=True,
        timeout=30
    )

    assert result.returncode == 0

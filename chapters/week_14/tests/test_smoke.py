"""
Smoke tests for Week 14 examples

这些测试验证示例代码能够成功运行，不会抛出异常。
"""
import pytest
import subprocess
import sys
from pathlib import Path


# 示例文件路径
EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


@pytest.mark.parametrize("example_file", [
    "01_bayesian_ab_test.py",
    "02_prior_influence.py",
    "03_bayesian_regression.py",
    "04_mcmc_sampling.py",
    "05_hierarchical_model.py",
])
def test_example_runs(example_file):
    """
    测试示例文件能够成功运行

    注意：这些测试会实际运行 Python 脚本
    """
    example_path = EXAMPLES_DIR / example_file

    # 如果示例文件不存在，跳过测试
    if not example_path.exists():
        pytest.skip(f"示例文件 {example_file} 尚未创建")

    # 运行示例脚本
    result = subprocess.run(
        [sys.executable, str(example_path)],
        capture_output=True,
        text=True,
        timeout=120  # 贝叶斯采样可能需要较长时间
    )

    # 检查退出码
    assert result.returncode == 0, f"示例 {example_file} 运行失败:\n{result.stderr}"


def test_statlab_example_runs():
    """
    测试 StatLab 示例能够成功运行
    """
    example_path = EXAMPLES_DIR / "06_statlab_bayesian.py"

    if not example_path.exists():
        pytest.skip("StatLab 示例文件尚未创建")

    result = subprocess.run(
        [sys.executable, str(example_path)],
        capture_output=True,
        text=True,
        timeout=120
    )

    assert result.returncode == 0, f"StatLab 示例运行失败:\n{result.stderr}"
    # 检查是否生成了报告
    assert "贝叶斯" in result.stdout or "Bayesian" in result.stdout


def test_solution_runs():
    """
    测试 starter_code/solution.py 能够成功运行
    """
    solution_path = Path(__file__).parent.parent / "starter_code" / "solution.py"

    if not solution_path.exists():
        pytest.skip("solution.py 尚未创建")

    result = subprocess.run(
        [sys.executable, str(solution_path)],
        capture_output=True,
        text=True,
        timeout=120
    )

    assert result.returncode == 0, f"solution.py 运行失败:\n{result.stderr}"


@pytest.mark.parametrize("import_name", [
    "scipy",
    "numpy",
    "pandas",
])
def test_basic_imports(import_name):
    """
    测试基础库能够导入
    """
    result = subprocess.run(
        [sys.executable, "-c", f"import {import_name}"],
        capture_output=True,
        text=True,
        timeout=10
    )

    assert result.returncode == 0, f"无法导入 {import_name}: {result.stderr}"


@pytest.mark.pymc
def test_pymc_import():
    """
    测试 PyMC 能够导入（如果安装）
    """
    result = subprocess.run(
        [sys.executable, "-c", "import pymc, arviz"],
        capture_output=True,
        text=True,
        timeout=10
    )

    # 如果 PyMC 未安装，测试失败是预期的
    if result.returncode != 0:
        pytest.skip("PyMC 未安装，跳过相关测试")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

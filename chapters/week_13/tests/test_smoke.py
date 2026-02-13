"""
Smoke tests for Week 13 examples

这些测试验证示例代码能够成功运行，不会抛出异常。
"""
import pytest
import subprocess
import sys
from pathlib import Path


# 示例文件路径
EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


@pytest.mark.parametrize("example_file", [
    "01_causal_ladder.py",
    "02_causal_dag.py",
    "03_backdoor_criterion.py",
    "04_causal_estimation.py",
])
def test_example_runs(example_file):
    """
    测试示例文件能够成功运行

    注意：这些测试会实际运行 Python 脚本
    """
    example_path = EXAMPLES_DIR / example_file

    # 运行示例脚本
    result = subprocess.run(
        [sys.executable, str(example_path)],
        capture_output=True,
        text=True,
        timeout=60
    )

    # 检查退出码
    assert result.returncode == 0, f"示例 {example_file} 运行失败:\n{result.stderr}"


def test_statlab_example_runs():
    """
    测试 StatLab 示例能够成功运行
    """
    example_path = EXAMPLES_DIR / "05_statlab_causal.py"

    result = subprocess.run(
        [sys.executable, str(example_path)],
        capture_output=True,
        text=True,
        timeout=60
    )

    assert result.returncode == 0, f"StatLab 示例运行失败:\n{result.stderr}"
    # 检查是否生成了报告
    assert "StatLab 因果推断报告生成器" in result.stdout


def test_solution_runs():
    """
    测试 starter_code/solution.py 能够成功运行
    """
    solution_path = Path(__file__).parent.parent / "starter_code" / "solution.py"

    result = subprocess.run(
        [sys.executable, str(solution_path)],
        capture_output=True,
        text=True,
        timeout=60
    )

    assert result.returncode == 0, f"solution.py 运行失败:\n{result.stderr}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

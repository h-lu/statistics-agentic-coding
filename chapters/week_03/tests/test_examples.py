#!/usr/bin/env python3
"""
Week 03 示例代码测试

运行方式：python3 -m pytest chapters/week_03/tests/test_examples.py -v
"""
import subprocess
import sys
from pathlib import Path

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


def run_example(script_name: str) -> tuple[int, str, str]:
    """运行示例脚本并返回结果"""
    script_path = EXAMPLES_DIR / script_name
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        timeout=30
    )
    return result.returncode, result.stdout, result.stderr


def test_01_missing_overview():
    """测试缺失值概览示例"""
    returncode, stdout, stderr = run_example("01_missing_overview.py")
    assert returncode == 0, f"Script failed: {stderr}"
    assert "缺失值概览报告" in stdout
    assert "MCAR" in stdout or "MAR" in stdout


def test_02_imputation_strategies():
    """测试填充策略对比示例"""
    returncode, stdout, stderr = run_example("02_imputation_strategies.py")
    assert returncode == 0, f"Script failed: {stderr}"
    assert "填充策略对比" in stdout
    assert "均值填充" in stdout


def test_03_outlier_detection():
    """测试异常值检测示例"""
    returncode, stdout, stderr = run_example("03_outlier_detection.py")
    assert returncode == 0, f"Script failed: {stderr}"
    assert "异常值检测" in stdout
    assert "IQR" in stdout


def test_04_feature_transform():
    """测试特征变换示例"""
    returncode, stdout, stderr = run_example("04_feature_transform.py")
    assert returncode == 0, f"Script failed: {stderr}"
    assert "StandardScaler" in stdout
    assert "OneHotEncoder" in stdout


def test_05_cleaning_logger():
    """测试清洗日志示例"""
    returncode, stdout, stderr = run_example("05_cleaning_logger.py")
    assert returncode == 0, f"Script failed: {stderr}"
    assert "清洗日志" in stdout


def test_99_statlab():
    """测试 StatLab 超级线示例"""
    returncode, stdout, stderr = run_example("99_statlab.py")
    assert returncode == 0, f"Script failed: {stderr}"
    assert "StatLab Week 03" in stdout

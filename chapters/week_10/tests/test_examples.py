"""
Week 10 示例代码测试

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


def test_01_classification_vs_regression():
    """测试示例 01：分类 vs 回归"""
    assert run_example("01_classification_vs_regression.py")


def test_02_logistic_regression():
    """测试示例 02：逻辑回归"""
    assert run_example("02_logistic_regression.py")


def test_03_confusion_matrix_metrics():
    """测试示例 03：混淆矩阵与评估指标"""
    assert run_example("03_confusion_matrix_metrics.py")


def test_04_roc_auc():
    """测试示例 04：ROC 曲线与 AUC"""
    assert run_example("04_roc_auc.py")


def test_05_pipeline_data_leakage():
    """测试示例 05：Pipeline 与数据泄漏"""
    assert run_example("05_pipeline_data_leakage.py")


def test_99_statlab_classification():
    """测试 StatLab 示例"""
    assert run_example("99_statlab_classification.py")


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

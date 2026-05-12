from __future__ import annotations

import ast
from pathlib import Path


def _read_docstring(path: Path) -> str:
    return ast.get_docstring(ast.parse(path.read_text(encoding="utf-8"))) or ""


def test_week10_example_docstrings_have_required_sections():
    examples_dir = Path(__file__).resolve().parents[1] / "examples"
    required_sections = ["输入", "输出", "核心概念", "常见错误"]

    for filename in [
        "01_classification_vs_regression.py",
        "02_logistic_regression.py",
        "03_confusion_matrix_metrics.py",
        "04_roc_auc.py",
        "05_pipeline_data_leakage.py",
    ]:
        docstring = _read_docstring(examples_dir / filename)
        for section in required_sections:
            assert section in docstring, f"{filename} missing {section}"

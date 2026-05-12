from __future__ import annotations

from pathlib import Path


def test_week12_documents_mention_optional_dependency_path():
    chapter = Path(__file__).resolve().parents[1] / "CHAPTER.md"
    assignment = Path(__file__).resolve().parents[1] / "ASSIGNMENT.md"

    chapter_text = chapter.read_text(encoding="utf-8")
    assignment_text = assignment.read_text(encoding="utf-8")

    for text in [chapter_text, assignment_text]:
        assert "requirements-advanced.txt" in text
        assert "pip install -r requirements-advanced.txt" in text
        assert "默认路径" in text
        assert "高级路径" in text


def test_week12_images_are_referenced_in_chapter():
    chapter = Path(__file__).resolve().parents[1] / "CHAPTER.md"
    text = chapter.read_text(encoding="utf-8")

    for image_name in [
        "shap_summary_plot.png",
        "shap_waterfall_plot.png",
        "group_confusion_matrices.png",
        "fairness_metrics_comparison.png",
    ]:
        assert f"images/{image_name}" in text

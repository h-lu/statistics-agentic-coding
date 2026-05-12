from __future__ import annotations

from pathlib import Path


def test_week11_assignment_requires_starter_code_first():
    assignment = Path(__file__).resolve().parents[1] / "ASSIGNMENT.md"
    text = assignment.read_text(encoding="utf-8")

    assert "python3 chapters/week_11/starter_code/week_11.py" in text


def test_week11_chapter_keeps_core_images_referenced():
    chapter = Path(__file__).resolve().parents[1] / "CHAPTER.md"
    text = chapter.read_text(encoding="utf-8")

    for image_name in [
        "decision_tree_depth_3.png",
        "decision_tree_overfitting.png",
        "baseline_comparison_auc.png",
    ]:
        assert f"images/{image_name}" in text

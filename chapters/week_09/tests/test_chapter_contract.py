from __future__ import annotations

from pathlib import Path


def test_week09_chapter_mentions_core_formulas_and_images():
    chapter = Path(__file__).resolve().parents[1] / "CHAPTER.md"
    text = chapter.read_text(encoding="utf-8")

    assert "β0 + β1x + ε" in text
    assert "VIF_j = 1/(1-R_j²)" in text or "VIF_j = 1 / (1 - R_j²)" in text
    assert "Cook's D > 4/n" in text

    for image_name in [
        "01_scatter_with_regression.png",
        "03_residuals_vs_fitted.png",
        "03_qq_plot.png",
        "04_cooks_distance.png",
    ]:
        assert f"images/{image_name}" in text


def test_week09_assignment_requires_diagnostics_and_sensitivity():
    assignment = Path(__file__).resolve().parents[1] / "ASSIGNMENT.md"
    text = assignment.read_text(encoding="utf-8")

    for needle in [
        "残差 vs 拟合值图",
        "QQ 图",
        "Cook's 距离",
        "阈值（4/n）",
        "删除后模型",
        "处理建议",
    ]:
        assert needle in text

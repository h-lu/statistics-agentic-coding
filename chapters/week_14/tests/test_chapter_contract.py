from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_bayes_vs_frequentist_flowchart_present() -> None:
    chapter = (ROOT / "CHAPTER.md").read_text(encoding="utf-8")

    assert "flowchart TD" in chapter
    assert "优先频率学派" in chapter
    assert "优先贝叶斯学派" in chapter
    assert "Expected Loss" in chapter


def test_optional_pymc_fallback_is_explicit() -> None:
    chapter = (ROOT / "CHAPTER.md").read_text(encoding="utf-8")
    assignment = (ROOT / "ASSIGNMENT.md").read_text(encoding="utf-8")

    text = chapter + "\n" + assignment
    assert "PyMC（可选扩展" in text or "PyMC 和 ArviZ 是额外的贝叶斯建模库" in text
    assert "SciPy fallback" in text or "SciPy 解析解" in text


def test_unreferenced_images_are_documented() -> None:
    chapter = (ROOT / "CHAPTER.md").read_text(encoding="utf-8")
    readme = (ROOT / "images" / "README.md").read_text(encoding="utf-8")

    image_names = [p.name for p in (ROOT / "images").glob("*.png")]
    referenced = set(re.findall(r"images/([A-Za-z0-9_.-]+\.png)", chapter))
    documented = {name for name in image_names if name in readme}

    assert image_names
    assert referenced <= set(image_names)
    assert set(image_names) - referenced <= documented


def test_expected_loss_threshold_language_is_present() -> None:
    assignment = (ROOT / "ASSIGNMENT.md").read_text(encoding="utf-8")

    assert "假阳性成本" in assignment
    assert "假阴性成本" in assignment
    assert "阈值" in assignment

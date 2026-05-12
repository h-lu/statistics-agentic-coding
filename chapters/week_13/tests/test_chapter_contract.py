from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_did_parallel_trends_artifact_exists_and_is_referenced() -> None:
    chapter = (ROOT / "CHAPTER.md").read_text(encoding="utf-8")
    image = ROOT / "images" / "did_parallel_trends.png"

    assert image.is_file()
    assert "did_parallel_trends.png" in chapter


def test_observational_methods_are_marked_as_optional() -> None:
    chapter = (ROOT / "CHAPTER.md").read_text(encoding="utf-8")

    assert "观察研究中的因果推断（选读拓展）" in chapter
    assert "拓展阅读" in chapter


def test_code_block_budget_has_exemption_comment() -> None:
    chapter = (ROOT / "CHAPTER.md").read_text(encoding="utf-8")
    code_blocks = chapter.count("```") // 2

    if code_blocks > 10:
        assert "<!-- code_block_exemption:" in chapter


def test_parallel_trends_script_uses_file_font_loading() -> None:
    script = (ROOT / "examples" / "06_did_parallel_trends.py").read_text(encoding="utf-8")

    assert "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc" in script
    assert "/usr/share/fonts/google-droid-fonts/DroidSansFallback.ttf" in script
    assert "raise RuntimeError" in script

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_cluster_stability_artifact_exists_and_script_uses_file_fonts() -> None:
    image = ROOT / "images" / "04_cluster_stability.png"
    script = (ROOT / "examples" / "04_cluster_stability.py").read_text(encoding="utf-8")

    assert image.is_file()
    assert "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc" in script
    assert "/usr/share/fonts/google-droid-fonts/DroidSansFallback.ttf" in script
    assert "raise RuntimeError" in script


def test_dimension_curse_table_mentions_2_10_50_100() -> None:
    chapter = (ROOT / "CHAPTER.md").read_text(encoding="utf-8")

    for token in ["2 维", "10 维", "50 维", "100 维", "变异系数"]:
        assert token in chapter


def test_statlab_template_fields_are_unified() -> None:
    assignment = (ROOT / "ASSIGNMENT.md").read_text(encoding="utf-8")

    expected_fields = [
        "输入数据",
        "特征选择理由",
        "标准化方式",
        "主成分解释",
        "聚类数选择",
        "稳定性检查",
        "非技术解释",
    ]
    for field in expected_fields:
        assert field in assignment


def test_ai_sidebar_mentions_k_recommendation_and_stability_checks() -> None:
    chapter = (ROOT / "CHAPTER.md").read_text(encoding="utf-8")

    assert "先让 AI 猜一个 K" in chapter
    assert "肘部法则" in chapter
    assert "轮廓系数" in chapter
    assert "稳定性检查" in chapter


def test_unreferenced_image_is_documented() -> None:
    chapter = (ROOT / "CHAPTER.md").read_text(encoding="utf-8")
    readme = (ROOT / "images" / "README.md").read_text(encoding="utf-8")

    image_names = [p.name for p in (ROOT / "images").glob("*.png")]
    referenced = {name for name in image_names if name in chapter}
    documented = {name for name in image_names if name in readme}

    assert "04_cluster_radar_chart.png" in documented
    assert set(image_names) - referenced <= documented

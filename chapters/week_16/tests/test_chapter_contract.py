from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_pipeline_diagram_is_present() -> None:
    chapter = (ROOT / "CHAPTER.md").read_text(encoding="utf-8")

    assert "flowchart LR" in chapter
    assert "raw data" in chapter
    assert "cleaning" in chapter
    assert "analysis" in chapter
    assert "ai_review_log" in chapter


def test_good_vs_bad_display_comparison_exists() -> None:
    chapter = (ROOT / "CHAPTER.md").read_text(encoding="utf-8")

    assert "好展示 vs 差展示" in chapter
    assert "127 页 PPT" in chapter
    assert "AI 先给建议，人类负责审查与定稿" in chapter


def test_ai_usage_log_template_has_required_columns() -> None:
    template = (ROOT / "starter_code" / "ai_usage_log_template.md").read_text(encoding="utf-8")

    required_columns = [
        "日期",
        "AI 工具/模型",
        "使用场景",
        "文件/位置",
        "采纳的建议",
        "拒绝的建议",
        "人工修改",
        "复核结论",
    ]
    for column in required_columns:
        assert column in template


def test_final_audit_checklist_mentions_ai_usage_records() -> None:
    chapter = (ROOT / "CHAPTER.md").read_text(encoding="utf-8")
    assignment = (ROOT / "ASSIGNMENT.md").read_text(encoding="utf-8")
    text = chapter + "\n" + assignment

    for keyword in ["数据来源", "清洗决策", "诚实图表", "不确定性", "模型诊断", "因果表述", "AI 使用记录"]:
        assert keyword in text

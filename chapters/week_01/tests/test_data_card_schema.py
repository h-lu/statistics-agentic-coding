"""Schema tests for Week 01 data cards."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

# Add starter_code to path for imports
starter_code_path = Path(__file__).parent.parent / "starter_code"
if str(starter_code_path) not in sys.path:
    sys.path.insert(0, str(starter_code_path))

from solution import generate_data_card


REQUIRED_METADATA_FIELDS = [
    "dataset_name",
    "source",
    "license_or_terms",
    "collection_process",
    "unit_of_analysis",
    "row_count",
    "column_count",
    "missing_summary",
    "known_limitations",
    "not_suitable_for",
    "human_review_notes",
]

REQUIRED_LABELS = [
    "数据集名称",
    "来源",
    "许可证/使用条款",
    "收集过程",
    "分析单位",
    "行数",
    "列数",
    "缺失概览",
    "已知限制",
    "不适合什么场景",
    "人工复核说明",
]


def test_data_card_includes_canonical_schema(sample_dataframe):
    metadata = {
        "dataset_name": "Palmer Penguins",
        "source": "seaborn 内置数据集",
        "license_or_terms": "示例数据，可用于教学",
        "collection_process": "Palmer Station, Antarctica LTER 的企鹅形态测量记录",
        "unit_of_analysis": "单只企鹅",
        "known_limitations": "样本来自特定地点和时间",
        "not_suitable_for": "不能直接推广到所有企鹅",
        "human_review_notes": "字段解释需要人工复核",
    }

    data_card = generate_data_card(sample_dataframe, metadata)

    for label in REQUIRED_LABELS:
        assert label in data_card

    assert "row_count" not in data_card
    assert "column_count" not in data_card


def test_data_card_distinguishes_numeric_and_categorical(sample_dataframe):
    metadata = {
        "dataset_name": "Palmer Penguins",
        "source": "seaborn 内置数据集",
        "unit_of_analysis": "单只企鹅",
    }

    data_card = generate_data_card(sample_dataframe, metadata)

    assert "统计学类型" in data_card
    assert "bill_length_mm" in data_card and "数值型-连续" in data_card
    assert "species" in data_card and "分类型-名义" in data_card


def test_data_card_does_not_fabricate_unknown_source_or_license(sample_dataframe):
    metadata = {
        "dataset_name": "Unknown Dataset",
        "unit_of_analysis": "单行记录",
    }

    data_card = generate_data_card(sample_dataframe, metadata)

    # 未提供来源/许可时，必须保留待补充，而不是编造
    assert data_card.count("（待补充）") >= 2
    assert "许可证/使用条款" in data_card
    assert "收集过程" in data_card

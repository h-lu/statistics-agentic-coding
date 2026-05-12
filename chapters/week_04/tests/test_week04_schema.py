"""
Week 04 schema tests.

These tests make the chapter requirements concrete:
1. The EDA dataset must expose the documented fields.
2. The time-series workflow must support year_week grouping.
3. Hypothesis entries must keep the required core fields.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

solution = pytest.importorskip("solution")

aggregate_by_year_week = getattr(solution, "aggregate_by_year_week", None)
validate_hypothesis = getattr(solution, "validate_hypothesis", None)


def load_week04_data() -> pd.DataFrame:
    """Load the chapter dataset from the starter code directory."""
    data_path = Path(__file__).parent.parent / "starter_code" / "week_04_data.csv"
    return pd.read_csv(data_path)


def test_week04_data_fields_present():
    """The chapter dataset should expose the documented business fields."""
    df = load_week04_data()
    required_columns = {
        "age",
        "time_on_site",
        "purchase_amount",
        "visit_count",
        "is_returning",
        "source",
        "date",
    }

    assert required_columns.issubset(df.columns), (
        f"Missing expected fields: {required_columns - set(df.columns)}"
    )


def test_year_week_grouping():
    """The time-series workflow should support year_week aggregation."""
    if aggregate_by_year_week is None:
        pytest.skip("aggregate_by_year_week 函数不存在")

    df = load_week04_data()
    grouped = aggregate_by_year_week(df, date_col="date", value_col="purchase_amount", agg_func="mean")

    assert isinstance(grouped, pd.DataFrame), "year_week 聚合应返回 DataFrame"
    assert "year_week" in grouped.columns, "结果应包含 year_week 列"
    assert "purchase_amount" in grouped.columns, "结果应包含聚合值列"
    assert grouped["year_week"].str.match(r"\d{4}-W\d{2}").all(), \
        "year_week 应使用 YYYY-Www 格式"
    assert len(grouped) > 0, "year_week 分组不应为空"


def test_hypothesis_required_fields():
    """Hypothesis records should keep the required core fields."""
    if validate_hypothesis is None:
        pytest.skip("validate_hypothesis 函数不存在")

    valid_hypothesis = {
        "observation": "搜索渠道的平均购买金额高于社交渠道",
        "explanation": "搜索用户意图更明确",
        "test_method": "双样本 t 检验",
        "priority": "high",
    }
    missing_test_method = {
        "observation": "搜索渠道的平均购买金额高于社交渠道",
        "explanation": "搜索用户意图更明确",
        "priority": "high",
    }

    assert validate_hypothesis(valid_hypothesis) is True
    assert validate_hypothesis(missing_test_method) is False

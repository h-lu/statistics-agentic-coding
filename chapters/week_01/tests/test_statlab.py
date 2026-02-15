"""Tests for StatLab report generation functionality."""

from __future__ import annotations

import pandas as pd
import pytest
from pathlib import Path
import re


def test_generate_report_basic(sample_dataframe, temp_output_dir):
    """Test basic report generation."""
    output_path = temp_output_dir / "report.md"

    # TODO: Implement after solution.py has generate_report function
    # result = generate_report(sample_dataframe, str(output_path))
    #
    # assert result == output_path
    # assert output_path.exists()
    pass


def test_report_contains_title(sample_dataframe, temp_output_dir):
    """Test that report contains proper title."""
    output_path = temp_output_dir / "report.md"

    # TODO: Implement after solution.py has generate_report function
    # generate_report(sample_dataframe, str(output_path))
    #
    # content = output_path.read_text(encoding='utf-8')
    # assert "# StatLab 分析报告" in content or "# StatLab Report" in content
    pass


def test_report_contains_data_card(sample_dataframe, temp_output_dir):
    """Test that report includes data card section."""
    output_path = temp_output_dir / "report.md"

    # TODO: Implement after solution.py has generate_report function
    # generate_report(sample_dataframe, str(output_path))
    #
    # content = output_path.read_text(encoding='utf-8')
    # assert "数据卡" in content
    # assert "数据来源" in content
    # assert "字段字典" in content
    pass


def test_report_contains_generation_timestamp(sample_dataframe, temp_output_dir):
    """Test that report includes generation timestamp."""
    output_path = temp_output_dir / "report.md"

    # TODO: Implement after solution.py has generate_report function
    # generate_report(sample_dataframe, str(output_path))
    #
    # content = output_path.read_text(encoding='utf-8')
    # # Check for date pattern (YYYY-MM-DD)
    # date_pattern = r"\d{4}-\d{2}-\d{2}"
    # assert re.search(date_pattern, content) is not None
    pass


def test_report_creates_parent_directory(sample_dataframe, tmp_path):
    """Test that report generation creates parent directories if needed."""
    nested_path = tmp_path / "nested" / "dir" / "report.md"

    # TODO: Implement after solution.py has generate_report function
    # generate_report(sample_dataframe, str(nested_path))
    #
    # assert nested_path.exists()
    # assert nested_path.parent.exists()
    pass


def test_report_overwrites_existing(sample_dataframe, temp_output_dir):
    """Test that generating report overwrites existing file."""
    output_path = temp_output_dir / "report.md"

    # Create existing file
    output_path.write_text("Old report content", encoding='utf-8')

    # TODO: Implement after solution.py has generate_report function
    # generate_report(sample_dataframe, str(output_path))
    #
    # content = output_path.read_text(encoding='utf-8')
    # assert "Old report content" not in content
    # assert "StatLab" in content
    pass


def test_report_with_penguins_dataset(penguins_dataset, temp_output_dir):
    """Test report generation with Palmer Penguins dataset."""
    output_path = temp_output_dir / "penguins_report.md"

    # TODO: Implement after solution.py has generate_report function
    # generate_report(penguins_dataset, str(output_path))
    #
    # content = output_path.read_text(encoding='utf-8')
    # assert "Palmer Penguins" in content
    # assert "species" in content
    # assert "bill_length_mm" in content
    pass


def test_report_contains_next_steps(sample_dataframe, temp_output_dir):
    """Test that report includes next steps section."""
    output_path = temp_output_dir / "report.md"

    # TODO: Implement after solution.py has generate_report function
    # generate_report(sample_dataframe, str(output_path))
    #
    # content = output_path.read_text(encoding='utf-8')
    # assert "下一步" in content or "Next Steps" in content
    pass


def test_report_markdown_format(sample_dataframe, temp_output_dir):
    """Test that report is valid Markdown."""
    output_path = temp_output_dir / "report.md"

    # TODO: Implement after solution.py has generate_report function
    # generate_report(sample_dataframe, str(output_path))
    #
    # content = output_path.read_text(encoding='utf-8')
    #
    # # Check for Markdown elements
    # assert content.strip().startswith("#")
    # assert "##" in content
    #
    # # Check for horizontal rules
    # assert "---" in content
    pass


def test_report_default_output_path(sample_dataframe, tmp_path):
    """Test report generation with default output path."""
    # Change to temp directory
    import os
    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        # TODO: Implement after solution.py has generate_report function with default path
        # result = generate_report(sample_dataframe)
        # assert Path(result).name == "report.md"
        # assert Path(result).exists()
        pass
    finally:
        os.chdir(original_cwd)


def test_report_invalid_input_not_dataframe(temp_output_dir):
    """Test that non-DataFrame input raises appropriate error."""
    output_path = temp_output_dir / "report.md"

    # TODO: Implement after solution.py has generate_report function
    # with pytest.raises(TypeError, match="DataFrame"):
    #     generate_report([1, 2, 3], str(output_path))
    #
    # with pytest.raises(TypeError, match="DataFrame"):
    #     generate_report("not a dataframe", str(output_path))
    pass


def test_report_empty_dataframe(empty_dataframe, temp_output_dir):
    """Test report generation with empty DataFrame."""
    output_path = temp_output_dir / "report.md"

    # TODO: Implement after solution.py has generate_report function
    # generate_report(empty_dataframe, str(output_path))
    #
    # assert output_path.exists()
    # content = output_path.read_text(encoding='utf-8')
    # # Should still generate a report, even if empty
    # assert "StatLab" in content
    pass


def test_report_encoding_utf8(sample_dataframe, temp_output_dir):
    """Test that report file is UTF-8 encoded."""
    output_path = temp_output_dir / "report.md"

    # TODO: Implement after solution.py has generate_report function
    # generate_report(sample_dataframe, str(output_path))
    #
    # # Read with UTF-8 encoding
    # with open(output_path, 'r', encoding='utf-8') as f:
    #     content = f.read()
    #
    # # Check Chinese characters are present
    # assert "数据卡" in content
    # assert "分析报告" in content
    pass


def test_report_return_value(sample_dataframe, temp_output_dir):
    """Test that generate_report returns the output path."""
    output_path = temp_output_dir / "report.md"

    # TODO: Implement after solution.py has generate_report function
    # result = generate_report(sample_dataframe, str(output_path))
    #
    # assert isinstance(result, Path)
    # assert result == output_path
    # assert result.exists()
    pass


def test_report_reproducibility(sample_dataframe, temp_output_dir):
    """Test that running report generation twice produces identical output."""
    output_path1 = temp_output_dir / "report1.md"
    output_path2 = temp_output_dir / "report2.md"

    # TODO: Implement after solution.py has generate_report function
    # generate_report(sample_dataframe, str(output_path1))
    # generate_report(sample_dataframe, str(output_path2))
    #
    # content1 = output_path1.read_text(encoding='utf-8')
    # content2 = output_path2.read_text(encoding='utf-8')
    #
    # # Timestamps might differ, so we check the main content
    # # Strip timestamp lines before comparing
    # content1_no_timestamp = re.sub(r'\d{4}-\d{2}-\d{2}', '', content1)
    # content2_no_timestamp = re.sub(r'\d{4}-\d{2}-\d{2}', '', content2)
    #
    # assert content1_no_timestamp == content2_no_timestamp
    pass


def test_report_structure_sections(sample_dataframe, temp_output_dir):
    """Test that report has expected structure with all sections."""
    output_path = temp_output_dir / "report.md"

    # TODO: Implement after solution.py has generate_report function
    # generate_report(sample_dataframe, str(output_path))
    #
    # content = output_path.read_text(encoding='utf-8')
    #
    # # Check for main sections
    # sections = [
    #     "# StatLab",  # Title
    #     "数据卡",
    #     "数据来源",
    #     "字段字典",
    #     "规模概览",
    #     "缺失概览",
    #     "下一步"
    # ]
    #
    # for section in sections:
    #     assert section in content, f"Section '{section}' not found in report"
    pass

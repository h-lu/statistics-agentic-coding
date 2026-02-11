"""
Week 02 测试：一页分布报告（One Page Distribution Report）

测试覆盖：
1. generate_descriptive_summary() - 生成描述统计摘要
2. create_distribution_plots() - 创建分布图（批量）
3. append_to_report() - 追加到报告文件
4. generate_one_page_report() - 生成完整的一页报告

测试用例类型：
- 正例：正常数据下的报告生成
- 边界：空数据、单列数据
- 反例：错误的数据类型
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

# 导入被测试的模块
import sys
starter_code_path = Path(__file__).parent.parent / "starter_code"
sys.path.insert(0, str(starter_code_path))

# 尝试导入 solution 模块
try:
    from solution import (
        generate_descriptive_summary,
        create_distribution_plots,
        append_to_report,
        generate_one_page_report,
    )
except ImportError:
    pytest.skip("solution.py not yet created", allow_module_level=True)


# =============================================================================
# Test: generate_descriptive_summary()
# =============================================================================

class TestGenerateDescriptiveSummary:
    """测试描述统计摘要生成函数"""

    # --------------------
    # 正例（happy path）
    # --------------------

    def test_generate_basic_summary(self, sample_dataframe: pd.DataFrame):
        """
        测试基本摘要生成

        期望：应返回包含均值、中位数、标准差、IQR 的字典
        """
        result = generate_descriptive_summary(sample_dataframe)

        assert isinstance(result, dict)

        # 应该包含所有数值列的摘要
        numeric_cols = sample_dataframe.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            assert col in result, f"应包含列 '{col}' 的摘要"

            col_summary = result[col]
            assert isinstance(col_summary, dict)

            # 验证关键字段存在
            required_fields = ['mean', 'median', 'std', 'iqr']
            for field in required_fields:
                assert field in col_summary, f"列 '{col}' 应包含 '{field}'"

            # 验证数据类型正确
            assert isinstance(col_summary['mean'], (int, float))
            assert isinstance(col_summary['median'], (int, float))
            assert isinstance(col_summary['std'], (int, float))
            assert isinstance(col_summary['iqr'], (int, float))

    def test_generate_summary_correctness(self, sample_dataframe: pd.DataFrame):
        """
        测试摘要值的正确性

        期望：计算值应与 pandas 直接计算一致
        """
        result = generate_descriptive_summary(sample_dataframe)

        # 选择第一个数值列验证
        numeric_col = sample_dataframe.select_dtypes(include=['number']).columns[0]
        series = sample_dataframe[numeric_col]
        summary = result[numeric_col]

        # 验证均值
        expected_mean = series.mean()
        assert summary['mean'] == pytest.approx(expected_mean, rel=1e-5)

        # 验证中位数
        expected_median = series.median()
        assert summary['median'] == pytest.approx(expected_median, rel=1e-5)

        # 验证标准差
        expected_std = series.std()
        assert summary['std'] == pytest.approx(expected_std, rel=1e-5)

        # 验证 IQR
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        expected_iqr = q3 - q1
        assert summary['iqr'] == pytest.approx(expected_iqr, rel=1e-5)

    def test_generate_summary_with_quantiles(self, sample_dataframe: pd.DataFrame):
        """
        测试包含四分位数的摘要

        期望：如果实现，应包含 Q1, Q3
        """
        result = generate_descriptive_summary(sample_dataframe)

        numeric_col = sample_dataframe.select_dtypes(include=['number']).columns[0]
        summary = result[numeric_col]

        # 可选字段：Q1, Q3
        if 'q1' in summary and 'q3' in summary:
            # 验证 IQR = Q3 - Q1
            assert summary['iqr'] == pytest.approx(summary['q3'] - summary['q1'], rel=1e-5)

    # --------------------
    # 边界情况
    # --------------------

    def test_generate_empty_dataframe(self):
        """
        测试空 DataFrame

        期望：应返回空字典或提示
        """
        empty_df = pd.DataFrame()
        result = generate_descriptive_summary(empty_df)

        assert isinstance(result, dict)
        assert len(result) == 0

    def test_generate_no_numeric_columns(self):
        """
        测试无数值列的 DataFrame

        期望：应返回空字典或警告
        """
        categorical_df = pd.DataFrame({
            'city': ['北京', '上海', '深圳'],
            'category': ['A', 'B', 'C'],
        })

        result = generate_descriptive_summary(categorical_df)

        assert isinstance(result, dict)
        assert len(result) == 0

    def test_generate_with_missing_values(self, dataframe_with_missing: pd.DataFrame):
        """
        测试包含缺失值的数据

        期望：应自动忽略缺失值
        """
        result = generate_descriptive_summary(dataframe_with_missing)

        # 应该能计算，不抛出异常
        assert isinstance(result, dict)

        # 验证结果不包含 NaN
        for col_summary in result.values():
            assert not np.isnan(col_summary['mean'])
            assert not np.isnan(col_summary['median'])

    # --------------------
    # 反例（错误输入）
    # --------------------

    def test_generate_non_dataframe_input(self):
        """
        测试非 DataFrame 输入

        期望：应抛出异常
        """
        with pytest.raises((TypeError, ValueError)):
            generate_descriptive_summary([1, 2, 3])


# =============================================================================
# Test: create_distribution_plots()
# =============================================================================

class TestCreateDistributionPlots:
    """测试批量创建分布图函数"""

    # --------------------
    # 正例（happy path）
    # --------------------

    def test_create_plots_for_dataframe(self, sample_dataframe: pd.DataFrame, temp_output_dir: Path):
        """
        测试为 DataFrame 创建所有分布图

        期望：应为每个数值列生成直方图和箱线图
        """
        result = create_distribution_plots(sample_dataframe, output_dir=str(temp_output_dir))

        assert isinstance(result, (list, dict))

        if isinstance(result, list):
            # 应返回文件路径列表
            assert len(result) > 0

            # 验证所有文件存在
            for path in result:
                assert Path(path).exists()
                assert Path(path).stat().st_size > 0

        elif isinstance(result, dict):
            assert 'paths' in result or 'histogram' in result or 'boxplot' in result

    def test_create_plots_correct_count(self, sample_dataframe: pd.DataFrame, temp_output_dir: Path):
        """
        测试生成的图表数量

        期望：每个数值列应生成至少 2 张图（直方图 + 箱线图）
        """
        result = create_distribution_plots(sample_dataframe, output_dir=str(temp_output_dir))

        numeric_cols = sample_dataframe.select_dtypes(include=['number']).columns
        expected_min_plots = len(numeric_cols) * 2  # 每列至少 2 张图

        if isinstance(result, list):
            assert len(result) >= expected_min_plots

    def test_create_plots_custom_types(self, sample_dataframe: pd.DataFrame, temp_output_dir: Path):
        """
        测试自定义图表类型

        期望：应支持指定生成哪些类型的图
        """
        result = create_distribution_plots(
            sample_dataframe,
            output_dir=str(temp_output_dir),
            plot_types=['histogram']  # 只要直方图
        )

        assert result is not None

    # --------------------
    # 边界情况
    # --------------------

    def test_create_empty_dataframe(self, temp_output_dir: Path):
        """
        测试空 DataFrame

        期望：应返回空列表
        """
        empty_df = pd.DataFrame()
        result = create_distribution_plots(empty_df, output_dir=str(temp_output_dir))

        if isinstance(result, list):
            assert len(result) == 0
        elif isinstance(result, dict):
            assert len(result.get('paths', [])) == 0

    def test_create_no_numeric_columns(self, temp_output_dir: Path):
        """
        测试无数值列的 DataFrame

        期望：应返回空列表
        """
        categorical_df = pd.DataFrame({
            'city': ['北京', '上海', '深圳'],
            'category': ['A', 'B', 'C'],
        })

        result = create_distribution_plots(categorical_df, output_dir=str(temp_output_dir))

        if isinstance(result, list):
            assert len(result) == 0

    def test_create_plots_creates_directory(self, sample_dataframe: pd.DataFrame, tmp_path: Path):
        """
        测试自动创建输出目录

        期望：如果目录不存在，应自动创建
        """
        new_dir = tmp_path / "new_figures"
        assert not new_dir.exists()

        result = create_distribution_plots(sample_dataframe, output_dir=str(new_dir))

        # 目录应被创建
        assert new_dir.exists()
        assert new_dir.is_dir()

    # --------------------
    # 反例（错误输入）
    # --------------------

    def test_create_plots_non_dataframe_input(self, temp_output_dir: Path):
        """
        测试非 DataFrame 输入

        期望：应抛出异常
        """
        with pytest.raises((TypeError, ValueError)):
            create_distribution_plots([1, 2, 3], output_dir=str(temp_output_dir))


# =============================================================================
# Test: append_to_report()
# =============================================================================

class TestAppendToReport:
    """测试追加报告内容函数"""

    # --------------------
    # 正例（happy path）
    # --------------------

    def test_append_to_new_report(self, sample_dataframe: pd.DataFrame, sample_report_path: Path):
        """
        测试追加到新报告文件

        期望：应创建报告文件并写入内容
        """
        append_to_report(sample_dataframe, report_path=str(sample_report_path))

        # 验证文件被创建
        assert sample_report_path.exists()

        # 读取内容验证
        content = sample_report_path.read_text(encoding='utf-8')

        # 应包含关键标题
        assert '描述统计' in content or 'Descriptive' in content
        assert '核心指标' in content or 'Summary' in content

    def test_append_content_structure(self, sample_dataframe: pd.DataFrame, sample_report_path: Path):
        """
        测试报告内容结构

        期望：应包含 Markdown 表格和图表引用
        """
        append_to_report(sample_dataframe, report_path=str(sample_report_path))

        content = sample_report_path.read_text(encoding='utf-8')

        # 应包含 Markdown 表格
        assert '|' in content  # Markdown 表格语法

        # 应包含数值列名
        for col in sample_dataframe.select_dtypes(include=['number']).columns:
            assert col in content

    def test_append_skewness_interpretation(self, sample_dataframe: pd.DataFrame, sample_report_path: Path):
        """
        测试偏态解释生成

        期望：应根据均值和中位数的关系生成解释
        """
        append_to_report(sample_dataframe, report_path=str(sample_report_path))

        content = sample_report_path.read_text(encoding='utf-8')

        # 应包含解释性文字
        has_interpretation = any(kw in content for kw in ['偏', 'skew', '均值', '中位数'])
        assert has_interpretation

    # --------------------
    # 边界情况
    # --------------------

    def test_append_to_existing_report(self, sample_dataframe: pd.DataFrame, sample_report_path: Path):
        """
        测试追加到已存在的报告

        期望：应在文件末尾追加，不覆盖原有内容
        """
        # 先写入一些内容
        sample_report_path.write_text("# 原有内容\n\n", encoding='utf-8')

        # 追加新内容
        append_to_report(sample_dataframe, report_path=str(sample_report_path))

        content = sample_report_path.read_text(encoding='utf-8')

        # 原有内容应保留
        assert '原有内容' in content
        # 新内容也应存在
        assert '描述统计' in content or 'Descriptive' in content

    def test_append_empty_dataframe(self, sample_report_path: Path):
        """
        测试追加空 DataFrame

        期望：应写入空表格或提示
        """
        empty_df = pd.DataFrame()

        append_to_report(empty_df, report_path=str(sample_report_path))

        content = sample_report_path.read_text(encoding='utf-8')

        # 至少应该创建了文件
        assert sample_report_path.exists()

    # --------------------
    # 反例（错误输入）
    # --------------------

    def test_append_non_dataframe_input(self, sample_report_path: Path):
        """
        测试非 DataFrame 输入

        期望：应抛出异常
        """
        with pytest.raises((TypeError, ValueError)):
            append_to_report([1, 2, 3], report_path=str(sample_report_path))


# =============================================================================
# Test: generate_one_page_report()
# =============================================================================

class TestGenerateOnePageReport:
    """测试完整一页报告生成函数"""

    # --------------------
    # 正例（happy path）
    # --------------------

    def test_generate_complete_report(self, sample_dataframe: pd.DataFrame, temp_output_dir: Path):
        """
        测试生成完整的一页报告

        期望：应生成摘要、图表和 Markdown 报告
        """
        report_path = temp_output_dir / "report.md"
        figures_dir = temp_output_dir / "figures"

        result = generate_one_page_report(
            sample_dataframe,
            report_path=str(report_path),
            figures_dir=str(figures_dir)
        )

        assert isinstance(result, dict) or isinstance(result, str)

        # 验证报告文件被创建
        assert report_path.exists()

        # 验证图表目录被创建
        assert figures_dir.exists()

        # 验证报告内容
        content = report_path.read_text(encoding='utf-8')

        # 应包含关键部分
        assert any(kw in content for kw in ['描述统计', 'Descriptive', '摘要', 'Summary'])

    def test_generate_report_with_metadata(self, sample_dataframe: pd.DataFrame, temp_output_dir: Path):
        """
        测试包含元数据的报告生成

        期望：应在报告中包含标题、数据源等信息
        """
        report_path = temp_output_dir / "report_with_meta.md"

        result = generate_one_page_report(
            sample_dataframe,
            report_path=str(report_path),
            title="用户数据分析报告",
            data_source="用户数据库",
            author="分析师",
        )

        content = report_path.read_text(encoding='utf-8')

        # 应包含元数据
        assert '用户数据分析报告' in content or '用户数据库' in content

    def test_generate_report_structure(self, sample_dataframe: pd.DataFrame, temp_output_dir: Path):
        """
        测试报告结构

        期望：应包含标题、摘要表格、图表、解释
        """
        report_path = temp_output_dir / "report_structure.md"

        generate_one_page_report(
            sample_dataframe,
            report_path=str(report_path),
        )

        content = report_path.read_text(encoding='utf-8')

        # 应包含 Markdown 标题
        assert '#' in content

        # 应包含表格（Markdown 表格以 | 开头）
        assert '|' in content

        # 应包含图片引用（Markdown 图片语法）
        has_image_ref = '![(' in content or '![' in content or '.png' in content or '.jpg' in content

    # --------------------
    # 边界情况
    # --------------------

    def test_generate_report_empty_dataframe(self, temp_output_dir: Path):
        """
        测试空 DataFrame 的报告生成

        期望：应生成空报告或提示
        """
        report_path = temp_output_dir / "empty_report.md"
        empty_df = pd.DataFrame()

        result = generate_one_page_report(
            empty_df,
            report_path=str(report_path),
        )

        # 应该能生成，不崩溃
        assert report_path.exists()

    # --------------------
    # 反例（错误输入）
    # --------------------

    def test_generate_report_non_dataframe_input(self, temp_output_dir: Path):
        """
        测试非 DataFrame 输入

        期望：应抛出异常
        """
        report_path = temp_output_dir / "error_report.md"

        with pytest.raises((TypeError, ValueError)):
            generate_one_page_report(
                [1, 2, 3],
                report_path=str(report_path),
            )


# =============================================================================
# Test: 报告内容质量
# =============================================================================

class TestReportQuality:
    """测试报告内容质量"""

    def test_report_has_clear_headers(self, sample_dataframe: pd.DataFrame, temp_output_dir: Path):
        """
        测试报告标题清晰度

        期望：应包含清晰的章节标题
        """
        report_path = temp_output_dir / "quality_report.md"

        generate_one_page_report(
            sample_dataframe,
            report_path=str(report_path),
        )

        content = report_path.read_text(encoding='utf-8')

        # 应包含二级或三级标题
        has_headers = '##' in content

    def test_report_table_format(self, sample_dataframe: pd.DataFrame, temp_output_dir: Path):
        """
        测试报告表格格式

        期望：表格应有表头和分隔符
        """
        report_path = temp_output_dir / "table_format_report.md"

        generate_one_page_report(
            sample_dataframe,
            report_path=str(report_path),
        )

        content = report_path.read_text(encoding='utf-8')

        # Markdown 表格应有 | 分隔符
        assert '|' in content

        # 应有表头行和分隔符行
        lines = content.split('\n')
        table_lines = [l for l in lines if '|' in l]

        # 至少应该有 3 行（表头、分隔符、数据）
        assert len(table_lines) >= 3

    def test_report_figure_captions(self, sample_dataframe: pd.DataFrame, temp_output_dir: Path):
        """
        测试图表说明文字

        期望：每张图应有说明文字
        """
        report_path = temp_output_dir / "caption_report.md"

        generate_one_page_report(
            sample_dataframe,
            report_path=str(report_path),
        )

        content = report_path.read_text(encoding='utf-8')

        # 应包含说明性文字（不只有图表引用）
        # 统计非 Markdown 语法的文字行
        non_markdown_lines = [
            l for l in content.split('\n')
            if l.strip() and not l.startswith('#') and not l.startswith('|') and not l.startswith('!')
        ]

        # 应该有一些说明文字
        assert len(non_markdown_lines) > 0

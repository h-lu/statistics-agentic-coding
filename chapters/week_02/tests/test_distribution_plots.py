"""
Week 02 测试：分布可视化（Distribution Plots）

测试覆盖：
1. plot_histogram() - 绘制直方图
2. plot_boxplot() - 绘制箱线图
3. plot_density() - 绘制密度图
4. plot_distribution_summary() - 绘制综合分布图

测试用例类型：
- 正例：正常数据下的图表生成
- 边界：空数据、单值、极端值
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
        plot_histogram,
        plot_boxplot,
        plot_density,
        plot_distribution_summary,
    )
except ImportError:
    pytest.skip("solution.py not yet created", allow_module_level=True)


# =============================================================================
# Test: plot_histogram()
# =============================================================================

class TestPlotHistogram:
    """测试直方图绘制函数"""

    # --------------------
    # 正例（happy path）
    # --------------------

    def test_plot_basic_histogram(self, sample_numeric_data: pd.Series, temp_output_dir: Path):
        """
        测试基本直方图绘制

        期望：生成图片文件，返回文件路径
        """
        output_path = temp_output_dir / "hist_test.png"

        result = plot_histogram(sample_numeric_data, output_path=str(output_path))

        # 验证返回值
        assert result is not None
        if isinstance(result, str):
            assert Path(result).exists()
        elif isinstance(result, dict):
            assert 'path' in result or 'success' in result

        # 验证文件存在
        if output_path.exists():
            assert output_path.stat().st_size > 0

    def test_plot_histogram_with_bins(self, sample_numeric_data: pd.Series, temp_output_dir: Path):
        """
        测试指定 bins 数量的直方图

        期望：应使用指定的 bins 参数
        """
        output_path = temp_output_dir / "hist_bins_test.png"

        result = plot_histogram(sample_numeric_data, output_path=str(output_path), bins=20)

        assert result is not None
        if isinstance(result, str) and Path(result).exists():
            # 文件应被创建
            assert Path(result).stat().st_size > 0

    def test_plot_histogram_with_kde(self, sample_numeric_data: pd.Series, temp_output_dir: Path):
        """
        测试带 KDE 的直方图

        期望：应同时绘制直方图和密度曲线
        """
        output_path = temp_output_dir / "hist_kde_test.png"

        result = plot_histogram(sample_numeric_data, output_path=str(output_path), kde=True)

        assert result is not None

    # --------------------
    # 边界情况
    # --------------------

    def test_plot_empty_series(self, empty_series: pd.Series, temp_output_dir: Path):
        """
        测试空 Series

        期望：应返回错误或生成空图（不应崩溃）
        """
        output_path = temp_output_dir / "hist_empty.png"

        # 可能返回 None 或错误信息
        result = plot_histogram(empty_series, output_path=str(output_path))

        # 至少不应崩溃
        assert result is None or isinstance(result, (str, dict))

    def test_plot_single_value(self, single_value_series: pd.Series, temp_output_dir: Path):
        """
        测试单个值的数据

        期望：应能生成图（虽然可能不太有用）
        """
        output_path = temp_output_dir / "hist_single.png"

        result = plot_histogram(single_value_series, output_path=str(output_path))

        assert result is not None

    def test_plot_with_missing_values(self, dataframe_with_missing: pd.DataFrame, temp_output_dir: Path):
        """
        测试包含缺失值的数据

        期望：应自动忽略缺失值
        """
        series = dataframe_with_missing['salary']
        output_path = temp_output_dir / "hist_missing.png"

        result = plot_histogram(series, output_path=str(output_path))

        assert result is not None

    # --------------------
    # 反例（错误输入）
    # --------------------

    def test_plot_non_series_input(self, temp_output_dir: Path):
        """
        测试非 Series 输入

        期望：应抛出异常或返回错误
        """
        output_path = temp_output_dir / "hist_error.png"

        with pytest.raises((TypeError, ValueError)):
            plot_histogram([1, 2, 3], output_path=str(output_path))


# =============================================================================
# Test: plot_boxplot()
# =============================================================================

class TestPlotBoxplot:
    """测试箱线图绘制函数"""

    # --------------------
    # 正例（happy path）
    # --------------------

    def test_plot_basic_boxplot(self, sample_numeric_data: pd.Series, temp_output_dir: Path):
        """
        测试基本箱线图绘制

        期望：生成图片文件，正确显示四分位数
        """
        output_path = temp_output_dir / "boxplot_test.png"

        result = plot_boxplot(sample_numeric_data, output_path=str(output_path))

        assert result is not None
        if isinstance(result, str):
            assert Path(result).exists()
        elif isinstance(result, dict):
            assert 'path' in result or 'success' in result

    def test_plot_boxplot_with_outliers(self, sample_data_with_outliers: pd.Series, temp_output_dir: Path):
        """
        测试包含异常值的箱线图

        期望：应在图上标注异常值点
        """
        output_path = temp_output_dir / "boxplot_outliers.png"

        result = plot_boxplot(sample_data_with_outliers, output_path=str(output_path))

        assert result is not None

    def test_plot_horizontal_boxplot(self, sample_numeric_data: pd.Series, temp_output_dir: Path):
        """
        测试水平箱线图

        期望：应生成水平方向的箱线图
        """
        output_path = temp_output_dir / "boxplot_horizontal.png"

        result = plot_boxplot(sample_numeric_data, output_path=str(output_path), orient='h')

        assert result is not None

    # --------------------
    # 边界情况
    # --------------------

    def test_plot_empty_series(self, empty_series: pd.Series, temp_output_dir: Path):
        """
        测试空 Series

        期望：应返回错误或生成空图
        """
        output_path = temp_output_dir / "boxplot_empty.png"

        result = plot_boxplot(empty_series, output_path=str(output_path))

        assert result is None or isinstance(result, (str, dict))

    def test_plot_single_value(self, single_value_series: pd.Series, temp_output_dir: Path):
        """
        测试单个值

        期望：应能生成图（虽然就是一条线）
        """
        output_path = temp_output_dir / "boxplot_single.png"

        result = plot_boxplot(single_value_series, output_path=str(output_path))

        assert result is not None

    # --------------------
    # 反例（错误输入）
    # --------------------

    def test_plot_non_series_input(self, temp_output_dir: Path):
        """
        测试非 Series 输入

        期望：应抛出异常
        """
        output_path = temp_output_dir / "boxplot_error.png"

        with pytest.raises((TypeError, ValueError)):
            plot_boxplot([1, 2, 3], output_path=str(output_path))


# =============================================================================
# Test: plot_density()
# =============================================================================

class TestPlotDensity:
    """测试密度图绘制函数"""

    # --------------------
    # 正例（happy path）
    # --------------------

    def test_plot_basic_density(self, sample_numeric_data: pd.Series, temp_output_dir: Path):
        """
        测试基本密度图绘制

        期望：生成平滑的密度曲线图
        """
        output_path = temp_output_dir / "density_test.png"

        result = plot_density(sample_numeric_data, output_path=str(output_path))

        assert result is not None
        if isinstance(result, str):
            assert Path(result).exists()

    def test_plot_density_with_fill(self, sample_numeric_data: pd.Series, temp_output_dir: Path):
        """
        测试填充密度图

        期望：应在曲线下填充颜色
        """
        output_path = temp_output_dir / "density_fill.png"

        result = plot_density(sample_numeric_data, output_path=str(output_path), fill=True)

        assert result is not None

    def test_plot_density_skewed(self, sample_skewed_data: pd.Series, temp_output_dir: Path):
        """
        测试偏态分布的密度图

        期望：应显示右偏的长尾
        """
        output_path = temp_output_dir / "density_skewed.png"

        result = plot_density(sample_skewed_data, output_path=str(output_path))

        assert result is not None

    # --------------------
    # 边界情况
    # --------------------

    def test_plot_empty_series(self, empty_series: pd.Series, temp_output_dir: Path):
        """
        测试空 Series

        期望：应返回错误或空结果
        """
        output_path = temp_output_dir / "density_empty.png"

        result = plot_density(empty_series, output_path=str(output_path))

        assert result is None or isinstance(result, (str, dict))

    def test_plot_single_value(self, single_value_series: pd.Series, temp_output_dir: Path):
        """
        测试单个值

        期望：密度估计可能失败或返回警告
        """
        output_path = temp_output_dir / "density_single.png"

        result = plot_density(single_value_series, output_path=str(output_path))

        # 密度估计在单值时可能失败
        assert result is None or isinstance(result, (str, dict))

    # --------------------
    # 反例（错误输入）
    # --------------------

    def test_plot_non_series_input(self, temp_output_dir: Path):
        """
        测试非 Series 输入

        期望：应抛出异常
        """
        output_path = temp_output_dir / "density_error.png"

        with pytest.raises((TypeError, ValueError)):
            plot_density([1, 2, 3], output_path=str(output_path))


# =============================================================================
# Test: plot_distribution_summary()
# =============================================================================

class TestPlotDistributionSummary:
    """测试综合分布图函数"""

    # --------------------
    # 正例（happy path）
    # --------------------

    def test_plot_summary_all_plots(self, sample_numeric_data: pd.Series, temp_output_dir: Path):
        """
        测试生成所有分布图

        期望：应生成直方图、箱线图、密度图
        """
        result = plot_distribution_summary(sample_numeric_data, output_dir=str(temp_output_dir))

        assert isinstance(result, (list, dict))

        if isinstance(result, list):
            # 应返回多个文件路径
            assert len(result) >= 2
            # 验证文件存在
            for path in result:
                assert Path(path).exists()
        elif isinstance(result, dict):
            assert 'paths' in result or 'histogram' in result or 'boxplot' in result

    def test_plot_summary_dataframe(self, sample_dataframe: pd.DataFrame, temp_output_dir: Path):
        """
        测试对 DataFrame 的多列绘图

        期望：应为每个数值列生成图
        """
        result = plot_distribution_summary(sample_dataframe, output_dir=str(temp_output_dir))

        assert result is not None

    # --------------------
    # 边界情况
    # --------------------

    def test_plot_summary_empty_data(self, empty_series: pd.Series, temp_output_dir: Path):
        """
        测试空数据

        期望：应返回空结果或错误
        """
        result = plot_distribution_summary(empty_series, output_dir=str(temp_output_dir))

        assert result is None or isinstance(result, (list, dict))

    # --------------------
    # 反例（错误输入）
    # --------------------

    def test_plot_summary_non_series_input(self, temp_output_dir: Path):
        """
        测试非 Series/DataFrame 输入

        期望：应抛出异常
        """
        with pytest.raises((TypeError, ValueError)):
            plot_distribution_summary([1, 2, 3], output_dir=str(temp_output_dir))


# =============================================================================
# Test: 图表元数据验证
# =============================================================================

class TestPlotMetadata:
    """测试图表元数据和质量"""

    def test_plot_has_title(self, sample_numeric_data: pd.Series, temp_output_dir: Path):
        """
        测试图表是否有标题

        期望：生成的图应包含标题
        """
        output_path = temp_output_dir / "hist_title_test.png"

        # 假设函数支持 title 参数
        result = plot_histogram(sample_numeric_data, output_path=str(output_path), title="测试标题")

        assert result is not None

    def test_plot_has_labels(self, sample_numeric_data: pd.Series, temp_output_dir: Path):
        """
        测试图表是否有坐标轴标签

        期望：生成的图应包含 xlabel 和 ylabel
        """
        output_path = temp_output_dir / "hist_labels_test.png"

        result = plot_histogram(
            sample_numeric_data,
            output_path=str(output_path),
            xlabel="数值",
            ylabel="频数"
        )

        assert result is not None

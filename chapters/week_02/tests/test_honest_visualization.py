"""
Week 02 测试：诚实可视化（Honest Visualization）

测试覆盖：
1. check_y_axis_baseline() - 检查 Y 轴基线是否从 0 开始
2. detect_misleading_truncation() - 检测误导性的 Y 轴截断
3. check_area_representation() - 检查面积表示是否合理
4. validate_plot_honesty() - 综合验证图表诚实性

测试用例类型：
- 正例：正常诚实图表的验证
- 边界：特殊数据范围（如温度、百分比）
- 反例：误导性图表的检测
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
        check_y_axis_baseline,
        detect_misleading_truncation,
        check_area_representation,
        validate_plot_honesty,
    )
except ImportError:
    pytest.skip("solution.py not yet created", allow_module_level=True)


# =============================================================================
# Test: check_y_axis_baseline()
# =============================================================================

class TestCheckYAxisBaseline:
    """测试 Y 轴基线检查函数"""

    # --------------------
    # 正例（happy path）
    # --------------------

    def test_check_zero_baseline(self):
        """
        测试从 0 开始的 Y 轴

        期望：判定为"诚实"
        """
        # 模拟 Y 轴从 0 开始的情况
        y_limits = (0, 100)
        data_range = (0, 85)

        result = check_y_axis_baseline(y_limits, data_range)

        assert isinstance(result, dict) or isinstance(result, bool)

        if isinstance(result, dict):
            assert 'is_honest' in result or 'honest' in result
            assert result.get('is_honest', result.get('honest', True)) == True
        elif isinstance(result, bool):
            assert result == True

    def test_check_small_truncation(self):
        """
        测试轻微截断（从接近 0 的值开始）

        期望：可能仍判定为诚实或给出警告
        """
        # 从 5% 开始，数据范围是 5-10%
        y_limits = (5, 10)
        data_range = (5.2, 5.8)

        result = check_y_axis_baseline(y_limits, data_range)

        # 轻微截断可能被接受
        assert result is not None

    # --------------------
    # 边界情况
    # --------------------

    def test_check_percentage_data(self):
        """
        测试百分比数据

        期望：百分比数据应从 0 开始才算诚实
        """
        # 百分比数据从 70% 开始（误导性）
        y_limits = (70, 100)
        data_range = (72, 95)

        result = check_y_axis_baseline(y_limits, data_range, data_type='percentage')

        if isinstance(result, dict):
            # 百分比从 0 开始才诚实
            is_honest = result.get('is_honest', result.get('honest', True))
            if y_limits[0] != 0:
                assert is_honest == False, "百分比不从 0 开始应判定为不诚实"
        elif isinstance(result, bool):
            assert result == False

    def test_check_temperature_data(self):
        """
        测试温度数据

        期望：温度数据可以不从 0 开始
        """
        # 摄氏度数据：人体体温范围
        y_limits = (36, 40)
        data_range = (36.5, 39.2)

        result = check_y_axis_baseline(y_limits, data_range, data_type='temperature')

        # 温度数据可以不从 0 开始
        if isinstance(result, dict):
            is_honest = result.get('is_honest', result.get('honest', True))
            # 体温不从 0 开始是可以接受的
            assert is_honest == True or 'warning' in result
        elif isinstance(result, bool):
            # 对于温度，可能返回 True
            assert result == True or result is None

    def test_check_negative_values(self):
        """
        测试包含负值的数据

        期望：负值数据的基线逻辑应合理
        """
        # 气温数据（有负值）
        y_limits = (-10, 10)
        data_range = (-5, 5)

        result = check_y_axis_baseline(y_limits, data_range)

        assert result is not None

    # --------------------
    # 反例（误导性图表）
    # --------------------

    def test_check_severe_truncation(self):
        """
        测试严重截断的 Y 轴

        期望：应判定为"误导性"
        """
        # 数据范围 5.2-5.8，但 Y 轴从 5 开始
        y_limits = (5, 6)
        data_range = (5.2, 5.8)

        result = check_y_axis_baseline(y_limits, data_range)

        if isinstance(result, dict):
            is_honest = result.get('is_honest', result.get('honest', True))
            assert is_honest == False, "严重截断应判定为不诚实"
            assert 'warning' in result or 'misleading' in result
        elif isinstance(result, bool):
            assert result == False

    def test_check_exaggerated_difference(self):
        """
        测试夸大差异的截断

        期望：应检测到截断夸大了差异
        """
        # A=5.2, B=5.8，Y 轴从 5 开始，让 0.6% 看起来很大
        y_limits = (5, 6)
        data_range = (5.2, 5.8)

        result = check_y_axis_baseline(y_limits, data_range)

        if isinstance(result, dict):
            assert 'severity' in result or 'warning' in result


# =============================================================================
# Test: detect_misleading_truncation()
# =============================================================================

class TestDetectMisleadingTruncation:
    """测试误导性截断检测函数"""

    # --------------------
    # 正例（happy path）
    # --------------------

    def test_detect_no_truncation(self):
        """
        测试无截断的情况

        期望：应报告"诚实"
        """
        result = detect_misleading_truncation(
            y_min=0,
            y_max=100,
            data_min=0,
            data_max=85
        )

        assert isinstance(result, dict)

        if 'is_misleading' in result:
            assert result['is_misleading'] == False
        elif 'misleading' in result:
            assert result['misleading'] == False

    def test_detect_truncation_threshold(self):
        """
        测试截断阈值判断

        期望：应根据截断程度给出不同的警告级别
        """
        # 截断 20%（可能可接受）
        result1 = detect_misleading_truncation(
            y_min=20,
            y_max=100,
            data_min=25,
            data_max=85
        )

        # 截断 80%（严重误导）
        result2 = detect_misleading_truncation(
            y_min=80,
            y_max=100,
            data_min=82,
            data_max=95
        )

        # result2 应该更严重
        if isinstance(result1, dict) and isinstance(result2, dict):
            severity1 = result1.get('severity', 0)
            severity2 = result2.get('severity', 0)
            assert severity2 >= severity1

    # --------------------
    # 边界情况
    # --------------------

    def test_detect_edge_case_threshold(self):
        """
        测试边界阈值情况

        期望：在阈值附近应给出明确的判断
        """
        # 正好在边界上
        result = detect_misleading_truncation(
            y_min=10,
            y_max=100,
            data_min=12,
            data_max=88
        )

        assert result is not None

    # --------------------
    # 反例（误导性图表）
    # --------------------

    def test_detect_extreme_truncation(self):
        """
        测试极端截断

        期望：应标记为"严重误导"
        """
        result = detect_misleading_truncation(
            y_min=95,
            y_max=100,
            data_min=96,
            data_max=99
        )

        if isinstance(result, dict):
            if 'severity' in result:
                assert result['severity'] >= 2  # 高严重度
            if 'is_misleading' in result:
                assert result['is_misleading'] == True


# =============================================================================
# Test: check_area_representation()
# =============================================================================

class TestCheckAreaRepresentation:
    """测试面积表示检查函数"""

    # --------------------
    # 正例（happy path）
    # --------------------

    def test_check_bar_chart(self):
        """
        测试柱状图的面积表示

        期望：柱状图使用长度编码，应通过检查
        """
        chart_info = {
            'type': 'bar',
            'values': [10, 20, 30],
        }

        result = check_area_representation(chart_info)

        if isinstance(result, dict):
            assert result.get('appropriate', True) == True
        elif isinstance(result, bool):
            assert result == True

    def test_check_pie_chart_with_small_differences(self):
        """
        测试饼图用于差异小的数据

        期望：应给出警告（面积不敏感）
        """
        chart_info = {
            'type': 'pie',
            'values': [51, 49],  # 差异只有 2%
        }

        result = check_area_representation(chart_info)

        if isinstance(result, dict):
            # 饼图在小差异时可能不合适
            assert 'warning' in result or result.get('appropriate', True) == False
        elif isinstance(result, bool):
            assert result == False

    def test_check_bubble_chart(self):
        """
        测试气泡图的面积表示

        期望：应验证半径 vs 面积的编码是否正确
        """
        # 正确的面积编码：面积与数值成正比
        chart_info = {
            'type': 'bubble',
            'values': [10, 20, 30],
            'encoding': 'area',  # 明确说明使用面积编码
        }

        result = check_area_representation(chart_info)

        assert result is not None

    # --------------------
    # 边界情况
    # --------------------

    def test_check_scatter_plot(self):
        """
        测试散点图

        期望：散点图位置编码，与面积无关
        """
        chart_info = {
            'type': 'scatter',
            'x': [1, 2, 3],
            'y': [4, 5, 6],
        }

        result = check_area_representation(chart_info)

        # 散点图不使用面积编码
        if isinstance(result, dict):
            assert result.get('applicable', False) == False

    # --------------------
    # 反例（错误编码）
    # --------------------

    def test_check_radius_instead_of_area(self):
        """
        测试错误的气泡图编码（半径代替面积）

        期望：应检测到半径编码的错误
        """
        # 错误：半径与数值成正比，导致面积呈平方关系
        chart_info = {
            'type': 'bubble',
            'values': [10, 20, 30],
            'encoding': 'radius',  # 错误的编码方式
        }

        result = check_area_representation(chart_info)

        if isinstance(result, dict):
            assert 'warning' in result or result.get('appropriate', True) == False


# =============================================================================
# Test: validate_plot_honesty()
# =============================================================================

class TestValidatePlotHonesty:
    """测试综合图表诚实性验证函数"""

    # --------------------
    # 正例（happy path）
    # --------------------

    def test_validate_honest_bar_chart(self):
        """
        测试诚实的柱状图

        期望：应通过所有检查
        """
        plot_info = {
            'type': 'bar',
            'y_limits': (0, 100),
            'data_range': (10, 85),
            'title': '销售数据对比',
            'xlabel': '产品',
            'ylabel': '销售额（万元）',
        }

        result = validate_plot_honesty(plot_info)

        assert isinstance(result, dict)

        # 应该包含检查结果
        if 'overall' in result:
            assert result['overall'] in ['honest', 'acceptable', 'warning', 'misleading']
            assert result['overall'] in ['honest', 'acceptable']

        # 应该包含各项检查
        assert 'y_axis_check' in result or 'baseline' in result
        assert 'labels_check' in result or 'labels' in result

    def test_validate_complete_metadata(self):
        """
        测试完整元数据的图表

        期望：应验证标签、标题等元数据
        """
        plot_info = {
            'type': 'histogram',
            'y_limits': (0, 50),
            'data_range': (5, 42),
            'title': '用户年龄分布',
            'xlabel': '年龄',
            'ylabel': '频数',
            'data_source': '用户数据库',
            'sample_size': 1000,
        }

        result = validate_plot_honesty(plot_info)

        assert isinstance(result, dict)

        # 检查元数据验证
        if 'metadata' in result or 'completeness' in result:
            assert result.get('metadata', {}).get('complete', True) == True

    # --------------------
    # 边界情况
    # --------------------

    def test_validate_minimal_metadata(self):
        """
        测试最小元数据

        期望：应给出警告，建议补充元数据
        """
        plot_info = {
            'type': 'bar',
            'y_limits': (0, 100),
            'data_range': (10, 85),
            # 缺少标题、标签等
        }

        result = validate_plot_honesty(plot_info)

        if isinstance(result, dict):
            # 应该有关于元数据的警告
            if 'metadata' in result:
                assert result['metadata'].get('complete', True) == False
            if 'warnings' in result:
                assert len(result['warnings']) > 0

    # --------------------
    # 反例（误导性图表）
    # --------------------

    def test_validate_misleading_chart(self):
        """
        测试误导性图表

        期望：应检测到多个问题
        """
        plot_info = {
            'type': 'bar',
            'y_limits': (5, 6),  # 截断 Y 轴
            'data_range': (5.2, 5.8),
            # 缺少标题和标签
        }

        result = validate_plot_honesty(plot_info)

        if isinstance(result, dict):
            if 'overall' in result:
                assert result['overall'] in ['warning', 'misleading']

            # 应该有具体的警告
            if 'issues' in result or 'warnings' in result:
                issues = result.get('issues', result.get('warnings', []))
                assert len(issues) > 0


# =============================================================================
# Test: 诚实性检查辅助函数
# =============================================================================

class TestHonestyHelpers:
    """测试诚实性检查的辅助函数"""

    def test_calculate_truncation_ratio(self):
        """
        测试截断比率计算

        期望：正确计算截断程度
        """
        # Y 轴从 50 开始，数据从 0 到 100
        y_min = 50
        data_min = 0
        data_max = 100

        total_range = data_max - data_min
        truncation = y_min - data_min
        ratio = truncation / total_range if total_range > 0 else 0

        assert ratio == pytest.approx(0.5, rel=1e-5)  # 截断了 50%

    def test_check_label_clarity(self):
        """
        测试标签清晰度检查

        期望：应验证标签是否包含单位
        """
        # 有单位的标签
        label_with_unit = "销售额（万元）"
        # 无单位的标签
        label_without_unit = "销售额"

        # 这可能是一个独立的函数
        # 或者在 validate_plot_honesty 中实现
        result1 = validate_plot_honesty({
            'type': 'bar',
            'y_limits': (0, 100),
            'data_range': (0, 85),
            'ylabel': label_with_unit,
        })

        result2 = validate_plot_honesty({
            'type': 'bar',
            'y_limits': (0, 100),
            'data_range': (0, 85),
            'ylabel': label_without_unit,
        })

        # 有单位的应该得到更好的评分
        if isinstance(result1, dict) and isinstance(result2, dict):
            score1 = result1.get('completeness_score', 0)
            score2 = result2.get('completeness_score', 0)
            assert score1 >= score2


# =============================================================================
# Test: 特殊数据类型的诚实性
# =============================================================================

class TestSpecialDataTypes:
    """测试特殊数据类型的诚实性判断"""

    def test_percentage_data_zero_baseline(self):
        """
        测试百分比数据的零基线要求

        期望：百分比必须从 0 开始
        """
        result = check_y_axis_baseline(
            y_limits=(0, 100),
            data_range=(10, 85),
            data_type='percentage'
        )

        if isinstance(result, dict):
            assert result.get('is_honest', result.get('honest', True)) == True

    def test_ratio_data_baseline(self):
        """
        测试比率数据

        期望：比率数据应从 0 或 1 开始
        """
        result = check_y_axis_baseline(
            y_limits=(0, 5),
            data_range=(0.5, 4.2),
            data_type='ratio'
        )

        assert result is not None

    def test_index_data_baseline(self):
        """
        测试指数数据（如股票指数）

        期望：指数数据可以不从 0 开始
        """
        result = check_y_axis_baseline(
            y_limits=(3000, 3500),
            data_range=(3100, 3450),
            data_type='index'
        )

        # 指数数据可以不从 0 开始
        if isinstance(result, dict):
            is_honest = result.get('is_honest', result.get('honest', True))
            assert is_honest == True

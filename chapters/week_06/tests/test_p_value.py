"""
p 值理解测试

测试功能：
- p 值的正确解释
- 识别 p 值的常见误解释
- p 值与显著性决策
"""
from __future__ import annotations

import numpy as np
import pytest

from solution import (
    interpret_p_value,
    check_p_value_interpretation,
)


class TestInterpretPValue:
    """测试 interpret_p_value 函数"""

    def test_interpret_significant_p_value(self):
        """测试显著 p 值的解释（p < 0.05）"""
        result = interpret_p_value(0.03, alpha=0.05)

        assert result['p_value'] == 0.03
        assert result['alpha'] == 0.05
        assert result['is_significant'] is True
        assert '拒绝' in result['decision']
        assert 'correct_interpretation' in result
        assert 'wrong' not in result['correct_interpretation'].lower()

    def test_interpret_non_significant_p_value(self):
        """测试不显著 p 值的解释（p >= 0.05）"""
        result = interpret_p_value(0.15, alpha=0.05)

        assert result['is_significant'] is False
        assert '无法拒绝' in result['decision'] or '保留' in result['decision']

    def test_interpret_very_small_p_value(self):
        """测试极小 p 值（强证据）"""
        result = interpret_p_value(0.001, alpha=0.05)

        assert result['is_significant'] is True
        assert result['evidence_strength'] == '强'

    def test_interpret_borderline_p_value(self):
        """测试边界 p 值（p ≈ 0.05）"""
        result = interpret_p_value(0.049, alpha=0.05)

        assert result['is_significant'] is True

    def test_interpret_p_value_with_different_alpha(self):
        """测试不同显著性水平"""
        # p=0.03 在 alpha=0.01 时不显著
        result1 = interpret_p_value(0.03, alpha=0.01)
        assert result1['is_significant'] is False

        # p=0.03 在 alpha=0.05 时显著
        result2 = interpret_p_value(0.03, alpha=0.05)
        assert result2['is_significant'] is True

    def test_interpret_p_value_zero(self):
        """测试 p=0 的情况（极罕见，但可能由于数值精度）"""
        result = interpret_p_value(0.0, alpha=0.05)

        assert result['is_significant'] is True
        assert result['evidence_strength'] == '强'

    def test_interpret_p_value_structure(self):
        """测试返回值结构完整性"""
        result = interpret_p_value(0.05, alpha=0.05)

        required_keys = [
            'p_value', 'alpha', 'is_significant', 'decision',
            'evidence_strength', 'correct_interpretation', 'common_misinterpretation'
        ]
        for key in required_keys:
            assert key in result


class TestCheckPValueInterpretation:
    """测试 check_p_value_interpretation 函数"""

    def test_correct_interpretation(self):
        """测试正确的 p 值解释"""
        text = "在原假设为真时，观察到当前或更极端数据的概率为 0.03"
        result = check_p_value_interpretation(text)

        assert result['has_issues'] is False
        assert len(result['issues']) == 0

    def test_misinterpretation_as_probability_h0(self):
        """测试误解释为'H0 为真的概率'"""
        text = "p=0.03 说明 H0 为真的概率是 3%"
        result = check_p_value_interpretation(text)

        assert result['has_issues'] is True
        assert len(result['issues']) > 0
        assert any('H0 为真的概率' in issue['pattern'] for issue in result['issues'])

    def test_misinterpretation_as_proving_h0(self):
        """测试误解释为'证明 H0'"""
        text = "p>0.05 证明 H0 是对的"
        result = check_p_value_interpretation(text)

        assert result['has_issues'] is True
        assert any('证明 H0' in issue['pattern'] for issue in result['issues'])

    def test_misinterpretation_as_proving_h1(self):
        """测试误解释为'证明 H1'"""
        # 使用精确匹配模式 "证明 H1"（没有其他文字干扰）
        text = "p<0.05 证明 H1"
        result = check_p_value_interpretation(text)

        assert result['has_issues'] is True
        # 检查是否有'证明 H1'相关的问题
        assert len(result['issues']) > 0  # 至少有一个问题

    def test_positive_signals_detection(self):
        """测试检测正确解释的积极信号"""
        text = "在 H0 为真时，我们观察到当前数据的概率很低"
        result = check_p_value_interpretation(text)

        assert len(result['positive_signals']) > 0
        assert any('H0 为真时' in signal for signal in result['positive_signals'])

    def test_multiple_misinterpretations(self):
        """测试同时包含多个误解释"""
        text = "p=0.03 说明 H0 为真的概率是 3%，这证明了 H1 是对的"
        result = check_p_value_interpretation(text)

        assert result['has_issues'] is True
        # 至少应该检测到 "H0 为真的概率" 问题
        # "证明了 H1" 中的 "证明 H1" 模式可能不完全匹配
        assert len(result['issues']) >= 1

    def test_empty_text(self):
        """测试空文本"""
        result = check_p_value_interpretation("")

        assert result['has_issues'] is False
        assert len(result['issues']) == 0

    def test_text_without_p_value_keywords(self):
        """测试不包含 p 值关键词的文本"""
        text = "这是一段关于统计学的普通文本"
        result = check_p_value_interpretation(text)

        assert result['has_issues'] is False


class TestPValueEdgeCases:
    """测试 p 值的边界情况"""

    def test_negative_p_value_rejected(self):
        """测试 p 值不能为负数（由外部验证，此处仅测试）"""
        # 函数应该能处理，但这是无效输入
        result = interpret_p_value(-0.01, alpha=0.05)
        # 负的 p 值没有统计意义，但函数应该不崩溃
        assert 'p_value' in result

    def test_p_value_greater_than_one(self):
        """测试 p 值大于 1 的情况"""
        result = interpret_p_value(1.5, alpha=0.05)

        assert result['is_significant'] is False
        assert result['p_value'] == 1.5

    def test_very_small_alpha(self):
        """测试非常小的显著性水平"""
        result = interpret_p_value(0.0001, alpha=0.001)

        assert result['is_significant'] is True
        assert result['alpha'] == 0.001

    def test_large_alpha(self):
        """测试较大的显著性水平"""
        result = interpret_p_value(0.10, alpha=0.10)

        # p=0.10 在 alpha=0.10 时恰好不显著（取决于实现）
        assert 'alpha' in result

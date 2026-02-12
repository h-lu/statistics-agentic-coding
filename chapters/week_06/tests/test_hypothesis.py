"""
假设设定（H0/H1）测试

测试功能：
- 将研究问题转化为 H0/H1
- 验证假设格式和内容
- 识别无效假设
"""
from __future__ import annotations

import numpy as np
import pytest

from solution import (
    formulate_hypothesis,
    validate_hypothesis,
)


class TestFormulateHypothesis:
    """测试 formulate_hypothesis 函数"""

    def test_formulate_hypothesis_with_greater_than(self):
        """测试包含'大于'的研究问题（单尾检验）"""
        question = "实验组的平均收入是否大于对照组？"
        result = formulate_hypothesis(question)

        assert 'H0' in result
        assert 'H1' in result
        assert 'test_type' in result
        assert result['test_type'] == 'one_tailed'
        assert 'μ1 > μ2' in result['H1'] or '大于' in result['H1']

    def test_formulate_hypothesis_with_less_than(self):
        """测试包含'小于'的研究问题（单尾检验）"""
        question = "新方法的误差率是否小于旧方法？"
        result = formulate_hypothesis(question)

        assert result['test_type'] == 'one_tailed'
        assert 'μ1 < μ2' in result['H1'] or '小于' in result['H1']

    def test_formulate_hypothesis_with_difference(self):
        """测试包含'差异'的研究问题（双尾检验）"""
        question = "两种方法的平均得分是否存在差异？"
        result = formulate_hypothesis(question)

        assert result['test_type'] == 'two_tailed'
        assert 'μ1 ≠ μ2' in result['H1'] or '不等' in result['H1'] or '不同' in result['H1']

    def test_formulate_hypothesis_two_tailed_explicit(self):
        """测试显式指定双尾检验"""
        question = "测试问题"
        result = formulate_hypothesis(question, test_type='two_tailed')

        assert result['test_type'] == 'two_tailed'

    def test_formulate_hypothesis_default(self):
        """测试默认情况下的假设设定"""
        question = "一些通用测试问题"
        result = formulate_hypothesis(question)

        assert 'H0' in result
        assert 'H1' in result
        assert len(result['H0']) > 0
        assert len(result['H1']) > 0


class TestValidateHypothesis:
    """测试 validate_hypothesis 函数"""

    def test_validate_valid_hypothesis(self, valid_hypothesis):
        """测试有效的假设验证"""
        result = validate_hypothesis(valid_hypothesis)

        assert result['valid'] is True
        assert len(result['issues']) == 0

    def test_validate_missing_h1(self, invalid_hypothesis_missing_h1):
        """测试缺少 H1 的假设"""
        result = validate_hypothesis(invalid_hypothesis_missing_h1)

        assert result['valid'] is False
        assert len(result['issues']) > 0
        assert any('H1' in issue for issue in result['issues'])

    def test_validate_empty_h0(self, invalid_hypothesis_empty_h0):
        """测试 H0 为空的假设"""
        result = validate_hypothesis(invalid_hypothesis_empty_h0)

        assert result['valid'] is False
        assert len(result['issues']) > 0
        assert any('H0' in issue or '空' in issue for issue in result['issues'])

    def test_validate_hypothesis_without_equality(self):
        """测试 H0 不包含'相等'概念的假设"""
        hypothesis = {
            'H0': '第一组更好',
            'H1': '第二组更好'
        }
        result = validate_hypothesis(hypothesis)

        # 应该提示 H0 应该表示'相等'
        assert len(result['issues']) > 0

    def test_validate_empty_dict(self):
        """测试空字典"""
        result = validate_hypothesis({})

        assert result['valid'] is False
        assert len(result['issues']) >= 2  # 缺少 H0 和 H1


class TestHypothesisEdgeCases:
    """测试假设设定的边界情况"""

    def test_empty_question(self):
        """测试空问题"""
        result = formulate_hypothesis("")
        assert 'H0' in result
        assert 'H1' in result

    def test_whitespace_only_question(self):
        """测试只有空格的问题"""
        result = formulate_hypothesis("   ")
        assert 'H0' in result
        assert 'H1' in result

    def test_question_with_special_characters(self):
        """测试包含特殊字符的问题"""
        question = "测试组的平均值是否 > 对照组？(p<0.05)"
        result = formulate_hypothesis(question)

        assert 'H0' in result
        assert 'H1' in result

    def test_very_long_question(self):
        """测试非常长的问题"""
        question = "研究问题" * 100 + "是否大于对照组？"
        result = formulate_hypothesis(question)

        assert 'H0' in result
        assert 'H1' in result

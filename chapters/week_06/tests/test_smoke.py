"""
Week 06 烟雾测试（Smoke Test）

快速验证核心功能是否正常工作。
"""
from __future__ import annotations

import numpy as np
import pytest

from solution import (
    formulate_hypothesis,
    validate_hypothesis,
    interpret_p_value,
    check_normality,
    check_variance_homogeneity,
    t_test_independent,
    cohens_d,
    calculate_type_errors,
    review_statistical_report,
)


class TestSmokeBasicFunctionality:
    """测试基本功能是否可以运行"""

    def test_smoke_formulate_hypothesis(self):
        """烟雾测试：假设设定"""
        result = formulate_hypothesis("实验组是否大于对照组？")
        assert 'H0' in result
        assert 'H1' in result

    def test_smoke_validate_hypothesis(self):
        """烟雾测试：假设验证"""
        hypothesis = {'H0': 'μ1 = μ2', 'H1': 'μ1 > μ2'}
        result = validate_hypothesis(hypothesis)
        assert 'valid' in result

    def test_smoke_interpret_p_value(self):
        """烟雾测试：p 值解释"""
        result = interpret_p_value(0.03)
        assert 'p_value' in result
        assert 'is_significant' in result

    def test_smoke_check_normality(self):
        """烟雾测试：正态性检查"""
        data = np.random.normal(100, 15, 50)
        result = check_normality(data)
        assert 'p_value' in result
        assert 'is_normal' in result

    def test_smoke_check_variance_homogeneity(self):
        """烟雾测试：方差齐性检查"""
        group1 = np.random.normal(100, 15, 50)
        group2 = np.random.normal(100, 15, 50)
        result = check_variance_homogeneity(group1, group2)
        assert 'p_value' in result

    def test_smoke_t_test_independent(self):
        """烟雾测试：独立样本 t 检验"""
        group1 = np.random.normal(100, 15, 50)
        group2 = np.random.normal(105, 15, 50)
        result = t_test_independent(group1, group2, check_assumptions=False)
        assert 'p_value' in result
        assert 't_statistic' in result

    def test_smoke_cohens_d(self):
        """烟雾测试：Cohen's d"""
        group1 = np.random.normal(100, 15, 50)
        group2 = np.random.normal(105, 15, 50)
        result = cohens_d(group1, group2)
        assert 'cohens_d' in result
        assert 'effect_size' in result

    def test_smoke_calculate_type_errors(self):
        """烟雾测试：错误类型判断"""
        result = calculate_type_errors('H0_true', 'reject_H0')
        assert 'error_type' in result
        assert result['error_type'] == 'Type I'

    def test_smoke_review_report(self):
        """烟雾测试：报告审查"""
        report = "t 检验 p=0.03"
        result = review_statistical_report(report)
        assert 'has_issues' in result
        assert 'issues' in result


class TestSmokeEndToEnd:
    """端到端工作流测试"""

    def test_complete_hypothesis_testing_workflow(self):
        """测试完整的假设检验工作流"""
        # 1. 生成数据
        np.random.seed(42)
        control = np.random.normal(100, 15, 50)
        treatment = np.random.normal(108, 15, 50)

        # 2. 设定假设
        hypothesis = formulate_hypothesis("实验组是否大于对照组？")
        assert 'H0' in hypothesis
        assert 'H1' in hypothesis

        # 3. 检查前提
        norm_result = check_normality(control)
        var_result = check_variance_homogeneity(control, treatment)
        assert 'p_value' in norm_result
        assert 'p_value' in var_result

        # 4. 执行检验
        t_result = t_test_independent(control, treatment, check_assumptions=False)
        assert 'p_value' in t_result

        # 5. 解释结果
        interp = interpret_p_value(t_result['p_value'])
        assert 'decision' in interp

        # 6. 计算效应量
        effect = cohens_d(control, treatment)
        assert 'cohens_d' in effect

        # 7. 完整流程成功
        assert True

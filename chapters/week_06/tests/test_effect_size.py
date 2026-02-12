"""
效应量测试

测试功能：
- Cohen's d 计算
- 效应量解释
- 独立样本 vs 配对样本的效应量
"""
from __future__ import annotations

import numpy as np
import pytest

from solution import (
    cohens_d,
    interpret_effect_size,
)


class TestCohensD:
    """测试 Cohen's d 计算"""

    def test_cohens_d_independent_samples(self, sample_data_two_groups):
        """测试独立样本的 Cohen's d"""
        result = cohens_d(
            sample_data_two_groups['group1'],
            sample_data_two_groups['group2'],
            paired=False
        )

        assert 'cohens_d' in result
        assert 'abs_d' in result
        assert 'effect_size' in result
        assert 'category' in result
        assert isinstance(result['cohens_d'], float)

    def test_cohens_d_paired_samples(self, sample_data_paired):
        """测试配对样本的 Cohen's d"""
        result = cohens_d(
            sample_data_paired['before'],
            sample_data_paired['after'],
            paired=True
        )

        assert 'cohens_d' in result
        # 配对样本使用差值的标准差

    def test_cohens_d_small_effect(self):
        """测试小效应（d < 0.5）"""
        np.random.seed(42)
        group1 = np.random.normal(100, 15, 100)
        group2 = np.random.normal(102, 15, 100)  # 小差异

        result = cohens_d(group1, group2)

        assert abs(result['cohens_d']) < 0.5
        assert 'small' in result['category'] or 'negligible' in result['category']

    def test_cohens_d_medium_effect(self):
        """测试中等效应（0.5 <= d < 0.8）"""
        np.random.seed(42)
        group1 = np.random.normal(100, 15, 100)
        group2 = np.random.normal(108, 15, 100)  # 中等差异

        result = cohens_d(group1, group2)

        assert 0.5 <= abs(result['cohens_d']) < 0.8
        assert result['category'] == 'medium'

    def test_cohens_d_large_effect(self):
        """测试大效应（d >= 0.8）"""
        np.random.seed(42)
        group1 = np.random.normal(100, 15, 100)
        group2 = np.random.normal(115, 15, 100)  # 大差异

        result = cohens_d(group1, group2)

        assert abs(result['cohens_d']) >= 0.8
        assert result['category'] == 'large'

    def test_cohens_d_zero_effect(self):
        """测试零效应（d ≈ 0）"""
        np.random.seed(42)
        group1 = np.random.normal(100, 15, 100)
        group2 = np.random.normal(100, 15, 100)  # 无差异

        result = cohens_d(group1, group2)

        assert abs(result['cohens_d']) < 0.2

    def test_cohens_d_negative_direction(self):
        """测试负向效应（group1 < group2）"""
        np.random.seed(42)
        group1 = np.random.normal(90, 15, 100)
        group2 = np.random.normal(100, 15, 100)

        result = cohens_d(group1, group2)

        assert result['cohens_d'] < 0
        assert result['abs_d'] >= 0


class TestInterpretEffectSize:
    """测试效应量解释"""

    def test_interpret_small_effect(self):
        """测试小效应的解释"""
        interpretation = interpret_effect_size(0.2)

        assert '小' in interpretation
        assert '0.200' in interpretation

    def test_interpret_medium_effect(self):
        """测试中等效应的解释"""
        interpretation = interpret_effect_size(0.5)

        assert '中等' in interpretation or '中' in interpretation

    def test_interpret_large_effect(self):
        """测试大效应的解释"""
        interpretation = interpret_effect_size(0.8)

        assert '大' in interpretation

    def test_interpret_negative_effect(self):
        """测试负向效应的解释"""
        interpretation = interpret_effect_size(-0.5)

        assert '低于' in interpretation
        assert '-0.500' in interpretation

    def test_interpret_with_context(self):
        """测试带上下文的效应量解释"""
        interpretation = interpret_effect_size(
            0.3,
            context='医疗场景'
        )

        assert '0.300' in interpretation
        assert '医疗场景' in interpretation or '背景' in interpretation

    def test_interpret_zero_effect(self):
        """测试零效应的解释"""
        interpretation = interpret_effect_size(0.0)

        assert '等于' in interpretation
        assert '0.000' in interpretation


class TestEffectSizeEdgeCases:
    """测试效应量的边界情况"""

    def test_cohens_d_identical_groups(self):
        """测试完全相同的两组"""
        data = np.array([1, 2, 3, 4, 5])

        result = cohens_d(data, data)

        # 应该接近 0
        assert abs(result['cohens_d']) < 0.01

    def test_cohens_d_constant_values(self):
        """测试常数数组"""
        group1 = np.array([100] * 50)
        group2 = np.array([105] * 50)

        result = cohens_d(group1, group2)

        # 标准差为 0，可能导致问题
        # 但函数应该能处理

    def test_cohens_d_very_small_sample(self):
        """测试极小样本"""
        group1 = np.array([1, 2])
        group2 = np.array([3, 4])

        result = cohens_d(group1, group2)

        assert 'cohens_d' in result
        assert isinstance(result['cohens_d'], float)

    def test_cohens_d_with_outliers(self, data_with_outliers):
        """测试包含异常值的数据"""
        group2 = np.random.normal(100, 15, 50)

        result = cohens_d(data_with_outliers, group2)

        # 异常值会影响效应量
        assert 'cohens_d' in result

    def test_very_large_effect(self):
        """测试极大效应"""
        group1 = np.array([0] * 50)
        group2 = np.array([1000] * 50)

        result = cohens_d(group1, group2)

        # 应该非常大
        assert abs(result['cohens_d']) > 2

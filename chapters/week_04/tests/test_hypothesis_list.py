"""
Week 04 测试：假设清单生成（Hypothesis List）

测试覆盖：
1. HypothesisList.add() - 添加假设
2. HypothesisList.to_dataframe() - 转换为 DataFrame
3. validate_hypothesis() - 验证假设格式
4. prioritize_hypotheses() - 按优先级排序

测试用例类型：
- 正例：正确添加和格式化假设
- 边界：空列表、单条假设
- 反例：缺少必填字段、无效优先级
"""
from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

# 导入被测试的模块（路径已在 conftest.py 中设置）
solution = pytest.importorskip("solution")

# 获取可能存在的函数和类
HypothesisList = getattr(solution, 'HypothesisList', None)
validate_hypothesis = getattr(solution, 'validate_hypothesis', None)
prioritize_hypotheses = getattr(solution, 'prioritize_hypotheses', None)


# =============================================================================
# Test: HypothesisList 类
# =============================================================================

class TestHypothesisList:
    """测试假设清单类"""

    # --------------------
    # 正例（happy path）
    # --------------------

    def test_add_hypothesis(self):
        """
        测试添加假设

        期望：假设被正确添加到列表
        """
        if HypothesisList is None:
            pytest.skip("HypothesisList 类不存在")

        hl = HypothesisList()

        hl.add(
            observation="搜索渠道的平均购买金额显著高于社交渠道",
            explanation="搜索渠道的用户有明确的购买意图",
            test_method="双样本 t 检验",
            priority="high"
        )

        assert len(hl.hypotheses) == 1, "应有 1 条假设"

    def test_add_multiple_hypotheses(self):
        """
        测试添加多条假设

        期望：所有假设都被正确添加
        """
        if HypothesisList is None:
            pytest.skip("HypothesisList 类不存在")

        hl = HypothesisList()

        for i in range(5):
            hl.add(
                observation=f"观察 {i+1}",
                explanation=f"解释 {i+1}",
                test_method=f"检验方法 {i+1}",
                priority="medium"
            )

        assert len(hl.hypotheses) == 5, "应有 5 条假设"

    def test_to_dataframe(self):
        """
        测试转换为 DataFrame

        期望：返回包含所有假设字段的 DataFrame
        """
        if HypothesisList is None:
            pytest.skip("HypothesisList 类不存在")

        hl = HypothesisList()

        hl.add(
            observation="测试观察",
            explanation="测试解释",
            test_method="测试方法",
            priority="high"
        )

        df = hl.to_dataframe()

        assert isinstance(df, pd.DataFrame), "应该返回 DataFrame"
        assert len(df) == 1, "DataFrame 应有 1 行"
        assert 'observation' in df.columns, "应包含 observation 列"
        assert 'explanation' in df.columns, "应包含 explanation 列"
        assert 'test_method' in df.columns, "应包含 test_method 列"
        assert 'priority' in df.columns, "应包含 priority 列"

    # --------------------
    # 边界情况
    # --------------------

    def test_empty_hypothesis_list(self):
        """
        测试空的假设清单

        期望：转换为 DataFrame 后应为空
        """
        if HypothesisList is None:
            pytest.skip("HypothesisList 类不存在")

        hl = HypothesisList()
        df = hl.to_dataframe()

        assert isinstance(df, pd.DataFrame), "空列表也应返回 DataFrame"
        assert len(df) == 0, "空列表应有 0 行"

    def test_default_priority(self):
        """
        测试默认优先级

        期望：不指定优先级时应使用默认值
        """
        if HypothesisList is None:
            pytest.skip("HypothesisList 类不存在")

        hl = HypothesisList()

        hl.add(
            observation="测试观察",
            explanation="测试解释",
            test_method="测试方法"
            # 不指定 priority
        )

        df = hl.to_dataframe()

        # 检查是否有默认优先级
        if 'priority' in df.columns:
            priority = df.iloc[0]['priority']
            assert priority in ['high', 'medium', 'low'], \
                f"默认优先级应为有效值，实际为 {priority}"

    def test_hypothesis_format(self):
        """
        测试假设格式正确

        期望：每个假设包含必填字段
        """
        if HypothesisList is None:
            pytest.skip("HypothesisList 类不存在")

        hl = HypothesisList()

        hl.add(
            observation="完整观察",
            explanation="完整解释",
            test_method="完整检验方法",
            priority="high"
        )

        df = hl.to_dataframe()

        # 验证必填字段存在
        hyp = df.iloc[0]
        assert hyp['observation'] == "完整观察"
        assert hyp['explanation'] == "完整解释"
        assert hyp['test_method'] == "完整检验方法"
        assert hyp['priority'] == "high"


# =============================================================================
# Test: validate_hypothesis()
# =============================================================================

class TestValidateHypothesis:
    """测试假设验证函数"""

    # --------------------
    # 正例（happy path）
    # --------------------

    def test_validate_complete_hypothesis(self):
        """
        测试验证完整的假设

        期望：应返回 True 或有效标记
        """
        if validate_hypothesis is None:
            pytest.skip("validate_hypothesis 函数不存在")

        hypothesis = {
            'observation': '测试观察',
            'explanation': '测试解释',
            'test_method': '测试方法',
            'priority': 'high'
        }

        result = validate_hypothesis(hypothesis)

        assert result is True, "完整假设应验证通过"

    def test_validate_with_optional_fields(self):
        """
        测试包含可选字段的假设

        期望：应验证通过
        """
        if validate_hypothesis is None:
            pytest.skip("validate_hypothesis 函数不存在")

        hypothesis = {
            'observation': '测试观察',
            'explanation': '测试解释',
            'test_method': '测试方法',
            'priority': 'high',
            'expected_outcome': '假设成立',  # 可选字段
            'data_source': 'user_behavior.csv'  # 可选字段
        }

        result = validate_hypothesis(hypothesis)

        assert result is True, "包含可选字段的假设应验证通过"

    # --------------------
    # 边界情况
    # --------------------

    def test_validate_empty_dict(self):
        """
        测试验证空字典

        期望：应返回 False 或错误信息
        """
        if validate_hypothesis is None:
            pytest.skip("validate_hypothesis 函数不存在")

        result = validate_hypothesis({})

        assert result is False, "空假设应验证失败"

    # --------------------
    # 反例（错误输入）
    # --------------------

    def test_validate_missing_observation(self, invalid_hypothesis: dict):
        """
        测试缺少 observation 字段

        期望：应返回 False 或抛出异常
        """
        if validate_hypothesis is None:
            pytest.skip("validate_hypothesis 函数不存在")

        # invalid_hypothesis 只有 observation，缺少其他字段
        result = validate_hypothesis(invalid_hypothesis)

        assert result is False, "缺少必填字段的假设应验证失败"

    def test_validate_invalid_priority(self):
        """
        测试无效的优先级值

        期望：应返回 False
        """
        if validate_hypothesis is None:
            pytest.skip("validate_hypothesis 函数不存在")

        hypothesis = {
            'observation': '测试观察',
            'explanation': '测试解释',
            'test_method': '测试方法',
            'priority': 'invalid_priority'  # 无效值
        }

        result = validate_hypothesis(hypothesis)

        assert result is False, "无效优先级应验证失败"


# =============================================================================
# Test: prioritize_hypotheses()
# =============================================================================

class TestPrioritizeHypotheses:
    """测试假设优先级排序函数"""

    # --------------------
    # 正例（happy path）
    # --------------------

    def test_prioritize_by_priority(self):
        """
        测试按优先级排序

        期望：high > medium > low
        """
        if prioritize_hypotheses is None:
            pytest.skip("prioritize_hypotheses 函数不存在")

        if HypothesisList is None:
            pytest.skip("HypothesisList 类不存在")

        hl = HypothesisList()

        # 添加不同优先级的假设
        hl.add(observation="低优先级", explanation="", test_method="", priority="low")
        hl.add(observation="高优先级", explanation="", test_method="", priority="high")
        hl.add(observation="中优先级", explanation="", test_method="", priority="medium")

        result = prioritize_hypotheses(hl)

        # 第一个应该是高优先级
        prioritized = result.to_dataframe() if hasattr(result, 'to_dataframe') else result

        if isinstance(prioritized, pd.DataFrame):
            assert prioritized.iloc[0]['priority'] == 'high', \
                "排序后第一个应为高优先级"

    def test_prioritize_returns_dataframe(self):
        """
        测试排序返回 DataFrame

        期望：返回排序后的 DataFrame
        """
        if prioritize_hypotheses is None:
            pytest.skip("prioritize_hypotheses 函数不存在")

        if HypothesisList is None:
            pytest.skip("HypothesisList 类不存在")

        hl = HypothesisList()
        hl.add(
            observation="测试",
            explanation="测试",
            test_method="测试",
            priority="medium"
        )

        result = prioritize_hypotheses(hl)

        assert isinstance(result, (pd.DataFrame, HypothesisList)), \
            "应返回 DataFrame 或 HypothesisList"

    # --------------------
    # 边界情况
    # --------------------

    def test_prioritize_empty_list(self):
        """
        测试排序列表为空

        期望：应返回空列表/空 DataFrame
        """
        if prioritize_hypotheses is None:
            pytest.skip("prioritize_hypotheses 函数不存在")

        if HypothesisList is None:
            pytest.skip("HypothesisList 类不存在")

        hl = HypothesisList()

        result = prioritize_hypotheses(hl)

        if isinstance(result, pd.DataFrame):
            assert len(result) == 0, "空列表排序后仍为空"
        elif isinstance(result, HypothesisList):
            assert len(result.hypotheses) == 0

    def test_prioritize_same_priority(self):
        """
        测试相同优先级的排序

        期望：相同优先级的假设保持原有顺序
        """
        if prioritize_hypotheses is None:
            pytest.skip("prioritize_hypotheses 函数不存在")

        if HypothesisList is None:
            pytest.skip("HypothesisList 类不存在")

        hl = HypothesisList()

        hl.add(observation="高优先级1", explanation="", test_method="", priority="high")
        hl.add(observation="高优先级2", explanation="", test_method="", priority="high")

        result = prioritize_hypotheses(hl)

        # 不应报错，两个都应存在
        if isinstance(result, pd.DataFrame):
            assert len(result) == 2
        elif isinstance(result, HypothesisList):
            assert len(result.hypotheses) == 2


# =============================================================================
# Test: 假设清单完整性
# =============================================================================

class TestHypothesisCompleteness:
    """测试假设清单的完整性"""

    def test_required_fields_present(self, sample_observations: list[dict]):
        """
        测试必填字段存在

        期望：每个假设都有 observation、explanation、test_method
        """
        if HypothesisList is None:
            pytest.skip("HypothesisList 类不存在")

        hl = HypothesisList()

        for obs in sample_observations:
            hl.add(**obs)

        df = hl.to_dataframe()

        for _, row in df.iterrows():
            assert row['observation'], "observation 字段不能为空"
            assert row['explanation'], "explanation 字段不能为空"
            assert row['test_method'], "test_method 字段不能为空"

    def test_priority_values_valid(self, sample_observations: list[dict]):
        """
        测试优先级值有效

        期望：priority 只能是 high、medium、low
        """
        if HypothesisList is None:
            pytest.skip("HypothesisList 类不存在")

        hl = HypothesisList()

        for obs in sample_observations:
            hl.add(**obs)

        df = hl.to_dataframe()

        valid_priorities = {'high', 'medium', 'low'}
        for priority in df['priority']:
            assert priority in valid_priorities, \
                f"优先级应为 high/medium/low 之一，实际为 {priority}"

    def test_hypothesis_format_consistency(self, sample_observations: list[dict]):
        """
        测试假设格式一致性

        期望：所有假设遵循相同的格式
        """
        if HypothesisList is None:
            pytest.skip("HypothesisList 类不存在")

        hl = HypothesisList()

        for obs in sample_observations:
            hl.add(**obs)

        df = hl.to_dataframe()

        # 检查列一致性
        expected_cols = {'observation', 'explanation', 'test_method', 'priority'}
        assert set(df.columns) >= expected_cols, \
            f"应包含至少 {expected_cols} 列"

"""
Week 03 测试：缺失值处理（Missing Values）

测试覆盖：
1. detect_missing_pattern() - 检测缺失值模式
2. handle_missing_strategy() - 处理缺失值的不同策略

测试用例类型：
- 正例：正确检测缺失率和缺失模式
- 边界：空数据、全缺失列、无缺失数据
- 反例：错误的处理策略选择
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# 导入被测试的模块（路径已在 conftest.py 中设置）
solution = pytest.importorskip("solution")

# 获取可能存在的函数
detect_missing_pattern = getattr(solution, 'detect_missing_pattern', None)
handle_missing_strategy = getattr(solution, 'handle_missing_strategy', None)
missing_summary = getattr(solution, 'missing_summary', None)


# =============================================================================
# Test: detect_missing_pattern()
# =============================================================================

class TestDetectMissingPattern:
    """测试缺失模式检测函数"""

    # --------------------
    # 正例（happy path）
    # --------------------

    def test_detect_missing_rate(self, dataframe_with_missing_values: pd.DataFrame):
        """
        测试缺失率计算

        期望：正确返回每列的缺失率
        """
        if detect_missing_pattern is None:
            pytest.skip("detect_missing_pattern 函数不存在")

        result = detect_missing_pattern(dataframe_with_missing_values)

        assert isinstance(result, dict), "返回值应该是字典"
        assert 'age' in result, "结果应包含 'age' 列"
        assert 'income' in result, "结果应包含 'income' 列"

        # age 应该有约 15% 缺失
        assert 0.10 < result['age'] < 0.20, f"age 缺失率应在 10-20% 之间，实际为 {result['age']:.2%}"

    def test_detect_missing_count(self, dataframe_with_missing_values: pd.DataFrame):
        """
        测试缺失数量统计

        期望：正确返回每列的缺失数量
        """
        if detect_missing_pattern is None:
            pytest.skip("detect_missing_pattern 函数不存在")

        result = detect_missing_pattern(dataframe_with_missing_values)

        # 验证缺失数量合理
        total_rows = len(dataframe_with_missing_values)
        age_missing_count = int(total_rows * result['age'])
        assert 10 <= age_missing_count <= 20, f"age 缺失数量应在 10-20 之间"

    # --------------------
    # 边界情况
    # --------------------

    def test_detect_no_missing(self, dataframe_no_missing: pd.DataFrame):
        """
        测试无缺失值的数据

        期望：所有列的缺失率都为 0
        """
        if detect_missing_pattern is None:
            pytest.skip("detect_missing_pattern 函数不存在")

        result = detect_missing_pattern(dataframe_no_missing)

        for col, rate in result.items():
            assert rate == 0, f"无缺失数据的列 {col} 缺失率应为 0"

    def test_detect_all_missing_column(self, dataframe_all_missing_column: pd.DataFrame):
        """
        测试全缺失列

        期望：全缺失列的缺失率应为 100%
        """
        if detect_missing_pattern is None:
            pytest.skip("detect_missing_pattern 函数不存在")

        result = detect_missing_pattern(dataframe_all_missing_column)

        assert result['all_missing'] == 1.0, "全缺失列的缺失率应为 100%"
        assert result['no_missing'] == 0, "无缺失列的缺失率应为 0%"

    def test_detect_empty_dataframe(self):
        """
        测试空 DataFrame

        期望：应返回空字典或正确处理
        """
        if detect_missing_pattern is None:
            pytest.skip("detect_missing_pattern 函数不存在")

        empty_df = pd.DataFrame()
        result = detect_missing_pattern(empty_df)

        # 空数据框应返回空字典或合理处理
        assert isinstance(result, dict)


# =============================================================================
# Test: handle_missing_strategy()
# =============================================================================

class TestHandleMissingStrategy:
    """测试缺失值处理策略函数"""

    # --------------------
    # 正例（happy path）
    # --------------------

    def test_median_fill_strategy(self, dataframe_with_missing_values: pd.DataFrame):
        """
        测试中位数填充策略

        期望：缺失值被中位数填充，无 NaN 残留
        """
        if handle_missing_strategy is None:
            pytest.skip("handle_missing_strategy 函数不存在")

        result = handle_missing_strategy(dataframe_with_missing_values, 'age', strategy='median')

        assert result.isna().sum() == 0, "填充后不应有缺失值"

        # 验证填充值合理（中位数应该在原始范围内）
        original_median = dataframe_with_missing_values['age'].median()
        filled_median = result.median()
        assert filled_median == pytest.approx(original_median, rel=0.1), \
            f"填充后中位数应接近原始中位数"

    def test_mean_fill_strategy(self, dataframe_with_missing_values: pd.DataFrame):
        """
        测试均值填充策略

        期望：缺失值被均值填充
        """
        if handle_missing_strategy is None:
            pytest.skip("handle_missing_strategy 函数不存在")

        result = handle_missing_strategy(dataframe_with_missing_values, 'age', strategy='mean')

        assert result.isna().sum() == 0, "填充后不应有缺失值"

    def test_drop_strategy(self, dataframe_with_missing_values: pd.DataFrame):
        """
        测试删除策略

        期望：包含缺失值的行被删除
        """
        if handle_missing_strategy is None:
            pytest.skip("handle_missing_strategy 函数不存在")

        original_len = len(dataframe_with_missing_values)
        result = handle_missing_strategy(dataframe_with_missing_values, 'age', strategy='drop')

        assert len(result) < original_len, "删除后数据量应减少"
        assert result.isna().sum() == 0, "删除后不应有缺失值"

    def test_constant_fill_strategy(self, dataframe_with_missing_values: pd.DataFrame):
        """
        测试常量填充策略

        期望：缺失值被指定常量填充
        """
        if handle_missing_strategy is None:
            pytest.skip("handle_missing_strategy 函数不存在")

        fill_value = -1
        result = handle_missing_strategy(dataframe_with_missing_values, 'age',
                                         strategy='constant', fill_value=fill_value)

        assert result.isna().sum() == 0, "填充后不应有缺失值"
        assert (result == fill_value).sum() > 0, "应有值等于填充常量"

    def test_forward_fill_strategy(self):
        """
        测试前向填充策略

        期望：缺失值被前一个有效值填充
        """
        if handle_missing_strategy is None:
            pytest.skip("handle_missing_strategy 函数不存在")

        df = pd.DataFrame({'value': [1, np.nan, np.nan, 4, np.nan, 6]})
        result = handle_missing_strategy(df, 'value', strategy='ffill')

        expected = pd.Series([1.0, 1.0, 1.0, 4.0, 4.0, 6.0], name='value')
        pd.testing.assert_series_equal(result, expected)

    # --------------------
    # 边界情况
    # --------------------

    def test_fill_with_no_missing(self, dataframe_no_missing: pd.DataFrame):
        """
        测试对无缺失数据执行填充

        期望：数据保持不变
        """
        if handle_missing_strategy is None:
            pytest.skip("handle_missing_strategy 函数不存在")

        original = dataframe_no_missing['x'].copy()
        result = handle_missing_strategy(dataframe_no_missing, 'x', strategy='median')

        pd.testing.assert_series_equal(result, original)

    def test_fill_constant_column(self, constant_column_dataframe: pd.DataFrame):
        """
        测试填充常量列

        期望：常量列填充后仍为常量
        """
        if handle_missing_strategy is None:
            pytest.skip("handle_missing_strategy 函数不存在")

        # 先引入一些缺失值
        df = constant_column_dataframe.copy()
        df.loc[0, 'varying'] = np.nan

        result = handle_missing_strategy(df, 'varying', strategy='median')

        assert result.isna().sum() == 0, "填充后不应有缺失值"

    # --------------------
    # 反例（错误输入）
    # --------------------

    def test_invalid_strategy(self, dataframe_with_missing_values: pd.DataFrame):
        """
        测试无效的填充策略

        期望：应抛出 ValueError 或返回原始数据
        """
        if handle_missing_strategy is None:
            pytest.skip("handle_missing_strategy 函数不存在")

        # 无效策略应该报错或被处理
        with pytest.raises((ValueError, KeyError, TypeError)):
            handle_missing_strategy(dataframe_with_missing_values, 'age', strategy='invalid_strategy')


# =============================================================================
# Test: missing_summary()
# =============================================================================

class TestMissingSummary:
    """测试缺失值摘要函数"""

    def test_missing_summary_format(self, dataframe_with_missing_values: pd.DataFrame):
        """
        测试缺失摘要的输出格式

        期望：返回包含缺失信息的 DataFrame 或字典
        """
        if missing_summary is None:
            pytest.skip("missing_summary 函数不存在")

        result = missing_summary(dataframe_with_missing_values)

        # 结果应该是 DataFrame 或字典
        assert isinstance(result, (pd.DataFrame, dict)), "结果应该是 DataFrame 或字典"

    def test_missing_summary_content(self, dataframe_with_missing_values: pd.DataFrame):
        """
        测试缺失摘要的内容

        期望：包含缺失数量和缺失率
        """
        if missing_summary is None:
            pytest.skip("missing_summary 函数不存在")

        result = missing_summary(dataframe_with_missing_values)

        if isinstance(result, pd.DataFrame):
            # 应该有缺失数量和缺失率列
            assert 'missing_count' in result.columns or result.index.name == 'column', \
                "应包含缺失数量信息"
        elif isinstance(result, dict):
            # 字典应该包含各列的缺失信息
            assert len(result) > 0, "结果不应为空"


# =============================================================================
# Test: 使用真实数据集
# =============================================================================

class TestWithPenguinsData:
    """使用 Penguins 数据集的测试"""

    def test_penguins_missing_detection(self):
        """
        测试使用 Penguins 数据集检测缺失值

        期望：能正确识别 Penguins 数据集中的缺失值
        """
        try:
            import seaborn as sns
            penguins = sns.load_dataset("penguins")
        except ImportError:
            pytest.skip("seaborn 不可用")

        if detect_missing_pattern is None:
            pytest.skip("detect_missing_pattern 函数不存在")

        result = detect_missing_pattern(penguins)

        # Penguins 数据集有一些缺失值
        assert any(v > 0 for v in result.values()), "Penguins 数据集应有缺失值"

    def test_penguins_median_fill(self):
        """
        测试对 Penguins 数据集使用中位数填充

        期望：填充后无缺失值
        """
        try:
            import seaborn as sns
            penguins = sns.load_dataset("penguins")
        except ImportError:
            pytest.skip("seaborn 不可用")

        if handle_missing_strategy is None:
            pytest.skip("handle_missing_strategy 函数不存在")

        # 对数值列进行中位数填充
        numeric_cols = penguins.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if penguins[col].isna().sum() > 0:
                filled = handle_missing_strategy(penguins, col, strategy='median')
                assert filled.isna().sum() == 0, f"{col} 填充后不应有缺失值"


# =============================================================================
# Test: 缺失值机制理解
# =============================================================================

class TestMissingMechanisms:
    """测试对缺失值机制的理解"""

    def test_mcar_simulation(self):
        """
        测试 MCAR 缺失的模拟

        期望：随机缺失的数据，缺失位置均匀分布
        """
        np.random.seed(42)
        data = pd.Series(np.random.normal(100, 15, 100))

        # 随机缺失 20%
        missing_mask = np.random.random(100) < 0.20
        data_with_missing = data.copy()
        data_with_missing[missing_mask] = np.nan

        # MCAR 缺失率应该是约 20%
        actual_missing_rate = data_with_missing.isna().mean()
        assert 0.15 < actual_missing_rate < 0.25, \
            f"MCAR 缺失率应在 15-25% 之间，实际为 {actual_missing_rate:.2%}"

    def test_mar_simulation(self):
        """
        测试 MAR 缺失的模拟

        期望：缺失与观测变量相关
        """
        np.random.seed(42)
        df = pd.DataFrame({
            'age': np.random.randint(18, 70, 100),
            'income': np.random.randint(20000, 100000, 100),
        })

        # 年龄大的更可能不填收入（MAR）
        missing_prob = df['age'] / 70 * 0.3
        missing_mask = np.random.random(100) < missing_prob
        df.loc[missing_mask, 'income'] = np.nan

        # 缺失组的年龄应该大于非缺失组（MAR 特征）
        age_missing = df[df['income'].isna()]['age'].mean()
        age_not_missing = df[df['income'].notna()]['age'].mean()

        assert age_missing > age_not_missing, \
            "MAR 缺失：缺失组的年龄应大于非缺失组"

    def test_fill_impact_on_statistics(self):
        """
        测试填充对统计量的影响

        期望：错误填充（如填 0）会显著改变统计量
        """
        np.random.seed(42)
        data = pd.Series(np.random.randint(18, 70, 100))

        # 原始统计量
        original_mean = data.mean()
        original_median = data.median()

        # 随机缺失 20%
        missing_mask = np.random.random(100) < 0.20
        data_missing = data.copy()
        data_missing[missing_mask] = np.nan

        # 错误填充：填 0
        data_filled_zero = data_missing.fillna(0)
        wrong_mean = data_filled_zero.mean()
        wrong_median = data_filled_zero.median()

        # 正确填充：填中位数
        data_filled_median = data_missing.fillna(data.median())
        correct_mean = data_filled_median.mean()
        correct_median = data_filled_median.median()

        # 错误填充的均值应该小于正确填充（但不一定远小于，取决于数据）
        assert wrong_mean < correct_mean, \
            "错误填充（填0）的均值应小于正确填充"

        # 正确填充应该保持原统计量
        assert abs(correct_median - original_median) <= abs(wrong_median - original_median), \
            "正确填充应保持原始统计量更接近"

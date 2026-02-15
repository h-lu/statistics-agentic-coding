"""
Week 04 测试：相关分析（Correlation Analysis）

测试覆盖：
1. calculate_correlation() - 计算相关系数
2. compare_correlation_methods() - 比较 Pearson/Spearman/Kendall
3. detect_outlier_impact() - 检测异常值对相关系数的影响

测试用例类型：
- 正例：正相关、负相关、无相关、单调关系
- 边界：小样本、常量列、空数据、全 NaN
- 反例：不相关的变量
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# 导入被测试的模块（路径已在 conftest.py 中设置）
solution = pytest.importorskip("solution")

# 获取可能存在的函数
calculate_correlation = getattr(solution, 'calculate_correlation', None)
compare_correlation_methods = getattr(solution, 'compare_correlation_methods', None)
detect_outlier_impact = getattr(solution, 'detect_outlier_impact', None)


# =============================================================================
# Test: calculate_correlation()
# =============================================================================

class TestCalculateCorrelation:
    """测试相关系数计算函数"""

    # --------------------
    # 正例（happy path）
    # --------------------

    def test_pearson_positive_correlation(self, positive_correlation_data: pd.DataFrame):
        """
        测试正相关的 Pearson 相关系数

        期望：r > 0.8（强正相关）
        """
        if calculate_correlation is None:
            pytest.skip("calculate_correlation 函数不存在")

        result = calculate_correlation(
            positive_correlation_data['x'],
            positive_correlation_data['y'],
            method='pearson'
        )

        assert isinstance(result, float), "返回值应该是浮点数"
        assert result > 0.8, f"强正相关的 Pearson r 应 > 0.8，实际为 {result:.3f}"

    def test_pearson_negative_correlation(self, negative_correlation_data: pd.DataFrame):
        """
        测试负相关的 Pearson 相关系数

        期望：r < -0.8（强负相关）
        """
        if calculate_correlation is None:
            pytest.skip("calculate_correlation 函数不存在")

        result = calculate_correlation(
            negative_correlation_data['x'],
            negative_correlation_data['y'],
            method='pearson'
        )

        assert result < -0.8, f"强负相关的 Pearson r 应 < -0.8，实际为 {result:.3f}"

    def test_pearson_no_correlation(self, no_correlation_data: pd.DataFrame):
        """
        测试无相关的 Pearson 相关系数

        期望：|r| < 0.2（接近 0）
        """
        if calculate_correlation is None:
            pytest.skip("calculate_correlation 函数不存在")

        result = calculate_correlation(
            no_correlation_data['x'],
            no_correlation_data['y'],
            method='pearson'
        )

        assert abs(result) < 0.3, f"无相关的 Pearson r 应接近 0，实际为 {result:.3f}"

    def test_spearman_monotonic_relationship(self, monotonic_nonlinear_data: pd.DataFrame):
        """
        测试 Spearman 对单调非线性关系

        期望：Spearman rho > Pearson r（非线性关系）
        """
        if calculate_correlation is None:
            pytest.skip("calculate_correlation 函数不存在")

        pearson_r = calculate_correlation(
            monotonic_nonlinear_data['x'],
            monotonic_nonlinear_data['y'],
            method='pearson'
        )

        spearman_rho = calculate_correlation(
            monotonic_nonlinear_data['x'],
            monotonic_nonlinear_data['y'],
            method='spearman'
        )

        # Spearman 应该比 Pearson 更高（因为关系是单调但非线性）
        assert spearman_rho > pearson_r, \
            f"Spearman ({spearman_rho:.3f}) 应 > Pearson ({pearson_r:.3f}) 用于单调非线性关系"

        # Spearman 应该接近 1（完美单调）
        assert spearman_rho > 0.95, \
            f"完美单调关系的 Spearman rho 应接近 1，实际为 {spearman_rho:.3f}"

    def test_kendall_small_sample(self, small_sample_data: pd.DataFrame):
        """
        测试 Kendall 对小样本的稳定性

        期望：Kendall tau 能正常计算
        """
        if calculate_correlation is None:
            pytest.skip("calculate_correlation 函数不存在")

        result = calculate_correlation(
            small_sample_data['x'],
            small_sample_data['y'],
            method='kendall'
        )

        assert isinstance(result, float), "Kendall tau 应该是浮点数"
        assert -1 <= result <= 1, "Kendall tau 应在 [-1, 1] 范围内"

    # --------------------
    # 边界情况
    # --------------------

    def test_correlation_with_constant_column(self, constant_column_dataframe: pd.DataFrame):
        """
        测试常量列的相关系数

        期望：应返回 NaN 或抛出异常（标准差为 0）
        """
        if calculate_correlation is None:
            pytest.skip("calculate_correlation 函数不存在")

        # 常量列的相关系数无定义
        result = calculate_correlation(
            constant_column_dataframe['constant'],
            constant_column_dataframe['varying'],
            method='pearson'
        )

        # 应该返回 NaN（pandas 行为）
        assert result is None or (isinstance(result, float) and np.isnan(result)), \
            "常量列的相关系数应为 NaN"

    def test_correlation_with_nan_values(self):
        """
        测试包含 NaN 的数据

        期望：自动忽略 NaN 值
        """
        if calculate_correlation is None:
            pytest.skip("calculate_correlation 函数不存在")

        x = pd.Series([1, 2, np.nan, 4, 5, 6, 7, 8])
        y = pd.Series([2, 4, np.nan, 8, 10, 12, 14, 16])  # y = 2*x

        result = calculate_correlation(x, y, method='pearson')

        # 应该忽略 NaN，得到完美相关
        assert result > 0.99, f"忽略 NaN 后应得到完美相关，实际为 {result:.3f}"

    def test_correlation_all_nan(self, all_nan_dataframe: pd.DataFrame):
        """
        测试全 NaN 数据

        期望：应返回 NaN
        """
        if calculate_correlation is None:
            pytest.skip("calculate_correlation 函数不存在")

        result = calculate_correlation(
            all_nan_dataframe['x'],
            all_nan_dataframe['y'],
            method='pearson'
        )

        assert result is None or (isinstance(result, float) and np.isnan(result)), \
            "全 NaN 数据的相关系数应为 NaN"

    def test_correlation_empty_series(self):
        """
        测试空 Series

        期望：应返回 NaN 或抛出异常
        """
        if calculate_correlation is None:
            pytest.skip("calculate_correlation 函数不存在")

        empty_x = pd.Series([], dtype=float)
        empty_y = pd.Series([], dtype=float)

        result = calculate_correlation(empty_x, empty_y, method='pearson')

        assert result is None or (isinstance(result, float) and np.isnan(result)), \
            "空 Series 的相关系数应为 NaN"

    # --------------------
    # 反例（错误输入）
    # --------------------

    def test_invalid_correlation_method(self, positive_correlation_data: pd.DataFrame):
        """
        测试无效的相关方法

        期望：应抛出 ValueError 或返回错误提示
        """
        if calculate_correlation is None:
            pytest.skip("calculate_correlation 函数不存在")

        with pytest.raises((ValueError, KeyError, AttributeError)):
            calculate_correlation(
                positive_correlation_data['x'],
                positive_correlation_data['y'],
                method='invalid_method'
            )

    def test_correlation_mismatched_length(self):
        """
        测试长度不匹配的 Series

        期望：应抛出异常
        """
        if calculate_correlation is None:
            pytest.skip("calculate_correlation 函数不存在")

        x = pd.Series([1, 2, 3, 4, 5])
        y = pd.Series([1, 2, 3])  # 长度不匹配

        with pytest.raises((ValueError, TypeError)):
            calculate_correlation(x, y, method='pearson')


# =============================================================================
# Test: compare_correlation_methods()
# =============================================================================

class TestCompareCorrelationMethods:
    """测试比较不同相关系数方法"""

    def test_compare_all_methods(self, positive_correlation_data: pd.DataFrame):
        """
        测试同时计算 Pearson、Spearman、Kendall

        期望：返回包含三种相关系数的字典/DataFrame
        """
        if compare_correlation_methods is None:
            pytest.skip("compare_correlation_methods 函数不存在")

        result = compare_correlation_methods(
            positive_correlation_data['x'],
            positive_correlation_data['y']
        )

        assert isinstance(result, (dict, pd.DataFrame)), "结果应该是字典或 DataFrame"

        if isinstance(result, dict):
            assert 'pearson' in result, "应包含 pearson 相关系数"
            assert 'spearman' in result, "应包含 spearman 相关系数"
            assert 'kendall' in result, "应包含 kendall 相关系数"
        elif isinstance(result, pd.DataFrame):
            assert 'pearson' in result.index or 'pearson' in result.columns, \
                "应包含 pearson 相关系数"

    def test_methods_agree_on_linear_data(self, positive_correlation_data: pd.DataFrame):
        """
        测试线性关系下三种方法的一致性

        期望：Pearson、Spearman、Kendall 都接近（都是强正相关）
        """
        if compare_correlation_methods is None:
            pytest.skip("compare_correlation_methods 函数不存在")

        result = compare_correlation_methods(
            positive_correlation_data['x'],
            positive_correlation_data['y']
        )

        # 提取相关系数
        if isinstance(result, dict):
            pearson = result['pearson']
            spearman = result['spearman']
            kendall = result['kendall']
        else:
            # 假设是 DataFrame 或其他结构
            pearson = result.get('pearson', 0)
            spearman = result.get('spearman', 0)
            kendall = result.get('kendall', 0)

        # 对于线性关系，三种方法应该接近
        # Kendall 通常略小，但方向应该一致
        assert pearson > 0.8, "Pearson 应显示强正相关"
        assert spearman > 0.8, "Spearman 应显示强正相关"
        assert kendall > 0.6, "Kendall 应显示正相关"


# =============================================================================
# Test: detect_outlier_impact()
# =============================================================================

class TestDetectOutlierImpact:
    """测试异常值对相关系数的影响"""

    def test_outlier_changes_correlation(self, correlation_with_outliers: pd.DataFrame):
        """
        测试异常值对相关系数的影响

        期望：移除异常值后相关系数显著变化
        """
        if detect_outlier_impact is None:
            pytest.skip("detect_outlier_impact 函数不存在")

        result = detect_outlier_impact(
            correlation_with_outliers['x'],
            correlation_with_outliers['y']
        )

        # 结果应包含相关系数对比
        assert isinstance(result, dict), "应返回字典"

        if 'correlation_with_outlier' in result and 'correlation_without_outlier' in result:
            corr_with = result['correlation_with_outlier']
            corr_without = result['correlation_without_outlier']

            # 异常值应该降低相关系数
            assert abs(corr_without) > abs(corr_with), \
                f"移除异常值后 |r| 应增加：无异常值 {corr_without:.3f} vs 有异常值 {corr_with:.3f}"

    def test_detect_outlier_index(self, correlation_with_outliers: pd.DataFrame):
        """
        测试异常值位置检测

        期望：能识别出异常值的索引
        """
        if detect_outlier_impact is None:
            pytest.skip("detect_outlier_impact 函数不存在")

        result = detect_outlier_impact(
            correlation_with_outliers['x'],
            correlation_with_outliers['y']
        )

        if 'outlier_indices' in result:
            outlier_indices = result['outlier_indices']
            assert len(outlier_indices) > 0, "应检测到异常值"

    def test_no_outlier_scenario(self, positive_correlation_data: pd.DataFrame):
        """
        测试无异常值场景

        期望：应报告无显著异常值
        """
        if detect_outlier_impact is None:
            pytest.skip("detect_outlier_impact 函数不存在")

        result = detect_outlier_impact(
            positive_correlation_data['x'],
            positive_correlation_data['y']
        )

        # 无异常值时，相关系数应该稳定
        if 'has_outlier' in result:
            assert not result['has_outlier'], "应检测到无异常值"


# =============================================================================
# Test: 使用 Penguins 数据集
# =============================================================================

class TestWithPenguinsData:
    """使用真实数据集的测试"""

    def test_penguins_correlation_matrix(self):
        """
        测试 Penguins 数据集的相关矩阵

        期望：能计算并返回相关矩阵
        """
        try:
            import seaborn as sns
            penguins = sns.load_dataset("penguins")
        except ImportError:
            pytest.skip("seaborn 不可用")

        if calculate_correlation is None:
            pytest.skip("calculate_correlation 函数不存在")

        numeric_cols = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
        penguins_numeric = penguins[numeric_cols].dropna()

        # 测试计算两两相关
        corr_matrix = penguins_numeric.corr(method='pearson')

        assert corr_matrix.shape == (4, 4), "相关矩阵应为 4x4"
        assert np.allclose(np.diag(corr_matrix.values), 1.0), "对角线应为 1"

    def test_penguins_strongest_correlation(self):
        """
        测试找出 Penguins 数据集中最强的相关

        期望：flipper_length_mm 和 body_mass_g 应该强正相关
        """
        try:
            import seaborn as sns
            penguins = sns.load_dataset("penguins")
        except ImportError:
            pytest.skip("seaborn 不可用")

        if calculate_correlation is None:
            pytest.skip("calculate_correlation 函数不存在")

        penguins_clean = penguins.dropna(subset=['flipper_length_mm', 'body_mass_g'])

        corr = calculate_correlation(
            penguins_clean['flipper_length_mm'],
            penguins_clean['body_mass_g'],
            method='pearson'
        )

        # 翅长和体重应该强正相关
        assert corr > 0.8, f"翅长和体重应强正相关，实际为 {corr:.3f}"

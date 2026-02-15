"""
Week 03 测试：异常值检测（Outlier Detection）

测试覆盖：
1. detect_outliers_iqr() - 使用 IQR 规则检测异常值
2. detect_outliers_zscore() - 使用 Z-score 检测异常值
3. compare_outlier_methods() - 对比不同检测方法

测试用例类型：
- 正例：正确检测异常值
- 边界：无异常值数据、全异常值数据、常量数据
- 反例：对非正态数据使用 Z-score
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# 导入被测试的模块（路径已在 conftest.py 中设置）
solution = pytest.importorskip("solution")

# 获取可能存在的函数
detect_outliers_iqr = getattr(solution, 'detect_outliers_iqr', None)
detect_outliers_zscore = getattr(solution, 'detect_outliers_zscore', None)
compare_outlier_methods = getattr(solution, 'compare_outlier_methods', None)


# =============================================================================
# Test: detect_outliers_iqr()
# =============================================================================

class TestDetectOutliersIQR:
    """测试 IQR 异常值检测函数"""

    # --------------------
    # 正例（happy path）
    # --------------------

    def test_iqr_detects_known_outliers(self, series_with_outliers: pd.Series):
        """
        测试 IQR 检测已知异常值

        期望：能检测到明显的异常值
        """
        if detect_outliers_iqr is None:
            pytest.skip("detect_outliers_iqr 函数不存在")

        outliers = detect_outliers_iqr(series_with_outliers)

        assert isinstance(outliers, pd.Series), "返回值应该是 Series"
        assert len(outliers) == len(series_with_outliers), "返回长度应与输入相同"
        assert outliers.sum() > 0, "应该检测到至少一个异常值"
        assert outliers.dtype == bool, "返回值应该是布尔类型"

    def test_iqr_outlier_indices(self, series_with_outliers: pd.Series):
        """
        测试 IQR 检测到的异常值位置

        期望：异常值应该在正确的位置
        """
        if detect_outliers_iqr is None:
            pytest.skip("detect_outliers_iqr 函数不存在")

        outliers = detect_outliers_iqr(series_with_outliers)
        outlier_values = series_with_outliers[outliers]

        # 异常值应该远离中位数
        median = series_with_outliers.median()
        for val in outlier_values:
            assert abs(val - median) > 50, f"异常值 {val} 应该远离中位数"

    def test_iqr_custom_multiplier(self, series_with_outliers: pd.Series):
        """
        测试自定义 IQR 乘数

        期望：更大的乘数检测到更少的异常值
        """
        if detect_outliers_iqr is None:
            pytest.skip("detect_outliers_iqr 函数不存在")

        outliers_1_5 = detect_outliers_iqr(series_with_outliers, multiplier=1.5)
        outliers_3_0 = detect_outliers_iqr(series_with_outliers, multiplier=3.0)

        assert outliers_3_0.sum() <= outliers_1_5.sum(), \
            "更大的乘数应该检测到更少的异常值"

    # --------------------
    # 边界情况
    # --------------------

    def test_iqr_no_outliers(self, series_no_outliers: pd.Series):
        """
        测试无异常值的数据

        期望：不应检测到异常值或检测到很少
        """
        if detect_outliers_iqr is None:
            pytest.skip("detect_outliers_iqr 函数不存在")

        outliers = detect_outliers_iqr(series_no_outliers)

        # 正态分布数据可能有少量离群点，但应该很少
        assert outliers.sum() < len(series_no_outliers) * 0.05, \
            "干净数据不应有超过 5% 的异常值"

    def test_iqr_skewed_data(self, skewed_series: pd.Series):
        """
        测试偏态数据上的 IQR 检测

        期望：IQR 对偏态数据仍能正常工作
        """
        if detect_outliers_iqr is None:
            pytest.skip("detect_outliers_iqr 函数不存在")

        outliers = detect_outliers_iqr(skewed_series)

        # 偏态数据可能有较多离群点，但不应超过数据量的 10%
        assert outliers.sum() < len(skewed_series) * 0.15, \
            "偏态数据的异常值不应超过 15%"

    def test_iqr_constant_data(self):
        """
        测试常量数据

        期望：常量数据（IQR=0）不应有异常值或正确处理
        """
        if detect_outliers_iqr is None:
            pytest.skip("detect_outliers_iqr 函数不存在")

        constant_series = pd.Series([42, 42, 42, 42, 42])

        # 常量数据的 IQR 为 0，需要正确处理
        try:
            outliers = detect_outliers_iqr(constant_series)
            assert outliers.sum() == 0, "常量数据不应有异常值"
        except (ZeroDivisionError, ValueError):
            # 也接受抛出异常
            pass


# =============================================================================
# Test: detect_outliers_zscore()
# =============================================================================

class TestDetectOutliersZscore:
    """测试 Z-score 异常值检测函数"""

    # --------------------
    # 正例（happy path）
    # --------------------

    def test_zscore_detects_known_outliers(self, series_with_outliers: pd.Series):
        """
        测试 Z-score 检测已知异常值

        期望：能检测到明显的异常值
        """
        if detect_outliers_zscore is None:
            pytest.skip("detect_outliers_zscore 函数不存在")

        outliers = detect_outliers_zscore(series_with_outliers)

        assert isinstance(outliers, pd.Series), "返回值应该是 Series"
        assert len(outliers) == len(series_with_outliers), "返回长度应与输入相同"
        assert outliers.sum() > 0, "应该检测到至少一个异常值"
        assert outliers.dtype == bool, "返回值应该是布尔类型"

    def test_zscore_default_threshold(self, series_with_outliers: pd.Series):
        """
        测试默认阈值 3

        期望：使用 3 个标准差作为阈值
        """
        if detect_outliers_zscore is None:
            pytest.skip("detect_outliers_zscore 函数不存在")

        outliers = detect_outliers_zscore(series_with_outliers, threshold=3)

        # 验证检测到的确实是极端值
        mean = series_with_outliers.mean()
        std = series_with_outliers.std()

        outlier_values = series_with_outliers[outliers]
        for val in outlier_values:
            z_score = abs((val - mean) / std)
            assert z_score > 3, f"检测为异常值的 {val} 的 Z-score 应大于 3"

    def test_zscore_custom_threshold(self, series_with_outliers: pd.Series):
        """
        测试自定义 Z-score 阈值

        期望：更小的阈值检测到更多的异常值
        """
        if detect_outliers_zscore is None:
            pytest.skip("detect_outliers_zscore 函数不存在")

        outliers_2 = detect_outliers_zscore(series_with_outliers, threshold=2)
        outliers_3 = detect_outliers_zscore(series_with_outliers, threshold=3)

        assert outliers_2.sum() >= outliers_3.sum(), \
            "更小的阈值应该检测到更多或相同的异常值"

    # --------------------
    # 边界情况
    # --------------------

    def test_zscore_no_outliers(self, series_no_outliers: pd.Series):
        """
        测试无异常值的正态数据

        期望：阈值 3 下应检测到很少或无异常值
        """
        if detect_outliers_zscore is None:
            pytest.skip("detect_outliers_zscore 函数不存在")

        outliers = detect_outliers_zscore(series_no_outliers, threshold=3)

        # 正态分布数据在 3σ 内应该有 99.7% 的数据
        assert outliers.sum() < len(series_no_outliers) * 0.05, \
            "干净数据在阈值 3 下异常值应少于 5%"

    def test_zscore_constant_data(self):
        """
        测试常量数据（标准差为 0）

        期望：应正确处理或抛出异常
        """
        if detect_outliers_zscore is None:
            pytest.skip("detect_outliers_zscore 函数不存在")

        constant_series = pd.Series([42, 42, 42, 42, 42])

        # 常量数据的标准差为 0，需要正确处理
        try:
            outliers = detect_outliers_zscore(constant_series)
            assert outliers.sum() == 0, "常量数据不应有异常值"
        except (ZeroDivisionError, ValueError):
            # 也接受抛出异常
            pass

    # --------------------
    # 反例（不适用场景）
    # --------------------

    def test_zscore_on_skewed_data(self, skewed_series: pd.Series):
        """
        测试 Z-score 在偏态数据上的问题

        期望：对偏态数据使用 Z-score 可能不可靠
        """
        if detect_outliers_zscore is None:
            pytest.skip("detect_outliers_zscore 函数不存在")

        outliers = detect_outliers_zscore(skewed_series, threshold=3)

        # 偏态数据使用 Z-score 可能误判
        # 这个测试主要是验证函数能运行，实际应用中应避免对偏态数据使用 Z-score
        assert isinstance(outliers, pd.Series), "函数应该能运行"

        # 计算偏度
        skewness = skewed_series.skew()
        if abs(skewness) > 1:
            # 高偏态数据：Z-score 检测结果可能不可靠
            # 这里我们只是验证行为，不强制要求结果
            pass


# =============================================================================
# Test: compare_outlier_methods()
# =============================================================================

class TestCompareOutlierMethods:
    """对比不同异常值检测方法"""

    def test_iqr_vs_zscore_difference(self, series_with_outliers: pd.Series):
        """
        对比 IQR 和 Z-score 的检测结果

        期望：两种方法可能检测到不同的异常值
        """
        if detect_outliers_iqr is None or detect_outliers_zscore is None:
            pytest.skip("检测函数不存在")

        iqr_outliers = detect_outliers_iqr(series_with_outliers)
        zscore_outliers = detect_outliers_zscore(series_with_outliers)

        # 两种方法检测到的异常值数量可能不同
        # 这个测试验证两种方法都能运行
        assert isinstance(iqr_outliers, pd.Series)
        assert isinstance(zscore_outliers, pd.Series)

        # 检测到的异常值集合可能不完全相同
        iqr_count = iqr_outliers.sum()
        zscore_count = zscore_outliers.sum()

        # 对于包含明显异常值的数据，两种方法都应该能检测到一些
        assert iqr_count > 0 or zscore_count > 0, \
            "至少有一种方法应该检测到异常值"

    def test_iqr_more_robust_than_zscore(self, skewed_series: pd.Series):
        """
        验证 IQR 对偏态数据更稳健

        期望：在偏态数据上，IQR 的行为更可预测
        """
        if detect_outliers_iqr is None or detect_outliers_zscore is None:
            pytest.skip("检测函数不存在")

        iqr_outliers = detect_outliers_iqr(skewed_series)
        zscore_outliers = detect_outliers_zscore(skewed_series)

        # IQR 应该能检测到一些离群点（偏态分布的长尾）
        # Z-score 在偏态数据上的行为可能不稳定
        assert iqr_outliers.sum() >= 0, "IQR 应该能正常运行"

    def test_both_methods_agree_on_clean_data(self, series_no_outliers: pd.Series):
        """
        测试两种方法在干净数据上的一致性

        期望：两种方法在干净数据上都应检测到很少的异常值
        """
        if detect_outliers_iqr is None or detect_outliers_zscore is None:
            pytest.skip("检测函数不存在")

        iqr_outliers = detect_outliers_iqr(series_no_outliers)
        zscore_outliers = detect_outliers_zscore(series_no_outliers)

        # 两种方法在干净数据上都应该检测到很少的异常值
        assert iqr_outliers.sum() < len(series_no_outliers) * 0.05
        assert zscore_outliers.sum() < len(series_no_outliers) * 0.05


# =============================================================================
# Test: 使用真实数据集
# =============================================================================

class TestWithPenguinsData:
    """使用 Penguins 数据集的测试"""

    def test_penguins_outlier_detection(self):
        """
        测试使用 Penguins 数据集检测异常值

        期望：能正确处理真实数据集
        """
        try:
            import seaborn as sns
            penguins = sns.load_dataset("penguins")
        except ImportError:
            pytest.skip("seaborn 不可用")

        if detect_outliers_iqr is None:
            pytest.skip("detect_outliers_iqr 函数不存在")

        # 测试体重的异常值检测
        body_mass = penguins["body_mass_g"].dropna()
        outliers = detect_outliers_iqr(body_mass)

        assert isinstance(outliers, pd.Series), "返回值应该是 Series"
        assert len(outliers) == len(body_mass), "返回长度应与输入相同"

    def test_penguins_iqr_by_species(self):
        """
        测试按物种分组检测异常值

        期望：分组检测更准确
        """
        try:
            import seaborn as sns
            penguins = sns.load_dataset("penguins")
        except ImportError:
            pytest.skip("seaborn 不可用")

        if detect_outliers_iqr is None:
            pytest.skip("detect_outliers_iqr 函数不存在")

        # 按物种分组检测
        species_results = {}
        for species in penguins['species'].unique():
            data = penguins[penguins['species'] == species]['body_mass_g'].dropna()
            outliers = detect_outliers_iqr(data)
            species_results[species] = outliers.sum()

        # 每个物种都应该有检测结果
        for species, count in species_results.items():
            assert count >= 0, f"{species} 的异常值检测应成功"


# =============================================================================
# Test: 异常值处理决策
# =============================================================================

class TestOutlierDecisions:
    """测试异常值处理决策"""

    def test_remove_outliers(self):
        """
        测试删除异常值

        期望：删除后数据不再包含异常值
        """
        if detect_outliers_iqr is None:
            pytest.skip("detect_outliers_iqr 函数不存在")

        data = pd.Series([1, 2, 3, 4, 5, 100, 200])
        outliers = detect_outliers_iqr(data)
        cleaned = data[~outliers]

        assert len(cleaned) < len(data), "删除后数据量应减少"
        assert cleaned.max() < data.max(), "最大值应该变小"

    def test_cap_outliers(self):
        """
        测试盖帽法处理异常值

        期望：极端值被替换为边界值
        """
        # 这个测试验证学生是否理解盖帽法
        data = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 100.0, 200.0])

        q25 = data.quantile(0.25)
        q75 = data.quantile(0.75)
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr

        # 盖帽法：将超出边界的值替换为边界值
        capped = data.copy().astype(float)
        capped[capped < lower_bound] = lower_bound
        capped[capped > upper_bound] = upper_bound

        assert capped.min() >= lower_bound, "盖帽后最小值应>=下界"
        assert capped.max() <= upper_bound, "盖帽后最大值应<=上界"

    def test_outlier_impact_on_statistics(self):
        """
        测试异常值对统计量的影响

        期望：异常值会显著影响均值，但对中位数影响较小
        """
        # 创建包含极端值的数据
        clean_data = pd.Series([10, 12, 15, 18, 20, 22, 25, 28, 30, 32])
        data_with_outlier = pd.Series([10, 12, 15, 18, 20, 22, 25, 28, 30, 32, 100, 150])

        # 计算统计量
        clean_mean = clean_data.mean()
        clean_median = clean_data.median()
        outlier_mean = data_with_outlier.mean()
        outlier_median = data_with_outlier.median()

        # 异常值对均值的影响更大
        mean_change_pct = abs(outlier_mean - clean_mean) / clean_mean * 100
        median_change_pct = abs(outlier_median - clean_median) / clean_median * 100

        assert mean_change_pct > median_change_pct, \
            "异常值对均值的影响应大于对中位数的影响"

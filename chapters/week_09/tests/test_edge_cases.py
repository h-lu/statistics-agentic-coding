"""
Edge cases and boundary tests for Week 09 solution.py

边界情况与错误处理测试：
- 空数据集
- 单点/两点数据
- 零方差数据
- NaN/Inf 处理
- 维度不匹配
- 完全共线性
- 极端离群点
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Add starter_code to path
starter_code_path = Path(__file__).parent.parent / "starter_code"
sys.path.insert(0, str(starter_code_path))

try:
    import solution
except ImportError:
    solution = None


# =============================================================================
# 1. 空数据与极小样本测试
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestEmptyAndMinimalData:
    """测试空数据和极小样本"""

    def test_empty_data(self, empty_data):
        """
        反例：空数据应报错或返回有意义的结果

        空数组无法拟合回归
        """
        if not hasattr(solution, 'fit_ols'):
            pytest.skip("fit_ols not implemented")

        x = empty_data['x']
        y = empty_data['y']

        # Should raise an error
        with pytest.raises((ValueError, IndexError, RuntimeError)):
            solution.fit_ols(x, y)

    def test_single_point(self, single_point_data):
        """
        边界：单点数据无法拟合回归

        至少需要 2 个点来拟合直线
        """
        if not hasattr(solution, 'fit_ols'):
            pytest.skip("fit_ols not implemented")

        x = single_point_data['x']
        y = single_point_data['y']

        # Should raise an error
        with pytest.raises((ValueError, IndexError, RuntimeError)):
            solution.fit_ols(x, y)

    def test_two_points(self, two_points_data):
        """
        边界：两点数据可拟合完美直线

        两点确定一条直线，R² = 1
        """
        if not hasattr(solution, 'fit_ols'):
            pytest.skip("fit_ols not implemented")

        x = two_points_data['x']
        y = two_points_data['y']

        model = solution.fit_ols(x, y)

        assert model is not None, "Should fit line with 2 points"

        # R² should be 1 (perfect fit)
        if hasattr(model, 'rsquared'):
            assert abs(model.rsquared - 1.0) < 1e-10, "R² should be 1 for 2 points"

    def test_three_points_minimum(self):
        """
        边界：三点是回归的有意义最小样本

        有 1 个自由度用于误差估计
        """
        if not hasattr(solution, 'fit_ols'):
            pytest.skip("fit_ols not implemented")

        x = np.array([1.0, 2.0, 3.0])
        y = np.array([2.0, 4.0, 6.0])

        model = solution.fit_ols(x, y)

        assert model is not None, "Should fit line with 3 points"


# =============================================================================
# 2. 零方差数据测试
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestZeroVarianceData:
    """测试零方差数据"""

    def test_constant_y(self, constant_y_data):
        """
        边界：y 为常量（零方差）

        所有 y 值相同，无法拟合斜率
        """
        if not hasattr(solution, 'fit_ols'):
            pytest.skip("fit_ols not implemented")

        x = constant_y_data['x']
        y = constant_y_data['y']

        # May raise error or produce slope = 0
        try:
            model = solution.fit_ols(x, y)
            # If it succeeds, slope should be ~0
            if hasattr(model, 'params'):
                slope = model.params[1] if len(model.params) > 1 else None
                if slope is not None:
                    assert abs(slope) < 1e-10, f"Slope should be ~0 for constant y, got {slope}"
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            # Raising an error is also acceptable
            assert True

    def test_constant_x(self, constant_x_data):
        """
        反例：x 为常量（零方差）

        所有 x 值相同，无法估计斜率
        """
        if not hasattr(solution, 'fit_ols'):
            pytest.skip("fit_ols not implemented")

        x = constant_x_data['x']
        y = constant_x_data['y']

        # Should raise an error (cannot compute slope with constant x)
        with pytest.raises((ValueError, RuntimeError, np.linalg.LinAlgError)):
            solution.fit_ols(x, y)

    def test_all_values_identical(self):
        """
        边界：x 和 y 都是常量

        完全无变异的数据
        """
        if not hasattr(solution, 'fit_ols'):
            pytest.skip("fit_ols not implemented")

        x = np.full(10, 5.0)
        y = np.full(10, 10.0)

        # Should raise an error
        with pytest.raises((ValueError, RuntimeError, np.linalg.LinAlgError)):
            solution.fit_ols(x, y)


# =============================================================================
# 3. NaN 和 Inf 处理测试
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestNaNAndInfHandling:
    """测试 NaN 和 Inf 处理"""

    def test_nan_in_x(self, nan_data):
        """
        反例：x 中有 NaN

        应报错或自动处理（删除/插补）
        """
        if not hasattr(solution, 'fit_ols'):
            pytest.skip("fit_ols not implemented")

        x = nan_data['x']
        y = nan_data['y']

        # Should either raise error or handle gracefully
        try:
            model = solution.fit_ols(x, y)
            # If it succeeds, check result is valid
            if hasattr(model, 'params'):
                assert np.all(np.isfinite(model.params)), "Coefficients should be finite"
        except (ValueError, RuntimeError):
            # Raising an error is acceptable
            assert True

    def test_nan_in_y(self, nan_data):
        """
        反例：y 中有 NaN

        应报错或自动处理
        """
        if not hasattr(solution, 'fit_ols'):
            pytest.skip("fit_ols not implemented")

        x = nan_data['x']
        y = nan_data['y']

        # Same as above
        try:
            model = solution.fit_ols(x, y)
            if hasattr(model, 'params'):
                assert np.all(np.isfinite(model.params)), "Coefficients should be finite"
        except (ValueError, RuntimeError):
            assert True

    def test_inf_in_x(self, infinite_data):
        """
        反例：x 中有 Inf

        应报错
        """
        if not hasattr(solution, 'fit_ols'):
            pytest.skip("fit_ols not implemented")

        x = infinite_data['x']
        y = infinite_data['y']

        # Should raise an error
        with pytest.raises((ValueError, RuntimeError)):
            solution.fit_ols(x, y)

    def test_inf_in_y(self, infinite_data):
        """
        反例：y 中有 Inf

        应报错
        """
        if not hasattr(solution, 'fit_ols'):
            pytest.skip("fit_ols not implemented")

        x = infinite_data['x']
        y = infinite_data['y']

        # Should raise an error
        with pytest.raises((ValueError, RuntimeError)):
            solution.fit_ols(x, y)


# =============================================================================
# 4. 维度不匹配测试
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestDimensionMismatch:
    """测试维度不匹配"""

    def test_different_length_arrays(self, mismatched_dimensions):
        """
        反例：x 和 y 长度不同

        应报错
        """
        if not hasattr(solution, 'fit_ols'):
            pytest.skip("fit_ols not implemented")

        x = mismatched_dimensions['x']
        y = mismatched_dimensions['y']

        # Should raise an error
        with pytest.raises((ValueError, IndexError, AssertionError)):
            solution.fit_ols(x, y)

    def test_wrong_dimensions_for_multiple_regression(self):
        """
        反例：多元回归时 X 维度错误

        X 应该是 2D 数组或 DataFrame
        """
        if not hasattr(solution, 'multiple_regression'):
            pytest.skip("multiple_regression not implemented")

        np.random.seed(42)
        # Wrong: X is 1D
        x_1d = np.random.normal(0, 1, 100)
        y = np.random.normal(0, 1, 100)

        # Should raise an error or handle it
        try:
            result = solution.multiple_regression(x_1d, y)
            # If it succeeds, it might treat it as single predictor
            assert result is not None
        except (ValueError, IndexError):
            # Raising an error is also acceptable
            assert True

    def test_predict_with_wrong_dimensions(self):
        """
        反例：预测时维度不匹配

        新数据的特征维度应与训练数据一致
        """
        if not hasattr(solution, 'fit_ols') or not hasattr(solution, 'predict'):
            pytest.skip("fit_ols or predict not implemented")

        np.random.seed(42)
        x_train = np.random.normal(50, 15, (100, 2))  # 2 features
        y_train = 10 + 0.5 * x_train[:, 0] + 0.3 * x_train[:, 1] + np.random.normal(0, 5, 100)

        # Assuming fit_ols can handle 2D X
        try:
            model = solution.fit_ols(x_train, y_train)
            x_new = np.random.normal(50, 15, (10, 3))  # 3 features (wrong!)

            # Should raise an error
            with pytest.raises((ValueError, IndexError)):
                solution.predict(model, x_new)
        except Exception:
            # Implementation may vary
            pass


# =============================================================================
# 5. 极端离群点测试
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestExtremeOutliers:
    """测试极端离群点"""

    def test_single_extreme_outlier(self, diagnostics_with_outlier):
        """
        边界：单个极端离群点

        应能识别离群点但不应崩溃
        """
        if not hasattr(solution, 'fit_ols'):
            pytest.skip("fit_ols not implemented")

        x = diagnostics_with_outlier['x']
        y = diagnostics_with_outlier['y']

        # Should still fit without crashing
        model = solution.fit_ols(x, y)

        assert model is not None, "Should fit even with outlier"
        if hasattr(model, 'params'):
            assert np.all(np.isfinite(model.params)), "Coefficients should be finite"

    def test_multiple_extreme_outliers(self, diagnostics_multiple_outliers):
        """
        边界：多个极端离群点

        模型应仍然可拟合
        """
        if not hasattr(solution, 'fit_ols'):
            pytest.skip("fit_ols not implemented")

        x = diagnostics_multiple_outliers['x']
        y = diagnostics_multiple_outliers['y']

        model = solution.fit_ols(x, y)

        assert model is not None, "Should fit even with multiple outliers"
        if hasattr(model, 'params'):
            assert np.all(np.isfinite(model.params)), "Coefficients should be finite"

    def test_outlier_far_from_data(self):
        """
        边界：离群点远离数据中心

        Cook's 距离应非常高
        """
        if not hasattr(solution, 'calculate_cooks_distance') or not hasattr(solution, 'simple_linear_regression'):
            pytest.skip("calculate_cooks_distance or simple_linear_regression not implemented")

        np.random.seed(42)
        x = np.random.normal(50, 10, 95)
        y = 10 + 0.5 * x + np.random.normal(0, 5, 95)

        # Add extreme outlier
        x_with_outlier = np.concatenate([x, [1000]])
        y_with_outlier = np.concatenate([y, [5000]])

        # Fit regression first
        reg_result = solution.simple_linear_regression(x_with_outlier, y_with_outlier)
        cooks_result = solution.calculate_cooks_distance(reg_result)
        cooks_d = cooks_result['cooks_d']

        # The last point should have very high Cook's distance
        max_cooks = np.max(cooks_d)
        assert cooks_d[-1] > 0.1, "Outlier should have high Cook's distance"
        assert cooks_d[-1] == max_cooks, "Outlier should have the highest Cook's distance"


# =============================================================================
# 6. 完全共线性测试
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestPerfectCollinearity:
    """测试完全共线性"""

    def test_perfect_collinearity_two_variables(self):
        """
        反例：两个变量完全相关

        x2 = 2 * x1
        """
        if not hasattr(solution, 'multiple_regression'):
            pytest.skip("multiple_regression not implemented")

        np.random.seed(42)
        n = 100
        x1 = np.random.normal(0, 1, n)
        x2 = 2 * x1  # Perfectly correlated
        X = np.column_stack([x1, x2])
        y = 10 + 2 * x1 + 3 * x2 + np.random.normal(0, 1, n)

        # Should raise an error (singular matrix)
        with pytest.raises((ValueError, RuntimeError, np.linalg.LinAlgError)):
            solution.multiple_regression(X, y)

    def test_perfect_collinearity_linear_combination(self, multicollinearity_perfect):
        """
        反例：一个变量是其他变量的线性组合

        x3 = 2*x1 + 3*x2
        """
        if not hasattr(solution, 'multiple_regression'):
            pytest.skip("multiple_regression not implemented")

        X = multicollinearity_perfect['X']
        y = multicollinearity_perfect['y']

        # Should raise an error
        with pytest.raises((ValueError, RuntimeError, np.linalg.LinAlgError)):
            solution.multiple_regression(X, y)

    def test_near_perfect_collinearity(self):
        """
        边界：近乎完全共线性

        VIF 应非常高
        """
        if not hasattr(solution, 'calculate_vif'):
            pytest.skip("calculate_vif not implemented")

        np.random.seed(42)
        n = 200
        x1 = np.random.normal(0, 1, n)
        x2 = 0.999 * x1 + np.random.normal(0, 0.001, n)  # Very highly correlated
        X = np.column_stack([x1, x2])

        vif_df = solution.calculate_vif(X)
        vif_values = vif_df['vif'].values if hasattr(vif_df, 'vif') else vif_df

        # VIFs should be extremely high
        assert max(vif_values) > 100, "VIF should be > 100 for near-perfect collinearity"


# =============================================================================
# 7. 数值范围测试
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestNumericalRanges:
    """测试极端数值范围"""

    def test_very_large_values(self):
        """
        边界：非常大的数值

        应保持数值稳定性
        """
        if not hasattr(solution, 'fit_ols'):
            pytest.skip("fit_ols not implemented")

        np.random.seed(42)
        x = np.random.uniform(1e10, 1e11, 100)
        y = 1e9 + 0.5 * x + np.random.normal(0, 1e9, 100)

        model = solution.fit_ols(x, y)

        assert model is not None
        if hasattr(model, 'params'):
            assert np.all(np.isfinite(model.params)), "Should handle very large values"

    def test_very_small_values(self):
        """
        边界：非常小的数值

        应保持数值稳定性
        """
        if not hasattr(solution, 'fit_ols'):
            pytest.skip("fit_ols not implemented")

        np.random.seed(42)
        x = np.random.uniform(1e-10, 1e-9, 100)
        y = 1e-10 + 0.5 * x + np.random.normal(0, 1e-10, 100)

        model = solution.fit_ols(x, y)

        assert model is not None
        if hasattr(model, 'params'):
            assert np.all(np.isfinite(model.params)), "Should handle very small values"

    def test_mixed_scales(self):
        """
        边界：混合量级的数据

        x1 量级为 1，x2 量级为 1e6
        """
        if not hasattr(solution, 'multiple_regression'):
            pytest.skip("multiple_regression not implemented")

        np.random.seed(42)
        n = 200
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1e6, n)
        X = np.column_stack([x1, x2])
        y = 10 + 2 * x1 + 3 * x2 + np.random.normal(0, 1000, n)

        result = solution.multiple_regression(X, y)

        assert result is not None
        # Coefficients should reflect the different scales
        # x2 coefficient should be ~3, x1 coefficient ~2

    def test_negative_values(self):
        """
        边界：全为负值的数据

        应正常处理
        """
        if not hasattr(solution, 'fit_ols'):
            pytest.skip("fit_ols not implemented")

        np.random.seed(42)
        x = np.random.uniform(-100, -10, 100)
        y = -10 + 0.5 * x + np.random.normal(0, 5, 100)

        model = solution.fit_ols(x, y)

        assert model is not None
        if hasattr(model, 'params'):
            assert np.all(np.isfinite(model.params)), "Should handle negative values"


# =============================================================================
# 8. 预测边界测试
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestPredictionBoundaries:
    """测试预测功能的边界情况"""

    def test_predict_single_value(self):
        """
        边界：预测单个值

        """
        if not hasattr(solution, 'fit_ols') or not hasattr(solution, 'predict'):
            pytest.skip("fit_ols or predict not implemented")

        np.random.seed(42)
        x = np.random.normal(50, 15, 100)
        y = 10 + 0.5 * x + np.random.normal(0, 5, 100)

        model = solution.fit_ols(x, y)
        prediction = solution.predict(model, np.array([50]))

        assert prediction is not None
        # Should predict around 35 (10 + 0.5*50)
        assert 25 <= prediction[0] <= 45, "Prediction should be reasonable"

    def test_predict_out_of_range(self):
        """
        边界：预测超出训练数据范围的值

        预测 x = 1000，但训练数据 x 在 [10, 90]
        """
        if not hasattr(solution, 'fit_ols') or not hasattr(solution, 'predict'):
            pytest.skip("fit_ols or predict not implemented")

        np.random.seed(42)
        x = np.random.uniform(10, 90, 100)
        y = 10 + 0.5 * x + np.random.normal(0, 5, 100)

        model = solution.fit_ols(x, y)
        prediction = solution.predict(model, np.array([1000]))

        assert prediction is not None
        # Should still give a prediction (though may be unreliable)

    def test_predict_negative_input(self):
        """
        边界：预测负值输入

        即使训练数据没有负值
        """
        if not hasattr(solution, 'fit_ols') or not hasattr(solution, 'predict'):
            pytest.skip("fit_ols or predict not implemented")

        np.random.seed(42)
        x = np.random.uniform(10, 90, 100)
        y = 10 + 0.5 * x + np.random.normal(0, 5, 100)

        model = solution.fit_ols(x, y)
        prediction = solution.predict(model, np.array([-50]))

        assert prediction is not None
        # May give negative prediction, which might not make sense in context


# =============================================================================
# 9. 诊断工具边界测试
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestDiagnosticsBoundaries:
    """测试诊断工具的边界情况"""

    def test_cooks_distance_with_perfect_fit(self):
        """
        边界：完美拟合时的 Cook's 距离

        对于完美拟合，残差为 0，但边界点由于杠杆效应仍有 Cook's 距离
        """
        if not hasattr(solution, 'calculate_cooks_distance') or not hasattr(solution, 'simple_linear_regression'):
            pytest.skip("calculate_cooks_distance or simple_linear_regression not implemented")

        # Perfect linear relationship with more points
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        y = 5 + 2 * x  # Perfect fit

        # Fit regression first
        reg_result = solution.simple_linear_regression(x, y)
        cooks_result = solution.calculate_cooks_distance(reg_result)
        cooks_d = cooks_result['cooks_d']

        # For perfect fit, residuals are 0 but leverage still contributes to Cook's D
        # Center points should have very small Cook's D (low leverage)
        # Edge points may have higher Cook's D due to leverage
        assert cooks_d is not None, "Should return Cook's distances"
        assert len(cooks_d) == len(x), "Should have one Cook's distance per observation"
        # Center points (indices 4-6) should have smaller Cook's D than edge points
        center_cooks = cooks_d[4:7]
        assert np.all(center_cooks < np.max(cooks_d)), "Center points should have lower Cook's D than max"

    def test_vif_with_single_predictor(self):
        """
        边界：单个预测变量的 VIF

        单个变量不存在多重共线性问题
        """
        if not hasattr(solution, 'calculate_vif'):
            pytest.skip("calculate_vif not implemented")

        np.random.seed(42)
        x = np.random.normal(0, 1, (100, 1))

        # VIF for single predictor should be 1
        vif_df = solution.calculate_vif(x)
        vif_values = vif_df['vif'].values if hasattr(vif_df, 'vif') else vif_df

        assert len(vif_values) == 1, "Should have 1 VIF value"
        assert abs(vif_values[0] - 1.0) < 0.1, "VIF should be 1 for single predictor"

    def test_vif_with_two_identical_variables(self):
        """
        反例：两个完全相同的变量

        VIF 应为无穷大或报错
        """
        if not hasattr(solution, 'calculate_vif'):
            pytest.skip("calculate_vif not implemented")

        np.random.seed(42)
        x = np.random.normal(0, 1, 100)
        X = np.column_stack([x, x])  # Identical columns

        # Should either give very high VIF or raise error
        try:
            vif_df = solution.calculate_vif(X)
            vif_values = vif_df['vif'].values if hasattr(vif_df, 'vif') else vif_df
            assert any(vif > 1e6 for vif in vif_values), "VIF should be extremely high"
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            # Raising an error is also acceptable
            assert True


# =============================================================================
# 10. 数据类型测试
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestDataTypes:
    """测试不同数据类型"""

    def test_integer_input(self):
        """
        边界：整数输入

        应能正确处理整数
        """
        if not hasattr(solution, 'fit_ols'):
            pytest.skip("fit_ols not implemented")

        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        y = np.array([3, 5, 7, 9, 11, 13, 15, 17, 19, 21])

        model = solution.fit_ols(x, y)

        assert model is not None
        # Should handle integer input correctly

    def test_list_input(self):
        """
        边界：列表输入（而非 NumPy 数组）

        应能转换或报错
        """
        if not hasattr(solution, 'fit_ols'):
            pytest.skip("fit_ols not implemented")

        x = [1, 2, 3, 4, 5]
        y = [3, 5, 7, 9, 11]

        # Should either convert or raise error
        try:
            model = solution.fit_ols(x, y)
            assert model is not None
        except TypeError:
            # Raising TypeError for list input is acceptable
            assert True

    def test_pandas_series_input(self):
        """
        边界：Pandas Series 输入

        应能正确处理
        """
        if not hasattr(solution, 'fit_ols'):
            pytest.skip("fit_ols not implemented")

        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not available")

        x = pd.Series([1, 2, 3, 4, 5])
        y = pd.Series([3, 5, 7, 9, 11])

        model = solution.fit_ols(x, y)

        assert model is not None
        # Should handle pandas Series correctly

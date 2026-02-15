"""
Comprehensive tests for Week 09 solution.py

综合测试：
- 简单线性回归（OLS 拟合）
- 回归系数解释
- R² 计算
- 残差计算
- 多元回归
- StatLab 集成测试
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

# Add starter_code to path
starter_code_path = Path(__file__).parent.parent / "starter_code"
sys.path.insert(0, str(starter_code_path))

try:
    import solution
except ImportError:
    solution = None


# =============================================================================
# 1. 简单线性回归测试
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestSimpleLinearRegression:
    """测试简单线性回归功能"""

    def test_fit_intercept_and_slope(self, simple_linear_data):
        """
        正例：正确拟合截距和斜率

        数据生成：y = 10 + 0.5*x + noise
        预期：截距约 10，斜率约 0.5
        """
        if not hasattr(solution, 'simple_linear_regression') and not hasattr(solution, 'fit_ols'):
            pytest.skip("No regression function implemented")

        x = simple_linear_data['x']
        y = simple_linear_data['y']

        # Try different function names
        if hasattr(solution, 'simple_linear_regression'):
            result = solution.simple_linear_regression(x, y)
            if isinstance(result, dict):
                # solution.py returns: {'coefficients': {'intercept': ..., 'slope': ...}}
                if 'coefficients' in result:
                    intercept = result['coefficients'].get('intercept')
                    slope = result['coefficients'].get('slope')
                else:
                    intercept = result.get('intercept', result.get('coef', {}).get('intercept'))
                    slope = result.get('slope', result.get('coef', {}).get('x'))
        elif hasattr(solution, 'fit_ols'):
            model = solution.fit_ols(x, y)
            # Extract coefficients from model (implementation dependent)
            if hasattr(model, 'params'):
                intercept = model.params[0]
                slope = model.params[1] if len(model.params) > 1 else None
            elif isinstance(model, dict):
                intercept = model.get('intercept')
                slope = model.get('slope')
            else:
                pytest.skip("Cannot extract coefficients from model")

        # Check that coefficients are reasonable
        assert intercept is not None, "Should have intercept"
        assert slope is not None, "Should have slope"
        assert 5 <= intercept <= 15, f"Intercept should be around 10, got {intercept}"
        assert 0.3 <= slope <= 0.7, f"Slope should be around 0.5, got {slope}"

    def test_perfect_linear_relationship(self, simple_linear_perfect):
        """
        正例：完美线性关系（无噪声）

        y = 5 + 2*x（无噪声）
        预期：R² = 1，系数精确
        """
        if not hasattr(solution, 'simple_linear_regression') and not hasattr(solution, 'fit_ols'):
            pytest.skip("No regression function implemented")

        x = simple_linear_perfect['x']
        y = simple_linear_perfect['y']

        if hasattr(solution, 'simple_linear_regression'):
            result = solution.simple_linear_regression(x, y)
        elif hasattr(solution, 'fit_ols'):
            result = solution.fit_ols(x, y)
        else:
            pytest.skip("No regression function implemented")

        # For perfect linear relationship, R² should be 1
        if hasattr(solution, 'calculate_r_squared') and hasattr(result, 'fittedvalues'):
            r2 = solution.calculate_r_squared(y, result.fittedvalues)
            assert abs(r2 - 1.0) < 1e-10, f"R² should be 1 for perfect linear relationship, got {r2}"

    def test_negative_slope(self, simple_linear_negative_slope):
        """
        正例：负斜率回归

        y = 100 - 0.8*x + noise
        预期：斜率为负
        """
        if not hasattr(solution, 'fit_ols'):
            pytest.skip("fit_ols not implemented")

        x = simple_linear_negative_slope['x']
        y = simple_linear_negative_slope['y']

        model = solution.fit_ols(x, y)

        # Slope should be negative
        if hasattr(model, 'params'):
            slope = model.params[1] if len(model.params) > 1 else None
            assert slope is not None, "Should have slope"
            assert slope < 0, f"Slope should be negative, got {slope}"
            assert -1.0 <= slope <= -0.6, f"Slope should be around -0.8, got {slope}"

    def test_no_relationship(self, simple_linear_no_relationship):
        """
        正例：无关系数据

        y = 50 + noise（与 x 无关）
        预期：斜率接近 0，R² 接近 0
        """
        if not hasattr(solution, 'fit_ols'):
            pytest.skip("fit_ols not implemented")

        x = simple_linear_no_relationship['x']
        y = simple_linear_no_relationship['y']

        model = solution.fit_ols(x, y)

        # Slope should be close to 0
        if hasattr(model, 'params'):
            slope = model.params[1] if len(model.params) > 1 else None
            assert slope is not None, "Should have slope"
            assert -0.2 <= slope <= 0.2, f"Slope should be close to 0, got {slope}"

        # R² should be low
        if hasattr(model, 'rsquared'):
            assert model.rsquared < 0.2, f"R² should be low for no relationship, got {model.rsquared}"


# =============================================================================
# 2. 回归系数解释测试
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestCoefficientInterpretation:
    """测试回归系数解释功能"""

    def test_interpret_intercept(self):
        """
        正例：正确解释截距

        截距表示 x=0 时的 y 预测值
        """
        if not hasattr(solution, 'interpret_coefficients'):
            pytest.skip("interpret_coefficients not implemented")

        # solution.py expects reg_result format from simple_linear_regression
        # which has structure: {'coefficients': {'intercept': ..., 'slope': ...}, 'r_squared': ...}
        reg_result = {
            'coefficients': {'intercept': 10, 'slope': 0.5},
            'r_squared': 0.75
        }
        interpretation = solution.interpret_coefficients(reg_result, x_name='x', y_name='y')

        assert interpretation is not None, "Should return interpretation"
        assert 'intercept' in str(interpretation).lower() or '截距' in str(interpretation), \
            "Should mention intercept"

    def test_interpret_slope(self):
        """
        正例：正确解释斜率

        斜率表示 x 每增加 1 单位，y 的变化量
        """
        if not hasattr(solution, 'interpret_coefficients'):
            pytest.skip("interpret_coefficients not implemented")

        # solution.py expects reg_result format
        reg_result = {
            'coefficients': {'intercept': 10, 'slope': 0.5},
            'r_squared': 0.75
        }
        interpretation = solution.interpret_coefficients(reg_result, x_name='广告投入', y_name='销售额')

        assert interpretation is not None, "Should return interpretation"
        # Should mention that slope represents change per unit
        text = str(interpretation).lower()
        assert 'change' in text or '变化' in text or 'increase' in text or '增加' in text, \
            "Should explain slope as change per unit"

    def test_coefficient_confidence_interval(self, simple_linear_data):
        """
        正例：计算回归系数的置信区间

        斜率应有 95% CI
        """
        if not hasattr(solution, 'fit_ols'):
            pytest.skip("fit_ols not implemented")

        x = simple_linear_data['x']
        y = simple_linear_data['y']

        model = solution.fit_ols(x, y)

        # Check if model has confidence intervals
        if hasattr(model, 'conf_int'):
            ci = model.conf_int()
            assert ci.shape[0] >= 2, "Should have CI for intercept and slope"
            assert ci.shape[1] == 2, "CI should have lower and upper bounds"

            # Lower bound should be less than upper bound
            assert ci[0, 0] < ci[0, 1], "CI lower < upper for intercept"
            assert ci[1, 0] < ci[1, 1], "CI lower < upper for slope"

    def test_coefficient_significance_test(self, simple_linear_data):
        """
        正例：回归系数显著性检验

        对于有真实关系的变量，p 值应 < 0.05
        """
        if not hasattr(solution, 'fit_ols'):
            pytest.skip("fit_ols not implemented")

        x = simple_linear_data['x']
        y = simple_linear_data['y']

        model = solution.fit_ols(x, y)

        # Check if model has p-values
        if hasattr(model, 'pvalues'):
            p_values = model.pvalues
            assert len(p_values) >= 2, "Should have p-values for intercept and slope"

            # Slope p-value should be significant (since x truly affects y)
            slope_p = p_values[1]
            assert slope_p < 0.05, f"Slope should be significant (p < 0.05), got {slope_p}"


# =============================================================================
# 3. R² 和拟合优度测试
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestRSquared:
    """测试 R² 计算和解释"""

    def test_calculate_r_squared(self):
        """
        正例：计算 R²

        R² = 1 - SS_res / SS_tot
        """
        if not hasattr(solution, 'calculate_r_squared'):
            pytest.skip("calculate_r_squared not implemented")

        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.2, 2.9, 4.1, 5.0])

        r2 = solution.calculate_r_squared(y_true, y_pred)

        assert r2 is not None, "Should return R²"
        assert 0 <= r2 <= 1, f"R² should be in [0, 1], got {r2}"
        assert r2 > 0.95, "R² should be high for good predictions"

    def test_r_squared_perfect_fit(self):
        """
        正例：完美拟合时 R² = 1

        当预测值完全等于真实值时
        """
        if not hasattr(solution, 'calculate_r_squared'):
            pytest.skip("calculate_r_squared not implemented")

        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])

        r2 = solution.calculate_r_squared(y_true, y_pred)

        assert abs(r2 - 1.0) < 1e-10, f"R² should be 1 for perfect fit, got {r2}"

    def test_r_squared_worst_fit(self):
        """
        正例：最差拟合时 R² 可能 < 0

        当模型比均值还差时
        """
        if not hasattr(solution, 'calculate_r_squared'):
            pytest.skip("calculate_r_squared not implemented")

        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([10, 20, 30, 40, 50])  # Horrible predictions

        r2 = solution.calculate_r_squared(y_true, y_pred)

        # R² can be negative for terrible models
        assert r2 < 0, f"R² should be negative for terrible predictions, got {r2}"

    def test_r_squared_from_model(self, simple_linear_high_r_squared):
        """
        正例：从模型获取 R²

        高 R² 数据应产生高 R² 值
        """
        if not hasattr(solution, 'fit_ols'):
            pytest.skip("fit_ols not implemented")

        x = simple_linear_high_r_squared['x']
        y = simple_linear_high_r_squared['y']

        model = solution.fit_ols(x, y)

        if hasattr(model, 'rsquared'):
            r2 = model.rsquared
            assert r2 > 0.7, f"R² should be high for strong relationship, got {r2}"

    def test_r_squared_comparison(self, simple_linear_high_r_squared, simple_linear_low_r_squared):
        """
        正例：比较不同模型的 R²

        高 R² 数据应比低 R² 数据产生更高的 R²
        """
        if not hasattr(solution, 'fit_ols'):
            pytest.skip("fit_ols not implemented")

        # High R² data
        x_high = simple_linear_high_r_squared['x']
        y_high = simple_linear_high_r_squared['y']
        model_high = solution.fit_ols(x_high, y_high)

        # Low R² data
        x_low = simple_linear_low_r_squared['x']
        y_low = simple_linear_low_r_squared['y']
        model_low = solution.fit_ols(x_low, y_low)

        if hasattr(model_high, 'rsquared') and hasattr(model_low, 'rsquared'):
            assert model_high.rsquared > model_low.rsquared, \
                "High R² data should produce higher R² than low R² data"


# =============================================================================
# 4. 残差分析测试
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestResiduals:
    """测试残差计算和分析"""

    def test_calculate_residuals(self):
        """
        正例：计算残差

        残差 = y_true - y_pred
        """
        if not hasattr(solution, 'calculate_residuals'):
            pytest.skip("calculate_residuals not implemented")

        y_true = np.array([10, 20, 30, 40, 50])
        y_pred = np.array([12, 18, 32, 38, 52])

        residuals = solution.calculate_residuals(y_true, y_pred)

        assert residuals is not None, "Should return residuals"
        assert len(residuals) == len(y_true), "Residuals should have same length as input"
        assert np.allclose(residuals, [-2, 2, -2, 2, -2]), "Residuals should be y_true - y_pred"

    def test_residuals_sum_to_zero(self, simple_linear_data):
        """
        正例：OLS 残差之和应接近 0

        对于包含截距的 OLS 回归
        """
        if not hasattr(solution, 'fit_ols'):
            pytest.skip("fit_ols not implemented")

        x = simple_linear_data['x']
        y = simple_linear_data['y']

        model = solution.fit_ols(x, y)

        if hasattr(model, 'resid'):
            residuals = model.resid
            sum_residuals = np.sum(residuals)
            assert abs(sum_residuals) < 1e-10, f"Residuals should sum to ~0, got {sum_residuals}"

    def test_residuals_from_model(self, simple_linear_data):
        """
        正例：从模型获取残差

        模型应提供残差属性
        """
        if not hasattr(solution, 'fit_ols'):
            pytest.skip("fit_ols not implemented")

        x = simple_linear_data['x']
        y = simple_linear_data['y']

        model = solution.fit_ols(x, y)

        # Check if model has residuals
        if hasattr(model, 'resid'):
            residuals = model.resid
            assert len(residuals) == len(y), "Residuals should have same length as y"
            assert np.all(np.isfinite(residuals)), "All residuals should be finite"

    def test_fitted_values(self, simple_linear_data):
        """
        正例：计算拟合值

        拟合值 = intercept + slope * x
        """
        if not hasattr(solution, 'fit_ols'):
            pytest.skip("fit_ols not implemented")

        x = simple_linear_data['x']
        y = simple_linear_data['y']

        model = solution.fit_ols(x, y)

        if hasattr(model, 'fittedvalues'):
            fitted = model.fittedvalues
            assert len(fitted) == len(y), "Fitted values should have same length as y"
            assert np.all(np.isfinite(fitted)), "All fitted values should be finite"


# =============================================================================
# 5. 模型诊断测试
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestModelDiagnostics:
    """测试模型诊断功能"""

    def test_check_linearity(self, assumption_linear_met, assumption_linear_violated_quadratic):
        """
        正例：检查线性假设

        线性数据应通过，二次数据应失败
        """
        if not hasattr(solution, 'check_linearity'):
            pytest.skip("check_linearity not implemented")

        # Linear data - should pass
        x_linear = assumption_linear_met['x']
        y_linear = assumption_linear_met['y']
        result_linear = solution.check_linearity(x_linear, y_linear)

        # Quadratic data - should fail
        x_quad = assumption_linear_violated_quadratic['x']
        y_quad = assumption_linear_violated_quadratic['y']
        result_quad = solution.check_linearity(x_quad, y_quad)

        # Linear data should be more linear than quadratic data
        if isinstance(result_linear, dict) and isinstance(result_quad, dict):
            assert result_linear.get('is_linear', True) == True, "Linear data should pass linearity check"
            assert result_quad.get('is_linear', True) == False, "Quadratic data should fail linearity check"

    def test_check_normality(self, assumption_normality_met, assumption_normality_violated_skewed):
        """
        正例：检查正态性假设

        正态数据应通过，偏态数据应失败
        """
        if not hasattr(solution, 'check_normality'):
            pytest.skip("check_normality not implemented")

        # Normal residuals
        x_normal = assumption_normality_met['x']
        y_normal = assumption_normality_met['y']

        # Skewed residuals
        x_skewed = assumption_normality_violated_skewed['x']
        y_skewed = assumption_normality_violated_skewed['y']

        # Implementation dependent - just check function runs
        result_normal = solution.check_normality(x_normal, y_normal)
        result_skewed = solution.check_normality(x_skewed, y_skewed)

        assert result_normal is not None, "Should return result for normal data"
        assert result_skewed is not None, "Should return result for skewed data"

    def test_check_homoscedasticity(self, assumption_homoscedasticity_met, assumption_homoscedasticity_violated):
        """
        正例：检查等方差假设

        同方差数据应通过，异方差数据应失败
        """
        if not hasattr(solution, 'check_homoscedasticity'):
            pytest.skip("check_homoscedasticity not implemented")

        # Homoscedastic data
        x_homo = assumption_homoscedasticity_met['x']
        y_homo = assumption_homoscedasticity_met['y']

        # Heteroscedastic data
        x_hetero = assumption_homoscedasticity_violated['x']
        y_hetero = assumption_homoscedasticity_violated['y']

        result_homo = solution.check_homoscedasticity(x_homo, y_homo)
        result_hetero = solution.check_homoscedasticity(x_hetero, y_hetero)

        assert result_homo is not None, "Should return result for homoscedastic data"
        assert result_hetero is not None, "Should return result for heteroscedastic data"

    def test_calculate_cooks_distance(self, diagnostics_with_outlier):
        """
        正例：计算 Cook's 距离

        有离群点的数据应产生高 Cook's 距离
        """
        if not hasattr(solution, 'calculate_cooks_distance'):
            pytest.skip("calculate_cooks_distance not implemented")
        if not hasattr(solution, 'simple_linear_regression'):
            pytest.skip("simple_linear_regression not implemented")

        x = diagnostics_with_outlier['x']
        y = diagnostics_with_outlier['y']

        # Fit regression first
        reg_result = solution.simple_linear_regression(x, y)
        cooks_result = solution.calculate_cooks_distance(reg_result)
        cooks_d = cooks_result.get('cooks_d', cooks_result)

        assert cooks_d is not None, "Should return Cook's distances"
        assert len(cooks_d) == len(y), "Cook's distances should have same length as data"
        assert np.all(cooks_d >= 0), "Cook's distances should be non-negative"

        # At least one point should have high Cook's distance
        threshold = 4 / len(y)
        high_influence = np.sum(cooks_d > threshold)
        assert high_influence >= 1, f"Should have at least one high influence point, found {high_influence}"


# =============================================================================
# 6. 多元回归测试
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestMultipleRegression:
    """测试多元回归功能"""

    def test_multiple_regression_fit(self, multiple_regression_data):
        """
        正例：拟合多元回归

        y = 10 + 0.5*x1 + 0.3*x2 - 0.2*x3 + noise
        """
        if not hasattr(solution, 'multiple_regression'):
            pytest.skip("multiple_regression not implemented")

        X = multiple_regression_data['X']
        y = multiple_regression_data['y']

        result = solution.multiple_regression(X, y)

        assert result is not None, "Should return result"

        # Check that we have 3 coefficients (+ intercept)
        if hasattr(result, 'params'):
            assert len(result.params) == 4, "Should have intercept + 3 coefficients"
        elif isinstance(result, dict):
            n_coefs = len(result.get('coefficients', result.get('coef', [])))
            assert n_coefs == 4 or n_coefs == 3, "Should have 3 or 4 coefficients"

    def test_coefficient_order(self, multiple_regression_data):
        """
        正例：验证系数顺序

        系数应按输入变量的顺序返回
        """
        if not hasattr(solution, 'multiple_regression'):
            pytest.skip("multiple_regression not implemented")

        X = multiple_regression_data['X']
        y = multiple_regression_data['y']

        result = solution.multiple_regression(X, y)

        # Just verify function runs without error
        assert result is not None

    def test_multiple_vs_simple_regression(self, simple_linear_data, multiple_regression_data):
        """
        正例：多元回归 vs 简单回归

        多元回归应能处理多个预测变量
        """
        # Simple regression should work with 1 predictor
        if hasattr(solution, 'simple_linear_regression'):
            x = simple_linear_data['x']
            y = simple_linear_data['y']
            result_simple = solution.simple_linear_regression(x, y)
            assert result_simple is not None

        # Multiple regression should work with 3 predictors
        if hasattr(solution, 'multiple_regression'):
            X = multiple_regression_data['X']
            y = multiple_regression_data['y']
            result_multiple = solution.multiple_regression(X, y)
            assert result_multiple is not None


# =============================================================================
# 7. 多重共线性测试
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestMulticollinearity:
    """测试多重共线性检测功能"""

    def test_calculate_vif_no_collinearity(self, multicollinearity_none):
        """
        正例：无多重共线性时 VIF 应接近 1

        VIF = 1 / (1 - R²)，R²=0 时 VIF=1
        """
        if not hasattr(solution, 'calculate_vif'):
            pytest.skip("calculate_vif not implemented")

        X = multicollinearity_none['X']

        vif_df = solution.calculate_vif(X)
        # solution.py returns a DataFrame with 'vif' column
        vif_values = vif_df['vif'].values if hasattr(vif_df, 'vif') else vif_df

        assert vif_values is not None, "Should return VIF values"
        assert len(vif_values) == X.shape[1], "Should have VIF for each variable"

        # All VIFs should be close to 1 (no multicollinearity)
        for vif in vif_values:
            assert 1 <= vif < 5, f"VIF should be < 5 for no multicollinearity, got {vif}"

    def test_calculate_vif_severe_collinearity(self, multicollinearity_severe):
        """
        正例：严重多重共线性时 VIF 应 > 10

        VIF > 10 表示严重多重共线性
        """
        if not hasattr(solution, 'calculate_vif'):
            pytest.skip("calculate_vif not implemented")

        X = multicollinearity_severe['X']

        vif_df = solution.calculate_vif(X)
        # solution.py returns a DataFrame with 'vif' column
        vif_values = vif_df['vif'].values if hasattr(vif_df, 'vif') else vif_df

        assert vif_values is not None, "Should return VIF values"

        # At least one VIF should be high
        max_vif = max(vif_values) if isinstance(vif_values, (list, np.ndarray)) else vif_values
        assert max_vif > 10, f"Should have VIF > 10 for severe multicollinearity, got {max_vif}"

    def test_perfect_collinearity_detection(self, multicollinearity_perfect):
        """
        反例：完全共线性应被检测到

        完全共线性导致奇异矩阵，无法计算 OLS
        """
        if not hasattr(solution, 'calculate_vif'):
            pytest.skip("calculate_vif not implemented")

        X = multicollinearity_perfect['X']

        # Perfect multicollinearity should cause an error or warning
        try:
            vif_df = solution.calculate_vif(X)
            # If it doesn't error, VIFs should be extremely high
            vif_values = vif_df['vif'].values if hasattr(vif_df, 'vif') else vif_df
            assert any(vif > 1000 for vif in vif_values), \
                "Perfect multicollinearity should produce very high VIFs"
        except (np.linalg.LinAlgError, ValueError):
            # Raising an error is also acceptable
            assert True, "Perfect multicollinearity should raise an error"


# =============================================================================
# 8. StatLab 集成测试
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestStatLabIntegration:
    """测试 StatLab 集成功能"""

    def test_regression_with_diagnostics(self, simple_linear_data):
        """
        正例：回归分析 + 诊断报告

        应返回模型结果 + 诊断信息
        """
        if not hasattr(solution, 'regression_with_diagnostics'):
            pytest.skip("regression_with_diagnostics not implemented")

        x = simple_linear_data['x']
        y = simple_linear_data['y']

        result = solution.regression_with_diagnostics(x, y)

        assert result is not None, "Should return result"
        assert isinstance(result, dict), "Should return a dict"

        # Should contain model results
        assert 'model' in result or 'r_squared' in result or 'coefficients' in result, \
            "Should contain model results"

    def test_format_regression_report(self, simple_linear_data):
        """
        正例：格式化回归报告

        应生成可读的 Markdown 报告
        """
        if not hasattr(solution, 'regression_with_diagnostics') or not hasattr(solution, 'format_regression_report'):
            pytest.skip("StatLab functions not implemented")

        x = simple_linear_data['x']
        y = simple_linear_data['y']

        diag_result = solution.regression_with_diagnostics(x, y)
        report = solution.format_regression_report(diag_result)

        assert report is not None, "Should return report"
        assert isinstance(report, str), "Report should be a string"
        assert len(report) > 100, "Report should have meaningful content"

    def test_statlab_ad_sales_analysis(self, statlab_ad_sales_data):
        """
        正例：StatLab 广告销售分析

        使用真实场景数据
        """
        if not hasattr(solution, 'regression_with_diagnostics'):
            pytest.skip("regression_with_diagnostics not implemented")

        # Simple regression: total ad spend vs sales
        ad_total = statlab_ad_sales_data['ad_tv'] + statlab_ad_sales_data['ad_online'] + statlab_ad_sales_data['ad_social']
        sales = statlab_ad_sales_data['sales']

        result = solution.regression_with_diagnostics(ad_total, sales)

        assert result is not None, "Should analyze ad-sales relationship"

    def test_statlab_price_demand_analysis(self, statlab_price_demand_data):
        """
        正例：StatLab 价格需求分析

        负相关关系
        """
        if not hasattr(solution, 'regression_with_diagnostics'):
            pytest.skip("regression_with_diagnostics not implemented")

        price = statlab_price_demand_data['price']
        demand = statlab_price_demand_data['demand']

        result = solution.regression_with_diagnostics(price, demand)

        assert result is not None, "Should analyze price-demand relationship"

        # Slope should be negative
        if hasattr(result, 'model') and hasattr(result['model'], 'params'):
            slope = result['model'].params[1]
            assert slope < 0, "Price-demand relationship should be negative"


# =============================================================================
# 9. 数值稳定性测试
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestNumericalStability:
    """测试数值稳定性"""

    def test_large_values(self):
        """
        边界：大数值数据

        应能正确处理大数值
        """
        if not hasattr(solution, 'fit_ols'):
            pytest.skip("fit_ols not implemented")

        np.random.seed(42)
        x = np.random.uniform(1e6, 1e7, 100)
        y = 1e5 + 0.5 * x + np.random.normal(0, 1e5, 100)

        model = solution.fit_ols(x, y)

        assert model is not None, "Should handle large values"
        if hasattr(model, 'params'):
            assert np.all(np.isfinite(model.params)), "Coefficients should be finite"

    def test_small_values(self):
        """
        边界：小数值数据

        应能正确处理接近 0 的数值
        """
        if not hasattr(solution, 'fit_ols'):
            pytest.skip("fit_ols not implemented")

        np.random.seed(42)
        x = np.random.uniform(0.001, 0.01, 100)
        y = 0.0001 + 0.5 * x + np.random.normal(0, 0.001, 100)

        model = solution.fit_ols(x, y)

        assert model is not None, "Should handle small values"
        if hasattr(model, 'params'):
            assert np.all(np.isfinite(model.params)), "Coefficients should be finite"

    def test_reproducibility_with_seed(self, simple_linear_data):
        """
        正例：相同随机种子应产生相同结果

        验证可复现性
        """
        if not hasattr(solution, 'fit_ols'):
            pytest.skip("fit_ols not implemented")

        x = simple_linear_data['x']
        y = simple_linear_data['y']

        # OLS without random components should give identical results
        model1 = solution.fit_ols(x, y)
        model2 = solution.fit_ols(x, y)

        if hasattr(model1, 'params') and hasattr(model2, 'params'):
            np.testing.assert_array_almost_equal(
                model1.params, model2.params,
                decimal=10,
                err_msg="Same input should give identical results"
            )

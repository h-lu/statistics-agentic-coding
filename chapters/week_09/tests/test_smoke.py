"""
Week 09 烟雾测试（Smoke Test）

快速验证核心功能是否正常工作。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# 导入需要测试的函数
# 注意：当 starter_code/solution.py 不存在时，这些测试会跳过
try:
    from solution import (
        fit_linear_regression,
        calculate_regression_coefficients,
        calculate_vif,
        check_residuals_normality,
        check_residuals_homoscedasticity,
        calculate_cooks_distance,
        review_regression_report,
    )
except ImportError:
    pytest.skip("starter_code/solution.py not implemented yet", allow_module_level=True)
except ImportError:
    pytest.skip("solution.py not found", allow_module_level=True)


class TestSmokeBasicFunctionality:
    """测试基本功能是否可以运行"""

    @pytest.fixture
    def sample_regression_data(self):
        """创建回归测试数据"""
        np.random.seed(42)
        n = 100
        X = pd.DataFrame({
            'x1': np.random.normal(0, 1, n),
            'x2': np.random.normal(0, 1, n),
        })
        y = 5 + 2 * X['x1'] + 1.5 * X['x2'] + np.random.normal(0, 1, n)
        return X, y

    def test_smoke_fit_linear_regression(self, sample_regression_data):
        """烟雾测试：线性回归拟合"""
        X, y = sample_regression_data
        result = fit_linear_regression(X, y)

        # 应该返回模型对象或包含系数的字典
        assert result is not None
        assert hasattr(result, 'coef_') or 'coef' in result or \
               hasattr(result, 'params') or 'params' in result

    def test_smoke_calculate_regression_coefficients(self, sample_regression_data):
        """烟雾测试：回归系数计算"""
        X, y = sample_regression_data
        result = calculate_regression_coefficients(X, y)

        # 应该包含截距和斜率
        assert 'intercept' in result or 'const' in result
        assert 'coef' in result or 'slope' in result or 'coefs' in result

    def test_smoke_calculate_vif(self):
        """烟雾测试：VIF 计算"""
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            'x1': np.random.normal(0, 1, n),
            'x2': np.random.normal(0, 1, n),
            'x3': np.random.normal(0, 1, n),
        })

        result = calculate_vif(df)

        # 应该返回每个变量的 VIF
        assert isinstance(result, (pd.DataFrame, dict, list))
        if isinstance(result, pd.DataFrame):
            assert 'VIF' in result.columns or 'vif' in result.columns

    def test_smoke_check_residuals_normality(self, sample_regression_data):
        """烟雾测试：残差正态性检验"""
        X, y = sample_regression_data

        # 先拟合模型
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X, y)
        residuals = y - model.predict(X)

        result = check_residuals_normality(residuals)

        # 应该返回检验结果
        assert 'statistic' in result or 'stat' in result
        assert 'p_value' in result or 'p' in result

    def test_smoke_check_residuals_homoscedasticity(self, sample_regression_data):
        """烟雾测试：残差同方差性检验"""
        X, y = sample_regression_data

        # 先拟合模型
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X, y)
        residuals = y - model.predict(X)
        fitted = model.predict(X)

        result = check_residuals_homoscedasticity(fitted, residuals)

        # 应该返回检验结果
        assert 'statistic' in result or 'stat' in result or 'is_homoscedastic' in result

    def test_smoke_calculate_cooks_distance(self, sample_regression_data):
        """烟雾测试：Cook's 距离计算"""
        X, y = sample_regression_data

        result = calculate_cooks_distance(X, y)

        # 应该返回 Cook's 距离数组
        assert isinstance(result, (np.ndarray, list, pd.Series))

    def test_smoke_review_regression_report(self):
        """烟雾测试：回归报告审查"""
        report = """
        回归分析报告：

        R² = 0.82
        面积系数 = 1.25 (p < 0.001)

        结论：面积显著影响房价。
        """

        result = review_regression_report(report)

        # 应该返回审查结果
        assert 'has_issues' in result or 'score' in result
        assert 'issues' in result or 'recommendations' in result


class TestSmokeEndToEnd:
    """端到端工作流测试"""

    def test_complete_regression_workflow(self):
        """测试完整的回归分析工作流"""
        # 1. 生成数据
        np.random.seed(42)
        n = 100
        X = pd.DataFrame({
            'x1': np.random.normal(0, 1, n),
            'x2': np.random.normal(0, 1, n),
        })
        y = 5 + 2 * X['x1'] + 1.5 * X['x2'] + np.random.normal(0, 1, n)

        # 2. 拟合回归
        model = fit_linear_regression(X, y)
        assert model is not None

        # 3. 计算系数
        coefs = calculate_regression_coefficients(X, y)
        assert 'intercept' in coefs or 'const' in coefs

        # 4. 计算 VIF
        vif_result = calculate_vif(X)
        assert vif_result is not None

        # 5. 残差诊断
        from sklearn.linear_model import LinearRegression
        sklearn_model = LinearRegression()
        sklearn_model.fit(X, y)
        residuals = y - sklearn_model.predict(X)
        fitted = sklearn_model.predict(X)

        normality_result = check_residuals_normality(residuals)
        assert normality_result is not None

        homoscedasticity_result = check_residuals_homoscedasticity(fitted, residuals)
        assert homoscedasticity_result is not None

        # 6. Cook's 距离
        cooks_d = calculate_cooks_distance(X, y)
        assert cooks_d is not None

        # 7. 流程成功
        assert True

    def test_complete_review_workflow(self):
        """测试完整的报告审查工作流"""
        # 有问题的报告
        bad_report = """
        回归分析报告：

        R² = 0.82
        面积系数 = 1.25, p < 0.001

        结论：
        1. 面积会导致房价上涨。
        2. 模型拟合良好。
        """

        # 审查报告
        result = review_regression_report(bad_report)

        # 应该检测到问题（因果解释）
        assert 'has_issues' in result or 'score' in result
        if 'has_issues' in result:
            # 如果返回 has_issues，应该检测到因果解释问题
            assert result['has_issues'] is True

        # 流程成功
        assert True

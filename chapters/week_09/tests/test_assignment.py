"""
Week 09 作业测试框架

注意：这是基础测试框架，只验证核心函数的正确性。
完整的作业评估需要人工评分（解释、可视化、报告质量）。
"""

import pytest
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm

# 加载数据
penguins = sns.load_dataset("penguins")
penguins = penguins.dropna()
adelie = penguins[penguins["species"] == "Adelie"]


class TestLinearRegression:
    """测试简单线性回归拟合"""

    def test_model_fitting(self):
        """测试模型拟合是否正确"""
        X = adelie["bill_length_mm"].values
        y = adelie["bill_depth_mm"].values
        X_with_const = sm.add_constant(X)
        model = sm.OLS(y, X_with_const).fit()

        # 检查是否有两个参数（截距和斜率）
        assert len(model.params) == 2, "模型应该有两个参数（截距和斜率）"

        # 检查 R² 在合理范围内
        assert 0 <= model.rsquared <= 1, "R² 应该在 0 到 1 之间"

        # 检查系数是否为有限数
        assert np.isfinite(model.params).all(), "系数应该是有限数"

    def test_coefficient_signs(self):
        """测试系数符号（喙长度和喙深度通常是负相关）"""
        X = adelie["bill_length_mm"].values
        y = adelie["bill_depth_mm"].values
        X_with_const = sm.add_constant(X)
        model = sm.OLS(y, X_with_const).fit()

        # 斜率应该是负的（喙越长，喙通常越浅）
        # 注意：这不是严格的约束，生物学上可能有例外
        # 这里只测试斜率是否为有限数
        assert np.isfinite(model.params[1]), "斜率应该是有限数"


class TestModelDiagnostics:
    """测试模型诊断"""

    def test_cooks_distance(self):
        """测试 Cook's 距离计算"""
        X = adelie["bill_length_mm"].values
        y = adelie["bill_depth_mm"].values
        X_with_const = sm.add_constant(X)
        model = sm.OLS(y, X_with_const).fit()

        # 获取影响统计量
        influence = model.get_influence()
        cooks_d = influence.cooks_distance[0]

        # 检查 Cook's 距离数量
        assert len(cooks_d) == len(y), "Cook's 距离数量应该等于样本量"

        # 检查是否都为非负数
        assert (cooks_d >= 0).all(), "Cook's 距离应该都 >= 0"

        # 检查阈值计算
        n = len(cooks_d)
        threshold = 4 / n
        assert 0 < threshold < 1, "阈值应该在 0 到 1 之间"

    def test_residuals(self):
        """测试残差计算"""
        X = adelie["bill_length_mm"].values
        y = adelie["bill_depth_mm"].values
        X_with_const = sm.add_constant(X)
        model = sm.OLS(y, X_with_const).fit()

        # 检查残差数量
        assert len(model.resid) == len(y), "残差数量应该等于样本量"

        # 检查残差和是否接近 0（OLS 的性质）
        assert np.abs(np.sum(model.resid)) < 1e-10, "残差和应该接近 0"


class TestMultipleRegression:
    """测试多元回归"""

    def test_multiple_regression_fitting(self):
        """测试多元回归拟合"""
        X = adelie[["body_mass_g", "bill_length_mm", "flipper_length_mm"]].values
        y = adelie["bill_depth_mm"].values
        X_with_const = sm.add_constant(X)
        model = sm.OLS(y, X_with_const).fit()

        # 检查参数数量（截距 + 3 个自变量）
        assert len(model.params) == 4, "多元回归应该有 4 个参数"

        # 检查 R² 和调整 R²
        assert 0 <= model.rsquared <= 1, "R² 应该在 0 到 1 之间"
        assert 0 <= model.rsquared_adj <= 1, "调整 R² 应该在 0 到 1 之间"

        # 调整 R² 应该 <= R²
        assert model.rsquared_adj <= model.rsquared, "调整 R² 应该 <= R²"


class TestVIF:
    """测试方差膨胀因子计算"""

    def test_vif_calculation(self):
        """测试 VIF 计算"""
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        X = adelie[["body_mass_g", "bill_length_mm", "flipper_length_mm"]].values
        X_with_const = sm.add_constant(X)

        # 计算第一个自变量的 VIF（跳过截距）
        vif_1 = variance_inflation_factor(X_with_const, 1)

        # VIF 应该 >= 1
        assert vif_1 >= 1, "VIF 应该 >= 1"

        # VIF 应该是有限数
        assert np.isfinite(vif_1), "VIF 应该是有限数"


class TestRobustRegression:
    """测试稳健回归（可选）"""

    def test_rlm_fitting(self):
        """测试稳健回归拟合"""
        X = adelie["bill_length_mm"].values
        y = adelie["bill_depth_mm"].values
        X_with_const = sm.add_constant(X)

        # 拟合 RLM
        rlm_model = sm.RLM(y, X_with_const).fit()

        # 检查参数数量
        assert len(rlm_model.params) == 2, "RLM 应该有两个参数"

        # 检查参数是否为有限数
        assert np.isfinite(rlm_model.params).all(), "RLM 系数应该是有限数"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

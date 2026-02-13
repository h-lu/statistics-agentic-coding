"""
Week 11 烟雾测试（Smoke Test）

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
        fit_decision_tree,
        fit_random_forest,
        calculate_feature_importance,
        calculate_permutation_importance,
        tune_hyperparameters_grid,
        tune_hyperparameters_random,
        detect_overfitting,
        compare_tree_models,
        review_tree_model_code,
    )
except ImportError:
    pytest.skip("starter_code/solution.py not implemented yet", allow_module_level=True)


class TestSmokeBasicFunctionality:
    """测试基本功能是否可以运行"""

    @pytest.fixture
    def sample_regression_data(self):
        """创建回归测试数据"""
        np.random.seed(42)
        n = 100
        X = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, n),
            'feature_2': np.random.normal(0, 1, n),
        })
        y = 2 * X['feature_1'] + 0.5 * X['feature_2'] + np.random.normal(0, 0.5, n)
        return X, y

    def test_smoke_fit_decision_tree(self, sample_regression_data):
        """烟雾测试：决策树拟合"""
        X, y = sample_regression_data
        model = fit_decision_tree(X, y, max_depth=3)

        # 应该返回模型对象
        assert model is not None
        assert hasattr(model, 'predict') or 'tree' in str(type(model)).lower()

    def test_smoke_fit_random_forest(self, sample_regression_data):
        """烟雾测试：随机森林拟合"""
        X, y = sample_regression_data
        model = fit_random_forest(X, y, n_estimators=50)

        # 应该返回模型对象
        assert model is not None
        assert hasattr(model, 'predict') or 'forest' in str(type(model)).lower()

    def test_smoke_calculate_feature_importance(self, sample_regression_data):
        """烟雾测试：特征重要性计算"""
        X, y = sample_regression_data
        model = fit_random_forest(X, y, n_estimators=10)

        importance = calculate_feature_importance(model, X.columns)

        # 应该返回特征重要性
        assert isinstance(importance, (dict, pd.DataFrame, list))
        if isinstance(importance, dict):
            assert len(importance) > 0

    def test_smoke_calculate_permutation_importance(self, sample_regression_data):
        """烟雾测试：置换重要性计算"""
        X, y = sample_regression_data
        model = fit_random_forest(X, y, n_estimators=10)

        importance = calculate_permutation_importance(model, X, y)

        # 应该返回置换重要性
        assert isinstance(importance, (dict, pd.DataFrame, list))

    def test_smoke_tune_hyperparameters_grid(self, sample_regression_data):
        """烟雾测试：网格搜索调优"""
        X, y = sample_regression_data

        param_grid = {
            'max_depth': [3, 5],
            'min_samples_leaf': [1, 5]
        }

        result = tune_hyperparameters_grid(X, y, param_grid, cv=3)

        # 应该返回调优结果
        assert isinstance(result, dict)
        assert 'best_params' in result or 'best_params_' in result

    def test_smoke_tune_hyperparameters_random(self, sample_regression_data):
        """烟雾测试：随机搜索调优"""
        X, y = sample_regression_data

        param_dist = {
            'max_depth': [3, 5, 7, 10],
            'min_samples_leaf': [1, 5, 10]
        }

        result = tune_hyperparameters_random(X, y, param_dist, n_iter=5, cv=3)

        # 应该返回调优结果
        assert isinstance(result, dict)
        assert 'best_params' in result or 'best_params_' in result

    def test_smoke_detect_overfitting(self, sample_regression_data):
        """烟雾测试：过拟合检测"""
        X, y = sample_regression_data

        # 创建一个可能过拟合的树
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        tree_overfit = DecisionTreeRegressor(max_depth=None, random_state=42)
        tree_overfit.fit(X_train, y_train)

        overfitting_report = detect_overfitting(
            tree_overfit, X_train, y_train, X_test, y_test
        )

        # 应该返回过拟合检测报告
        assert isinstance(overfitting_report, dict)
        assert 'is_overfitting' in overfitting_report or 'overfitting' in overfitting_report

    def test_smoke_compare_tree_models(self, sample_regression_data):
        """烟雾测试：模型对比"""
        X, y = sample_regression_data

        comparison = compare_tree_models(X, y)

        # 应该返回对比结果
        assert isinstance(comparison, dict)
        assert 'decision_tree' in comparison or 'tree' in comparison

    def test_smoke_review_tree_model_code(self):
        """烟雾测试：树模型代码审查"""
        code = """
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X, y)
"""

        review_result = review_tree_model_code(code)

        # 应该返回审查结果
        assert isinstance(review_result, dict)
        assert 'has_issues' in review_result or 'issues' in review_result


class TestSmokeEndToEnd:
    """端到端工作流测试"""

    def test_complete_tree_workflow(self):
        """测试完整的树模型工作流"""
        # 1. 生成数据
        np.random.seed(42)
        n = 100
        X = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, n),
            'feature_2': np.random.normal(0, 1, n),
        })
        y = 2 * X['feature_1'] + 0.5 * X['feature_2'] + np.random.normal(0, 0.5, n)

        # 2. 拟合决策树
        dt = fit_decision_tree(X, y, max_depth=3)
        assert dt is not None

        # 3. 拟合随机森林
        rf = fit_random_forest(X, y, n_estimators=50)
        assert rf is not None

        # 4. 计算特征重要性
        importance = calculate_feature_importance(rf, X.columns)
        assert importance is not None

        # 5. 检测过拟合
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        overfitting_report = detect_overfitting(rf, X_train, y_train, X_test, y_test)
        assert overfitting_report is not None

        # 6. 超参数调优
        param_grid = {'max_depth': [3, 5]}
        tuning_result = tune_hyperparameters_grid(X_train, y_train, param_grid, cv=3)
        assert tuning_result is not None

        # 流程成功
        assert True

    def test_complete_review_workflow(self):
        """测试完整的代码审查工作流"""
        # 有问题的代码（容易过拟合）
        bad_code = """
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
train_score = dt.score(X_train, y_train)
test_score = dt.score(X_test, y_test)
print(f"训练集 R²: {train_score}")
print(f"测试集 R²: {test_score}")
"""

        # 审查代码
        result = review_tree_model_code(bad_code)

        # 应该检测到问题
        assert 'has_issues' in result or 'issues' in result
        if 'has_issues' in result:
            # 应该检测到缺少 max_depth 限制等
            assert result['has_issues'] is True

        # 流程成功
        assert True

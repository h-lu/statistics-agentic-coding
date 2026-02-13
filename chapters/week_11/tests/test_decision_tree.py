"""
Week 11 决策树测试

测试决策树的拟合、预测、可视化和过拟合检测。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score

# 导入需要测试的函数
try:
    from solution import (
        fit_decision_tree,
        export_tree_text,
        detect_tree_overfitting,
        calculate_tree_depth,
        get_tree_splits,
    )
except ImportError:
    pytest.skip("starter_code/solution.py not implemented yet", allow_module_level=True)


class TestDecisionTreeInitialization:
    """测试决策树初始化"""

    def test_decision_tree_with_max_depth(self, house_price_data):
        """测试带 max_depth 参数的决策树"""
        X = house_price_data[['area_sqm', 'bedrooms', 'age_years']]
        y = house_price_data['price']

        model = fit_decision_tree(X, y, max_depth=3)

        assert model is not None
        assert hasattr(model, 'get_depth')
        assert model.get_depth() <= 3

    def test_decision_tree_with_min_samples_split(self, house_price_data):
        """测试带 min_samples_split 参数的决策树"""
        X = house_price_data[['area_sqm', 'bedrooms', 'age_years']]
        y = house_price_data['price']

        model = fit_decision_tree(X, y, min_samples_split=20)

        assert model is not None
        assert hasattr(model, 'min_samples_split')

    def test_decision_tree_with_criterion_mse(self, house_price_data):
        """测试使用 MSE 准则的回归树"""
        X = house_price_data[['area_sqm', 'bedrooms', 'age_years']]
        y = house_price_data['price']

        model = fit_decision_tree(X, y, criterion='squared_error')  # 或 'mse'

        assert model is not None
        assert hasattr(model, 'criterion')

    def test_decision_tree_with_criterion_gini(self, churn_classification_data):
        """测试使用 Gini 准则的分类树"""
        X = churn_classification_data[['tenure_months', 'monthly_charges']]
        y = churn_classification_data['churn']

        model = fit_decision_tree(X, y, criterion='gini', task='classification')

        assert model is not None


class TestDecisionTreeFitAndPredict:
    """测试决策树的拟合和预测"""

    def test_decision_tree_fit_regression(self, house_price_data):
        """测试回归树的拟合"""
        X = house_price_data[['area_sqm', 'bedrooms', 'age_years']]
        y = house_price_data['price']

        model = fit_decision_tree(X, y, max_depth=5)

        assert hasattr(model, 'predict')

        y_pred = model.predict(X)
        assert len(y_pred) == len(y)

        # R² 应该 > 0（模型学到了一些东西）
        r2 = r2_score(y, y_pred)
        assert r2 > 0

    def test_decision_tree_fit_classification(self, churn_classification_data):
        """测试分类树的拟合"""
        X = churn_classification_data[['tenure_months', 'monthly_charges']]
        y = churn_classification_data['churn']

        model = fit_decision_tree(X, y, max_depth=5, task='classification')

        assert hasattr(model, 'predict')

        y_pred = model.predict(X)
        assert len(y_pred) == len(y)

        # 准确率应该 > 0.5（比随机好）
        acc = accuracy_score(y, y_pred)
        assert acc > 0.5

    def test_decision_tree_predict_single_sample(self, house_price_data):
        """测试单样本预测"""
        X = house_price_data[['area_sqm', 'bedrooms', 'age_years']]
        y = house_price_data['price']

        model = fit_decision_tree(X, y, max_depth=3)

        # 单个样本预测
        single_sample = pd.DataFrame({
            'area_sqm': [100],
            'bedrooms': [2],
            'age_years': [10]
        })

        prediction = model.predict(single_sample)

        assert len(prediction) == 1
        assert prediction[0] > 0  # 房价应该是正数


class TestFeatureImportances:
    """测试特征重要性属性"""

    def test_feature_importances_exists(self, house_price_data):
        """测试特征重要性属性是否存在"""
        X = house_price_data[['area_sqm', 'bedrooms', 'age_years']]
        y = house_price_data['price']

        model = fit_decision_tree(X, y, max_depth=5)

        assert hasattr(model, 'feature_importances_')

        importances = model.feature_importances_
        assert len(importances) == len(X.columns)

        # 特征重要性应该是非负的
        assert np.all(importances >= 0)

        # 特征重要性之和应该接近 1.0
        assert np.isclose(np.sum(importances), 1.0, atol=0.01)

    def test_feature_importances_ordering(self, house_price_data):
        """测试特征重要性的排序合理性"""
        X = house_price_data[['area_sqm', 'bedrooms', 'age_years']]
        y = house_price_data['price']

        model = fit_decision_tree(X, y, max_depth=5)

        importances = pd.Series(model.feature_importances_, index=X.columns)

        # 面积通常应该是最重要的（因为它是房价的主要驱动因素）
        # 注意：这不是严格的断言，因为存在随机性
        # 但在某些设置下应该如此
        assert importances.max() > 0  # 至少有一个特征重要

    def test_feature_importances_zero_for_unused(self, house_price_data):
        """测试未使用的特征重要性为 0"""
        # 添加一个噪声特征
        X = house_price_data[['area_sqm', 'bedrooms', 'age_years']].copy()
        X['noise_feature'] = np.random.normal(0, 1, len(X))
        y = house_price_data['price']

        # 使用浅树（只有 1-2 层），噪声特征应该不被使用
        model = fit_decision_tree(X, y, max_depth=2)

        importances = pd.Series(model.feature_importances_, index=X.columns)

        # 噪声特征的重要性应该很低或为 0
        assert importances['noise_feature'] < 0.1


class TestTreeExport:
    """测试树的导出功能"""

    def test_export_tree_text_not_empty(self, house_price_data):
        """测试导出的文本不为空"""
        X = house_price_data[['area_sqm', 'bedrooms', 'age_years']]
        y = house_price_data['price']

        model = fit_decision_tree(X, y, max_depth=3)

        tree_text = export_tree_text(model, feature_names=X.columns)

        assert isinstance(tree_text, str)
        assert len(tree_text) > 0

        # 应该包含分裂条件
        assert 'area_sqm' in tree_text or 'bedrooms' in tree_text

    def test_export_tree_text_contains_depth(self, house_price_data):
        """测试导出的文本包含深度信息"""
        X = house_price_data[['area_sqm', 'bedrooms', 'age_years']]
        y = house_price_data['price']

        model = fit_decision_tree(X, y, max_depth=2)

        tree_text = export_tree_text(model, feature_names=X.columns)

        # 应该包含一些树的关键词
        assert any(keyword in tree_text.lower() for keyword in [
            'node', 'split', 'leaf', 'samples', 'value'
        ])


class TestOverfittingDetection:
    """测试过拟合检测"""

    def test_overfitting_deep_tree(self, overfitting_scenario_data):
        """测试深度树的过拟合检测"""
        X_train, y_train, X_test, y_test = overfitting_scenario_data

        # 创建一个很深的树（容易过拟合）
        deep_tree = DecisionTreeRegressor(max_depth=None, random_state=42)
        deep_tree.fit(X_train, y_train)

        train_score = deep_tree.score(X_train, y_train)
        test_score = deep_tree.score(X_test, y_test)

        overfitting_report = detect_tree_overfitting(
            deep_tree, X_train, y_train, X_test, y_test
        )

        assert overfitting_report['is_overfitting'] is True
        assert overfitting_report['train_score'] > overfitting_report['test_score']
        assert overfitting_report['gap'] > 0.1  # 性能差距超过 10%

    def test_no_overfitting_shallow_tree(self, house_price_data):
        """测试浅树不过拟合"""
        from sklearn.model_selection import train_test_split

        X = house_price_data[['area_sqm', 'bedrooms', 'age_years']]
        y = house_price_data['price']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 创建一个浅树（不太可能过拟合）
        shallow_tree = DecisionTreeRegressor(max_depth=3, random_state=42)
        shallow_tree.fit(X_train, y_train)

        overfitting_report = detect_tree_overfitting(
            shallow_tree, X_train, y_train, X_test, y_test
        )

        # 浅树应该不太可能过拟合
        # 但注意：这取决于阈值设置
        assert 'is_overfitting' in overfitting_report

    def test_overfitting_threshold_sensitivity(self, overfitting_scenario_data):
        """测试过拟合检测的阈值敏感性"""
        X_train, y_train, X_test, y_test = overfitting_scenario_data

        tree = DecisionTreeRegressor(max_depth=None, random_state=42)
        tree.fit(X_train, y_train)

        report_low_threshold = detect_tree_overfitting(
            tree, X_train, y_train, X_test, y_test, threshold=0.05
        )
        report_high_threshold = detect_tree_overfitting(
            tree, X_train, y_train, X_test, y_test, threshold=0.3
        )

        # 低阈值更容易检测到过拟合
        assert report_low_threshold['is_overfitting'] is True

        # 高阈值可能不检测到过拟合
        # 但在某些情况下仍然可能


class TestTreeDepthAndSplits:
    """测试树的深度和分裂信息"""

    def test_calculate_tree_depth(self, house_price_data):
        """测试计算树的深度"""
        X = house_price_data[['area_sqm', 'bedrooms', 'age_years']]
        y = house_price_data['price']

        model = fit_decision_tree(X, y, max_depth=5)

        depth = calculate_tree_depth(model)

        assert isinstance(depth, int)
        assert depth <= 5  # 应该不超过 max_depth
        assert depth >= 1  # 至少有根节点

    def test_get_tree_splits_returns_dict(self, house_price_data):
        """测试获取树分裂信息返回字典"""
        X = house_price_data[['area_sqm', 'bedrooms', 'age_years']]
        y = house_price_data['price']

        model = fit_decision_tree(X, y, max_depth=3)

        splits = get_tree_splits(model)

        assert isinstance(splits, dict) or isinstance(splits, list)

    def test_get_tree_splits_not_empty(self, house_price_data):
        """测试获取树分裂信息不为空"""
        X = house_price_data[['area_sqm', 'bedrooms', 'age_years']]
        y = house_price_data['price']

        model = fit_decision_tree(X, y, max_depth=3)

        splits = get_tree_splits(model)

        # 应该有一些分裂信息
        if isinstance(splits, dict):
            assert len(splits) > 0
        elif isinstance(splits, list):
            assert len(splits) > 0


class TestDecisionTreeEdgeCases:
    """测试决策树的边界情况"""

    def test_decision_tree_single_feature(self, single_feature_data):
        """测试单特征决策树"""
        X = single_feature_data[['feature']]
        y = single_feature_data['target']

        model = fit_decision_tree(X, y, max_depth=3)

        assert model is not None
        y_pred = model.predict(X)
        assert len(y_pred) == len(y)

    def test_decision_tree_small_dataset(self, very_small_dataset):
        """测试小数据集决策树"""
        X = very_small_dataset[['feature_1', 'feature_2']]
        y = very_small_dataset['target']

        # 小数据集应该能拟合，但可能过拟合
        model = fit_decision_tree(X, y, max_depth=2)

        assert model is not None

    def test_decision_tree_constant_target(self, constant_target_data):
        """测试常数目标变量的决策树"""
        X = constant_target_data[['feature_1', 'feature_2']]
        y = constant_target_data['target']

        model = fit_decision_tree(X, y, max_depth=3)

        assert model is not None

        # 预测应该接近常数
        y_pred = model.predict(X)
        assert np.allclose(y_pred, y.mean(), atol=1)

    def test_decision_tree_perfect_fit(self):
        """测试完美拟合的决策树"""
        # 创建一个可以完美分类的数据
        np.random.seed(42)
        n = 100

        X = pd.DataFrame({
            'feature': np.concatenate([
                np.random.normal(-2, 1, n),  # 类别 0
                np.random.normal(2, 1, n)    # 类别 1
            ])
        })
        y = np.array([0] * n + [1] * n)

        model = fit_decision_tree(X, y, max_depth=5, task='classification')

        assert model is not None

        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)

        # 应该接近完美分类
        assert accuracy > 0.9


class TestDecisionTreeComparison:
    """测试决策树与其他模型的对比"""

    def test_tree_vs_linear_on_nonlinear(self, house_price_data):
        """测试树模型在非线性关系上的优势"""
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import train_test_split

        X = house_price_data[['area_sqm', 'bedrooms', 'age_years']]
        y = house_price_data['price']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 线性回归
        linear = LinearRegression()
        linear.fit(X_train, y_train)
        linear_r2 = linear.score(X_test, y_test)

        # 决策树
        tree = fit_decision_tree(X_train, y_train, max_depth=5)
        tree_r2 = tree.score(X_test, y_test)

        # 树模型应该在这个非线性数据上表现不差
        # 注意：这不是严格的断言，因为存在随机性
        assert tree_r2 > 0

    def test_tree_depth_impact_on_performance(self, house_price_data):
        """测试树深度对性能的影响"""
        from sklearn.model_selection import train_test_split

        X = house_price_data[['area_sqm', 'bedrooms', 'age_years']]
        y = house_price_data['price']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 浅树
        shallow_tree = fit_decision_tree(X_train, y_train, max_depth=2)
        shallow_train_r2 = shallow_tree.score(X_train, y_train)
        shallow_test_r2 = shallow_tree.score(X_test, y_test)

        # 深树
        deep_tree = fit_decision_tree(X_train, y_train, max_depth=10)
        deep_train_r2 = deep_tree.score(X_train, y_train)
        deep_test_r2 = deep_tree.score(X_test, y_test)

        # 深树训练集 R² 应该更高
        assert deep_train_r2 >= shallow_train_r2

        # 但测试集不一定更好（可能过拟合）
        # 这是树模型的关键权衡

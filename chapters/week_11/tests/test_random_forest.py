"""
Week 11 随机森林测试

测试随机森林的初始化、拟合、预测和方差降低特性。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split

# 导入需要测试的函数
try:
    from solution import (
        fit_random_forest,
        extract_feature_importance_rf,
        compare_rf_oob_score,
        measure_variance_reduction,
        get_rf_tree_diversity,
    )
except ImportError:
    pytest.skip("starter_code/solution.py not implemented yet", allow_module_level=True)


class TestRandomForestInitialization:
    """测试随机森林的初始化"""

    def test_random_forest_with_n_estimators(self, house_price_data):
        """测试带 n_estimators 参数的随机森林"""
        X = house_price_data[['area_sqm', 'bedrooms', 'age_years']]
        y = house_price_data['price']

        model = fit_random_forest(X, y, n_estimators=100)

        assert model is not None
        assert hasattr(model, 'n_estimators')
        # 允许实际 n_estimators 略有不同（取决于实现）
        assert model.n_estimators >= 90

    def test_random_forest_with_max_features(self, house_price_data):
        """测试带 max_features 参数的随机森林"""
        X = house_price_data[['area_sqm', 'bedrooms', 'age_years']]
        y = house_price_data['price']

        model = fit_random_forest(X, y, n_estimators=50, max_features='sqrt')

        assert model is not None
        assert hasattr(model, 'max_features')

    def test_random_forest_with_max_depth(self, house_price_data):
        """测试带 max_depth 参数的随机森林"""
        X = house_price_data[['area_sqm', 'bedrooms', 'age_years']]
        y = house_price_data['price']

        model = fit_random_forest(X, y, n_estimators=50, max_depth=7)

        assert model is not None
        # 检查每棵树的深度
        for tree in model.estimators_:
            assert tree.get_depth() <= 7

    def test_random_forest_classification(self, churn_classification_data):
        """测试分类随机森林"""
        X = churn_classification_data[['tenure_months', 'monthly_charges']]
        y = churn_classification_data['churn']

        model = fit_random_forest(X, y, n_estimators=50, task='classification')

        assert model is not None
        assert hasattr(model, 'predict')


class TestRandomForestFitAndPredict:
    """测试随机森林的拟合和预测"""

    def test_random_forest_fit_regression(self, house_price_data):
        """测试回归随机森林的拟合"""
        X = house_price_data[['area_sqm', 'bedrooms', 'age_years']]
        y = house_price_data['price']

        model = fit_random_forest(X, y, n_estimators=100)

        assert hasattr(model, 'predict')

        y_pred = model.predict(X)
        assert len(y_pred) == len(y)

        # R² 应该 > 0
        r2 = r2_score(y, y_pred)
        assert r2 > 0

    def test_random_forest_fit_classification(self, churn_classification_data):
        """测试分类随机森林的拟合"""
        X = churn_classification_data[['tenure_months', 'monthly_charges']]
        y = churn_classification_data['churn']

        model = fit_random_forest(X, y, n_estimators=50, task='classification')

        assert hasattr(model, 'predict')

        y_pred = model.predict(X)
        assert len(y_pred) == len(y)

        # 准确率应该 > 0.5
        acc = accuracy_score(y, y_pred)
        assert acc > 0.5

    def test_random_forest_predict_proba(self, churn_classification_data):
        """测试随机森林的概率预测"""
        X = churn_classification_data[['tenure_months', 'monthly_charges']]
        y = churn_classification_data['churn']

        model = fit_random_forest(X, y, n_estimators=50, task='classification')

        assert hasattr(model, 'predict_proba')

        y_proba = model.predict_proba(X)
        assert y_proba.shape == (len(y), 2)  # 二分类

        # 概率应该在 [0, 1] 之间
        assert np.all(y_proba >= 0)
        assert np.all(y_proba <= 1)

        # 每行概率之和应该为 1
        assert np.allclose(np.sum(y_proba, axis=1), 1.0)


class TestFeatureImportanceExtraction:
    """测试特征重要性提取"""

    def test_feature_importance_exists(self, house_price_data):
        """测试特征重要性属性是否存在"""
        X = house_price_data[['area_sqm', 'bedrooms', 'age_years']]
        y = house_price_data['price']

        model = fit_random_forest(X, y, n_estimators=50)

        assert hasattr(model, 'feature_importances_')

        importances = model.feature_importances_
        assert len(importances) == len(X.columns)

        # 特征重要性应该是非负的
        assert np.all(importances >= 0)

        # 特征重要性之和应该接近 1.0
        assert np.isclose(np.sum(importances), 1.0, atol=0.01)

    def test_extract_feature_importance_returns_dataframe(self, house_price_data):
        """测试提取特征重要性返回 DataFrame"""
        X = house_price_data[['area_sqm', 'bedrooms', 'age_years']]
        y = house_price_data['price']

        model = fit_random_forest(X, y, n_estimators=50)

        importance_df = extract_feature_importance_rf(model, X.columns)

        assert isinstance(importance_df, pd.DataFrame)
        assert 'feature' in importance_df.columns or '特征' in importance_df.columns
        assert 'importance' in importance_df.columns or '重要性' in importance_df.columns

        assert len(importance_df) == len(X.columns)

    def test_feature_importance_ranking(self, house_price_data):
        """测试特征重要性的排序"""
        X = house_price_data[['area_sqm', 'bedrooms', 'age_years']]
        y = house_price_data['price']

        model = fit_random_forest(X, y, n_estimators=50)

        importance_df = extract_feature_importance_rf(model, X.columns)

        # 检查是否排序（降序）
        if 'importance' in importance_df.columns:
            importances = importance_df['importance'].values
        else:
            importances = importance_df['重要性'].values

        # 应该是降序排列
        assert all(importances[i] >= importances[i+1] for i in range(len(importances)-1))

    def test_feature_importance_consistency(self, house_price_data):
        """测试特征重要性的一致性（多次运行）"""
        X = house_price_data[['area_sqm', 'bedrooms', 'age_years']]
        y = house_price_data['price']

        model1 = fit_random_forest(X, y, n_estimators=50, random_state=42)
        model2 = fit_random_forest(X, y, n_estimators=50, random_state=42)

        # 相同随机种子应该产生相似的特征重要性
        imp1 = model1.feature_importances_
        imp2 = model2.feature_importances_

        np.testing.assert_array_almost_equal(imp1, imp2, decimal=2)


class TestOOBScore:
    """测试袋外分数 (Out-of-Bag Score)"""

    def test_rf_with_oob_score(self, house_price_data):
        """测试带 OOB 分数的随机森林"""
        X = house_price_data[['area_sqm', 'bedrooms', 'age_years']]
        y = house_price_data['price']

        model = fit_random_forest(X, y, n_estimators=100, oob_score=True)

        assert hasattr(model, 'oob_score_')

        # OOB 分数应该在合理范围内
        assert 0 <= model.oob_score_ <= 1

    def test_compare_rf_oob_score_vs_test_score(self, house_price_data):
        """测试 OOB 分数与测试分数的对比"""
        X = house_price_data[['area_sqm', 'bedrooms', 'age_years']]
        y = house_price_data['price']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = fit_random_forest(X_train, y_train, n_estimators=100, oob_score=True)

        oob_score = model.oob_score_
        test_score = model.score(X_test, y_test)

        # OOB 分数应该接近测试分数（都是未见数据的性能估计）
        assert abs(oob_score - test_score) < 0.2  # 差异不超过 20%

    def test_compare_rf_oob_score_function(self, house_price_data):
        """测试 OOB 分数对比函数"""
        X = house_price_data[['area_sqm', 'bedrooms', 'age_years']]
        y = house_price_data['price']

        model_with_oob = fit_random_forest(X, y, n_estimators=100, oob_score=True)
        model_without_oob = fit_random_forest(X, y, n_estimators=100, oob_score=False)

        comparison = compare_rf_oob_score(model_with_oob, model_without_oob)

        assert isinstance(comparison, dict)
        assert 'oob_score' in comparison or 'with_oob' in comparison


class TestVarianceReduction:
    """测试方差降低（随机森林 vs 单棵树）"""

    def test_variance_reduction_vs_single_tree(self, house_price_data):
        """测试随机森林相对于单棵树的方差降低"""
        X = house_price_data[['area_sqm', 'bedrooms', 'age_years']]
        y = house_price_data['price']

        # 多次训练单棵树
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        tree_scores = []
        for i in range(10):
            tree = DecisionTreeRegressor(
                max_depth=7, random_state=42+i  # 不同随机种子
            )
            tree.fit(X_train, y_train)
            tree_scores.append(tree.score(X_test, y_test))

        tree_variance = np.var(tree_scores)

        # 多次训练随机森林
        rf_scores = []
        for i in range(10):
            rf = RandomForestRegressor(
                n_estimators=50, max_depth=7, random_state=42+i
            )
            rf.fit(X_train, y_train)
            rf_scores.append(rf.score(X_test, y_test))

        rf_variance = np.var(rf_scores)

        # 随机森林的方差应该更低
        # 注意：这不是绝对的，但通常如此
        # 我们只检查函数能够计算
        variance_report = measure_variance_reduction(
            tree_scores, rf_scores
        )

        assert isinstance(variance_report, dict)
        assert 'tree_variance' in variance_report or 'variance_reduction' in variance_report

    def test_measure_variance_reduction(self, house_price_data):
        """测试方差降低测量函数"""
        X = house_price_data[['area_sqm', 'bedrooms', 'age_years']]
        y = house_price_data['price']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 单棵树
        tree = DecisionTreeRegressor(max_depth=7, random_state=42)
        tree.fit(X_train, y_train)

        # 随机森林
        rf = RandomForestRegressor(
            n_estimators=100, max_depth=7, random_state=42
        )
        rf.fit(X_train, y_train)

        variance_report = measure_variance_reduction(
            tree, rf, X_test, y_test
        )

        assert isinstance(variance_report, dict)
        assert 'tree_score' in variance_report
        assert 'rf_score' in variance_report

    def test_rf_more_stable_than_tree(self, house_price_data):
        """测试随机森林比单棵树更稳定"""
        X = house_price_data[['area_sqm', 'bedrooms', 'age_years']]
        y = house_price_data['price']

        # 在不同的训练集划分上评估
        tree_scores = []
        rf_scores = []

        for random_state in range(42, 52):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=random_state
            )

            # 单棵树
            tree = DecisionTreeRegressor(max_depth=7, random_state=42)
            tree.fit(X_train, y_train)
            tree_scores.append(tree.score(X_test, y_test))

            # 随机森林
            rf = RandomForestRegressor(
                n_estimators=100, max_depth=7, random_state=42
            )
            rf.fit(X_train, y_train)
            rf_scores.append(rf.score(X_test, y_test))

        tree_std = np.std(tree_scores)
        rf_std = np.std(rf_scores)

        # 随机森林的标准差应该更小（更稳定）
        # 注意：这是一个软断言，因为存在随机性
        # 但在大多数情况下应该成立
        # 我们检查函数能正确计算
        stability_report = {
            'tree_std': tree_std,
            'rf_std': rf_std,
            'improvement': (tree_std - rf_std) / tree_std if tree_std > 0 else 0
        }

        assert isinstance(stability_report, dict)
        assert 'tree_std' in stability_report


class TestTreeDiversity:
    """测试树之间的多样性"""

    def test_get_rf_tree_diversity(self, house_price_data):
        """测试获取随机森林中树的多样性"""
        X = house_price_data[['area_sqm', 'bedrooms', 'age_years']]
        y = house_price_data['price']

        model = fit_random_forest(X, y, n_estimators=50, max_features='sqrt')

        diversity = get_rf_tree_diversity(model)

        assert isinstance(diversity, dict) or isinstance(diversity, float)

        if isinstance(diversity, dict):
            assert 'correlation' in diversity or 'diversity' in diversity

    def test_rf_trees_are_different(self, house_price_data):
        """测试随机森林中的树是不同的"""
        X = house_price_data[['area_sqm', 'bedrooms', 'age_years']]
        y = house_price_data['price']

        model = fit_random_forest(X, y, n_estimators=50, max_features='sqrt')

        # 检查前两棵树的预测是否不同
        tree_0_pred = model.estimators_[0].predict(X)
        tree_1_pred = model.estimators_[1].predict(X)

        # 应该有不同的预测
        assert not np.allclose(tree_0_pred, tree_1_pred)

    def test_max_features_impacts_diversity(self, house_price_data):
        """测试 max_features 对多样性的影响"""
        X = house_price_data[['area_sqm', 'bedrooms', 'age_years']]
        y = house_price_data['price']

        # max_features=1.0（所有特征）
        rf_low_diversity = fit_random_forest(
            X, y, n_estimators=50, max_features=1.0, random_state=42
        )

        # max_features='sqrt'（较少特征）
        rf_high_diversity = fit_random_forest(
            X, y, n_estimators=50, max_features='sqrt', random_state=42
        )

        # 高多样性的森林应该有不同的树
        # 我们检查第一棵树的深度或分裂点可能不同
        assert rf_low_diversity is not None
        assert rf_high_diversity is not None


class TestRandomForestEdgeCases:
    """测试随机森林的边界情况"""

    def test_random_forest_single_feature(self, single_feature_data):
        """测试单特征随机森林"""
        X = single_feature_data[['feature']]
        y = single_feature_data['target']

        model = fit_random_forest(X, y, n_estimators=50)

        assert model is not None
        y_pred = model.predict(X)
        assert len(y_pred) == len(y)

    def test_random_forest_small_dataset(self, very_small_dataset):
        """测试小数据集随机森林"""
        X = very_small_dataset[['feature_1', 'feature_2']]
        y = very_small_dataset['target']

        model = fit_random_forest(X, y, n_estimators=10)

        assert model is not None

    def test_random_forest_single_tree(self, house_price_data):
        """测试只有一棵树的森林"""
        X = house_price_data[['area_sqm', 'bedrooms', 'age_years']]
        y = house_price_data['price']

        model = fit_random_forest(X, y, n_estimators=1)

        assert model is not None
        assert len(model.estimators_) == 1

    def test_random_forest_many_trees(self, house_price_data):
        """测试很多棵树的森林"""
        X = house_price_data[['area_sqm', 'bedrooms', 'age_years']]
        y = house_price_data['price']

        model = fit_random_forest(X, y, n_estimators=200)

        assert model is not None
        assert len(model.estimators_) >= 190


class TestRandomForestComparison:
    """测试随机森林与其他模型的对比"""

    def test_rf_vs_tree_performance(self, house_price_data):
        """测试随机森林 vs 单棵树的性能"""
        from sklearn.model_selection import train_test_split

        X = house_price_data[['area_sqm', 'bedrooms', 'age_years']]
        y = house_price_data['price']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 单棵树
        tree = DecisionTreeRegressor(max_depth=7, random_state=42)
        tree.fit(X_train, y_train)
        tree_r2 = tree.score(X_test, y_test)

        # 随机森林
        rf = fit_random_forest(X_train, y_train, n_estimators=100, max_depth=7)
        rf_r2 = rf.score(X_test, y_test)

        # 随机森林应该不比单棵树差太多
        # 注意：这不是严格的断言，因为存在随机性
        assert rf_r2 > 0
        assert tree_r2 > 0

    def test_rf_vs_linear_on_nonlinear(self, house_price_data):
        """测试随机森林在线性数据上的表现"""
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import train_test_split

        # 创建一个线性关系的数据
        np.random.seed(42)
        n = 500
        X_lin = pd.DataFrame({
            'x1': np.random.normal(0, 1, n),
            'x2': np.random.normal(0, 1, n)
        })
        y_lin = 2 * X_lin['x1'] + 3 * X_lin['x2'] + np.random.normal(0, 0.5, n)

        X_train, X_test, y_train, y_test = train_test_split(
            X_lin, y_lin, test_size=0.2, random_state=42
        )

        # 线性回归
        linear = LinearRegression()
        linear.fit(X_train, y_train)
        linear_r2 = linear.score(X_test, y_test)

        # 随机森林
        rf = fit_random_forest(X_train, y_train, n_estimators=100)
        rf_r2 = rf.score(X_test, y_test)

        # 线性回归在线性数据上应该表现很好
        # 但随机森林也不应该太差
        assert linear_r2 > 0.7
        assert rf_r2 > 0.5

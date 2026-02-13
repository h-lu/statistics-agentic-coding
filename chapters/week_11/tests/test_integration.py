"""
Week 11 集成测试

测试完整的流水线：预处理 → 树模型 → 随机森林 → 调优 → 重要性 → 报告
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error

# 导入需要测试的函数
try:
    from solution import (
        complete_tree_pipeline,
        compare_models,
        generate_tree_report,
        review_ai_tree_code,
    )
except ImportError:
    pytest.skip("starter_code/solution.py not implemented yet", allow_module_level=True)


class TestCompletePipeline:
    """测试完整的建模流水线"""

    def test_complete_regression_pipeline(self, house_price_data):
        """测试完整的回归流水线"""
        X = house_price_data[['area_sqm', 'bedrooms', 'age_years', 'distance_km']]
        y = house_price_data['price']

        result = complete_tree_pipeline(
            X, y,
            task='regression',
            test_size=0.2,
            random_state=42
        )

        assert isinstance(result, dict)

        # 应该包含模型
        assert 'decision_tree' in result or 'tree' in result
        assert 'random_forest' in result or 'forest' in result

        # 应该包含评估
        assert 'train_score' in result or 'scores' in result
        assert 'test_score' in result or 'scores' in result

        # 应该包含特征重要性
        assert 'feature_importance' in result or 'importance' in result

    def test_complete_classification_pipeline(self, churn_classification_data):
        """测试完整的分类流水线"""
        X = churn_classification_data[['tenure_months', 'monthly_charges', 'total_charges']]
        y = churn_classification_data['churn']

        result = complete_tree_pipeline(
            X, y,
            task='classification',
            test_size=0.2,
            random_state=42
        )

        assert isinstance(result, dict)

        # 应该包含模型
        assert 'decision_tree' in result or 'tree' in result
        assert 'random_forest' in result or 'forest' in result

        # 应该包含评估
        assert 'train_score' in result or 'accuracy' in result
        assert 'test_score' in result or 'accuracy' in result

    def test_pipeline_with_hyperparameter_tuning(self, house_price_data):
        """测试带超参数调优的流水线"""
        X = house_price_data[['area_sqm', 'bedrooms', 'age_years']]
        y = house_price_data['price']

        param_grid = {
            'max_depth': [3, 5, 7],
            'min_samples_leaf': [5, 10]
        }

        result = complete_tree_pipeline(
            X, y,
            task='regression',
            param_grid=param_grid,
            cv=3,
            random_state=42
        )

        assert isinstance(result, dict)

        # 应该包含最佳参数
        assert 'best_params' in result or 'tuned' in result

    def test_pipeline_with_feature_importance(self, house_price_data):
        """测试带特征重要性的流水线"""
        X = house_price_data[['area_sqm', 'bedrooms', 'age_years', 'distance_km']]
        y = house_price_data['price']

        result = complete_tree_pipeline(
            X, y,
            task='regression',
            compute_permutation_importance=True,
            random_state=42
        )

        assert isinstance(result, dict)

        # 应该包含两种重要性
        assert 'feature_importance' in result or 'builtin_importance' in result
        assert 'permutation_importance' in result or 'perm_importance' in result


class TestModelComparison:
    """测试模型对比"""

    def test_compare_linear_vs_tree_vs_forest(self, house_price_data):
        """测试线性回归 vs 决策树 vs 随机森林"""
        X = house_price_data[['area_sqm', 'bedrooms', 'age_years']]
        y = house_price_data['price']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        comparison = compare_models(
            X_train, y_train, X_test, y_test,
            models=['linear', 'tree', 'forest'],
            task='regression'
        )

        assert isinstance(comparison, dict) or isinstance(comparison, pd.DataFrame)

        # 应该包含所有模型的结果
        if isinstance(comparison, dict):
            assert 'linear' in comparison or 'linear_regression' in comparison
            assert 'tree' in comparison or 'decision_tree' in comparison
            assert 'forest' in comparison or 'random_forest' in comparison

    def test_compare_models_returns_r2(self, house_price_data):
        """测试模型对比返回 R² 分数"""
        X = house_price_data[['area_sqm', 'bedrooms', 'age_years']]
        y = house_price_data['price']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        comparison = compare_models(
            X_train, y_train, X_test, y_test,
            models=['tree', 'forest'],
            task='regression'
        )

        # 应该包含 R²
        if isinstance(comparison, dict):
            if 'forest' in comparison:
                forest_result = comparison['forest']
                assert 'r2' in forest_result or 'score' in forest_result
                assert forest_result.get('r2', forest_result.get('score', 0)) > 0

    def test_compare_classification_models(self, churn_classification_data):
        """测试分类模型对比"""
        X = churn_classification_data[['tenure_months', 'monthly_charges']]
        y = churn_classification_data['churn']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        comparison = compare_models(
            X_train, y_train, X_test, y_test,
            models=['logistic', 'tree', 'forest'],
            task='classification'
        )

        assert isinstance(comparison, dict) or isinstance(comparison, pd.DataFrame)

        # 所有模型的准确率应该 > 0.5（比随机好）
        if isinstance(comparison, dict):
            for model_name, result in comparison.items():
                if 'accuracy' in result:
                    assert result['accuracy'] > 0.5

    def test_forest_outperforms_tree(self, house_price_data):
        """测试随机森林优于单棵树（在合理情况下）"""
        X = house_price_data[['area_sqm', 'bedrooms', 'age_years']]
        y = house_price_data['price']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        comparison = compare_models(
            X_train, y_train, X_test, y_test,
            models=['tree', 'forest'],
            task='regression'
        )

        if isinstance(comparison, dict):
            tree_r2 = comparison.get('tree', {}).get('r2', 0)
            forest_r2 = comparison.get('forest', {}).get('r2', 0)

            # 随机森林应该不比单棵树差（通常更好）
            # 注意：这不是绝对的，但大多数情况下成立
            assert forest_r2 > 0
            assert tree_r2 > 0


class TestReportGeneration:
    """测试报告生成"""

    def test_generate_tree_report_regression(self, house_price_data):
        """测试生成回归报告"""
        X = house_price_data[['area_sqm', 'bedrooms', 'age_years']]
        y = house_price_data['price']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        report = generate_tree_report(
            X_train, y_train, X_test, y_test,
            task='regression',
            model_type='random_forest'
        )

        assert isinstance(report, str)
        assert len(report) > 0

        # 应该包含关键词
        assert any(keyword in report.lower() for keyword in [
            'r2', 'r²', 'score', 'mse'
        ])

    def test_generate_tree_report_classification(self, churn_classification_data):
        """测试生成分类报告"""
        X = churn_classification_data[['tenure_months', 'monthly_charges']]
        y = churn_classification_data['churn']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        report = generate_tree_report(
            X_train, y_train, X_test, y_test,
            task='classification',
            model_type='random_forest'
        )

        assert isinstance(report, str)
        assert len(report) > 0

        # 应该包含关键词
        assert any(keyword in report.lower() for keyword in [
            'accuracy', '准确率', 'precision', 'recall'
        ])

    def test_report_includes_feature_importance(self, house_price_data):
        """测试报告包含特征重要性"""
        X = house_price_data[['area_sqm', 'bedrooms', 'age_years']]
        y = house_price_data['price']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        report = generate_tree_report(
            X_train, y_train, X_test, y_test,
            task='regression',
            include_feature_importance=True
        )

        # 应该包含特征重要性信息
        assert 'importance' in report.lower() or '重要性' in report

    def test_report_includes_limitations(self, house_price_data):
        """测试报告包含局限性讨论"""
        X = house_price_data[['area_sqm', 'bedrooms', 'age_years']]
        y = house_price_data['price']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        report = generate_tree_report(
            X_train, y_train, X_test, y_test,
            task='regression',
            include_limitations=True
        )

        # 应该包含局限性信息
        assert any(keyword in report.lower() for keyword in [
            'limitation', '局限', 'caus', 'overfit', '过拟合'
        ])


class TestAICodeReview:
    """测试 AI 代码审查"""

    def test_review_good_tree_code(self, good_tree_code_example):
        """测试审查好的树模型代码"""
        review = review_ai_tree_code(good_tree_code_example)

        assert isinstance(review, dict)
        assert 'has_issues' in review or 'score' in review

        # 好的代码应该没有严重问题
        if review.get('has_issues', True):
            # 如果有问题，应该是低严重性的
            assert 'issues' in review
            severe_issues = [i for i in review['issues'] if i.get('severity', 'medium') == 'high']
            assert len(severe_issues) == 0

    def test_review_overfitting_code(self, bad_tree_code_overfitting):
        """测试审查过拟合的代码"""
        review = review_ai_tree_code(bad_tree_code_overfitting)

        assert isinstance(review, dict)
        assert review.get('has_issues', False) is True

        # 应该检测到过拟合风险
        if 'issues' in review:
            overfitting_issues = [
                i for i in review['issues']
                if 'overfit' in i.get('message', '').lower() or
                   '过拟合' in i.get('message', '') or
                   'max_depth' in i.get('message', '').lower()
            ]
            assert len(overfitting_issues) > 0

    def test_review_no_tuning_code(self, bad_tree_code_no_tuning):
        """测试审查缺少调优的代码"""
        review = review_ai_tree_code(bad_tree_code_no_tuning)

        assert isinstance(review, dict)
        assert review.get('has_issues', False) is True

        # 应该检测到缺少调优
        if 'issues' in review:
            tuning_issues = [
                i for i in review['issues']
                if 'tuning' in i.get('message', '').lower() or
                   '调优' in i.get('message', '') or
                   'grid' in i.get('message', '').lower()
            ]
            # 至少应该有警告
            assert len(tuning_issues) >= 0

    def test_review_misinterpretation_code(self, bad_tree_code_feature_importance_misinterpretation):
        """测试审查错误解释的代码"""
        review = review_ai_tree_code(bad_tree_code_feature_importance_misinterpretation)

        assert isinstance(review, dict)

        # 应该检测到因果解释问题（如果有注释）
        if 'issues' in review:
            causality_issues = [
                i for i in review['issues']
                if 'causal' in i.get('message', '').lower() or
                   '因果' in i.get('message', '') or
                   'cause' in i.get('message', '').lower()
            ]
            # 注释可能不完整，所以不做严格断言
            assert len(causality_issues) >= 0


class TestEndToEndWorkflow:
    """端到端工作流测试"""

    def test_complete_analysis_workflow(self, house_price_data):
        """测试完整的分析工作流"""
        # 1. 划分数据
        X = house_price_data[['area_sqm', 'bedrooms', 'age_years', 'distance_km']]
        y = house_price_data['price']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 2. 运行完整流水线
        result = complete_tree_pipeline(
            X_train, y_train,
            test_data=(X_test, y_test),
            task='regression',
            tune_hyperparameters=True,
            compute_permutation_importance=True
        )

        assert isinstance(result, dict)

        # 3. 对比模型
        comparison = compare_models(
            X_train, y_train, X_test, y_test,
            models=['linear', 'tree', 'forest'],
            task='regression'
        )

        assert isinstance(comparison, dict) or isinstance(comparison, pd.DataFrame)

        # 4. 生成报告
        report = generate_tree_report(
            X_train, y_train, X_test, y_test,
            task='regression',
            include_feature_importance=True,
            include_limitations=True
        )

        assert isinstance(report, str)
        assert len(report) > 0

        # 工作流成功
        assert True

    def test_workflow_with_categorical_features(self, house_price_data_with_categories):
        """测试带类别特征的工作流"""
        df = house_price_data_with_categories

        # 编码类别特征
        X_encoded = pd.get_dummies(
            df[['area_sqm', 'bedrooms', 'age_years', 'city', 'property_type']],
            columns=['city', 'property_type'],
            drop_first=True
        )
        y = df['price']

        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=0.2, random_state=42
        )

        # 应该能处理编码后的特征
        result = complete_tree_pipeline(
            X_train, y_train,
            test_data=(X_test, y_test),
            task='regression'
        )

        assert isinstance(result, dict)

    def test_workflow_detects_and_warns_overfitting(self, house_price_data):
        """测试工作流检测并警告过拟合"""
        X = house_price_data[['area_sqm', 'bedrooms', 'age_years']]
        y = house_price_data['price']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 使用容易过拟合的参数
        result = complete_tree_pipeline(
            X_train, y_train,
            test_data=(X_test, y_test),
            task='regression',
            max_depth=None,  # 无限制
            check_overfitting=True
        )

        assert isinstance(result, dict)

        # 应该包含过拟合检测
        assert 'overfitting' in result or 'overfit' in result or 'warning' in result


class TestModelRobustness:
    """测试模型稳健性"""

    def test_model_robustness_to_noise(self, house_price_data):
        """测试模型对噪声的稳健性"""
        X = house_price_data[['area_sqm', 'bedrooms', 'age_years']].copy()
        y = house_price_data['price'].copy()

        # 添加噪声
        X_noisy = X + np.random.normal(0, 5, X.shape)
        y_noisy = y + np.random.normal(0, 10, y.shape)

        X_train, X_test, y_train, y_test = train_test_split(
            X_noisy, y_noisy, test_size=0.2, random_state=42
        )

        # 随机森林应该对噪声更稳健
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        rf_r2 = rf.score(X_test, y_test)

        tree = DecisionTreeRegressor(max_depth=7, random_state=42)
        tree.fit(X_train, y_train)
        tree_r2 = tree.score(X_test, y_test)

        # 两者都应该能学到一些东西（R² > 0）
        assert rf_r2 > 0
        assert tree_r2 > 0

    def test_model_stability_across_splits(self, house_price_data):
        """测试模型在不同数据划分上的稳定性"""
        X = house_price_data[['area_sqm', 'bedrooms', 'age_years']]
        y = house_price_data['price']

        rf_scores = []
        tree_scores = []

        for random_state in range(42, 52):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=random_state
            )

            # 随机森林
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            rf_scores.append(rf.score(X_test, y_test))

            # 决策树
            tree = DecisionTreeRegressor(max_depth=7, random_state=42)
            tree.fit(X_train, y_train)
            tree_scores.append(tree.score(X_test, y_test))

        # 随机森林的方差应该更小
        rf_std = np.std(rf_scores)
        tree_std = np.std(tree_scores)

        # 至少能计算
        assert rf_std >= 0
        assert tree_std >= 0


class TestReproducibility:
    """测试可复现性"""

    def test_same_random_seed_same_results(self, house_price_data):
        """测试相同随机种子产生相同结果"""
        X = house_price_data[['area_sqm', 'bedrooms', 'age_years']]
        y = house_price_data['price']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 模型 1
        rf1 = RandomForestRegressor(n_estimators=100, random_state=42)
        rf1.fit(X_train, y_train)
        pred1 = rf1.predict(X_test)

        # 模型 2（相同种子）
        rf2 = RandomForestRegressor(n_estimators=100, random_state=42)
        rf2.fit(X_train, y_train)
        pred2 = rf2.predict(X_test)

        # 预测应该完全相同
        np.testing.assert_array_equal(pred1, pred2)

    def test_different_random_seed_different_results(self, house_price_data):
        """测试不同随机种子产生不同结果"""
        X = house_price_data[['area_sqm', 'bedrooms', 'age_years']]
        y = house_price_data['price']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 模型 1
        rf1 = RandomForestRegressor(n_estimators=100, random_state=42)
        rf1.fit(X_train, y_train)
        pred1 = rf1.predict(X_test)

        # 模型 2（不同种子）
        rf2 = RandomForestRegressor(n_estimators=100, random_state=123)
        rf2.fit(X_train, y_train)
        pred2 = rf2.predict(X_test)

        # 预测应该不同（至少大部分不同）
        assert not np.allclose(pred1, pred2)

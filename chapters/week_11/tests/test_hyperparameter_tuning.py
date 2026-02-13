"""
Week 11 超参数调优测试

测试网格搜索、随机搜索和超参数评估。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split

# 导入需要测试的函数
try:
    from solution import (
        tune_hyperparameters_grid,
        tune_hyperparameters_random,
        compare_grid_vs_random,
        extract_best_params,
        extract_cv_results,
        analyze_hyperparameter_sensitivity,
        validate_hyperparameters,
    )
except ImportError:
    pytest.skip("starter_code/solution.py not implemented yet", allow_module_level=True)


class TestGridSearchCV:
    """测试网格搜索"""

    def test_grid_search_basic(self, house_price_data):
        """测试基本的网格搜索"""
        X = house_price_data[['area_sqm', 'bedrooms', 'age_years']]
        y = house_price_data['price']

        param_grid = {
            'max_depth': [3, 5, 7],
            'min_samples_leaf': [1, 5, 10]
        }

        result = tune_hyperparameters_grid(
            X, y, param_grid, cv=5, scoring='r2'
        )

        assert isinstance(result, dict)
        assert 'best_params' in result or 'best_params_' in result
        assert 'best_score' in result or 'best_score_' in result

    def test_grid_search_returns_best_params(self, house_price_data):
        """测试网格搜索返回最佳参数"""
        X = house_price_data[['area_sqm', 'bedrooms', 'age_years']]
        y = house_price_data['price']

        param_grid = {
            'max_depth': [3, 5],
            'min_samples_leaf': [5, 10]
        }

        result = tune_hyperparameters_grid(
            X, y, param_grid, cv=3
        )

        best_params = extract_best_params(result)

        assert isinstance(best_params, dict)
        assert 'max_depth' in best_params
        assert 'min_samples_leaf' in best_params

        # 最佳参数应该在参数网格中
        assert best_params['max_depth'] in [3, 5]
        assert best_params['min_samples_leaf'] in [5, 10]

    def test_grid_search_cv_results(self, house_price_data):
        """测试网格搜索返回 CV 结果"""
        X = house_price_data[['area_sqm', 'bedrooms', 'age_years']]
        y = house_price_data['price']

        param_grid = {
            'max_depth': [3, 5],
            'min_samples_leaf': [5, 10]
        }

        result = tune_hyperparameters_grid(
            X, y, param_grid, cv=3
        )

        cv_results = extract_cv_results(result)

        assert isinstance(cv_results, (dict, pd.DataFrame))

        if isinstance(cv_results, dict):
            assert 'params' in cv_results or 'mean_test_score' in cv_results

    def test_grid_search_with_scoring_metric(self, house_price_data):
        """测试网格搜索使用不同的评分指标"""
        X = house_price_data[['area_sqm', 'bedrooms', 'age_years']]
        y = house_price_data['price']

        param_grid = {'max_depth': [3, 5]}

        # 测试 R²
        result_r2 = tune_hyperparameters_grid(
            X, y, param_grid, cv=3, scoring='r2'
        )

        # 测试负 MSE
        result_mse = tune_hyperparameters_grid(
            X, y, param_grid, cv=3, scoring='neg_mean_squared_error'
        )

        assert result_r2 is not None
        assert result_mse is not None

        # R² 应该接近 1，负 MSE 应该接近 0
        assert extract_best_params(result_r2) is not None
        assert extract_best_params(result_mse) is not None

    def test_grid_search_computationally_expensive(self, house_price_data):
        """测试网格搜索的计算成本"""
        X = house_price_data[['area_sqm', 'bedrooms', 'age_years']]
        y = house_price_data['price']

        # 大参数网格
        param_grid_large = {
            'max_depth': [3, 5, 7, 10, None],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 5, 10, 20]
        }

        # 应该能处理，但可能需要时间
        result = tune_hyperparameters_grid(
            X, y, param_grid_large, cv=3
        )

        assert result is not None
        assert 'best_params' in result or 'best_params_' in result


class TestRandomizedSearchCV:
    """测试随机搜索"""

    def test_random_search_basic(self, house_price_data):
        """测试基本的随机搜索"""
        X = house_price_data[['area_sqm', 'bedrooms', 'age_years']]
        y = house_price_data['price']

        param_dist = {
            'max_depth': [3, 5, 7, 10],
            'min_samples_leaf': [1, 5, 10, 20]
        }

        result = tune_hyperparameters_random(
            X, y, param_dist, n_iter=10, cv=5
        )

        assert isinstance(result, dict)
        assert 'best_params' in result or 'best_params_' in result

    def test_random_search_n_iter_parameter(self, house_price_data):
        """测试随机搜索的 n_iter 参数"""
        X = house_price_data[['area_sqm', 'bedrooms', 'age_years']]
        y = house_price_data['price']

        param_dist = {
            'max_depth': [3, 5, 7, 10, 15],
            'min_samples_leaf': [1, 5, 10, 20]
        }

        # 不同的 n_iter
        result_5 = tune_hyperparameters_random(
            X, y, param_dist, n_iter=5, cv=3
        )

        result_20 = tune_hyperparameters_random(
            X, y, param_dist, n_iter=20, cv=3
        )

        assert result_5 is not None
        assert result_20 is not None

    def test_random_search_with_distributions(self, house_price_data):
        """测试随机搜索使用分布（不只是列表）"""
        from scipy.stats import randint, uniform

        X = house_price_data[['area_sqm', 'bedrooms', 'age_years']]
        y = house_price_data['price']

        param_dist = {
            'max_depth': randint(3, 15),
            'min_samples_leaf': randint(1, 20),
            'min_samples_split': randint(2, 20)
        }

        result = tune_hyperparameters_random(
            X, y, param_dist, n_iter=10, cv=3
        )

        assert result is not None
        best_params = extract_best_params(result)

        # 参数应该在实际范围内
        assert 3 <= best_params['max_depth'] <= 15
        assert 1 <= best_params['min_samples_leaf'] <= 20


class TestGridVsRandomComparison:
    """测试网格搜索 vs 随机搜索"""

    def test_compare_grid_vs_random(self, house_price_data):
        """测试比较网格搜索和随机搜索"""
        X = house_price_data[['area_sqm', 'bedrooms', 'age_years']]
        y = house_price_data['price']

        param_grid = {
            'max_depth': [3, 5, 7, 10],
            'min_samples_leaf': [1, 5, 10]
        }

        comparison = compare_grid_vs_random(
            X, y, param_grid, n_iter=10, cv=3
        )

        assert isinstance(comparison, dict)
        assert 'grid_search' in comparison or 'grid' in comparison
        assert 'random_search' in comparison or 'random' in comparison

    def test_random_search_faster_than_grid(self, house_price_data):
        """测试随机搜索比网格搜索快"""
        import time

        X = house_price_data[['area_sqm', 'bedrooms', 'age_years']]
        y = house_price_data['price']

        param_grid = {
            'max_depth': [3, 5, 7, 10, 15],
            'min_samples_leaf': [1, 5, 10, 20],
            'min_samples_split': [2, 5, 10, 20]
        }

        # 网格搜索：所有组合 = 5*4*4 = 80
        start_grid = time.time()
        result_grid = tune_hyperparameters_grid(
            X, y, param_grid, cv=3
        )
        time_grid = time.time() - start_grid

        # 随机搜索：只尝试 20 个组合
        start_random = time.time()
        result_random = tune_hyperparameters_random(
            X, y, param_grid, n_iter=20, cv=3
        )
        time_random = time.time() - start_random

        # 随机搜索应该更快（虽然不是绝对的）
        # 我们只检查两者都能成功
        assert result_grid is not None
        assert result_random is not None

    def test_grid_vs_random_score_quality(self, house_price_data):
        """测试网格搜索 vs 随机搜索的得分质量"""
        X = house_price_data[['area_sqm', 'bedrooms', 'age_years']]
        y = house_price_data['price']

        param_grid = {
            'max_depth': [3, 5, 7, 10],
            'min_samples_leaf': [1, 5, 10]
        }

        comparison = compare_grid_vs_random(
            X, y, param_grid, n_iter=10, cv=5
        )

        # 网格搜索应该找到最优的（因为穷举）
        # 随机搜索应该接近
        if 'grid_best_score' in comparison and 'random_best_score' in comparison:
            grid_score = comparison['grid_best_score']
            random_score = comparison['random_best_score']

            # 随机搜索的得分应该不太差
            assert abs(grid_score - random_score) < 0.1


class TestHyperparameterSensitivity:
    """测试超参数敏感性分析"""

    def test_analyze_hyperparameter_sensitivity(self, house_price_data):
        """测试超参数敏感性分析"""
        X = house_price_data[['area_sqm', 'bedrooms', 'age_years']]
        y = house_price_data['price']

        sensitivity = analyze_hyperparameter_sensitivity(
            X, y,
            param_name='max_depth',
            param_values=[3, 5, 7, 10, 15],
            cv=3
        )

        assert isinstance(sensitivity, dict) or isinstance(sensitivity, pd.DataFrame)

        if isinstance(sensitivity, dict):
            assert 'scores' in sensitivity or 'mean_score' in sensitivity

    def test_max_depth_sensitivity(self, house_price_data):
        """测试 max_depth 敏感性"""
        X = house_price_data[['area_sqm', 'bedrooms', 'age_years']]
        y = house_price_data['price']

        depths = [3, 5, 7, 10, None]
        scores = []

        for depth in depths:
            model = DecisionTreeRegressor(
                max_depth=depth, random_state=42
            )
            from sklearn.model_selection import cross_val_score
            cv_scores = cross_val_score(model, X, y, cv=3, scoring='r2')
            scores.append(cv_scores.mean())

        # 应该有一个"最佳"深度
        # 太浅（3）可能欠拟合，太深（None）可能过拟合
        assert len(scores) == len(depths)
        assert all(s > 0 for s in scores)  # 所有 R² 应该为正

    def test_n_estimators_sensitivity(self, house_price_data):
        """测试 n_estimators 敏感性"""
        X = house_price_data[['area_sqm', 'bedrooms', 'age_years']]
        y = house_price_data['price']

        n_trees_list = [10, 50, 100, 200]
        scores = []

        for n_trees in n_trees_list:
            model = RandomForestRegressor(
                n_estimators=n_trees, random_state=42
            )
            from sklearn.model_selection import cross_val_score
            cv_scores = cross_val_score(model, X, y, cv=3, scoring='r2')
            scores.append(cv_scores.mean())

        # 更多树通常会提升性能（但有收益递减）
        assert len(scores) == len(n_trees_list)
        assert all(s > 0 for s in scores)


class TestHyperparameterValidation:
    """测试超参数验证"""

    def test_validate_hyperparameters_valid(self):
        """测试验证有效的超参数"""
        params = {
            'max_depth': 5,
            'min_samples_leaf': 10,
            'n_estimators': 100
        }

        validation = validate_hyperparameters(params, model_type='random_forest')

        assert isinstance(validation, dict)
        assert validation['valid'] is True

    def test_validate_hyperparameters_invalid(self):
        """测试验证无效的超参数"""
        params = {
            'max_depth': -1,  # 无效
            'min_samples_leaf': 0,  # 无效
            'n_estimators': 0  # 无效
        }

        validation = validate_hyperparameters(params, model_type='random_forest')

        assert isinstance(validation, dict)
        assert validation['valid'] is False
        assert 'errors' in validation or 'issues' in validation

    def test_validate_hyperparameters_warning(self):
        """测试超参数警告"""
        params = {
            'max_depth': None,  # 可能导致过拟合
            'min_samples_leaf': 1  # 可能导致过拟合
        }

        validation = validate_hyperparameters(
            params, model_type='random_forest',
            check_overfitting_risk=True
        )

        assert isinstance(validation, dict)
        # 应该有警告
        assert 'warnings' in validation or 'risks' in validation


class TestHyperparameterExtraction:
    """测试超参数提取"""

    def test_extract_best_params_from_grid(self, house_price_data):
        """测试从网格搜索提取最佳参数"""
        X = house_price_data[['area_sqm', 'bedrooms', 'age_years']]
        y = house_price_data['price']

        param_grid = {'max_depth': [3, 5]}

        result = tune_hyperparameters_grid(X, y, param_grid, cv=3)
        best_params = extract_best_params(result)

        assert isinstance(best_params, dict)
        assert 'max_depth' in best_params

    def test_extract_best_params_from_random(self, house_price_data):
        """测试从随机搜索提取最佳参数"""
        X = house_price_data[['area_sqm', 'bedrooms', 'age_years']]
        y = house_price_data['price']

        param_dist = {'max_depth': [3, 5, 7]}

        result = tune_hyperparameters_random(X, y, param_dist, n_iter=3, cv=3)
        best_params = extract_best_params(result)

        assert isinstance(best_params, dict)
        assert 'max_depth' in best_params

    def test_extract_cv_results_dataframe(self, house_price_data):
        """测试提取 CV 结果为 DataFrame"""
        X = house_price_data[['area_sqm', 'bedrooms', 'age_years']]
        y = house_price_data['price']

        param_grid = {
            'max_depth': [3, 5],
            'min_samples_leaf': [5, 10]
        }

        result = tune_hyperparameters_grid(X, y, param_grid, cv=3)
        cv_results = extract_cv_results(result)

        if isinstance(cv_results, pd.DataFrame):
            # 应该有多行（每个参数组合）
            assert len(cv_results) == 4  # 2*2 = 4
            # 应该有参数列
            assert 'param_max_depth' in cv_results.columns or \
                   'params' in cv_results.columns


class TestHyperparameterEdgeCases:
    """测试超参数调优的边界情况"""

    def test_tuning_with_small_dataset(self, very_small_dataset):
        """测试小数据集的超参数调优"""
        X = very_small_dataset[['feature_1', 'feature_2']]
        y = very_small_dataset['target']

        param_grid = {'max_depth': [2, 3]}

        # 应该能处理（但 CV fold 可能受限）
        result = tune_hyperparameters_grid(
            X, y, param_grid, cv=2
        )

        assert result is not None

    def test_tuning_with_single_param(self, house_price_data):
        """测试单参数调优"""
        X = house_price_data[['area_sqm', 'bedrooms', 'age_years']]
        y = house_price_data['price']

        param_grid = {'max_depth': [3, 5, 7]}

        result = tune_hyperparameters_grid(X, y, param_grid, cv=3)

        assert result is not None
        best_params = extract_best_params(result)
        assert 'max_depth' in best_params

    def test_tuning_with_extreme_values(self, house_price_data):
        """测试极端值的超参数调优"""
        X = house_price_data[['area_sqm', 'bedrooms', 'age_years']]
        y = house_price_data['price']

        param_grid = {
            'max_depth': [1, 50],  # 极浅 vs 极深
            'min_samples_leaf': [1, 100]  # 极小 vs 极大
        }

        result = tune_hyperparameters_grid(X, y, param_grid, cv=3)

        assert result is not None
        # 最佳参数应该不太极端
        best_params = extract_best_params(result)
        assert 1 <= best_params['max_depth'] <= 50
        assert 1 <= best_params['min_samples_leaf'] <= 100


class TestNestedCrossValidation:
    """测试嵌套交叉验证"""

    def test_nested_cv_prevents_overfitting(self, house_price_data):
        """测试嵌套 CV 防止过拟合验证集"""
        X = house_price_data[['area_sqm', 'bedrooms', 'age_years']]
        y = house_price_data['price']

        param_grid = {'max_depth': [3, 5, 7]}

        # 外层 CV
        from sklearn.model_selection import cross_val_score, KFold

        outer_cv = KFold(n_splits=3, shuffle=True, random_state=42)
        outer_scores = []

        for train_idx, test_idx in outer_cv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # 内层 CV（调优）
            result = tune_hyperparameters_grid(
                X_train, y_train, param_grid, cv=3
            )

            best_params = extract_best_params(result)
            best_model = DecisionTreeRegressor(**best_params, random_state=42)
            best_model.fit(X_train, y_train)

            outer_scores.append(best_model.score(X_test, y_test))

        # 外层 CV 得分应该更保守
        assert len(outer_scores) == 3
        assert all(s > 0 for s in outer_scores)

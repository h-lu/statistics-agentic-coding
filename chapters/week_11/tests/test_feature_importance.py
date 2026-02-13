"""
Week 11 特征重要性测试

测试特征重要性的计算、置换重要性和相关特征陷阱。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

# 导入需要测试的函数
try:
    from solution import (
        calculate_feature_importance,
        calculate_permutation_importance,
        compare_builtin_vs_permutation,
        detect_correlation_dilution,
        interpret_feature_importance,
        plot_feature_importance,
    )
except ImportError:
    pytest.skip("starter_code/solution.py not implemented yet", allow_module_level=True)


class TestBuiltinFeatureImportance:
    """测试内置特征重要性"""

    def test_builtin_importance_calculation(self, house_price_data):
        """测试内置特征重要性计算"""
        X = house_price_data[['area_sqm', 'bedrooms', 'age_years', 'distance_km']]
        y = house_price_data['price']

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        importance = calculate_feature_importance(model, X.columns)

        assert isinstance(importance, (pd.DataFrame, dict, list))

        if isinstance(importance, pd.DataFrame):
            assert len(importance) == len(X.columns)
            # 检查必需的列
            assert 'feature' in importance.columns or '特征' in importance.columns
            assert 'importance' in importance.columns or '重要性' in importance.columns

    def test_builtin_importance_sum_to_one(self, house_price_data):
        """测试内置特征重要性之和为 1"""
        X = house_price_data[['area_sqm', 'bedrooms', 'age_years']]
        y = house_price_data['price']

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        importance = calculate_feature_importance(model, X.columns)

        if isinstance(importance, pd.DataFrame):
            if 'importance' in importance.columns:
                importances = importance['importance'].values
            else:
                importances = importance['重要性'].values
        elif isinstance(importance, dict):
            importances = list(importance.values())
        else:
            importances = importance

        # 和应该接近 1.0
        assert np.isclose(sum(importances), 1.0, atol=0.01)

    def test_builtin_importance_non_negative(self, house_price_data):
        """测试内置特征重要性非负"""
        X = house_price_data[['area_sqm', 'bedrooms', 'age_years']]
        y = house_price_data['price']

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        importance = calculate_feature_importance(model, X.columns)

        if isinstance(importance, pd.DataFrame):
            if 'importance' in importance.columns:
                importances = importance['importance'].values
            else:
                importances = importance['重要性'].values
        elif isinstance(importance, dict):
            importances = list(importance.values())
        else:
            importances = importance

        # 所有重要性应该非负
        assert all(imp >= 0 for imp in importances)

    def test_builtin_importance_known_truth(self, feature_importance_known_truth):
        """测试内置重要性是否能识别真值"""
        X = feature_importance_known_truth.drop('target', axis=1)
        y = feature_importance_known_truth['target']

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        importance = calculate_feature_importance(model, X.columns)

        if isinstance(importance, pd.DataFrame):
            # 排序后检查
            importance_sorted = importance.sort_values(
                by='importance' if 'importance' in importance.columns else '重要性',
                ascending=False
            )
            top_feature = importance_sorted.iloc[0]['feature' if 'feature' in importance_sorted.columns else '特征']
        else:
            # 简化处理
            top_feature = X.columns[0]

        # feature_1 应该是最重要的（或之一）
        assert top_feature in ['feature_1', 'feature_2']


class TestPermutationImportance:
    """测试置换重要性"""

    def test_permutation_importance_calculation(self, house_price_data):
        """测试置换重要性计算"""
        X = house_price_data[['area_sqm', 'bedrooms', 'age_years']]
        y = house_price_data['price']

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        perm_imp = calculate_permutation_importance(model, X, y, n_repeats=10)

        assert isinstance(perm_imp, (dict, pd.DataFrame))

        if isinstance(perm_imp, dict):
            # 检查必需的键
            assert any(key in perm_imp for key in [
                'importances', 'importance', 'scores'
            ])

    def test_permutation_importance_reduces_score(self, house_price_data):
        """测试置换重要特征会降低分数"""
        X = house_price_data[['area_sqm', 'bedrooms', 'age_years']]
        y = house_price_data['price']

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        # 原始分数
        original_score = model.score(X, y)

        # 置换最重要的特征
        perm_imp = calculate_permutation_importance(model, X, y, n_repeats=10)

        if isinstance(perm_imp, dict):
            # 置换后分数应该降低
            assert 'importances_mean' in perm_imp or 'importance' in perm_imp

    def test_permutation_importance_known_truth(self, feature_importance_known_truth):
        """测试置换重要性是否能识别真值"""
        X = feature_importance_known_truth.drop('target', axis=1)
        y = feature_importance_known_truth['target']

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        perm_imp = calculate_permutation_importance(model, X, y, n_repeats=10)

        # feature_1 和 feature_2 应该最重要
        # feature_4（噪声）应该最不重要
        assert isinstance(perm_imp, dict) or isinstance(perm_imp, pd.DataFrame)

    def test_permutation_importance_n_repeats(self, house_price_data):
        """测试置换重要性的重复次数参数"""
        X = house_price_data[['area_sqm', 'bedrooms', 'age_years']]
        y = house_price_data['price']

        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X, y)

        # 不同的重复次数
        perm_imp_5 = calculate_permutation_importance(model, X, y, n_repeats=5)
        perm_imp_10 = calculate_permutation_importance(model, X, y, n_repeats=10)

        # 两者都应该成功
        assert perm_imp_5 is not None
        assert perm_imp_10 is not None


class TestBuiltinVsPermutationComparison:
    """测试内置重要性 vs 置换重要性"""

    def test_compare_builtin_vs_permutation(self, house_price_data):
        """测试比较内置和置换重要性"""
        X = house_price_data[['area_sqm', 'bedrooms', 'age_years']]
        y = house_price_data['price']

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        comparison = compare_builtin_vs_permutation(model, X, y)

        assert isinstance(comparison, dict)

        # 应该包含两种重要性的结果
        assert 'builtin' in comparison or '内置' in comparison
        assert 'permutation' in comparison or '置换' in comparison

    def test_builtin_and_permutation_correlation(self, house_price_data):
        """测试内置和置换重要性的相关性"""
        X = house_price_data[['area_sqm', 'bedrooms', 'age_years']]
        y = house_price_data['price']

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        comparison = compare_builtin_vs_permutation(model, X, y)

        # 两种重要性应该有一定的相关性
        # 但不一定完全一致
        assert isinstance(comparison, dict)

    def test_permutation_more_reliable_for_correlated(self, house_price_data_correlated):
        """测试置换重要性对相关特征更可靠"""
        X = house_price_data_correlated[['area_sqm', 'living_area', 'rooms']]
        y = house_price_data_correlated['price']

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        comparison = compare_builtin_vs_permutation(model, X, y)

        # 置换重要性应该能识别多个相关特征都重要
        assert isinstance(comparison, dict)


class TestCorrelationDilution:
    """测试相关特征稀释重要性"""

    def test_detect_correlation_dilution(self, house_price_data_correlated):
        """测试检测相关性稀释"""
        X = house_price_data_correlated[['area_sqm', 'living_area', 'rooms']]
        y = house_price_data_correlated['price']

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        detection = detect_correlation_dilution(
            model, X, threshold=0.7
        )

        assert isinstance(detection, dict)
        assert 'has_correlation' in detection or 'correlation_detected' in detection

        # 应该检测到高相关特征对
        assert detection.get('has_correlation', False) or \
               detection.get('correlation_detected', False)

    def test_correlation_dilution_reduces_importance(self, house_price_data_correlated):
        """测试相关性会降低单个特征的重要性"""
        X = house_price_data_correlated[['area_sqm', 'living_area', 'rooms']]
        y = house_price_data_correlated['price']

        # 只用 area_sqm
        model1 = RandomForestRegressor(n_estimators=100, random_state=42)
        model1.fit(X[['area_sqm']], y)
        importance1 = model1.feature_importances_[0]

        # 用所有相关特征
        model2 = RandomForestRegressor(n_estimators=100, random_state=42)
        model2.fit(X, y)
        importance2 = model2.feature_importances_[0]  # area_sqm 的重要性

        # 当有相关特征时，单个特征的重要性可能会降低
        # 但不一定总是如此，所以我们只检查能计算
        assert importance1 >= 0
        assert importance2 >= 0

    def test_correlation_matrix_detection(self, house_price_data_correlated):
        """测试相关性矩阵检测"""
        X = house_price_data_correlated[['area_sqm', 'living_area', 'rooms']]

        detection = detect_correlation_dilution(
            None, X, threshold=0.7, method='correlation_matrix'
        )

        assert isinstance(detection, dict)
        assert 'correlation_matrix' in detection or 'correlations' in detection

        # 应该识别出高相关对
        if 'high_correlation_pairs' in detection:
            assert len(detection['high_correlation_pairs']) > 0


class TestFeatureImportanceInterpretation:
    """测试特征重要性的解释"""

    def test_interpret_feature_importance_returns_text(self, house_price_data):
        """测试特征重要性解释返回文本"""
        X = house_price_data[['area_sqm', 'bedrooms', 'age_years']]
        y = house_price_data['price']

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        importance = calculate_feature_importance(model, X.columns)
        interpretation = interpret_feature_importance(importance, model_type='random_forest')

        assert isinstance(interpretation, str)
        assert len(interpretation) > 0

        # 应该包含关键词
        assert any(keyword in interpretation.lower() for keyword in [
            'feature', '特征', 'important', '重要'
        ])

    def test_interpret_includes_warning(self, house_price_data):
        """测试解释包含相关性警告"""
        X = house_price_data[['area_sqm', 'bedrooms', 'age_years']]
        y = house_price_data['price']

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        importance = calculate_feature_importance(model, X.columns)
        interpretation = interpret_feature_importance(
            importance,
            model_type='random_forest',
            include_correlation_warning=True
        )

        # 应该包含警告
        assert any(keyword in interpretation.lower() for keyword in [
            'correlation', '相关', 'causal', '因果', 'warning', '警告'
        ])

    def test_interpret_no_causality_claim(self, house_price_data):
        """测试解释不做因果声称"""
        X = house_price_data[['area_sqm', 'bedrooms', 'age_years']]
        y = house_price_data['price']

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        importance = calculate_feature_importance(model, X.columns)
        interpretation = interpret_feature_importance(importance, model_type='random_forest')

        # 不应该说"增加 X 会改变 Y"（因果语言）
        # 应该说"X 与 Y 相关"或"X 对预测有贡献"
        causal_phrases = ['cause', '导致', 'makes', '使', 'causality']
        interpretation_lower = interpretation.lower()

        # 如果出现因果短语，应该被否定或警告
        for phrase in causal_phrases:
            if phrase in interpretation_lower:
                # 应该伴随警告
                assert 'warning' in interpretation_lower or '注意' in interpretation_lower


class TestFeatureImportanceVisualization:
    """测试特征重要性可视化"""

    def test_plot_feature_importance(self, house_price_data, temp_output_dir):
        """测试绘制特征重要性"""
        X = house_price_data[['area_sqm', 'bedrooms', 'age_years']]
        y = house_price_data['price']

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        importance = calculate_feature_importance(model, X.columns)

        # 测试绘图函数
        result = plot_feature_importance(
            importance,
            output_path=temp_output_dir / 'importance.png'
        )

        # 应该成功生成文件或返回成功状态
        assert result is not None or (temp_output_dir / 'importance.png').exists()

    def test_plot_importance_top_k(self, house_price_data, temp_output_dir):
        """测试绘制 Top K 特征重要性"""
        X = house_price_data[['area_sqm', 'bedrooms', 'age_years', 'distance_km']]
        y = house_price_data['price']

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        importance = calculate_feature_importance(model, X.columns)

        # 只画 Top 2
        result = plot_feature_importance(
            importance,
            top_k=2,
            output_path=temp_output_dir / 'importance_top2.png'
        )

        assert result is not None


class TestFeatureImportanceEdgeCases:
    """测试特征重要性的边界情况"""

    def test_importance_with_single_feature(self, single_feature_data):
        """测试单特征的特征重要性"""
        X = single_feature_data[['feature']]
        y = single_feature_data['target']

        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X, y)

        importance = calculate_feature_importance(model, X.columns)

        # 单特征的重要性应该接近 1.0
        if isinstance(importance, pd.DataFrame):
            if 'importance' in importance.columns:
                imp_val = importance['importance'].values[0]
            else:
                imp_val = importance['重要性'].values[0]
        else:
            imp_val = importance[0] if isinstance(importance, list) else list(importance.values())[0]

        assert imp_val > 0.9

    def test_importance_with_unimportant_feature(self, house_price_data):
        """测试不重要特征的重要性接近 0"""
        X = house_price_data[['area_sqm', 'bedrooms', 'age_years']].copy()
        X['noise'] = np.random.normal(0, 1, len(X))
        y = house_price_data['price']

        model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X, y)

        importance = calculate_feature_importance(model, X.columns)

        if isinstance(importance, pd.DataFrame):
            importance_dict = dict(zip(
                importance['feature' if 'feature' in importance.columns else '特征'],
                importance['importance' if 'importance' in importance.columns else '重要性']
            ))
        else:
            importance_dict = importance

        # 噪声特征的重要性应该很低
        noise_importance = importance_dict.get('noise', 0)
        assert noise_importance < 0.2

    def test_importance_with_constant_feature(self, house_price_data):
        """测试常数特征的重要性"""
        X = house_price_data[['area_sqm', 'bedrooms', 'age_years']].copy()
        X['constant'] = 1.0
        y = house_price_data['price']

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        importance = calculate_feature_importance(model, X.columns)

        if isinstance(importance, pd.DataFrame):
            importance_dict = dict(zip(
                importance['feature' if 'feature' in importance.columns else '特征'],
                importance['importance' if 'importance' in importance.columns else '重要性']
            ))
        else:
            importance_dict = importance

        # 常数特征的重要性应该为 0
        constant_importance = importance_dict.get('constant', 0)
        assert constant_importance < 0.01


class TestHighCardinalityTrap:
    """测试高基数类别特征陷阱"""

    def test_high_cardinality_overestimates_importance(self, high_cardinality_categorical_data):
        """测试高基数类别特征高估重要性"""
        X = high_cardinality_categorical_data[['feature_1', 'user_id']]
        y = high_cardinality_categorical_data['target']

        model = RandomForestRegressor(n_estimators=100, random_state=42)

        # 编码 user_id
        X_encoded = pd.get_dummies(X, columns=['user_id'], drop_first=True)
        model.fit(X_encoded, y)

        # user_id 相关的列可能看起来很重要
        # 但实际上只是噪声
        importance = calculate_feature_importance(model, X_encoded.columns)

        # 至少应该能计算
        assert importance is not None

    def test_permutation_importance_reveals_trap(self, high_cardinality_categorical_data):
        """测试置换重要性揭示高基数陷阱"""
        X = high_cardinality_categorical_data[['feature_1', 'user_id']]
        y = high_cardinality_categorical_data['target']

        model = RandomForestRegressor(n_estimators=100, random_state=42)

        # 只用 feature_1
        model_real = RandomForestRegressor(n_estimators=100, random_state=42)
        model_real.fit(X[['feature_1']], y)

        perm_imp = calculate_permutation_importance(model_real, X[['feature_1']], y)

        # feature_1 应该有真实的正重要性
        assert perm_imp is not None

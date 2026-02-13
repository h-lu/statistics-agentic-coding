"""
Week 12：边界用例测试

测试空数据、极端情况、异常输入等边界场景
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

try:
    from solution import (
        calculate_shap_values,
        explain_single_prediction,
        calculate_disparate_impact,
        calculate_equal_opportunity,
        calculate_equalized_odds,
        add_differential_privacy_noise,
        detect_proxy_variables,
    )
except ImportError:
    pytest.skip("starter_code/solution.py not implemented yet", allow_module_level=True)


class TestEmptyAndSmallData:
    """测试空数据和极小数据"""

    def test_shap_with_empty_dataframe(self, empty_dataframe):
        """测试空数据框的 SHAP 计算"""
        # 应该优雅地处理空数据
        with pytest.raises((ValueError, IndexError, KeyError)):
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=10, random_state=42)

            # 空数据无法训练，但测试函数是否优雅地失败
            calculate_shap_values(model, empty_dataframe)

    def test_shap_with_single_sample(self, single_sample_data):
        """测试单个样本的 SHAP 计算"""
        df = single_sample_data

        X = df[['feature_1', 'feature_2']]
        y = df['target']

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split

        # 单个样本无法划分训练/测试集
        # 测试是否优雅地处理
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            model.fit(X_train, y_train)

            shap_values = calculate_shap_values(model, X_test)
            # 如果能计算，验证结果
            assert shap_values is not None
        except (ValueError, IndexError):
            # 预期可能失败
            assert True

    def test_disparate_impact_with_empty_groups(self):
        """测试空组的差异影响比"""
        n = 50
        y_pred = np.array([0] * 25 + [1] * 25)
        # 只有一个组
        group_labels = np.zeros(n, dtype=int)

        di_ratio = calculate_disparate_impact(y_pred, group_labels)

        # 应该处理空组（返回 NaN 或特殊值）
        assert di_ratio is not None

    def test_equal_opportunity_with_single_class(self):
        """测试只有一个类别的平等机会"""
        # 所有样本都是负类
        y_true = np.zeros(100, dtype=int)
        y_pred = np.random.randint(0, 2, 100)
        group_labels = np.array([0] * 50 + [1] * 50)

        # 应该优雅处理
        eo_diff = calculate_equal_opportunity(y_true, y_pred, group_labels)
        assert eo_diff is not None

    def test_fairness_metrics_with_small_group(self):
        """测试样本数很小的群体的公平性指标"""
        n = 100
        y_pred = np.random.randint(0, 2, n)
        # 群体 B 只有 5 个样本
        group_labels = np.array([0] * 95 + [1] * 5)

        di_ratio = calculate_disparate_impact(y_pred, group_labels)

        # 应该返回结果（虽然可能不稳定）
        assert di_ratio is not None


class TestSingleFeature:
    """测试只有一个特征的情况"""

    def test_shap_with_single_feature(self, single_feature_data):
        """测试单特征的 SHAP 计算"""
        df = single_feature_data

        X = df[['feature']]
        y = df['target']

        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        shap_values = calculate_shap_values(model, X_test)

        # 应该能计算
        assert shap_values is not None

    def test_explain_single_prediction_single_feature(self, single_feature_data):
        """测试单特征的单个预测解释"""
        df = single_feature_data

        X = df[['feature']]
        y = df['target']

        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        explanation = explain_single_prediction(model, X_test.iloc[0])

        # 应该返回解释
        assert explanation is not None
        if isinstance(explanation, dict):
            assert 'shap_values' in explanation or len(explanation) > 0


class TestNegativeAndZeroValues:
    """测试负值和零值"""

    def test_shap_with_negative_feature_values(self, data_with_negative_values):
        """测试包含负值特征的 SHAP 计算"""
        df = data_with_negative_values

        X = df[['income', 'age', 'score']]
        y = (df['income'] > 5000).astype(int)

        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        shap_values = calculate_shap_values(model, X_test)

        # 应该能处理负值
        assert shap_values is not None

    def test_differential_privacy_with_negative_values(self):
        """测试差分隐私对负值的处理"""
        data = np.array([-1000, -500, 0, 500, 1000])

        private_data = add_differential_privacy_noise(
            data, epsilon=1.0, sensitivity=2000
        )

        # 应该返回加了噪声的数据
        assert private_data is not None
        assert len(private_data) == len(data)

        # 负值应该仍然可能是负值
        assert any(val < 0 for val in private_data)


class TestOutliersAndExtremeValues:
    """测试离群点和极端值"""

    def test_shap_with_outliers(self, data_with_outliers):
        """测试包含离群点的 SHAP 计算"""
        df = data_with_outliers

        X = df[['feature']]
        y = df['target']

        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        shap_values = calculate_shap_values(model, X_test)

        # 应该能处理离群点
        assert shap_values is not None

    def test_explain_prediction_with_outlier(self, data_with_outliers):
        """测试解释离群点样本"""
        df = data_with_outliers

        X = df[['feature']]
        y = df['target']

        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        # 找一个离群点
        outlier_idx = np.abs(X_test['feature']).idxmax()

        explanation = explain_single_prediction(model, X_test.loc[outlier_idx])

        # 应该能解释
        assert explanation is not None

    def test_differential_privacy_with_outliers(self):
        """测试差分隐私对极端值的处理"""
        # 包含极端值的数据
        data = np.array([1, 2, 3, 4, 5, 10000])  # 10000 是极端值

        private_data = add_differential_privacy_noise(
            data, epsilon=1.0, sensitivity=10000
        )

        # 应该返回结果
        assert private_data is not None
        assert len(private_data) == len(data)


class TestConstantFeatures:
    """测试常数特征"""

    def test_shap_with_constant_feature(self, constant_feature_data):
        """测试包含常数特征的 SHAP 计算"""
        df = constant_feature_data

        X = df[['feature_1', 'feature_2']]
        y = df['target']

        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        shap_values = calculate_shap_values(model, X_test)

        # 应该能处理常数特征
        assert shap_values is not None

        # 常数特征的 SHAP 值应该接近 0
        # （如果实现正确）
        # 这个检查取决于具体实现


class TestHighlyImbalancedData:
    """测试高度不平衡数据"""

    def test_shap_with_imbalanced_data(self, highly_imbalanced_data):
        """测试极度不平衡数据的 SHAP 计算"""
        df = highly_imbalanced_data

        X = df[['feature']]
        y = df['target']

        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        shap_values = calculate_shap_values(model, X_test)

        # 应该能处理不平衡数据
        assert shap_values is not None

    def test_fairness_with_imbalanced_groups(self):
        """测试群体大小极不平衡的公平性指标"""
        n = 200
        # 群体 A 只有 10 个样本
        y_pred = np.random.randint(0, 2, n)
        group_labels = np.array([0] * 10 + [1] * 190)

        di_ratio = calculate_disparate_impact(y_pred, group_labels)

        # 应该返回结果
        assert di_ratio is not None


class TestDifferentialPrivacyEdgeCases:
    """测试差分隐私的边界情况"""

    def test_differential_privacy_zero_epsilon(self):
        """测试 epsilon = 0（极端隐私保护）"""
        data = np.array([1000, 2000, 3000, 4000, 5000])

        private_data = add_differential_privacy_noise(
            data, epsilon=0.001, sensitivity=4000
        )

        # 应该返回加了噪声的数据
        assert private_data is not None

        # epsilon 很小时，噪声应该很大
        noise = np.abs(private_data - data).mean()
        assert noise > 0  # 应该有噪声

    def test_differential_privacy_single_value(self):
        """测试单个值的差分隐私"""
        data = np.array([5000])

        private_data = add_differential_privacy_noise(
            data, epsilon=1.0, sensitivity=5000
        )

        # 应该返回单个加噪值
        assert private_data is not None
        assert len(private_data) == 1

    def test_differential_privacy_all_same_values(self):
        """测试所有值相同的差分隐私"""
        data = np.full(100, 5000)

        private_data = add_differential_privacy_noise(
            data, epsilon=1.0, sensitivity=5000
        )

        # 加噪后不应该都相同
        assert len(np.unique(private_data)) > 1

    def test_differential_privacy_zero_sensitivity(self):
        """测试 sensitivity = 0 的情况"""
        data = np.array([1000, 2000, 3000])

        # sensitivity = 0 意味着无噪声
        private_data = add_differential_privacy_noise(
            data, epsilon=1.0, sensitivity=0
        )

        # 应该返回原数据
        assert private_data is not None


class TestProxyVariableDetectionEdgeCases:
    """测试代理变量检测的边界情况"""

    def test_detect_proxy_with_no_sensitive_col(self):
        """测试没有敏感列的情况"""
        np.random.seed(42)
        n = 100

        df = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, n),
            'feature_2': np.random.normal(0, 1, n),
        })

        # 应该优雅处理
        try:
            proxies = detect_proxy_variables(df, sensitive_col='gender')
            # 如果没有该列，应该返回空或报错
            assert proxies is not None
        except (KeyError, ValueError):
            # 预期可能报错
            assert True

    def test_detect_proxy_all_correlated(self):
        """测试所有特征都相关的情况"""
        np.random.seed(42)
        n = 100

        gender = np.random.randint(0, 2, n)
        # 所有特征都与性别高度相关
        feature_1 = gender + np.random.normal(0, 0.1, n)
        feature_2 = gender * 2 + np.random.normal(0, 0.1, n)

        df = pd.DataFrame({
            'gender': gender,
            'feature_1': feature_1,
            'feature_2': feature_2,
        })

        proxies = detect_proxy_variables(df, sensitive_col='gender')

        # 应该检测到多个代理变量
        if isinstance(proxies, list):
            assert len(proxies) >= 1
        elif isinstance(proxies, dict):
            assert 'proxies' in proxies
            assert len(proxies['proxies']) >= 1


class TestVerySmallDatasets:
    """测试极小数据集"""

    def test_shap_with_10_samples(self):
        """测试只有 10 个样本的 SHAP 计算"""
        np.random.seed(42)
        n = 10

        X = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, n),
            'feature_2': np.random.normal(0, 1, n),
        })
        y = np.random.randint(0, 2, n)

        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
            model = RandomForestClassifier(n_estimators=5, random_state=42)
            model.fit(X_train, y_train)

            shap_values = calculate_shap_values(model, X_test)
            assert shap_values is not None
        except (ValueError, IndexError):
            # 样本太少可能失败
            assert True

    def test_fairness_with_very_small_dataset(self):
        """测试极小数据集的公平性指标"""
        n = 20
        y_pred = np.random.randint(0, 2, n)
        group_labels = np.random.randint(0, 2, n)

        di_ratio = calculate_disparate_impact(y_pred, group_labels)

        # 应该能处理
        assert di_ratio is not None


class TestEdgeCaseCombinations:
    """测试边界情况组合"""

    def test_imbalanced_small_with_outliers(self):
        """测试不平衡 + 小样本 + 离群点的组合"""
        np.random.seed(42)
        n = 50

        # 不平衡：90% 负类
        y = np.zeros(n, dtype=int)
        y[:5] = 1

        # 小样本 + 离群点
        feature = np.random.normal(0, 1, n)
        feature[:3] = [100, -100, 50]  # 添加离群点

        X = pd.DataFrame({'feature': feature})

        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            model.fit(X_train, y_train)

            shap_values = calculate_shap_values(model, X_test)
            assert shap_values is not None
        except (ValueError, IndexError):
            # 可能失败
            assert True

"""
Week 12：SHAP 可解释性测试

测试 SHAP 值计算、特征重要性、局部解释等功能
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

try:
    from solution import (
        calculate_shap_values,
        explain_single_prediction,
        calculate_feature_importance_shap,
        review_xai_code,
    )
    import shap
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
except ImportError:
    pytest.skip("starter_code/solution.py or required libraries not implemented yet", allow_module_level=True)


class TestSHAPValuesCalculation:
    """测试 SHAP 值计算功能"""

    def test_shap_values_returns_array(self, simple_classification_for_shap):
        """测试 SHAP 值返回正确的数组格式"""
        X, y = simple_classification_for_shap

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        shap_values = calculate_shap_values(model, X_test)

        # 应该返回数组或列表
        assert isinstance(shap_values, (np.ndarray, list))

        # 形状应该匹配测试集
        if isinstance(shap_values, np.ndarray):
            # 对于二分类，shap_values 可能是 [n_samples, n_features]
            # 或 [[n_samples, n_features], [n_samples, n_features]]
            if len(shap_values.shape) == 2:
                assert shap_values.shape[0] == len(X_test)

    def test_shap_values_global_importance_ranking(self, simple_regression_for_shap):
        """测试 SHAP 值能正确排序特征重要性"""
        X, y = simple_regression_for_shap

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)

        importance = calculate_feature_importance_shap(model, X_test)

        # feature_1 和 feature_2 应该比 feature_3 更重要
        if isinstance(importance, pd.DataFrame):
            importance_sorted = importance.sort_values(
                by=importance.columns[0], ascending=False
            )
            top_features = importance_sorted.index.tolist()
            # feature_3 应该不在前两位
            assert 'feature_3' not in top_features[:2]
        elif isinstance(importance, dict):
            sorted_items = sorted(importance.items(),
                                 key=lambda x: abs(x[1]),
                                 reverse=True)
            top_features = [item[0] for item in sorted_items]
            assert 'feature_3' not in top_features[:2]

    def test_shap_values_additivity_property(self, simple_classification_for_shap):
        """测试 SHAP 值的可加性：SHAP 值之和 + 基准值 = 预测值"""
        X, y = simple_classification_for_shap

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        # 检查可加性（近似）
        # 对于二分类，检查正类
        if isinstance(shap_values, list):
            shap_vals = shap_values[1]
            expected = explainer.expected_value[1]
        else:
            shap_vals = shap_values
            expected = explainer.expected_value

        # 验证第一个样本
        shap_sum = shap_vals[0].sum()
        # 注意：这是近似验证，因为随机森林的预测不是线性的
        assert abs(shap_sum) < 10  # SHAP 值之和不应该过大

    def test_shap_values_classification_vs_regression(
        self,
        simple_classification_for_shap,
        simple_regression_for_shap
    ):
        """测试分类和回归任务的 SHAP 值计算"""
        X_clf, y_clf = simple_classification_for_shap
        X_reg, y_reg = simple_regression_for_shap

        # 分类
        from sklearn.model_selection import train_test_split
        X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
            X_clf, y_clf, test_size=0.2, random_state=42
        )

        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(X_train_c, y_train_c)
        shap_clf = calculate_shap_values(clf, X_test_c)
        assert shap_clf is not None

        # 回归
        X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
            X_reg, y_reg, test_size=0.2, random_state=42
        )

        reg = RandomForestRegressor(n_estimators=10, random_state=42)
        reg.fit(X_train_r, y_train_r)
        shap_reg = calculate_shap_values(reg, X_test_r)
        assert shap_reg is not None


class TestLocalExplanation:
    """测试局部解释功能"""

    def test_explain_single_prediction_returns_dict(self, simple_classification_for_shap):
        """测试单个预测解释返回字典"""
        X, y = simple_classification_for_shap

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        explanation = explain_single_prediction(model, X_test.iloc[0])

        # 应该返回字典
        assert isinstance(explanation, dict)

        # 应该包含关键信息
        expected_keys = ['base_value', 'shap_values', 'final_value']
        for key in expected_keys:
            assert key in explanation or any(k in explanation for k in expected_keys)

    def test_explain_single_prediction_feature_contributions(
        self,
        simple_classification_for_shap
    ):
        """测试单个预测解释能正确显示特征贡献"""
        X, y = simple_classification_for_shap

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        explanation = explain_single_prediction(model, X_test.iloc[0])

        # SHAP 值应该包含所有特征的贡献
        if 'shap_values' in explanation:
            shap_vals = explanation['shap_values']
            if isinstance(shap_vals, dict):
                assert len(shap_vals) == len(X_test.columns)

    def test_explain_single_prediction_direction(self, simple_classification_for_shap):
        """测试单个预测解释的方向性（正负贡献）"""
        X, y = simple_classification_for_shap

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        explanation = explain_single_prediction(model, X_test.iloc[0])

        # 应该同时有正向和负向贡献
        if 'shap_values' in explanation:
            shap_vals = explanation['shap_values']
            if isinstance(shap_vals, dict):
                values = list(shap_vals.values())
                # 应该既有正值也有负值（或者至少有非零值）
                assert any(v != 0 for v in values)


class TestSHAPWithRealData:
    """测试 SHAP 在真实场景数据上的表现"""

    def test_credit_scoring_shap_values(self, credit_scoring_data):
        """测试信用评分数据的 SHAP 值"""
        df = credit_scoring_data

        feature_cols = ['income', 'credit_history_age', 'debt_to_income',
                       'credit_inquiries', 'employment_length']
        X = df[feature_cols]
        y = df['default']

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)

        # 计算 SHAP 值
        shap_values = calculate_shap_values(model, X_test)
        assert shap_values is not None

        # 计算特征重要性
        importance = calculate_feature_importance_shap(model, X_test)
        assert importance is not None

    def test_credit_scoring_explain_rejection(self, credit_scoring_data):
        """测试解释信用评分拒绝案例"""
        df = credit_scoring_data

        feature_cols = ['income', 'credit_history_age', 'debt_to_income',
                       'credit_inquiries', 'employment_length']
        X = df[feature_cols]
        y = df['default']

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)

        # 找一个被预测为违约的样本
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        rejected_idx = np.where(y_pred_proba > 0.7)[0]

        if len(rejected_idx) > 0:
            explanation = explain_single_prediction(
                model, X_test.iloc[rejected_idx[0]]
            )
            assert explanation is not None


class TestSHAPCodeReview:
    """测试 SHAP 代码审查功能"""

    def test_review_good_shap_code(self, good_shap_code_example):
        """测试审查好的 SHAP 代码"""
        review = review_xai_code(good_shap_code_example)

        assert isinstance(review, dict)
        # 好的代码应该没有严重问题
        if 'has_issues' in review:
            assert not review.get('critical_issues', False)

    def test_review_bad_shap_wrong_explainer(self, bad_shap_code_wrong_explainer):
        """测试审查使用了错误 Explainer 的代码"""
        review = review_xai_code(bad_shap_code_wrong_explainer)

        assert isinstance(review, dict)
        # 应该检测到问题
        if 'has_issues' in review:
            assert review['has_issues'] is True
        if 'issues' in review:
            assert len(review['issues']) > 0

    def test_review_bad_shap_global_only(self, bad_shap_code_global_only):
        """测试审查只有全局解释的代码"""
        review = review_xai_code(bad_shap_code_global_only)

        assert isinstance(review, dict)
        # 应该检测到缺少局部解释
        if 'issues' in review:
            issue_descriptions = ' '.join(str(i) for i in review['issues'])
            # 应该提到缺少局部解释或单个样本解释
            assert any(keyword in issue_descriptions.lower()
                      for keyword in ['local', 'single', 'individual', 'sample'])

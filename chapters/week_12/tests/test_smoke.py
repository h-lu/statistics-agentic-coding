"""
Week 12 烟雾测试（Smoke Test）

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
        calculate_shap_values,
        explain_single_prediction,
        calculate_feature_importance_shap,
        detect_proxy_variables,
        calculate_disparate_impact,
        calculate_equal_opportunity,
        calculate_equalized_odds,
        add_differential_privacy_noise,
        review_xai_code,
        explain_to_nontechnical,
    )
except ImportError:
    pytest.skip("starter_code/solution.py not implemented yet", allow_module_level=True)


class TestSmokeBasicFunctionality:
    """测试基本功能是否可以运行"""

    @pytest.fixture
    def sample_classification_data(self):
        """创建分类测试数据"""
        np.random.seed(42)
        n = 100

        X = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, n),
            'feature_2': np.random.normal(0, 1, n),
            'feature_3': np.random.normal(0, 1, n),
        })

        logit = -1 + 2 * X['feature_1'] + X['feature_2']
        prob = 1 / (1 + np.exp(-logit))
        y = (np.random.random(n) < prob).astype(int)

        return X, y

    def test_smoke_calculate_shap_values(self, sample_classification_data):
        """烟雾测试：SHAP 值计算"""
        X, y = sample_classification_data

        # 训练一个简单的随机森林
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        # 计算 SHAP 值
        shap_values = calculate_shap_values(model, X)

        # 应该返回 SHAP 值
        assert shap_values is not None
        assert isinstance(shap_values, (np.ndarray, list))

    def test_smoke_explain_single_prediction(self, sample_classification_data):
        """烟雾测试：单个预测解释"""
        X, y = sample_classification_data

        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        # 解释第一个样本
        explanation = explain_single_prediction(model, X.iloc[0])

        # 应该返回解释
        assert explanation is not None
        assert isinstance(explanation, dict)

    def test_smoke_calculate_feature_importance_shap(self, sample_classification_data):
        """烟雾测试：基于 SHAP 的特征重要性"""
        X, y = sample_classification_data

        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        importance = calculate_feature_importance_shap(model, X)

        # 应该返回特征重要性
        assert isinstance(importance, (dict, pd.DataFrame, list))
        if isinstance(importance, dict):
            assert len(importance) > 0

    def test_smoke_detect_proxy_variables(self):
        """烟雾测试：代理变量检测"""
        np.random.seed(42)
        n = 100

        df = pd.DataFrame({
            'gender': np.random.randint(0, 2, n),
            'income': np.random.uniform(3000, 30000, n),
            'occupation': np.random.choice(['A', 'B', 'C'], n)
        })

        proxies = detect_proxy_variables(df, sensitive_col='gender')

        # 应该返回代理变量检测结果
        assert isinstance(proxies, (dict, list))

    def test_smoke_calculate_disparate_impact(self):
        """烟雾测试：差异影响比计算"""
        np.random.seed(42)
        n = 100

        y_pred = np.random.randint(0, 2, n)
        group_labels = np.random.randint(0, 2, n)

        di_ratio = calculate_disparate_impact(y_pred, group_labels)

        # 应该返回差异影响比
        assert isinstance(di_ratio, float)
        assert di_ratio >= 0

    def test_smoke_calculate_equal_opportunity(self):
        """烟雾测试：平等机会计算"""
        np.random.seed(42)
        n = 100

        y_true = np.random.randint(0, 2, n)
        y_pred = np.random.randint(0, 2, n)
        group_labels = np.random.randint(0, 2, n)

        eo_diff = calculate_equal_opportunity(y_true, y_pred, group_labels)

        # 应该返回平等机会差异
        assert isinstance(eo_diff, (float, dict))

    def test_smoke_calculate_equalized_odds(self):
        """烟雾测试：均等几率计算"""
        np.random.seed(42)
        n = 100

        y_true = np.random.randint(0, 2, n)
        y_pred = np.random.randint(0, 2, n)
        group_labels = np.random.randint(0, 2, n)

        eo_result = calculate_equalized_odds(y_true, y_pred, group_labels)

        # 应该返回均等几率结果
        assert isinstance(eo_result, dict)

    def test_smoke_add_differential_privacy_noise(self):
        """烟雾测试：差分隐私噪声添加"""
        np.random.seed(42)
        data = np.array([1000, 2000, 3000, 4000, 5000])

        private_data = add_differential_privacy_noise(
            data, epsilon=1.0, sensitivity=4000
        )

        # 应该返回加了噪声的数据
        assert private_data is not None
        assert len(private_data) == len(data)

    def test_smoke_review_xai_code(self):
        """烟雾测试：XAI 代码审查"""
        code = """
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
"""

        review = review_xai_code(code)

        # 应该返回审查结果
        assert isinstance(review, dict)
        assert 'has_issues' in review or 'issues' in review

    def test_smoke_explain_to_nontechnical(self):
        """烟雾测试：向非技术人员解释"""
        prediction_explanation = {
            'base_value': 0.2,
            'shap_values': {'income': -0.3, 'credit_history': -0.1},
            'final_value': 0.35
        }

        explanation = explain_to_nontechnical(
            prediction_explanation,
            audience='customer'
        )

        # 应该返回易读的解释
        assert isinstance(explanation, str)
        assert len(explanation) > 0


class TestSmokeEndToEnd:
    """端到端工作流测试"""

    def test_complete_shap_workflow(self, sample_classification_data):
        """测试完整的 SHAP 工作流"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split

        X, y = sample_classification_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 1. 训练模型
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        # 2. 计算 SHAP 值
        shap_values = calculate_shap_values(model, X_test)
        assert shap_values is not None

        # 3. 计算特征重要性
        importance = calculate_feature_importance_shap(model, X_test)
        assert importance is not None

        # 4. 解释单个预测
        explanation = explain_single_prediction(model, X_test.iloc[0])
        assert explanation is not None

        # 流程成功
        assert True

    def test_complete_fairness_workflow(self):
        """测试完整的公平性评估工作流"""
        np.random.seed(42)
        n = 200

        # 生成模拟数据
        y_true = np.random.randint(0, 2, n)
        y_pred = np.random.randint(0, 2, n)
        group_labels = np.random.randint(0, 2, n)

        # 1. 计算差异影响比
        di_ratio = calculate_disparate_impact(y_pred, group_labels)
        assert di_ratio is not None

        # 2. 计算平等机会
        eo_diff = calculate_equal_opportunity(y_true, y_pred, group_labels)
        assert eo_diff is not None

        # 3. 计算均等几率
        eo_result = calculate_equalized_odds(y_true, y_pred, group_labels)
        assert eo_result is not None

        # 流程成功
        assert True

    def test_complete_ethics_review_workflow(self):
        """测试完整的伦理审查工作流"""
        code = """
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)
"""

        # 审查代码
        review = review_xai_code(code)
        assert review is not None

        # 生成解释
        mock_explanation = {
            'base_value': 0.2,
            'shap_values': {'feature_1': 0.3, 'feature_2': -0.1},
            'final_value': 0.4
        }

        explanation = explain_to_nontechnical(mock_explanation, 'customer')
        assert explanation is not None

        # 流程成功
        assert True

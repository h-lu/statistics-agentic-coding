"""
Week 10 烟雾测试（Smoke Test）

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
        fit_logistic_regression,
        calculate_confusion_matrix,
        calculate_precision_recall_f1,
        calculate_roc_auc,
        cross_validate_model,
        detect_data_leakage,
        review_classification_report,
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
        })
        y = np.random.randint(0, 2, n)
        return X, y

    def test_smoke_fit_logistic_regression(self, sample_classification_data):
        """烟雾测试：逻辑回归拟合"""
        X, y = sample_classification_data
        model = fit_logistic_regression(X, y)

        # 应该返回模型对象
        assert model is not None
        assert hasattr(model, 'coef_') or hasattr(model, 'params') or 'model' in str(type(model)).lower()

    def test_smoke_calculate_confusion_matrix(self):
        """烟雾测试：混淆矩阵计算"""
        y_true = np.array([0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 0, 1])

        result = calculate_confusion_matrix(y_true, y_pred)

        # 应该返回 TP, TN, FP, FN
        assert isinstance(result, (dict, tuple, np.ndarray))
        if isinstance(result, dict):
            assert 'tp' in result or 'TP' in result or 'true_positive' in result

    def test_smoke_calculate_precision_recall_f1(self):
        """烟雾测试：精确率、召回率、F1 计算"""
        y_true = np.array([0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 0, 1])

        result = calculate_precision_recall_f1(y_true, y_pred)

        # 应该返回精确率、召回率、F1
        assert isinstance(result, dict)
        assert 'precision' in result or '精确率' in result
        assert 'recall' in result or '召回率' in result
        assert 'f1' in result or 'f1_score' in result

    def test_smoke_calculate_roc_auc(self):
        """烟雾测试：ROC-AUC 计算"""
        y_true = np.array([0, 1, 0, 1, 1])
        y_proba = np.array([0.2, 0.8, 0.3, 0.6, 0.9])

        result = calculate_roc_auc(y_true, y_proba)

        # 应该返回 AUC 值和（可选）ROC 曲线数据
        assert isinstance(result, (dict, float, tuple))
        if isinstance(result, dict):
            assert 'auc' in result or 'AUC' in result

    def test_smoke_cross_validate_model(self, sample_classification_data):
        """烟雾测试：交叉验证"""
        X, y = sample_classification_data

        result = cross_validate_model(X, y, cv=5)

        # 应该返回交叉验证结果
        assert isinstance(result, dict)
        assert 'scores' in result or 'mean_score' in result or 'cv_scores' in result

    def test_smoke_detect_data_leakage(self, sample_classification_data):
        """烟雾测试：数据泄漏检测"""
        X, y = sample_classification_data

        # 模拟两种情况
        # 1. 全局预处理（有泄漏）
        from sklearn.preprocessing import StandardScaler
        scaler_global = StandardScaler()
        X_scaled_global = scaler_global.fit_transform(X)

        leakage_report = detect_data_leakage(X_scaled_global, y)

        # 应该返回检测结果
        assert isinstance(leakage_report, dict)
        assert 'has_leakage' in result or 'leakage_detected' in leakage_report or 'risk' in leakage_report

    def test_smoke_review_classification_report(self):
        """烟雾测试：分类报告审查"""
        report = """
        分类评估报告：

        准确率: 85.0%
        精确率: 60.0%
        召回率: 40.0%
        AUC: 0.75

        结论：模型性能良好。
        """

        result = review_classification_report(report)

        # 应该返回审查结果
        assert isinstance(result, dict)
        assert 'has_issues' in result or 'score' in result
        assert 'issues' in result or 'recommendations' in result or 'warnings' in result


class TestSmokeEndToEnd:
    """端到端工作流测试"""

    def test_complete_classification_workflow(self):
        """测试完整的分类分析工作流"""
        # 1. 生成数据
        np.random.seed(42)
        n = 100
        X = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, n),
            'feature_2': np.random.normal(0, 1, n),
        })
        y = np.random.randint(0, 2, n)

        # 2. 拟合逻辑回归
        model = fit_logistic_regression(X, y)
        assert model is not None

        # 3. 预测
        y_pred = model.predict(X) if hasattr(model, 'predict') else \
                  (model.predict(sm.add_constant(X)) > 0.5).astype(int)
        y_proba = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else \
                  model.predict(sm.add_constant(X))

        # 4. 计算混淆矩阵
        cm_result = calculate_confusion_matrix(y, y_pred)
        assert cm_result is not None

        # 5. 计算指标
        metrics = calculate_precision_recall_f1(y, y_pred)
        assert metrics is not None

        # 6. 计算 AUC
        auc_result = calculate_roc_auc(y, y_proba)
        assert auc_result is not None

        # 7. 交叉验证
        cv_result = cross_validate_model(X, y, cv=5)
        assert cv_result is not None

        # 8. 流程成功
        assert True

    def test_complete_review_workflow(self):
        """测试完整的报告审查工作流"""
        # 有问题的报告（只有准确率）
        bad_report = """
        分类评估报告：

        准确率: 85.0%
        模型训练成功。

        结论：模型表现良好。
        """

        # 审查报告
        result = review_classification_report(bad_report)

        # 应该检测到问题
        assert 'has_issues' in result or 'score' in result
        if 'has_issues' in result:
            # 应该检测到缺少混淆矩阵、精确率/召回率等
            assert result['has_issues'] is True

        # 流程成功
        assert True

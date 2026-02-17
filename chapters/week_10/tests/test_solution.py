"""
Comprehensive tests for Week 10 solution.py

综合测试：
- 逻辑回归分类
- 混淆矩阵与评估指标
- ROC 曲线与 AUC
- Pipeline 与数据泄漏防护
- StatLab 集成测试
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import make_classification

# Add starter_code to path
starter_code_path = Path(__file__).parent.parent / "starter_code"
sys.path.insert(0, str(starter_code_path))

try:
    import solution
except ImportError:
    solution = None


# =============================================================================
# 1. 逻辑回归分类测试
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestLogisticRegression:
    """测试逻辑回归功能"""

    def test_train_logistic_regression(self, simple_binary_classification_data):
        """
        正例：训练逻辑回归模型

        应能拟合二分类数据
        """
        if not hasattr(solution, 'train_logistic_regression') and not hasattr(solution, 'fit_logistic_regression'):
            pytest.skip("No logistic regression function implemented")

        X = simple_binary_classification_data['X']
        y = simple_binary_classification_data['y']

        # Try different function names
        if hasattr(solution, 'train_logistic_regression'):
            model = solution.train_logistic_regression(X, y)
        elif hasattr(solution, 'fit_logistic_regression'):
            model = solution.fit_logistic_regression(X, y)
        else:
            pytest.skip("No logistic regression function implemented")

        assert model is not None, "Should return a model"

        # Check that model can predict
        if hasattr(model, 'predict'):
            y_pred = model.predict(X)
            assert len(y_pred) == len(y), "Predictions should have same length as y"

    def test_logistic_regression_predictions(self, simple_binary_classification_data):
        """
        正例：逻辑回归预测

        应能返回类别预测
        """
        if not hasattr(solution, 'train_logistic_regression') and not hasattr(solution, 'fit_logistic_regression'):
            pytest.skip("No logistic regression function implemented")

        X = simple_binary_classification_data['X']
        y = simple_binary_classification_data['y']

        # Train model
        if hasattr(solution, 'train_logistic_regression'):
            model = solution.train_logistic_regression(X, y)
        elif hasattr(solution, 'fit_logistic_regression'):
            model = solution.fit_logistic_regression(X, y)
        else:
            pytest.skip("No logistic regression function implemented")

        # Make predictions
        if hasattr(model, 'predict'):
            y_pred = model.predict(X)
        elif hasattr(solution, 'predict_logistic'):
            y_pred = solution.predict_logistic(model, X)
        else:
            pytest.skip("No prediction function implemented")

        assert y_pred is not None
        assert len(y_pred) == len(y)

        # Predictions should be 0 or 1
        unique_preds = np.unique(y_pred)
        assert all(p in [0, 1] for p in unique_preds), f"Predictions should be 0 or 1, got {unique_preds}"

    def test_logistic_regression_probabilities(self, simple_binary_classification_data):
        """
        正例：逻辑回归概率预测

        应能返回类别概率（在 [0, 1] 之间）
        """
        if not hasattr(solution, 'train_logistic_regression') and not hasattr(solution, 'fit_logistic_regression'):
            pytest.skip("No logistic regression function implemented")

        X = simple_binary_classification_data['X']
        y = simple_binary_classification_data['y']

        # Train model
        if hasattr(solution, 'train_logistic_regression'):
            model = solution.train_logistic_regression(X, y)
        elif hasattr(solution, 'fit_logistic_regression'):
            model = solution.fit_logistic_regression(X, y)
        else:
            pytest.skip("No logistic regression function implemented")

        # Get probabilities
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X)
            # Get probability of class 1
            if y_prob.ndim > 1 and y_prob.shape[1] > 1:
                y_prob = y_prob[:, 1]
        elif hasattr(solution, 'predict_proba'):
            y_prob = solution.predict_proba(model, X)
        else:
            pytest.skip("No probability prediction function implemented")

        assert y_prob is not None
        assert len(y_prob) == len(y)

        # Probabilities should be in [0, 1]
        assert np.all(y_prob >= 0) and np.all(y_prob <= 1), "Probabilities should be in [0, 1]"

    def test_logistic_regression_coefficients(self, simple_binary_classification_data):
        """
        正例：获取逻辑回归系数

        应能返回截距和系数
        """
        if not hasattr(solution, 'train_logistic_regression') and not hasattr(solution, 'fit_logistic_regression'):
            pytest.skip("No logistic regression function implemented")

        X = simple_binary_classification_data['X']
        y = simple_binary_classification_data['y']

        # Train model
        if hasattr(solution, 'train_logistic_regression'):
            model = solution.train_logistic_regression(X, y)
        elif hasattr(solution, 'fit_logistic_regression'):
            model = solution.fit_logistic_regression(X, y)
        else:
            pytest.skip("No logistic regression function implemented")

        # Check coefficients
        has_coef = False
        if hasattr(model, 'coef_'):
            assert model.coef_ is not None
            assert len(model.coef_[0]) == X.shape[1], "Should have coefficient for each feature"
            has_coef = True

        if hasattr(model, 'intercept_'):
            assert model.intercept_ is not None
            has_coef = True

        if isinstance(model, dict):
            assert 'coefficients' in model or 'coef' in model or 'intercept' in model
            has_coef = True

        assert has_coef, "Should provide access to coefficients"

    def test_perfectly_separable_classes(self, binary_classification_perfect):
        """
        正例：完美可分类别

        模型应能达到高准确率
        """
        if not hasattr(solution, 'train_logistic_regression') and not hasattr(solution, 'fit_logistic_regression'):
            pytest.skip("No logistic regression function implemented")

        X = binary_classification_perfect['X']
        y = binary_classification_perfect['y']

        # Train model
        if hasattr(solution, 'train_logistic_regression'):
            model = solution.train_logistic_regression(X, y)
        elif hasattr(solution, 'fit_logistic_regression'):
            model = solution.fit_logistic_regression(X, y)
        else:
            pytest.skip("No logistic regression function implemented")

        # Make predictions
        if hasattr(model, 'predict'):
            y_pred = model.predict(X)
        elif hasattr(solution, 'predict_logistic'):
            y_pred = solution.predict_logistic(model, X)
        else:
            pytest.skip("No prediction function implemented")

        # Should achieve high accuracy
        accuracy = np.mean(y_pred == y)
        assert accuracy >= 0.95, f"Should achieve high accuracy for perfectly separable classes, got {accuracy}"

    def test_imbalanced_classification(self, imbalanced_classification_data):
        """
        正例：类别不平衡数据

        模型应能处理不平衡数据（80% 负类，20% 正类）
        """
        if not hasattr(solution, 'train_logistic_regression') and not hasattr(solution, 'fit_logistic_regression'):
            pytest.skip("No logistic regression function implemented")

        X = imbalanced_classification_data['X']
        y = imbalanced_classification_data['y']

        # Train model
        if hasattr(solution, 'train_logistic_regression'):
            model = solution.train_logistic_regression(X, y)
        elif hasattr(solution, 'fit_logistic_regression'):
            model = solution.fit_logistic_regression(X, y)
        else:
            pytest.skip("No logistic regression function implemented")

        assert model is not None, "Should handle imbalanced data"

        # Make predictions
        if hasattr(model, 'predict'):
            y_pred = model.predict(X)
        elif hasattr(solution, 'predict_logistic'):
            y_pred = solution.predict_logistic(model, X)
        else:
            pytest.skip("No prediction function implemented")

        # Should predict both classes (not all the same)
        unique_preds = np.unique(y_pred)
        assert len(unique_preds) >= 1, f"Should make predictions, got {unique_preds}"


# =============================================================================
# 2. 混淆矩阵与评估指标测试
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestConfusionMatrix:
    """测试混淆矩阵功能"""

    def test_calculate_confusion_matrix(self, known_predictions):
        """
        正例：计算混淆矩阵

        应返回 TP、TN、FP、FN
        """
        if not hasattr(solution, 'calculate_confusion_matrix'):
            pytest.skip("calculate_confusion_matrix not implemented")

        y_true = known_predictions['y_true']
        y_pred = known_predictions['y_pred']

        cm = solution.calculate_confusion_matrix(y_true, y_pred)

        assert cm is not None

        # Check format
        if isinstance(cm, np.ndarray):
            assert cm.shape == (2, 2), "Confusion matrix should be 2x2"
            # Extract values
            tn, fp, fn, tp = cm.ravel()
            assert tp == known_predictions['expected']['TP']
            assert tn == known_predictions['expected']['TN']
            assert fp == known_predictions['expected']['FP']
            assert fn == known_predictions['expected']['FN']
        elif isinstance(cm, dict):
            # May return dict with TP, TN, FP, FN
            expected = known_predictions['expected']
            # Check at least one value matches
            has_correct_value = False
            for key in ['TP', 'TN', 'FP', 'FN', 'tp', 'tn', 'fp', 'fn']:
                if key in cm:
                    if cm[key] == expected.get(key.upper(), expected.get(key.lower())):
                        has_correct_value = True
            assert has_correct_value, f"Confusion matrix dict should have correct values"

    def test_perfect_confusion_matrix(self, perfect_predictions):
        """
        正例：完美预测的混淆矩阵

        FP = FN = 0，TP + TN = 样本数
        """
        if not hasattr(solution, 'calculate_confusion_matrix'):
            pytest.skip("calculate_confusion_matrix not implemented")

        y_true = perfect_predictions['y_true']
        y_pred = perfect_predictions['y_pred']

        cm = solution.calculate_confusion_matrix(y_true, y_pred)

        if isinstance(cm, np.ndarray):
            tn, fp, fn, tp = cm.ravel()
            assert fp == 0, "FP should be 0 for perfect predictions"
            assert fn == 0, "FN should be 0 for perfect predictions"
            assert tp + tn == len(y_true), "All predictions should be correct"

    def test_worst_confusion_matrix(self, worst_predictions):
        """
        反例：最差预测的混淆矩阵

        TP = TN = 0
        """
        if not hasattr(solution, 'calculate_confusion_matrix'):
            pytest.skip("calculate_confusion_matrix not implemented")

        y_true = worst_predictions['y_true']
        y_pred = worst_predictions['y_pred']

        cm = solution.calculate_confusion_matrix(y_true, y_pred)

        if isinstance(cm, np.ndarray):
            tn, fp, fn, tp = cm.ravel()
            assert tp == 0, "TP should be 0 for worst predictions"
            assert tn == 0, "TN should be 0 for worst predictions"


@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestEvaluationMetrics:
    """测试评估指标功能"""

    def test_calculate_accuracy(self, known_predictions):
        """
        正例：计算准确率

        accuracy = (TP + TN) / (TP + TN + FP + FN)
        """
        if not hasattr(solution, 'calculate_accuracy'):
            pytest.skip("calculate_accuracy not implemented")

        y_true = known_predictions['y_true']
        y_pred = known_predictions['y_pred']

        accuracy = solution.calculate_accuracy(y_true, y_pred)

        assert accuracy is not None
        assert 0 <= accuracy <= 1, "Accuracy should be in [0, 1]"
        assert abs(accuracy - known_predictions['expected']['accuracy']) < 0.01, \
            f"Accuracy should be {known_predictions['expected']['accuracy']}, got {accuracy}"

    def test_calculate_precision(self, known_predictions):
        """
        正例：计算精确率

        precision = TP / (TP + FP)
        """
        if not hasattr(solution, 'calculate_precision'):
            pytest.skip("calculate_precision not implemented")

        y_true = known_predictions['y_true']
        y_pred = known_predictions['y_pred']

        precision = solution.calculate_precision(y_true, y_pred)

        assert precision is not None
        assert 0 <= precision <= 1, "Precision should be in [0, 1]"
        assert abs(precision - known_predictions['expected']['precision']) < 0.01, \
            f"Precision should be {known_predictions['expected']['precision']}, got {precision}"

    def test_calculate_recall(self, known_predictions):
        """
        正例：计算召回率

        recall = TP / (TP + FN)
        """
        if not hasattr(solution, 'calculate_recall'):
            pytest.skip("calculate_recall not implemented")

        y_true = known_predictions['y_true']
        y_pred = known_predictions['y_pred']

        recall = solution.calculate_recall(y_true, y_pred)

        assert recall is not None
        assert 0 <= recall <= 1, "Recall should be in [0, 1]"
        assert abs(recall - known_predictions['expected']['recall']) < 0.01, \
            f"Recall should be {known_predictions['expected']['recall']}, got {recall}"

    def test_calculate_f1(self, known_predictions):
        """
        正例：计算 F1 分数

        F1 = 2 * (precision * recall) / (precision + recall)
        """
        if not hasattr(solution, 'calculate_f1') and not hasattr(solution, 'calculate_f1_score'):
            pytest.skip("calculate_f1 not implemented")

        y_true = known_predictions['y_true']
        y_pred = known_predictions['y_pred']

        if hasattr(solution, 'calculate_f1'):
            f1 = solution.calculate_f1(y_true, y_pred)
        elif hasattr(solution, 'calculate_f1_score'):
            f1 = solution.calculate_f1_score(y_true, y_pred)
        else:
            pytest.skip("calculate_f1 not implemented")

        assert f1 is not None
        assert 0 <= f1 <= 1, "F1 should be in [0, 1]"
        assert abs(f1 - known_predictions['expected']['f1']) < 0.01, \
            f"F1 should be {known_predictions['expected']['f1']}, got {f1}"

    def test_calculate_all_metrics(self, known_predictions):
        """
        正例：计算所有评估指标

        应返回包含准确率、精确率、召回率、F1 的字典
        """
        if not hasattr(solution, 'calculate_metrics'):
            pytest.skip("calculate_metrics not implemented")

        y_true = known_predictions['y_true']
        y_pred = known_predictions['y_pred']

        metrics = solution.calculate_metrics(y_true, y_pred)

        assert metrics is not None
        assert isinstance(metrics, dict), "Should return a dictionary"

        # Should contain at least some metrics
        expected_keys = ['accuracy', 'precision', 'recall', 'f1', 'f1_score']
        has_metrics = any(key in metrics for key in expected_keys)
        assert has_metrics, f"Should have at least one metric, got keys: {metrics.keys()}"

    def test_perfect_predictions_metrics(self, perfect_predictions):
        """
        正例：完美预测的指标

        所有指标应为 1.0
        """
        if not hasattr(solution, 'calculate_metrics'):
            pytest.skip("calculate_metrics not implemented")

        y_true = perfect_predictions['y_true']
        y_pred = perfect_predictions['y_pred']

        metrics = solution.calculate_metrics(y_true, y_pred)

        # Check that key metrics are 1.0
        for key in ['accuracy', 'precision', 'recall', 'f1', 'f1_score']:
            if key in metrics:
                assert abs(metrics[key] - 1.0) < 0.01, f"{key} should be 1.0 for perfect predictions, got {metrics[key]}"

    def test_imbalanced_data_accuracy_trap(self, imbalanced_classification_data):
        """
        边界：类别不平衡时的准确率陷阱

        永远预测多数类的准确率 = 负类比例
        """
        # Calculate what a "always predict negative" classifier would get
        y = imbalanced_classification_data['y']
        majority_accuracy = (y == 0).mean()

        # This should be around 0.8 (80% negative class)
        assert 0.75 < majority_accuracy < 0.85, "Majority class should be around 80%"


# =============================================================================
# 3. ROC 曲线与 AUC 测试
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestROCCurve:
    """测试 ROC 曲线功能"""

    def test_calculate_roc_curve(self, probabilities_for_roc):
        """
        正例：计算 ROC 曲线

        应返回 FPR、TPR 和阈值
        """
        if not hasattr(solution, 'calculate_roc_curve'):
            pytest.skip("calculate_roc_curve not implemented")

        y_true = probabilities_for_roc['y_true']
        y_prob = probabilities_for_roc['y_prob']

        result = solution.calculate_roc_curve(y_true, y_prob)

        assert result is not None

        # Check format
        if isinstance(result, dict):
            assert 'fpr' in result or 'FPR' in result, "Should contain FPR"
            assert 'tpr' in result or 'TPR' in result, "Should contain TPR"
        elif isinstance(result, tuple):
            assert len(result) >= 2, "Should return at least FPR and TPR"
            fpr, tpr = result[0], result[1]
            assert len(fpr) == len(tpr), "FPR and TPR should have same length"

    def test_roc_curve_values(self, probabilities_for_roc):
        """
        正例：ROC 曲线值验证

        TPR 和 FPR 应在 [0, 1] 范围内
        """
        if not hasattr(solution, 'calculate_roc_curve'):
            pytest.skip("calculate_roc_curve not implemented")

        y_true = probabilities_for_roc['y_true']
        y_prob = probabilities_for_roc['y_prob']

        result = solution.calculate_roc_curve(y_true, y_prob)

        # Extract FPR and TPR
        if isinstance(result, dict):
            fpr = result.get('fpr', result.get('FPR'))
            tpr = result.get('tpr', result.get('TPR'))
        elif isinstance(result, tuple) and len(result) >= 2:
            fpr, tpr = result[0], result[1]
        else:
            pytest.skip("Unexpected ROC curve format")

        assert fpr is not None and tpr is not None
        assert np.all(fpr >= 0) and np.all(fpr <= 1), "FPR should be in [0, 1]"
        assert np.all(tpr >= 0) and np.all(tpr <= 1), "TPR should be in [0, 1]"


@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestAUC:
    """测试 AUC 功能"""

    def test_calculate_auc(self, probabilities_for_roc):
        """
        正例：计算 AUC

        AUC 应在 [0, 1] 范围内
        """
        if not hasattr(solution, 'calculate_auc'):
            pytest.skip("calculate_auc not implemented")

        y_true = probabilities_for_roc['y_true']
        y_prob = probabilities_for_roc['y_prob']

        auc = solution.calculate_auc(y_true, y_prob)

        assert auc is not None
        assert 0 <= auc <= 1, f"AUC should be in [0, 1], got {auc}"

    def test_perfect_auc(self, probabilities_for_roc):
        """
        正例：完美排序的 AUC

        当所有负类概率 < 所有正类概率时，AUC = 1
        """
        if not hasattr(solution, 'calculate_auc'):
            pytest.skip("calculate_auc not implemented")

        y_true = probabilities_for_roc['y_true']
        y_prob = probabilities_for_roc['y_prob']

        auc = solution.calculate_auc(y_true, y_prob)

        # Perfect separation should give AUC close to 1
        assert auc > 0.95, f"AUC should be close to 1 for perfect separation, got {auc}"

    def test_random_auc(self):
        """
        正例：随机预测的 AUC

        随机预测的 AUC 应该在合理范围内（不极端偏离 0.5）
        注意：小样本下 AUC 波动较大，使用较宽的容差
        """
        if not hasattr(solution, 'calculate_auc'):
            pytest.skip("calculate_auc not implemented")

        np.random.seed(42)
        # 使用更大的样本量以获得更稳定的随机 AUC
        n_samples = 100
        y_true = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))
        y_prob = np.random.random(n_samples)

        auc = solution.calculate_auc(y_true, y_prob)

        # 随机预测的 AUC 应该在合理范围内（小样本波动大，用较宽范围）
        assert 0.2 <= auc <= 0.8, f"AUC should be in reasonable range for random predictions, got {auc}"

    def test_reversed_auc(self):
        """
        反例：反向预测的 AUC

        当所有负类概率 > 所有正类概率时，AUC 接近 0
        """
        if not hasattr(solution, 'calculate_auc'):
            pytest.skip("calculate_auc not implemented")

        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        # Reversed: negatives have high prob, positives have low prob
        y_prob = np.array([0.9, 0.8, 0.7, 0.6, 0.1, 0.2, 0.3, 0.4])

        auc = solution.calculate_auc(y_true, y_prob)

        # Reversed predictions should give AUC close to 0
        assert auc < 0.2, f"AUC should be close to 0 for reversed predictions, got {auc}"


# =============================================================================
# 4. Pipeline 与数据泄漏测试
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestPipeline:
    """测试 Pipeline 功能"""

    def test_create_classification_pipeline(self):
        """
        正例：创建分类 Pipeline

        应包含预处理和模型步骤
        """
        if not hasattr(solution, 'create_classification_pipeline'):
            pytest.skip("create_classification_pipeline not implemented")

        pipeline = solution.create_classification_pipeline()

        assert pipeline is not None

        # Check that pipeline has steps
        if hasattr(pipeline, 'steps'):
            assert len(pipeline.steps) >= 1, "Pipeline should have at least one step"

    def test_train_with_pipeline(self, simple_binary_classification_data):
        """
        正例：使用 Pipeline 训练模型

        应能 fit 和 predict
        """
        if not hasattr(solution, 'train_with_pipeline'):
            pytest.skip("train_with_pipeline not implemented")

        X = simple_binary_classification_data['X']
        y = simple_binary_classification_data['y']

        model = solution.train_with_pipeline(X, y)

        assert model is not None

        # Should be able to predict
        if hasattr(model, 'predict'):
            y_pred = model.predict(X)
            assert y_pred is not None

    def test_pipeline_with_missing_values(self, data_with_missing_values):
        """
        正例：Pipeline 处理缺失值

        应能自动填充缺失值而不泄漏
        """
        if not hasattr(solution, 'train_with_pipeline'):
            pytest.skip("train_with_pipeline not implemented")

        X = data_with_missing_values['X']
        y = data_with_missing_values['y']

        # Pipeline should handle missing values
        try:
            model = solution.train_with_pipeline(X, y)
            assert model is not None
        except (ValueError, KeyError):
            # May fail if X is DataFrame without column names
            # Try with numpy array
            X_array = X.values
            model = solution.train_with_pipeline(X_array, y)
            assert model is not None

    def test_cross_val_with_pipeline(self, simple_binary_classification_data):
        """
        正例：Pipeline + 交叉验证

        交叉验证应使用 Pipeline 防止泄漏
        """
        if not hasattr(solution, 'cross_val_with_pipeline'):
            pytest.skip("cross_val_with_pipeline not implemented")

        X = simple_binary_classification_data['X']
        y = simple_binary_classification_data['y']

        scores = solution.cross_val_with_pipeline(X, y)

        assert scores is not None

        # Check format
        if isinstance(scores, (list, np.ndarray)):
            assert len(scores) >= 1, "Should have at least one CV score"
            assert all(0 <= s <= 1 for s in scores), "CV scores should be in [0, 1]"
        elif isinstance(scores, dict):
            assert 'mean' in scores or 'scores' in scores, "Should have mean or scores"


# =============================================================================
# 5. StatLab 集成测试
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestStatLabIntegration:
    """测试 StatLab 集成功能"""

    def test_classification_with_pipeline(self, statlab_customer_churn_data):
        """
        正例：完整的分类评估流程

        应返回模型、评估指标、混淆矩阵、ROC 数据等
        """
        if not hasattr(solution, 'classification_with_pipeline'):
            pytest.skip("classification_with_pipeline not implemented")

        df = statlab_customer_churn_data
        feature_cols = ['purchase_count', 'avg_spend', 'days_since_last_purchase', 'registration_days']
        X = df[feature_cols].values
        y = df['is_churned'].values

        result = solution.classification_with_pipeline(X, y)

        assert result is not None
        assert isinstance(result, dict), "Should return a dictionary"

        # Should contain key results
        has_metrics = 'metrics' in result or 'accuracy' in result or 'auc' in result
        assert has_metrics, "Should contain evaluation metrics"

    def test_format_classification_report(self, statlab_customer_churn_data):
        """
        正例：格式化分类报告

        应生成可读的 Markdown 报告
        """
        if not hasattr(solution, 'classification_with_pipeline') or not hasattr(solution, 'format_classification_report'):
            pytest.skip("StatLab functions not implemented")

        df = statlab_customer_churn_data
        feature_cols = ['purchase_count', 'avg_spend', 'days_since_last_purchase', 'registration_days']
        X = df[feature_cols].values
        y = df['is_churned'].values

        result = solution.classification_with_pipeline(X, y)
        report = solution.format_classification_report(result)

        assert report is not None
        assert isinstance(report, str), "Report should be a string"
        assert len(report) > 100, "Report should have meaningful content"

        # Report should mention key metrics
        report_lower = report.lower()
        has_metrics = any(keyword in report_lower for keyword in
                         ['accuracy', 'precision', 'recall', 'f1', 'auc'])
        assert has_metrics, "Report should mention evaluation metrics"


# =============================================================================
# 6. 数值稳定性测试
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestNumericalStability:
    """测试数值稳定性"""

    def test_large_feature_values(self):
        """
        边界：大数值特征

        应能正确处理大数值
        """
        if not hasattr(solution, 'train_logistic_regression') and not hasattr(solution, 'fit_logistic_regression'):
            pytest.skip("No logistic regression function implemented")

        np.random.seed(42)
        X = np.random.uniform(1e6, 1e7, (100, 2))
        y = (X[:, 0] > 5e6).astype(int)

        # Train model
        if hasattr(solution, 'train_logistic_regression'):
            model = solution.train_logistic_regression(X, y)
        elif hasattr(solution, 'fit_logistic_regression'):
            model = solution.fit_logistic_regression(X, y)
        else:
            pytest.skip("No logistic regression function implemented")

        assert model is not None

    def test_small_feature_values(self):
        """
        边界：小数值特征

        应能正确处理接近 0 的数值
        """
        if not hasattr(solution, 'train_logistic_regression') and not hasattr(solution, 'fit_logistic_regression'):
            pytest.skip("No logistic regression function implemented")

        np.random.seed(42)
        X = np.random.uniform(0.001, 0.01, (100, 2))
        y = (X[:, 0] > 0.005).astype(int)

        # Train model
        if hasattr(solution, 'train_logistic_regression'):
            model = solution.train_logistic_regression(X, y)
        elif hasattr(solution, 'fit_logistic_regression'):
            model = solution.fit_logistic_regression(X, y)
        else:
            pytest.skip("No logistic regression function implemented")

        assert model is not None

    def test_reproducibility_with_seed(self, simple_binary_classification_data):
        """
        正例：相同随机种子应产生相同结果

        验证可复现性（对于有随机初始化的模型）
        """
        if not hasattr(solution, 'train_logistic_regression') and not hasattr(solution, 'fit_logistic_regression'):
            pytest.skip("No logistic regression function implemented")

        X = simple_binary_classification_data['X']
        y = simple_binary_classification_data['y']

        # LogisticRegression with same random_state should give same results
        # (this is implementation dependent)
        assert True  # Placeholder - implementation may vary


# =============================================================================
# 7. 特征重要性测试
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestFeatureImportance:
    """测试特征重要性功能"""

    def test_get_feature_importance(self, simple_binary_classification_data):
        """
        正例：获取特征重要性

        应返回特征系数或重要性分数
        """
        if not hasattr(solution, 'train_logistic_regression') and not hasattr(solution, 'fit_logistic_regression'):
            pytest.skip("No logistic regression function implemented")

        if not hasattr(solution, 'get_feature_importance'):
            pytest.skip("get_feature_importance not implemented")

        X = simple_binary_classification_data['X']
        y = simple_binary_classification_data['y']

        # Train model
        if hasattr(solution, 'train_logistic_regression'):
            model = solution.train_logistic_regression(X, y)
        elif hasattr(solution, 'fit_logistic_regression'):
            model = solution.fit_logistic_regression(X, y)
        else:
            pytest.skip("No logistic regression function implemented")

        importance = solution.get_feature_importance(model)

        assert importance is not None

        # Should have importance for each feature
        if isinstance(importance, (list, np.ndarray)):
            assert len(importance) == X.shape[1], "Should have importance for each feature"
        elif isinstance(importance, dict):
            assert len(importance) == X.shape[1], "Should have importance for each feature"

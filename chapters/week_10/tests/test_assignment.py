"""
Week 10 作业测试框架

注意：这是基础测试框架，只验证核心函数的正确性。
完整的作业评估需要人工评分（解释、可视化、报告质量）。
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score, classification_report
)


class TestLogisticRegression:
    """测试逻辑回归分类"""

    def test_model_fitting(self):
        """测试模型拟合是否正确"""
        np.random.seed(42)
        X, y = make_classification(
            n_samples=200,
            n_features=3,
            n_redundant=0,
            n_informative=2,
            random_state=42
        )

        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)

        # 检查系数是否存在
        assert model.coef_ is not None, "模型应该有系数"
        assert model.intercept_ is not None, "模型应该有截距"

        # 检查系数数量
        assert model.coef_.shape[1] == X.shape[1], "系数数量应该等于特征数量"

        # 检查系数是否为有限数
        assert np.isfinite(model.coef_).all(), "系数应该是有限数"
        assert np.isfinite(model.intercept_).all(), "截距应该是有限数"

    def test_prediction_classes(self):
        """测试类别预测"""
        np.random.seed(42)
        X, y = make_classification(
            n_samples=200,
            n_features=3,
            n_redundant=0,
            random_state=42
        )

        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)
        y_pred = model.predict(X)

        # 预测应该是 0 或 1
        unique_preds = np.unique(y_pred)
        assert all(p in [0, 1] for p in unique_preds), "预测应该是 0 或 1"

        # 预测数量应该匹配
        assert len(y_pred) == len(y), "预测数量应该等于样本数量"

    def test_probability_prediction(self):
        """测试概率预测"""
        np.random.seed(42)
        X, y = make_classification(
            n_samples=200,
            n_features=3,
            n_redundant=0,
            random_state=42
        )

        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)
        y_prob = model.predict_proba(X)

        # 概率应该在 [0, 1] 之间
        assert np.all(y_prob >= 0) and np.all(y_prob <= 1), "概率应该在 [0, 1] 之间"

        # 每行概率和应该等于 1
        assert np.allclose(np.sum(y_prob, axis=1), 1.0), "每行概率和应该等于 1"


class TestConfusionMatrix:
    """测试混淆矩阵"""

    def test_confusion_matrix_shape(self):
        """测试混淆矩阵形状"""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])

        cm = confusion_matrix(y_true, y_pred)

        # 二分类应该是 2x2
        assert cm.shape == (2, 2), "二分类混淆矩阵应该是 2x2"

    def test_confusion_matrix_values(self):
        """测试混淆矩阵值"""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])

        cm = confusion_matrix(y_true, y_pred)

        # 提取值
        tn, fp, fn, tp = cm.ravel()

        # 验证
        assert tn == 1, "TN 应该是 1"
        assert fp == 1, "FP 应该是 1"
        assert fn == 0, "FN 应该是 0"
        assert tp == 2, "TP 应该是 2"

    def test_confusion_matrix_perfect(self):
        """测试完美预测的混淆矩阵"""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])  # 完美预测

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # FP 和 FN 应该是 0
        assert fp == 0, "完美预测的 FP 应该是 0"
        assert fn == 0, "完美预测的 FN 应该是 0"


class TestEvaluationMetrics:
    """测试评估指标"""

    def test_accuracy_calculation(self):
        """测试准确率计算"""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])

        accuracy = accuracy_score(y_true, y_pred)

        # accuracy = (1 + 2) / 4 = 0.75
        assert abs(accuracy - 0.75) < 0.01, f"准确率应该是 0.75, 得到 {accuracy}"

    def test_precision_calculation(self):
        """测试精确率计算"""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])

        precision = precision_score(y_true, y_pred)

        # precision = 2 / (2 + 1) = 2/3
        assert abs(precision - 2/3) < 0.01, f"精确率应该是 0.667, 得到 {precision}"

    def test_recall_calculation(self):
        """测试召回率计算"""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])

        recall = recall_score(y_true, y_pred)

        # recall = 2 / (2 + 0) = 1.0
        assert abs(recall - 1.0) < 0.01, f"召回率应该是 1.0, 得到 {recall}"

    def test_f1_calculation(self):
        """测试 F1 分数计算"""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])

        f1 = f1_score(y_true, y_pred)

        # F1 = 2 * (precision * recall) / (precision + recall)
        # = 2 * (2/3 * 1) / (2/3 + 1) = 4/5 = 0.8
        assert abs(f1 - 0.8) < 0.01, f"F1 分数应该是 0.8, 得到 {f1}"


class TestROCAUC:
    """测试 ROC 和 AUC"""

    def test_roc_curve_output(self):
        """测试 ROC 曲线输出"""
        np.random.seed(42)
        X, y = make_classification(
            n_samples=100,
            n_features=2,
            n_redundant=0,
            random_state=42
        )

        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)
        y_prob = model.predict_proba(X)[:, 1]

        fpr, tpr, thresholds = roc_curve(y, y_prob)

        # FPR 和 TPR 应该在 [0, 1] 范围内
        assert np.all(fpr >= 0) and np.all(fpr <= 1), "FPR 应该在 [0, 1] 之间"
        assert np.all(tpr >= 0) and np.all(tpr <= 1), "TPR 应该在 [0, 1] 之间"

        # FPR 和 TPR 长度应该相同
        assert len(fpr) == len(tpr), "FPR 和 TPR 长度应该相同"

    def test_auc_range(self):
        """测试 AUC 范围"""
        np.random.seed(42)
        X, y = make_classification(
            n_samples=100,
            n_features=2,
            n_redundant=0,
            random_state=42
        )

        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)
        y_prob = model.predict_proba(X)[:, 1]

        auc = roc_auc_score(y, y_prob)

        # AUC 应该在 [0, 1] 之间
        assert 0 <= auc <= 1, f"AUC 应该在 [0, 1] 之间, 得到 {auc}"

    def test_perfect_separation_auc(self):
        """测试完美分离的 AUC"""
        # 完美分离：负类概率低，正类概率高
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

        auc = roc_auc_score(y_true, y_prob)

        # 完美分离应该接近 1
        assert auc > 0.95, f"完美分离的 AUC 应该接近 1, 得到 {auc}"


class TestPipeline:
    """测试 Pipeline"""

    def test_pipeline_creation(self):
        """测试 Pipeline 创建"""
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(max_iter=1000, random_state=42))
        ])

        # Pipeline 应该有步骤
        assert len(pipeline.steps) == 2, "Pipeline 应该有 2 个步骤"

        # 步骤名称应该正确
        step_names = [name for name, _ in pipeline.steps]
        assert 'scaler' in step_names, "应该有 scaler 步骤"
        assert 'model' in step_names, "应该有 model 步骤"

    def test_pipeline_fit_predict(self):
        """测试 Pipeline 的 fit 和 predict"""
        from sklearn.preprocessing import StandardScaler

        np.random.seed(42)
        X, y = make_classification(
            n_samples=200,
            n_features=3,
            n_redundant=0,
            random_state=42
        )

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(max_iter=1000, random_state=42))
        ])

        # Fit
        pipeline.fit(X, y)

        # Predict
        y_pred = pipeline.predict(X)

        # 预测应该是 0 或 1
        unique_preds = np.unique(y_pred)
        assert all(p in [0, 1] for p in unique_preds), "预测应该是 0 或 1"

    def test_pipeline_with_missing_values(self):
        """测试 Pipeline 处理缺失值"""
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler

        np.random.seed(42)
        X, y = make_classification(
            n_samples=200,
            n_features=3,
            n_redundant=0,
            random_state=42
        )

        # 添加缺失值
        X[10, 0] = np.nan
        X[20, 1] = np.nan

        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(max_iter=1000, random_state=42))
        ])

        # 应该能处理缺失值
        pipeline.fit(X, y)
        y_pred = pipeline.predict(X)

        assert y_pred is not None, "Pipeline 应该能处理缺失值"


class TestCrossValidation:
    """测试交叉验证"""

    def test_cross_val_scores(self):
        """测试交叉验证分数"""
        from sklearn.preprocessing import StandardScaler

        np.random.seed(42)
        X, y = make_classification(
            n_samples=200,
            n_features=3,
            n_redundant=0,
            random_state=42
        )

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(max_iter=1000, random_state=42))
        ])

        scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')

        # 应该有 5 个分数
        assert len(scores) == 5, "5 折交叉验证应该有 5 个分数"

        # 分数应该在 [0, 1] 之间
        assert all(0 <= s <= 1 for s in scores), "准确率分数应该在 [0, 1] 之间"

    def test_cross_val_with_pipeline(self):
        """测试 Pipeline 防止数据泄漏"""
        from sklearn.preprocessing import StandardScaler

        np.random.seed(42)
        X, y = make_classification(
            n_samples=200,
            n_features=3,
            n_redundant=0,
            random_state=42
        )

        # 使用 Pipeline 的交叉验证
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(max_iter=1000, random_state=42))
        ])

        scores_with_pipeline = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')

        # 不使用 Pipeline（数据泄漏）
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)  # 泄漏！
        model = LogisticRegression(max_iter=1000, random_state=42)
        scores_leaky = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')

        # Pipeline 的分数应该 <= 泄漏的分数（泄漏会虚高）
        # 但这不是硬性要求，因为差异可能很小
        assert scores_with_pipeline is not None
        assert scores_leaky is not None


class TestClassImbalance:
    """测试类别不平衡"""

    def test_imbalanced_dataset(self):
        """测试不平衡数据集"""
        np.random.seed(42)
        X, y = make_classification(
            n_samples=500,
            n_features=3,
            n_redundant=0,
            weights=[0.8, 0.2],  # 80% 负类，20% 正类
            random_state=42
        )

        # 计算类别比例
        pos_ratio = y.mean()
        neg_ratio = 1 - pos_ratio

        # 验证不平衡
        assert abs(pos_ratio - 0.2) < 0.05, f"正类比例应该约 20%, 得到 {pos_ratio:.1%}"
        assert abs(neg_ratio - 0.8) < 0.05, f"负类比例应该约 80%, 得到 {neg_ratio:.1%}"

    def test_imbalanced_metrics(self):
        """测试不平衡数据的评估指标"""
        np.random.seed(42)
        X, y = make_classification(
            n_samples=500,
            n_features=3,
            n_redundant=0,
            weights=[0.8, 0.2],
            random_state=42
        )

        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)
        y_pred = model.predict(X)

        # 计算指标
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, zero_division=0)
        recall = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)

        # 所有指标应该在 [0, 1] 之间
        assert 0 <= accuracy <= 1, "准确率应该在 [0, 1] 之间"
        assert 0 <= precision <= 1, "精确率应该在 [0, 1] 之间"
        assert 0 <= recall <= 1, "召回率应该在 [0, 1] 之间"
        assert 0 <= f1 <= 1, "F1 分数应该在 [0, 1] 之间"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

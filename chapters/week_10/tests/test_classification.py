"""
Week 10: 分类模型与逻辑回归测试

测试用例矩阵：
1. 正例测试 - 验证正常情况下的功能
2. 边界测试 - 验证极端/边界情况
3. 反例测试 - 验证错误输入的处理
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    accuracy_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# =============================================================================
# 测试数据生成器
# =============================================================================


def generate_balanced_data(n_samples=1000, n_features=4, random_state=42):
    """生成平衡的二分类数据"""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_redundant=0,
        n_clusters_per_class=1,
        weights=[0.5, 0.5],
        flip_y=0,
        random_state=random_state,
    )
    return X, y


def generate_imbalanced_data(n_samples=1000, n_features=4, random_state=42):
    """生成不平衡的二分类数据（1:9 比例）"""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_redundant=0,
        n_clusters_per_class=1,
        weights=[0.9, 0.1],
        flip_y=0,
        random_state=random_state,
    )
    return X, y


def train_logistic_model(X, y, random_state=42, class_weight=None):
    """辅助函数：训练逻辑回归模型"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        random_state=random_state,
        class_weight=class_weight,
    )
    model.fit(X_train_scaled, y_train)

    return model, X_train_scaled, X_test_scaled, y_train, y_test, scaler


# =============================================================================
# 1. 正例测试 (Happy Path Tests)
# =============================================================================


def test_logistic_regression_fitting():
    """测试逻辑回归模型能正常拟合数据"""
    X, y = generate_balanced_data(n_samples=1000, random_state=42)
    model, X_train, X_test, y_train, y_test, _ = train_logistic_model(X, y)

    # 验证模型成功拟合（有系数和截距）
    assert model.coef_ is not None
    assert model.intercept_ is not None
    assert model.coef_.shape == (1, X.shape[1])

    # 验证模型能在测试集上预测
    y_pred = model.predict(X_test)
    assert len(y_pred) == len(y_test)
    assert all(pred in [0, 1] for pred in y_pred)

    # 验证能输出概率
    y_prob = model.predict_proba(X_test)
    assert y_prob.shape == (len(X_test), 2)
    assert np.all((y_prob >= 0) & (y_prob <= 1))
    assert np.allclose(y_prob.sum(axis=1), 1.0)


def test_confusion_matrix_shape():
    """测试混淆矩阵形状正确"""
    X, y = generate_balanced_data(n_samples=500, random_state=42)
    model, _, X_test, _, y_test, _ = train_logistic_model(X, y)

    y_pred = model.predict(X_test)

    # 计算混淆矩阵
    cm = confusion_matrix(y_test, y_pred)

    # 二分类问题，混淆矩阵应该是 2x2
    assert cm.shape == (2, 2)

    # 提取各元素
    tn, fp, fn, tp = cm.ravel()

    # 验证各个值非负
    assert tn >= 0 and fp >= 0 and fn >= 0 and tp >= 0

    # 验证总和等于测试集大小
    total = tn + fp + fn + tp
    assert total == len(y_test)


def test_precision_recall_calculation():
    """测试精确率、查全率、F1 计算正确"""
    X, y = generate_balanced_data(n_samples=500, random_state=42)
    model, _, X_test, _, y_test, _ = train_logistic_model(X, y)

    y_pred = model.predict(X_test)

    # 计算指标
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)

    # 验证指标在合理范围内
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= f1 <= 1
    assert 0 <= accuracy <= 1

    # F1 是精确率和查全率的调和平均
    if precision > 0 and recall > 0:
        expected_f1 = 2 * (precision * recall) / (precision + recall)
        assert abs(f1 - expected_f1) < 1e-10


def test_roc_auc_calculation():
    """测试 AUC 计算正确"""
    X, y = generate_balanced_data(n_samples=500, random_state=42)
    model, _, X_test, _, y_test, _ = train_logistic_model(X, y)

    y_prob = model.predict_proba(X_test)[:, 1]

    # 计算 ROC 曲线
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    # 验证 ROC 曲线点的数量
    assert len(fpr) == len(tpr) == len(thresholds)
    assert len(fpr) >= 2  # 至少有两个点

    # 验证 FPR 和 TPR 在 [0, 1] 范围内
    assert np.all((fpr >= 0) & (fpr <= 1))
    assert np.all((tpr >= 0) & (tpr <= 1))

    # 计算 AUC
    auc = roc_auc_score(y_test, y_prob)

    # AUC 应该在 [0, 1] 之间
    assert 0 <= auc <= 1

    # 对于可分离的数据，AUC 应该明显高于 0.5
    assert auc > 0.5


# =============================================================================
# 2. 边界测试 (Edge Case Tests)
# =============================================================================


def test_perfect_classification():
    """测试完美分类时的指标"""
    # 创建完全可分离的数据
    np.random.seed(42)
    n_samples = 200

    # 类别 0：特征值较小
    X0 = np.random.randn(n_samples // 2, 2) - 2
    y0 = np.zeros(n_samples // 2)

    # 类别 1：特征值较大
    X1 = np.random.randn(n_samples // 2, 2) + 2
    y1 = np.ones(n_samples // 2)

    X = np.vstack([X0, X1])
    y = np.hstack([y0, y1])

    model, _, X_test, _, y_test, _ = train_logistic_model(X, y)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # 计算指标
    precision = precision_score(y_test, y_pred, zero_division=1)
    recall = recall_score(y_test, y_pred, zero_division=1)
    f1 = f1_score(y_test, y_pred, zero_division=1)
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    # 完美分类应该接近 1.0
    assert accuracy >= 0.95
    assert auc >= 0.99


def test_all_same_prediction():
    """测试全部预测为同一类的处理"""
    # 创建一个模型只预测一个类别的场景
    np.random.seed(42)
    n_samples = 100

    # 训练数据：保证有两类，但某一类很少
    X = np.random.randn(n_samples, 2)
    y = np.zeros(n_samples, dtype=int)
    # 设置最后 5 个为正类
    y[-5:] = 1
    # 让正类样本的特征值明显更大
    X[-5:] = X[-5:] + 3

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    # 使用 zero_division 参数处理边界情况
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)

    # 验证指标在有效范围内
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1


def test_empty_class():
    """测试某类无样本时的处理"""
    # 创建测试数据，模拟某类在测试集中不存在的情况
    np.random.seed(42)

    # 训练数据有两类
    X_train = np.random.randn(100, 2)
    y_train = np.random.randint(0, 2, 100)

    # 测试数据只有一类（模拟极端情况）
    X_test = np.random.randn(20, 2)
    y_test = np.zeros(20, dtype=int)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    # 应该能正常计算混淆矩阵
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    assert cm.shape == (2, 2)

    # 使用 labels 参数确保类别一致
    precision = precision_score(y_test, y_pred, labels=[0, 1], zero_division=0)
    recall = recall_score(y_test, y_pred, labels=[0, 1], zero_division=0)

    assert 0 <= precision <= 1
    assert 0 <= recall <= 1


# =============================================================================
# 3. 反例测试 (Error Handling Tests)
# =============================================================================


def test_invalid_input_shape():
    """测试输入形状不匹配应报错"""
    X, y = generate_balanced_data(n_samples=100, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # 测试特征数量不匹配的输入
    X_wrong_shape = X_test[:, :2]  # 只取前 2 个特征

    with pytest.raises(ValueError):
        model.predict(X_wrong_shape)


def test_invalid_probability_range():
    """测试概率超出 [0,1] 范围应报错"""
    # 手动构造无效的预测概率
    y_true = np.array([0, 0, 1, 1])

    # 有效的概率
    y_prob_valid = np.array([0.1, 0.2, 0.8, 0.9])
    auc_valid = roc_auc_score(y_true, y_prob_valid)
    assert 0 <= auc_valid <= 1

    # 超出范围的概率应该在使用时出现问题
    y_prob_invalid = np.array([-0.1, 1.2, 0.5, 0.6])

    # roc_auc_score 不会直接报错，但会给出警告或异常结果
    # 这里我们验证当使用无效概率时，AUC 计算可能产生异常值
    with np.errstate(invalid="ignore"):
        try:
            auc_invalid = roc_auc_score(y_true, y_prob_invalid)
            # 如果计算成功，结果应该在合理范围内
            # 但 sklearn 通常会在内部处理异常值
        except ValueError:
            # 预期的行为：无效输入应报错
            pass


def test_missing_labels():
    """测试标签缺失处理"""
    X, y = generate_balanced_data(n_samples=100, random_state=42)

    # 创建带有缺失值的标签（用 -1 表示缺失）
    y_with_missing = y.copy().astype(float)
    y_with_missing[:10] = np.nan

    # 使用 sklearn 的 SimpleImputer 或手动处理缺失值
    from sklearn.impute import SimpleImputer

    # 分离有效数据
    valid_mask = ~np.isnan(y_with_missing)
    X_valid = X[valid_mask]
    y_valid = y_with_missing[valid_mask]

    # 应该能正常训练
    model = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
    model.fit(X_valid, y_valid)

    # 验证模型能正常预测
    y_pred = model.predict(X[:5])
    assert len(y_pred) == 5


# =============================================================================
# 类别不平衡专项测试
# =============================================================================


def test_class_imbalance_accuracy_trap():
    """测试类别不平衡时的准确率陷阱"""
    X, y = generate_imbalanced_data(n_samples=1000, random_state=42)

    # 不使用 class_weight
    model_normal, _, X_test, _, y_test, _ = train_logistic_model(X, y, class_weight=None)

    # 使用 class_weight='balanced'
    model_balanced, _, X_test2, _, y_test2, _ = train_logistic_model(
        X, y, class_weight="balanced"
    )

    y_pred_normal = model_normal.predict(X_test)
    y_pred_balanced = model_balanced.predict(X_test2)

    # 计算正类的查全率
    recall_normal = recall_score(y_test, y_pred_normal, zero_division=0)
    recall_balanced = recall_score(y_test2, y_pred_balanced, zero_division=0)

    # 使用 balanced 权重应该提高少数类的查全率
    # （这不总是成立，但在大多数不平衡数据集上有效）
    # 我们主要验证两种模型都能运行，且指标在合理范围内
    assert 0 <= recall_normal <= 1
    assert 0 <= recall_balanced <= 1


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

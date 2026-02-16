"""
Week 10: StatLab 集成测试

测试 StatLab 的分类分析模块功能：
- 分类报告生成
- ROC 曲线绘制
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    roc_auc_score,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
import pandas as pd


# =============================================================================
# StatLab 分类分析模块（复制自 CHAPTER.md）
# =============================================================================


def classification_with_evaluation(
    X, y, var_names=None, test_size=0.2, random_state=42
):
    """
    拟合逻辑回归模型并输出完整评估报告

    参数:
    - X: 特征矩阵
    - y: 目标变量（0/1）
    - var_names: 特征名称列表
    - test_size: 测试集比例
    - random_state: 随机种子

    返回:
    - dict: 包含模型、评估结果、图表数据的字典
    """
    # 划分训练集和测试集（分层抽样）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 检查类别分布
    class_dist = pd.Series(y).value_counts().sort_index()
    imbalance_ratio = class_dist.max() / class_dist.min()

    # 特征缩放
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 拟合模型（如果类别不平衡，使用 balanced 权重）
    if imbalance_ratio > 5:
        model = LogisticRegression(
            solver="lbfgs",
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state,
        )
    else:
        model = LogisticRegression(
            solver="lbfgs", max_iter=1000, random_state=random_state
        )

    model.fit(X_train_scaled, y_train)

    # 预测
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    # 1. 基础评估指标
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    # 2. ROC 曲线
    fpr, tpr, roc_thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    # 3. 交叉验证
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring="f1")

    # 4. 特征重要性（系数）
    if var_names is None:
        var_names = [f"feature_{i}" for i in range(X.shape[1])]

    feature_importance = pd.DataFrame(
        {
            "feature": var_names,
            "coefficient": model.coef_[0],
            "abs_coefficient": np.abs(model.coef_[0]),
        }
    ).sort_values("abs_coefficient", ascending=False)

    results = {
        "model": model,
        "scaler": scaler,
        "class_distribution": class_dist.to_dict(),
        "imbalance_ratio": imbalance_ratio,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "roc_auc": roc_auc,
        "cv_f1_mean": cv_scores.mean(),
        "cv_f1_std": cv_scores.std(),
        "feature_importance": feature_importance.to_dict("records"),
        "plots": {
            "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": roc_auc},
            "confusion_matrix": cm.tolist(),
        },
        "test_data": {
            "y_test": y_test.tolist(),
            "y_pred": y_pred.tolist(),
            "y_prob": y_prob.tolist(),
        },
    }

    return results


def format_classification_report(results):
    """格式化分类结果为 Markdown 报告"""
    md = ["## 分类分析\n\n"]

    # 1. 数据概况
    md.append("### 数据概况\n\n")
    md.append(f"- 类别分布: {results['class_distribution']}\n")
    md.append(f"- 不平衡比例: 1:{results['imbalance_ratio']:.1f}\n")
    if results["imbalance_ratio"] > 5:
        md.append("- ⚠️ 检测到类别不平衡，已使用 class_weight='balanced'\n")
    md.append("\n")

    # 2. 混淆矩阵
    cm = results["confusion_matrix"]
    md.append("### 混淆矩阵\n\n")
    md.append("|  | 预测负类 | 预测正类 |\n")
    md.append("|--|---------|---------|\n")
    md.append(f"| 实际负类 | {cm['tn']} | {cm['fp']} |\n")
    md.append(f"| 实际正类 | {cm['fn']} | {cm['tp']} |\n\n")

    # 3. 评估指标
    md.append("### 评估指标\n\n")
    md.append("| 指标 | 值 |\n")
    md.append("|------|-----|\n")
    m = results["metrics"]
    md.append(f"| 准确率 (Accuracy) | {m['accuracy']:.2%} |\n")
    md.append(f"| 精确率 (Precision) | {m['precision']:.2%} |\n")
    md.append(f"| 查全率 (Recall) | {m['recall']:.2%} |\n")
    md.append(f"| F1 分数 | {m['f1']:.2%} |\n")
    md.append(f"| AUC-ROC | {results['roc_auc']:.4f} |\n")
    md.append(
        f"| CV F1 (mean±std) | {results['cv_f1_mean']:.2%} ± {results['cv_f1_std']:.2%} |\n\n"
    )

    # 4. 特征重要性
    md.append("### 特征重要性\n\n")
    md.append("| 特征 | 系数 | 绝对值 |\n")
    md.append("|------|------|--------|\n")
    for feat in results["feature_importance"]:
        md.append(
            f"| {feat['feature']} | {feat['coefficient']:.4f} | {feat['abs_coefficient']:.4f} |\n"
        )
    md.append("\n")

    # 5. 诊断结论
    md.append("### 诊断结论\n\n")
    if results["roc_auc"] < 0.7:
        md.append("- ⚠️ AUC < 0.7，模型区分能力较弱\n")
    elif results["roc_auc"] > 0.9:
        md.append("- ✅ AUC > 0.9，模型区分能力优秀（但需检查是否过拟合）\n")
    else:
        md.append("- ✅ 模型区分能力尚可\n")

    if m["precision"] < 0.5 and m["recall"] < 0.5:
        md.append(
            "- ⚠️ 精确率和查全率均较低，建议重新审视特征或尝试其他模型\n"
        )

    if results["imbalance_ratio"] > 10:
        md.append("- ⚠️ 类别严重不平衡，建议关注查全率而非准确率\n")

    md.append(f"- 交叉验证 F1: {results['cv_f1_mean']:.2%} ± {results['cv_f1_std']:.2%}\n")

    return "".join(md)


# =============================================================================
# 测试数据生成
# =============================================================================


def generate_test_data(n_samples=1000, weights=None, random_state=42):
    """生成测试数据"""
    if weights is None:
        weights = [0.5, 0.5]
    X, y = make_classification(
        n_samples=n_samples,
        n_features=4,
        n_redundant=0,
        n_clusters_per_class=1,
        weights=weights,
        flip_y=0,
        random_state=random_state,
    )
    return X, y


# =============================================================================
# StatLab 测试用例
# =============================================================================


def test_classification_report_generation():
    """测试分类报告生成功能"""
    X, y = generate_test_data(n_samples=500, random_state=42)

    results = classification_with_evaluation(
        X, y, var_names=["特征A", "特征B", "特征C", "特征D"]
    )

    # 验证返回结果的结构
    assert "model" in results
    assert "scaler" in results
    assert "class_distribution" in results
    assert "imbalance_ratio" in results
    assert "confusion_matrix" in results
    assert "metrics" in results
    assert "roc_auc" in results
    assert "cv_f1_mean" in results
    assert "cv_f1_std" in results
    assert "feature_importance" in results
    assert "plots" in results
    assert "test_data" in results

    # 验证混淆矩阵结构
    cm = results["confusion_matrix"]
    assert "tn" in cm
    assert "fp" in cm
    assert "fn" in cm
    assert "tp" in cm

    # 验证各值非负
    assert cm["tn"] >= 0
    assert cm["fp"] >= 0
    assert cm["fn"] >= 0
    assert cm["tp"] >= 0

    # 验证指标范围
    metrics = results["metrics"]
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["precision"] <= 1
    assert 0 <= metrics["recall"] <= 1
    assert 0 <= metrics["f1"] <= 1

    # 验证 AUC 范围
    assert 0 <= results["roc_auc"] <= 1

    # 验证交叉验证分数
    assert 0 <= results["cv_f1_mean"] <= 1
    assert results["cv_f1_std"] >= 0

    # 验证特征重要性
    assert len(results["feature_importance"]) == X.shape[1]
    for feat in results["feature_importance"]:
        assert "feature" in feat
        assert "coefficient" in feat
        assert "abs_coefficient" in feat


def test_classification_report_markdown():
    """测试 Markdown 报告格式"""
    X, y = generate_test_data(n_samples=500, random_state=42)

    results = classification_with_evaluation(
        X, y, var_names=["特征A", "特征B", "特征C", "特征D"]
    )

    report = format_classification_report(results)

    # 验证报告包含必要章节
    assert "## 分类分析" in report
    assert "### 数据概况" in report
    assert "### 混淆矩阵" in report
    assert "### 评估指标" in report
    assert "### 特征重要性" in report
    assert "### 诊断结论" in report

    # 验证包含关键指标
    assert "准确率" in report
    assert "精确率" in report
    assert "查全率" in report
    assert "F1" in report or "F1 分数" in report
    assert "AUC-ROC" in report


def test_roc_curve_plotting():
    """测试 ROC 曲线数据生成"""
    X, y = generate_test_data(n_samples=500, random_state=42)

    results = classification_with_evaluation(X, y)

    # 验证 ROC 数据存在
    assert "roc" in results["plots"]
    roc_data = results["plots"]["roc"]

    # 验证 ROC 数据结构
    assert "fpr" in roc_data
    assert "tpr" in roc_data
    assert "auc" in roc_data

    fpr = roc_data["fpr"]
    tpr = roc_data["tpr"]
    auc_value = roc_data["auc"]

    # 验证 FPR 和 TPR 长度相同
    assert len(fpr) == len(tpr)

    # 验证 FPR 和 TPR 在 [0, 1] 范围内
    assert all(0 <= val <= 1 for val in fpr)
    assert all(0 <= val <= 1 for val in tpr)

    # 验证 ROC 曲线起始于 (0,0) 并终止于 (1,1)
    assert fpr[0] == 0
    assert tpr[0] == 0
    assert fpr[-1] == 1
    assert tpr[-1] == 1

    # 验证 AUC 值与 results 中的一致
    assert abs(auc_value - results["roc_auc"]) < 1e-10

    # 验证 AUC > 0.5（对于可分离数据）
    assert auc_value > 0.5


def test_class_imbalance_detection():
    """测试类别不平衡检测"""
    # 不平衡数据（1:9）
    X_imbalanced, y_imbalanced = generate_test_data(
        n_samples=1000, weights=[0.9, 0.1], random_state=42
    )

    results_imbalanced = classification_with_evaluation(X_imbalanced, y_imbalanced)

    # 验证不平衡比例
    assert results_imbalanced["imbalance_ratio"] > 5

    # 验证使用了 class_weight='balanced'
    assert results_imbalanced["model"].class_weight == "balanced"

    # 验证报告包含不平衡警告
    report = format_classification_report(results_imbalanced)
    assert "class_weight='balanced'" in report

    # 平衡数据
    X_balanced, y_balanced = generate_test_data(
        n_samples=500, weights=[0.5, 0.5], random_state=42
    )

    results_balanced = classification_with_evaluation(X_balanced, y_balanced)

    # 验证不平衡比例接近 1
    assert results_balanced["imbalance_ratio"] < 2

    # 验证没有使用 class_weight
    assert results_balanced["model"].class_weight is None


def test_cross_validation_stability():
    """测试交叉验证的稳定性"""
    X, y = generate_test_data(n_samples=500, random_state=42)

    results = classification_with_evaluation(X, y)

    # 验证交叉验证分数合理
    cv_mean = results["cv_f1_mean"]
    cv_std = results["cv_f1_std"]

    # 均值应在合理范围内
    assert 0 <= cv_mean <= 1

    # 标准差不应过大（表示模型稳定）
    assert cv_std < 0.5  # 通常 CV std 应该小于 0.5

    # 交叉验证分数应与测试集分数接近
    test_f1 = results["metrics"]["f1"]
    assert abs(cv_mean - test_f1) < 0.3  # 允许一定差异


def test_feature_importance_ranking():
    """测试特征重要性排序"""
    X, y = generate_test_data(n_samples=500, random_state=42)

    var_names = ["特征A", "特征B", "特征C", "特征D"]
    results = classification_with_evaluation(X, y, var_names=var_names)

    # 验证特征名称正确（顺序可能因模型拟合结果而不同）
    importance_df = pd.DataFrame(results["feature_importance"])
    assert set(importance_df["feature"]) == set(var_names)

    # 验证按绝对值排序（降序）
    abs_coeffs = importance_df["abs_coefficient"].values
    assert all(abs_coeffs[i] >= abs_coeffs[i + 1] for i in range(len(abs_coeffs) - 1))


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

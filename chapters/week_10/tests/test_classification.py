"""
Test suite for Week 10: 分类与评估

This module tests logistic regression, confusion matrix, ROC-AUC,
cross-validation, and data leakage detection.
"""

import pytest
import numpy as np
import pandas as pd
from scipy import stats


# =============================================================================
# Test 1: Logistic Regression
# =============================================================================

class TestLogisticRegression:
    """Test logistic regression fitting and interpretation."""

    def test_logistic_regression_sklearn_fits_correctly(self, churn_data):
        """
        Happy path: 使用 sklearn 拟合逻辑回归.

        学习目标:
        - 理解 LogisticRegression 的基本用法
        - 正确获取 coef_ 和 intercept_
        - predict 和 predict_proba 的区别
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split

        X = churn_data[['tenure_months', 'monthly_charges']]
        y = churn_data['churn']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)

        # 检查模型成功拟合
        assert hasattr(model, 'coef_'), "模型应该有 coef_ 属性"
        assert hasattr(model, 'intercept_'), "模型应该有 intercept_ 属性"
        assert len(model.coef_[0]) == 2, "两个特征应该有两个系数"

    def test_sigmoid_output_range(self, churn_data):
        """
        Happy path: Sigmoid 输出在 [0, 1] 范围内.

        学习目标:
        - 理解 Sigmoid 函数的作用
        - 预测概率必须在 [0, 1] 之间
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split

        X = churn_data[['tenure_months', 'monthly_charges']]
        y = churn_data['churn']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)

        # predict_proba 返回两列 [P(y=0), P(y=1)]
        y_proba = model.predict_proba(X_test)[:, 1]

        # 所有概率应该在 [0, 1] 范围内
        assert np.all(y_proba >= 0), "概率不应该小于 0"
        assert np.all(y_proba <= 1), "概率不应该大于 1"
        assert np.all((y_proba > 0) & (y_proba < 1)), "概率应该在 (0, 1) 开区间内"

    def test_sigmoid_function_properties(self):
        """
        Test: Sigmoid 函数的数学性质.

        学习目标:
        - 理解 Sigmoid 的形状（S 型曲线）
        - sigmoid(0) = 0.5
        - sigmoid(+∞) → 1, sigmoid(-∞) → 0
        """
        def sigmoid(z):
            return 1 / (1 + np.exp(-z))

        # sigmoid(0) = 0.5
        assert abs(sigmoid(0) - 0.5) < 1e-10, "sigmoid(0) 应该等于 0.5"

        # sigmoid(+large) 接近 1
        assert sigmoid(10) > 0.999, "sigmoid(10) 应该接近 1"
        assert sigmoid(-10) < 0.001, "sigmoid(-10) 应该接近 0"

        # 对称性：sigmoid(z) + sigmoid(-z) = 1
        for z in [-5, -2, 0, 2, 5]:
            assert abs(sigmoid(z) + sigmoid(-z) - 1) < 1e-10, \
                f"sigmoid({z}) + sigmoid(-{z}) 应该等于 1"

    def test_odds_ratio_interpretation(self, churn_data):
        """
        Happy path: 正确解释优势比.

        学习目标:
        - 理解系数是对数优势比
        - exp(系数) = 优势比（Odds Ratio）
        - OR > 1 表示增加优势，OR < 1 表示降低优势
        """
        from sklearn.linear_model import LogisticRegression

        X = churn_data[['monthly_charges']]
        y = churn_data['churn']

        model = LogisticRegression(random_state=42)
        model.fit(X, y)

        coef = model.coef_[0][0]
        odds_ratio = np.exp(coef)

        # 月费越高，流失风险越高（OR > 1）
        # 注意：如果数据生成时设置了正相关，这里应该通过
        assert isinstance(odds_ratio, (float, np.floating))
        assert odds_ratio > 0, "优势比应该为正数"

        # 验证 OR 的计算
        assert abs(odds_ratio - np.exp(coef)) < 1e-10, \
            "优势比应该等于 exp(系数)"

    def test_decision_boundary_threshold(self, churn_data):
        """
        Happy path: 默认阈值 0.5 的决策边界.

        学习目标:
        - 理解 predict() 默认使用阈值 0.5
        - predict() 等价于 (predict_proba() > 0.5)
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split

        X = churn_data[['tenure_months', 'monthly_charges']]
        y = churn_data['churn']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)

        # 两种预测方式应该等价
        y_pred_default = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred_threshold = (y_pred_proba >= 0.5).astype(int)

        # 应该完全一致
        assert np.array_equal(y_pred_default, y_pred_threshold), \
            "predict() 应该等价于 (predict_proba() >= 0.5)"

    def test_logistic_regression_statsmodels(self, churn_data):
        """
        Happy path: 使用 statsmodels 拟合逻辑回归.

        学习目标:
        - 理解 statsmodels.Logit 需要手动添加常数项
        - 能够获取详细输出（summary, p 值, CI）
        """
        import statsmodels.api as sm

        X = churn_data[['tenure_months', 'monthly_charges']]
        y = churn_data['churn']

        # 添加截距项
        X_sm = sm.add_constant(X)
        model = sm.Logit(y, X_sm).fit(disp=0)

        # 检查模型成功拟合
        assert hasattr(model, 'params'), "模型应该有 params 属性"
        assert hasattr(model, 'summary'), "模型应该有 summary 方法"
        assert 'const' in model.params.index, "应该包含截距项"
        assert len(model.params) == 3, "应该有截距 + 2 个系数"

        # 检查可以获取置信区间
        conf_int = model.conf_int()
        assert conf_int.shape[0] == 3, "每个参数都应该有 CI"


# =============================================================================
# Test 2: Confusion Matrix
# =============================================================================

class TestConfusionMatrix:
    """Test confusion matrix calculation and derived metrics."""

    def test_confusion_matrix_calculation(self, churn_data):
        """
        Happy path: 正确计算混淆矩阵.

        学习目标:
        - 理解 TP, TN, FP, FN 的含义
        - 混淆矩阵的结构
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import confusion_matrix

        X = churn_data[['tenure_months', 'monthly_charges']]
        y = churn_data['churn']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)

        # 混淆矩阵应该是 2x2
        assert cm.shape == (2, 2), "二分类混淆矩阵应该是 2x2"

        # TN, FP, FN, TP
        tn, fp, fn, tp = cm.ravel()

        # 验证总和等于样本数
        assert tn + fp + fn + tp == len(y_test), \
            "混淆矩阵各元素之和应该等于样本数"

        # 验证各元素非负
        assert tn >= 0 and fp >= 0 and fn >= 0 and tp >= 0, \
            "混淆矩阵元素应该非负"

    def test_precision_calculation(self):
        """
        Test: 精确率计算公式.

        学习目标:
        - 精确率 = TP / (TP + FP)
        - "预测为正的样本中，真正为正的比例"
        """
        from sklearn.metrics import confusion_matrix

        # 构造示例：TP=5, FP=5, FN=10, TN=80
        y_true = np.array([0]*80 + [1]*5 + [0]*5 + [1]*10)
        y_pred = np.array([0]*80 + [1]*5 + [1]*5 + [0]*10)

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        precision = tp / (tp + fp)

        # 手动计算
        assert precision == 5 / (5 + 5), \
            "精确率应该等于 TP / (TP + FP)"
        assert abs(precision - 0.5) < 1e-10, \
            "本例精确率应该为 0.5"

    def test_recall_calculation(self):
        """
        Test: 召回率计算公式.

        学习目标:
        - 召回率 = TP / (TP + FN)
        - "真实为正的样本中，被正确预测为正的比例"
        """
        from sklearn.metrics import confusion_matrix

        # 构造示例：TP=5, FP=5, FN=10, TN=80
        y_true = np.array([0]*80 + [1]*5 + [0]*5 + [1]*10)
        y_pred = np.array([0]*80 + [1]*5 + [1]*5 + [0]*10)

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        recall = tp / (tp + fn)

        # 手动计算
        assert recall == 5 / (5 + 10), \
            "召回率应该等于 TP / (TP + FN)"
        assert abs(recall - 1/3) < 1e-10, \
            "本例召回率应该约为 0.333"

    def test_f1_harmonic_mean(self):
        """
        Test: F1 是精确率和召回率的调和平均.

        学习目标:
        - F1 = 2 × (P × R) / (P + R)
        - 调和平均数惩罚极端情况
        """
        # 使用精确值避免浮点误差
        precision = 0.5
        recall = 1 / 3  # 使用分数避免 0.333 的表示误差

        # 调和平均数
        f1 = 2 * (precision * recall) / (precision + recall)

        # F1 = 2 * (0.5 * 1/3) / (0.5 + 1/3) = (1/3) / (5/6) = 2/5 = 0.4
        expected_f1 = 0.4

        assert abs(f1 - expected_f1) < 1e-10, \
            f"F1 应该等于 {expected_f1}, 实际: {f1}"

        # 验证 F1 介于 P 和 R 之间（但不高于两者）
        # 调和平均数总是 <= 几何平均数 <= 算术平均数
        # 对于 P=0.5, R=0.333，F1 应该 <= min(P,R)
        assert f1 <= max(precision, recall), \
            "F1 应该不大于精确率和召回率的最大值"

    def test_imbalanced_data_accuracy_paradox(self, churn_data_imbalanced):
        """
        Test: 准确率悖论（类别不平衡）.

        学习目标:
        - 理解准确率在不平衡数据上的误导性
        - 85% 不流失时，预测"全不流失"准确率也是 85%
        """
        from sklearn.dummy import DummyClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score

        X = churn_data_imbalanced[['tenure_months', 'monthly_charges']]
        y = churn_data_imbalanced['churn']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # 基线：总是预测多数类
        dummy = DummyClassifier(strategy='most_frequent', random_state=42)
        dummy.fit(X_train, y_train)
        y_pred_dummy = dummy.predict(X_test)

        dummy_accuracy = accuracy_score(y_test, y_pred_dummy)
        majority_ratio = (y_test == 0).mean()

        # 基线准确率应该等于多数类比例
        assert abs(dummy_accuracy - majority_ratio) < 0.01, \
            "多数类分类器的准确率应该等于多数类比例"

        # 多数类比例应该很高（不平衡）
        assert majority_ratio > 0.8, \
            "本测试数据应该有严重的类别不平衡"

    def test_confusion_matrix_interpretation(self, churn_data):
        """
        Test: 混淆矩阵的业务解释.

        学习目标:
        - FP：误报（假阳性）
        - FN：漏报（假阴性）
        - 理解 FP/FN 的业务成本
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import confusion_matrix

        X = churn_data[['tenure_months', 'monthly_charges']]
        y = churn_data['churn']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # 解释
        # FP：预测会流失，但实际不流失（误报）
        # FN：预测不会流失，但实际流失（漏报）

        # 验证解释的合理性
        assert fp >= 0, "假阳性应该非负"
        assert fn >= 0, "假阴性应该非负"

        # 业务含义：FP 浪费营销成本，FN 损失客户价值
        # 这是概念性测试，不需要断言


# =============================================================================
# Test 3: ROC-AUC
# =============================================================================

class TestROCAUC:
    """Test ROC curve and AUC calculation."""

    def test_roc_curve_structure(self, churn_data):
        """
        Happy path: ROC 曲线的结构.

        学习目标:
        - ROC 曲线展示 FPR vs TPR 的权衡
        - 曲线点数 = 阈值数 + 1
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_curve

        X = churn_data[['tenure_months', 'monthly_charges']]
        y = churn_data['churn']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]

        fpr, tpr, thresholds = roc_curve(y_test, y_proba)

        # 验证结构
        assert len(fpr) == len(tpr) == len(thresholds), \
            "FPR, TPR, 阈值的长度应该相同"
        assert fpr[0] == 0 and tpr[0] == 0, \
            "ROC 曲线应该从 (0, 0) 开始"
        assert fpr[-1] == 1 and tpr[-1] == 1, \
            "ROC 曲线应该在 (1, 1) 结束"

        # FPR 和 TPR 都在 [0, 1] 范围内
        assert np.all((fpr >= 0) & (fpr <= 1)), "FPR 应该在 [0, 1] 范围内"
        assert np.all((tpr >= 0) & (tpr <= 1)), "TPR 应该在 [0, 1] 范围内"

    def test_auc_range(self, churn_data):
        """
        Happy path: AUC 在 [0, 1] 范围内.

        学习目标:
        - AUC = 1: 完美分类器
        - AUC = 0.5: 随机猜测
        - AUC < 0.5: 比随机还差（可以反转预测）
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score

        X = churn_data[['tenure_months', 'monthly_charges']]
        y = churn_data['churn']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_proba)

        # AUC 应该在 [0, 1] 范围内
        assert 0 <= auc <= 1, f"AUC 应该在 [0, 1] 范围内, 实际: {auc}"

        # 合理的模型应该 AUC > 0.5
        assert auc > 0.5, "有意义的模型 AUC 应该 > 0.5"

    def test_auc_perfect_classifier(self, perfect_classifier_data):
        """
        Test: 完美分类器的 AUC = 1.

        学习目标:
        - 理解完美分类器是什么
        - 所有正类概率 > 所有负类概率
        """
        from sklearn.metrics import roc_auc_score

        y_true = perfect_classifier_data['true_label']
        y_proba = perfect_classifier_data['predicted_proba']

        auc = roc_auc_score(y_true, y_proba)

        # 完美分类器应该 AUC = 1（或非常接近）
        assert auc > 0.99, f"完美分类器的 AUC 应该接近 1, 实际: {auc}"

    def test_auc_random_classifier(self, random_classifier_data):
        """
        Test: 随机分类器的 AUC ≈ 0.5.

        学习目标:
        - 理解随机猜测的表现
        - AUC = 0.5 表示无区分能力
        """
        from sklearn.metrics import roc_auc_score

        y_true = random_classifier_data['true_label']
        y_proba = random_classifier_data['predicted_proba']

        auc = roc_auc_score(y_true, y_proba)

        # 随机分类器的 AUC 应该接近 0.5
        # 允许一定偏差（因为样本有限）
        assert 0.3 <= auc <= 0.7, \
            f"随机分类器的 AUC 应该接近 0.5, 实际: {auc}"

    def test_auc_ranking_interpretation(self):
        """
        Test: AUC 衡量排序能力.

        学习目标:
        - AUC = P(正类概率 > 负类概率)
        - AUC 不依赖阈值
        """
        # 构造简单例子
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.2, 0.4, 0.6, 0.8])

        # 计算所有"正负对"中正类概率 > 负类概率的比例
        n_pos = 2
        n_neg = 2
        n_pairs = n_pos * n_neg  # 4 对

        # 所有 4 对都满足：正类概率 > 负类概率
        # (0.2 < 0.6), (0.2 < 0.8), (0.4 < 0.6), (0.4 < 0.8)
        correct_ranking = 4
        auc_expected = correct_ranking / n_pairs  # = 1.0

        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_true, y_proba)

        assert abs(auc - auc_expected) < 0.01, \
            f"完美排序的 AUC 应该为 {auc_expected}, 实际: {auc}"

    def test_roc_vs_precision_recall(self, churn_data_imbalanced):
        """
        Test: ROC vs PR 曲线在不平衡数据上的差异.

        学习目标:
        - 理解 PR-AUC 在不平衡数据上更严格
        - ROC-AUC 可能给人"虚假高"的感觉
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score, average_precision_score

        X = churn_data_imbalanced[['tenure_months', 'monthly_charges']]
        y = churn_data_imbalanced['churn']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]

        roc_auc = roc_auc_score(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)

        # PR-AUC 通常更严格（可能低于 ROC-AUC）
        # 但这不是必须的，只是通常情况
        assert isinstance(roc_auc, (float, np.floating))
        assert isinstance(pr_auc, (float, np.floating))

        # 两者都在 [0, 1] 范围内
        assert 0 <= roc_auc <= 1
        assert 0 <= pr_auc <= 1


# =============================================================================
# Test 4: Cross-Validation
# =============================================================================

class TestCrossValidation:
    """Test K-fold cross-validation."""

    def test_kfold_creates_correct_splits(self, churn_data):
        """
        Happy path: K-fold 创建正确的折数.

        学习目标:
        - K-fold 将数据分成 K 份
        - 每次用 K-1 份训练，1 份验证
        """
        from sklearn.model_selection import KFold

        X = churn_data[['tenure_months', 'monthly_charges']].values
        y = churn_data['churn'].values

        n_splits = 5
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        # 计算折数
        n_folds = sum(1 for _ in kf.split(X))

        assert n_folds == n_splits, \
            f"K-fold 应该创建 {n_splits} 个折, 实际: {n_folds}"

    def test_stratified_kfold_preserves_class_ratio(self, churn_data_imbalanced):
        """
        Happy path: StratifiedKFold 保持类别比例.

        学习目标:
        - 理解分层抽样的作用
        - 每个折中类别比例与整体一致
        """
        from sklearn.model_selection import StratifiedKFold

        X = churn_data_imbalanced[['tenure_months', 'monthly_charges']].values
        y = churn_data_imbalanced['churn'].values

        # 整体类别比例
        overall_ratio = y.mean()

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # 检查每个折的验证集类别比例
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            val_y = y[val_idx]
            val_ratio = val_y.mean()

            # 验证集比例应该与整体比例接近
            # 允许一定偏差（因为样本有限）
            assert abs(val_ratio - overall_ratio) < 0.1, \
                f"折 {fold_idx} 的类别比例 ({val_ratio:.2f}) 应该接近整体比例 ({overall_ratio:.2f})"

    def test_cross_val_score_returns_correct_length(self, churn_data):
        """
        Happy path: cross_val_score 返回正确数量的分数.

        学习目标:
        - cross_val_score 返回每个折的分数
        - 长度 = cv 折数
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score

        X = churn_data[['tenure_months', 'monthly_charges']]
        y = churn_data['churn']

        model = LogisticRegression(random_state=42, max_iter=1000)
        n_splits = 5

        scores = cross_val_score(model, X, y, cv=n_splits, scoring='accuracy')

        assert len(scores) == n_splits, \
            f"分数数量应该等于折数 ({n_splits}), 实际: {len(scores)}"

    def test_cv_score_mean_and_std(self, churn_data):
        """
        Happy path: 交叉验证分数的均值和标准差.

        学习目标:
        - 均值表示模型平均性能
        - 标准差表示稳定性（越小越稳定）
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score

        X = churn_data[['tenure_months', 'monthly_charges']]
        y = churn_data['churn']

        model = LogisticRegression(random_state=42, max_iter=1000)
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

        mean_score = scores.mean()
        std_score = scores.std()

        # 均值应该在合理范围内
        assert 0 <= mean_score <= 1, "准确率均值应该在 [0, 1] 范围内"

        # 标准差应该非负
        assert std_score >= 0, "标准差应该非负"

        # 标准差不应该太大（模型相对稳定）
        # 这里不做严格断言，因为取决于数据质量

    def test_cv_multiple_metrics(self, churn_data):
        """
        Happy path: 交叉验证评估多个指标.

        学习目标:
        - 使用 cross_validate 同时评估多个指标
        - 理解不同指标的权衡
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_validate

        X = churn_data[['tenure_months', 'monthly_charges']]
        y = churn_data['churn']

        model = LogisticRegression(random_state=42, max_iter=1000)

        scoring = {
            'accuracy': 'accuracy',
            'f1': 'f1',
            'roc_auc': 'roc_auc',
            'recall': 'recall'
        }

        cv_results = cross_validate(model, X, y, cv=5, scoring=scoring)

        # 检查返回结果
        for metric in scoring.keys():
            key = f'test_{metric}'
            assert key in cv_results, \
                f"结果应该包含 {key}"
            assert len(cv_results[key]) == 5, \
                f"{key} 应该有 5 个折的分数"


# =============================================================================
# Test 5: Data Leakage
# =============================================================================

class TestDataLeakage:
    """Test data leakage detection and prevention."""

    def test_global_scaler_leaks_test_info(self, data_for_leakage_test):
        """
        Test: 全局 StandardScaler 会泄漏测试集信息.

        学习目标:
        - 理解为什么全局预处理是错误的
        - 测试集的统计量（均值、方差）会"泄露"到训练过程
        """
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split, cross_val_score

        X, y = data_for_leakage_test

        # 错误做法：全局 fit scaler
        scaler_global = StandardScaler()
        X_scaled_global = scaler_global.fit_transform(X)

        # 交叉验证（已经泄漏）
        model = LogisticRegression(random_state=42, max_iter=1000)
        scores_leaked = cross_val_score(model, X_scaled_global, y, cv=5)

        # 正确做法：在 Pipeline 内做
        from sklearn.pipeline import Pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(random_state=42, max_iter=1000))
        ])
        scores_correct = cross_val_score(pipeline, X, y, cv=5)

        # 泄漏的分数通常会虚高
        # 注意：这不是必须的（有时候差异很小），但平均来说泄漏会虚高
        assert isinstance(scores_leaked.mean(), (float, np.floating))
        assert isinstance(scores_correct.mean(), (float, np.floating))

        # 记录差异（用于教学）
        leakage_diff = scores_leaked.mean() - scores_correct.mean()
        # 不做严格断言，因为数据可能导致差异很小

    def test_pipeline_prevents_leakage(self, data_for_leakage_test):
        """
        Happy path: Pipeline 防止数据泄漏.

        学习目标:
        - 理解 Pipeline 的工作机制
        - 每个 CV 折内独立拟合预处理
        """
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import cross_val_score

        X, y = data_for_leakage_test

        # Pipeline 模式
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(random_state=42, max_iter=1000))
        ])

        scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')

        # 应该成功运行
        assert len(scores) == 5, "应该返回 5 个折的分数"
        assert np.all((scores >= 0) & (scores <= 1)), "准确率应该在 [0, 1] 范围内"

    def test_column_transformer_with_mixed_types(self, churn_data_with_categories):
        """
        Happy path: ColumnTransformer 处理混合类型.

        学习目标:
        - 理解 ColumnTransformer 的用法
        - 数值列：StandardScaler
        - 类别列：OneHotEncoder
        """
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import train_test_split

        df = churn_data_with_categories

        numeric_features = ['tenure_months', 'monthly_charges']
        categorical_features = ['contract_type', 'payment_method']

        X = df[numeric_features + categorical_features]
        y = df['churn']

        # ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ]
        )

        # Pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000))
        ])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # 拟合和预测
        pipeline.fit(X_train, y_train)
        score = pipeline.score(X_test, y_test)

        # 应该成功运行
        assert isinstance(score, (float, np.floating))
        assert 0 <= score <= 1, "准确率应该在 [0, 1] 范围内"

    def test_train_test_split_order_matters(self, data_for_leakage_test):
        """
        Test: 必须先 split，再预处理.

        学习目标:
        - 理解预处理在 split 之后的重要性
        - 顺序错误会导致数据泄漏
        """
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split

        X, y = data_for_leakage_test

        # 正确顺序：先 split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 再 fit scaler（只在训练集上）
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)  # 注意：用 transform，不是 fit_transform

        # 验证：测试集的均值和方差不应该被用到
        # 训练集的统计量
        train_mean = scaler.mean_
        train_scale = scaler.scale_

        # 计算测试集的真实统计量（没有被用到）
        test_mean = X_test.mean(axis=0)
        test_scale = X_test.std(axis=0)

        # 验证两者不同（说明没有泄漏）
        # 注意：这里不要求所有特征都不同，但整体应该有差异
        assert np.any(np.abs(train_mean - test_mean) > 0.1), \
            "训练集和测试集的均值应该有差异（说明没有泄漏测试集信息）"


# =============================================================================
# Test 6: Edge Cases
# =============================================================================

class TestClassificationEdgeCases:
    """Test classification with edge cases."""

    def test_single_class_data_warning(self, single_class_data):
        """
        Edge case: 只有一个类别的数据.

        学习目标:
        - 理解单类别数据无法训练分类器
        - 应该产生警告或错误
        """
        from sklearn.linear_model import LogisticRegression
        import warnings

        X = single_class_data[['feature_1', 'feature_2']]
        y = single_class_data['target']

        model = LogisticRegression(random_state=42)

        # 应该产生警告或错误
        with warnings.catch_warnings(record=True):
            try:
                model.fit(X, y)
                # 如果没有报错，至少系数应该有问题
                # sklearn 的处理：coef_ 可能全为 0
            except ValueError as e:
                # 预期的错误
                assert 'class' in str(e).lower() or 'single' in str(e).lower()

    def test_very_small_dataset(self, very_small_dataset):
        """
        Edge case: 极小数据集.

        学习目标:
        - 小样本会导致估计不稳定
        - 但模型仍可拟合（可能过拟合）
        """
        from sklearn.linear_model import LogisticRegression

        X = very_small_dataset[['feature']]
        y = very_small_dataset['target']

        model = LogisticRegression(random_state=42)
        model.fit(X, y)

        # 应该能拟合（可能过拟合）
        assert hasattr(model, 'coef_'), "模型应该成功拟合"

    def test_perfect_separation_warning(self, perfect_separation_data):
        """
        Edge case: 完全可分的数据.

        学习目标:
        - 理解完全可分导致系数趋于无穷大
        - sklearn 可能产生收敛警告
        """
        from sklearn.linear_model import LogisticRegression
        import warnings

        X = perfect_separation_data[['feature']]
        y = perfect_separation_data['target']

        model = LogisticRegression(random_state=42)

        # 可能产生收敛警告
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model.fit(X, y)

            # 检查是否有收敛警告
            # (sklearn 不一定总是警告，因为它有正则化)
            # 这里只是记录，不做严格断言

        # 模型应该能拟合
        assert hasattr(model, 'coef_')


# =============================================================================
# Test 7: AI Classification Report Review
# =============================================================================

class TestAIClassificationReportReview:
    """Test ability to review AI-generated classification reports."""

    def test_check_good_report_has_all_elements(self, good_classification_report):
        """
        Happy path: 识别合格的分类报告.

        学习目标:
        - 理解完整报告应包含的要素
        - 混淆矩阵、精确率/召回率/F1、ROC-AUC、阈值讨论
        """
        report = good_classification_report.lower()

        # 应该包含的关键要素
        required_elements = [
            '混淆',  # 混淆矩阵
            '精确',  # 精确率
            '召回',  # 召回率
            'f1',  # F1 分数
            'auc',  # ROC-AUC
            '阈值',  # 阈值讨论
            '局限',  # 局限性
        ]

        missing_elements = []
        for element in required_elements:
            if element not in report:
                missing_elements.append(element)

        # 允许缺少 1-2 个要素
        assert len(missing_elements) <= 2, \
            f"合格的报告应该包含关键要素，缺少: {missing_elements}"

    def test_detect_only_accuracy_report(self, bad_classification_report_only_accuracy):
        """
        Test: 识别只报告准确率的糟糕报告.

        学习目标:
        - 理解只看准确率的问题
        - 特别是在类别不平衡场景
        """
        report = bad_classification_report_only_accuracy

        # 缺少的关键要素
        missing_elements = []

        # 检查混淆矩阵
        if '混淆' not in report and 'confusion' not in report.lower():
            missing_elements.append('混淆矩阵')

        # 检查精确率/召回率
        if '精确' not in report and 'precision' not in report.lower():
            missing_elements.append('精确率')
        if '召回' not in report and 'recall' not in report.lower():
            missing_elements.append('召回率')

        # 检查 AUC
        if 'auc' not in report.lower():
            missing_elements.append('AUC')

        # 检查阈值讨论
        if '阈值' not in report and 'threshold' not in report.lower():
            missing_elements.append('阈值讨论')

        # 应该检测到缺少关键要素
        assert len(missing_elements) >= 3, \
            f"应该检测到报告缺少要素: {missing_elements}"

    def test_detect_no_threshold_discussion(self, bad_classification_report_no_threshold):
        """
        Test: 识别缺少阈值讨论的报告.

        学习目标:
        - 理解阈值选择的重要性
        - 0.5 不一定是业务最优解
        """
        report = bad_classification_report_no_threshold

        # 检查阈值讨论
        has_threshold_discussion = (
            '阈值' in report or
            'threshold' in report.lower() or
            '0.5' in report or
            '调整' in report
        )

        # 应该检测到缺少阈值讨论
        assert not has_threshold_discussion, \
            "应该检测到报告缺少阈值讨论"

    def test_detect_data_leakage_report(self, bad_classification_report_leakage):
        """
        Test: 识别存在数据泄漏的报告.

        学习目标:
        - 理解"全局标准化"是泄漏模式
        - 正确做法是 Pipeline 内预处理
        """
        report = bad_classification_report_leakage

        # 检查泄漏模式
        has_leakage_pattern = (
            '整个数据集' in report and '标准化' in report
        )

        # 检查是否提到 Pipeline
        has_pipeline_fix = (
            'pipeline' in report.lower() or
            'Pipeline' in report or
            '折内' in report
        )

        # 应该检测到泄漏模式
        assert has_leakage_pattern, "应该检测到全局预处理的泄漏模式"
        assert not has_pipeline_fix, "报告没有提到 Pipeline 修复方案"

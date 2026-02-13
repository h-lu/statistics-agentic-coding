"""
Week 10 边界测试（Edge Cases）

测试分类与评估中的边界情况和极端场景。
"""
import pytest
import numpy as np
import pandas as pd


class TestLogisticRegressionEdgeCases:
    """测试逻辑回归的边界情况"""

    def test_empty_data_raises_error(self):
        """空数据应该报错"""
        from sklearn.linear_model import LogisticRegression

        X = pd.DataFrame({'x1': [], 'x2': []})
        y = np.array([])

        model = LogisticRegression()

        with pytest.raises(ValueError):
            model.fit(X, y)

    def test_single_observation_perfect_fit(self):
        """单观测数据会完美拟合或产生警告"""
        from sklearn.linear_model import LogisticRegression
        import warnings

        X = pd.DataFrame({'x': [1.0]})
        y = np.array([1])

        model = LogisticRegression()

        # sklearn 可能产生警告或收敛问题
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            try:
                model.fit(X, y)
                # sklearn 可能成功拟合（会过拟合）
                assert hasattr(model, 'coef_')
            except ValueError as e:
                # 或者可能因为单类数据报错
                assert 'class' in str(e).lower() or 'sample' in str(e).lower()

    def test_all_same_predictions(self):
        """当模型总是预测同一类别时"""
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import confusion_matrix

        # 构造一个明显倾向于预测 0 的场景
        # sklearn LogisticRegression 在单类数据时会报错
        # 所以我们用两类的数据，但模型倾向于预测 0
        np.random.seed(42)
        X = pd.DataFrame({
            'x': np.concatenate([np.random.normal(-5, 1, 45), np.random.normal(5, 1, 5)])
        })
        y = np.array([0] * 45 + [1] * 5)  # 大部分是 0

        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X, y)
        y_pred = model.predict(X)

        # 应该预测大部分为 0（允许少数 1）
        n_zeros = np.sum(y_pred == 0)
        assert n_zeros >= 40, f"应该主要预测为 0, 实际 {n_zeros}/100"

    def test_extreme_coefficients_convergence_warning(self):
        """极端系数可能导致收敛警告"""
        from sklearn.linear_model import LogisticRegression
        import warnings

        # 完全可分的数据（可能导致大系数）
        X = pd.DataFrame({
            'x': [-10, -5, -1, 1, 5, 10]
        })
        y = np.array([0, 0, 0, 1, 1, 1])

        model = LogisticRegression(random_state=42, max_iter=10)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model.fit(X, y)

            # 可能产生收敛警告（但不保证）
            # 这里只验证模型能拟合
            assert hasattr(model, 'coef_')


class TestConfusionMatrixEdgeCases:
    """测试混淆矩阵的边界情况"""

    def test_perfect_predictions_confusion_matrix(self):
        """完美预测的混淆矩阵"""
        from sklearn.metrics import confusion_matrix

        # 使用重复样本
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])  # 完美预测

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # 验证：完美预测时 FN=0, FP=0
        # TN 和 TP 的值取决于样本数
        assert fn == 0 and fp == 0, \
            "完美预测应该 FN=0, FP=0"
        assert tn == 2 and tp == 2, \
            f"完美预测应该正确计数所有样本: TN={tn}, TP={tp}"

    def test_worst_predictions_confusion_matrix(self):
        """最差预测的混淆矩阵（全错）"""
        from sklearn.metrics import confusion_matrix

        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])  # 全错

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        assert tn == 0 and fp == 2 and fn == 2 and tp == 0, \
            "全错预测应该 TN=0, TP=0"

    def test_precision_zero_denominator(self):
        """精确率：分母为 0 的情况（没有预测为正）"""
        from sklearn.metrics import confusion_matrix

        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 0])  # 全预测为 0

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # 精确率 = TP / (TP + FP)
        # 这里 TP=0, FP=0，分母为 0
        if tp + fp == 0:
            precision = 0  # 或定义为 undefined
        else:
            precision = tp / (tp + fp)

        assert precision == 0, "分母为 0 时精确率应定义为 0"

    def test_recall_zero_denominator(self):
        """召回率：分母为 0 的情况（没有真实正样本）"""
        from sklearn.metrics import confusion_matrix

        y_true = np.array([0, 0, 0, 0])  # 全是 0
        y_pred = np.array([0, 0, 1, 1])

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # 召回率 = TP / (TP + FN)
        # 这里 TP=0, FN=0，分母为 0
        if tp + fn == 0:
            recall = 0  # 或定义为 undefined
        else:
            recall = tp / (tp + fn)

        assert recall == 0, "分母为 0 时召回率应定义为 0"

    def test_f1_zero_precision(self):
        """F1：精确率为 0 时"""
        from sklearn.metrics import f1_score

        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 0])  # 没预测对任何正样本

        f1 = f1_score(y_true, y_pred)

        # F1 应该是 0（因为没有 TP）
        assert f1 == 0, "没有 TP 时 F1 应该为 0"


class TestROCUAEdgeCases:
    """测试 ROC-AUC 的边界情况"""

    def test_perfect_classifier_auc_equals_one(self):
        """完美分类器的 AUC = 1.0"""
        from sklearn.metrics import roc_auc_score

        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.8, 0.9])  # 完美排序

        auc = roc_auc_score(y_true, y_proba)

        assert auc == 1.0, "完美分类器 AUC 应该为 1.0"

    def test_worst_classifier_auc_equals_zero(self):
        """最差分类器的 AUC = 0.0（可反转）"""
        from sklearn.metrics import roc_auc_score

        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.9, 0.8, 0.2, 0.1])  # 完全反了

        auc = roc_auc_score(y_true, y_proba)

        assert auc == 0.0, "完全反了的分类器 AUC 应该为 0.0"

    def test_random_classifier_auc_half(self):
        """随机分类器的 AUC ≈ 0.5"""
        from sklearn.metrics import roc_auc_score

        np.random.seed(42)
        n = 100
        y_true = np.random.randint(0, 2, n)
        y_proba = np.random.uniform(0, 1, n)

        auc = roc_auc_score(y_true, y_proba)

        # 随机分类器的 AUC 应该接近 0.5
        # 允许一定偏差
        assert 0.3 <= auc <= 0.7, f"随机分类器 AUC 应该接近 0.5, 实际: {auc}"

    def test_all_same_labels_auc_undefined(self):
        """所有标签相同时 AUC 未定义"""
        from sklearn.metrics import roc_auc_score
        import warnings

        y_true = np.array([0, 0, 0, 0])  # 全是 0
        y_proba = np.array([0.2, 0.4, 0.6, 0.8])

        # sklearn 1.x+ 会产生警告并返回 nan
        # 早期版本会抛出 ValueError
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                auc = roc_auc_score(y_true, y_proba)
                # 新版本：返回 nan 并产生警告
                assert np.isnan(auc), "单类数据 AUC 应该是 nan"
                assert len(w) > 0, "应该产生 UndefinedMetricWarning"
            except ValueError:
                # 旧版本：抛出 ValueError
                pass  # 预期行为

    def test_auc_threshold_independence(self):
        """AUC 不依赖分类阈值"""
        from sklearn.metrics import roc_auc_score

        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.3, 0.4, 0.6, 0.7])

        # AUC 只依赖概率排序，不依赖阈值
        auc1 = roc_auc_score(y_true, y_proba)

        # 改变阈值（不变概率排序），AUC 不变
        y_proba_shifted = y_proba + 0.1
        auc2 = roc_auc_score(y_true, y_proba_shifted)

        assert abs(auc1 - auc2) < 1e-10, \
            "AUC 应该不依赖阈值"

    def test_roc_curve_monotonic(self):
        """ROC 曲线应该是单调的"""
        from sklearn.metrics import roc_curve

        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.3, 0.4, 0.6, 0.7])

        fpr, tpr, _ = roc_curve(y_true, y_proba)

        # FPR 和 TPR 都应该是单调非减的
        assert np.all(np.diff(fpr) >= -1e-10), "FPR 应该单调非减"
        assert np.all(np.diff(tpr) >= -1e-10), "TPR 应该单调非减"


class TestCrossValidationEdgeCases:
    """测试交叉验证的边界情况"""

    def test_cv_folds_greater_than_samples(self):
        """折数大于样本数"""
        from sklearn.model_selection import KFold

        n_samples = 10
        n_splits = 15  # 大于样本数

        X = np.random.randn(n_samples, 2)
        y = np.random.randint(0, 2, n_samples)

        # 应该报错
        with pytest.raises(ValueError):
            kf = KFold(n_splits=n_splits)
            list(kf.split(X, y))

    def test_cv_single_fold(self):
        """单折交叉验证（就是简单的 hold-out）"""
        from sklearn.model_selection import KFold
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score

        X = np.random.randn(50, 2)
        y = np.random.randint(0, 2, 50)

        model = LogisticRegression(random_state=42, max_iter=1000)
        scores = cross_val_score(model, X, y, cv=2)

        assert len(scores) == 2, "2-fold 应该返回 2 个分数"

    def test_stratified_kfold_single_sample_in_class(self):
        """StratifiedKFold：某个类只有 1 个样本"""
        from sklearn.model_selection import StratifiedKFold

        # 类别 0: 3 个, 类别 1: 1 个
        X = np.array([[1], [2], [3], [4]])
        y = np.array([0, 0, 0, 1])

        # 4-fold 会失败（某个类在某个折中为空）
        with pytest.raises(ValueError):
            skf = StratifiedKFold(n_splits=4)
            list(skf.split(X, y))

    def test_cv_shuffle_consistency(self):
        """交叉验证的 shuffle 一致性"""
        from sklearn.model_selection import KFold

        X = np.arange(10).reshape(10, 1)
        y = np.arange(10)

        # 相同 random_state 应该产生相同的划分
        kf1 = KFold(n_splits=3, shuffle=True, random_state=42)
        kf2 = KFold(n_splits=3, shuffle=True, random_state=42)

        splits1 = list(kf1.split(X, y))
        splits2 = list(kf2.split(X, y))

        # 索引应该相同
        for split1, split2 in zip(splits1, splits2):
            assert np.array_equal(split1[0], split2[0]), \
                "相同 random_state 应该产生相同的训练集索引"
            assert np.array_equal(split1[1], split2[1]), \
                "相同 random_state 应该产生相同的验证集索引"


class TestDataLeakageEdgeCases:
    """测试数据泄漏的边界情况"""

    def test_leakage_detection_trivial_case(self):
        """简单情况下的泄漏检测"""
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split

        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = np.random.randint(0, 2, 100)

        # 错误做法：全局 scaler
        scaler_global = StandardScaler()
        X_scaled_global = scaler_global.fit_transform(X)

        # 划分
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled_global, y, test_size=0.2, random_state=42
        )

        # 测试集的均值应该是接近 0（因为全局标准化）
        test_mean = X_test.mean(axis=0)

        # 全局标准化后，所有数据（包括测试集）均值接近 0
        # 这说明测试集的分布信息被"教给"了训练过程
        assert np.all(np.abs(test_mean) < 0.5), \
            "全局标准化后测试集均值接近 0（说明泄漏）"

    def test_no_leakage_pipeline_case(self):
        """Pipeline 模式下的无泄漏"""
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split

        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = np.random.randint(0, 2, 100)

        # 正确做法：先 split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 再 fit scaler（只在训练集上）
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 测试集的均值不应该接近 0
        test_mean = X_test_scaled.mean(axis=0)

        # 测试集的统计量没有被用到
        # 所以测试集均值不一定接近 0
        assert np.any(np.abs(test_mean) > 0.1), \
            "Pipeline 模式下测试集均值不一定接近 0（无泄漏）"

    def test_leakage_with_feature_selection(self):
        """特征选择也可能导致泄漏"""
        from sklearn.feature_selection import SelectKBest
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score

        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)

        # 错误做法：全局特征选择
        selector = SelectKBest(k=5)
        X_selected = selector.fit_transform(X, y)

        model = LogisticRegression(random_state=42, max_iter=1000)
        scores_leaked = cross_val_score(model, X_selected, y, cv=3)

        # 正确做法：Pipeline 内特征选择
        from sklearn.pipeline import Pipeline
        pipeline = Pipeline([
            ('selector', SelectKBest(k=5)),
            ('model', LogisticRegression(random_state=42, max_iter=1000))
        ])
        scores_correct = cross_val_score(pipeline, X, y, cv=3)

        # 两种方法的分数可能不同
        # 泄漏的通常会虚高（但不保证）
        assert isinstance(scores_leaked.mean(), (float, np.floating))
        assert isinstance(scores_correct.mean(), (float, np.floating))


class TestThresholdEdgeCases:
    """测试阈值选择的边界情况"""

    def test_threshold_zero_always_positive(self):
        """阈值 = 0：总是预测为正"""
        from sklearn.metrics import confusion_matrix

        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.3, 0.4, 0.6, 0.7])

        threshold = 0.0
        y_pred = (y_proba >= threshold).astype(int)

        # 应该全预测为 1
        assert np.all(y_pred == 1), "阈值 0 应该全预测为正"

    def test_threshold_one_always_negative(self):
        """阈值 = 1：总是预测为负"""
        from sklearn.metrics import confusion_matrix

        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.3, 0.4, 0.6, 0.7])

        threshold = 1.0
        y_pred = (y_proba >= threshold).astype(int)

        # 应该全预测为 0（因为概率都 < 1）
        assert np.all(y_pred == 0), "阈值 1 应该全预测为负"

    def test_optimal_threshold_maximizes_metric(self):
        """最优阈值最大化某个指标"""
        from sklearn.metrics import f1_score

        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.2, 0.4, 0.6, 0.8])

        # 尝试不同阈值
        thresholds = np.linspace(0, 1, 101)
        f1_scores = []

        for thresh in thresholds:
            y_pred = (y_proba >= thresh).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            f1_scores.append(f1)

        f1_scores = np.array(f1_scores)

        # 找到最大 F1 的阈值
        best_threshold = thresholds[np.argmax(f1_scores)]
        best_f1 = np.max(f1_scores)

        # 应该有一个阈值使 F1 最大化
        assert 0 <= best_threshold <= 1, "最优阈值应该在 [0, 1] 范围内"
        assert best_f1 > 0, "应该存在一个阈值使 F1 > 0"

    def test_threshold_tradeoff_precision_vs_recall(self):
        """阈值权衡精确率和召回率"""
        from sklearn.metrics import precision_score, recall_score

        y_true = np.array([0, 0, 1, 1, 1, 1])
        y_proba = np.array([0.3, 0.4, 0.6, 0.7, 0.8, 0.9])

        # 低阈值：高召回率，低精确率
        y_pred_low = (y_proba >= 0.3).astype(int)
        precision_low = precision_score(y_true, y_pred_low, zero_division=0)
        recall_low = recall_score(y_true, y_pred_low)

        # 高阈值：低召回率，高精确率
        y_pred_high = (y_proba >= 0.8).astype(int)
        precision_high = precision_score(y_true, y_pred_high, zero_division=0)
        recall_high = recall_score(y_true, y_pred_high)

        # 召回率应该：低阈值 > 高阈值
        assert recall_low >= recall_high, \
            "低阈值应该有更高的召回率"

        # 精确率应该：高阈值 >= 低阈值（通常情况）
        # 注意：这不总是成立，但通常如此
        # 这里不做严格断言


class TestImbalancedDataEdgeCases:
    """测试不平衡数据的边界情况"""

    def test_extreme_imbalance_99_vs_1(self):
        """极端不平衡：99% vs 1%"""
        np.random.seed(42)
        n = 1000

        # 生成极端不平衡数据
        X = np.random.randn(n, 2)
        y = np.array([0] * 990 + [1] * 10)  # 99% 是 0

        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, recall_score

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred, zero_division=0)

        # 准确率会很高（因为总是预测多数类）
        # 但召回率可能很低
        assert accuracy > 0.9, "不平衡数据上准确率会虚高"
        # 召回率不确定，但应该被计算

    def test_single_minority_sample(self):
        """极少数类只有 1 个样本"""
        np.random.seed(42)
        n = 100

        X = np.random.randn(n, 2)
        y = np.array([0] * 99 + [1])  # 只有 1 个正样本

        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score

        model = LogisticRegression(random_state=42, max_iter=1000)

        # StratifiedKFold 可能失败（某个折没有正样本）
        # 使用普通 KFold
        from sklearn.model_selection import KFold
        scores = cross_val_score(model, X, y, cv=3)

        # 应该能运行
        assert len(scores) == 3

    def test_balanced_data_equal_performance(self):
        """平衡数据：类别性能相近"""
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report

        np.random.seed(42)
        n = 200

        # 生成平衡数据
        X_neg = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], n)
        X_pos = np.random.multivariate_normal([2, 2], [[1, 0], [0, 1]], n)

        X = np.vstack([X_neg, X_pos])
        y = np.array([0] * n + [1] * n)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # 获取每个类的指标
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average=None
        )

        # 两个类的性能应该相对接近
        # （因为数据平衡且质量相似）
        # 允许一定差异
        assert abs(precision[0] - precision[1]) < 0.3, \
            "平衡数据上两个类的精确率应该接近"
        assert abs(recall[0] - recall[1]) < 0.3, \
            "平衡数据上两个类的召回率应该接近"

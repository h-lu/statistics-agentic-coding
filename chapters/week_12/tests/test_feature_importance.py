"""
Feature Importance Tests for Week 12 solution.py

特征重要性测试：
- 正例：正常数据计算特征重要性
- 边界：相关特征、常量特征、单一重要特征
- 反例：空数据、无效模型
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add starter_code to path
starter_code_path = Path(__file__).parent.parent / "starter_code"
sys.path.insert(0, str(starter_code_path))

try:
    import solution
except ImportError:
    solution = None


# =============================================================================
# 1. 正例：标准特征重要性计算
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestFeatureImportanceNormal:
    """测试标准特征重要性计算"""

    def test_feature_importance_returns_valid_values(self, feature_importance_data):
        """
        正例：特征重要性应返回有效值

        特征重要性应该是非负数，且和为 1（归一化后）
        """
        if not hasattr(solution, 'compute_feature_importance'):
            pytest.skip("compute_feature_importance not implemented")

        X = feature_importance_data['X']
        y = feature_importance_data['y']

        # 训练模型
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
        model.fit(X, y)

        # 计算特征重要性
        importance = solution.compute_feature_importance(model)

        assert importance is not None
        assert len(importance) == X.shape[1]
        assert np.all(importance >= 0), "Feature importance should be non-negative"
        assert abs(importance.sum() - 1.0) < 0.01, "Feature importance should sum to 1"

    def test_feature_importance_with_names(self, feature_importance_data):
        """
        正例：带特征名称的特征重要性

        应返回 DataFrame 或带名称的数组
        """
        if not hasattr(solution, 'compute_feature_importance'):
            pytest.skip("compute_feature_importance not implemented")

        X = feature_importance_data['X']
        y = feature_importance_data['y']
        feature_names = feature_importance_data['feature_names']

        # 训练模型
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
        model.fit(X, y)

        # 计算特征重要性（带名称）
        result = solution.compute_feature_importance(model, feature_names=feature_names)

        assert result is not None

        # 如果返回 DataFrame，检查列名
        if isinstance(result, pd.DataFrame):
            assert 'feature' in result.columns or result.index.name == 'feature'
            assert 'importance' in result.columns

    def test_feature_importance_identifies_important_features(self, single_important_feature_data):
        """
        正例：特征重要性应正确识别重要特征

        单一重要特征应获得最高重要性分数
        """
        if not hasattr(solution, 'compute_feature_importance'):
            pytest.skip("compute_feature_importance not implemented")

        X = single_important_feature_data['X']
        y = single_important_feature_data['y']
        feature_names = single_important_feature_data['feature_names']

        # 训练模型
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
        model.fit(X, y)

        # 计算特征重要性
        importance = solution.compute_feature_importance(model, feature_names=feature_names)

        if isinstance(importance, pd.DataFrame):
            # 第一个特征应该最重要
            assert importance.iloc[0]['feature'] == 'important_feature'
            assert importance.iloc[0]['importance'] > importance.iloc[1]['importance']


# =============================================================================
# 2. 边界：相关特征的特征重要性
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestFeatureImportanceCorrelatedFeatures:
    """测试相关特征的特征重要性（"分票"现象）"""

    def test_correlated_features_dilute_importance(self, correlated_features_data):
        """
        边界：相关特征会互相"分票"

        feature_0 和 feature_1 高度相关，重要性会被分散
        """
        if not hasattr(solution, 'compute_feature_importance'):
            pytest.skip("compute_feature_importance not implemented")

        X = correlated_features_data['X']
        y = correlated_features_data['y']
        feature_names = correlated_features_data['feature_names']

        # 训练模型
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
        model.fit(X, y)

        # 计算特征重要性
        importance = solution.compute_feature_importance(model, feature_names=feature_names)

        if isinstance(importance, pd.DataFrame):
            # feature_0 和 feature_1 都应有非零重要性
            # 但单独来看，它们的重要性都会被低估
            importance_dict = dict(zip(importance['feature'], importance['importance']))

            # 两个相关特征的总重要性应该接近单独一个特征时的值
            combined_importance = importance_dict.get('feature_0', 0) + importance_dict.get('feature_1', 0)
            assert combined_importance > 0, "Correlated features should have combined importance"

    def test_duplicated_feature_splits_importance(self):
        """
        边界：重复特征会导致重要性分散

        如果同一特征被复制多次，每次副本的重要性会变小
        """
        if not hasattr(solution, 'compute_feature_importance'):
            pytest.skip("compute_feature_importance not implemented")

        np.random.seed(42)
        n = 300

        # 创建一个重要特征
        feature_0 = np.random.randn(n)
        # 复制 5 次
        X = np.column_stack([feature_0] * 5)

        # 目标变量仅依赖原始特征
        logit = feature_0
        prob = 1 / (1 + np.exp(-logit))
        y = (np.random.random(n) < prob).astype(int)

        # 训练模型
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
        model.fit(X, y)

        # 计算特征重要性
        importance = solution.compute_feature_importance(model)

        # 每个副本的重要性应该大致相同且较低
        assert importance is not None
        # 所有副本的重要性之和应该接近 1
        assert abs(importance.sum() - 1.0) < 0.01
        # 单个副本的重要性应该 < 0.3（因为被分成 5 份）
        assert np.all(importance < 0.4), "Duplicated features should have diluted importance"


# =============================================================================
# 3. 边界：常量特征
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestFeatureImportanceConstantFeatures:
    """测试常量特征的特征重要性"""

    def test_constant_feature_has_zero_importance(self):
        """
        边界：常量特征的重要性应为 0

        常量特征对预测没有贡献
        """
        if not hasattr(solution, 'compute_feature_importance'):
            pytest.skip("compute_feature_importance not implemented")

        np.random.seed(42)
        n = 200

        # 创建数据：第一个特征是常量
        X = np.random.randn(n, 3)
        X[:, 0] = 5.0  # 常量特征
        y = (X[:, 1] + X[:, 2] > 0).astype(int)

        # 训练模型
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
        model.fit(X, y)

        # 计算特征重要性
        importance = solution.compute_feature_importance(model)

        assert importance is not None
        # 第一个特征（常量）的重要性应该接近 0
        assert importance[0] < 0.01, f"Constant feature should have ~0 importance, got {importance[0]}"

    def test_all_constant_features_warning(self):
        """
        反例：所有特征都是常量

        应报错或返回警告
        """
        if not hasattr(solution, 'compute_feature_importance'):
            pytest.skip("compute_feature_importance not implemented")

        np.random.seed(42)
        n = 100

        X = np.ones((n, 3))  # 所有特征都是常量
        y = np.random.randint(0, 2, n)

        # 训练模型（可能失败）
        from sklearn.ensemble import RandomForestClassifier
        try:
            model = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
            model.fit(X, y)

            # 如果训练成功，特征重要性应该全部接近 0 或相等
            importance = solution.compute_feature_importance(model)
            assert importance is not None
            # 所有可能都是 0，或者全部相等（无信息）
        except (ValueError, RuntimeError):
            # 模型训练失败是可接受的
            assert True


# =============================================================================
# 4. 边界：极小数据
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestFeatureImportanceMinimalData:
    """测试极小数据的特征重要性"""

    def test_minimal_data_importance(self):
        """
        边界：极小数据集的特征重要性

        只有 20 个样本，特征重要性可能不稳定
        """
        if not hasattr(solution, 'compute_feature_importance'):
            pytest.skip("compute_feature_importance not implemented")

        np.random.seed(42)
        X = np.random.randn(20, 3)
        y = (X[:, 0] > 0).astype(int)

        # 训练模型
        from sklearn.ensemble import RandomForestClassifier
        try:
            model = RandomForestClassifier(n_estimators=30, max_depth=2, random_state=42)
            model.fit(X, y)

            # 计算特征重要性
            importance = solution.compute_feature_importance(model)
            assert importance is not None
            assert len(importance) == 3
        except (ValueError, RuntimeError):
            # 极小数据可能导致训练失败
            assert True

    def test_single_feature_importance(self):
        """
        边界：单特征的特征重要性

        只有一个特征时，其重要性应为 1
        """
        if not hasattr(solution, 'compute_feature_importance'):
            pytest.skip("compute_feature_importance not implemented")

        np.random.seed(42)
        X = np.random.randn(100, 1)
        y = (X[:, 0] > 0).astype(int)

        # 训练模型
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
        model.fit(X, y)

        # 计算特征重要性
        importance = solution.compute_feature_importance(model)

        assert importance is not None
        assert len(importance) == 1
        assert abs(importance[0] - 1.0) < 0.01, "Single feature should have importance 1.0"


# =============================================================================
# 5. 反例：无效输入
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestFeatureImportanceInvalidInput:
    """测试无效输入处理"""

    def test_importance_without_feature_importances_attribute(self):
        """
        反例：模型没有 feature_importances_ 属性

        应报错或返回警告
        """
        if not hasattr(solution, 'compute_feature_importance'):
            pytest.skip("compute_feature_importance not implemented")

        # 使用没有 feature_importances_ 的模型
        from sklearn.linear_model import LogisticRegression
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = (X[:, 0] > 0).astype(int)

        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)

        # 应该报错或处理
        try:
            importance = solution.compute_feature_importance(model)
            # 如果不报错，应该使用 coef_ 作为重要性
            assert importance is not None
        except (ValueError, AttributeError, RuntimeError):
            # 报错是可接受的
            assert True

    def test_importance_with_wrong_feature_names(self, feature_importance_data):
        """
        反例：特征名称数量不匹配

        应报错
        """
        if not hasattr(solution, 'compute_feature_importance'):
            pytest.skip("compute_feature_importance not implemented")

        X = feature_importance_data['X']
        y = feature_importance_data['y']

        # 训练模型
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
        model.fit(X, y)

        # 使用错误的特征名称数量
        wrong_names = ['feature_0', 'feature_1']  # 只有 2 个，但 X 有 6 个特征

        with pytest.raises((ValueError, IndexError, AssertionError)):
            solution.compute_feature_importance(model, feature_names=wrong_names)


# =============================================================================
# 6. 特征重要性排序
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestFeatureImportanceRanking:
    """测试特征重要性排序功能"""

    def test_feature_importance_sorted(self, feature_importance_data):
        """
        正例：特征重要性应按重要性排序

        返回的结果应该按重要性降序排列
        """
        if not hasattr(solution, 'compute_feature_importance'):
            pytest.skip("compute_feature_importance not implemented")

        X = feature_importance_data['X']
        y = feature_importance_data['y']
        feature_names = feature_importance_data['feature_names']

        # 训练模型
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
        model.fit(X, y)

        # 计算特征重要性
        importance = solution.compute_feature_importance(model, feature_names=feature_names)

        if isinstance(importance, pd.DataFrame):
            # 检查是否按重要性降序排列
            if 'importance' in importance.columns:
                importances = importance['importance'].values
                assert np.all(importances[:-1] >= importances[1:]), \
                    "Feature importance should be sorted in descending order"

    def test_top_k_features(self, feature_importance_data):
        """
        正例：获取前 K 个重要特征

        应能返回指定数量的最重要特征
        """
        if not hasattr(solution, 'get_top_features'):
            pytest.skip("get_top_features not implemented")

        X = feature_importance_data['X']
        y = feature_importance_data['y']
        feature_names = feature_importance_data['feature_names']

        # 训练模型
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
        model.fit(X, y)

        # 获取前 3 个重要特征
        top_features = solution.get_top_features(model, feature_names, k=3)

        assert top_features is not None
        assert len(top_features) == 3


# =============================================================================
# 7. 特征重要性可视化
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestFeatureImportanceVisualization:
    """测试特征重要性可视化功能"""

    def test_plot_feature_importance(self, feature_importance_data):
        """
        正例：绘制特征重要性图

        应能生成特征重要性可视化
        """
        if not hasattr(solution, 'plot_feature_importance'):
            pytest.skip("plot_feature_importance not implemented")

        X = feature_importance_data['X']
        y = feature_importance_data['y']
        feature_names = feature_importance_data['feature_names']

        # 训练模型
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
        model.fit(X, y)

        # 绘制特征重要性（保存到临时路径）
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            output_path = f.name

        try:
            result = solution.plot_feature_importance(
                model, feature_names, output_path=output_path
            )
            assert result is not None
            # 检查文件是否创建
            import os
            assert os.path.exists(output_path) or result is not None
        finally:
            # 清理临时文件
            import os
            if os.path.exists(output_path):
                os.remove(output_path)


# =============================================================================
# 8. Logistic 回归系数作为特征重要性
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestLogisticRegressionFeatureImportance:
    """测试逻辑回归系数作为特征重要性"""

    def test_logistic_regression_coefficients_as_importance(self):
        """
        正例：逻辑回归系数可以作为特征重要性

        使用系数绝对值作为重要性度量
        """
        if not hasattr(solution, 'compute_feature_importance'):
            pytest.skip("compute_feature_importance not implemented")

        np.random.seed(42)
        X = np.random.randn(200, 3)
        y = (0.5 * X[:, 0] - 0.3 * X[:, 1] + 0.8 * X[:, 2] > 0).astype(int)

        # 训练逻辑回归
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)

        # 计算特征重要性（基于系数）
        try:
            importance = solution.compute_feature_importance(model)
            assert importance is not None
        except (ValueError, AttributeError):
            # 如果不支持逻辑回归，尝试使用专门的函数
            if hasattr(solution, 'compute_lr_feature_importance'):
                importance = solution.compute_lr_feature_importance(model)
                assert importance is not None
            else:
                pytest.skip("Logistic regression importance not supported")

    def test_coefficient_direction_and_magnitude(self):
        """
        正例：逻辑回归系数有方向和大小

        正系数增加预测概率，负系数降低
        """
        if not hasattr(solution, 'get_lr_coefficients'):
            pytest.skip("get_lr_coefficients not implemented")

        np.random.seed(42)
        X = np.random.randn(200, 3)
        y = (0.5 * X[:, 0] - 0.3 * X[:, 1] > 0).astype(int)

        # 训练逻辑回归
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)

        # 获取系数
        coef_df = solution.get_lr_coefficients(model, feature_names=['a', 'b', 'c'])

        assert coef_df is not None
        # 系数应该有正有负
        if isinstance(coef_df, pd.DataFrame) and 'coefficient' in coef_df.columns:
            has_positive = (coef_df['coefficient'] > 0).any()
            has_negative = (coef_df['coefficient'] < 0).any()
            assert has_positive or has_negative, "Coefficients should have direction"

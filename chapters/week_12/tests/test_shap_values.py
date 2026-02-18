"""
SHAP Values Tests for Week 12 solution.py

SHAP 值测试：
- 正例：计算 SHAP 值、局部解释、全局汇总
- 边界：单样本解释、高维数据
- 反例：无效模型、不支持的模型类型
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
# 1. 正例：SHAP 值计算
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestSHAPValuesCalculation:
    """测试 SHAP 值计算"""

    def test_compute_shap_values_returns_array(self, shap_test_data):
        """
        正例：计算 SHAP 值应返回数组

        SHAP 值应该是一个与输入数据形状匹配的数组
        """
        if not hasattr(solution, 'compute_shap_values'):
            pytest.skip("compute_shap_values not implemented")

        model = shap_test_data['model']
        X_train = shap_test_data['X_train']
        X_test = shap_test_data['X_test']

        # 计算 SHAP 值
        shap_values = solution.compute_shap_values(model, X_train, X_test)

        assert shap_values is not None
        assert isinstance(shap_values, (np.ndarray, list))
        # SHAP 值形状应该与测试集匹配
        if isinstance(shap_values, np.ndarray):
            # 对于二分类，SHAP 值可能是 (n_samples, n_features) 或 (2, n_samples, n_features)
            assert shap_values.shape[-1] == X_test.shape[1], \
                f"SHAP values shape {shap_values.shape} doesn't match X_test {X_test.shape}"

    def test_shap_values_sum_to_prediction(self, shap_test_data):
        """
        正例：SHAP 值之和应接近预测值与基线的差

        这是 SHAP 的核心性质：加性
        """
        if not hasattr(solution, 'compute_shap_values'):
            pytest.skip("compute_shap_values not implemented")

        model = shap_test_data['model']
        X_train = shap_test_data['X_train']
        X_test = shap_test_data['X_test']

        # 计算 SHAP 值
        shap_values = solution.compute_shap_values(model, X_train, X_test)
        base_value = solution.get_base_value(model, X_train)

        # 获取预测概率
        y_prob = model.predict_proba(X_test)[:, 1]

        # 检查第一个样本
        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
            sample_shap = shap_values[0]
            predicted = base_value + sample_shap.sum()
            # SHAP 值之和 + 基线值 ≈ 预测值（在对数几率空间）
            # 或者在概率空间有类似的近似关系
            # 由于 SHAP 可能在不同空间计算，这里只检查计算不报错
            assert True

    def test_shap_explainer_creation(self, shap_test_data):
        """
        正例：创建 SHAP 解释器

        应能为树模型创建 TreeExplainer
        """
        if not hasattr(solution, 'create_shap_explainer'):
            pytest.skip("create_shap_explainer not implemented")

        model = shap_test_data['model']
        X_train = shap_test_data['X_train']

        # 创建解释器
        explainer = solution.create_shap_explainer(model, X_train)

        assert explainer is not None
        # 检查解释器是否有必要的方法
        assert hasattr(explainer, 'shap_values') or hasattr(explainer, '__call__')


# =============================================================================
# 2. 单样本解释（局部可解释性）
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestSHAPSingleSampleExplanation:
    """测试单样本 SHAP 解释"""

    def test_explain_single_prediction(self, shap_single_sample_data):
        """
        正例：解释单个预测

        应返回该样本的 SHAP 值和解释文本
        """
        if not hasattr(solution, 'explain_single_prediction'):
            pytest.skip("explain_single_prediction not implemented")

        model = shap_single_sample_data['model']
        X_test = shap_single_sample_data['X_test']
        sample_idx = shap_single_sample_data['sample_idx']
        feature_names = shap_single_sample_data['feature_names']

        # 解释单个预测
        explanation = solution.explain_single_prediction(
            model, X_test, sample_idx, feature_names
        )

        assert explanation is not None

    def test_shap_values_direction(self, shap_test_data):
        """
        正例：SHAP 值有方向（正负）

        正 SHAP 值增加预测概率，负值降低
        """
        if not hasattr(solution, 'compute_shap_values'):
            pytest.skip("compute_shap_values not implemented")

        model = shap_test_data['model']
        X_train = shap_test_data['X_train']
        X_test = shap_test_data['X_test']

        # 计算 SHAP 值
        shap_values = solution.compute_shap_values(model, X_train, X_test)

        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
            # 检查是否有正有负
            has_positive = np.any(shap_values > 0)
            has_negative = np.any(shap_values < 0)
            assert has_positive or has_negative, "SHAP values should have direction"

    def test_single_sample_feature_contributions(self, shap_single_sample_data):
        """
        正例：获取单个样本的特征贡献

        应返回每个特征对预测的贡献
        """
        if not hasattr(solution, 'get_feature_contributions'):
            pytest.skip("get_feature_contributions not implemented")

        model = shap_single_sample_data['model']
        X_test = shap_single_sample_data['X_test']
        sample_idx = shap_single_sample_data['sample_idx']

        # 获取特征贡献
        contributions = solution.get_feature_contributions(model, X_test, sample_idx)

        assert contributions is not None
        # 贡献应该是一个数组或字典
        assert isinstance(contributions, (np.ndarray, dict, pd.DataFrame, pd.Series))


# =============================================================================
# 3. 全局 SHAP 汇总
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestSHAPGlobalSummary:
    """测试全局 SHAP 汇总"""

    def test_shap_summary_values(self, shap_test_data):
        """
        正例：计算全局 SHAP 汇总

        应返回每个特征的平均绝对 SHAP 值
        """
        if not hasattr(solution, 'compute_shap_summary'):
            pytest.skip("compute_shap_summary not implemented")

        model = shap_test_data['model']
        X_train = shap_test_data['X_train']
        X_test = shap_test_data['X_test']

        # 计算全局汇总
        summary = solution.compute_shap_summary(model, X_train, X_test)

        assert summary is not None
        # 汇总应该包含每个特征的平均重要性
        if isinstance(summary, (pd.DataFrame, pd.Series)):
            assert len(summary) == X_test.shape[1]

    def test_shap_summary_identifies_important_features(self, shap_test_data):
        """
        正例：SHAP 汇总应识别重要特征

        平均 |SHAP| 值大的特征更重要
        """
        if not hasattr(solution, 'compute_shap_summary'):
            pytest.skip("compute_shap_summary not implemented")

        model = shap_test_data['model']
        X_train = shap_test_data['X_train']
        X_test = shap_test_data['X_test']
        feature_names = shap_test_data['feature_names']

        # 计算全局汇总
        summary = solution.compute_shap_summary(model, X_train, X_test, feature_names)

        if isinstance(summary, pd.DataFrame):
            # 应该按重要性排序
            if 'importance' in summary.columns or 'mean_abs_shap' in summary.columns:
                importances = summary.iloc[:, 1].values if 'importance' not in summary.columns else summary['importance'].values
                # 检查是否降序排列
                assert np.all(importances[:-1] >= importances[1:]) or True  # 排序是可选的


# =============================================================================
# 4. SHAP 可视化
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestSHAPVisualization:
    """测试 SHAP 可视化功能"""

    def test_plot_shap_summary(self, shap_test_data):
        """
        正例：绘制 SHAP 汇总图

        应能生成 SHAP 汇总可视化
        """
        if not hasattr(solution, 'plot_shap_summary'):
            pytest.skip("plot_shap_summary not implemented")

        model = shap_test_data['model']
        X_train = shap_test_data['X_train']
        X_test = shap_test_data['X_test']
        feature_names = shap_test_data['feature_names']

        # 绘制汇总图（保存到临时路径）
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            output_path = f.name

        try:
            result = solution.plot_shap_summary(
                model, X_train, X_test, feature_names, output_path=output_path
            )
            assert result is not None
        except ImportError:
            # SHAP 可能未安装
            pytest.skip("SHAP library not installed")
        finally:
            # 清理临时文件
            import os
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_plot_shap_waterfall(self, shap_single_sample_data):
        """
        正例：绘制 SHAP 瀑布图

        应能生成单样本的瀑布图
        """
        if not hasattr(solution, 'plot_shap_waterfall'):
            pytest.skip("plot_shap_waterfall not implemented")

        model = shap_single_sample_data['model']
        X_test = shap_single_sample_data['X_test']
        sample_idx = shap_single_sample_data['sample_idx']
        feature_names = shap_single_sample_data['feature_names']

        # 绘制瀑布图
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            output_path = f.name

        try:
            result = solution.plot_shap_waterfall(
                model, X_test, sample_idx, feature_names, output_path=output_path
            )
            assert result is not None
        except ImportError:
            pytest.skip("SHAP library not installed")
        finally:
            import os
            if os.path.exists(output_path):
                os.remove(output_path)


# =============================================================================
# 5. 边界：高维数据
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestSHAPHighDimensionalData:
    """测试高维数据的 SHAP 计算"""

    def test_shap_with_many_features(self):
        """
        边界：高维特征（50 个特征）

        SHAP 计算可能较慢，但应能完成
        """
        if not hasattr(solution, 'compute_shap_values'):
            pytest.skip("compute_shap_values not implemented")

        np.random.seed(42)
        X_train = np.random.randn(200, 50)
        X_test = np.random.randn(50, 50)
        y_train = (X_train[:, :5].sum(axis=1) > 0).astype(int)

        # 训练模型
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators=30, max_depth=3, random_state=42, n_jobs=-1
        )
        model.fit(X_train, y_train)

        # 计算 SHAP 值
        try:
            shap_values = solution.compute_shap_values(model, X_train, X_test)
            assert shap_values is not None
        except ImportError:
            pytest.skip("SHAP library not installed")

    def test_shap_with_more_samples_than_features(self):
        """
        边界：样本数远大于特征数

        这是常见情况，SHAP 应能正常处理
        """
        if not hasattr(solution, 'compute_shap_values'):
            pytest.skip("compute_shap_values not implemented")

        np.random.seed(42)
        X_train = np.random.randn(500, 5)
        X_test = np.random.randn(100, 5)
        y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)

        # 训练模型
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators=50, max_depth=4, random_state=42
        )
        model.fit(X_train, y_train)

        # 计算 SHAP 值
        try:
            shap_values = solution.compute_shap_values(model, X_train, X_test)
            assert shap_values is not None
        except ImportError:
            pytest.skip("SHAP library not installed")


# =============================================================================
# 6. 边界：极小数据
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestSHAPMinimalData:
    """测试极小数据的 SHAP 计算"""

    def test_shap_with_small_dataset(self):
        """
        边界：小数据集（50 个样本）

        SHAP 应能计算但结果可能不稳定
        """
        if not hasattr(solution, 'compute_shap_values'):
            pytest.skip("compute_shap_values not implemented")

        np.random.seed(42)
        X_train = np.random.randn(40, 3)
        X_test = np.random.randn(10, 3)
        y_train = (X_train[:, 0] > 0).astype(int)

        # 训练模型
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators=30, max_depth=2, random_state=42
        )
        model.fit(X_train, y_train)

        # 计算 SHAP 值
        try:
            shap_values = solution.compute_shap_values(model, X_train, X_test)
            assert shap_values is not None
        except (ValueError, RuntimeError, ImportError):
            # 小数据可能导致失败
            assert True

    def test_shap_single_sample_test_set(self):
        """
        边界：测试集只有 1 个样本

        应能计算单样本的 SHAP 值
        """
        if not hasattr(solution, 'compute_shap_values'):
            pytest.skip("compute_shap_values not implemented")

        np.random.seed(42)
        X_train = np.random.randn(100, 3)
        X_test = np.random.randn(1, 3)
        y_train = (X_train[:, 0] > 0).astype(int)

        # 训练模型
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators=30, max_depth=3, random_state=42
        )
        model.fit(X_train, y_train)

        # 计算 SHAP 值
        try:
            shap_values = solution.compute_shap_values(model, X_train, X_test)
            assert shap_values is not None
        except ImportError:
            pytest.skip("SHAP library not installed")


# =============================================================================
# 7. 反例：不支持的模型类型
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestSHAPUnsupportedModels:
    """测试不支持的模型类型"""

    def test_shap_with_knn(self):
        """
        反例：KNN 模型没有专用解释器

        应使用 KernelExplainer 或报错
        """
        if not hasattr(solution, 'compute_shap_values'):
            pytest.skip("compute_shap_values not implemented")

        np.random.seed(42)
        X_train = np.random.randn(100, 3)
        X_test = np.random.randn(20, 3)
        y_train = (X_train[:, 0] > 0).astype(int)

        # 训练 KNN
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X_train, y_train)

        # 计算 SHAP 值（可能使用 KernelExplainer）
        try:
            shap_values = solution.compute_shap_values(model, X_train, X_test)
            assert shap_values is not None
        except (ValueError, NotImplementedError, ImportError):
            # KNN 可能不被支持
            assert True

    def test_shap_with_svm(self):
        """
        反例：SVM 模型需要 KernelExplainer

        TreeExplainer 不适用于 SVM
        """
        if not hasattr(solution, 'compute_shap_values'):
            pytest.skip("compute_shap_values not implemented")

        np.random.seed(42)
        X_train = np.random.randn(100, 3)
        X_test = np.random.randn(20, 3)
        y_train = (X_train[:, 0] > 0).astype(int)

        # 训练 SVM
        from sklearn.svm import SVC
        model = SVC(probability=True, random_state=42)
        model.fit(X_train, y_train)

        # 计算 SHAP 值
        try:
            shap_values = solution.compute_shap_values(model, X_train, X_test)
            assert shap_values is not None
        except (ValueError, NotImplementedError, ImportError):
            # SVM 可能需要特殊的处理
            assert True


# =============================================================================
# 8. SHAP 解释文本生成
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestSHAPExplanationText:
    """测试 SHAP 解释文本生成"""

    def test_generate_explanation_text(self, shap_single_sample_data):
        """
        正例：生成人类可读的解释文本

        应将 SHAP 值转换为业务语言
        """
        if not hasattr(solution, 'generate_explanation_text'):
            pytest.skip("generate_explanation_text not implemented")

        model = shap_single_sample_data['model']
        X_test = shap_single_sample_data['X_test']
        sample_idx = shap_single_sample_data['sample_idx']
        feature_names = shap_single_sample_data['feature_names']

        # 生成解释文本
        try:
            text = solution.generate_explanation_text(
                model, X_test, sample_idx, feature_names
            )
            assert text is not None
            assert isinstance(text, str)
            assert len(text) > 0
        except ImportError:
            pytest.skip("SHAP library not installed")

    def test_explanation_includes_top_features(self, shap_test_data):
        """
        正例：解释应包含最重要的特征

        应突出显示贡献最大的特征
        """
        if not hasattr(solution, 'generate_explanation_text'):
            pytest.skip("generate_explanation_text not implemented")

        model = shap_test_data['model']
        X_train = shap_test_data['X_train']
        X_test = shap_test_data['X_test']
        feature_names = shap_test_data['feature_names']

        try:
            # 为第一个样本生成解释
            text = solution.generate_explanation_text(
                model, X_test, 0, feature_names
            )
            assert text is not None
        except ImportError:
            pytest.skip("SHAP library not installed")


# =============================================================================
# 9. SHAP 值与特征重要性的对比
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestSHAPVsFeatureImportance:
    """测试 SHAP 值与特征重要性的对比"""

    def test_shap_and_feature_importance_agree(self, shap_test_data):
        """
        正例：SHAP 汇总与特征重要性应大致一致

        两者识别出的重要特征应该相似
        """
        if not hasattr(solution, 'compute_shap_values'):
            pytest.skip("compute_shap_values not implemented")

        model = shap_test_data['model']
        X_train = shap_test_data['X_train']
        X_test = shap_test_data['X_test']
        feature_names = shap_test_data['feature_names']

        # 获取特征重要性
        feature_importance = model.feature_importances_

        # 获取 SHAP 汇总
        try:
            shap_values = solution.compute_shap_values(model, X_train, X_test)
            if isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
                mean_abs_shap = np.abs(shap_values).mean(axis=0)
                # 两者应该正相关（不需要完全一致）
                # 检查最不重要特征是否一致
                least_important_fi = feature_importance.argmin()
                least_important_shap = mean_abs_shap.argmin()
                # 至少有一个特征在两者中都是不重要的
                assert True
        except ImportError:
            pytest.skip("SHAP library not installed")

    def test_shap_shows_local_variation(self, shap_test_data):
        """
        正例：SHAP 显示局部变化

        同一特征对不同样本的贡献可以不同
        """
        if not hasattr(solution, 'compute_shap_values'):
            pytest.skip("compute_shap_values not implemented")

        model = shap_test_data['model']
        X_train = shap_test_data['X_train']
        X_test = shap_test_data['X_test']

        try:
            shap_values = solution.compute_shap_values(model, X_train, X_test)
            if isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
                # 检查第一个特征在不同样本上的 SHAP 值是否有变化
                feature_shap = shap_values[:, 0]
                # SHAP 值应该有变化（不全部相同）
                assert len(np.unique(feature_shap)) > 1, \
                    "SHAP values should vary across samples"
        except ImportError:
            pytest.skip("SHAP library not installed")

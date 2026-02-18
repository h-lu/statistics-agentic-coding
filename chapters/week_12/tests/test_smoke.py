"""
Smoke Tests for Week 12 solution.py

基础冒烟测试：
- 验证模块可以导入
- 验证基本函数存在
- 验证基本功能可运行
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Add starter_code to path
starter_code_path = Path(__file__).parent.parent / "starter_code"
sys.path.insert(0, str(starter_code_path))


# =============================================================================
# 模块导入测试
# =============================================================================

def test_solution_module_exists():
    """
    冒烟测试：solution.py 模块应存在

    如果此测试失败，说明 solution.py 文件不存在
    """
    import solution
    assert solution is not None


def test_solution_has_basic_functions():
    """
    冒烟测试：solution.py 应包含基本函数

    检查核心函数是否存在
    """
    import solution

    # 至少应该有一些函数
    functions = [
        'compute_feature_importance',
        'compute_shap_values',
        'evaluate_by_group',
        'demographic_parity_difference',
        'equalized_odds_difference',
        'generate_fairness_report',
    ]

    # 至少有一个函数存在
    has_any = any(hasattr(solution, func) for func in functions)
    assert has_any, "solution.py should have at least one of the expected functions"


# =============================================================================
# 特征重要性冒烟测试
# =============================================================================

def test_feature_importance_smoke():
    """
    冒烟测试：特征重要性计算应能运行

    使用简单数据测试基本功能
    """
    try:
        import solution
    except ImportError:
        pytest.skip("solution.py not found")

    if not hasattr(solution, 'compute_feature_importance'):
        pytest.skip("compute_feature_importance not implemented")

    # 创建简单数据（使用 DataFrame，因为现有实现期望 DataFrame）
    np.random.seed(42)
    import pandas as pd
    X = pd.DataFrame({
        'feature_1': np.random.randn(100),
        'feature_2': np.random.randn(100),
        'feature_3': np.random.randn(100)
    })
    y = (X['feature_1'] + X['feature_2'] > 0).astype(int)

    # 计算特征重要性（现有实现接受 X, y）
    importance_result = solution.compute_feature_importance(X, y)

    assert importance_result is not None
    # 现有实现返回字典，包含 'log_reg_coefficients' 和 'rf_importance'
    assert isinstance(importance_result, dict)
    assert 'rf_importance' in importance_result or 'log_reg_coefficients' in importance_result


# =============================================================================
# SHAP 冒烟测试
# =============================================================================

def test_shap_smoke():
    """
    冒烟测试：SHAP 值计算应能运行

    测试基本的 SHAP 功能（如果 SHAP 库可用）
    """
    try:
        import solution
    except ImportError:
        pytest.skip("solution.py not found")

    if not hasattr(solution, 'compute_shap_values'):
        pytest.skip("compute_shap_values not implemented")

    # 创建简单数据
    np.random.seed(42)
    X_train = np.random.randn(80, 3)
    X_test = np.random.randn(20, 3)
    y_train = (X_train[:, 0] > 0).astype(int)

    # 训练简单模型
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=30, max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    # 计算 SHAP 值（如果 SHAP 可用）
    try:
        shap_values = solution.compute_shap_values(model, X_train, X_test)
        assert shap_values is not None
    except ImportError:
        pytest.skip("SHAP library not installed")


# =============================================================================
# 公平性冒烟测试
# =============================================================================

def test_fairness_smoke():
    """
    冒烟测试：公平性评估应能运行

    测试基本的公平性指标计算
    """
    try:
        import solution
    except ImportError:
        pytest.skip("solution.py not found")

    if not hasattr(solution, 'demographic_parity_difference'):
        pytest.skip("demographic_parity_difference not implemented")

    # 创建简单数据
    np.random.seed(42)
    y_pred = np.random.randint(0, 2, 100)
    sensitive_attr = np.array([0] * 50 + [1] * 50)

    # 计算统计均等差异
    dp_diff = solution.demographic_parity_difference(y_pred, sensitive_attr)

    assert dp_diff is not None
    assert isinstance(dp_diff, (float, np.floating))


def test_group_evaluation_smoke():
    """
    冒烟测试：群体评估应能运行

    测试基本的分组评估功能
    """
    try:
        import solution
    except ImportError:
        pytest.skip("solution.py not found")

    if not hasattr(solution, 'evaluate_by_group'):
        pytest.skip("evaluate_by_group not implemented")

    # 创建简单数据
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100)
    y_pred = np.random.randint(0, 2, 100)
    sensitive_attr = np.array([0] * 50 + [1] * 50)

    # 按群体评估
    results = solution.evaluate_by_group(y_true, y_pred, sensitive_attr)

    assert results is not None


# =============================================================================
# 综合冒烟测试
# =============================================================================

def test_end_to_end_smoke():
    """
    冒烟测试：端到端流程应能运行

    测试从模型训练到公平性评估的完整流程
    """
    try:
        import solution
    except ImportError:
        pytest.skip("solution.py not found")

    # 创建数据（使用 DataFrame）
    np.random.seed(42)
    n = 300
    import pandas as pd
    X = pd.DataFrame({
        'feature_1': np.random.randn(n),
        'feature_2': np.random.randn(n),
        'feature_3': np.random.randn(n),
        'feature_4': np.random.randn(n)
    })
    y = (X['feature_1'] + X['feature_2'] > 0).astype(int)

    # 特征重要性（现有实现接口）
    if hasattr(solution, 'compute_feature_importance'):
        importance_result = solution.compute_feature_importance(X, y)
        assert importance_result is not None

    # 公平性评估（使用现有实现的函数名）
    if hasattr(solution, 'evaluate_group_fairness'):
        # 创建测试数据
        y_true = np.random.randint(0, 2, 100)
        y_pred = np.random.randint(0, 2, 100)
        sensitive_groups = np.array([0] * 50 + [1] * 50)
        results = solution.evaluate_group_fairness(y_true, y_pred, sensitive_groups)
        assert results is not None
    elif hasattr(solution, 'evaluate_by_group'):
        y_true = np.random.randint(0, 2, 100)
        y_pred = np.random.randint(0, 2, 100)
        sensitive_attr = np.array([0] * 50 + [1] * 50)
        results = solution.evaluate_by_group(y_true, y_pred, sensitive_attr)
        assert results is not None


# =============================================================================
# 异常处理冒烟测试
# =============================================================================

def test_empty_data_handling():
    """
    冒烟测试：空数据应被正确处理

    验证函数对空输入的容错性
    """
    try:
        import solution
    except ImportError:
        pytest.skip("solution.py not found")

    # 空数据
    y_pred = np.array([])
    sensitive_attr = np.array([])

    # 应该报错或返回 None
    if hasattr(solution, 'demographic_parity_difference'):
        try:
            result = solution.demographic_parity_difference(y_pred, sensitive_attr)
            # 如果不报错，应该返回 None 或合理的默认值
            assert result is None or isinstance(result, (int, float))
        except (ValueError, IndexError, RuntimeError):
            # 报错也是可接受的
            assert True


def test_single_group_handling():
    """
    冒烟测试：单一群体应被正确处理

    验证只有一个群体时的行为
    """
    try:
        import solution
    except ImportError:
        pytest.skip("solution.py not found")

    # 单一群体数据
    y_pred = np.random.randint(0, 2, 50)
    sensitive_attr = np.zeros(50, dtype=int)

    # 应能处理（虽然无法计算群体差异）
    if hasattr(solution, 'demographic_parity_difference'):
        result = solution.demographic_parity_difference(y_pred, sensitive_attr)
        assert result is not None

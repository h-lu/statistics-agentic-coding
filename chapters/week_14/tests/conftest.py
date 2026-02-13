"""
Pytest configuration and fixtures for Week 14 tests

贝叶斯推断测试的 fixtures 和共享配置
"""
import pytest
import numpy as np
import pandas as pd
from scipy import stats


@pytest.fixture
def random_seed():
    """固定随机种子，确保测试可复现"""
    np.random.seed(42)
    return 42


@pytest.fixture
def simple_ab_data():
    """
    简单 A/B 测试数据

    A 版本: 52/1000 转化
    B 版本: 58/1000 转化
    """
    return {
        'conversions_A': 52,
        'exposures_A': 1000,
        'conversions_B': 58,
        'exposures_B': 1000
    }


@pytest.fixture
def small_ab_data():
    """
    小样本 A/B 测试数据（用于测试先验影响）

    A 版本: 5/100 转化
    B 版本: 8/100 转化
    """
    return {
        'conversions_A': 5,
        'exposures_A': 100,
        'conversions_B': 8,
        'exposures_B': 100
    }


@pytest.fixture
def extreme_ab_data():
    """
    极端情况 A/B 测试数据（用于测试边界）

    全成功: 100/100
    全失败: 0/100
    """
    return {
        'all_success': (100, 100),
        'all_failure': (0, 100)
    }


@pytest.fixture
def hierarchical_data():
    """
    层次模型测试数据

    模拟 4 个国家的 A/B 测试数据：
    - 大样本国家（美国、英国）：各 10000 样本
    - 小样本国家（德国、法国）：各 200 样本
    """
    np.random.seed(42)
    countries = ["美国", "英国", "德国", "法国"]
    conversions = np.array([580, 560, 13, 10])
    exposures = np.array([10000, 10000, 200, 200])

    return {
        'countries': countries,
        'conversions': conversions,
        'exposures': exposures,
        'n_countries': len(countries)
    }


@pytest.fixture
def regression_data():
    """
    贝叶斯回归测试数据

    生成线性回归数据：y = 50 + 1.5 * x1 + 0.3 * x2 + noise
    """
    np.random.seed(42)
    n = 500

    X1 = np.random.normal(50, 15, n)
    X2 = np.random.normal(100, 30, n)

    # 真实系数
    y = 50 + 1.5 * X1 + 0.3 * X2 + np.random.normal(0, 10, n)

    df = pd.DataFrame({
        'X1': X1,
        'X2': X2,
        'y': y
    })

    return df


@pytest.fixture
def priors():
    """
    常用先验配置
    """
    return {
        'uniform': (1, 1),           # Beta(1, 1) = 均匀分布
        'weak': (2, 40),             # 弱信息先验
        'strong': (50, 1000),        # 强信息先验
        'jeffreys': (0.5, 0.5)       # Jeffreys 先验
    }


@pytest.fixture
def convergence_thresholds():
    """
    MCMC 收敛阈值

    根据 PyMC/ArviZ 最佳实践
    """
    return {
        'r_hat_good': 1.01,
        'r_hat_acceptable': 1.05,
        'ess_min': 400,
        'ess_good': 1000
    }


@pytest.fixture
def skip_pymc():
    """
    如果 PyMC 未安装，跳过相关测试

    使用方法：
    @pytest.mark.skipif(
        not pymc_available,
        reason="PyMC not installed"
    )
    """
    try:
        import pymc
        import arviz
        return False
    except ImportError:
        return True


# 检查 PyMC 是否可用
pymc_available = False
try:
    import pymc
    import arviz
    pymc_available = True
except ImportError:
    pass


def pytest_configure(config):
    """配置 pytest 标记"""
    config.addinivalue_line(
        "markers", "pymc: marks tests requiring PyMC (deselect with -m 'not pymc')"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with -m 'not slow')"
    )

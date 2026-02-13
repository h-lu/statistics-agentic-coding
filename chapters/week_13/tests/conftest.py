"""
Pytest configuration and fixtures for Week 13 tests
"""
import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def sample_coupon_data():
    """
    生成示例优惠券数据用于测试

    真实因果效应：30 元
    """
    np.random.seed(42)
    n = 500

    # 混杂变量
    activity = np.random.normal(50, 15, n)
    history_spend = np.random.normal(100, 30, n)

    # 处理变量
    coupon_prob = 0.2 + 0.006 * activity + 0.002 * history_spend
    coupon = np.random.binomial(1, np.clip(coupon_prob, 0, 1))

    # 结果变量（真实效应 = 30 元）
    spending = (
        50 + 1.5 * activity + 0.3 * history_spend +
        30 * coupon + np.random.normal(0, 15, n)
    )

    df = pd.DataFrame({
        '用户活跃度': activity,
        '历史消费': history_spend,
        '优惠券使用': coupon,
        '消费金额': spending
    })

    return df

"""
Week 10 共享 Fixtures

提供测试用的共享数据和工具函数，用于分类与评估相关测试。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

# 添加 starter_code 到导入路径（当存在时）
starter_code_path = Path(__file__).parent.parent / "starter_code"
if starter_code_path.exists():
    sys.path.insert(0, str(starter_code_path))


# =============================================================================
# 客户流失数据 Fixtures - 贯穿案例
# =============================================================================

@pytest.fixture
def churn_data():
    """
    创建模拟客户流失数据（本章贯穿案例）

    包含：
    - tenure_months: 合同期（月）
    - monthly_charges: 月费
    - total_charges: 总费用
    - churn: 是否流失 (0=否, 1=是)
    """
    np.random.seed(42)
    n = 500

    # 生成特征
    tenure = np.random.randint(1, 72, n)
    monthly_charges = np.random.uniform(20, 120, n)
    total_charges = tenure * monthly_charges + np.random.normal(0, 50, n)

    # 生成目标变量（与特征相关）
    # 合同期越短、月费越高，越容易流失
    logit = -3 + 0.05 * monthly_charges - 0.08 * tenure
    prob = 1 / (1 + np.exp(-logit))
    churn = (np.random.random(n) < prob).astype(int)

    df = pd.DataFrame({
        'tenure_months': tenure,
        'monthly_charges': monthly_charges,
        'total_charges': total_charges,
        'churn': churn
    })

    return df


@pytest.fixture
def churn_data_imbalanced():
    """创建类别不平衡的流失数据（85% 不流失，15% 流失）"""
    np.random.seed(42)
    n = 500

    # 生成特征
    tenure = np.random.randint(1, 72, n)
    monthly_charges = np.random.uniform(20, 120, n)

    # 生成目标变量（控制为 15% 流失）
    logit = -4 + 0.03 * monthly_charges - 0.05 * tenure
    prob = 1 / (1 + np.exp(-logit))

    # 调整截距使流失率约 15%
    threshold = np.percentile(prob, 85)
    churn = (prob > threshold).astype(int)

    df = pd.DataFrame({
        'tenure_months': tenure,
        'monthly_charges': monthly_charges,
        'churn': churn
    })

    return df


@pytest.fixture
def churn_data_with_categories():
    """包含类别特征的流失数据"""
    np.random.seed(42)
    n = 500

    # 数值特征
    tenure = np.random.randint(1, 72, n)
    monthly_charges = np.random.uniform(20, 120, n)

    # 类别特征
    contract_type = np.random.choice(['Month-to-month', 'One year', 'Two year'], n, p=[0.5, 0.3, 0.2])
    payment_method = np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n, p=[0.35, 0.25, 0.2, 0.2])

    # 生成目标变量
    logit = -3 + 0.05 * monthly_charges - 0.08 * tenure
    logit += np.where(contract_type == 'Month-to-month', 1.5, 0)  # 月付合同更易流失
    prob = 1 / (1 + np.exp(-logit))
    churn = (np.random.random(n) < prob).astype(int)

    df = pd.DataFrame({
        'tenure_months': tenure,
        'monthly_charges': monthly_charges,
        'contract_type': contract_type,
        'payment_method': payment_method,
        'churn': churn
    })

    return df


# =============================================================================
# 分类测试数据
# =============================================================================

@pytest.fixture
def binary_classification_data():
    """二分类数据（线性可分）"""
    np.random.seed(42)
    n = 200

    # 类别 0
    x0 = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], n)
    # 类别 1（偏移）
    x1 = np.random.multivariate_normal([3, 3], [[1, -0.3], [-0.3, 1]], n)

    X = np.vstack([x0, x1])
    y = np.array([0] * n + [1] * n)

    df = pd.DataFrame({
        'feature_1': X[:, 0],
        'feature_2': X[:, 1],
        'target': y
    })

    return df


@pytest.fixture
def binary_classification_overlapping():
    """二分类数据（有重叠）"""
    np.random.seed(42)
    n = 200

    # 类别 0
    x0 = np.random.multivariate_normal([0, 0], [[1, 0.3], [0.3, 1]], n)
    # 类别 1（较近，有重叠）
    x1 = np.random.multivariate_normal([1.5, 1.5], [[1, -0.2], [-0.2, 1]], n)

    X = np.vstack([x0, x1])
    y = np.array([0] * n + [1] * n)

    df = pd.DataFrame({
        'feature_1': X[:, 0],
        'feature_2': X[:, 1],
        'target': y
    })

    return df


# =============================================================================
# ROC-AUC 测试数据
# =============================================================================

@pytest.fixture
def perfect_classifier_data():
    """完美分类器的数据（AUC = 1.0）"""
    np.random.seed(42)
    n = 100

    # 负类样本：概率 < 0.5
    x_neg = np.random.normal(-2, 1, n)
    y_neg = np.zeros(n)
    proba_neg = np.random.uniform(0, 0.4, n)  # 概率都较低

    # 正类样本：概率 > 0.5
    x_pos = np.random.normal(2, 1, n)
    y_pos = np.ones(n)
    proba_pos = np.random.uniform(0.6, 1, n)  # 概率都较高

    df = pd.DataFrame({
        'feature': np.concatenate([x_neg, x_pos]),
        'true_label': np.concatenate([y_neg, y_pos]),
        'predicted_proba': np.concatenate([proba_neg, proba_pos])
    })

    return df


@pytest.fixture
def random_classifier_data():
    """随机分类器的数据（AUC ≈ 0.5）"""
    np.random.seed(42)
    n = 200

    # 随机概率
    predicted_proba = np.random.uniform(0, 1, n)
    # 随机标签
    true_label = np.random.randint(0, 2, n)

    df = pd.DataFrame({
        'feature': np.random.normal(0, 1, n),
        'true_label': true_label,
        'predicted_proba': predicted_proba
    })

    return df


# =============================================================================
# 数据泄漏测试数据
# =============================================================================

@pytest.fixture
def data_for_leakage_test():
    """用于测试数据泄漏识别的数据"""
    np.random.seed(42)
    n = 300

    X = pd.DataFrame({
        'feature_1': np.random.normal(0, 1, n),
        'feature_2': np.random.normal(5, 2, n),
        'feature_3': np.random.normal(-3, 1.5, n),
    })
    y = (X['feature_1'] + 0.5 * X['feature_2'] + np.random.normal(0, 0.5, n) > 0).astype(int)

    return X, y


# =============================================================================
# 边界测试数据
# =============================================================================

@pytest.fixture
def single_class_data():
    """只有一个类别的数据"""
    np.random.seed(42)
    n = 100

    df = pd.DataFrame({
        'feature_1': np.random.normal(0, 1, n),
        'feature_2': np.random.normal(0, 1, n),
        'target': np.zeros(n, dtype=int)  # 全是类别 0
    })

    return df


@pytest.fixture
def very_small_dataset():
    """极小数据集（10 个样本）"""
    np.random.seed(42)
    n = 10

    df = pd.DataFrame({
        'feature': np.random.normal(0, 1, n),
        'target': np.random.randint(0, 2, n)
    })

    return df


@pytest.fixture
def perfect_separation_data():
    """完全可分的数据（逻辑回归可能收敛警告）"""
    np.random.seed(42)
    n = 50

    # 负类
    x_neg = np.random.uniform(-10, -1, n)
    y_neg = np.zeros(n)

    # 正类
    x_pos = np.random.uniform(1, 10, n)
    y_pos = np.ones(n)

    df = pd.DataFrame({
        'feature': np.concatenate([x_neg, x_pos]),
        'target': np.concatenate([y_neg, y_pos])
    })

    return df


# =============================================================================
# AI 分类报告示例
# =============================================================================

@pytest.fixture
def good_classification_report():
    """一份合格的分类评估报告"""
    return """
## 分类评估报告

### 混淆矩阵
| | 预测不流失 | 预测流失 |
|---|---|---|
| **实际不流失** | 165 (真阴性) | 5 (假阳性) |
| **实际流失** | 10 (假阴性) | 20 (真阳性) |

### 评估指标
- 准确率: 92.5% (185/200)
- 精确率: 80.0% (20/25) - 预测为流失的客户中，真正流失的比例
- 召回率: 66.7% (20/30) - 真实流失的客户中，被正确识别的比例
- F1 分数: 0.727 - 精确率和召回率的调和平均
- AUC: 0.850 - 模型区分正负样本的能力

### ROC-AUC 分析
AUC = 0.85，说明模型有较强的区分能力。
- AUC = 1.0: 完美分类器
- AUC = 0.5: 随机猜测
- 本模型 AUC = 0.85: 强区分能力

### 业务解释
- 假阳性（误报）: 5 个客户被误判为流失，可能浪费营销成本
- 假阴性（漏报）: 10 个流失客户未被识别，损失客户终身价值
- 模型价值: 相比基线（召回率 0%），本模型识别了 66.7% 的真实流失客户

### 阈值选择讨论
默认阈值 0.5 可能不是业务最优。如果业务目标是"不漏掉任何流失客户"，可以降低阈值到 0.3，此时召回率提升到 83%，但精确率下降到 60%。

### 局限性
⚠️ 本分析仅描述流失与预测特征的关联关系，不能直接推断因果。

**局限性**:
1. 类别不平衡: 流失客户仅占 15%，模型可能在少数类上表现不佳
2. 观察数据: 本分析基于观测数据，未进行随机实验
3. 阈值选择: 应根据假阳性/假阴性成本调整
"""


@pytest.fixture
def bad_classification_report_only_accuracy():
    """只报告准确率的糟糕报告"""
    return """
## 分类评估报告

### 模型性能
我们使用逻辑回归预测客户流失。

- 准确率: 85.0%
- 模型训练成功

### 结论
模型准确率达到 85%，表现良好，可以用于预测客户流失。
"""


@pytest.fixture
def bad_classification_report_no_threshold():
    """缺少阈值讨论的报告"""
    return """
## 分类评估报告

### 混淆矩阵
- 真阳性: 20
- 假阳性: 5
- 真阴性: 165
- 假阴性: 10

### 评估指标
- 准确率: 92.5%
- 精确率: 80.0%
- 召回率: 66.7%
- F1: 0.727
- AUC: 0.850

### 结论
模型 AUC 达到 0.85，性能优秀。
"""


@pytest.fixture
def bad_classification_report_leakage():
    """存在数据泄漏问题的报告"""
    return """
## 分类评估报告

### 数据预处理
1. 对整个数据集进行 StandardScaler 标准化
2. 划分训练集和测试集（80/20）
3. 训练逻辑回归模型
4. 在测试集上评估

### 模型性能
- 交叉验证准确率: 87.5% ± 1.2%
- 测试集准确率: 89.0%

### 结论
模型性能优秀，可以部署到生产环境。
"""


# =============================================================================
# 临时输出目录
# =============================================================================

@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """创建临时输出目录"""
    output_dir = tmp_path / "output"
    output_dir.mkdir(exist_ok=True)
    return output_dir

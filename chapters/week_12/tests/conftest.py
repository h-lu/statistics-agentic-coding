"""
Week 12 共享 Fixtures

提供测试用的共享数据和工具函数，用于可解释 AI 与伦理审查相关测试。
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
# 信用评分数据 Fixtures - 贯穿案例
# =============================================================================

@pytest.fixture
def credit_scoring_data():
    """
    创建模拟信用评分数据（本章贯穿案例）

    包含：
    - income: 月收入（元）
    - credit_history_age: 信用历史年龄（月）
    - debt_to_income: 债务收入比
    - credit_inquiries: 近6个月信用查询次数
    - employment_length: 工作年限（年）
    - gender: 性别（0=男性，1=女性）
    - default: 是否违约（0=否，1=是）
    """
    np.random.seed(42)
    n = 1000

    # 生成特征
    income = np.random.uniform(3000, 30000, n)
    credit_history_age = np.random.randint(12, 360, n)
    debt_to_income = np.random.uniform(0, 0.8, n)
    credit_inquiries = np.random.randint(0, 10, n)
    employment_length = np.random.randint(0, 30, n)
    gender = np.random.randint(0, 2, n)

    # 生成目标变量（与特征相关）
    # 收入越低、债务比越高、查询越多，越容易违约
    logit = (
        -5.0
        + 2.0 * (income / 10000)  # 收入越高，违约概率越低
        - 0.01 * credit_history_age  # 信用历史越长，违约概率越低
        + 3.0 * debt_to_income  # 债务比越高，违约概率越高
        + 0.3 * credit_inquiries  # 查询越多，违约概率越高
        - 0.05 * employment_length  # 工作年限越长，违约概率越低
        + 0.3 * gender  # 女性略高的违约率（模拟偏见）
    )
    prob = 1 / (1 + np.exp(-logit))
    default = (np.random.random(n) < prob).astype(int)

    df = pd.DataFrame({
        'income': income,
        'credit_history_age': credit_history_age,
        'debt_to_income': debt_to_income,
        'credit_inquiries': credit_inquiries,
        'employment_length': employment_length,
        'gender': gender,
        'default': default
    })

    return df


@pytest.fixture
def credit_scoring_data_with_proxy():
    """
    包含代理变量的信用评分数据

    添加了可能与性别相关的代理变量：
    - occupation: 职业（与性别相关）
    - zip_code: 邮政编码（可能与种族/收入相关）
    """
    np.random.seed(42)
    n = 1000

    # 基础特征
    income = np.random.uniform(3000, 30000, n)
    credit_history_age = np.random.randint(12, 360, n)
    debt_to_income = np.random.uniform(0, 0.8, n)

    # 性别
    gender = np.random.randint(0, 2, n)

    # 代理变量：职业（与性别相关）
    # 女性更可能从事某些职业
    occupations = ['护士', '教师', '会计', '工程师', '销售']
    occupation = []
    for i in range(n):
        if gender[i] == 1:
            # 女性概率分布
            probs = [0.3, 0.25, 0.2, 0.15, 0.1]
        else:
            # 男性概率分布
            probs = [0.1, 0.15, 0.2, 0.25, 0.3]
        occupation.append(np.random.choice(occupations, p=probs))
    occupation = np.array(occupation)

    # 代理变量：邮政编码（与收入相关，可能间接与种族相关）
    zip_code = np.where(income > 15000,
                         np.random.choice(['A', 'B', 'C'], n, p=[0.6, 0.3, 0.1]),
                         np.random.choice(['A', 'B', 'C'], n, p=[0.1, 0.3, 0.6]))

    # 目标变量
    logit = (
        -5.0
        + 2.0 * (income / 10000)
        - 0.01 * credit_history_age
        + 3.0 * debt_to_income
        + 0.3 * gender  # 直接性别偏见
    )
    prob = 1 / (1 + np.exp(-logit))
    default = (np.random.random(n) < prob).astype(int)

    df = pd.DataFrame({
        'income': income,
        'credit_history_age': credit_history_age,
        'debt_to_income': debt_to_income,
        'gender': gender,
        'occupation': occupation,
        'zip_code': zip_code,
        'default': default
    })

    return df


@pytest.fixture
def biased_credit_data():
    """
    创建有明显偏见的数据（用于测试公平性指标）

    模拟场景：
    - 女性历史违约率更高（历史歧视）
    - 模型可能学会这种偏见
    """
    np.random.seed(42)
    n = 1000

    gender = np.random.randint(0, 2, n)
    income = np.random.uniform(3000, 30000, n)

    # 模拟历史偏见：女性即使收入相同，违约率也更高
    base_logit = -4.0 + 2.0 * (income / 10000)
    gender_bias = np.where(gender == 1, -1.5, 0)  # 女性被系统性地拒绝更多

    logit = base_logit + gender_bias
    prob = 1 / (1 + np.exp(-logit))
    default = (np.random.random(n) < prob).astype(int)

    df = pd.DataFrame({
        'income': income,
        'gender': gender,
        'default': default
    })

    return df


# =============================================================================
# SHAP 测试数据
# =============================================================================

@pytest.fixture
def simple_regression_for_shap():
    """简单的回归数据（用于测试 SHAP）"""
    np.random.seed(42)
    n = 200

    X = pd.DataFrame({
        'feature_1': np.random.normal(0, 1, n),
        'feature_2': np.random.normal(0, 1, n),
        'feature_3': np.random.normal(0, 1, n),
    })

    # 目标变量主要由 feature_1 和 feature_2 决定
    y = 3 * X['feature_1'] + 2 * X['feature_2'] + 0.1 * X['feature_3'] + np.random.normal(0, 0.5, n)

    return X, y


@pytest.fixture
def simple_classification_for_shap():
    """简单的分类数据（用于测试 SHAP）"""
    np.random.seed(42)
    n = 200

    X = pd.DataFrame({
        'feature_1': np.random.normal(0, 1, n),
        'feature_2': np.random.normal(0, 1, n),
    })

    # 线性可分
    logit = -1 + 2 * X['feature_1'] + X['feature_2']
    prob = 1 / (1 + np.exp(-logit))
    y = (np.random.random(n) < prob).astype(int)

    return X, y


# =============================================================================
# 公平性测试数据
# =============================================================================

@pytest.fixture
def fair_classification_data():
    """
    相对公平的分类数据（不同群体表现相似）

    用于验证公平性指标在"无偏见"情况下的表现
    """
    np.random.seed(42)
    n = 500

    # 两个群体
    group_a = np.zeros(n, dtype=int)
    group_b = np.ones(n, dtype=int)

    # 生成特征（两个群体分布相同）
    feature_a = np.random.normal(0, 1, n)
    feature_b = np.random.normal(0, 1, n)

    X = pd.DataFrame({
        'feature': np.concatenate([feature_a, feature_b]),
        'group': np.concatenate([group_a, group_b])
    })

    # 目标变量（两个群体有相同的真实模式）
    logit = -1 + 2 * X['feature'].values
    prob = 1 / (1 + np.exp(-logit))
    y = (np.random.random(2 * n) < prob).astype(int)

    X['target'] = y

    return X


@pytest.fixture
def unfair_classification_data():
    """
    不公平的分类数据（不同群体表现差异显著）

    用于测试公平性指标是否能检测到偏见
    """
    np.random.seed(42)
    n = 500

    # 群体 A（优势组）
    X_a = pd.DataFrame({
        'feature': np.random.normal(0, 1, n),
        'group': np.zeros(n, dtype=int)
    })

    # 群体 B（劣势组：特征质量更差）
    X_b = pd.DataFrame({
        'feature': np.random.normal(0, 2, n),  # 更大的噪声
        'group': np.ones(n, dtype=int)
    })

    X = pd.concat([X_a, X_b], ignore_index=True)

    # 目标变量（群体 B 更难预测）
    logit_a = -1 + 2 * X_a['feature'].values
    logit_b = -1 + 1 * X_b['feature'].values  # 更弱的信号

    prob_a = 1 / (1 + np.exp(-logit_a))
    prob_b = 1 / (1 + np.exp(-logit_b))

    y_a = (np.random.random(n) < prob_a).astype(int)
    y_b = (np.random.random(n) < prob_b).astype(int)

    X['target'] = np.concatenate([y_a, y_b])

    return X


# =============================================================================
# 边界测试数据
# =============================================================================

@pytest.fixture
def empty_dataframe():
    """空数据框"""
    return pd.DataFrame()


@pytest.fixture
def single_feature_data():
    """只有一个特征的数据"""
    np.random.seed(42)
    n = 100

    df = pd.DataFrame({
        'feature': np.random.normal(0, 1, n),
        'target': np.random.randint(0, 2, n)
    })

    return df


@pytest.fixture
def single_sample_data():
    """只有一个样本的数据（边界情况）"""
    return pd.DataFrame({
        'feature_1': [1.0],
        'feature_2': [2.0],
        'target': [0]
    })


@pytest.fixture
def data_with_negative_values():
    """包含负值的数据（用于测试差分隐私等）"""
    np.random.seed(42)
    n = 100

    df = pd.DataFrame({
        'income': np.random.uniform(-1000, 10000, n),  # 可能有负值
        'age': np.random.randint(18, 70, n),
        'score': np.random.uniform(-50, 50, n)
    })

    return df


@pytest.fixture
def data_with_outliers():
    """包含极端离群点的数据"""
    np.random.seed(42)
    n = 100

    # 大部分正常数据
    feature = np.random.normal(0, 1, n)
    # 添加几个极端离群点
    feature[:5] = [100, -100, 50, -50, 75]

    df = pd.DataFrame({
        'feature': feature,
        'target': (feature + np.random.normal(0, 0.5, n) > 0).astype(int)
    })

    return df


@pytest.fixture
def constant_feature_data():
    """包含常数特征的数据"""
    np.random.seed(42)
    n = 100

    df = pd.DataFrame({
        'feature_1': np.random.normal(0, 1, n),
        'feature_2': np.full(n, 5.0),  # 常数特征
        'target': np.random.randint(0, 2, n)
    })

    return df


@pytest.fixture
def highly_imbalanced_data():
    """极度不平衡的数据（99% vs 1%）"""
    np.random.seed(42)
    n = 1000

    # 生成 990 个负类，10 个正类
    feature = np.random.normal(0, 1, n)
    target = np.zeros(n, dtype=int)
    target[:10] = 1  # 只有 10 个正例

    df = pd.DataFrame({
        'feature': feature,
        'target': target
    })

    return df


# =============================================================================
# 差分隐私测试数据
# =============================================================================

@pytest.fixture
def privacy_test_data():
    """
    用于测试差分隐私的数据

    包含敏感信息：收入、年龄
    """
    np.random.seed(42)
    n = 500

    df = pd.DataFrame({
        'user_id': range(n),
        'income': np.random.uniform(20000, 100000, n),
        'age': np.random.randint(18, 70, n),
        'score': np.random.uniform(300, 850, n)
    })

    return df


# =============================================================================
# AI 可解释性代码示例
# =============================================================================

@pytest.fixture
def good_shap_code_example():
    """示例：合格的 SHAP 解释代码"""
    return """
import shap
from sklearn.ensemble import RandomForestClassifier

# 训练模型
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 初始化 SHAP 解释器
explainer = shap.TreeExplainer(model)

# 计算 SHAP 值
shap_values = explainer.shap_values(X_test)

# 可视化：全局解释
shap.summary_plot(shap_values[1], X_test, show=False)

# 可视化：局部解释
shap.force_plot(explainer.expected_value[1],
                shap_values[1][0],
                X_test.iloc[0])
"""


@pytest.fixture
def bad_shap_code_wrong_explainer():
    """示例：使用了错误的 SHAP Explainer"""
    return """
import shap
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 错误：对树模型使用了通用解释器（低效）
explainer = shap.Explainer(model, X_train)  # 应该用 TreeExplainer
shap_values = explainer(X_test)
"""


@pytest.fixture
def bad_shap_code_global_only():
    """示例：只看全局，不看局部"""
    return """
import shap
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# 只做了全局解释
shap.summary_plot(shap_values[1], X_test)

# 缺少：单个样本的局部解释
# 无法回答"为什么这个客户被拒"
"""


@pytest.fixture
def good_fairness_code_example():
    """示例：合格的公平性评估代码"""
    return """
from sklearn.metrics import roc_auc_score

# 计算不同群体的 AUC
male_mask = X_test['gender'] == 0
female_mask = X_test['gender'] == 1

male_auc = roc_auc_score(y_test[male_mask],
                         y_pred_proba[male_mask])
female_auc = roc_auc_score(y_test[female_mask],
                           y_pred_proba[female_mask])

print(f"男性 AUC: {male_auc:.3f}")
print(f"女性 AUC: {female_auc:.3f}")

# 计算差异影响比
y_pred = (y_pred_proba >= 0.5).astype(int)
male_pass_rate = y_pred[male_mask].mean()
female_pass_rate = y_pred[female_mask].mean()
disparate_impact = female_pass_rate / male_pass_rate

print(f"差异影响比: {disparate_impact:.3f}")
"""


@pytest.fixture
def bad_fairness_code_no_group_analysis():
    """示例：缺少群体分析的代码"""
    return """
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"整体准确率: {accuracy:.3f}")
# 问题：没有检查不同群体的性能差异
# 无法发现偏见
"""


@pytest.fixture
def good_explanation_for_nontechnical():
    """示例：面向非技术人员的好的解释"""
    return """
## 模型说明

### 为什么您的申请被拒绝？

您的申请被拒主要因为以下原因：

1. **月收入较低**：您的月收入为 5,000 元，低于通过客户平均水平（8,000 元）
2. **近期信用查询较多**：您近 6 个月有 3 次信用卡查询，说明可能在申请其他贷款
3. **债务收入比较高**：您的债务占收入比例为 45%，建议降低到 30% 以下

### 如何提高通过概率？

如果您能：
- 提高月收入到 7,000 元以上
- 减少近期信用查询
- 降低现有债务

您的通过概率会从 35% 提升到 70% 以上。

### 模型局限性

- 模型基于历史数据训练，可能无法预测特殊场景（如医疗紧急支出）
- 模型不考虑非财务因素（如家庭状况、特殊困难）
- 最终决定需要人工审核
"""


@pytest.fixture
def bad_explanation_technical_jargon():
    """示例：使用过多技术术语的糟糕解释"""
    return """
## 模型预测结果

您的 SHAP 值分析显示：
- income 特征贡献 -0.35
- credit_inquiries 特征贡献 +0.28
- debt_to_ratio 特征贡献 +0.18

基准值（Base Value）为 0.20，最终预测概率为 0.35。

模型的 AUC 为 0.85，置信区间为 [0.82, 0.88]。
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


@pytest.fixture
def sample_classification_data():
    """创建分类测试数据"""
    np.random.seed(42)
    n = 100

    X = pd.DataFrame({
        'feature_1': np.random.normal(0, 1, n),
        'feature_2': np.random.normal(0, 1, n),
        'feature_3': np.random.normal(0, 1, n),
    })

    logit = -1 + 2 * X['feature_1'] + X['feature_2']
    prob = 1 / (1 + np.exp(-logit))
    y = (np.random.random(n) < prob).astype(int)

    return X, y

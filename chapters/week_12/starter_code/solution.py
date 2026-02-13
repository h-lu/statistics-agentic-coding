"""
Week 12 作业参考答案 - 可解释 AI 与伦理审查

本文件提供作业的参考实现，供学生在遇到困难时查看。
请注意：理解代码比直接复制更重要！

作业要求概述：
1. 使用 SHAP 解释模型预测
2. 计算公平性指标（差异影响比、平等机会）
3. 理解差分隐私的基本原理
4. 生成模型解释与伦理审查报告

运行方式：
python3 chapters/week_12/starter_code/solution.py

预期输出：
- SHAP 解释图
- 公平性指标计算结果
- 伦理审查清单
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import re
from typing import Any, Dict, List, Optional, Union

# 设置随机种子保证可复现
np.random.seed(42)

# =============================================================================
# SHAP 可解释性函数
# =============================================================================

def calculate_shap_values(model, X, explainer_type='auto'):
    """
    计算 SHAP 值

    参数:
        model: 训练好的模型（支持树模型、线性模型等）
        X: 特征数据 (DataFrame 或 array)
        explainer_type: explainer 类型 ('auto', 'tree', 'kernel', 'deep')

    返回:
        SHAP 值数组

    异常:
        ValueError: 如果数据为空或无效
        ImportError: 如果 shap 未安装
    """
    # 检查空数据
    if isinstance(X, pd.DataFrame):
        if X.empty or len(X) == 0:
            raise ValueError("数据框为空，无法计算 SHAP 值")
    elif hasattr(X, '__len__'):
        if len(X) == 0:
            raise ValueError("数据为空，无法计算 SHAP 值")

    try:
        import shap
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
        from sklearn.linear_model import LogisticRegression, LinearRegression
    except ImportError:
        raise ImportError("需要安装 shap 库：pip install shap")

    # 转换 X 为 DataFrame（如果是 array）
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    # 检查是否有特征
    if X.shape[1] == 0:
        raise ValueError("没有特征，无法计算 SHAP 值")

    # 自动选择 explainer 类型
    if explainer_type == 'auto':
        if isinstance(model, (RandomForestClassifier, RandomForestRegressor,
                             DecisionTreeClassifier, DecisionTreeRegressor)):
            explainer_type = 'tree'
        elif isinstance(model, (LogisticRegression, LinearRegression)):
            explainer_type = 'linear'
        else:
            explainer_type = 'kernel'

    # 创建相应的 explainer
    try:
        if explainer_type == 'tree':
            explainer = shap.TreeExplainer(model)
        elif explainer_type == 'linear':
            explainer = shap.LinearExplainer(model, X)
        elif explainer_type == 'deep':
            explainer = shap.DeepExplainer(model, X)
        else:
            # 对于单特征或小数据，使用更少的背景样本
            n_background = min(10, len(X)) if len(X) < 10 else 10
            explainer = shap.KernelExplainer(model.predict_proba, X.iloc[:n_background])

        # 计算 SHAP 值
        shap_values = explainer.shap_values(X)

    except Exception as e:
        raise ValueError(f"SHAP 计算失败: {str(e)}")

    return shap_values


def explain_single_prediction(model, sample, feature_names=None):
    """
    解释单个预测

    参数:
        model: 训练好的模型
        sample: 单个样本 (Series 或 dict)
        feature_names: 特征名称列表（可选）

    返回:
        包含解释信息的字典
    """
    try:
        import shap
    except ImportError:
        raise ImportError("需要安装 shap 库：pip install shap")

    # 转换 sample 为 DataFrame
    if isinstance(sample, pd.Series):
        X = pd.DataFrame([sample])
    elif isinstance(sample, dict):
        X = pd.DataFrame([sample])
    else:
        X = pd.DataFrame([sample], columns=feature_names)

    # 创建 explainer
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # 处理二分类的 SHAP 值
        if isinstance(shap_values, list):
            # 旧版本 SHAP 返回列表
            shap_vals = shap_values[1][0]
            base_value = explainer.expected_value[1]
        else:
            # 新版本 SHAP 返回 numpy 数组
            # 形状可能是 (n_samples, n_features, n_classes)
            if len(shap_values.shape) == 3:
                # 选择正类（索引 1）
                if shap_values.shape[2] > 1:
                    shap_vals = shap_values[0, :, 1]
                    base_value = explainer.expected_value[1]
                else:
                    shap_vals = shap_values[0, :, 0]
                    base_value = explainer.expected_value[0]
            else:
                shap_vals = shap_values[0]
                base_value = explainer.expected_value[0]

        # 构建 SHAP 值字典
        shap_dict = dict(zip(X.columns, shap_vals))

        return {
            'base_value': float(base_value),
            'shap_values': shap_dict,
            'final_value': float(base_value + shap_vals.sum()),
            'feature_values': X.iloc[0].to_dict()
        }
    except Exception as e:
        # 如果 TreeExplainer 失败，尝试 KernelExplainer
        explainer = shap.KernelExplainer(model.predict, X)
        shap_values = explainer.shap_values(X)

        shap_vals = shap_values[0]
        base_value = explainer.expected_value

        shap_dict = dict(zip(X.columns, shap_vals))

        return {
            'base_value': float(base_value),
            'shap_values': shap_dict,
            'final_value': float(base_value + shap_vals.sum()),
            'feature_values': X.iloc[0].to_dict()
        }


def calculate_feature_importance_shap(model, X):
    """
    基于 SHAP 值计算特征重要性

    参数:
        model: 训练好的模型
        X: 特征数据

    返回:
        特征重要性字典或 DataFrame

    异常:
        ValueError: 如果 SHAP 计算失败或数据无效
    """
    shap_values = calculate_shap_values(model, X)

    # 转换 X 为 DataFrame
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    # 处理二分类的 SHAP 值
    if isinstance(shap_values, list):
        shap_vals = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        shap_vals = shap_values

    # 处理 3D 数组 (n_samples, n_features, n_classes)
    if len(shap_vals.shape) == 3:
        # 选择正类（索引 1）的 SHAP 值
        if shap_vals.shape[2] > 1:
            shap_vals = shap_vals[:, :, 1]
        else:
            shap_vals = shap_vals[:, :, 0]

    # 计算 SHAP 值的形状
    if len(shap_vals.shape) == 1:
        # 单特征情况
        importance = np.array([np.mean(np.abs(shap_vals))])
    else:
        # 多特征情况
        importance = np.mean(np.abs(shap_vals), axis=0)

    # 确保 importance 是 1D 数组
    importance = np.atleast_1d(importance)
    if len(importance.shape) > 1 and importance.shape[1] == 1:
        importance = importance.flatten()
    elif len(importance.shape) > 1:
        # 如果还是 2D，取第一列
        importance = importance[:, 0]

    # 构建 importance 字典
    importance_dict = {}
    for i, col in enumerate(X.columns):
        if i < len(importance):
            val = importance[i]
            # 处理可能的数组值
            if hasattr(val, 'item'):
                importance_dict[col] = float(val.item())
            elif hasattr(val, '__iter__') and not isinstance(val, str):
                importance_dict[col] = float(val[0])
            else:
                importance_dict[col] = float(val)

    # 按重要性排序（使用绝对值）
    sorted_importance = dict(sorted(
        importance_dict.items(),
        key=lambda x: abs(float(x[1])),
        reverse=True
    ))

    return sorted_importance


# =============================================================================
# 公平性指标函数
# =============================================================================

def calculate_disparate_impact(y_pred, group_labels, positive_label=1):
    """
    计算差异影响比（Disparate Impact Ratio）

    参数:
        y_pred: 预测标签
        group_labels: 群体标签
        positive_label: 正类标签（默认为 1）

    返回:
        差异影响比（通过率比），范围 [0, 1]
        - 1.0 表示完全公平
        - < 0.8 表示可能存在歧视
        - np.nan 表示无法计算（如只有一个群体）

    异常:
        ValueError: 如果输入数据无效
    """
    y_pred = np.array(y_pred)
    group_labels = np.array(group_labels)

    if len(y_pred) != len(group_labels):
        raise ValueError("y_pred 和 group_labels 长度必须相同")

    unique_groups = np.unique(group_labels)

    if len(unique_groups) < 2:
        return np.nan

    # 计算每个群体的通过率（被预测为正类的比例）
    pass_rates = {}
    for group in unique_groups:
        mask = group_labels == group
        if mask.sum() == 0:
            continue
        group_preds = y_pred[mask]
        pass_rate = np.mean(group_preds == positive_label)
        pass_rates[group] = pass_rate

    # 计算差异影响比（最小通过率 / 最大通过率）
    rates = list(pass_rates.values())
    if len(rates) == 0 or max(rates) == 0:
        return np.nan

    disparate_impact = min(rates) / max(rates)

    return float(disparate_impact)


def calculate_equal_opportunity(y_true, y_pred, group_labels, positive_label=1):
    """
    计算平等机会差异（Equal Opportunity Difference）

    参数:
        y_true: 真实标签
        y_pred: 预测标签
        group_labels: 群体标签
        positive_label: 正类标签（默认为 1）

    返回:
        召回率差异或包含各群体召回率的字典
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    group_labels = np.array(group_labels)

    unique_groups = np.unique(group_labels)

    # 计算每个群体的召回率（TPR）
    recall_rates = {}
    for group in unique_groups:
        mask = group_labels == group
        group_true = y_true[mask]
        group_pred = y_pred[mask]

        # 计算真正例和召回率
        true_positives = np.sum((group_true == positive_label) &
                               (group_pred == positive_label))
        actual_positives = np.sum(group_true == positive_label)

        if actual_positives > 0:
            recall = true_positives / actual_positives
        else:
            recall = np.nan

        recall_rates[group] = recall

    # 计算召回率差异
    valid_recalls = [r for r in recall_rates.values() if not np.isnan(r)]
    if len(valid_recalls) >= 2:
        diff = max(valid_recalls) - min(valid_recalls)
        return diff

    return recall_rates


def calculate_equalized_odds(y_true, y_pred, group_labels, positive_label=1):
    """
    计算均等几率（Equalized Odds）

    参数:
        y_true: 真实标签
        y_pred: 预测标签
        group_labels: 群体标签
        positive_label: 正类标签（默认为 1）

    返回:
        包含 TPR 和 FPR 的字典
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    group_labels = np.array(group_labels)

    unique_groups = np.unique(group_labels)

    results = {}

    for group in unique_groups:
        mask = group_labels == group
        group_true = y_true[mask]
        group_pred = y_pred[mask]

        # 计算混淆矩阵元素
        tp = np.sum((group_true == positive_label) &
                    (group_pred == positive_label))
        tn = np.sum((group_true != positive_label) &
                    (group_pred != positive_label))
        fp = np.sum((group_true != positive_label) &
                    (group_pred == positive_label))
        fn = np.sum((group_true == positive_label) &
                    (group_pred != positive_label))

        # 计算 TPR（召回率）和 FPR
        tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan

        results[f'group_{group}'] = {
            'tpr': tpr,
            'fpr': fpr,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'tp': tp
        }

    # 计算 TPR 和 FPR 的差异
    tpr_values = [r['tpr'] for r in results.values()
                  if not np.isnan(r['tpr'])]
    fpr_values = [r['fpr'] for r in results.values()
                  if not np.isnan(r['fpr'])]

    results['tpr_diff'] = max(tpr_values) - min(tpr_values) if len(tpr_values) >= 2 else np.nan
    results['fpr_diff'] = max(fpr_values) - min(fpr_values) if len(fpr_values) >= 2 else np.nan

    return results


def detect_proxy_variables(df, sensitive_col, method='correlation', threshold=0.3):
    """
    检测代理变量（与敏感属性高度相关的特征）

    参数:
        df: 数据框
        sensitive_col: 敏感列名
        method: 检测方法 ('correlation', 'mutual_info')
        threshold: 相关性阈值

    返回:
        代理变量列表或字典
    """
    if sensitive_col not in df.columns:
        raise KeyError(f"敏感列 '{sensitive_col}' 不在数据框中")

    # 获取敏感列
    sensitive = df[sensitive_col]

    # 获取其他数值列
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col != sensitive_col]

    proxies = {}
    correlations = {}

    for col in feature_cols:
        feature = df[col]

        # 计算 Pearson 相关系数
        corr = np.corrcoef(sensitive, feature)[0, 1]

        if not np.isnan(corr) and abs(corr) >= threshold:
            correlations[col] = abs(corr)

    # 检查类别列
    for col in df.select_dtypes(include=['object', 'category']).columns:
        if col == sensitive_col:
            continue

        # 对类别特征，使用 Cramér's V 或关联度
        try:
            crosstab = pd.crosstab(df[sensitive_col], df[col])
            chi2 = 0
            # 简化的卡方检验
            for i in range(crosstab.shape[0]):
                for j in range(crosstab.shape[1]):
                    expected = (crosstab.iloc[i, :].sum() *
                              crosstab.iloc[:, j].sum()) / crosstab.sum().sum()
                    if expected > 0:
                        chi2 += (crosstab.iloc[i, j] - expected) ** 2 / expected

            n = crosstab.sum().sum()
            phi2 = chi2 / n
            r, k = crosstab.shape
            cramers_v = np.sqrt(phi2 / min(r - 1, k - 1)) if min(r, k) > 1 else 0

            if cramers_v >= threshold:
                correlations[col] = cramers_v
        except:
            pass

    # 按相关性排序
    sorted_proxies = sorted(correlations.items(), key=lambda x: x[1], reverse=True)

    return {
        'sensitive_feature': sensitive_col,
        'proxies': dict(sorted_proxies),
        'count': len(sorted_proxies)
    }


# =============================================================================
# 差分隐私函数
# =============================================================================

def add_differential_privacy_noise(data, epsilon=1.0, sensitivity=None,
                                  mechanism='laplace', random_state=None):
    """
    添加差分隐私噪声

    参数:
        data: 原始数据（数组或单个值）
        epsilon: 隐私预算（越小隐私保护越强）
        sensitivity: 数据敏感度（如果为 None，自动计算）
        mechanism: 噪声机制 ('laplace', 'gaussian')
        random_state: 随机种子

    返回:
        加噪后的数据
    """
    data = np.array(data, dtype=float)
    original_shape = data.shape

    # 展平数据
    if data.ndim == 0:
        data = data.reshape(1)
    else:
        data = data.flatten()

    # 设置随机种子
    if random_state is not None:
        np.random.seed(random_state)

    # 自动计算敏感度（最大值 - 最小值）
    if sensitivity is None:
        sensitivity = data.max() - data.min()

    # 拉普拉斯机制
    if mechanism == 'laplace':
        scale = sensitivity / epsilon if epsilon > 0 else np.inf
        noise = np.random.laplace(0, scale, size=data.shape)

    # 高斯机制（需要 delta 参数）
    elif mechanism == 'gaussian':
        # 使用默认 delta = 1 / (len(data) * sqrt(len(data)))
        delta = 1 / (len(data) * np.sqrt(len(data)))
        sigma = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon
        noise = np.random.normal(0, sigma, size=data.shape)

    else:
        raise ValueError(f"未知的噪声机制: {mechanism}")

    # 添加噪声并恢复形状
    private_data = data + noise

    return private_data.reshape(original_shape)


def check_privacy_budget(epsilon_or_epsilons, total_budget=1.0):
    """
    检查隐私预算

    参数:
        epsilon_or_epsilons: 单个 epsilon 或 epsilon 列表
        total_budget: 总隐私预算

    返回:
        预算状态字典或布尔值
    """
    # 处理单个 epsilon
    if isinstance(epsilon_or_epsilons, (int, float)):
        total_epsilon = epsilon_or_epsilons
    else:
        total_epsilon = sum(epsilon_or_epsilons)

    exceeded = total_epsilon > total_budget

    return {
        'total_epsilon': total_epsilon,
        'budget': total_budget,
        'exceeded': exceeded,
        'remaining': max(0, total_budget - total_epsilon),
        'status': 'over_budget' if exceeded else 'within_budget'
    }


# =============================================================================
# 伦理审查函数
# =============================================================================

def create_ethics_checklist(model_info=None, fairness_metrics=None,
                           privacy_info=None):
    """
    创建伦理审查清单

    参数:
        model_info: 模型信息字典
        fairness_metrics: 公平性指标字典
        privacy_info: 隐私信息字典

    返回:
        审查清单字典
    """
    checklist = {
        'data_bias': {
            'checked': False,
            'items': [
                '训练数据是否存在历史歧视？',
                '某些群体的样本量是否不足？',
                '数据收集过程是否存在偏见？'
            ]
        },
        'algorithm_bias': {
            'checked': False,
            'items': [
                '模型是否放大数据中的模式？',
                '模型是否对某些群体过拟合？',
                '评估指标是否考虑了群体差异？'
            ]
        },
        'proxy_variables': {
            'checked': False,
            'items': [
                '是否存在敏感属性的代理变量？',
                '是否检测了特征与敏感属性的相关性？',
                '是否删除或调整了代理变量？'
            ]
        },
        'fairness': {
            'checked': False,
            'items': [
                '差异影响比是否 >= 0.8？',
                '平等机会差异是否 < 0.1？',
                '是否在不同群体上分别评估了模型？'
            ]
        },
        'privacy': {
            'checked': False,
            'items': [
                '数据发布是否使用差分隐私？',
                '是否可能通过模型反推训练数据？',
                '是否限制了模型访问权限？'
            ]
        },
        'reproducibility': {
            'checked': True,  # 默认已检查
            'items': [
                '随机种子是否记录？',
                '数据来源是否记录？',
                '代码版本是否记录？'
            ]
        },
        'limitations': {
            'checked': True,  # 默认已检查
            'items': [
                '模型的局限性是否明确？',
                '是否说明了模型的适用范围？',
                '是否避免了因果声明？'
            ]
        }
    }

    # 如果提供了公平性指标，更新审查结果
    if fairness_metrics:
        if fairness_metrics.get('disparate_impact', 1.0) < 0.8:
            checklist['fairness']['issues'] = ['差异影响比 < 0.8，可能存在法律风险']
        if fairness_metrics.get('equal_opportunity_diff', 0) > 0.1:
            checklist['fairness']['issues'] = checklist['fairness'].get('issues', [])
            checklist['fairness']['issues'].append('平等机会差异 > 0.1，模型在不同群体上性能差异显著')

    # 如果提供了模型信息，添加到清单
    if model_info:
        checklist['model_info'] = model_info

    # 如果提供了隐私信息，更新审查结果
    if privacy_info:
        checklist['privacy_info'] = privacy_info

    return checklist


# =============================================================================
# 解释与代码审查函数
# =============================================================================

def explain_to_nontechnical(prediction_explanation, audience='customer',
                           include_recommendations=False, **kwargs):
    """
    向非技术人员解释预测

    参数:
        prediction_explanation: 预测解释字典
        audience: 受众类型 ('customer', 'product_manager', 'compliance')
        include_recommendations: 是否包含建议
        **kwargs: 额外参数（兼容性）

    返回:
        易读的解释文本
    """
    base_value = prediction_explanation.get('base_value', 0.2)
    shap_values = prediction_explanation.get('shap_values', {})
    final_value = prediction_explanation.get('final_value', 0.5)
    feature_values = prediction_explanation.get('feature_values', {})

    # 找出最重要的正向和负向贡献特征
    contrib_items = list(shap_values.items())
    contrib_items.sort(key=lambda x: abs(x[1]), reverse=True)

    top_positive = [(f, v) for f, v in contrib_items if v > 0][:2]
    top_negative = [(f, v) for f, v in contrib_items if v < 0][:2]

    if audience == 'customer':
        # 客户版本：简单、易懂
        explanation_parts = []

        # 总体结论
        if final_value > 0.5:
            explanation_parts.append("您的申请有风险，主要因为：")
        else:
            explanation_parts.append("您的申请通过概率较高，主要因为：")

        # 添加主要原因
        for feature, value in top_negative[:2]:
            feature_name_cn = translate_feature_name(feature)
            feature_val = feature_values.get(feature, 0)
            explanation_parts.append(f"- {feature_name_cn}（{feature_val:.2f}）增加了风险")

        # 添加建议
        if include_recommendations:
            explanation_parts.append("\n建议：")
            for feature, value in top_negative[:2]:
                feature_name_cn = translate_feature_name(feature)
                explanation_parts.append(f"- 改善{feature_name_cn}可以提高通过概率")

        return "\n".join(explanation_parts)

    elif audience == 'product_manager':
        # 产品经理版本：平衡技术性和可读性
        explanation = f"模型预测结果为 {final_value:.2%}，\n\n"
        explanation += "主要影响因素：\n"

        for feature, value in contrib_items[:3]:
            direction = "提高" if value > 0 else "降低"
            explanation += f"- {feature}: {direction}了概率 ({value:+.3f})\n"

        return explanation

    elif audience == 'compliance':
        # 合规版本：正式、详细
        explanation = "模型预测与公平性评估报告\n\n"
        explanation += f"基准值: {base_value:.3f}\n"
        explanation += f"预测值: {final_value:.3f}\n\n"

        explanation += "特征贡献：\n"
        for feature, value in contrib_items[:5]:
            explanation += f"- {feature}: {value:+.4f}\n"

        # 如果有公平性指标
        if 'model_metadata' in prediction_explanation:
            metadata = prediction_explanation['model_metadata']
            if 'fairness_metrics' in metadata:
                explanation += "\n公平性指标：\n"
                for metric, value in metadata['fairness_metrics'].items():
                    explanation += f"- {metric}: {value}\n"

        return explanation

    else:
        # 默认版本
        return f"预测值为 {final_value:.2f}，主要受 {list(shap_values.keys())[:3]} 等特征影响。"


def translate_feature_name(feature_name):
    """
    翻译特征名为中文（简化版）

    参数:
        feature_name: 特征名

    返回:
        中文名称
    """
    translations = {
        'income': '月收入',
        'credit_history_age': '信用历史长度',
        'debt_to_income': '债务收入比',
        'credit_inquiries': '信用查询次数',
        'employment_length': '工作年限',
        'age': '年龄',
        'feature_1': '特征1',
        'feature_2': '特征2',
        'feature_3': '特征3',
    }

    return translations.get(feature_name, feature_name)


def review_xai_code(code):
    """
    审查 XAI 代码（检测常见问题）

    参数:
        code: 代码字符串

    返回:
        审查结果字典，包含:
        - has_issues: 是否有问题
        - issues: 问题列表（可以是字符串或字典）
        - suggestions: 建议列表
        - critical_issues: 严重问题数量
        - total_issues: 总问题数
    """
    issues = []
    suggestions = []

    # 检查 1: 是否使用了错误的 explainer
    if 'shap.Explainer(' in code and 'RandomForest' in code:
        issues.append('对树模型使用通用 Explainer 效率较低，建议使用 TreeExplainer')

    # 检查 2: 是否只做了全局解释
    has_summary_plot = 'summary_plot' in code
    has_force_plot = 'force_plot' in code or 'dependence_plot' in code

    if has_summary_plot and not has_force_plot:
        issues.append('Code only includes global explanation (summary_plot), missing single sample or local explanation (force_plot)')

    # 检查 3: 是否进行了群体分析（只在代码涉及群体分析时检查）
    # 如果代码包含 'auc'、'group'、'gender' 等公平性相关关键词，
    # 但没有进行群体分析，则标记为问题
    has_fairness_context = (
        'auc' in code.lower() or
        'roc' in code.lower() or
        'accuracy' in code.lower()
    )
    has_group_analysis = (
        'group' in code.lower() or
        'gender' in code.lower() or
        'disparate' in code.lower() or
        'fairness' in code.lower()
    )

    # 只有在公平性上下文中才要求群体分析
    if has_fairness_context and not has_group_analysis and not 'TreeExplainer' in code:
        issues.append('建议按敏感特征（如性别）分别评估模型以检查公平性')

    # 检查 4: 是否使用了差分隐私
    has_privacy = (
        'differential_privacy' in code.lower() or
        'laplace' in code.lower() or
        'epsilon' in code.lower()
    )

    if not has_privacy:
        suggestions.append('考虑使用差分隐私保护数据发布')

    # 检查 5: 是否有随机种子
    has_random_seed = 'random_state' in code or 'np.random.seed' in code

    if not has_random_seed:
        issues.append('建议设置随机种子以提高可复现性')

    # 计算严重问题数量（第一个问题是 warning）
    critical_issues = 1 if 'shap.Explainer(' in code and 'RandomForest' in code else 0

    return {
        'has_issues': len(issues) > 0,
        'issues': issues,
        'suggestions': suggestions,
        'critical_issues': critical_issues,
        'total_issues': len(issues)
    }


# =============================================================================
# 主函数
# =============================================================================

def main():
    """运行所有作业参考答案"""
    print("\n" + "=" * 60)
    print("Week 12 作业参考答案")
    print("=" * 60)

    print("\n本文件提供以下函数：")
    print("\n1. SHAP 可解释性:")
    print("   - calculate_shap_values(model, X)")
    print("   - explain_single_prediction(model, sample)")
    print("   - calculate_feature_importance_shap(model, X)")

    print("\n2. 公平性指标:")
    print("   - calculate_disparate_impact(y_pred, group_labels)")
    print("   - calculate_equal_opportunity(y_true, y_pred, group_labels)")
    print("   - calculate_equalized_odds(y_true, y_pred, group_labels)")
    print("   - detect_proxy_variables(df, sensitive_col)")

    print("\n3. 差分隐私:")
    print("   - add_differential_privacy_noise(data, epsilon, sensitivity)")
    print("   - check_privacy_budget(epsilon_or_epsilons, total_budget)")

    print("\n4. 伦理审查:")
    print("   - create_ethics_checklist(model_info, fairness_metrics, privacy_info)")
    print("   - explain_to_nontechnical(prediction_explanation, audience)")
    print("   - review_xai_code(code)")

    print("\n" + "=" * 60)
    print("参考答案实现完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()

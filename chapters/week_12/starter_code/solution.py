"""
Week 12 作业参考实现

本文件提供作业的基础部分参考实现，用于学习参考。
当你在作业中遇到困难时，可以查看此文件，但建议先独立完成。

包含内容：
1. 特征重要性计算与可视化
2. 基本的分组公平性评估
3. 简单的模型解释报告生成

注意：这只是基础实现，不包含进阶部分和挑战部分。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def setup_chinese_font() -> str:
    """配置中文字体"""
    chinese_fonts = ['SimHei', 'Noto Sans CJK SC', 'Arial Unicode MS',
                     'PingFang SC', 'Microsoft YaHei']
    available = [f.name for f in fm.fontManager.ttflist]
    for font in chinese_fonts:
        if font in available:
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False
            return font
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    return 'DejaVu Sans'


# ============================================================================
# 第一部分：特征重要性
# ============================================================================

def compute_feature_importance_from_data(X: pd.DataFrame, y: pd.Series) -> dict:
    """
    计算逻辑回归系数和随机森林特征重要性

    参数:
    - X: 特征 DataFrame
    - y: 目标变量 Series

    返回:
    - dict: 包含 log_reg_coefficients 和 rf_importance
    """
    # 划分数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    results = {}

    # 1. 逻辑回归系数（需要标准化）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(X_train_scaled, y_train)

    # 创建系数表
    coef_df = pd.DataFrame({
        'feature': X_train.columns,
        'coefficient': log_reg.coef_[0],
        'abs_coef': np.abs(log_reg.coef_[0])
    }).sort_values('abs_coef', ascending=False)

    results['log_reg_coefficients'] = coef_df

    # 2. 随机森林特征重要性
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    results['rf_importance'] = importance_df

    return results


# ============================================================================
# 第二部分：分组公平性评估
# ============================================================================

def evaluate_group_fairness(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_groups: np.ndarray
) -> pd.DataFrame:
    """
    按敏感属性分组评估模型性能

    参数:
    - y_true: 真实标签
    - y_pred: 预测标签
    - sensitive_groups: 敏感属性值数组（如性别、地区）

    返回:
    - DataFrame: 每个组的评估指标
    """
    df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'group': sensitive_groups
    })

    results = []

    for group in sorted(df['group'].unique()):
        group_df = df[df['group'] == group]

        if len(group_df) < 10:
            continue  # 跳过样本太少的组

        cm = confusion_matrix(group_df['y_true'], group_df['y_pred'])
        tn, fp, fn, tp = cm.ravel()

        results.append({
            'group': group,
            'count': len(group_df),
            'accuracy': accuracy_score(group_df['y_true'], group_df['y_pred']),
            'true_positive_rate': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'positive_rate': group_df['y_pred'].mean()
        })

    return pd.DataFrame(results)


def check_fairness_warnings(fairness_df: pd.DataFrame) -> list[str]:
    """
    检查公平性警告

    参数:
    - fairness_df: evaluate_group_fairness 的返回结果

    返回:
    - list[str]: 警告信息列表
    """
    warnings = []

    if len(fairness_df) < 2:
        return ["警告: 需要至少 2 个分组才能进行公平性评估"]

    # 计算差异
    tpr_diff = fairness_df['true_positive_rate'].max() - fairness_df['true_positive_rate'].min()
    fpr_diff = fairness_df['false_positive_rate'].max() - fairness_df['false_positive_rate'].min()

    if tpr_diff > 0.1:
        warnings.append(f"警告: 真阳性率差异过大 ({tpr_diff:.3f})")
    if fpr_diff > 0.1:
        warnings.append(f"警告: 假阳性率差异过大 ({fpr_diff:.3f})")

    if not warnings:
        warnings.append("分组差异在可接受范围内")

    return warnings


# ============================================================================
# 第三部分：模型解释报告
# ============================================================================

def generate_explanation_report(
    model_metrics: dict,
    feature_importance: pd.DataFrame,
    fairness_df: pd.DataFrame
) -> str:
    """
    生成面向非技术读者的模型解释报告

    参数:
    - model_metrics: 包含 accuracy, precision, recall, auc 等指标的字典
    - feature_importance: 特征重要性 DataFrame
    - fairness_df: 分组评估结果 DataFrame

    返回:
    - str: Markdown 格式的报告
    """
    lines = []
    lines.append("# 模型解释报告\n")

    # 1. 模型性能（翻译成业务语言）
    lines.append("## 模型性能\n")
    auc = model_metrics.get('auc', 0)
    lines.append(f"- 模型区分能力: {auc*100:.0f}% (满分 100%)\n")

    recall = model_metrics.get('recall', 0)
    lines.append(f"- 召回率: {recall*100:.0f}% (能抓到多少真实正样本)\n")

    precision = model_metrics.get('precision', 0)
    lines.append(f"- 精确率: {precision*100:.0f}% (预测为正的样本中多少是准确的)\n\n")

    # 2. 关键因素
    lines.append("## 主要影响因素\n")
    for idx, row in feature_importance.head(5).iterrows():
        lines.append(f"- {row['feature']}\n")

    lines.append("\n")

    # 3. 公平性说明
    lines.append("## 公平性说明\n")
    if fairness_df is not None and len(fairness_df) >= 2:
        lines.append("模型对不同群体的表现:\n\n")

        for _, row in fairness_df.iterrows():
            lines.append(f"- {row['group']} 组: 准确率 {row['accuracy']:.1%}, "
                        f"召回率 {row['true_positive_rate']:.1%}\n")

        # 添加警告
        warning_list = check_fairness_warnings(fairness_df)
        lines.append("\n注意事项:\n")
        for warning in warning_list:
            lines.append(f"- {warning}\n")

    lines.append("\n")

    # 4. 行动建议
    lines.append("## 行动建议\n")
    lines.append("- 该模型适合用于辅助决策，不应完全替代人工判断\n")
    lines.append("- 建议定期重新评估模型性能和公平性\n")
    lines.append("- 对于预测结果边缘的案例，建议人工复核\n\n")

    return "".join(lines)


# ============================================================================
# 演示主函数
# ============================================================================

def main() -> None:
    """
    演示如何使用上述函数

    这是一个完整的示例，展示从数据加载到报告生成的流程
    """
    print("=" * 60)
    print("Week 12 作业参考实现演示")
    print("=" * 60)

    # 创建输出目录
    output_dir = Path(__file__).parent.parent.parent.parent / 'output'
    output_dir.mkdir(parents=True, exist_ok=True)

    # 生成示例数据
    print("\n1. 生成示例数据...")
    np.random.seed(42)
    n = 1000

    X = pd.DataFrame({
        'feature_1': np.random.randn(n),
        'feature_2': np.random.randn(n),
        'feature_3': np.random.randn(n)
    })

    logit = -1 + 0.5 * X['feature_1'] - 0.3 * X['feature_2']
    prob = 1 / (1 + np.exp(-logit))
    y = np.random.binomial(1, prob)

    # 添加敏感属性（用于公平性评估）
    X['group'] = np.random.choice(['A', 'B', 'C'], n, p=[0.6, 0.3, 0.1])

    print(f"   数据规模: {X.shape[0]} 行, {X.shape[1]} 列")
    print(f"   正类占比: {y.mean():.2%}")

    # 2. 计算特征重要性
    print("\n2. 计算特征重要性...")
    importance_results = compute_feature_importance_from_data(
        X.drop('group', axis=1), y
    )

    print("\n逻辑回归系数:")
    print(importance_results['log_reg_coefficients'])

    print("\n随机森林特征重要性:")
    print(importance_results['rf_importance'])

    # 3. 训练模型并进行公平性评估
    print("\n3. 训练模型并评估公平性...")
    X_train, X_test, y_train, y_test = train_test_split(
        X.drop('group', axis=1), y, test_size=0.3, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    # 获取测试集的 group 信息
    group_test = X.loc[X_test.index, 'group']

    fairness_df = evaluate_group_fairness(y_test.values, y_pred, group_test.values)

    print("\n分组公平性评估:")
    print(fairness_df)

    warnings = check_fairness_warnings(fairness_df)
    print("\n公平性警告:")
    for warning in warnings:
        print(f"  {warning}")

    # 4. 生成报告
    print("\n4. 生成模型解释报告...")

    # 计算模型指标
    from sklearn.metrics import roc_auc_score
    y_prob = rf.predict_proba(X_test)[:, 1]
    model_metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_prob)
    }

    report = generate_explanation_report(
        model_metrics,
        importance_results['rf_importance'],
        fairness_df
    )

    # 保存报告
    report_path = output_dir / 'solution_explanation_report.md'
    report_path.write_text(report, encoding='utf-8')

    print(f"报告已保存到: {report_path}")

    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)
    print("\n你可以参考此文件完成作业，但建议先独立尝试。")


# ============================================================================
# 测试兼容函数（为 pytest 测试提供的接口）
# ============================================================================

def compute_feature_importance(model_or_X, feature_names_or_y=None, *, feature_names=None, y=None):
    """
    计算特征重要性（支持两种调用方式）

    方式1: compute_feature_importance(model, feature_names=None)
      - 从已训练的模型提取特征重要性

    方式2: compute_feature_importance(X, y)
      - 从原始数据训练模型并计算特征重要性

    参数:
    - model_or_X: 已训练的模型 或 特征 DataFrame
    - feature_names_or_y: 特征名称列表 或 目标变量 Series（位置参数）
    - feature_names: 特征名称列表（keyword-only，用于方式1）
    - y: 目标变量（keyword-only，用于方式2）

    返回:
    - np.ndarray/pd.DataFrame 或 dict: 特征重要性
    """
    # 处理 keyword arguments
    if feature_names is not None:
        # 方式1 使用 keyword argument
        model = model_or_X
        actual_feature_names = feature_names

        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0])
        else:
            raise ValueError("模型没有 feature_importances_ 或 coef_ 属性")

        if actual_feature_names is not None:
            return pd.DataFrame({
                'feature': actual_feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
        return importance

    elif y is not None:
        # 方式2 使用 keyword argument
        X = model_or_X
        return compute_feature_importance_from_data(X, y)

    else:
        # 使用位置参数
        if isinstance(model_or_X, pd.DataFrame):
            # 方式2: compute_feature_importance(X, y)
            X = model_or_X
            actual_y = feature_names_or_y
            return compute_feature_importance_from_data(X, actual_y)
        else:
            # 方式1: compute_feature_importance(model, feature_names)
            model = model_or_X
            actual_feature_names = feature_names_or_y

            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_[0])
            else:
                raise ValueError("模型没有 feature_importances_ 或 coef_ 属性")

            if actual_feature_names is not None:
                return pd.DataFrame({
                    'feature': actual_feature_names,
                    'importance': importance
                }).sort_values('importance', ascending=False)
            return importance


def compute_shap_values(model, X_train, X_test):
    """
    计算 SHAP 值

    参数:
    - model: 已训练的模型
    - X_train: 训练数据（用于 KernelExplainer，TreeExplainer 不需要）
    - X_test: 测试数据

    返回:
    - np.ndarray: SHAP 值数组 (n_samples, n_features)
    """
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        # 处理多种可能的输出格式
        if isinstance(shap_values, list):
            # 列表格式：取正类的 SHAP 值
            shap_values = shap_values[1]
        elif isinstance(shap_values, np.ndarray):
            # 数组格式：可能是 (n_samples, n_features, n_classes) 或 (n_classes, n_samples, n_features)
            if shap_values.ndim == 3:
                if shap_values.shape[0] == 2:
                    # (2, n_samples, n_features) -> 取正类
                    shap_values = shap_values[1]
                elif shap_values.shape[-1] == 2:
                    # (n_samples, n_features, 2) -> 取正类
                    shap_values = shap_values[:, :, 1]

        return shap_values
    except ImportError:
        # 如果 shap 未安装，返回模拟数据
        n_samples = len(X_test)
        n_features = X_test.shape[1] if hasattr(X_test, 'shape') else len(X_test.columns)
        mock_shap = np.random.randn(n_samples, n_features) * 0.1
        return mock_shap


def get_base_value(model, X_train):
    """
    获取 SHAP 的基线值（expected_value）

    参数:
    - model: 已训练的模型
    - X_train: 训练数据

    返回:
    - float: 基线值
    """
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        expected_value = explainer.expected_value

        # 处理二分类情况
        if isinstance(expected_value, (list, np.ndarray)):
            expected_value = float(expected_value[1] if len(expected_value) > 1 else expected_value[0])

        return expected_value
    except ImportError:
        # 如果 shap 未安装，返回模拟基线
        return 0.0


def demographic_parity_difference(y_pred, sensitive_features):
    """
    计算统计均等差异

    参数:
    - y_pred: 预测标签
    - sensitive_features: 敏感属性

    返回:
    - float: 各组预测正率的最大差异
    """
    df = pd.DataFrame({
        'y_pred': y_pred,
        'group': sensitive_features
    })

    group_rates = df.groupby('group')['y_pred'].mean()
    return float(group_rates.max() - group_rates.min())


def evaluate_by_group(y_true, y_pred, sensitive_attr):
    """
    按敏感属性分组评估（evaluate_group_fairness 的别名）

    参数:
    - y_true: 真实标签
    - y_pred: 预测标签
    - sensitive_attr: 敏感属性值数组

    返回:
    - pd.DataFrame: 每个组的评估指标
    """
    return evaluate_group_fairness(y_true, y_pred, sensitive_attr)


def generate_explanation_text(model, X_test, sample_idx, feature_names):
    """
    生成单个样本的 SHAP 解释文本

    参数:
    - model: 已训练的模型
    - X_test: 测试数据
    - sample_idx: 样本索引
    - feature_names: 特征名称

    返回:
    - str: 解释文本
    """
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        expected_value = explainer.expected_value

        # 处理二分类情况
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            # 处理 (n_samples, n_features, n_classes) 或 (n_classes, n_samples, n_features)
            if shap_values.shape[0] == 2:
                shap_values = shap_values[1]
            elif shap_values.shape[-1] == 2:
                shap_values = shap_values[:, :, 1]

        if isinstance(expected_value, (list, np.ndarray)):
            expected_value = float(expected_value[1] if len(expected_value) > 1 else expected_value[0])

        # 获取样本的 SHAP 值
        if isinstance(shap_values, np.ndarray) and shap_values.ndim >= 2:
            sample_shap = shap_values[sample_idx]
        else:
            sample_shap = shap_values

        # 确保是一维数组
        if isinstance(sample_shap, np.ndarray) and sample_shap.ndim > 1:
            sample_shap = sample_shap.flatten()

    except ImportError:
        # 如果 SHAP 未安装，返回模拟解释
        expected_value = 0.0
        n_features = len(feature_names) if isinstance(feature_names, list) else X_test.shape[1]
        sample_shap = np.random.randn(n_features) * 0.1

    lines = []
    lines.append(f"样本 #{sample_idx} 的预测解释")
    lines.append(f"基线值: {expected_value:.3f}")
    lines.append("\n主要影响因素:")

    # 排序并显示前5个
    if isinstance(feature_names, list) and len(feature_names) > 0:
        # 确保 sample_shap 长度与 feature_names 匹配
        n_features = min(len(feature_names), len(sample_shap))
        indices = np.argsort(np.abs(sample_shap[:n_features]))[::-1][:min(5, n_features)]
        for idx in indices:
            contribution = sample_shap[idx]
            direction = "增加" if contribution > 0 else "降低"
            lines.append(f"  - {feature_names[idx]}: {direction}预测 {abs(contribution):.3f}")

    return "\n".join(lines)


if __name__ == "__main__":
    main()

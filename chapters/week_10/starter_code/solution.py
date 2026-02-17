"""
Week 10 作业参考答案：准确率陷阱——分类模型与评估

本文件提供了 Week 10 作业的参考实现，供学生在遇到困难时查看。
建议学生先自己尝试完成作业，遇到问题后再参考此答案。

运行方式：python3 chapters/week_10/starter_code/solution.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, roc_auc_score)
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

# 配置中文字体
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


# ========================================
# 第 1 题：分类 vs 回归
# ========================================

def question1_classification_vs_regression():
    """
    第 1 题：区分分类和回归问题

    题目：判断以下问题是分类还是回归
    """
    print("=" * 60)
    print("第 1 题：分类 vs 回归")
    print("=" * 60)

    problems = {
        "预测客户的销售额": "回归",
        "预测客户是否会流失": "分类",
        "预测房屋价格": "回归",
        "预测邮件是否为垃圾邮件": "分类",
        "预测明天的温度": "回归",
        "预测手写数字是 0-9 中的哪个": "分类"
    }

    print("\n问题类型判断：")
    for problem, answer in problems.items():
        print(f"  {problem:<30} -> {answer}")

    # 代码示例：为什么不能用线性回归做分类
    print("\n为什么不能用线性回归做分类：")
    print("  1. 预测值可能超出 [0, 1] 范围")
    print("  2. 假设残差正态，但 0/1 数据的残差不可能正态")
    print("  3. 对异常值敏感")

    # 正确做法：逻辑回归
    print("\n正确做法：使用逻辑回归")
    print("  - 输出概率值在 [0, 1] 之间")
    print("  - 使用 Sigmoid 函数将线性组合映射为概率")


# ========================================
# 第 2 题：逻辑回归与混淆矩阵
# ========================================

def question2_logistic_regression_and_confusion_matrix():
    """
    第 2 题：训练逻辑回归模型并计算混淆矩阵
    """
    print("\n" + "=" * 60)
    print("第 2 题：逻辑回归与混淆矩阵")
    print("=" * 60)

    # 加载数据
    titanic = sns.load_dataset("titanic")
    features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
    X = titanic[features].copy()
    y = titanic['survived']

    # 预处理
    numeric_features = ['age', 'sibsp', 'parch', 'fare']
    categorical_features = ['pclass', 'sex', 'embarked']

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    # 创建 Pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', LogisticRegression(max_iter=1000, random_state=42))
    ])

    # 划分数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 训练
    pipeline.fit(X_train, y_train)

    # 预测
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # 计算混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print("\n混淆矩阵:")
    print(f"                预测负类  预测正类")
    print(f"实际负类 (0)  |   {tn:3d}   |   {fp:3d}   |")
    print(f"实际正类 (1)  |   {fn:3d}   |   {tp:3d}   |")

    # 计算评估指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"\n评估指标:")
    print(f"  准确率   = {accuracy:.4f}")
    print(f"  精确率   = {precision:.4f}")
    print(f"  召回率   = {recall:.4f}")
    print(f"  F1 分数  = {f1:.4f}")

    # 计算公式说明
    print(f"\n计算公式:")
    print(f"  准确率   = (TP + TN) / 总数 = ({tp} + {tn}) / {cm.sum()}")
    print(f"  精确率   = TP / (TP + FP) = {tp} / ({tp} + {fp})")
    print(f"  召回率   = TP / (TP + FN) = {tp} / ({tp} + {fn})")
    print(f"  F1 分数  = 2 * 精确率 * 召回率 / (精确率 + 召回率)")

    return {
        'confusion_matrix': cm,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'y_test': y_test,
        'y_prob': y_prob
    }


# ========================================
# 第 3 题：ROC 曲线与 AUC
# ========================================

def question3_roc_auc(results):
    """
    第 3 题：绘制 ROC 曲线并计算 AUC
    """
    print("\n" + "=" * 60)
    print("第 3 题：ROC 曲线与 AUC")
    print("=" * 60)

    y_test = results['y_test']
    y_prob = results['y_prob']

    # 计算 ROC 曲线
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    print(f"\nAUC: {auc:.4f}")

    # AUC 解读
    if auc > 0.9:
        quality = "优秀"
    elif auc > 0.8:
        quality = "良好"
    elif auc > 0.7:
        quality = "一般"
    else:
        quality = "较差"
    print(f"分类器质量: {quality}")

    # 绘制 ROC 曲线
    font = setup_chinese_font()
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', label='随机猜测 (AUC = 0.5)', linewidth=2)
    ax.set_xlabel('假正率 FPR (1 - 特异度)')
    ax.set_ylabel('真正率 TPR (召回率)')
    ax.set_title('ROC 曲线')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图片
    output_dir = Path(__file__).parent.parent / 'starter_code_output'
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'solution_roc_curve.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"\nROC 曲线已保存到: {output_dir / 'solution_roc_curve.png'}")

    # AUC vs 准确率
    print(f"\nAUC vs 准确率:")
    print(f"  AUC: {auc:.4f} - 综合所有阈值的分类器表现")
    print(f"  准确率: {results['accuracy']:.4f} - 单一阈值 (0.5) 下的表现")
    print(f"  AUC 的优势: 不依赖阈值选择，适用于类别不平衡数据")


# ========================================
# 第 4 题：数据泄漏与 Pipeline
# ========================================

def question4_data_leakage_and_pipeline():
    """
    第 4 题：对比有无 Pipeline 的交叉验证结果
    """
    print("\n" + "=" * 60)
    print("第 4 题：数据泄漏与 Pipeline")
    print("=" * 60)

    # 加载数据
    titanic = sns.load_dataset("titanic")
    features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
    X = titanic[features].copy()
    y = titanic['survived']

    # 错误做法：先填充缺失值
    print("\n错误做法：在划分之前填充缺失值（数据泄漏）")
    X_filled = X.copy()
    numeric_cols = ['age', 'sibsp', 'parch', 'fare']
    cat_cols = ['pclass', 'sex', 'embarked']

    for col in numeric_cols:
        X_filled[col] = X_filled[col].fillna(X_filled[col].mean())
    for col in cat_cols:
        X_filled[col] = X_filled[col].fillna(X_filled[col].mode()[0])

    X_encoded = pd.get_dummies(X_filled, columns=cat_cols, drop_first=True)

    model = LogisticRegression(max_iter=1000, random_state=42)
    scores_bad = cross_val_score(model, X_encoded, y, cv=5, scoring='roc_auc')
    print(f"交叉验证 AUC: {scores_bad.mean():.4f} (+/- {scores_bad.std():.4f})")
    print("问题: 填充缺失值时用了全部数据的统计量，测试集信息泄漏到训练集")

    # 正确做法：使用 Pipeline
    print("\n正确做法：使用 Pipeline（防止数据泄漏）")
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, cat_cols)
    ])
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', LogisticRegression(max_iter=1000, random_state=42))
    ])
    scores_good = cross_val_score(pipeline, X, y, cv=5, scoring='roc_auc')
    print(f"交叉验证 AUC: {scores_good.mean():.4f} (+/- {scores_good.std():.4f})")
    print("优势: 每一折的预处理都是独立的，只用训练集的统计量")

    print(f"\n对比:")
    print(f"  错误做法 AUC: {scores_bad.mean():.4f} (虚高)")
    print(f"  正确做法 AUC: {scores_good.mean():.4f} (真实)")


# ========================================
# 第 5 题：类别不平衡与准确率陷阱
# ========================================

def question5_imbalanced_data():
    """
    第 5 题：类别不平衡时的准确率陷阱
    """
    print("\n" + "=" * 60)
    print("第 5 题：类别不平衡与准确率陷阱")
    print("=" * 60)

    # 创建类别不平衡的数据
    np.random.seed(42)
    n_samples = 1000
    n_positive = 200  # 20% 正类
    n_negative = 800  # 80% 负类

    # 生成数据
    X_neg = np.random.randn(n_negative, 2) + np.array([0, 0])
    y_neg = np.zeros(n_negative, dtype=int)
    X_pos = np.random.randn(n_positive, 2) + np.array([2, 2])
    y_pos = np.ones(n_positive, dtype=int)

    X = np.vstack([X_neg, X_pos])
    y = np.hstack([y_neg, y_pos])

    print(f"\n数据集:")
    print(f"  总样本数: {len(y)}")
    print(f"  负类: {n_negative} (80%)")
    print(f"  正类: {n_positive} (20%)")

    # 训练模型
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 计算指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"\n模型评估:")
    print(f"  准确率: {accuracy:.4f}")
    print(f"  精确率: {precision:.4f}")
    print(f"  召回率: {recall:.4f}")
    print(f"  F1 分数: {f1:.4f}")

    # 傻瓜基线
    dummy_pred = np.zeros_like(y_test)
    dummy_accuracy = accuracy_score(y_test, dummy_pred)

    print(f"\n傻瓜基线（永远预测负类）:")
    print(f"  准确率: {dummy_accuracy:.4f}")
    print(f"  召回率: 0.0000 (完全漏掉所有正类)")

    print(f"\n结论:")
    print(f"  傻瓜基线的准确率是 {dummy_accuracy:.4f}，但召回率是 0")
    print(f"  类别不平衡时，准确率会误导，必须看精确率、召回率、F1")


# ========================================
# 主函数
# ========================================

def main():
    """运行所有题目的参考答案"""
    print("\n" + "=" * 60)
    print("Week 10 作业参考答案")
    print("=" * 60)

    # 第 1 题
    question1_classification_vs_regression()

    # 第 2 题
    results = question2_logistic_regression_and_confusion_matrix()

    # 第 3 题
    question3_roc_auc(results)

    # 第 4 题
    question4_data_leakage_and_pipeline()

    # 第 5 题
    question5_imbalanced_data()

    print("\n" + "=" * 60)
    print("所有题目完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()


# ========================================
# 测试框架所需的函数别名
# ========================================

def train_logistic_regression(X, y, **kwargs):
    """训练逻辑回归模型（测试框架别名）"""
    model = LogisticRegression(**kwargs)
    model.fit(X, y)
    return model

fit_logistic_regression = train_logistic_regression  # 别名

def predict_logistic(model, X):
    """使用逻辑回归模型预测（测试框架别名）"""
    return model.predict(X), model.predict_proba(X)

def calculate_confusion_matrix(y_true, y_pred):
    """计算混淆矩阵（测试框架别名）"""
    return confusion_matrix(y_true, y_pred)

def calculate_accuracy(y_true, y_pred):
    """计算准确率（测试框架别名）"""
    return accuracy_score(y_true, y_pred)

def calculate_precision(y_true, y_pred, **kwargs):
    """计算精确率（测试框架别名）"""
    return precision_score(y_true, y_pred, **kwargs)

def calculate_recall(y_true, y_pred, **kwargs):
    """计算召回率（测试框架别名）"""
    return recall_score(y_true, y_pred, **kwargs)

def calculate_f1(y_true, y_pred, **kwargs):
    """计算 F1 分数（测试框架别名）"""
    return f1_score(y_true, y_pred, **kwargs)

def calculate_metrics(y_true, y_pred, **kwargs):
    """计算所有评估指标（测试框架别名）"""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, **kwargs),
        'recall': recall_score(y_true, y_pred, **kwargs),
        'f1': f1_score(y_true, y_pred, **kwargs)
    }

def calculate_roc_curve(y_true, y_prob):
    """计算 ROC 曲线数据（测试框架别名）"""
    return roc_curve(y_true, y_prob)

def calculate_auc(y_true, y_prob):
    """计算 AUC（测试框架别名）"""
    return roc_auc_score(y_true, y_prob)

def plot_roc_curve(fpr, tpr, auc_value, save_path=None):
    """绘制 ROC 曲线（测试框架别名）"""
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_value:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    return plt.gcf()

def create_classification_pipeline(numeric_features=None, categorical_features=None, model=None):
    """创建分类 Pipeline（测试框架别名）

    支持两种调用方式：
    1. 无参数：创建简单的数值型 Pipeline（用于测试）
    2. 带参数：创建带 ColumnTransformer 的完整 Pipeline（用于实战）
    """
    if model is None:
        model = LogisticRegression(max_iter=1000)

    # 如果没有指定特征列表，创建简单的数值型 Pipeline（含 imputer 处理缺失值）
    if numeric_features is None and categorical_features is None:
        return Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('model', model)
        ])

    # 完整 Pipeline：带 ColumnTransformer
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    transformers = []
    if numeric_features:
        transformers.append(('num', numeric_transformer, numeric_features))
    if categorical_features:
        transformers.append(('cat', categorical_transformer, categorical_features))

    preprocessor = ColumnTransformer(transformers)

    return Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])


def train_with_pipeline(pipeline_or_X, X_or_y, y=None):
    """使用 Pipeline 训练模型（测试框架别名）

    支持两种调用方式：
    1. train_with_pipeline(pipeline, X, y): 传入已创建的 Pipeline
    2. train_with_pipeline(X, y): 自动创建简单 Pipeline
    """
    if y is None:
        # 调用方式: train_with_pipeline(X, y)
        X = pipeline_or_X
        y = X_or_y
        pipeline = create_classification_pipeline()
    else:
        # 调用方式: train_with_pipeline(pipeline, X, y)
        pipeline = pipeline_or_X
        X = X_or_y

    pipeline.fit(X, y)
    return pipeline


def cross_val_with_pipeline(pipeline_or_X, X_or_y=None, y=None, cv=5, scoring='roc_auc'):
    """使用 Pipeline 进行交叉验证（测试框架别名）

    支持两种调用方式：
    1. cross_val_with_pipeline(pipeline, X, y, cv, scoring): 传入已创建的 Pipeline
    2. cross_val_with_pipeline(X, y, cv, scoring): 自动创建简单 Pipeline
    """
    if y is None and X_or_y is not None:
        # 调用方式: cross_val_with_pipeline(X, y)
        X = pipeline_or_X
        y = X_or_y
        pipeline = create_classification_pipeline()
    else:
        # 调用方式: cross_val_with_pipeline(pipeline, X, y)
        pipeline = pipeline_or_X
        X = X_or_y

    return cross_val_score(pipeline, X, y, cv=cv, scoring=scoring)


def classification_with_pipeline(X, y, numeric_features=None, categorical_features=None,
                                 test_size=0.3, random_state=42):
    """完整的分类评估流水线，使用 Pipeline 防止数据泄漏（测试框架别名）

    支持两种调用方式：
    1. classification_with_pipeline(X, y): 自动推断所有特征为数值型
    2. classification_with_pipeline(X, y, numeric_features, categorical_features): 指定特征类型
    """
    from sklearn.model_selection import train_test_split

    # 自动推断特征类型
    if numeric_features is None and categorical_features is None:
        # 假设所有特征都是数值型
        if hasattr(X, 'shape'):
            n_features = X.shape[1]
        else:
            n_features = len(X.columns) if hasattr(X, 'columns') else len(X[0])
        numeric_features = list(range(n_features))
        categorical_features = []

    pipeline = create_classification_pipeline(numeric_features, categorical_features)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    results = {
        'pipeline': pipeline,
        'metrics': calculate_metrics(y_test, y_pred),
        'confusion_matrix': calculate_confusion_matrix(y_test, y_pred),
        'auc': calculate_auc(y_test, y_prob),
        'y_test': y_test,
        'y_pred': y_pred,
        'y_prob': y_prob
    }

    results['metrics']['auc'] = results['auc']

    # 交叉验证
    cv_scores = cross_val_with_pipeline(
        create_classification_pipeline(numeric_features, categorical_features),
        X, y, cv=5
    )
    results['cv_scores'] = {'mean': cv_scores.mean(), 'std': cv_scores.std()}

    return results

def format_classification_report(results):
    """格式化分类报告（测试框架别名）"""
    md = ["## 分类模型评估\n\n"]

    metrics = results['metrics']
    md.append("### 评估指标\n\n")
    md.append(f"| 指标 | 值 |\n")
    md.append(f"|------|-----|\n")
    md.append(f"| 准确率 | {metrics['accuracy']:.4f} |\n")
    md.append(f"| 精确率 | {metrics['precision']:.4f} |\n")
    md.append(f"| 召回率 | {metrics['recall']:.4f} |\n")
    md.append(f"| F1 分数 | {metrics['f1']:.4f} |\n")
    md.append(f"| AUC | {metrics['auc']:.4f} |\n\n")

    return "".join(md)

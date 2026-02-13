"""
Week 10 分类评估作业参考解答

本文件提供了作业的参考实现，供学生在遇到困难时查阅。
建议学生先自己尝试完成作业，遇到问题时再参考此解答。

作业要求：
1. 使用逻辑回归对二分类目标建模
2. 计算混淆矩阵、精确率、召回率、F1
3. 绘制 ROC 曲线并计算 AUC
4. 使用 Pipeline 防止数据泄漏
5. 与基线模型对比

运行方式：python3 chapters/week_10/starter_code/solution.py
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve
)
from sklearn.dummy import DummyClassifier

OUTPUT_DIR = Path(__file__).parent.parent.parent / "output" / "week_10"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 设置随机种子
np.random.seed(42)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_data():
    """
    1. 加载数据

    TODO: 替换为你的数据路径
    """
    # 示例：生成模拟数据
    # 实际使用时：df = pd.read_csv("data/clean_data.csv")
    n = 1000

    df = pd.DataFrame({
        'age': np.random.randint(18, 70, n),
        'income': np.random.lognormal(10, 0.5, n),
        'days_since_last_purchase': np.random.randint(1, 365, n),
        'gender': np.random.choice(['男', '女'], n),
        'membership_level': np.random.choice(['普通', '银卡', '金卡'], n, p=[0.6, 0.3, 0.1]),
    })

    # 目标变量：购买
    purchase_prob = (
        0.1 +
        0.2 * (df['income'] > df['income'].median()).astype(int) +
        0.3 * (df['membership_level'] == '金卡').astype(int) +
        0.15 * (df['membership_level'] == '银卡').astype(int)
    )
    df['purchase'] = np.random.binomial(1, np.clip(purchase_prob, 0, 1))

    return df


def prepare_data(df):
    """
    2. 准备数据

    TODO: 指定你的目标变量和特征
    """
    # 指定目标变量和特征
    target = "purchase"  # TODO: 替换为你的目标变量
    numeric_features = ["age", "income", "days_since_last_purchase"]  # TODO: 替换为你的数值特征
    categorical_features = ["gender", "membership_level"]  # TODO: 替换为你的类别特征

    X = df[numeric_features + categorical_features]
    y = df[target]

    # 划分训练集和测试集
    # TODO: 使用分层采样（stratify=y）保持类别比例
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test, numeric_features, categorical_features


def build_pipeline(numeric_features, categorical_features):
    """
    3. 构建 Pipeline

    TODO: 完成 Pipeline 的构建
    """
    # 数值特征预处理
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),  # TODO: 选择缺失值填充策略
        ('scaler', StandardScaler())  # TODO: 标准化
    ])

    # 类别特征预处理
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # TODO: 填充缺失值
        ('onehot', OneHotEncoder(handle_unknown='ignore'))  # TODO: One-Hot 编码
    ])

    # 组合预处理
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # 完整 Pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])

    return pipeline


def evaluate_model(pipeline, X_train, y_train, X_test, y_test):
    """
    4. 评估模型

    TODO: 计算混淆矩阵、精确率、召回率、F1、AUC
    """
    # 拟合模型
    pipeline.fit(X_train, y_train)

    # 预测
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # 计算混淆矩阵
    # TODO: 使用 confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # 计算指标
    # TODO: 计算精确率、召回率、F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # 计算 AUC
    # TODO: 使用 roc_auc_score
    auc = roc_auc_score(y_test, y_proba)

    # 打印结果
    print("\n" + "=" * 60)
    print("混淆矩阵:")
    print("=" * 60)
    print(f"  {'':>12} {'预测=0':>12} {'预测=1':>12}")
    print(f"  {'实际=0':>12} {tn:>12} {fp:>12}")
    print(f"  {'实际=1':>12} {fn:>12} {tp:>12}")

    print("\n" + "=" * 60)
    print("评估指标:")
    print("=" * 60)
    print(f"  准确率: {accuracy:.3f}")
    print(f"  精确率: {precision:.3f}")
    print(f"  召回率: {recall:.3f}")
    print(f"  F1 分数: {f1:.3f}")
    print(f"  AUC: {auc:.3f}")

    # 打印分类报告
    print("\n" + "=" * 60)
    print("分类报告 (classification_report):")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=['不购买', '购买']))

    return cm, y_proba, auc


def plot_roc_curve(y_test, y_proba, auc):
    """
    5. 绘制 ROC 曲线

    TODO: 使用 roc_curve 绘制 ROC 曲线
    """
    # 计算 ROC 曲线
    # TODO: 使用 roc_curve
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC 曲线 (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='随机猜测 (AUC = 0.5)')
    plt.xlabel('假阳性率 (FPR)', fontsize=12)
    plt.ylabel('真阳性率 (TPR / Recall)', fontsize=12)
    plt.title('ROC 曲线', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'roc_curve_solution.png', dpi=150, bbox_inches='tight')
    print("\n✅ ROC 曲线已保存为 roc_curve_solution.png")
    plt.close()


def compare_with_baseline(pipeline, X_train, y_train, X_test, y_test):
    """
    6. 与基线对比

    TODO: 使用 DummyClassifier 与多数类基线对比
    """
    # 训练基线模型
    # TODO: 使用 DummyClassifier(strategy='most_frequent')
    dummy = DummyClassifier(strategy='most_frequent', random_state=42)
    dummy.fit(X_train, y_train)

    # 预测
    y_pred_model = pipeline.predict(X_test)
    y_pred_dummy = dummy.predict(X_test)

    # 计算准确率和召回率
    acc_model = (y_pred_model == y_test).mean()
    acc_dummy = (y_pred_dummy == y_test).mean()

    recall_model = ((y_pred_model == 1) & (y_test == 1)).sum() / (y_test == 1).sum()
    recall_dummy = ((y_pred_dummy == 1) & (y_test == 1)).sum() / (y_test == 1).sum() if (y_test == 1).sum() > 0 else 0

    # 打印对比
    print("\n" + "=" * 60)
    print("与基线模型对比:")
    print("=" * 60)
    print(f"  {'指标':<15} {'基线':>15} {'逻辑回归':>15}")
    print("-" * 60)
    print(f"  {'准确率':<15} {acc_dummy:>15.3f} {acc_model:>15.3f}")
    print(f"  {'召回率':<15} {recall_dummy:>15.3f} {recall_model:>15.3f}")

    print(f"\n改进:")
    print(f"  准确率: {(acc_model - acc_dummy):.1%}")
    print(f"  召回率: {(recall_model - recall_dummy):.1%}")


def cross_validate_model(pipeline, X, y):
    """
    7. 交叉验证

    TODO: 使用 cross_val_score 进行 5-fold 交叉验证
    """
    # 5-fold 交叉验证
    # TODO: 使用 cross_val_score，scoring='accuracy'
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')

    print("\n" + "=" * 60)
    print("5-fold 交叉验证:")
    print("=" * 60)
    print(f"  准确率: {cv_scores}")
    print(f"  平均: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")


def main():
    """主函数：完整的分类评估流程"""
    print("=" * 60)
    print("Week 10 分类评估作业参考解答")
    print("=" * 60)

    # 1. 加载数据
    print("\n步骤 1: 加载数据...")
    df = load_data()
    print(f"  数据形状: {df.shape}")
    print(f"  目标变量比例: {df['purchase'].mean():.1%}")

    # 2. 准备数据
    print("\n步骤 2: 准备数据...")
    X_train, X_test, y_train, y_test, numeric_features, categorical_features = prepare_data(df)
    print(f"  训练集: {X_train.shape}")
    print(f"  测试集: {X_test.shape}")

    # 3. 构建 Pipeline
    print("\n步骤 3: 构建 Pipeline...")
    pipeline = build_pipeline(numeric_features, categorical_features)
    print(f"  Pipeline: ColumnTransformer -> LogisticRegression")

    # 4. 评估模型
    print("\n步骤 4: 评估模型...")
    cm, y_proba, auc = evaluate_model(pipeline, X_train, y_train, X_test, y_test)

    # 5. 绘制 ROC 曲线
    print("\n步骤 5: 绘制 ROC 曲线...")
    plot_roc_curve(y_test, y_proba, auc)

    # 6. 与基线对比
    print("\n步骤 6: 与基线对比...")
    compare_with_baseline(pipeline, X_train, y_train, X_test, y_test)

    # 7. 交叉验证
    print("\n步骤 7: 交叉验证...")
    X = df[numeric_features + categorical_features]
    y = df['purchase']
    cross_validate_model(pipeline, X, y)

    # 总结
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print("""
    本作业演示了完整的分类评估流程：

    1. 数据准备：划分训练集和测试集（分层采样）
    2. 模型构建：使用 Pipeline + ColumnTransformer 防止数据泄漏
    3. 模型评估：混淆矩阵、精确率、召回率、F1、AUC
    4. 可视化：绘制 ROC 曲线
    5. 基线对比：与多数类分类器对比
    6. 交叉验证：5-fold CV 估计泛化性能

    关键要点：
    - 准确率在类别不平衡时会误导
    - 需要关注精确率、召回率、F1
    - ROC-AUC 是阈值无关的评估指标
    - 使用 Pipeline 防止数据泄漏
    """)

    print("\n" + "=" * 60)
    print("✅ 完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
示例：完整的分类流水线——从预处理到评估

本例演示：
1. 使用 ColumnTransformer 处理混合数据类型
2. 构建 Pipeline（预处理 + 逻辑回归）
3. K-fold 分层交叉验证
4. 与基线模型对比

运行方式：python3 chapters/week_10/examples/05_complete_pipeline.py
预期输出：
- 交叉验证结果（准确率、F1、AUC）
- 混淆矩阵和分类报告
- 与基线模型对比
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, accuracy_score, f1_score
)
from sklearn.dummy import DummyClassifier

# 设置随机种子
np.random.seed(42)

# 设置中文字体
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def generate_mixed_type_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    生成混合数据类型（数值 + 类别）的客户流失数据

    参数:
        n_samples: 样本数量

    返回:
        包含数值和类别特征的 DataFrame
    """
    # 数值特征
    tenure_months = np.random.uniform(1, 72, n_samples)
    monthly_charges = np.random.uniform(20, 120, n_samples)
    total_charges = tenure_months * monthly_charges + np.random.normal(0, 50, n_samples)

    # 类别特征
    contract_type = np.random.choice(['月付', '一年', '两年'], n_samples, p=[0.5, 0.3, 0.2])
    payment_method = np.random.choice(['电子支票', '邮寄支票', '银行转账', '信用卡'], n_samples)
    internet_service = np.random.choice(['DSL', '光纤', '无'], n_samples, p=[0.3, 0.5, 0.2])

    # 生成目标变量（流失）
    # 合同期越短、月费越高，越容易流失
    prob_churn = (
        0.8 * (contract_type == '月付').astype(int) +
        0.3 * (contract_type == '一年').astype(int) +
        0.1 * (monthly_charges / 120) +
        0.05 * (tenure_months / 72)
    )
    prob_churn = np.clip(prob_churn, 0, 1)
    churn = np.random.binomial(1, prob_churn)

    # 创建 DataFrame
    df = pd.DataFrame({
        'tenure_months': tenure_months,
        'monthly_charges': monthly_charges,
        'total_charges': total_charges,
        'contract_type': contract_type,
        'payment_method': payment_method,
        'internet_service': internet_service,
        'churn': churn
    })

    # 添加一些缺失值（模拟真实数据）
    missing_indices = np.random.choice(df.index, size=int(n_samples * 0.05), replace=False)
    df.loc[missing_indices, 'total_charges'] = np.nan

    return df


def build_classification_pipeline(
    numeric_features: list,
    categorical_features: list
) -> Pipeline:
    """
    构建完整的分类 Pipeline

    参数:
        numeric_features: 数值特征列表
        categorical_features: 类别特征列表

    返回:
        sklearn Pipeline 对象
    """
    # 数值特征预处理：填充缺失值 + 标准化
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # 类别特征预处理：填充缺失值 + One-Hot 编码
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # ColumnTransformer：对不同列应用不同预处理
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'  # 丢弃未指定的列
    )

    # 完整 Pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])

    return pipeline


def evaluate_model(
    pipeline: Pipeline,
    X_train, y_train,
    X_test, y_test,
    model_name: str = "模型"
) -> dict:
    """
    评估模型性能

    返回:
        包含各种评估指标的字典
    """
    # 拟合
    pipeline.fit(X_train, y_train)

    # 预测
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # 计算指标
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    results = {
        'name': model_name,
        'confusion_matrix': cm,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_proba),
        'y_pred': y_pred,
        'y_proba': y_proba
    }

    return results


def cross_validate_pipeline(
    pipeline: Pipeline,
    X, y,
    n_folds: int = 5
) -> dict:
    """
    K-fold 分层交叉验证

    返回:
        包含交叉验证结果的字典
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    # 评估多个指标
    scoring = {
        'accuracy': 'accuracy',
        'f1': 'f1',
        'roc_auc': 'roc_auc',
        'recall': 'recall'
    }

    cv_results = cross_validate(
        pipeline, X, y,
        cv=skf,
        scoring=scoring,
        return_train_score=False
    )

    # 提取结果
    results = {}
    for metric in scoring.keys():
        scores = cv_results[f'test_{metric}']
        results[metric] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'values': scores
        }

    return results


def print_evaluation_summary(results: dict, cv_results: dict) -> None:
    """打印评估摘要"""
    print("\n" + "=" * 60)
    print(f"{results['name']}：评估结果")
    print("=" * 60)

    # 测试集指标
    print(f"\n【测试集性能】")
    print(f"  准确率: {results['accuracy']:.3f}")
    print(f"  精确率: {results['precision']:.3f}")
    print(f"  召回率: {results['recall']:.3f}")
    print(f"  F1 分数: {results['f1']:.3f}")
    print(f"  AUC: {results['auc']:.3f}")

    # 混淆矩阵
    cm = results['confusion_matrix']
    print(f"\n【混淆矩阵】")
    print(f"  {'':>12} {'预测不流失':>12} {'预测流失':>12}")
    print(f"  {'实际不流失':>12} {cm[0, 0]:>12} {cm[0, 1]:>12}")
    print(f"  {'实际流失':>12} {cm[1, 0]:>12} {cm[1, 1]:>12}")

    # 交叉验证结果
    print(f"\n【5-fold 交叉验证】")
    for metric, values in cv_results.items():
        print(f"  {metric:>10}: {values['mean']:.3f} ± {values['std']:.3f}")


def main() -> None:
    """主函数"""
    print("=" * 60)
    print("示例5: 完整的分类流水线")
    print("=" * 60)

    # 1. 生成数据
    df = generate_mixed_type_data(n_samples=1000)

    print(f"\n📊 数据概览:")
    print(df.head(10))
    print(f"\n数据类型:")
    print(df.dtypes)
    print(f"\n缺失值:")
    print(df.isnull().sum())
    print(f"\n流失率: {df['churn'].mean():.1%}")

    # 2. 准备数据
    numeric_features = ['tenure_months', 'monthly_charges', 'total_charges']
    categorical_features = ['contract_type', 'payment_method', 'internet_service']

    X = df[numeric_features + categorical_features]
    y = df['churn']

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"\n✅ 数据划分:")
    print(f"  训练集: {len(X_train)} 样本 (流失率 {y_train.mean():.1%})")
    print(f"  测试集: {len(X_test)} 样本 (流失率 {y_test.mean():.1%})")

    # 3. 构建 Pipeline
    print(f"\n✅ 构建 Pipeline...")
    pipeline = build_classification_pipeline(numeric_features, categorical_features)
    print(f"  Pipeline 结构:")
    print(f"    1. ColumnTransformer (预处理)")
    print(f"       - 数值特征: SimpleImputer(median) + StandardScaler")
    print(f"       - 类别特征: SimpleImputer(most_frequent) + OneHotEncoder")
    print(f"    2. LogisticRegression(random_state=42)")

    # 4. 交叉验证
    print(f"\n✅ 运行 5-fold 交叉验证...")
    cv_results = cross_validate_pipeline(pipeline, X, y, n_folds=5)

    # 5. 测试集评估
    print(f"\n✅ 在测试集上评估...")
    results = evaluate_model(
        pipeline, X_train, y_train, X_test, y_test,
        model_name="逻辑回归"
    )

    # 6. 打印结果
    print_evaluation_summary(results, cv_results)

    # 7. 与基线对比
    print("\n" + "=" * 60)
    print("与基线模型对比")
    print("=" * 60)

    # 基线：多数类分类器
    dummy = DummyClassifier(strategy='most_frequent', random_state=42)
    dummy_results = evaluate_model(
        dummy, X_train, y_train, X_test, y_test,
        model_name="多数类基线"
    )

    print(f"\n{'指标':<15} {'基线模型':>15} {'逻辑回归':>15} {'改进':>15}")
    print("-" * 60)
    print(f"{'准确率':<15} {dummy_results['accuracy']:>15.3f} {results['accuracy']:>15.3f} {(results['accuracy'] - dummy_results['accuracy']):>+15.1%}")
    print(f"{'召回率':<15} {dummy_results['recall']:>15.3f} {results['recall']:>15.3f} {(results['recall'] - dummy_results['recall']):>+15.1%}")
    print(f"{'F1 分数':<15} {dummy_results['f1']:>15.3f} {results['f1']:>15.3f} {(results['f1'] - dummy_results['f1']):>+15.1%}")
    print(f"{'AUC':<15} {dummy_results['auc']:>15.3f} {results['auc']:>15.3f} {(results['auc'] - dummy_results['auc']):>+15.1%}")

    # 8. 查看系数（可解释性）
    print("\n" + "=" * 60)
    print("模型可解释性：特征重要性")
    print("=" * 60)

    # 获取特征名（One-Hot 编码后）
    feature_names = numeric_features + list(
        pipeline.named_steps['preprocessor']
        .named_transformers_['cat']
        .named_steps['onehot']
        .get_feature_names_out(categorical_features)
    )

    # 获取系数
    coefs = pipeline.named_steps['classifier'].coef_[0]

    # 创建系数表
    coef_df = pd.DataFrame({
        '特征': feature_names,
        '系数': coefs,
        '|系数|': np.abs(coefs)
    }).sort_values('|系数|', ascending=False)

    print(f"\n前 10 个最重要的特征:")
    print(coef_df.head(10).to_string(index=False))

    # 9. 总结
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print(f"""
完整的分类流水线包含：

1. 数据预处理：
   - 数值特征：填充缺失值（中位数）+ 标准化
   - 类别特征：填充缺失值（众数）+ One-Hot 编码

2. 模型训练：
   - 逻辑回归（可解释性强）
   - 正则化防止过拟合

3. 评估方法：
   - K-fold 分层交叉验证（稳健估计）
   - 多指标评估（准确率、F1、AUC）
   - 与基线对比（证明模型价值）

4. 工程实践：
   - Pipeline 防止数据泄漏
   - ColumnTransformer 处理混合类型
   - 可复现的训练流程

关键结果：
- 测试集 AUC = {results['auc']:.3f}（{ '强' if results['auc'] > 0.8 else '中等' if results['auc'] > 0.7 else '弱'}区分能力）
- 交叉验证标准差较小（模型稳定）
- 召回率显著优于基线（{results['recall']:.1%} vs {dummy_results['recall']:.1%}）
    """)

    print("\n" + "=" * 60)
    print("✅ 示例5完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()

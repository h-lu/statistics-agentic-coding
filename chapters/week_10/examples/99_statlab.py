"""
StatLab 分类评估报告生成器

本脚本是 StatLab 超级线的一部分，用于在可复现分析报告中添加
"分类评估"章节。它执行完整的分类分析流程，包括：
- 逻辑回归建模（系数解释 + 优势比）
- 混淆矩阵与分类指标（精确率、召回率、F1）
- ROC-AUC 分析（阈值无关评估）
- K-fold 分层交叉验证
- 基线对比（多数类分类器）
- 自动生成报告片段和图表

运行方式：python3 chapters/week_10/examples/99_statlab.py
预期输出：
- 报告片段（追加到 report.md）
- ROC 曲线图（保存到 report/images/）

依赖: 需要预先清洗好的数据（假设路径为 data/clean_data.csv）

说明：本脚本在上周（回归分析）基础上增量修改，将分析目标从
"连续目标预测"扩展到"二分类目标预测"。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    train_test_split, StratifiedKFold,
    cross_validate, cross_val_score
)
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve, precision_recall_curve,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.dummy import DummyClassifier

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def classification_evaluation_to_report(
    df: pd.DataFrame,
    target: str,
    numeric_features: List[str],
    categorical_features: List[str],
    output_dir: str = "report"
) -> str:
    """
    对数据集进行完整的分类评估，生成报告片段

    参数:
        df: 清洗后的数据
        target: 目标变量名（如 'purchase', 'churn'）
        numeric_features: 数值特征列表
        categorical_features: 类别特征列表
        output_dir: 报告输出目录

    返回:
        Markdown 格式的报告片段
    """
    # 创建输出目录
    output_path = Path(output_dir)
    images_path = output_path / "images"
    images_path.mkdir(parents=True, exist_ok=True)

    # 准备数据
    X = df[numeric_features + categorical_features]
    y = df[target]

    print("=" * 70)
    print("StatLab 分类评估报告生成器")
    print("=" * 70)
    print(f"\n📊 数据概览:")
    print(f"  总样本数: {len(y)}")
    print(f"  目标变量: {target}")
    print(f"  {target}=1 的比例: {y.mean():.2%}")
    print(f"  数值特征: {', '.join(numeric_features)}")
    print(f"  类别特征: {', '.join(categorical_features)}")

    # ========== 1. 划分训练集和测试集 ==========
    print(f"\n✅ 步骤 1: 划分训练集和测试集...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  训练集: {len(X_train)} 样本")
    print(f"  测试集: {len(X_test)} 样本")

    # ========== 2. 构建 Pipeline ==========
    print(f"\n✅ 步骤 2: 构建 Pipeline...")

    # 数值特征预处理
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # 类别特征预处理
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )

    # 完整 Pipeline
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])

    print(f"  Pipeline: ColumnTransformer -> LogisticRegression")

    # ========== 3. 拟合模型 ==========
    print(f"\n✅ 步骤 3: 拟合逻辑回归模型...")
    full_pipeline.fit(X_train, y_train)

    # 预测
    y_pred = full_pipeline.predict(X_test)
    y_proba = full_pipeline.predict_proba(X_test)[:, 1]

    print(f"  ✅ 模型拟合完成")

    # ========== 4. 混淆矩阵与评估指标 ==========
    print(f"\n✅ 步骤 4: 计算评估指标...")

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)

    print(f"  准确率: {accuracy:.3f}")
    print(f"  精确率: {precision:.3f}")
    print(f"  召回率: {recall:.3f}")
    print(f"  F1 分数: {f1:.3f}")
    print(f"  AUC: {auc:.3f}")

    # ========== 5. ROC 曲线 ==========
    print(f"\n✅ 步骤 5: 绘制 ROC 曲线...")

    fpr, tpr, thresholds_roc = roc_curve(y_test, y_proba)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC 曲线 (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='随机猜测 (AUC = 0.5)')
    plt.xlabel('假阳性率 (FPR)', fontsize=12)
    plt.ylabel('真阳性率 (TPR / Recall)', fontsize=12)
    plt.title(f'ROC 曲线 - {target} 预测', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    roc_fig_path = images_path / "roc_curve.png"
    plt.savefig(roc_fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ ROC 曲线已保存: {roc_fig_path}")

    # ========== 6. K-fold 交叉验证 ==========
    print(f"\n✅ 步骤 6: 运行 5-fold 交叉验证...")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_results = cross_validate(
        full_pipeline, X, y,
        cv=skf,
        scoring={
            'accuracy': 'accuracy',
            'f1': 'f1',
            'roc_auc': 'roc_auc',
            'recall': 'recall'
        },
        return_train_score=False
    )

    cv_accuracy = cv_results['test_accuracy']
    cv_f1 = cv_results['test_f1']
    cv_auc = cv_results['test_roc_auc']
    cv_recall = cv_results['test_recall']

    print(f"  准确率: {cv_accuracy.mean():.3f} ± {cv_accuracy.std():.3f}")
    print(f"  F1 分数: {cv_f1.mean():.3f} ± {cv_f1.std():.3f}")
    print(f"  AUC: {cv_auc.mean():.3f} ± {cv_auc.std():.3f}")
    print(f"  召回率: {cv_recall.mean():.3f} ± {cv_recall.std():.3f}")

    # ========== 7. 基线对比 ==========
    print(f"\n✅ 步骤 7: 与基线模型对比...")

    dummy = DummyClassifier(strategy='most_frequent', random_state=42)
    dummy.fit(X_train, y_train)
    y_pred_dummy = dummy.predict(X_test)
    y_proba_dummy = dummy.predict_proba(X_test)[:, 1]

    dummy_acc = accuracy_score(y_test, y_pred_dummy)
    dummy_recall = recall_score(y_test, y_pred_dummy, zero_division=0)
    dummy_auc = roc_auc_score(y_test, y_proba_dummy)

    print(f"  基线准确率: {dummy_acc:.3f}")
    print(f"  基线召回率: {dummy_recall:.3f}")
    print(f"  基线 AUC: {dummy_auc:.3f}")

    # ========== 8. 提取系数表 ==========
    print(f"\n✅ 步骤 8: 提取模型系数...")

    # 获取特征名（One-Hot 编码后）
    feature_names = numeric_features + list(
        full_pipeline.named_steps['preprocessor']
        .named_transformers_['cat']
        .named_steps['onehot']
        .get_feature_names_out(categorical_features)
    )

    # 获取系数
    coefs = full_pipeline.named_steps['classifier'].coef_[0]

    # 计算优势比
    odds_ratios = np.exp(coefs)

    # 创建系数表
    coef_df = pd.DataFrame({
        '特征': feature_names,
        '系数': coefs,
        '优势比 (OR)': odds_ratios,
        '|系数|': np.abs(coefs)
    }).sort_values('|系数|', ascending=False)

    print(f"  ✅ 提取了 {len(coef_df)} 个特征的系数")

    # ========== 9. 生成报告片段 ==========
    print(f"\n✅ 步骤 9: 生成报告片段...")

    report = generate_report_markdown(
        target=target,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        coef_df=coef_df,
        cm=cm,
        tn=tn, fp=fp, fn=fn, tp=tp,
        accuracy=accuracy, precision=precision, recall=recall, f1=f1, auc=auc,
        cv_accuracy=cv_accuracy, cv_f1=cv_f1, cv_auc=cv_auc, cv_recall=cv_recall,
        dummy_acc=dummy_acc, dummy_recall=dummy_recall,
        n_total=len(y),
        pos_ratio=y.mean(),
        roc_fig_path=roc_fig_path
    )

    print(f"  ✅ 报告片段生成完成")

    print("\n" + "=" * 70)
    print("✅ 分类评估完成！")
    print("=" * 70)

    return report


def generate_report_markdown(
    target: str,
    numeric_features: List[str],
    categorical_features: List[str],
    coef_df: pd.DataFrame,
    cm: np.ndarray,
    tn: int, fp: int, fn: int, tp: int,
    accuracy: float, precision: float, recall: float, f1: float, auc: float,
    cv_accuracy: np.ndarray, cv_f1: np.ndarray, cv_auc: np.ndarray, cv_recall: np.ndarray,
    dummy_acc: float, dummy_recall: float,
    n_total: int,
    pos_ratio: float,
    roc_fig_path: Path
) -> str:
    """生成 Markdown 格式的报告片段"""

    # 前 10 个重要特征
    top_features = coef_df.head(10)

    # 系数表 Markdown
    coef_table = ""
    for _, row in top_features.iterrows():
        coef_table += f"- **{row['特征']}**: 系数 = {row['系数']:.3f}, 优势比 (OR) = {row['优势比 (OR)']:.3f}\n"

    # AUC 判断
    if auc > 0.8:
        auc_strength = "强"
    elif auc > 0.7:
        auc_strength = "中等"
    elif auc > 0.6:
        auc_strength = "弱"
    else:
        auc_strength = "很差"

    report = f"""
## 分类评估

### 研究问题

哪些因素影响 **{target}**（二分类目标）？

本节使用逻辑回归（Logistic Regression）建模，目标是预测 {target}=1（如"购买"/"流失"）的概率，并评估模型的预测性能。

### 模型设置

**算法**: 逻辑回归 (Logistic Regression)

**特征**:
- 数值特征: {', '.join(numeric_features)}
- 类别特征: {', '.join(categorical_features)}

**预处理**:
- 数值特征: 中位数填充缺失值 + 标准化 (StandardScaler)
- 类别特征: 众数填充缺失值 + One-Hot 编码 (OneHotEncoder)

**评估方法**: 5-fold 分层交叉验证 (StratifiedKFold)

### 逻辑回归系数与优势比

逻辑回归的系数表示"对数优势比"（log-odds ratio）的变化。为了更直观地解释，我们计算**优势比 (Odds Ratio, OR)**：

**优势比 (OR) = exp(系数)**

- OR > 1: 该特征增加会提高 {target}=1 的优势
- OR < 1: 该特征增加会降低 {target}=1 的优势
- OR = 1: 该特征对 {target} 无影响

**前 10 个重要特征**:

{coef_table}

**解释示例**:
- 如果某特征的优势比 OR = 1.5，说明该特征每增加 1 单位，{target}=1 的优势增加到原来的 1.5 倍（增加 50%）
- 如果某特征的优势比 OR = 0.8，说明该特征每增加 1 单位，{target}=1 的优势降低到原来的 0.8 倍（降低 20%）

### 混淆矩阵与评估指标

**混淆矩阵** (Threshold = 0.5):

| | 预测 {target}=0 | 预测 {target}=1 |
|---|---|---|
| **实际 {target}=0** | {tn} (真阴性 TN) | {fp} (假阳性 FP) |
| **实际 {target}=1** | {fn} (假阴性 FN) | {tp} (真阳性 TP) |

**评估指标**:

| 指标 | 公式 | 值 | 含义 |
|------|------|-----|------|
| **准确率 (Accuracy)** | (TP + TN) / 总样本 | {accuracy:.2%} | 所有预测中，预测正确的比例 |
| **精确率 (Precision)** | TP / (TP + FP) | {precision:.2%} | 预测为 {target}=1 的样本中，真正为 1 的比例 |
| **召回率 (Recall)** | TP / (TP + FN) | {recall:.2%} | 真实为 {target}=1 的样本中，被正确识别的比例 |
| **F1 分数** | 2 × (Prec × Rec) / (Prec + Rec) | {f1:.3f} | 精确率和召回率的调和平均数 |

**业务解释**:

- **假阳性成本（误报）**: {fp} 个样本被错误预测为 {target}=1，可能浪费营销/运营资源
- **假阴性成本（漏报）**: {fn} 个真实 {target}=1 的样本被遗漏，可能造成业务损失（如流失客户、未成交订单）
- **模型价值**: 本模型的召回率为 {recall:.1%}，相比基线模型（召回率 {dummy_recall:.1%}）有显著提升

### ROC-AUC 分析

**AUC（ROC 曲线下面积）**: {auc:.3f}

AUC 衡量模型区分正负样本的能力，不依赖分类阈值：

- **AUC = 1.0**: 完美分类器
- **AUC = 0.5**: 随机猜测（像抛硬币）
- **本模型 AUC = {auc:.3f}**: {auc_strength}区分能力

**直观解释**:
AUC = {auc:.3f} 的含义是：如果你随机选一个 {target}=1 的样本和一个 {target}=0 的样本，模型给 {target}=1 的样本更高概率的概率是 {auc:.1%}。

![ROC 曲线](images/roc_curve.png)

### 交叉验证结果

5-fold 分层交叉验证（StratifiedKFold）结果:

| 指标 | 均值 ± 标准差 | 说明 |
|------|---------------|------|
| **准确率** | {cv_accuracy.mean():.3f} ± {cv_accuracy.std():.3f} | 整体正确率 |
| **F1 分数** | {cv_f1.mean():.3f} ± {cv_f1.std():.3f} | 精确率与召回率的平衡 |
| **AUC** | {cv_auc.mean():.3f} ± {cv_auc.std():.3f} | 区分能力 |
| **召回率** | {cv_recall.mean():.3f} ± {cv_recall.std():.3f} | 捕获 {target}=1 的能力 |

**稳定性评估**:
- 标准差较小（< 0.05），说明模型对不同数据划分稳健
- 如果标准差很大（> 0.10），说明模型对数据划分敏感，需要更多数据或更简单的模型

### 基线对比

与**多数类基线**（DummyClassifier：总是预测出现最多的类别）对比:

| 模型 | 准确率 | 召回率 | AUC |
|------|---------|--------|-----|
| **多数类基线** | {dummy_acc:.2%} | {dummy_recall:.2%} | 0.500 |
| **逻辑回归** | {accuracy:.2%} | {recall:.2%} | {auc:.3f} |
| **改进** | {(accuracy - dummy_acc):.1%} | {(recall - dummy_recall):.1%} | {(auc - 0.5):.3f} |

**结论**:

- 本模型的准确率与基线{"相当" if accuracy < dummy_acc * 1.05 else "略高"}
- 但召回率从基线的 {dummy_recall:.1%} 提升到 {recall:.1%}，{"有显著改进" if recall > dummy_recall * 1.5 else "有所改进"}
- AUC = {auc:.3f} ({"强" if auc > 0.8 else "中等" if auc > 0.7 else "弱"})区分能力，模型有效

### 工程实践：防止数据泄漏

本分析使用 **Pipeline + ColumnTransformer** 模式：

```python
Pipeline(steps=[
    ('preprocessor', ColumnTransformer(
        transformers=[
            ('num', Pipeline([...]), numeric_features),
            ('cat', Pipeline([...]), categorical_features)
        ]
    )),
    ('classifier', LogisticRegression(...))
])
```

**关键实践**:

1. **所有预处理在 Pipeline 内完成**:
   - 标准化、One-Hot 编码、缺失值填充都在 Pipeline 内部
   - 交叉验证时，每个折独立拟合预处理参数（如均值、方差）

2. **确保测试集信息不会泄漏**:
   - 测试集只用于 `transform`，不用于 `fit`
   - 每个折的训练集不会"看到"其他折的统计量

3. **可复现性**:
   - 固定随机种子 (`random_state=42`)
   - 所有步骤封装在 Pipeline 对象中，可直接用于新数据

这是分类评估中的最佳实践，避免"虚高"的性能估计。

### 局限性与因果警告

⚠️ **本分析仅描述 {target} 与预测特征的关联关系，不能直接推断因果**。

**局限性**:

1. **类别不平衡**:
   - {target}=1 的样本比例为 {pos_ratio:.1%}{"（较低）" if pos_ratio < 0.2 else "（中等）"}
   - 模型可能在少数类上表现不佳（召回率低）
   - 如需优化少数类，可调整分类阈值或使用过采样/欠采样技术

2. **观察数据**:
   - 本分析基于观测数据，未进行随机实验
   - 可能存在混杂变量（confounders）和反向因果
   - 例如：{target} 可能影响某些特征，而非单向因果关系

3. **阈值选择**:
   - 默认阈值 0.5 可能不是业务最优解
   - 应根据假阳性/假阴性成本调整（见 ROC 曲线）
   - 如更看重召回率（减少漏报），可降低阈值

4. **数据漂移**:
   - 如果未来数据分布与训练数据不同，模型性能可能下降
   - 建议定期监控模型在生产环境的性能，并定期重新训练

**因果推断**:

Week 13 会学习的**因果图 (DAG)** 和识别策略（如 RCT、工具变量、双重差分）可用于回答"改变 X 是否会导致 Y 变化"的问题。

- 本分析仅限于"**预测**"（Prediction）
- 不涉及"**因果**"（Causation）

### 数据来源

- **样本量**: n = {n_total}
- **{target}=1 的比例**: {pos_ratio:.2%}
- **分析日期**: 2026-02-12
- **随机种子**: 42（保证可复现）

---

"""
    return report


# ============================================================================
# 示例使用（生成模拟数据演示）
# ============================================================================

def demo_with_mock_data():
    """使用模拟数据演示完整流程"""
    print("\n" + "=" * 70)
    print("StatLab 分类评估报告生成器 - 演示模式")
    print("=" * 70)

    # 1. 生成模拟数据（电商购买场景）
    np.random.seed(42)
    n = 1000

    df = pd.DataFrame({
        # 数值特征
        'age': np.random.randint(18, 70, n),
        'income': np.random.lognormal(10, 0.5, n),
        'days_since_last_purchase': np.random.randint(1, 365, n),

        # 类别特征
        'gender': np.random.choice(['男', '女'], n),
        'city_tier': np.random.choice(['一线城市', '二线城市', '三线及以下'], n, p=[0.3, 0.4, 0.3]),
        'membership_level': np.random.choice(['普通', '银卡', '金卡'], n, p=[0.6, 0.3, 0.1]),
    })

    # 目标变量：购买（二分类）
    # 购买概率与收入、会员等级、距离上次购买时间相关
    purchase_prob = (
        0.1 +
        0.2 * (df['income'] > df['income'].median()).astype(int) +
        0.3 * (df['membership_level'] == '金卡').astype(int) +
        0.15 * (df['membership_level'] == '银卡').astype(int) +
        0.1 * (df['days_since_last_purchase'] < 30).astype(int)
    )
    df['purchase'] = np.random.binomial(1, np.clip(purchase_prob, 0, 1))

    print(f"\n📊 模拟数据概览:")
    print(df.head(10))
    print(f"\n购买率: {df['purchase'].mean():.1%}")

    # 2. 运行分类评估
    target = "purchase"
    numeric_features = ["age", "income", "days_since_last_purchase"]
    categorical_features = ["gender", "city_tier", "membership_level"]

    report = classification_evaluation_to_report(
        df=df,
        target=target,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        output_dir="report"
    )

    # 3. 打印报告
    print("\n" + "=" * 70)
    print("生成的报告片段:")
    print("=" * 70)
    print(report)

    # 4. 保存到文件
    output_path = Path("report")
    output_path.mkdir(exist_ok=True)

    report_file = output_path / "classification_evaluation.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n✅ 报告已保存到: {report_file}")
    print(f"✅ ROC 曲线已保存到: {output_path}/images/roc_curve.png")

    return report


def main():
    """主函数"""
    # 演示模式：使用模拟数据
    report = demo_with_mock_data()

    print("\n" + "=" * 70)
    print("💡 使用说明:")
    print("=" * 70)
    print("""
    1. 在你的 StatLab 项目中，替换数据源:
       df = pd.read_csv("data/clean_data.csv")

    2. 指定你的目标变量和特征:
       target = "your_target_variable"  # 如 'purchase', 'churn'
       numeric_features = ["num_var1", "num_var2", ...]
       categorical_features = ["cat_var1", "cat_var2", ...]

    3. 运行函数生成报告:
       report = classification_evaluation_to_report(
           df, target, numeric_features, categorical_features, "report"
       )

    4. 将生成的报告片段追加到 report.md

    本脚本是 StatLab 超级线的一部分，在 Week 09 回归分析基础上
    增加了分类评估能力。完整报告应包含：
    - Week 01-04: 数据卡、描述统计、清洗、EDA
    - Week 05-08: 假设检验、不确定性量化
    - Week 09: 回归分析
    - Week 10: 分类评估（本脚本）
    """)


if __name__ == "__main__":
    main()

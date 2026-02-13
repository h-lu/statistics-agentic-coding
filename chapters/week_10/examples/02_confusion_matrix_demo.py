"""
示例：从准确率到混淆矩阵——类别不平衡的陷阱

本例演示：
1. 准确率在类别不平衡场景下的误导性
2. 混淆矩阵的四个组成：TP, TN, FP, FN
3. 精确率、召回率、F1 的计算与业务含义

运行方式：python3 chapters/week_10/examples/02_confusion_matrix_demo.py
预期输出：
- 混淆矩阵可视化（保存为 confusion_matrix.png）
- 控制台输出准确率、精确率、召回率、F1 等指标
- 与基线模型的对比
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_score, recall_score, f1_score, accuracy_score
)
from sklearn.dummy import DummyClassifier

OUTPUT_DIR = Path(__file__).parent.parent.parent / "output" / "week_10"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 设置随机种子
np.random.seed(42)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def generate_imbalanced_data(n_samples: int = 1000, imbalance_ratio: float = 0.15) -> pd.DataFrame:
    """
    生成类别不平衡的二分类数据

    参数:
        n_samples: 总样本数
        imbalance_ratio: 少数类的比例

    返回:
        包含特征和标签的 DataFrame
    """
    # 生成特征
    n_minority = int(n_samples * imbalance_ratio)
    n_majority = n_samples - n_minority

    # 多数类（不流失）
    X_majority = np.random.randn(n_majority, 2) + np.array([2, 2])
    y_majority = np.zeros(n_majority)

    # 少数类（流失）
    X_minority = np.random.randn(n_minority, 2) + np.array([-1, -1])
    y_minority = np.ones(n_minority)

    # 合并
    X = np.vstack([X_majority, X_minority])
    y = np.hstack([y_majority, y_minority])

    # 打乱
    indices = np.random.permutation(len(y))
    X = X[indices]
    y = y[indices]

    return pd.DataFrame({
        'feature_1': X[:, 0],
        'feature_2': X[:, 1],
        'churn': y
    })


def plot_confusion_matrix(cm: np.ndarray, class_names: list, title: str = "混淆矩阵") -> None:
    """画混淆矩阵热力图"""
    plt.figure(figsize=(8, 6))

    # 计算百分比
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # 创建标签
    labels = np.array([
        [f"{cm[i, j]}\n({cm_percent[i, j]:.1f}%)"
         for j in range(cm.shape[1])]
        for i in range(cm.shape[0])
    ])

    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': '样本数量'})

    plt.xlabel('预测标签', fontsize=12)
    plt.ylabel('真实标签', fontsize=12)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
    print(f"✅ {title}已保存为 confusion_matrix.png")
    plt.close()


def explain_confusion_matrix_metrics(tn: int, fp: int, fn: int, tp: int) -> None:
    """
    解释混淆矩阵指标的业务含义

    参数:
        tn, fp, fn, tp: 混淆矩阵的四个元素
    """
    print("\n" + "=" * 60)
    print("混淆矩阵指标详解")
    print("=" * 60)

    # 基本统计
    total = tn + fp + fn + tp
    actual_positive = tp + fn
    actual_negative = tn + fp
    predicted_positive = tp + fp
    predicted_negative = tn + fn

    print(f"\n混淆矩阵：")
    print(f"{'':>12} {'预测不流失':>12} {'预测流失':>12}")
    print(f"{'实际不流失':>12} {tn:>12} {fp:>12}")
    print(f"{'实际流失':>12} {fn:>12} {tp:>12}")

    print(f"\n基本统计：")
    print(f"  总样本数: {total}")
    print(f"  实际流失: {actual_positive} ({actual_positive/total*100:.1f}%)")
    print(f"  实际不流失: {actual_negative} ({actual_negative/total*100:.1f}%)")

    # 准确率
    accuracy = (tp + tn) / total
    print(f"\n{'='*60}")
    print("1. 准确率 (Accuracy)")
    print("=" * 60)
    print(f"  公式: (TP + TN) / (TP + TN + FP + FN)")
    print(f"  计算: ({tp} + {tn}) / {total} = {accuracy:.2%}")
    print(f"  含义: 所有预测中，预测正确的比例")
    print(f"  ⚠️  在类别不平衡时会误导！")

    # 精确率
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    print(f"\n{'='*60}")
    print("2. 精确率 (Precision) - 查准率")
    print("=" * 60)
    print(f"  公式: TP / (TP + FP)")
    print(f"  计算: {tp} / ({tp} + {fp}) = {precision:.2%}")
    print(f"  含义: 在所有预测为'流失'的样本中，真正流失的比例")
    print(f"  业务价值: 避免误报，减少营销成本浪费")
    print(f"  场景: 给'可能流失'客户发优惠券，精确率低意味着")
    print(f"        很多优惠券发给了本来不会流失的客户")

    # 召回率
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    print(f"\n{'='*60}")
    print("3. 召回率 (Recall) - 查全率 / 灵敏度 / TPR")
    print("=" * 60)
    print(f"  公式: TP / (TP + FN)")
    print(f"  计算: {tp} / ({tp} + {fn}) = {recall:.2%}")
    print(f"  含义: 在所有真实'流失'的样本中，被正确识别的比例")
    print(f"  业务价值: 减少漏报，抓住更多流失客户")
    print(f"  场景: 客户流失预警，召回率低意味着")
    print(f"        大量真实流失客户被遗漏，损失客户终身价值")

    # F1 分数
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    print(f"\n{'='*60}")
    print("4. F1 分数 (F1-Score)")
    print("=" * 60)
    print(f"  公式: 2 × (Precision × Recall) / (Precision + Recall)")
    print(f"  计算: 2 × {precision:.3f} × {recall:.3f} / ({precision:.3f} + {recall:.3f}) = {f1:.3f}")
    print(f"  含义: 精确率和召回率的调和平均数")
    print(f"  为什么用调和平均？惩罚极端情况")
    print(f"    例如: 精确率=1.0, 召回率=0.01")
    print(f"         算术平均=0.505 (看起来还行)")
    print(f"         调和平均≈0.02 (揭示模型几乎无用)")

    # 特异性和假阳性率
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
    print(f"\n{'='*60}")
    print("5. 其他指标")
    print("=" * 60)
    print(f"  假阳性率 (FPR): FP / (FP + TN) = {fp} / ({fp} + {tn}) = {fpr:.2%}")
    print(f"  真阴性率 (TNR/特异性): TN / (TN + FP) = {tn} / ({tn} + {fp}) = {tnr:.2%}")

    # 业务成本
    print(f"\n{'='*60}")
    print("6. 业务成本视角")
    print("=" * 60)
    cost_fp = 100  # 误报成本（元）
    cost_fn = 500  # 漏报成本（元）

    total_cost = fp * cost_fp + fn * cost_fn
    print(f"  假阳性成本（误报）: {fp} 个 × ¥{cost_fp} = ¥{fp * cost_fp:,}")
    print(f"  假阴性成本（漏报）: {fn} 个 × ¥{cost_fn} = ¥{fn * cost_fn:,}")
    print(f"  总成本: ¥{total_cost:,}")

    print(f"\n  💡 如果优化精确率（减少 FP），可以节省 ¥{fp * cost_fp:,}")
    print(f"  💡 如果优化召回率（减少 FN），可以节省 ¥{fn * cost_fn:,}")


def demonstrate_accuracy_paradox() -> None:
    """演示准确率悖论"""
    print("\n" + "=" * 60)
    print("准确率悖论演示")
    print("=" * 60)

    # 场景1：平衡数据
    print("\n【场景1：平衡数据】")
    tn1, fp1, fn1, tp1 = 80, 10, 10, 80
    acc1 = (tp1 + tn1) / (tp1 + tn1 + fp1 + fn1)
    recall1 = tp1 / (tp1 + fn1)

    print(f"  混淆矩阵: TN={tn1}, FP={fp1}, FN={fn1}, TP={tp1}")
    print(f"  准确率: {acc1:.2%}")
    print(f"  召回率: {recall1:.2%}")
    print(f"  评估: 模型表现良好")

    # 场景2：不平衡数据（傻瓜模型）
    print("\n【场景2：不平衡数据 - 总是预测多数类】")
    tn2, fp2, fn2, tp2 = 150, 0, 30, 0
    acc2 = (tp2 + tn2) / (tp2 + tn2 + fp2 + fn2)
    recall2 = tp2 / (tp2 + fn2) if (tp2 + fn2) > 0 else 0

    print(f"  混淆矩阵: TN={tn2}, FP={fp2}, FN={fn2}, TP={tp2}")
    print(f"  准确率: {acc2:.2%}")
    print(f"  召回率: {recall2:.2%}")
    print(f"  评估: 模型毫无价值！准确率高但召回率为0")

    # 场景3：不平衡数据（真实模型）
    print("\n【场景3：不平衡数据 - 真实模型】")
    tn3, fp3, fn3, tp3 = 140, 10, 15, 15
    acc3 = (tp3 + tn3) / (tp3 + tn3 + fp3 + fn3)
    recall3 = tp3 / (tp3 + fn3) if (tp3 + fn3) > 0 else 0

    print(f"  混淆矩阵: TN={tn3}, FP={fp3}, FN={fn3}, TP={tp3}")
    print(f"  准确率: {acc3:.2%}")
    print(f"  召回率: {recall3:.2%}")
    print(f"  评估: 模型有价值！准确率略低但召回率50%")

    print("\n" + "=" * 60)
    print("结论：")
    print("=" * 60)
    print(f"  场景2 的准确率（{acc2:.1%}）高于 场景3（{acc3:.1%}）")
    print(f"  但场景2的召回率为 0%，完全漏掉了所有流失客户")
    print(f"  场景3的召回率为 {recall3:.1%}，能识别一半的流失客户")
    print(f"  → 在类别不平衡场景下，准确率是误导性指标！")


def main() -> None:
    """主函数"""
    print("=" * 60)
    print("示例2: 从准确率到混淆矩阵")
    print("=" * 60)

    # 1. 生成类别不平衡数据
    df = generate_imbalanced_data(n_samples=1000, imbalance_ratio=0.15)

    print(f"\n📊 数据概览:")
    print(f"  总样本数: {len(df)}")
    print(f"  流失客户: {df['churn'].sum()} ({df['churn'].mean():.1%})")
    print(f"  不流失客户: {(df['churn'] == 0).sum()} ({(df['churn'] == 0).mean():.1%})")
    print(f"  ⚠️  这是一个类别不平衡的数据集！")

    # 2. 划分数据
    X = df[['feature_1', 'feature_2']]
    y = df['churn']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 3. 训练逻辑回归模型
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 4. 计算混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # 5. 画混淆矩阵
    plot_confusion_matrix(cm, ['不流失', '流失'], '逻辑回归混淆矩阵')

    # 6. 解释指标
    explain_confusion_matrix_metrics(tn, fp, fn, tp)

    # 7. 打印分类报告
    print("\n" + "=" * 60)
    print("分类报告 (sklearn.metrics.classification_report)")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=['不流失', '流失']))

    # 8. 与基线对比
    print("\n" + "=" * 60)
    print("与基线模型对比")
    print("=" * 60)

    dummy = DummyClassifier(strategy='most_frequent', random_state=42)
    dummy.fit(X_train, y_train)
    y_pred_dummy = dummy.predict(X_test)

    acc_model = accuracy_score(y_test, y_pred)
    acc_dummy = accuracy_score(y_test, y_pred_dummy)
    recall_model = recall_score(y_test, y_pred)
    recall_dummy = recall_score(y_test, y_pred_dummy)

    print(f"\n基线模型（总是预测多数类）:")
    print(f"  准确率: {acc_dummy:.2%}")
    print(f"  召回率: {recall_dummy:.2%}")

    print(f"\n逻辑回归模型:")
    print(f"  准确率: {acc_model:.2%}")
    print(f"  召回率: {recall_model:.2%}")

    print(f"\n改进:")
    print(f"  准确率变化: {(acc_model - acc_dummy):.1%}")
    print(f"  召回率变化: {(recall_model - recall_dummy):.1%}")

    # 9. 演示准确率悖论
    demonstrate_accuracy_paradox()

    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print("""
在类别不平衡场景下：
1. 准确率会撒谎：总是预测多数类的模型准确率很高但毫无价值
2. 需要看混淆矩阵：关注 TP, TN, FP, FN 的分布
3. 根据业务目标优化：
   - 想减少误报（浪费营销成本）→ 优化精确率
   - 想减少漏报（抓住更多流失客户）→ 优化召回率
   - 需要平衡两者 → 看 F1 分数
    """)

    print("\n" + "=" * 60)
    print("✅ 示例2完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
示例：ROC-AUC——阈值无关的评估

本例演示：
1. ROC 曲线的含义：假阳性率 vs 真阳性率的权衡
2. AUC 的直觉：随机样本对排序正确的概率
3. 不同阈值下的性能变化
4. 多个模型的 ROC 曲线对比

运行方式：python3 chapters/week_10/examples/03_roc_auc_demo.py
预期输出：
- ROC 曲线图（保存为 roc_curve.png）
- 控制台输出 AUC 值和不同阈值下的指标
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
from sklearn.datasets import make_classification

OUTPUT_DIR = Path(__file__).parent.parent.parent / "output" / "week_10"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 设置随机种子
np.random.seed(42)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def generate_binary_data(n_samples: int = 1000) -> tuple:
    """生成二分类数据"""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_clusters_per_class=1,
        weights=[0.85, 0.15],  # 类别不平衡
        random_state=42
    )
    return X, y


def plot_roc_curve_comparison(X_test, y_test, models: dict) -> None:
    """
    绘制多个模型的 ROC 曲线对比

    参数:
        X_test: 测试集特征
        y_test: 测试集标签
        models: 字典 {模型名: (模型, 预测概率)}
    """
    plt.figure(figsize=(10, 6))

    # 画对角线（随机猜测）
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1,
             label='随机猜测 (AUC = 0.5)')

    # 为每个模型画 ROC 曲线
    colors = ['steelblue', 'darkorange', 'forestgreen']
    for i, (name, (model, y_proba)) in enumerate(models.items()):
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)

        plt.plot(fpr, tpr, color=colors[i % len(colors)],
                linewidth=2, label=f'{name} (AUC = {auc:.3f})')

    plt.xlabel('假阳性率 (FPR = FP / (FP + TN))', fontsize=12)
    plt.ylabel('真阳性率 (TPR = Recall)', fontsize=12)
    plt.title('ROC 曲线：模型对比', fontsize=14)
    plt.legend(fontsize=11, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'roc_curve.png', dpi=150, bbox_inches='tight')
    print("✅ ROC 曲线已保存为 roc_curve.png")
    plt.close()


def explain_threshold_tradeoff(y_proba, y_test) -> None:
    """
    演示不同阈值下的性能权衡

    参数:
        y_proba: 预测概率
        y_test: 真实标签
    """
    print("\n" + "=" * 60)
    print("不同阈值下的性能权衡")
    print("=" * 60)

    # 测试不同阈值
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]

    print(f"\n{'阈值':<8} {'预测为正':<10} {'精确率':<10} {'召回率':<10} {'F1':<10}")
    print("-" * 60)

    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)

        n_pred_positive = y_pred.sum()
        precision = y_pred[y_pred == 1].mean() if n_pred_positive > 0 else 0

        tp = ((y_pred == 1) & (y_test == 1)).sum()
        fp = ((y_pred == 1) & (y_test == 0)).sum()
        fn = ((y_pred == 0) & (y_test == 1)).sum()

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"{thresh:<8.1f} {n_pred_positive:<10} {precision:<10.2%} {recall:<10.2%} {f1:<10.3f}")

    print("\n" + "=" * 60)
    print("观察：")
    print("=" * 60)
    print("  阈值越低 → 召回率越高（抓到更多流失客户）")
    print("           → 精确率越低（误报更多）")
    print("  阈值越高 → 召回率越低（漏掉更多流失客户）")
    print("           → 精确率越高（误报更少）")


def explain_auc_intuition(y_proba, y_test) -> None:
    """
    解释 AUC 的直观含义

    参数:
        y_proba: 预测概率
        y_test: 真实标签
    """
    auc = roc_auc_score(y_test, y_proba)

    print("\n" + "=" * 60)
    print("AUC 的直观含义")
    print("=" * 60)

    # 抽样演示
    positive_indices = np.where(y_test == 1)[0]
    negative_indices = np.where(y_test == 0)[0]

    print(f"\n模拟：随机选 10 对（流失, 不流失）样本")
    print(f"{'':>4} {'流失样本':<12} {'不流失样本':<12} {'模型判断':<12}")
    print("-" * 50)

    correct_count = 0
    for i in range(min(10, len(positive_indices))):
        pos_idx = np.random.choice(positive_indices)
        neg_idx = np.random.choice(negative_indices)

        pos_proba = y_proba[pos_idx]
        neg_proba = y_proba[neg_idx]

        # 判断模型是否正确排序（流失 > 不流失）
        is_correct = pos_proba > neg_proba
        if is_correct:
            correct_count += 1
            status = "✓ 正确"
        else:
            status = "✗ 错误"

        print(f"#{i+1:>3} {pos_proba:<12.3f} {neg_proba:<12.3f} {status:<12}")

    empirical_auc = correct_count / 10
    print(f"\n在这 10 对样本中，模型正确排序的比例: {empirical_auc:.1%}")
    print(f"真实 AUC（基于所有样本对）: {auc:.3f}")
    print(f"\n💡 AUC = {auc:.3f} 的含义：")
    print(f"   如果你随机选一个'流失'客户和一个'不流失'客户，")
    print(f"   模型给'流失'客户更高概率的概率是 {auc:.1%}")


def plot_precision_recall_curve(y_proba, y_test, auc_roc: float) -> None:
    """
    绘制精确率-召回率曲线

    参数:
        y_proba: 预测概率
        y_test: 真实标签
        auc_roc: ROC-AUC 值
    """
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, linewidth=2, color='darkorange')
    plt.xlabel('召回率 (Recall)', fontsize=12)
    plt.ylabel('精确率 (Precision)', fontsize=12)
    plt.title(f'精确率-召回率曲线 (ROC-AUC = {auc_roc:.3f})', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'pr_curve.png', dpi=150, bbox_inches='tight')
    print("✅ PR 曲线已保存为 pr_curve.png")
    plt.close()


def main() -> None:
    """主函数"""
    print("=" * 60)
    print("示例3: ROC-AUC——阈值无关的评估")
    print("=" * 60)

    # 1. 生成数据
    X, y = generate_binary_data(n_samples=1000)

    # 划分数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"\n📊 数据概览:")
    print(f"  训练集: {len(X_train)} 样本")
    print(f"  测试集: {len(X_test)} 样本")
    print(f"  正类比例: {y.mean():.1%}")

    # 2. 训练多个模型
    models = {}

    # 模型1：逻辑回归
    log_reg = LogisticRegression(random_state=42, max_iter=1000)
    log_reg.fit(X_train, y_train)
    y_proba_logreg = log_reg.predict_proba(X_test)[:, 1]
    models['逻辑回归'] = (log_reg, y_proba_logreg)

    # 模型2：随机森林
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_proba_rf = rf.predict_proba(X_test)[:, 1]
    models['随机森林'] = (rf, y_proba_rf)

    # 模型3：弱模型（特征随机）
    np.random.seed(42)
    y_proba_weak = np.random.uniform(0, 1, len(y_test))
    models['弱模型'] = (None, y_proba_weak)

    # 3. 画 ROC 曲线
    plot_roc_curve_comparison(X_test, y_test, models)

    # 4. 计算 AUC
    print("\n" + "=" * 60)
    print("AUC 值对比")
    print("=" * 60)

    for name, (_, y_proba) in models.items():
        auc = roc_auc_score(y_test, y_proba)
        if auc > 0.8:
            strength = "强"
        elif auc > 0.7:
            strength = "中等"
        elif auc > 0.6:
            strength = "弱"
        else:
            strength = "很差"
        print(f"  {name:<12} AUC = {auc:.3f} ({strength}区分能力)")

    # 5. 解释 AUC 直觉
    explain_auc_intuition(y_proba_logreg, y_test)

    # 6. 解释阈值权衡
    explain_threshold_tradeoff(y_proba_logreg, y_test)

    # 7. 画 PR 曲线
    plot_precision_recall_curve(y_proba_logreg, y_test, roc_auc_score(y_test, y_proba_logreg))

    # 8. 总结
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print("""
ROC-AUC 的核心价值：
1. 阈值无关：不依赖具体的分类阈值（0.5 或其他）
2. 整体评估：衡量模型在所有阈值下的综合性能
3. 模型对比：用于选择更好的模型（AUC 更高）

AUC 的直观解释：
- AUC = 1.0: 完美分类器
- AUC = 0.5: 随机猜测（像抛硬币）
- AUC = 0.75: 随机选一对样本，模型有 75% 概率正确排序

使用建议：
- 模型选择：用 AUC 比较（选 AUC 更高的）
- 阈值调整：用 PR 曲线或业务成本确定最优阈值
- 不平衡数据：PR-AUC 比 ROC-AUC 更现实
    """)

    print("\n" + "=" * 60)
    print("✅ 示例3完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()

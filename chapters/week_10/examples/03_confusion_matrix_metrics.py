"""
示例：混淆矩阵与评估指标——准确率、精确率、召回率、F1

运行方式：python3 chapters/week_10/examples/03_confusion_matrix.py
预期输出：混淆矩阵可视化、各类评估指标计算、类别不平衡时的准确率陷阱
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
import seaborn as sns

# 配置中文字体
def setup_chinese_font() -> str:
    """配置中文字体，返回使用的字体名称"""
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


def create_imbalanced_data(n_samples: int = 1000, imbalance_ratio: float = 0.2) -> tuple:
    """创建类别不平衡的数据集

    Args:
        n_samples: 样本数量
        imbalance_ratio: 正类（少数类）比例

    Returns:
        X, y: 特征和标签
    """
    np.random.seed(42)
    n_positive = int(n_samples * imbalance_ratio)
    n_negative = n_samples - n_positive

    # 负类样本（多数类）
    X_neg = np.random.randn(n_negative, 2) + np.array([0, 0])
    y_neg = np.zeros(n_negative, dtype=int)

    # 正类样本（少数类）
    X_pos = np.random.randn(n_positive, 2) + np.array([2, 2])
    y_pos = np.ones(n_positive, dtype=int)

    X = np.vstack([X_neg, X_pos])
    y = np.hstack([y_neg, y_pos])

    return X, y


def visualize_confusion_matrix(cm: np.ndarray, title: str = "混淆矩阵") -> plt.Figure:
    """可视化混淆矩阵"""
    font = setup_chinese_font()
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    # 设置标签
    classes = ['负类 (0)', '正类 (1)']
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='实际标签',
           xlabel='预测标签')

    # 在每个格子中添加数值
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            # 显示数值和术语
            text_value = format(cm[i, j], 'd')
            if i == 0 and j == 0:
                term = '\n(TN)'
            elif i == 0 and j == 1:
                term = '\n(FP)'
            elif i == 1 and j == 0:
                term = '\n(FN)'
            else:
                term = '\n(TP)'

            text_color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, text_value + term,
                   ha="center", va="center", color=text_color, fontsize=12)

    return fig


def compute_metrics_from_cm(cm: np.ndarray) -> dict:
    """从混淆矩阵计算所有评估指标"""
    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity
    }


def demonstrate_confusion_matrix_terms() -> None:
    """演示混淆矩阵术语"""
    print("=" * 60)
    print("混淆矩阵术语解释")
    print("=" * 60)

    cm = np.array([[50, 10], [5, 35]])
    print("\n混淆矩阵示例：")
    print(f"                预测负类  预测正类")
    print(f"实际负类 (0)  |   {cm[0,0]:2d}   |   {cm[0,1]:2d}   |")
    print(f"实际正类 (1)  |   {cm[1,0]:2d}   |   {cm[1,1]:2d}   |")

    print("\n术语解释：")
    print(f"  TN (True Negative)  = {cm[0,0]} : 预测负类，实际负类（拒对了）")
    print(f"  FP (False Positive) = {cm[0,1]} : 预测正类，实际负类（误判）")
    print(f"  FN (False Negative) = {cm[1,0]} : 预测负类，实际正类（漏判）")
    print(f"  TP (True Positive)  = {cm[1,1]} : 预测正类，实际正类（抓对了）")

    metrics = compute_metrics_from_cm(cm)
    print(f"\n评估指标：")
    print(f"  准确率   = {metrics['accuracy']:.4f}")
    print(f"  精确率   = {metrics['precision']:.4f}")
    print(f"  召回率   = {metrics['recall']:.4f}")
    print(f"  F1 分数  = {metrics['f1']:.4f}")
    print(f"  特异度   = {metrics['specificity']:.4f}")

    # 可视化
    fig = visualize_confusion_matrix(cm, "混淆矩阵示例")
    output_dir = Path(__file__).parent.parent / 'images'
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'confusion_matrix_example.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"\n图片已保存到: {output_dir / 'confusion_matrix_example.png'}")


def demonstrate_accuracy_trap() -> None:
    """演示准确率陷阱：类别不平衡时准确率会误导"""
    print("\n" + "=" * 60)
    print("准确率陷阱：类别不平衡时准确率会误导")
    print("=" * 60)

    # 创建类别不平衡的数据
    X, y = create_imbalanced_data(n_samples=1000, imbalance_ratio=0.2)
    print(f"\n数据集信息：")
    print(f"  总样本数: {len(y)}")
    print(f"  负类（未流失）: {(y == 0).sum()} ({(y == 0).sum() / len(y) * 100:.1f}%)")
    print(f"  正类（流失）: {(y == 1).sum()} ({(y == 1).sum() / len(y) * 100:.1f}%)")

    # 划分数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 训练模型
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 计算混淆矩阵和指标
    cm = confusion_matrix(y_test, y_pred)
    metrics = compute_metrics_from_cm(cm)

    print(f"\n模型评估：")
    print(f"  准确率: {metrics['accuracy']:.4f}")
    print(f"  精确率: {metrics['precision']:.4f}")
    print(f"  召回率: {metrics['recall']:.4f}")
    print(f"  F1 分数: {metrics['f1']:.4f}")

    # 傻瓜基线：总是预测多数类
    print(f"\n傻瓜基线（永远预测负类）：")
    dummy_pred = np.zeros_like(y_test)
    dummy_cm = confusion_matrix(y_test, dummy_pred)
    dummy_metrics = compute_metrics_from_cm(dummy_cm)

    print(f"  准确率: {dummy_metrics['accuracy']:.4f}")
    print(f"  精确率: {dummy_metrics['precision']:.4f} (无意义，没有预测正类)")
    print(f"  召回率: {dummy_metrics['recall']:.4f} (完全漏掉所有正类！)")
    print(f"  F1 分数: {dummy_metrics['f1']:.4f}")

    print(f"\n结论：")
    print(f"  傻瓜基线的准确率是 {dummy_metrics['accuracy']:.4f}，但召回率是 0！")
    print(f"  如果只看准确率，你会误以为模型不错。")
    print(f"  类别不平衡时，必须看精确率、召回率、F1 等指标。")

    # 可视化对比
    font = setup_chinese_font()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 模型的混淆矩阵
    im0 = axes[0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    axes[0].set_title(f'逻辑回归\n准确率={metrics["accuracy"]:.3f}, F1={metrics["f1"]:.3f}')
    axes[0].set_ylabel('实际标签')
    axes[0].set_xlabel('预测标签')
    axes[0].set_xticks([0, 1])
    axes[0].set_yticks([0, 1])
    axes[0].set_xticklabels(['负类', '正类'])
    axes[0].set_yticklabels(['负类', '正类'])
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[0].text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")

    # 傻瓜基线的混淆矩阵
    im1 = axes[1].imshow(dummy_cm, interpolation='nearest', cmap=plt.cm.Reds)
    axes[1].set_title(f'傻瓜基线（永远猜负类）\n准确率={dummy_metrics["accuracy"]:.3f}, 召回率={dummy_metrics["recall"]:.3f}')
    axes[1].set_xlabel('预测标签')
    axes[1].set_xticks([0, 1])
    axes[1].set_yticks([0, 1])
    axes[1].set_xticklabels(['负类', '正类'])
    axes[1].set_yticklabels(['负类', '正类'])
    thresh = dummy_cm.max() / 2.
    for i in range(dummy_cm.shape[0]):
        for j in range(dummy_cm.shape[1]):
            axes[1].text(j, i, format(dummy_cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if dummy_cm[i, j] > thresh else "black")

    plt.tight_layout()

    output_dir = Path(__file__).parent.parent / 'images'
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'accuracy_trap_comparison.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"\n图片已保存到: {output_dir / 'accuracy_trap_comparison.png'}")


def demonstrate_precision_recall_tradeoff() -> None:
    """演示精确率-召回率权衡"""
    print("\n" + "=" * 60)
    print("精确率-召回率权衡：阈值变化的影响")
    print("=" * 60)

    # 创建数据
    X, y = create_imbalanced_data(n_samples=1000, imbalance_ratio=0.3)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]

    # 不同阈值下的预测
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    results = []

    print(f"\n{'阈值':<8} {'精确率':<10} {'召回率':<10} {'F1':<10} {'策略'}")
    print("-" * 70)
    for threshold in thresholds:
        y_pred_thresh = (y_prob >= threshold).astype(int)
        precision = precision_score(y_test, y_pred_thresh, zero_division=0)
        recall = recall_score(y_test, y_pred_thresh, zero_division=0)
        f1 = f1_score(y_test, y_pred_thresh, zero_division=0)

        if threshold < 0.5:
            strategy = "宽松（预测更多正类）"
        elif threshold == 0.5:
            strategy = "平衡"
        else:
            strategy = "严格（预测更少正类）"

        print(f"{threshold:<8.1f} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {strategy}")
        results.append({'threshold': threshold, 'precision': precision,
                       'recall': recall, 'f1': f1})

    # 绘制权衡曲线
    font = setup_chinese_font()
    fig, ax = plt.subplots(figsize=(10, 6))

    thresholds_arr = [r['threshold'] for r in results]
    precisions = [r['precision'] for r in results]
    recalls = [r['recall'] for r in results]

    ax.plot(thresholds_arr, precisions, 'o-', label='精确率', linewidth=2, markersize=8)
    ax.plot(thresholds_arr, recalls, 's-', label='召回率', linewidth=2, markersize=8)
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='默认阈值 (0.5)')
    ax.set_xlabel('阈值')
    ax.set_ylabel('指标值')
    ax.set_title('精确率-召回率权衡：阈值变化的影响')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()

    output_dir = Path(__file__).parent.parent / 'images'
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'precision_recall_tradeoff.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"\n图片已保存到: {output_dir / 'precision_recall_tradeoff.png'}")

    print(f"\n关键洞察：")
    print(f"  - 阈值降低 -> 召回率上升（抓到更多正类），精确率下降（误判更多）")
    print(f"  - 阈值提高 -> 精确率上升（预测更谨慎），召回率下降（漏掉更多正类）")
    print(f"  - 没有完美的阈值，需要根据业务场景选择")


def main() -> None:
    """主函数"""
    from pathlib import Path

    demonstrate_confusion_matrix_terms()
    demonstrate_accuracy_trap()
    demonstrate_precision_recall_tradeoff()

    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print("""
    混淆矩阵与评估指标总结：

    1. 混淆矩阵的四个象限：
       - TN：正确拒绝的负类
       - TP：正确抓到的正类
       - FP：误判为正类的负类（第一类错误）
       - FN：漏掉的正类（第二类错误）

    2. 评估指标：
       - 准确率 = (TP + TN) / 总数 -> 整体正确率
       - 精确率 = TP / (TP + FP) -> 预测为正类中，真正为正类的比例
       - 召回率 = TP / (TP + FN) -> 实际正类中，被正确预测的比例
       - F1 = 2 * 精确率 * 召回率 / (精确率 + 召回率) -> 两者的调和平均

    3. 类别不平衡时：
       - 准确率会误导（永远猜多数类也有高准确率）
       - 必须看精确率、召回率、F1
       - 根据业务场景选择优先指标

    4. 业务场景与指标选择：
       - 流失预测/医疗诊断：优先召回率（不要漏掉）
       - 欺诈检测/垃圾邮件过滤：优先精确率（不要误判）
       - 推荐系统：优先 AUC（排序能力）
    """)


if __name__ == "__main__":
    main()

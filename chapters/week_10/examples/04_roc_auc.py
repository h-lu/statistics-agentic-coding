"""
示例：ROC 曲线与 AUC——全面评估分类器（不依赖单一阈值）

运行方式：python3 chapters/week_10/examples/04_roc_auc.py
预期输出：ROC 曲线可视化、AUC 计算与解读、不同阈值下的 TPR/FPR 变化
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.datasets import make_classification
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


def demonstrate_threshold_sensitivity() -> None:
    """演示不同阈值下的 TPR 和 FPR 变化"""
    print("=" * 60)
    print("阈值变化对 TPR 和 FPR 的影响")
    print("=" * 60)

    # 模拟预测概率和实际标签
    np.random.seed(42)
    y_prob = np.random.uniform(0, 1, 100)  # 预测概率
    y_true = (y_prob > 0.6).astype(int)   # 实际标签（制造可分的数据）

    # 计算不同阈值下的 TPR 和 FPR
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    results = []

    print(f"\n{'阈值':<8} {'TPR':<10} {'FPR':<10} {'策略'}")
    print("-" * 50)

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)

        # 计算 TP, FP, FN, TN
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        if threshold < 0.5:
            strategy = "宽松"
        elif threshold == 0.5:
            strategy = "平衡"
        else:
            strategy = "严格"

        print(f"{threshold:<8.1f} {tpr:<10.4f} {fpr:<10.4f} {strategy}")
        results.append({'threshold': threshold, 'tpr': tpr, 'fpr': fpr})

    print(f"\n关键洞察：")
    print(f"  - 阈值降低 -> TPR 上升（抓到更多正类），FPR 上升（误判更多）")
    print(f"  - 阈值提高 -> TPR 下降（漏掉更多），FPR 下降（误判更少）")
    print(f"  - ROC 曲线就是描绘这种权衡关系")


def plot_roc_curve_explanation() -> None:
    """绘制 ROC 曲线说明图"""
    font = setup_chinese_font()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左图：ROC 曲线示意图
    ax = axes[0]

    # 随机猜测
    ax.plot([0, 1], [0, 1], 'k--', label='随机猜测 (AUC=0.5)', linewidth=2)

    # 完美分类器
    ax.plot([0, 0, 1], [0, 1, 1], 'g-', label='完美分类器 (AUC=1.0)', linewidth=2)

    # 不错的分类器
    auc_x = np.linspace(0, 1, 100)
    auc_y = 1 - np.exp(-3 * auc_x)  # 类似逻辑函数的曲线
    ax.plot(auc_x, auc_y, 'b-', label='不错的分类器 (AUC≈0.85)', linewidth=2)

    ax.set_xlabel('假正率 FPR (1 - 特异度)')
    ax.set_ylabel('真正率 TPR (召回率)')
    ax.set_title('ROC 曲线：不同分类器的表现')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)

    # 添加区域标注
    ax.fill_between(auc_x, auc_y, alpha=0.1, color='blue')

    # 右图：AUC 值解读表
    ax = axes[1]
    ax.axis('off')

    auc_data = [
        ['AUC 值', '分类器质量'],
        ['1.0', '完美分类器'],
        ['0.9 - 1.0', '优秀'],
        ['0.8 - 0.9', '良好'],
        ['0.7 - 0.8', '一般'],
        ['0.5', '随机猜测'],
        ['< 0.5', '比随机还差（可能预测反了）']
    ]

    table = ax.table(cellText=auc_data[1:], colLabels=auc_data[0],
                     cellLoc='center', loc='center',
                     colWidths=[0.4, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    ax.set_title('AUC 值解读', pad=20)

    plt.tight_layout()

    # 保存图片
    output_dir = Path(__file__).parent.parent / 'images'
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'roc_curve_explanation.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"\n图片已保存到: {output_dir / 'roc_curve_explanation.png'}")


def train_and_plot_roc() -> dict:
    """训练模型并绘制 ROC 曲线"""
    print("\n" + "=" * 60)
    print("ROC 曲线与 AUC 实战")
    print("=" * 60)

    # 创建分类数据集
    X, y = make_classification(
        n_samples=1000,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        random_state=42,
        flip_y=0.1,  # 添加一些噪声
        weights=[0.7, 0.3]  # 类别不平衡
    )

    print(f"\n数据集信息：")
    print(f"  总样本数: {len(y)}")
    print(f"  负类: {(y == 0).sum()} ({(y == 0).sum() / len(y) * 100:.1f}%)")
    print(f"  正类: {(y == 1).sum()} ({(y == 1).sum() / len(y) * 100:.1f}%)")

    # 划分数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 训练模型
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    # 预测概率
    y_prob = model.predict_proba(X_test)[:, 1]

    # 计算 ROC 曲线
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    print(f"\n模型 AUC: {auc:.4f}")

    # 绘制 ROC 曲线
    font = setup_chinese_font()
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'逻辑回归 (AUC = {auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', label='随机猜测 (AUC = 0.5)', linewidth=2)

    # 标注最佳阈值附近点
    youden_index = tpr - fpr
    best_idx = np.argmax(youden_index)
    best_threshold = thresholds[best_idx]
    best_fpr = fpr[best_idx]
    best_tpr = tpr[best_idx]

    ax.plot(best_fpr, best_tpr, 'ro', markersize=10,
            label=f'最佳阈值约 {best_threshold:.3f} (Youden指数)')
    ax.annotate(f'({best_fpr:.3f}, {best_tpr:.3f})',
                xy=(best_fpr, best_tpr),
                xytext=(best_fpr + 0.1, best_tpr - 0.1),
                fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    ax.set_xlabel('假正率 FPR (1 - 特异度)')
    ax.set_ylabel('真正率 TPR (召回率)')
    ax.set_title('ROC 曲线：逻辑回归分类器评估')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()

    # 保存图片
    output_dir = Path(__file__).parent.parent / 'images'
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'roc_curve_actual.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"图片已保存到: {output_dir / 'roc_curve_actual.png'}")

    # 打印不同阈值下的指标
    print(f"\n不同阈值下的 TPR 和 FPR（部分采样）：")
    print(f"{'阈值':<10} {'FPR':<10} {'TPR':<10} {'Youden指数':<12}")
    print("-" * 50)
    step = max(1, len(thresholds) // 10)
    for i in range(0, len(thresholds), step):
        print(f"{thresholds[i]:<10.3f} {fpr[i]:<10.3f} {tpr[i]:<10.3f} {tpr[i] - fpr[i]:<12.3f}")

    return {'fpr': fpr, 'tpr': tpr, 'auc': auc, 'thresholds': thresholds}


def compare_classifiers_by_roc() -> None:
    """比较不同分类器的 ROC 曲线"""
    print("\n" + "=" * 60)
    print("比较不同分类器的 ROC 曲线")
    print("=" * 60)

    # 创建数据
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        random_state=42,
        weights=[0.7, 0.3]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import RandomForestClassifier

    # 训练不同的模型
    models = {
        '随机基线': DummyClassifier(strategy='uniform', random_state=42),
        '多数类基线': DummyClassifier(strategy='most_frequent'),
        '逻辑回归': LogisticRegression(max_iter=1000, random_state=42),
        '随机森林': RandomForestClassifier(n_estimators=100, random_state=42)
    }

    font = setup_chinese_font()
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ['gray', 'brown', 'blue', 'green']

    for (name, model), color in zip(models.items(), colors):
        model.fit(X_train, y_train)
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.predict(X_test)  # DummyClassifier 没有 predict_proba

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)

        ax.plot(fpr, tpr, color=color, linewidth=2, label=f'{name} (AUC = {auc:.4f})')

    ax.plot([0, 1], [0, 1], 'k--', label='随机猜测线', linewidth=1, alpha=0.5)
    ax.set_xlabel('假正率 FPR')
    ax.set_ylabel('真正率 TPR')
    ax.set_title('不同分类器的 ROC 曲线比较')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图片
    output_dir = Path(__file__).parent.parent / 'images'
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'roc_curve_comparison.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"图片已保存到: {output_dir / 'roc_curve_comparison.png'}")


def explain_auc_vs_accuracy() -> None:
    """解释 AUC 和准确率的区别"""
    print("\n" + "=" * 60)
    print("AUC vs 准确率：哪个更重要？")
    print("=" * 60)

    print("""
    AUC 和准确率的区别：

    | 维度       | 准确率                   | AUC                          |
    |-----------|-------------------------|-----------------------------|
    | 依赖阈值   | 是（需要选择阈值）         | 否（综合所有阈值）            |
    | 回答问题   | "整体预测正确的比例"       | "模型排序正负样本的能力"       |
    | 适用场景   | 类别平衡、需要单一阈值决策 | 类别不平衡、需要全面评估       |
    | 范围       | 0 到 1                   | 0 到 1（0.5 是随机）         |

    什么时候看准确率？
    - 类别相对平衡
    - 需要单一阈值决策（如"是否批准贷款"）
    - 业务关心整体正确率

    什么时候看 AUC？
    - 类别不平衡
    - 需要全面评估分类器（不依赖阈值）
    - 需要排序能力（如"优先联系哪些客户"）
    - 推荐系统、搜索排名

    经验法则：
    - 如果数据类别不平衡，AUC 通常比准确率更可靠
    - 如果需要单一阈值决策，看精确率和召回率
    - 如果需要排序能力，看 AUC
    """)


def main() -> None:
    """主函数"""
    from pathlib import Path

    demonstrate_threshold_sensitivity()
    plot_roc_curve_explanation()
    train_and_plot_roc()
    compare_classifiers_by_roc()
    explain_auc_vs_accuracy()

    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print("""
    ROC 曲线与 AUC 核心要点：

    1. ROC 曲线：
       - X 轴：FPR（假正率）= FP / (FP + TN)
       - Y 轴：TPR（真正率/召回率）= TP / (TP + FN)
       - 描述不同阈值下 TPR 和 FPR 的权衡

    2. AUC（ROC 曲线下面积）：
       - 1.0：完美分类器
       - 0.9-1.0：优秀
       - 0.8-0.9：良好
       - 0.5：随机猜测
       - < 0.5：比随机还差

    3. AUC 的优点：
       - 不依赖单一阈值
       - 衡量"排序能力"：正类是否有更高的预测概率
       - 适用于类别不平衡的数据

    4. AUC vs 准确率：
       - AUC：全面评估，适用于不平衡数据
       - 准确率：单一阈值评估，适用于平衡数据
    """)


if __name__ == "__main__":
    main()

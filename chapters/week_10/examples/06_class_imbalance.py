"""
示例：类别不平衡——识别、应对与评估。

本例演示：
1. 类别不平衡的识别与可视化
2. class_weight='balanced' 的用法
3. 过采样（SMOTE）和欠采样的应用
4. 不同策略的对比评估
5. 适合类别不平衡的评估指标

运行方式：python3 chapters/week_10/examples/06_class_imbalance.py
预期输出：
  - stdout 输出类别不平衡检测结果
  - 对比不同处理策略的效果
  - 保存图表到 images/06_class_imbalance.png

核心概念：
  - 类别不平衡比例 > 1:10 时需要特殊处理
  - class_weight='balanced': 自动调整类别权重
  - SMOTE: 合成少数类样本
  - 评估指标: 不要用准确率，用 F1、AUC-PR 等
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    balanced_accuracy_score, average_precision_score, confusion_matrix,
    classification_report
)
from pathlib import Path


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


def generate_imbalanced_data(n: int = 2000, random_state: int = 42) -> pd.DataFrame:
    """
    生成严重类别不平衡的数据（正类约占 5%）。
    
    参数:
        n: 样本量
        random_state: 随机种子
        
    返回:
        DataFrame 包含客户特征和高价值标签（少数类）
    """
    np.random.seed(random_state)
    
    data = pd.DataFrame({
        '注册月数': np.random.randint(1, 60, n),
        '月均浏览次数': np.random.poisson(25, n),
        '月均消费次数': np.random.poisson(2, n),
        '最近登录距今天数': np.random.randint(1, 90, n)
    })
    
    # 生成高价值标签（少数类，约 5%）
    score = (
        0.1 * data['注册月数'] +
        0.05 * data['月均浏览次数'] +
        0.5 * data['月均消费次数'] -
        0.03 * data['最近登录距今天数'] -
        6 +
        np.random.normal(0, 1, n)
    )
    data['是否高价值'] = (score > np.percentile(score, 95)).astype(int)
    
    return data


def detect_imbalance(df: pd.DataFrame) -> dict:
    """
    检测类别不平衡。
    
    返回:
        dict 包含类别分布信息
    """
    print("=" * 70)
    print("类别不平衡检测")
    print("=" * 70)
    
    y = df['是否高价值']
    class_counts = y.value_counts().sort_index()
    imbalance_ratio = class_counts.max() / class_counts.min()
    
    print(f"\n类别分布：")
    print(f"  低价值客户 (0): {class_counts[0]} ({class_counts[0]/len(y)*100:.2f}%)")
    print(f"  高价值客户 (1): {class_counts[1]} ({class_counts[1]/len(y)*100:.2f}%)")
    print(f"\n不平衡比例: 1:{imbalance_ratio:.1f}")
    
    print("\n不平衡程度判断：")
    if imbalance_ratio < 2:
        print("  ✅ 类别基本平衡（1:2 以内）")
        severity = "balanced"
    elif imbalance_ratio < 5:
        print("  ⚠️  轻度不平衡（1:2 到 1:5）")
        severity = "mild"
    elif imbalance_ratio < 10:
        print("  ⚠️  中度不平衡（1:5 到 1:10）")
        severity = "moderate"
    else:
        print("  🚨 严重不平衡（超过 1:10）")
        severity = "severe"
    
    print("\n准确率陷阱演示：")
    baseline_accuracy = class_counts[0] / len(y)
    print(f"  如果全部预测为低价值（多数类）：")
    print(f"  准确率 = {baseline_accuracy:.2%}")
    print(f"  但查全率 = 0%（所有高价值客户都被漏掉！）")
    
    return {
        'class_counts': class_counts,
        'imbalance_ratio': imbalance_ratio,
        'severity': severity
    }


def baseline_model(X_train: np.ndarray, X_test: np.ndarray, 
                   y_train: pd.Series, y_test: pd.Series) -> dict:
    """
    基准模型：不使用任何不平衡处理。
    """
    print("\n" + "=" * 70)
    print("策略 1：基准模型（无处理）")
    print("=" * 70)
    
    model = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    return evaluate_model(y_test, y_pred, y_prob, "基准模型")


def balanced_class_weight(X_train: np.ndarray, X_test: np.ndarray,
                          y_train: pd.Series, y_test: pd.Series) -> dict:
    """
    策略 2：使用 class_weight='balanced'。
    """
    print("\n" + "=" * 70)
    print("策略 2：class_weight='balanced'")
    print("=" * 70)
    
    # 计算权重
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    
    print(f"\n自动计算的类别权重：")
    print(f"  类别 0 (低价值): {weights[0]:.4f}")
    print(f"  类别 1 (高价值): {weights[1]:.4f}")
    print(f"\n权重计算公式：weight = n_samples / (n_classes * n_samples_in_class)")
    
    model = LogisticRegression(
        solver='lbfgs', 
        class_weight='balanced',
        max_iter=1000, 
        random_state=42
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    return evaluate_model(y_test, y_pred, y_prob, "class_weight='balanced'")


def smote_oversampling(X_train: np.ndarray, X_test: np.ndarray,
                       y_train: pd.Series, y_test: pd.Series) -> dict:
    """
    策略 3：使用 SMOTE 过采样。
    """
    print("\n" + "=" * 70)
    print("策略 3：SMOTE 过采样")
    print("=" * 70)
    
    try:
        from imblearn.over_sampling import SMOTE
        from collections import Counter
        
        print(f"\n原始训练集分布: {Counter(y_train)}")
        
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        
        print(f"SMOTE 后分布: {Counter(y_train_resampled)}")
        
        model = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)
        model.fit(X_train_resampled, y_train_resampled)
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        return evaluate_model(y_test, y_pred, y_prob, "SMOTE 过采样")
        
    except ImportError:
        print("\n⚠️  imbalanced-learn 未安装，跳过 SMOTE")
        print("   安装命令: pip install imbalanced-learn")
        return None


def undersampling(X_train: np.ndarray, X_test: np.ndarray,
                  y_train: pd.Series, y_test: pd.Series) -> dict:
    """
    策略 4：使用随机欠采样。
    """
    print("\n" + "=" * 70)
    print("策略 4：随机欠采样")
    print("=" * 70)
    
    try:
        from imblearn.under_sampling import RandomUnderSampler
        from collections import Counter
        
        print(f"\n原始训练集分布: {Counter(y_train)}")
        
        undersampler = RandomUnderSampler(random_state=42)
        X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train)
        
        print(f"欠采样后分布: {Counter(y_train_resampled)}")
        print(f"⚠️  注意：丢失了 {len(y_train) - len(y_train_resampled)} 个多数类样本")
        
        model = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)
        model.fit(X_train_resampled, y_train_resampled)
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        return evaluate_model(y_test, y_pred, y_prob, "随机欠采样")
        
    except ImportError:
        print("\n⚠️  imbalanced-learn 未安装，跳过欠采样")
        return None


def threshold_tuning(X_train: np.ndarray, X_test: np.ndarray,
                     y_train: pd.Series, y_test: pd.Series) -> dict:
    """
    策略 5：阈值调整（不重新训练模型）。
    """
    print("\n" + "=" * 70)
    print("策略 5：阈值调整")
    print("=" * 70)
    
    model = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # 计算不同阈值下的 F1
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    
    # 找到 F1 最大的阈值
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    optimal_idx = np.argmax(f1_scores[:-1])  # 排除最后一个 NaN
    optimal_threshold = thresholds[optimal_idx]
    
    print(f"\n最优阈值 (F1 最大): {optimal_threshold:.4f}")
    print(f"  该阈值下: Precision={precision[optimal_idx]:.2%}, Recall={recall[optimal_idx]:.2%}")
    
    # 使用新阈值做预测
    y_pred_new = (y_prob >= optimal_threshold).astype(int)
    
    return evaluate_model(y_test, y_pred_new, y_prob, f"阈值调整 (t={optimal_threshold:.2f})")


def evaluate_model(y_test: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray, 
                   method_name: str) -> dict:
    """
    评估模型表现。
    
    返回:
        dict 包含各项指标
    """
    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc_pr = average_precision_score(y_test, y_prob)
    
    print(f"\n{method_name} 评估结果：")
    print(f"  准确率 (Accuracy): {accuracy:.2%}")
    print(f"  平衡准确率 (Balanced Accuracy): {balanced_acc:.2%}")
    print(f"  精确率 (Precision): {precision:.2%}")
    print(f"  查全率 (Recall): {recall:.2%}")
    print(f"  F1 分数: {f1:.2%}")
    print(f"  AUC-PR: {auc_pr:.4f}")
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"\n  混淆矩阵: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    
    return {
        'method': method_name,
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_pr': auc_pr,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
    }


def compare_strategies(results: list) -> None:
    """
    对比不同策略的效果。
    """
    print("\n" + "=" * 70)
    print("策略对比总结")
    print("=" * 70)
    
    # 过滤掉 None 的结果
    results = [r for r in results if r is not None]
    
    print(f"\n{'策略':<25} {'准确率':<10} {'F1':<10} {'查全率':<10} {'AUC-PR':<10}")
    print("-" * 75)
    
    for r in results:
        print(f"{r['method']:<25} {r['accuracy']:<10.2%} {r['f1']:<10.2%} {r['recall']:<10.2%} {r['auc_pr']:<10.4f}")
    
    # 找出 F1 最高的策略
    best_f1 = max(results, key=lambda x: x['f1'])
    print(f"\n✅ F1 最高的策略: {best_f1['method']} (F1 = {best_f1['f1']:.2%})")
    
    # 找出查全率最高的策略
    best_recall = max(results, key=lambda x: x['recall'])
    print(f"✅ 查全率最高的策略: {best_recall['method']} (Recall = {best_recall['recall']:.2%})")


def plot_comparison(results: list) -> None:
    """绘制策略对比图"""
    setup_chinese_font()
    
    # 过滤掉 None 的结果
    results = [r for r in results if r is not None]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左图：指标对比
    ax1 = axes[0]
    methods = [r['method'] for r in results]
    x = np.arange(len(methods))
    width = 0.2
    
    metrics = ['precision', 'recall', 'f1']
    colors = ['#2E86AB', '#F18F01', '#C73E1D']
    labels = ['精确率', '查全率', 'F1']
    
    for i, (metric, color, label) in enumerate(zip(metrics, colors, labels)):
        values = [r[metric] for r in results]
        ax1.bar(x + i * width, values, width, label=label, color=color, edgecolor='black')
    
    ax1.set_xlabel('策略', fontsize=12)
    ax1.set_ylabel('分数', fontsize=12)
    ax1.set_title('不同策略的指标对比', fontsize=13, fontweight='bold')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(methods, rotation=15, ha='right')
    ax1.legend(fontsize=10)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 右图：混淆矩阵热力图（以最佳 F1 策略为例）
    ax2 = axes[1]
    best_result = max(results, key=lambda x: x['f1'])
    
    cm = np.array([[best_result['tn'], best_result['fp']],
                   [best_result['fn'], best_result['tp']]])
    
    im = ax2.imshow(cm, interpolation='nearest', cmap='Blues')
    ax2.set_title(f'最佳策略混淆矩阵\n{best_result["method"]}', fontsize=13, fontweight='bold')
    
    tick_marks = [0, 1]
    ax2.set_xticks(tick_marks)
    ax2.set_yticks(tick_marks)
    ax2.set_xticklabels(['低价值', '高价值'])
    ax2.set_yticklabels(['低价值', '高价值'])
    ax2.set_xlabel('预测类别', fontsize=12)
    ax2.set_ylabel('实际类别', fontsize=12)
    
    thresh = cm.max() / 2.
    for i in range(2):
        for j in range(2):
            ax2.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # 保存图片
    output_dir = Path(__file__).parent.parent / 'images'
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / '06_class_imbalance.png',
                dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"\n图表已保存到: images/06_class_imbalance.png")


def main() -> None:
    """主函数"""
    print("类别不平衡：识别、应对与评估\n")
    
    # 生成不平衡数据
    df = generate_imbalanced_data(n=2000, random_state=42)
    
    # 检测不平衡
    imbalance_info = detect_imbalance(df)
    
    # 准备数据
    X = df[['注册月数', '月均浏览次数', '月均消费次数', '最近登录距今天数']]
    y = df['是否高价值']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 特征缩放
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 测试不同策略
    results = []
    
    # 策略 1：基准
    results.append(baseline_model(X_train_scaled, X_test_scaled, y_train, y_test))
    
    # 策略 2：class_weight='balanced'
    results.append(balanced_class_weight(X_train_scaled, X_test_scaled, y_train, y_test))
    
    # 策略 3：SMOTE
    smote_result = smote_oversampling(X_train_scaled, X_test_scaled, y_train, y_test)
    if smote_result:
        results.append(smote_result)
    
    # 策略 4：欠采样
    under_result = undersampling(X_train_scaled, X_test_scaled, y_train, y_test)
    if under_result:
        results.append(under_result)
    
    # 策略 5：阈值调整
    results.append(threshold_tuning(X_train_scaled, X_test_scaled, y_train, y_test))
    
    # 对比策略
    compare_strategies(results)
    
    # 绘图
    plot_comparison(results)
    
    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print("\n类别不平衡的识别：")
    print("  - 不平衡比例 > 1:10 时需要特殊处理")
    print("  - 准确率是误导性指标（全猜多数类也能有高准确率）")
    print("\n应对策略：")
    print("  1. class_weight='balanced': 最简单，通常效果也不错")
    print("  2. SMOTE: 生成合成样本，增加少数类")
    print("  3. 欠采样: 减少多数类，可能丢失信息")
    print("  4. 阈值调整: 不改变模型，只调整决策阈值")
    print("\n评估指标选择：")
    print("  - 不要用准确率！")
    print("  - 优先使用 F1、AUC-PR、平衡准确率")
    print("  - 关注少数类的查全率")
    print("\n⚠️  重要：SMOTE/欠采样只能在训练集上使用，测试集必须保持原始分布！")
    print()


if __name__ == "__main__":
    main()

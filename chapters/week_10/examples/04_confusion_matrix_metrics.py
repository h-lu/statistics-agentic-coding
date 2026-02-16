"""
示例：混淆矩阵与分类指标——精确率、查全率、F1。

本例演示：
1. 混淆矩阵的计算与可视化
2. 精确率（Precision）、查全率（Recall）、F1 的计算
3. 分类报告的生成与解读
4. 不同业务场景下的指标选择

运行方式：python3 chapters/week_10/examples/04_confusion_matrix_metrics.py
预期输出：
  - stdout 输出混淆矩阵和分类指标
  - 展示混淆矩阵热力图
  - 保存图表到 images/04_confusion_matrix_metrics.png

核心概念：
  - TP: 真正例，FP: 假正例，TN: 真反例，FN: 假反例
  - Precision = TP / (TP + FP) - 预测为正类中，真的是正类的比例
  - Recall = TP / (TP + FN) - 真正类中，被预测为正类的比例
  - F1 = 2 * (Precision * Recall) / (Precision + Recall)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    precision_score, recall_score, f1_score, accuracy_score,
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


def generate_ecommerce_data(n: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """
    生成模拟电商客户数据（类别不平衡，约 15% 高价值客户）。
    
    参数:
        n: 样本量
        random_state: 随机种子
        
    返回:
        DataFrame 包含客户特征
    """
    np.random.seed(random_state)
    
    data = pd.DataFrame({
        '注册月数': np.random.randint(1, 48, n),
        '月均浏览次数': np.random.poisson(35, n),
        '购物车添加次数': np.random.poisson(6, n),
        '历史消费金额': np.random.exponential(scale=80, size=n)
    })
    
    # 生成高价值标签（约 15%）
    score = (
        0.08 * data['注册月数'] +
        0.04 * data['月均浏览次数'] +
        0.15 * data['购物车添加次数'] +
        0.008 * data['历史消费金额'] -
        4 +
        np.random.normal(0, 1, n)
    )
    data['是否高价值'] = (score > np.percentile(score, 85)).astype(int)
    
    return data


def fit_and_predict(df: pd.DataFrame) -> dict:
    """
    拟合逻辑回归模型并返回预测结果。
    
    返回:
        dict 包含真实标签和预测结果
    """
    X = df[['注册月数', '月均浏览次数', '购物车添加次数', '历史消费金额']]
    y = df['是否高价值']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    
    return {
        'y_test': y_test,
        'y_pred': y_pred
    }


def explain_confusion_matrix(y_test: pd.Series, y_pred: np.ndarray) -> dict:
    """
    解释混淆矩阵的四个象限。
    
    返回:
        dict 包含 TP, FP, TN, FN 的值
    """
    print("=" * 70)
    print("混淆矩阵详解")
    print("=" * 70)
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print("\n混淆矩阵的四象限：")
    print("┌─────────────────┬─────────────────┬─────────────────┐")
    print("│                 │   预测为负类    │   预测为正类    │")
    print("├─────────────────┼─────────────────┼─────────────────┤")
    print(f"│   实际为负类    │   TN = {tn:3d}      │   FP = {fp:3d}      │")
    print("│   (低价值)      │   ✅ 正确       │   ❌ 误报       │")
    print("├─────────────────┼─────────────────┼─────────────────┤")
    print(f"│   实际为正类    │   FN = {fn:3d}      │   TP = {tp:3d}      │")
    print("│   (高价值)      │   ❌ 漏报       │   ✅ 正确       │")
    print("└─────────────────┴─────────────────┴─────────────────┘")
    
    print("\n各象限含义：")
    print(f"  TN ({tn}): 实际是低价值，预测也是低价值 → 正确拒绝")
    print(f"  FP ({fp}): 实际是低价值，预测为高价值 → 误报（浪费营销资源）")
    print(f"  FN ({fn}): 实际是高价值，预测为低价值 → 漏报（损失重要客户）")
    print(f"  TP ({tp}): 实际是高价值，预测也是高价值 → 正确识别")
    
    return {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}


def calculate_metrics(cm_dict: dict, y_test: pd.Series, y_pred: np.ndarray) -> dict:
    """
    计算分类评估指标。
    
    返回:
        dict 包含各项指标
    """
    print("\n" + "=" * 70)
    print("分类评估指标")
    print("=" * 70)
    
    tn, fp, fn, tp = cm_dict['tn'], cm_dict['fp'], cm_dict['fn'], cm_dict['tp']
    
    # 手动计算
    accuracy_manual = (tp + tn) / (tp + tn + fp + fn)
    precision_manual = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_manual = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_manual = 2 * (precision_manual * recall_manual) / (precision_manual + recall_manual) if (precision_manual + recall_manual) > 0 else 0
    
    # sklearn 计算
    accuracy_sk = accuracy_score(y_test, y_pred)
    precision_sk = precision_score(y_test, y_pred, zero_division=0)
    recall_sk = recall_score(y_test, y_pred, zero_division=0)
    f1_sk = f1_score(y_test, y_pred, zero_division=0)
    
    print("\n手动计算 vs sklearn：")
    print(f"{'指标':<15} {'手动计算':<12} {'sklearn':<12}")
    print("-" * 45)
    print(f"{'准确率':<15} {accuracy_manual:<12.4f} {accuracy_sk:<12.4f}")
    print(f"{'精确率':<15} {precision_manual:<12.4f} {precision_sk:<12.4f}")
    print(f"{'查全率':<15} {recall_manual:<12.4f} {recall_sk:<12.4f}")
    print(f"{'F1 分数':<15} {f1_manual:<12.4f} {f1_sk:<12.4f}")
    
    print("\n公式详解：")
    print(f"  准确率 (Accuracy) = (TP + TN) / (TP + TN + FP + FN)")
    print(f"                    = ({tp} + {tn}) / {tp + tn + fp + fn}")
    print(f"                    = {accuracy_manual:.4f}")
    print()
    print(f"  精确率 (Precision) = TP / (TP + FP)")
    print(f"                     = {tp} / ({tp} + {fp})")
    print(f"                     = {precision_manual:.4f}")
    print(f"  含义：预测为高价值的客户中，{precision_manual:.1%} 真的是高价值")
    print()
    print(f"  查全率 (Recall) = TP / (TP + FN)")
    print(f"                  = {tp} / ({tp} + {fn})")
    print(f"                  = {recall_manual:.4f}")
    print(f"  含义：所有高价值客户中，只找到了 {recall_manual:.1%}")
    print()
    print(f"  F1 分数 = 2 × (Precision × Recall) / (Precision + Recall)")
    print(f"          = 2 × ({precision_manual:.4f} × {recall_manual:.4f}) / ({precision_manual:.4f} + {recall_manual:.4f})")
    print(f"          = {f1_manual:.4f}")
    print(f"  含义：精确率和查全率的调和平均")
    
    return {
        'accuracy': accuracy_manual,
        'precision': precision_manual,
        'recall': recall_manual,
        'f1': f1_manual
    }


def generate_classification_report(y_test: pd.Series, y_pred: np.ndarray) -> None:
    """
    生成分类报告。
    """
    print("\n" + "=" * 70)
    print("分类报告（Classification Report）")
    print("=" * 70)
    
    report = classification_report(
        y_test, y_pred, 
        target_names=['低价值', '高价值'],
        digits=4
    )
    print("\n" + report)
    
    print("报告解读：")
    print("  - precision: 每类的精确率")
    print("  - recall: 每类的查全率")
    print("  - f1-score: 每类的 F1 分数")
    print("  - support: 每类的样本数")
    print("  - macro avg: 各类别指标的无权平均（每类同等重要）")
    print("  - weighted avg: 按支持度加权的平均（考虑样本数）")


def business_scenarios() -> None:
    """
    讨论不同业务场景下的指标选择。
    """
    print("\n" + "=" * 70)
    print("业务场景：精确率 vs 查全率的权衡")
    print("=" * 70)
    
    scenarios = [
        {
            'name': '垃圾邮件过滤',
            'priority': '精确率',
            'reason': '宁可放过垃圾邮件，也别把正常邮件当垃圾',
            'fp_cost': '高（用户可能错过重要邮件）',
            'fn_cost': '低（垃圾邮件进入收件箱）'
        },
        {
            'name': '疾病筛查',
            'priority': '查全率',
            'reason': '宁可误报，也别漏掉真正患病的人',
            'fp_cost': '低（进一步检查的成本）',
            'fn_cost': '高（延误治疗，可能危及生命）'
        },
        {
            'name': '客户流失预警',
            'priority': '查全率',
            'reason': '宁可多给优惠，也别漏掉即将流失的客户',
            'fp_cost': '低（给忠诚客户发优惠券）',
            'fn_cost': '高（失去客户）'
        },
        {
            'name': '广告推荐',
            'priority': '精确率',
            'reason': '宁可少推荐，也别推荐不相关的广告',
            'fp_cost': '高（用户反感，降低体验）',
            'fn_cost': '低（少展示一个广告）'
        }
    ]
    
    for scenario in scenarios:
        print(f"\n📌 {scenario['name']}:")
        print(f"   优先指标: {scenario['priority']}")
        print(f"   原因: {scenario['reason']}")
        print(f"   误报代价: {scenario['fp_cost']}")
        print(f"   漏报代价: {scenario['fn_cost']}")
    
    print("\n" + "=" * 70)
    print("总结：没有完美的模型，只有适合业务的模型")
    print("=" * 70)
    print("\n关键问题：漏报和误报，哪个代价更高？")
    print("  - 漏报代价高 → 优先查全率（Recall）")
    print("  - 误报代价高 → 优先精确率（Precision）")
    print("  - 两者都要平衡 → 看 F1 分数")


def plot_confusion_matrix_and_metrics(y_test: pd.Series, y_pred: np.ndarray, metrics: dict) -> None:
    """绘制混淆矩阵和指标可视化"""
    setup_chinese_font()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左图：混淆矩阵热力图
    ax1 = axes[0]
    cm = confusion_matrix(y_test, y_pred)
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['低价值', '高价值'])
    disp.plot(ax=ax1, cmap='Blues', values_format='d', colorbar=True)
    ax1.set_title('混淆矩阵\nConfusion Matrix', fontsize=13, fontweight='bold')
    ax1.set_xlabel('预测类别', fontsize=12)
    ax1.set_ylabel('实际类别', fontsize=12)
    
    # 右图：指标对比柱状图
    ax2 = axes[1]
    metric_names = ['准确率\nAccuracy', '精确率\nPrecision', '查全率\nRecall', 'F1 分数\nF1-Score']
    metric_values = [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1']]
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    bars = ax2.bar(metric_names, metric_values, color=colors, edgecolor='black', linewidth=1.5)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel('分数', fontsize=12)
    ax2.set_title('分类评估指标对比', fontsize=13, fontweight='bold')
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 在柱状图上添加数值标签
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{value:.2%}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # 保存图片
    output_dir = Path(__file__).parent.parent / 'images'
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / '04_confusion_matrix_metrics.png',
                dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"\n图表已保存到: images/04_confusion_matrix_metrics.png")


def main() -> None:
    """主函数"""
    print("混淆矩阵与分类指标：精确率、查全率、F1\n")
    
    # 生成数据并拟合模型
    df = generate_ecommerce_data(n=1000, random_state=42)
    pred_result = fit_and_predict(df)
    y_test = pred_result['y_test']
    y_pred = pred_result['y_pred']
    
    # 混淆矩阵详解
    cm_dict = explain_confusion_matrix(y_test, y_pred)
    
    # 计算指标
    metrics = calculate_metrics(cm_dict, y_test, y_pred)
    
    # 分类报告
    generate_classification_report(y_test, y_pred)
    
    # 业务场景讨论
    business_scenarios()
    
    # 绘图
    plot_confusion_matrix_and_metrics(y_test, y_pred, metrics)
    
    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print("\n混淆矩阵是分类模型的'体检报告'：")
    print("  - TP: 正确识别的高价值客户")
    print("  - FP: 误报（浪费营销资源）")
    print("  - FN: 漏报（损失重要客户）")
    print("  - TN: 正确识别的低价值客户")
    print("\n核心指标：")
    print("  - 精确率：预测为正类的样本中，真正是正类的比例")
    print("  - 查全率：真正的正类中，被预测为正类的比例")
    print("  - F1：精确率和查全率的调和平均")
    print("\n业务选择：")
    print("  - 漏报代价高（疾病筛查、流失预警）→ 优先查全率")
    print("  - 误报代价高（垃圾邮件、广告推荐）→ 优先精确率")
    print()


if __name__ == "__main__":
    main()

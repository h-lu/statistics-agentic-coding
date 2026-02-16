"""
StatLab 超级线：分类分析模块。

本例在 Week 09 的回归分析基础上，添加分类分析功能：
- 逻辑回归拟合与评估
- 混淆矩阵与分类指标（精确率、查全率、F1）
- ROC 曲线与 AUC 计算
- 类别不平衡检测与处理
- 自动生成 Markdown 报告

这是 StatLab 超级线在 Week 10 的进展，添加了完整的分类分析能力。

运行方式：python3 chapters/week_10/examples/10_statlab_classification.py
预期输出：
  - stdout 输出分类分析结果
  - 生成图表到 images/
  - 生成报告到 output/classification_report.md

核心概念：
  - 分类问题的完整流程：拟合 → 评估 → 报告
  - 多维度评估：混淆矩阵 + 精确率/查全率/F1 + ROC-AUC
  - 类别不平衡检测与自动处理
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    precision_score, recall_score, f1_score, accuracy_score,
    balanced_accuracy_score, roc_curve, auc,
    RocCurveDisplay, classification_report
)
from pathlib import Path
from typing import Optional, Union


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


def classification_with_evaluation(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    var_names: Optional[list[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42
) -> dict:
    """
    拟合逻辑回归模型并输出完整评估报告。
    
    参数:
        X: 特征矩阵
        y: 目标变量（0/1）
        var_names: 特征名称列表
        test_size: 测试集比例
        random_state: 随机种子
        
    返回:
        dict: 包含模型、评估结果、图表数据的字典
    """
    # 转换为 DataFrame
    if isinstance(X, np.ndarray):
        if var_names is None:
            var_names = [f"feature_{i}" for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=var_names)
    else:
        X_df = X.copy()
        var_names = X_df.columns.tolist()
    
    y_array = np.array(y)
    
    # 划分训练集和测试集（分层抽样）
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_array, test_size=test_size, 
        random_state=random_state, stratify=y_array
    )
    
    # 检查类别分布
    class_dist = pd.Series(y_array).value_counts().sort_index()
    imbalance_ratio = class_dist.max() / class_dist.min()
    
    # 特征缩放
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 拟合模型（如果类别不平衡，使用 balanced 权重）
    if imbalance_ratio > 5:
        model = LogisticRegression(
            solver='lbfgs',
            class_weight='balanced', 
            max_iter=1000, 
            random_state=random_state
        )
        used_balanced = True
    else:
        model = LogisticRegression(
            solver='lbfgs',
            max_iter=1000, 
            random_state=random_state
        )
        used_balanced = False
    
    model.fit(X_train_scaled, y_train)
    
    # 预测
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    # 1. 基础评估指标
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    
    # 2. ROC 曲线
    fpr, tpr, roc_thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # 3. 交叉验证
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='f1')
    
    # 4. 特征重要性（系数）
    feature_importance = pd.DataFrame({
        'feature': var_names,
        'coefficient': model.coef_[0],
        'abs_coefficient': np.abs(model.coef_[0])
    }).sort_values('abs_coefficient', ascending=False)
    
    results = {
        'model': model,
        'scaler': scaler,
        'var_names': var_names,
        'class_distribution': class_dist.to_dict(),
        'imbalance_ratio': imbalance_ratio,
        'used_balanced_weight': used_balanced,
        'confusion_matrix': {
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
        },
        'metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'balanced_accuracy': balanced_acc
        },
        'roc_auc': roc_auc,
        'cv_f1_mean': cv_scores.mean(),
        'cv_f1_std': cv_scores.std(),
        'feature_importance': feature_importance.to_dict('records'),
        'plots': {
            'roc': {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'auc': roc_auc},
            'confusion_matrix': cm.tolist()
        },
        'test_data': {
            'y_test': y_test.tolist(),
            'y_pred': y_pred.tolist(),
            'y_prob': y_prob.tolist()
        }
    }
    
    return results


def plot_roc_curve(results: dict, figsize: tuple = (8, 6)) -> plt.Figure:
    """
    绘制 ROC 曲线。
    
    参数:
        results: classification_with_evaluation 的返回结果
        figsize: 图形大小
        
    返回:
        matplotlib Figure 对象
    """
    setup_chinese_font()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    roc_data = results['plots']['roc']
    ax.plot(roc_data['fpr'], roc_data['tpr'], 'b-', linewidth=2.5,
            label=f"ROC 曲线 (AUC = {roc_data['auc']:.3f})")
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='随机分类器')
    
    ax.set_xlabel('假正率 (FPR)', fontsize=12)
    ax.set_ylabel('真正率 (TPR)', fontsize=12)
    ax.set_title('ROC 曲线', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.fill_between(roc_data['fpr'], 0, roc_data['tpr'], alpha=0.2, color='blue')
    
    return fig


def plot_classification_results(results: dict, figsize: tuple = (15, 5)) -> plt.Figure:
    """
    画分类评估图表（综合版）。
    
    参数:
        results: classification_with_evaluation 的返回结果
        figsize: 图形大小
        
    返回:
        matplotlib Figure 对象
    """
    setup_chinese_font()
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # 1. 混淆矩阵热力图
    cm = np.array(results['plots']['confusion_matrix'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['低价值', '高价值'])
    disp.plot(ax=axes[0], cmap='Blues', values_format='d', colorbar=False)
    axes[0].set_title('混淆矩阵', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('预测类别', fontsize=11)
    axes[0].set_ylabel('实际类别', fontsize=11)
    
    # 2. ROC 曲线
    roc_data = results['plots']['roc']
    axes[1].plot(roc_data['fpr'], roc_data['tpr'], 'b-', linewidth=2.5,
                 label=f"ROC (AUC = {roc_data['auc']:.2f})")
    axes[1].plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='随机')
    axes[1].set_xlabel('假正率 (FPR)', fontsize=11)
    axes[1].set_ylabel('真正率 (TPR)', fontsize=11)
    axes[1].set_title('ROC 曲线', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=9, loc='lower right')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    
    # 3. 特征重要性
    importance = pd.DataFrame(results['feature_importance'])
    colors = ['#2E86AB' if c > 0 else '#A23B72' for c in importance['coefficient']]
    axes[2].barh(importance['feature'], importance['abs_coefficient'], color=colors)
    axes[2].set_xlabel('|系数|', fontsize=11)
    axes[2].set_title('特征重要性', fontsize=13, fontweight='bold')
    axes[2].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    return fig


def generate_classification_report(results: dict) -> str:
    """
    格式化分类结果为 Markdown 报告。
    
    参数:
        results: classification_with_evaluation 的返回结果
        
    返回:
        Markdown 格式的报告字符串
    """
    md = ["## 分类分析\n\n"]
    
    # 1. 数据概况
    md.append("### 数据概况\n\n")
    md.append(f"- 类别分布: {results['class_distribution']}\n")
    md.append(f"- 不平衡比例: 1:{results['imbalance_ratio']:.1f}\n")
    if results['used_balanced_weight']:
        md.append("- ⚠️ 检测到类别不平衡，已使用 class_weight='balanced'\n")
    md.append("\n")
    
    # 2. 混淆矩阵
    cm = results['confusion_matrix']
    md.append("### 混淆矩阵\n\n")
    md.append("|  | 预测负类 | 预测正类 |\n")
    md.append("|--|---------|---------|\n")
    md.append(f"| 实际负类 | {cm['tn']} | {cm['fp']} |\n")
    md.append(f"| 实际正类 | {cm['fn']} | {cm['tp']} |\n\n")
    
    # 3. 评估指标
    md.append("### 评估指标\n\n")
    md.append("| 指标 | 值 |\n")
    md.append("|------|-----|\n")
    m = results['metrics']
    md.append(f"| 准确率 (Accuracy) | {m['accuracy']:.2%} |\n")
    md.append(f"| 精确率 (Precision) | {m['precision']:.2%} |\n")
    md.append(f"| 查全率 (Recall) | {m['recall']:.2%} |\n")
    md.append(f"| F1 分数 | {m['f1']:.2%} |\n")
    md.append(f"| 平衡准确率 | {m['balanced_accuracy']:.2%} |\n")
    md.append(f"| AUC-ROC | {results['roc_auc']:.4f} |\n")
    md.append(f"| CV F1 (mean±std) | {results['cv_f1_mean']:.2%} ± {results['cv_f1_std']:.2%} |\n\n")
    
    # 4. 特征重要性
    md.append("### 特征重要性\n\n")
    md.append("| 特征 | 系数 | 绝对值 | 方向 |\n")
    md.append("|------|------|--------|------|\n")
    for feat in results['feature_importance']:
        direction = "正向 ⬆️" if feat['coefficient'] > 0 else "负向 ⬇️"
        md.append(f"| {feat['feature']} | {feat['coefficient']:.4f} | {feat['abs_coefficient']:.4f} | {direction} |\n")
    md.append("\n")
    
    # 5. 诊断结论
    md.append("### 诊断结论\n\n")
    
    # AUC 评估
    if results['roc_auc'] < 0.7:
        md.append("- ⚠️ AUC < 0.7，模型区分能力较弱，建议重新审视特征或尝试其他模型\n")
    elif results['roc_auc'] > 0.95:
        md.append("- ⚠️ AUC > 0.95，模型区分能力优秀，但需检查是否过拟合\n")
    else:
        md.append(f"- ✅ AUC = {results['roc_auc']:.4f}，模型区分能力尚可\n")
    
    # 精确率和查全率评估
    if m['precision'] < 0.5 and m['recall'] < 0.5:
        md.append("- ⚠️ 精确率和查全率均较低，模型可能需要优化\n")
    elif m['precision'] < 0.5:
        md.append("- ⚠️ 精确率较低，存在较多误报\n")
    elif m['recall'] < 0.5:
        md.append("- ⚠️ 查全率较低，存在较多漏报\n")
    
    # 类别不平衡警告
    if results['imbalance_ratio'] > 10:
        md.append("- ⚠️ 类别严重不平衡，建议关注查全率而非准确率\n")
    
    # 交叉验证结果
    md.append(f"- 交叉验证 F1: {results['cv_f1_mean']:.2%} ± {results['cv_f1_std']:.2%}\n")
    
    if results['cv_f1_std'] > 0.1:
        md.append("- ⚠️ 交叉验证标准差较大，模型稳定性有待提高\n")
    
    md.append("\n")
    
    return "".join(md)


def generate_ecommerce_data(n: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """生成模拟电商客户数据（类别不平衡）"""
    np.random.seed(random_state)
    
    data = pd.DataFrame({
        '历史消费金额': np.random.exponential(scale=100, size=n),
        '注册月数': np.random.randint(1, 48, n),
        '月均浏览次数': np.random.poisson(40, n),
        '购物车添加次数': np.random.poisson(8, n)
    })
    
    score = (
        0.01 * data['历史消费金额'] +
        0.1 * data['注册月数'] +
        0.05 * data['月均浏览次数'] +
        0.2 * data['购物车添加次数'] -
        5 +
        np.random.normal(0, 1, n)
    )
    data['是否高价值'] = (score > np.percentile(score, 90)).astype(int)
    
    return data


def main() -> None:
    """主函数：演示 StatLab 分类分析"""
    print("=" * 70)
    print("StatLab 超级线：分类分析与完整评估报告")
    print("=" * 70)
    
    # 加载数据
    print("\n加载数据...")
    df = generate_ecommerce_data(n=1000, random_state=42)
    
    print(f"数据集：{len(df)} 个客户样本")
    print(f"特征：历史消费金额、注册月数、月均浏览次数、购物车添加次数")
    print(f"目标：是否高价值客户")
    
    # 准备数据
    print("\n准备数据...")
    X = df[['历史消费金额', '注册月数', '月均浏览次数', '购物车添加次数']]
    y = df['是否高价值']
    
    print(f"类别分布：")
    print(f"  低价值客户 (0): {(y == 0).sum()} ({(y == 0).mean()*100:.1f}%)")
    print(f"  高价值客户 (1): {(y == 1).sum()} ({(y == 1).mean()*100:.1f}%)")
    
    # 运行分类分析
    print("\n" + "=" * 70)
    print("运行分类分析...")
    print("=" * 70)
    
    results = classification_with_evaluation(
        X, y, 
        var_names=['历史消费', '注册时长', '月均浏览', '购物车添加'],
        test_size=0.2,
        random_state=42
    )
    
    # 打印评估指标
    print("\n分类评估指标：")
    print(f"  准确率: {results['metrics']['accuracy']:.2%}")
    print(f"  精确率: {results['metrics']['precision']:.2%}")
    print(f"  查全率: {results['metrics']['recall']:.2%}")
    print(f"  F1 分数: {results['metrics']['f1']:.2%}")
    print(f"  AUC-ROC: {results['roc_auc']:.4f}")
    print(f"  交叉验证 F1: {results['cv_f1_mean']:.2%} ± {results['cv_f1_std']:.2%}")
    
    # 生成报告
    print("\n生成 Markdown 报告...")
    report = generate_classification_report(results)
    
    # 保存报告
    output_dir = Path(__file__).parent.parent / 'output'
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / 'classification_report.md'
    report_path.write_text(report, encoding='utf-8')
    print(f"报告已保存到: {report_path}")
    
    # 绘制 ROC 曲线
    print("\n绘制 ROC 曲线...")
    fig_roc = plot_roc_curve(results)
    
    images_dir = Path(__file__).parent.parent / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)
    fig_roc.savefig(images_dir / '10_roc_curve.png',
                    dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
    plt.close(fig_roc)
    print(f"ROC 曲线已保存到: images/10_roc_curve.png")
    
    # 绘制综合评估图
    print("\n绘制综合评估图...")
    fig_comprehensive = plot_classification_results(results)
    fig_comprehensive.savefig(images_dir / '10_classification_overview.png',
                              dpi=150, bbox_inches='tight',
                              facecolor='white', edgecolor='none')
    plt.close(fig_comprehensive)
    print(f"综合评估图已保存到: images/10_classification_overview.png")
    
    # 与上周的对比
    print("\n" + "=" * 70)
    print("StatLab 进度：本周 vs 上周")
    print("=" * 70)
    
    print("\n上周（Week 09）：")
    print("  - 回归分析（系数、R²、F 检验）")
    print("  - 假设检验（LINE 假设）")
    print("  - 模型诊断（残差图、QQ 图、Cook's 距离）")
    print("  - 多重共线性检查（VIF）")
    print("  → 量化了'关系' + 验证了'模型可信性'")
    
    print("\n本周（Week 10）：")
    print("  - 逻辑回归（sigmoid、概率预测）")
    print("  - 混淆矩阵（TP/FP/TN/FN）")
    print("  - 分类指标（精确率、查全率、F1）")
    print("  - ROC 曲线与 AUC")
    print("  - 类别不平衡检测与处理")
    print("  → 从'预测连续值'扩展到'预测离散类别'")
    
    print("\n老潘的点评：")
    print("  '上周你学会了说：广告投入每增加 1 万，销售增加 0.5'")
    print("  '                [95% CI: 0.4, 0.6], p < 0.001'")
    print("  '本周你学会了说：该客户有 85% 的概率是高价值客户'")
    print("  '                查全率 78%，精确率 72%，AUC = 0.85'")
    print("  '这才是完整的预测分析。'")
    
    print("\n" + "=" * 70)
    print("函数使用说明")
    print("=" * 70)
    print("""
# 在你的分析脚本中使用 StatLab 分类分析：

from chapters.week_10.examples.statlab_classification import (
    classification_with_evaluation,
    plot_roc_curve,
    generate_classification_report
)

# 1. 准备数据
X = df[['特征1', '特征2', '特征3']]
y = df['目标变量']  # 0/1

# 2. 运行分类分析
results = classification_with_evaluation(X, y, var_names=['特征1', '特征2', '特征3'])

# 3. 生成 Markdown 报告
report = generate_classification_report(results)
print(report)

# 4. 绘制 ROC 曲线
fig = plot_roc_curve(results)
fig.savefig('roc_curve.png')
    """)
    
    print("\n" + "=" * 70)
    print("完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()

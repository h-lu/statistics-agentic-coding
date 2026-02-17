"""
示例：StatLab 分类评估流水线——完整的分类模型评估报告

运行方式：python3 chapters/week_10/examples/10_statlab_classification.py
预期输出：生成完整的分类评估报告（Markdown + 图表），输出到 output/

这是 Week 10 的 StatLab 超级线代码，在上周回归分析的基础上增量添加分类评估模块。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, roc_auc_score,
                             classification_report)
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
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


def classification_with_pipeline(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    numeric_features: list[str],
    categorical_features: list[str],
    test_size: float = 0.3,
    random_state: int = 42
) -> dict:
    """
    用 Pipeline 训练分类模型并输出评估报告（防止数据泄漏）

    参数:
    - X: 特征 DataFrame
    - y: 目标变量（Series 或 array）
    - numeric_features: 数值型特征列表
    - categorical_features: 分类型特征列表
    - test_size: 测试集比例（默认 0.3）
    - random_state: 随机种子

    返回:
    - dict: 包含模型、评估指标、图表数据的字典
    """
    # 1. 定义预处理器
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # 2. 创建完整 Pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', LogisticRegression(max_iter=1000, random_state=random_state))
    ])

    # 3. 划分数据（分层采样）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 4. 训练
    pipeline.fit(X_train, y_train)

    # 5. 预测
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # 6. 计算评估指标
    results = {
        'pipeline': pipeline,
        'metrics': {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc': roc_auc_score(y_test, y_prob)
        },
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'y_test': y_test,
        'y_pred': y_pred,
        'y_prob': y_prob
    }

    # 7. ROC 曲线数据
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    results['roc_curve'] = {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds}

    # 8. 交叉验证 AUC（防止泄漏）
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='roc_auc')
    results['cv_scores'] = {
        'mean': cv_scores.mean(),
        'std': cv_scores.std(),
        'scores': cv_scores.tolist()
    }

    # 9. 特征重要性（逻辑回归系数）
    # 获取特征名称
    feature_names = numeric_features.copy()

    # 获取 One-Hot 编码后的特征名
    preprocessor_obj = pipeline.named_steps['preprocessor']
    cat_transformer = preprocessor_obj.named_transformers_['cat']
    cat_encoder = cat_transformer.named_steps['onehot']

    # 获取所有分类特征
    for i, cat_feat in enumerate(categorical_features):
        # 获取该特征的类别
        categories = cat_encoder.categories_[i]
        # OneHotEncoder 没有使用 drop_first，所以包含所有类别
        for cat in categories:
            feature_names.append(f"{cat_feat}_{cat}")

    # 获取系数
    coefficients = pipeline.named_steps['model'].coef_[0]
    results['feature_importance'] = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients
    }).sort_values('coefficient', key=abs, ascending=False).reset_index(drop=True)

    return results


def plot_classification_results(results: dict, output_dir: Path) -> None:
    """画分类评估图表并保存"""
    font = setup_chinese_font()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. 混淆矩阵
    cm = results['confusion_matrix']
    im = axes[0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    axes[0].set_title('混淆矩阵')
    axes[0].set_xlabel('预测标签')
    axes[0].set_ylabel('实际标签')
    axes[0].set_xticks([0, 1])
    axes[0].set_yticks([0, 1])
    axes[0].set_xticklabels(['负类 (0)', '正类 (1)'])
    axes[0].set_yticklabels(['负类 (0)', '正类 (1)'])

    # 添加数字
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = f'{cm[i, j]}'
            if i == 0 and j == 0:
                text += '\n(TN)'
            elif i == 0 and j == 1:
                text += '\n(FP)'
            elif i == 1 and j == 0:
                text += '\n(FN)'
            else:
                text += '\n(TP)'
            axes[0].text(j, i, text,
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=12)

    # 2. ROC 曲线
    roc = results['roc_curve']
    axes[1].plot(roc['fpr'], roc['tpr'], linewidth=2,
                label=f"ROC Curve (AUC = {results['metrics']['auc']:.4f})")
    axes[1].plot([0, 1], [0, 1], 'k--', label='随机猜测', linewidth=1)
    axes[1].set_xlabel('假正率 FPR (1 - 特异度)')
    axes[1].set_ylabel('真正率 TPR (召回率)')
    axes[1].set_title('ROC 曲线')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'classification_evaluation.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"图表已保存到: {output_dir / 'classification_evaluation.png'}")


def format_classification_report(results: dict) -> str:
    """格式化分类结果为 Markdown 报告"""
    md = ["## 分类模型评估\n\n"]

    # 1. 评估指标
    md.append("### 评估指标\n\n")
    metrics = results['metrics']
    md.append(f"| 指标 | 值 |\n")
    md.append(f"|------|-----|\n")
    md.append(f"| 准确率 | {metrics['accuracy']:.4f} |\n")
    md.append(f"| 精确率 | {metrics['precision']:.4f} |\n")
    md.append(f"| 召回率 | {metrics['recall']:.4f} |\n")
    md.append(f"| F1 分数 | {metrics['f1']:.4f} |\n")
    md.append(f"| AUC | {metrics['auc']:.4f} |\n\n")

    # 2. 交叉验证 AUC
    cv = results['cv_scores']
    md.append("### 交叉验证 AUC（防止数据泄漏）\n\n")
    md.append(f"- 平均 AUC: {cv['mean']:.4f} (+/- {cv['std']:.4f})\n")
    md.append(f"- 各折 AUC: {', '.join([f'{s:.4f}' for s in cv['scores']])}\n\n")
    md.append("**说明**: 使用 Pipeline + 交叉验证确保评估没有数据泄漏。每个模型的预处理都是独立的，只用训练集的统计量。\n\n")

    # 3. 混淆矩阵
    md.append("### 混淆矩阵\n\n")
    cm = results['confusion_matrix']
    md.append(f"| | 预测负类 | 预测正类 |\n")
    md.append(f"|---|---------|---------|\n")
    md.append(f"| **实际负类** | {cm[0, 0]} (TN) | {cm[0, 1]} (FP) |\n")
    md.append(f"| **实际正类** | {cm[1, 0]} (FN) | {cm[1, 1]} (TP) |\n\n")
    md.append("**术语解释**:\n")
    md.append(f"- TN (True Negative): 正确拒绝的负类\n")
    md.append(f"- FP (False Positive): 误判为正类的负类（误判）\n")
    md.append(f"- FN (False Negative): 漏掉的正类（漏判）\n")
    md.append(f"- TP (True Positive): 正确抓到的正类\n\n")

    # 4. 特征重要性（Top 10）
    md.append("### 特征重要性（逻辑回归系数绝对值 Top 10）\n\n")
    top_features = results['feature_importance'].head(10)
    md.append(f"| 排名 | 特征 | 系数 | 方向 |\n")
    md.append(f"|------|------|------|------|\n")
    for idx, row in top_features.iterrows():
        direction = "提高正类概率" if row['coefficient'] > 0 else "降低正类概率"
        md.append(f"| {idx + 1} | {row['feature']} | {row['coefficient']:.4f} | {direction} |\n")
    md.append("\n")

    # 5. 指标选择理由
    md.append("### 指标选择理由\n\n")
    md.append("**为什么选择这些评估指标？**\n\n")

    # 类别比例
    y_test = results['y_test']
    pos_ratio = y_test.mean() if isinstance(y_test, pd.Series) else y_test.mean()
    if pos_ratio < 0.3 or pos_ratio > 0.7:
        md.append(f"- **类别不平衡**: 正类占比 {pos_ratio:.1%}，准确率可能误导。优先参考精确率、召回率、F1 和 AUC。\n")
    else:
        md.append(f"- **类别相对平衡**: 正类占比 {pos_ratio:.1%}，准确率可用，但仍需参考其他指标。\n")

    # 业务场景
    md.append("\n**根据业务场景选择优先指标**:\n")
    md.append("- 如果优先'抓到所有正类'（如流失预测、医疗诊断）：关注**召回率**\n")
    md.append("- 如果优先'减少误判'（如欺诈检测、垃圾邮件过滤）：关注**精确率**\n")
    md.append("- 如果需要平衡：关注 **F1 分数**\n")
    md.append("- 如果需要排序能力（如推荐系统）：关注 **AUC**\n\n")

    # 6. 数据泄漏防护说明
    md.append("### 数据泄漏防护\n\n")
    md.append("**本评估已采取的防泄漏措施**:\n\n")
    md.append("1. **Pipeline 封装**: 预处理和模型绑定，确保测试集只用训练集的统计量\n")
    md.append("2. **交叉验证**: 使用 5 折交叉验证，每折的预处理都是独立的\n")
    md.append("3. **分层采样**: 划分训练/测试集时保持类别比例一致\n")
    md.append("4. **评估指标**: 使用 AUC（不依赖单一阈值）和交叉验证 AUC（防止过拟合）\n\n")
    md.append("**黄金法则**: 永远不要在测试数据上调用 fit()。Pipeline 自动确保这一点。\n\n")

    return "".join(md)


def main() -> None:
    """主函数：运行 StatLab 分类评估流水线"""
    print("=" * 60)
    print("StatLab 分类评估流水线")
    print("=" * 60)

    # 创建输出目录
    output_dir = Path(__file__).parent.parent.parent.parent / 'output'
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据（使用泰坦尼克数据集作为示例）
    print("\n加载数据...")
    titanic = sns.load_dataset("titanic")

    # 准备特征和目标
    feature_cols = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
    X = titanic[feature_cols].copy()
    y = titanic['survived']

    # 定义特征类型
    numeric_features = ['age', 'sibsp', 'parch', 'fare']
    categorical_features = ['pclass', 'sex', 'embarked']

    print(f"数据集规模: {X.shape[0]} 行, {X.shape[1]} 列")
    print(f"数值型特征: {numeric_features}")
    print(f"分类型特征: {categorical_features}")
    print(f"目标变量: survived (0=未生存, 1=生存)")
    print(f"类别分布: 0: {(y==0).sum()}, 1: {(y==1).sum()}")

    # 训练并评估
    print("\n训练分类模型...")
    results = classification_with_pipeline(X, y, numeric_features, categorical_features)

    # 打印评估指标
    print("\n评估指标:")
    for name, value in results['metrics'].items():
        print(f"  {name}: {value:.4f}")

    print(f"\n交叉验证 AUC: {results['cv_scores']['mean']:.4f} (+/- {results['cv_scores']['std']:.4f})")

    # 生成图表
    print("\n生成图表...")
    plot_classification_results(results, output_dir)

    # 生成报告
    print("生成报告...")
    report = format_classification_report(results)

    # 保存报告
    report_path = output_dir / 'classification_report.md'
    report_path.write_text(report, encoding='utf-8')
    print(f"\n报告已保存到: {report_path}")

    # 打印报告预览
    print("\n" + "=" * 60)
    print("报告预览")
    print("=" * 60)
    print(report[:1000] + "...\n")

    print("=" * 60)
    print("StatLab 分类评估流水线完成！")
    print("=" * 60)
    print("\n生成的文件:")
    print(f"  1. {output_dir / 'classification_evaluation.png'} - 评估图表")
    print(f"  2. {output_dir / 'classification_report.md'} - 评估报告")
    print("\n本周 StatLab 进展:")
    print("  - 添加了分类模型评估模块")
    print("  - 使用 Pipeline 防止数据泄漏")
    print("  - 输出混淆矩阵、精确率/召回率/F1、ROC-AUC")
    print("  - 包含指标选择理由和数据泄漏防护说明")


if __name__ == "__main__":
    main()

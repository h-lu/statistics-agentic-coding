"""
示例：StatLab 可解释性与伦理报告生成——完整的模型责任评估流水线

运行方式：python3 chapters/week_12/examples/12_statlab_interpretability.py
预期输出：生成完整的可解释性与伦理报告（Markdown + 图表），输出到 output/

这是 Week 12 的 StatLab 超级线代码，在 Week 11 树模型对比的基础上增量添加：
- SHAP 可解释性模块（局部解释）
- 公平性评估模块（分组指标）
- 面向非技术读者的模型解释报告
- 伦理风险清单

与上周对比：
- Week 11: 模型预测力评估（AUC、特征重要性）
- Week 12: 模型责任评估（SHAP、公平性、伦理风险）
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_auc_score)
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

# 尝试导入 shap
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("警告: shap 库未安装，SHAP 模块将被跳过。请运行: pip install shap")


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


def load_data() -> tuple:
    """
    加载数据（复用 Week 11 的泰坦尼克数据集）

    为了演示公平性评估，我们添加一个敏感属性：sex（性别）
    """
    print("加载数据...")
    titanic = sns.load_dataset("titanic")

    # 准备特征和目标
    feature_cols = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'class']
    X = titanic[feature_cols].copy()

    # 处理重复特征（class 和 pclass 重复）
    X = X.drop('class', axis=1)

    y = titanic['survived']

    # 定义特征类型
    numeric_features = ['age', 'sibsp', 'parch', 'fare']
    categorical_features = ['pclass', 'sex', 'embarked']

    return X, y, numeric_features, categorical_features


def train_models_with_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str]
) -> dict:
    """
    训练多个模型（复用 Week 11 的 Pipeline 结构）
    """
    print("\n训练模型...")

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

    # 2. 划分数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"  训练集: {X_train.shape[0]} 行")
    print(f"  测试集: {X_test.shape[0]} 行")

    # 3. 定义模型
    models = {}

    # 3.1 随机森林（选择最佳模型）
    models['random_forest'] = Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            max_features='sqrt',
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        ))
    ])

    # 4. 训练
    results = {}

    for name, pipeline in models.items():
        print(f"  训练 {name}...")
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        results[name] = {
            'pipeline': pipeline,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_prob': y_prob,
            'metrics': {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'auc': roc_auc_score(y_test, y_prob)
            }
        }

        print(f"    测试集 AUC: {results[name]['metrics']['auc']:.4f}")

    # 添加原始 X_test（用于公平性评估）
    results['random_forest']['X_test_original'] = X.loc[X_test.index]

    return results


def compute_shap_values(
    model_results: dict,
    output_dir: Path
) -> dict:
    """
    计算 SHAP 值并生成可视化
    """
    if not SHAP_AVAILABLE:
        print("  跳过 SHAP 模块（shap 未安装）")
        return {}

    print("\n计算 SHAP 值...")

    rf_pipeline = model_results['random_forest']['pipeline']
    X_test = model_results['random_forest']['X_test']

    # 获取预处理后的特征
    preprocessor = rf_pipeline.named_steps['preprocessor']
    X_test_processed = preprocessor.transform(X_test)

    # 获取特征名称
    feature_names = []
    for name, transformer, features in preprocessor.transformers_:
        if name == 'num':
            feature_names.extend(features)
        elif name == 'cat':
            encoder = transformer.named_steps['onehot']
            for i, feat in enumerate(features):
                categories = encoder.categories_[i]
                feature_names.extend([f"{feat}_{cat}" for cat in categories])

    # 训练 SHAP 解释器
    explainer = shap.TreeExplainer(rf_pipeline.named_steps['model'])
    shap_values = explainer.shap_values(X_test_processed)

    # shap_values 可能是嵌套列表（二分类）
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    expected_value = explainer.expected_value
    if isinstance(expected_value, list):
        expected_value = expected_value[1]

    # 绘制 SHAP 汇总图
    font = setup_chinese_font()
    fig = plt.figure(figsize=(10, 6))

    # 使用简化的可视化
    shap_summary = pd.DataFrame({
        'feature': feature_names,
        'importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('importance', ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(shap_summary)), shap_summary['importance'].values,
            color='steelblue', alpha=0.7)
    ax.set_yticks(range(len(shap_summary)))
    ax.set_yticklabels(shap_summary['feature'].values)
    ax.set_xlabel('平均 |SHAP 值|')
    ax.set_title('SHAP 特征重要性（全局）')
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_dir / 'shap_feature_importance.png',
                dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

    print(f"  SHAP 特征重要性图已保存")

    # 单样本解释
    sample_idx = 0
    single_explanation = {
        'sample_idx': sample_idx,
        'prediction': model_results['random_forest']['y_prob'][sample_idx],
        'top_features': []
    }

    sample_shap = shap_values[sample_idx]
    top_idx = np.argsort(np.abs(sample_shap))[-3:][::-1]

    for idx in top_idx:
        single_explanation['top_features'].append({
            'feature': feature_names[idx],
            'shap_value': sample_shap[idx]
        })

    return {
        'explainer': explainer,
        'shap_values': shap_values,
        'expected_value': expected_value,
        'feature_names': feature_names,
        'single_explanation': single_explanation
    }


def compute_fairness_metrics(
    model_results: dict,
    output_dir: Path
) -> dict:
    """
    计算公平性指标（按性别分组）
    """
    print("\n计算公平性指标...")

    y_test = model_results['random_forest']['y_test']
    y_pred = model_results['random_forest']['y_pred']
    y_prob = model_results['random_forest']['y_prob']
    X_test_original = model_results['random_forest']['X_test_original']

    # 按性别分组
    gender_results = {}

    for gender in X_test_original['sex'].unique():
        mask = X_test_original['sex'] == gender

        if mask.sum() < 10:
            continue

        y_true_g = y_test[mask]
        y_pred_g = y_pred[mask]
        y_prob_g = y_prob[mask]

        cm = confusion_matrix(y_true_g, y_pred_g)
        tn, fp, fn, tp = cm.ravel()

        gender_results[gender] = {
            'count': mask.sum(),
            'true_positive_rate': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'positive_rate': y_pred_g.mean(),
            'true_survival_rate': y_true_g.mean(),
            'avg_predicted_prob': y_prob_g.mean(),
            'accuracy': accuracy_score(y_true_g, y_pred_g)
        }

    # 计算差异
    if len(gender_results) >= 2:
        genders = list(gender_results.keys())
        g0, g1 = gender_results[genders[0]], gender_results[genders[1]]

        differences = {
            'tpr_diff': abs(g1['true_positive_rate'] - g0['true_positive_rate']),
            'fpr_diff': abs(g1['false_positive_rate'] - g0['false_positive_rate']),
            'positive_rate_diff': abs(g1['positive_rate'] - g0['positive_rate'])
        }
    else:
        differences = {}

    # 绘制分组混淆矩阵
    font = setup_chinese_font()
    fig, axes = plt.subplots(1, len(gender_results), figsize=(5 * len(gender_results), 4))
    if len(gender_results) == 1:
        axes = [axes]

    for ax, (gender, metrics) in zip(axes, gender_results.items()):
        cm = confusion_matrix(
            y_test[X_test_original['sex'] == gender],
            y_pred[X_test_original['sex'] == gender]
        )

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['未生还', '生还'],
                   yticklabels=['未生还', '生还'])
        ax.set_title(f'{gender} 组混淆矩阵 (n={metrics["count"]})')
        ax.set_xlabel('预测')
        ax.set_ylabel('真实')

    plt.tight_layout()
    plt.savefig(output_dir / 'group_confusion_matrices.png',
                dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

    print(f"  分组混淆矩阵图已保存")

    return {
        'gender_results': gender_results,
        'differences': differences
    }


def generate_non_technical_summary(
    model_results: dict,
    shap_results: dict,
    fairness_results: dict
) -> str:
    """
    生成面向非技术读者的模型总结
    """
    md = ["## 模型结论与行动建议（面向非技术读者）\n\n"]

    # 1. 一句话总结
    auc = model_results['random_forest']['metrics']['auc']
    md.append(f"**模型性能**: 该模型能正确识别约 {auc*100:.0f}% 的乘客生还状态。\n\n")

    # 2. 关键风险信号
    if shap_results and 'single_explanation' in shap_results:
        md.append("**主要影响因素**:\n\n")
        for feat in shap_results['single_explanation']['top_features']:
            direction = "增加" if feat['shap_value'] > 0 else "降低"
            md.append(f"- **{feat['feature']}**: {direction}生还概率\n")
        md.append("\n")

    # 3. 公平性说明
    if fairness_results and 'differences' in fairness_results:
        md.append("**公平性说明**:\n\n")

        diffs = fairness_results['differences']
        if diffs.get('tpr_diff', 0) < 0.1 and diffs.get('fpr_diff', 0) < 0.1:
            md.append("- 模型对不同性别的预测表现差异在可接受范围内\n")
        else:
            md.append("- 模型对不同性别的预测表现存在一定差异\n")
            md.append(f"- 真阳性率差异: {diffs.get('tpr_diff', 0):.1%}\n")
            md.append(f"- 假阳性率差异: {diffs.get('fpr_diff', 0):.1%}\n")
        md.append("\n")

    # 4. 行动建议
    md.append("**行动建议**:\n\n")
    md.append("- 该模型适用于：生还概率评估、救援资源分配决策支持\n")
    md.append("- 不适用于：直接决定个体生死（需要人工复核）\n")
    md.append("- 建议：定期重新训练，确保模型适应新数据\n\n")

    # 5. 模型边界
    md.append("**模型边界**:\n\n")
    md.append("- 模型基于泰坦尼克号历史数据，不能直接应用于其他场景\n")
    md.append("- 对于数据中样本量较少的群体，预测不确定性较高\n")
    md.append("- 建议每季度重新评估模型性能\n\n")

    return "".join(md)


def generate_ethics_risk_checklist(
    fairness_results: dict
) -> str:
    """
    生成伦理风险清单
    """
    md = ["## 伦理风险清单\n\n"]
    md.append("| 风险类型 | 风险等级 | 缓解措施 |\n")
    md.append("|---------|---------|----------|\n")

    # 根据公平性评估结果调整风险等级
    if fairness_results and 'differences' in fairness_results:
        diffs = fairness_results['differences']
        if diffs.get('tpr_diff', 0) > 0.15 or diffs.get('fpr_diff', 0) > 0.15:
            allocation_fairness = "高"
            mitigation = "检测到显著分组差异，建议调整决策阈值或使用后处理校准"
        else:
            allocation_fairness = "低"
            mitigation = f"分组真阳性率差异 {diffs.get('tpr_diff', 0):.1%}，在可接受范围内"
    else:
        allocation_fairness = "中"
        mitigation = "建议定期审计分组指标"

    md.append(f"| 分配不公 | {allocation_fairness} | {mitigation} |\n")
    md.append("| 数据偏见 | 中 | 定期审计数据来源，检查历史偏见 |\n")
    md.append("| 隐私风险 | 中 | 数据匿名化，限制访问权限 |\n")
    md.append("| 模型边界 | 中 | 明确有效期，定期重新训练 |\n")
    md.append("| 过度依赖 | 高 | 模型仅作为决策支持，不替代人工判断 |\n\n")

    return "".join(md)


def generate_full_report(
    model_results: dict,
    shap_results: dict,
    fairness_results: dict,
    output_dir: Path
) -> str:
    """
    生成完整的可解释性与伦理报告
    """
    md = ["# 模型可解释性与伦理评估报告\n\n"]
    md.append(f"**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    # 1. 模型性能摘要
    md.append("## 模型性能摘要\n\n")
    metrics = model_results['random_forest']['metrics']

    md.append("| 指标 | 值 |\n")
    md.append("|------|-----|\n")
    md.append(f"| 准确率 | {metrics['accuracy']:.4f} |\n")
    md.append(f"| 精确率 | {metrics['precision']:.4f} |\n")
    md.append(f"| 召回率 | {metrics['recall']:.4f} |\n")
    md.append(f"| F1 分数 | {metrics['f1']:.4f} |\n")
    md.append(f"| AUC | {metrics['auc']:.4f} |\n\n")

    # 2. SHAP 可解释性
    if shap_results:
        md.append("## 模型可解释性\n\n")
        md.append(f"![](shap_feature_importance.png)\n\n")

        if 'single_explanation' in shap_results:
            expl = shap_results['single_explanation']
            md.append("### 单样本预测解释\n\n")
            md.append(f"**样本 #{expl['sample_idx']}**\n\n")
            md.append(f"预测生还概率: {expl['prediction']:.1%}\n\n")
            md.append("**主要影响因素**:\n\n")
            for feat in expl['top_features']:
                md.append(f"- {feat['feature']}: SHAP 值 = {feat['shap_value']:.4f}\n")
            md.append("\n")

    # 3. 公平性评估
    if fairness_results:
        md.append("## 公平性评估\n\n")

        if 'gender_results' in fairness_results:
            md.append("### 按性别分组\n\n")
            md.append("| 分组 | 样本数 | 真阳性率 | 假阳性率 | 预测正率 | 真实生还率 |\n")
            md.append("|------|--------|---------|---------|---------|-----------|\n")

            for gender, metrics in fairness_results['gender_results'].items():
                md.append(f"| {gender} | {metrics['count']} | "
                         f"{metrics['true_positive_rate']:.3f} | "
                         f"{metrics['false_positive_rate']:.3f} | "
                         f"{metrics['positive_rate']:.3f} | "
                         f"{metrics['true_survival_rate']:.3f} |\n")

            md.append(f"![](group_confusion_matrices.png)\n\n")

    # 4. 非技术读者总结
    md.append(generate_non_technical_summary(model_results, shap_results, fairness_results))

    # 5. 伦理风险清单
    md.append(generate_ethics_risk_checklist(fairness_results))

    return "".join(md)


def main() -> None:
    """主函数：运行 StatLab 可解释性与伦理报告生成流水线"""
    print("=" * 60)
    print("StatLab 可解释性与伦理报告生成流水线")
    print("=" * 60)

    # 创建输出目录
    output_dir = Path(__file__).parent.parent.parent.parent / 'output'
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载数据
    X, y, numeric_features, categorical_features = load_data()

    # 2. 训练模型
    model_results = train_models_with_pipeline(X, y, numeric_features, categorical_features)

    # 3. 计算 SHAP 值
    shap_results = compute_shap_values(model_results, output_dir)

    # 4. 计算公平性指标
    fairness_results = compute_fairness_metrics(model_results, output_dir)

    # 5. 生成报告
    print("\n生成报告...")
    report = generate_full_report(model_results, shap_results, fairness_results, output_dir)

    # 保存报告
    report_path = output_dir / 'interpretability_ethics_report.md'
    report_path.write_text(report, encoding='utf-8')

    print(f"\n报告已保存到: {report_path}")

    # 6. 打印摘要
    print("\n" + "=" * 60)
    print("报告摘要")
    print("=" * 60)

    print("\n模型性能:")
    metrics = model_results['random_forest']['metrics']
    print(f"  AUC: {metrics['auc']:.4f}")
    print(f"  准确率: {metrics['accuracy']:.4f}")

    if fairness_results and 'differences' in fairness_results:
        print("\n公平性评估:")
        diffs = fairness_results['differences']
        print(f"  真阳性率差异: {diffs.get('tpr_diff', 0):.3f}")
        print(f"  假阳性率差异: {diffs.get('fpr_diff', 0):.3f}")

        if diffs.get('tpr_diff', 0) > 0.1 or diffs.get('fpr_diff', 0) > 0.1:
            print("  警告: 检测到显著的分组差异")
        else:
            print("  分组差异在可接受范围内")

    print("\n" + "=" * 60)
    print("StatLab 可解释性与伦理报告生成完成！")
    print("=" * 60)

    print("\n生成的文件:")
    if shap_results:
        print(f"  1. {output_dir / 'shap_feature_importance.png'} - SHAP 特征重要性图")
    print(f"  2. {output_dir / 'group_confusion_matrices.png'} - 分组混淆矩阵图")
    print(f"  3. {output_dir / 'interpretability_ethics_report.md'} - 完整报告")

    print("\n本周 StatLab 进展:")
    print("  - 添加了 SHAP 可解释性模块")
    print("  - 实现了按敏感属性分组的公平性评估")
    print("  - 生成了面向非技术读者的模型解释报告")
    print("  - 包含了伦理风险清单")

    print("\n与 Week 11 的对比:")
    print("  - 上周: 模型预测力评估（AUC、特征重要性）")
    print("  - 本周: 模型责任评估（SHAP、公平性、伦理风险）")
    print("  - 新增: 局部可解释性、分组评估、非技术沟通")


if __name__ == "__main__":
    main()

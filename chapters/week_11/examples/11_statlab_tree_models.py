"""
示例：StatLab 树模型报告生成——完整的模型对比与基线评估流水线

运行方式：python3 chapters/week_11/examples/11_statlab_tree_models.py
预期输出：生成完整的模型对比报告（Markdown + 图表），输出到 output/

这是 Week 11 的 StatLab 超级线代码，在 Week 10 分类评估的基础上增量添加：
- 树模型模块（决策树、随机森林）
- 多模型对比与基线评估
- 模型选择理由与权衡分析
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
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_auc_score)
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


def baseline_comparison_with_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
    test_size: float = 0.3,
    random_state: int = 42
) -> dict:
    """
    训练多个模型并与基线对比（使用 Pipeline 防止数据泄漏）

    参数:
    - X: 特征 DataFrame
    - y: 目标变量
    - numeric_features: 数值型特征列表
    - categorical_features: 分类型特征列表
    - test_size: 测试集比例
    - random_state: 随机种子

    返回:
    - dict: 包含所有模型评估结果的字典
    """
    print("=" * 60)
    print("StatLab 树模型对比流水线")
    print("=" * 60)

    # 1. 定义预处理器（复用 Week 10 的 Pipeline 结构）
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

    # 2. 划分数据（分层采样）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"\n数据集规模: {X.shape[0]} 行, {X.shape[1]} 列")
    print(f"训练集: {X_train.shape[0]} 行")
    print(f"测试集: {X_test.shape[0]} 行")
    print(f"正类占比: {y.mean():.2%}")

    # 3. 定义模型（都用同一个 Pipeline）
    models = {}

    # 3.1 傻瓜基线
    models['dummy'] = Pipeline([
        ('preprocessor', preprocessor),
        ('model', DummyClassifier(strategy='most_frequent'))
    ])

    # 3.2 逻辑回归基线
    models['logistic_regression'] = Pipeline([
        ('preprocessor', preprocessor),
        ('model', LogisticRegression(max_iter=1000, random_state=random_state))
    ])

    # 3.3 决策树
    models['decision_tree'] = Pipeline([
        ('preprocessor', preprocessor),
        ('model', DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=random_state
        ))
    ])

    # 3.4 随机森林
    models['random_forest'] = Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            max_features='sqrt',
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=random_state,
            n_jobs=-1
        ))
    ])

    # 4. 训练和评估
    results = {}

    for name, pipeline in models.items():
        print(f"\n训练 {name}...")

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        results[name] = {
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

        # 交叉验证 AUC（5 折）
        cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='roc_auc')
        results[name]['cv_scores'] = {
            'mean': cv_scores.mean(),
            'std': cv_scores.std(),
            'scores': cv_scores.tolist()
        }

        print(f"  测试集 AUC: {results[name]['metrics']['auc']:.4f}")
        print(f"  交叉验证 AUC: {results[name]['cv_scores']['mean']:.4f} (+/- {results[name]['cv_scores']['std']:.4f})")

    # 5. 获取特征重要性（随机森林）
    rf_pipeline = models['random_forest']

    # 获取特征名称（数值型 + One-Hot 后的分类型）
    feature_names = numeric_features.copy()

    preprocessor_obj = rf_pipeline.named_steps['preprocessor']
    cat_transformer = preprocessor_obj.named_transformers_['cat']
    cat_encoder = cat_transformer.named_steps['onehot']

    for i, cat_feat in enumerate(categorical_features):
        categories = cat_encoder.categories_[i]
        feature_names.extend([f"{cat_feat}_{cat}" for cat in categories])

    # 获取特征重要性
    importances = rf_pipeline.named_steps['model'].feature_importances_
    results['feature_importance'] = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).reset_index(drop=True)

    return results


def format_model_comparison_report(results: dict) -> str:
    """格式化模型对比结果为 Markdown 报告"""
    md = ["## 模型对比与选择\n\n"]

    # 1. 模型对比表
    md.append("### 评估指标对比\n\n")
    md.append(f"| 模型 | 准确率 | 精确率 | 召回率 | F1 | AUC | 交叉验证 AUC |\n")
    md.append(f"|------|--------|--------|--------|-----|-----|-------------|\n")

    for name in ['dummy', 'logistic_regression', 'decision_tree', 'random_forest']:
        res = results[name]
        metrics = res['metrics']
        cv = res['cv_scores']
        name_display = {
            'dummy': '傻瓜基线',
            'logistic_regression': '逻辑回归',
            'decision_tree': '决策树',
            'random_forest': '随机森林'
        }[name]

        md.append(f"| {name_display} | {metrics['accuracy']:.4f} | "
                  f"{metrics['precision']:.4f} | {metrics['recall']:.4f} | "
                  f"{metrics['f1']:.4f} | {metrics['auc']:.4f} | "
                  f"{cv['mean']:.4f} (+/- {cv['std']:.4f}) |\n")

    md.append("\n")

    # 2. 基线对比
    md.append("### 基线对比\n\n")
    dummy_auc = results['dummy']['metrics']['auc']
    lr_auc = results['logistic_regression']['metrics']['auc']
    tree_auc = results['decision_tree']['metrics']['auc']
    rf_auc = results['random_forest']['metrics']['auc']

    md.append(f"- **傻瓜基线 AUC**: {dummy_auc:.4f}\n")
    md.append(f"- **逻辑回归基线 AUC**: {lr_auc:.4f}\n")
    md.append(f"- **决策树 AUC**: {tree_auc:.4f}\n")
    md.append(f"- **随机森林 AUC**: {rf_auc:.4f}\n\n")

    md.append(f"**提升量**:\n")
    md.append(f"- 逻辑回归 vs 傻瓜基线: +{(lr_auc - dummy_auc):.4f}\n")
    md.append(f"- 决策树 vs 逻辑回归: +{(tree_auc - lr_auc):.4f} ({(tree_auc - lr_auc)/lr_auc*100:.1f}%)\n")
    md.append(f"- 随机森林 vs 逻辑回归: +{(rf_auc - lr_auc):.4f} ({(rf_auc - lr_auc)/lr_auc*100:.1f}%)\n\n")

    # 3. 特征重要性（Top 10）
    if 'feature_importance' in results:
        md.append("### 特征重要性（随机森林，Top 10）\n\n")
        top_features = results['feature_importance'].head(10)
        md.append(f"| 排名 | 特征 | 重要性 |\n")
        md.append(f"|------|------|--------|\n")
        for idx, row in top_features.iterrows():
            md.append(f"| {idx + 1} | {row['feature']} | {row['importance']:.4f} |\n")
        md.append("\n")

    # 4. 模型选择理由
    md.append("### 模型选择理由\n\n")

    md.append("**基线对比结论**:\n\n")
    md.append(f"- 所有模型的 AUC 都显著高于傻瓜基线（{dummy_auc:.4f}），说明模型比瞎猜好\n\n")

    md.append("**复杂度 vs 提升量权衡**:\n\n")
    improvement = (rf_auc - lr_auc) / lr_auc * 100

    if improvement < 2:
        md.append(f"- 随机森林比逻辑回归提升 {improvement:.1f}%，提升量较小\n")
        md.append("- 如果业务最关心预测力，选随机森林；如果需要向业务方解释规则，选逻辑回归\n\n")
    elif improvement < 5:
        md.append(f"- 随机森林比逻辑回归提升 {improvement:.1f}%，提升量中等\n")
        md.append("- 如果预测力是关键，选随机森林；如果需要可解释性，选逻辑回归\n\n")
    else:
        md.append(f"- 随机森林比逻辑回归提升 {improvement:.1f}%，提升量显著\n")
        md.append("- 建议选择随机森林\n\n")

    md.append("**可解释性考虑**:\n\n")
    md.append("- **逻辑回归**: 高可解释性，可以解读每个特征的系数方向和强度\n")
    md.append("- **决策树**: 高可解释性，可以画出树结构，直观展示决策规则\n")
    md.append("- **随机森林**: 中等可解释性，可以输出特征重要性，但无法画出完整决策过程\n\n")

    # 5. 场景推荐
    md.append("### 场景推荐\n\n")
    md.append("| 场景 | 优先级 | 推荐模型 |\n")
    md.append("|------|--------|----------|\n")
    md.append("| **需要向业务方解释** | 可解释性 | 逻辑回归 或 决策树 |\n")
    md.append("| **追求最高预测力** | AUC | 随机森林 |\n")
    md.append("| **实时预测（低延迟）** | 预测时间 | 逻辑回归 |\n")
    md.append("| **训练资源有限** | 训练时间 | 逻辑回归 或 决策树 |\n")
    md.append("| **边缘设备部署** | 模型大小 | 逻辑回归 |\n\n")

    return "".join(md)


def plot_model_comparison(results: dict, output_dir: Path) -> None:
    """画模型对比图表并保存"""
    font = setup_chinese_font()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. AUC 对比柱状图
    models = ['dummy', 'logistic_regression', 'decision_tree', 'random_forest']
    model_names = ['Dummy\n基线', '逻辑\n回归', '决策\n树', '随机\n森林']
    aucs = [results[m]['metrics']['auc'] for m in models]

    colors = ['gray', 'blue', 'green', 'darkgreen']
    bars = axes[0].bar(model_names, aucs, color=colors, alpha=0.7)

    axes[0].set_ylabel('AUC')
    axes[0].set_title('模型对比（测试集 AUC）')
    axes[0].set_ylim(0, 1)
    axes[0].axhline(y=0.5, color='red', linestyle='--', label='随机猜测', linewidth=1)
    axes[0].legend()

    # 添加数值标签
    for bar, auc in zip(bars, aucs):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{auc:.3f}', ha='center', va='bottom', fontsize=10)

    # 2. 交叉验证 AUC 分布（箱线图）
    cv_data = [results[m]['cv_scores']['scores'] for m in models]
    # 使用 tick_labels 替代 labels（兼容 matplotlib 3.9+）
    bp = axes[1].boxplot(cv_data, tick_labels=model_names, patch_artist=True)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    axes[1].set_ylabel('AUC')
    axes[1].set_title('交叉验证 AUC 分布（5 折）')
    axes[1].axhline(y=0.5, color='red', linestyle='--', label='随机猜测', linewidth=1)
    axes[1].legend()

    plt.tight_layout()

    # 保存图片
    plt.savefig(output_dir / 'model_comparison.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"模型对比图已保存到: {output_dir / 'model_comparison.png'}")


def plot_feature_importance(results: dict, output_dir: Path) -> None:
    """画特征重要性图并保存"""
    if 'feature_importance' not in results:
        return

    font = setup_chinese_font()
    fig, ax = plt.subplots(figsize=(10, 6))

    top_n = 15
    top_features = results['feature_importance'].head(top_n)

    bars = ax.barh(range(len(top_features)), top_features['importance'].values, color='steelblue')
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'].values)
    ax.set_xlabel('特征重要性')
    ax.set_title(f'随机森林特征重要性（Top {top_n}）')
    ax.invert_yaxis()

    # 添加数值标签
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2,
                f'{width:.3f}', ha='left', va='center', fontsize=9)

    plt.tight_layout()

    # 保存图片
    plt.savefig(output_dir / 'feature_importance.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"特征重要性图已保存到: {output_dir / 'feature_importance.png'}")


def main() -> None:
    """主函数：运行 StatLab 树模型对比流水线"""
    print("=" * 60)
    print("StatLab 树模型报告生成流水线")
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

    # 训练并对比模型
    print("\n训练模型...")
    results = baseline_comparison_with_pipeline(X, y, numeric_features, categorical_features)

    # 生成图表
    print("\n生成图表...")
    plot_model_comparison(results, output_dir)
    plot_feature_importance(results, output_dir)

    # 生成报告
    print("生成报告...")
    report = format_model_comparison_report(results)

    # 保存报告
    report_path = output_dir / 'model_comparison_report.md'
    report_path.write_text(report, encoding='utf-8')
    print(f"\n报告已保存到: {report_path}")

    # 打印报告预览
    print("\n" + "=" * 60)
    print("报告预览")
    print("=" * 60)
    print(report[:1500] + "...\n")

    print("=" * 60)
    print("StatLab 树模型对比流水线完成！")
    print("=" * 60)
    print("\n生成的文件:")
    print(f"  1. {output_dir / 'model_comparison.png'} - 模型对比图")
    print(f"  2. {output_dir / 'feature_importance.png'} - 特征重要性图")
    print(f"  3. {output_dir / 'model_comparison_report.md'} - 模型对比报告")

    print("\n本周 StatLab 进展:")
    print("  - 添加了树模型模块（决策树、随机森林）")
    print("  - 实现了多模型对比与基线评估")
    print("  - 包含模型选择理由和权衡分析")
    print("  - 使用 Pipeline 防止数据泄漏")

    print("\n与 Week 10 的对比:")
    print("  - 上周: 逻辑回归分类 + 单模型评估")
    print("  - 本周: 树模型 + 多模型对比 + 基线评估")
    print("  - 新增: 决策树可视化、特征重要性、模型选择框架")


if __name__ == "__main__":
    main()

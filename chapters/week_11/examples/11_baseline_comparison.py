"""
示例：基线对比——傻瓜基线、逻辑回归基线、单特征树基线

运行方式：python3 chapters/week_11/examples/11_baseline_comparison.py
预期输出：多模型对比表、Bootstrap AUC 分布图、模型选择建议
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
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


def load_titanic_data() -> tuple:
    """加载并准备泰坦尼克数据集"""
    print("=" * 60)
    print("加载泰坦尼克数据集")
    print("=" * 60)

    titanic = sns.load_dataset("titanic")

    # 选择特征
    feature_cols = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
    X = titanic[feature_cols].copy()
    y = titanic['survived']

    # 简化：删除缺失值
    X_clean = X.dropna()
    y_clean = y.loc[X_clean.index]

    # 编码分类型变量
    X_clean = X_clean.copy()
    X_clean['sex'] = X_clean['sex'].map({'male': 0, 'female': 1})
    X_clean['embarked'] = X_clean['embarked'].map({'C': 0, 'Q': 1, 'S': 2})
    X_clean = pd.get_dummies(X_clean, columns=['pclass'], drop_first=False)

    print(f"数据集规模: {X_clean.shape[0]} 行, {X_clean.shape[1]} 列")
    print(f"正类占比: {y_clean.mean():.2%}")

    return X_clean, y_clean


def train_all_baselines(X: pd.DataFrame, y: pd.Series) -> dict:
    """
    训练所有基线模型并对比

    返回:
    - dict: 包含所有模型评估结果的字典
    """
    print("\n" + "=" * 60)
    print("训练所有基线模型")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    results = {}

    # 1. 傻瓜基线（Dummy Baseline）
    print("\n【傻瓜基线】总是预测多数类")
    dummy = DummyClassifier(strategy='most_frequent')
    dummy.fit(X_train, y_train)

    y_pred = dummy.predict(X_test)
    y_prob = dummy.predict_proba(X_test)[:, 1]

    results['dummy'] = {
        'model': dummy,
        'name': '傻瓜基线',
        'metrics': evaluate_model(y_test, y_pred, y_prob),
        'cv_scores': cross_val_score(dummy, X, y, cv=5, scoring='roc_auc')
    }

    # 2. 逻辑回归基线
    print("\n【逻辑回归基线】最简单的线性分类器")
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(X_train, y_train)

    y_pred = log_reg.predict(X_test)
    y_prob = log_reg.predict_proba(X_test)[:, 1]

    results['logistic_regression'] = {
        'model': log_reg,
        'name': '逻辑回归',
        'metrics': evaluate_model(y_test, y_pred, y_prob),
        'cv_scores': cross_val_score(log_reg, X, y, cv=5, scoring='roc_auc')
    }

    # 3. 单特征树基线
    print("\n【单特征树基线】只用最重要的特征")

    # 找出最重要的特征（用逻辑回归系数）
    coef_abs = np.abs(log_reg.coef_[0])
    most_important_idx = np.argmax(coef_abs)
    most_important_feature = X.columns[most_important_idx]

    print(f"  最重要特征: {most_important_feature}")

    single_tree = DecisionTreeClassifier(
        max_depth=2,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42
    )
    single_tree.fit(X_train[[most_important_feature]], y_train)

    y_pred = single_tree.predict(X_test[[most_important_feature]])
    y_prob = single_tree.predict_proba(X_test[[most_important_feature]])[:, 1]

    results['single_feature_tree'] = {
        'model': single_tree,
        'name': f'单特征树 ({most_important_feature})',
        'metrics': evaluate_model(y_test, y_pred, y_prob),
        'cv_scores': cross_val_score(single_tree, X[[most_important_feature]], y, cv=5, scoring='roc_auc'),
        'feature': most_important_feature
    }

    # 4. 决策树
    print("\n【决策树】完整特征，限制深度")
    tree = DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42
    )
    tree.fit(X_train, y_train)

    y_pred = tree.predict(X_test)
    y_prob = tree.predict_proba(X_test)[:, 1]

    results['decision_tree'] = {
        'model': tree,
        'name': '决策树 (max_depth=5)',
        'metrics': evaluate_model(y_test, y_pred, y_prob),
        'cv_scores': cross_val_score(tree, X, y, cv=5, scoring='roc_auc')
    }

    # 5. 随机森林
    print("\n【随机森林】集成模型")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        max_features='sqrt',
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]

    results['random_forest'] = {
        'model': rf,
        'name': '随机森林 (100 棵树)',
        'metrics': evaluate_model(y_test, y_pred, y_prob),
        'cv_scores': cross_val_score(rf, X, y, cv=5, scoring='roc_auc')
    }

    # 保存测试数据用于后续 Bootstrap
    results['X_test'] = X_test
    results['y_test'] = y_test

    return results


def evaluate_model(y_test, y_pred, y_prob) -> dict:
    """计算评估指标"""
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'auc': roc_auc_score(y_test, y_prob)
    }


def print_comparison_table(results: dict) -> None:
    """打印模型对比表"""
    print("\n" + "=" * 60)
    print("模型对比表（测试集）")
    print("=" * 60)

    # 按模型顺序
    model_order = ['dummy', 'logistic_regression', 'single_feature_tree',
                   'decision_tree', 'random_forest']

    print(f"\n{'模型':<25} | {'准确率':<10} | {'精确率':<10} | {'召回率':<10} | {'F1':<10} | {'AUC':<10}")
    print("-" * 95)

    for key in model_order:
        res = results[key]
        m = res['metrics']
        print(f"{res['name']:<25} | {m['accuracy']:<10.4f} | "
              f"{m['precision']:<10.4f} | {m['recall']:<10.4f} | "
              f"{m['f1']:<10.4f} | {m['auc']:<10.4f}")

    # 交叉验证 AUC
    print("\n" + "=" * 60)
    print("交叉验证 AUC（5 折）")
    print("=" * 60)

    print(f"\n{'模型':<25} | {'平均 AUC':<12} | {'标准差':<12} | {'各折 AUC'}")
    print("-" * 90)

    for key in model_order:
        res = results[key]
        cv = res['cv_scores']
        scores_str = ', '.join([f"{s:.4f}" for s in cv])
        print(f"{res['name']:<25} | {cv.mean():<12.4f} | {cv.std():<12.4f} | {scores_str}")


def bootstrap_auc_distribution(model, X_test, y_test, n_bootstrap: int = 1000) -> np.ndarray:
    """
    Bootstrap 估计 AUC 的分布

    参数:
    - model: 训练好的模型
    - X_test: 测试集特征
    - y_test: 测试集标签
    - n_bootstrap: Bootstrap 次数

    返回:
    - np.ndarray: Bootstrap AUC 分数
    """
    np.random.seed(42)
    auc_scores = []
    n = len(X_test)

    for _ in range(n_bootstrap):
        # Bootstrap 采样（有放回）
        idx = np.random.choice(n, n, replace=True)
        X_boot = X_test.iloc[idx]
        y_boot = y_test.iloc[idx]

        # 预测并计算 AUC
        y_prob = model.predict_proba(X_boot)[:, 1]
        auc_scores.append(roc_auc_score(y_boot, y_prob))

    return np.array(auc_scores)


def test_significant_difference(model1, model2, X_test, y_test,
                                name1: str, name2: str) -> None:
    """
    检验两个模型的 AUC 是否有显著差异（使用 Mann-Whitney U 检验）

    参数:
    - model1, model2: 两个模型
    - X_test, y_test: 测试集
    - name1, name2: 模型名称
    """
    from scipy.stats import mannwhitneyu

    # Bootstrap AUC 分布
    auc1 = bootstrap_auc_distribution(model1, X_test, y_test)
    auc2 = bootstrap_auc_distribution(model2, X_test, y_test)

    # Mann-Whitney U 检验
    stat, p_value = mannwhitneyu(auc1, auc2, alternative='less')

    print(f"\n{name1} vs {name2}:")
    print(f"  {name1} AUC 均值: {auc1.mean():.4f} (±{auc1.std():.4f})")
    print(f"  {name2} AUC 均值: {auc2.mean():.4f} (±{auc2.std():.4f})")
    print(f"  提升: {(auc2.mean() - auc1.mean()):.4f} ({(auc2.mean() - auc1.mean())/auc1.mean()*100:.1f}%)")
    print(f"  Mann-Whitney U p 值: {p_value:.4f}")

    if p_value < 0.05:
        print(f"  结论: 提升显著 (p < 0.05)")
    else:
        print(f"  结论: 提升可能不显著 (p >= 0.05)")

    return auc1, auc2


def visualize_auc_distributions(results: dict) -> None:
    """可视化各模型的 AUC 分布（箱线图）"""
    print("\n" + "=" * 60)
    print("可视化 AUC 分布")
    print("=" * 60)

    font = setup_chinese_font()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 准备数据
    model_order = ['dummy', 'logistic_regression', 'decision_tree', 'random_forest']
    model_names = ['傻瓜基线', '逻辑回归', '决策树', '随机森林']
    test_aucs = [results[m]['metrics']['auc'] for m in model_order]
    cv_means = [results[m]['cv_scores'].mean() for m in model_order]
    cv_stds = [results[m]['cv_scores'].std() for m in model_order]

    # 1. 测试集 AUC 对比（柱状图）
    colors = ['gray', 'blue', 'green', 'darkgreen']
    bars = axes[0].bar(model_names, test_aucs, color=colors, alpha=0.7)
    axes[0].set_ylabel('AUC')
    axes[0].set_title('测试集 AUC 对比')
    axes[0].set_ylim(0, 1)
    axes[0].axhline(y=0.5, color='red', linestyle='--', label='随机猜测')
    axes[0].legend()

    # 添加数值标签
    for bar, auc in zip(bars, test_aucs):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{auc:.3f}', ha='center', va='bottom', fontsize=10)

    # 2. 交叉验证 AUC 分布（箱线图）
    cv_data = [results[m]['cv_scores'] for m in model_order]
    # 使用 tick_labels 替代 labels（兼容 matplotlib 3.9+）
    bp = axes[1].boxplot(cv_data, tick_labels=model_names, patch_artist=True)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    axes[1].set_ylabel('AUC')
    axes[1].set_title('交叉验证 AUC 分布（5 折）')
    axes[1].axhline(y=0.5, color='red', linestyle='--', label='随机猜测')
    axes[1].legend()

    plt.tight_layout()

    # 保存图片
    output_dir = Path(__file__).parent.parent / 'images'
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'baseline_comparison_auc.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"\n对比图已保存到: {output_dir / 'baseline_comparison_auc.png'}")


def generate_model_selection_report(results: dict) -> str:
    """生成模型选择报告"""
    print("\n" + "=" * 60)
    print("生成模型选择报告")
    print("=" * 60)

    dummy_auc = results['dummy']['metrics']['auc']
    lr_auc = results['logistic_regression']['metrics']['auc']
    rf_auc = results['random_forest']['metrics']['auc']

    report = []
    report.append("## 模型选择报告\n\n")

    # 1. 基线对比
    report.append("### 基线对比\n\n")
    report.append(f"- **傻瓜基线 AUC**: {dummy_auc:.4f}\n")
    report.append(f"- **逻辑回归基线 AUC**: {lr_auc:.4f}\n")
    report.append(f"- **随机森林 AUC**: {rf_auc:.4f}\n\n")

    report.append("**提升量**:\n")
    report.append(f"- 逻辑回归 vs 傻瓜基线: +{(lr_auc - dummy_auc):.4f}\n")
    improvement_rf_vs_lr = (rf_auc - lr_auc) / lr_auc * 100
    report.append(f"- 随机森林 vs 逻辑回归: +{(rf_auc - lr_auc):.4f} ({improvement_rf_vs_lr:.1f}%)\n\n")

    # 2. 复杂度 vs 提升量权衡
    report.append("### 复杂度 vs 提升量权衡\n\n")

    if improvement_rf_vs_lr < 2:
        report.append(f"- 随机森林比逻辑回归提升 {improvement_rf_vs_lr:.1f}%，提升量较小\n")
        report.append("- **建议**: 如果业务最关心预测力，选随机森林；如果需要向业务方解释规则，选逻辑回归\n\n")
    elif improvement_rf_vs_lr < 5:
        report.append(f"- 随机森林比逻辑回归提升 {improvement_rf_vs_lr:.1f}%，提升量中等\n")
        report.append("- **建议**: 如果预测力是关键，选随机森林；如果需要可解释性，选逻辑回归\n\n")
    else:
        report.append(f"- 随机森林比逻辑回归提升 {improvement_rf_vs_lr:.1f}%，提升量显著\n")
        report.append("- **建议**: 选择随机森林\n\n")

    # 3. 可解释性考虑
    report.append("### 可解释性考虑\n\n")
    report.append("**各模型的可解释性**:\n\n")
    report.append(f"- **傻瓜基线**: 无可解释性（总是预测多数类）\n")
    report.append(f"- **逻辑回归**: 高可解释性，可以解读每个特征的系数\n")
    report.append(f"- **单特征树**: 高可解释性，只有一个特征的简单规则\n")
    report.append(f"- **决策树**: 高可解释性，可以画出树结构\n")
    report.append(f"- **随机森林**: 中等可解释性，可以输出特征重要性，但无法画出完整决策过程\n\n")

    # 4. 场景推荐
    report.append("### 场景推荐\n\n")
    report.append("| 场景 | 优先级 | 推荐模型 |\n")
    report.append("|------|--------|----------|\n")
    report.append("| **需要向业务方解释** | 可解释性 | 逻辑回归 或 决策树 |\n")
    report.append("| **追求最高预测力** | AUC | 随机森林 |\n")
    report.append("| **实时预测（低延迟）** | 预测时间 | 逻辑回归 |\n")
    report.append("| **训练资源有限** | 训练时间 | 逻辑回归 或 决策树 |\n")
    report.append("| **边缘设备部署** | 模型大小 | 逻辑回归 |\n\n")

    return "".join(report)


def main() -> None:
    """主函数"""
    print("=" * 60)
    print("基线对比：傻瓜基线、逻辑回归、单特征树、决策树、随机森林")
    print("=" * 60)

    # 1. 加载数据
    X, y = load_titanic_data()

    # 2. 训练所有模型
    results = train_all_baselines(X, y)

    # 3. 打印对比表
    print_comparison_table(results)

    # 4. 显著性检验
    print("\n" + "=" * 60)
    print("显著性检验（Bootstrap + Mann-Whitney U）")
    print("=" * 60)

    lr_model = results['logistic_regression']['model']
    rf_model = results['random_forest']['model']
    X_test = results['X_test']
    y_test = results['y_test']

    lr_aucs, rf_aucs = test_significant_difference(
        lr_model, rf_model, X_test, y_test,
        "逻辑回归", "随机森林"
    )

    # 5. 可视化
    visualize_auc_distributions(results)

    # 6. 生成报告
    report = generate_model_selection_report(results)
    print("\n" + report)

    # 保存报告
    output_dir = Path(__file__).parent.parent.parent.parent / 'output'
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / 'baseline_comparison_report.md'
    report_path.write_text(report, encoding='utf-8')
    print(f"报告已保存到: {report_path}")

    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print("""
    基线对比核心要点：
    1. 为什么需要基线对比？
       - 评估"更复杂的模型是否值得"
       - 防止把"过拟合的提升"当成"真正的改进"
       - 权衡提升量、复杂度、可解释性

    2. 常见基线类型：
       - 傻瓜基线（Dummy）：总是预测多数类，检查模型是否比瞎猜好
       - 逻辑回归基线：最简单的线性分类器
       - 单特征树基线：只用最重要的特征

    3. 如何判断提升量是否显著？
       - Bootstrap 估计 AUC 的分布
       - 使用 Mann-Whitney U 检验
       - 不只看点估计，还要看不确定性

    4. 模型选择决策框架：
       - 提升量有多大？（<2% 小，2-5% 中，>5% 大）
       - 可解释性需求？（需要解释 vs 不需要）
       - 部署约束？（延迟、内存、训练时间）

    5. 工业界实践：
       - 永远先训练基线模型
       - 没有基线对比的模型选择不是分析，是炫技
       - 自动化工具（AutoML）也应该包含基线对比
    """)


if __name__ == "__main__":
    main()

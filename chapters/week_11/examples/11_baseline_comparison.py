"""
示例：基线对比——傻瓜基线、逻辑回归基线、单特征树基线

运行方式：python3 chapters/week_11/examples/11_baseline_comparison.py
预期输出：多模型对比表、paired bootstrap 差值区间、模型选择建议
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
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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


def load_churn_data() -> tuple[pd.DataFrame, pd.Series]:
    """加载 Week 10/11 共享的 churn 数据。"""
    print('=' * 60)
    print('加载共享 churn 数据')
    print('=' * 60)

    data_path = Path(__file__).resolve().parents[3] / 'data' / 'customer_churn.csv'
    df = pd.read_csv(data_path)
    X_raw = df.drop(columns=['is_churned'])
    X = pd.get_dummies(X_raw, columns=['contract_type'], drop_first=False)
    y = df['is_churned']

    print(f'数据集规模: {X.shape[0]} 行, {X.shape[1]} 列')
    print(f'正类占比: {y.mean():.2%}')

    return X, y


def make_logistic_pipeline() -> Pipeline:
    """构造逻辑回归基线，保持与 Week 10 的标准化主线一致。"""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=1000, random_state=42)),
    ])


def evaluate_model(y_true, y_pred, y_prob) -> dict:
    """计算评估指标。"""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auc': roc_auc_score(y_true, y_prob),
    }


def select_best_single_feature(X: pd.DataFrame, y: pd.Series) -> str:
    """用单特征浅树的交叉验证 AUC 选择最强单特征。"""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_feature = X.columns[0]
    best_auc = -np.inf

    for feature in X.columns:
        tree = DecisionTreeClassifier(
            max_depth=2,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
        )
        scores = cross_val_score(tree, X[[feature]], y, cv=cv, scoring='roc_auc')
        mean_auc = scores.mean()
        if mean_auc > best_auc:
            best_auc = mean_auc
            best_feature = feature

    return best_feature


def train_all_baselines(X: pd.DataFrame, y: pd.Series) -> dict:
    """训练所有基线模型并对比。"""
    print('\n' + '=' * 60)
    print('训练所有基线模型')
    print('=' * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    print('\n【傻瓜基线】总是预测多数类')
    dummy = DummyClassifier(strategy='most_frequent')
    dummy.fit(X_train, y_train)
    results['dummy'] = {
        'model': dummy,
        'name': '傻瓜基线',
        'metrics': evaluate_model(y_test, dummy.predict(X_test), dummy.predict_proba(X_test)[:, 1]),
        'cv_scores': cross_val_score(dummy, X, y, cv=cv, scoring='roc_auc'),
    }

    print('\n【逻辑回归基线】最简单的线性分类器')
    log_reg = make_logistic_pipeline()
    log_reg.fit(X_train, y_train)
    results['logistic_regression'] = {
        'model': log_reg,
        'name': '逻辑回归',
        'metrics': evaluate_model(y_test, log_reg.predict(X_test), log_reg.predict_proba(X_test)[:, 1]),
        'cv_scores': cross_val_score(make_logistic_pipeline(), X, y, cv=cv, scoring='roc_auc'),
    }

    print('\n【单特征树基线】只用一个最强单特征')
    best_feature = select_best_single_feature(X_train, y_train)
    print(f'  选择特征: {best_feature}')
    single_tree = DecisionTreeClassifier(
        max_depth=2,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
    )
    single_tree.fit(X_train[[best_feature]], y_train)
    results['single_feature_tree'] = {
        'model': single_tree,
        'name': f'单特征树 ({best_feature})',
        'metrics': evaluate_model(
            y_test,
            single_tree.predict(X_test[[best_feature]]),
            single_tree.predict_proba(X_test[[best_feature]])[:, 1],
        ),
        'cv_scores': cross_val_score(single_tree, X[[best_feature]], y, cv=cv, scoring='roc_auc'),
        'feature': best_feature,
    }

    print('\n【决策树】完整特征，限制深度')
    tree = DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
    )
    tree.fit(X_train, y_train)
    results['decision_tree'] = {
        'model': tree,
        'name': '决策树 (max_depth=5)',
        'metrics': evaluate_model(y_test, tree.predict(X_test), tree.predict_proba(X_test)[:, 1]),
        'cv_scores': cross_val_score(tree, X, y, cv=cv, scoring='roc_auc'),
    }

    print('\n【随机森林】集成模型')
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        max_features='sqrt',
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    results['random_forest'] = {
        'model': rf,
        'name': '随机森林 (100 棵树)',
        'metrics': evaluate_model(y_test, rf.predict(X_test), rf.predict_proba(X_test)[:, 1]),
        'cv_scores': cross_val_score(rf, X, y, cv=cv, scoring='roc_auc'),
    }

    results['X_test'] = X_test
    results['y_test'] = y_test
    return results


def print_comparison_table(results: dict) -> None:
    """打印模型对比表。"""
    print('\n' + '=' * 60)
    print('模型对比表（测试集）')
    print('=' * 60)

    model_order = ['dummy', 'logistic_regression', 'single_feature_tree', 'decision_tree', 'random_forest']
    print(f"\n{'模型':<25} | {'准确率':<10} | {'精确率':<10} | {'召回率':<10} | {'F1':<10} | {'AUC':<10}")
    print('-' * 95)

    for key in model_order:
        res = results[key]
        m = res['metrics']
        print(f"{res['name']:<25} | {m['accuracy']:<10.4f} | {m['precision']:<10.4f} | {m['recall']:<10.4f} | {m['f1']:<10.4f} | {m['auc']:<10.4f}")

    print('\n' + '=' * 60)
    print('交叉验证 AUC（5 折）')
    print('=' * 60)
    print(f"\n{'模型':<25} | {'平均 AUC':<12} | {'标准差':<12} | {'各折 AUC'}")
    print('-' * 90)

    for key in model_order:
        res = results[key]
        cv_scores = res['cv_scores']
        scores_str = ', '.join([f'{score:.4f}' for score in cv_scores])
        print(f"{res['name']:<25} | {cv_scores.mean():<12.4f} | {cv_scores.std():<12.4f} | {scores_str}")


def paired_bootstrap_auc_difference(model1, model2, X_test: pd.DataFrame, y_test: pd.Series,
                                    n_bootstrap: int = 2000, random_state: int = 42) -> dict:
    """在同一批 bootstrap 索引上估计 AUC 差值分布。"""
    rng = np.random.default_rng(random_state)
    y_true = y_test.to_numpy()
    prob1 = model1.predict_proba(X_test)[:, 1]
    prob2 = model2.predict_proba(X_test)[:, 1]
    n = len(y_true)

    auc1_scores = []
    auc2_scores = []
    diff_scores = []

    while len(diff_scores) < n_bootstrap:
        idx = rng.integers(0, n, size=n)
        y_boot = y_true[idx]
        if len(np.unique(y_boot)) < 2:
            continue
        auc1 = roc_auc_score(y_boot, prob1[idx])
        auc2 = roc_auc_score(y_boot, prob2[idx])
        auc1_scores.append(auc1)
        auc2_scores.append(auc2)
        diff_scores.append(auc2 - auc1)

    diff_scores = np.array(diff_scores)
    auc1_scores = np.array(auc1_scores)
    auc2_scores = np.array(auc2_scores)
    ci_low, ci_high = np.percentile(diff_scores, [2.5, 97.5])

    return {
        'auc1_scores': auc1_scores,
        'auc2_scores': auc2_scores,
        'diff_scores': diff_scores,
        'auc1_mean': auc1_scores.mean(),
        'auc2_mean': auc2_scores.mean(),
        'diff_mean': diff_scores.mean(),
        'diff_ci': (ci_low, ci_high),
        'supports_improvement': bool(ci_low > 0),
        'supports_baseline': bool(ci_high < 0),
    }


def test_significant_difference(model1, model2, X_test, y_test, name1: str, name2: str) -> dict:
    """用 paired bootstrap AUC 差值 CI 判断提升证据。"""
    result = paired_bootstrap_auc_difference(model1, model2, X_test, y_test)
    ci_low, ci_high = result['diff_ci']

    print(f'\n{name1} vs {name2}:')
    print(f'  {name1} bootstrap AUC 均值: {result["auc1_mean"]:.4f}')
    print(f'  {name2} bootstrap AUC 均值: {result["auc2_mean"]:.4f}')
    print(f'  平均提升量 ({name2} - {name1}): {result["diff_mean"]:.4f}')
    print(f'  95% CI: [{ci_low:.4f}, {ci_high:.4f}]')

    if result['supports_improvement']:
        print('  结论: 差值 CI 完全大于 0，支持随机森林优于逻辑回归。')
    elif result['supports_baseline']:
        print('  结论: 差值 CI 完全小于 0，支持逻辑回归优于随机森林。')
    else:
        print('  结论: 差值 CI 包含 0，当前证据不足以断言提升稳定存在。')

    return result


def visualize_auc_distributions(results: dict) -> None:
    """可视化测试集 AUC 与 paired bootstrap 差值分布。"""
    print('\n' + '=' * 60)
    print('可视化 AUC 与提升量分布')
    print('=' * 60)

    setup_chinese_font()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    model_order = ['dummy', 'logistic_regression', 'decision_tree', 'random_forest']
    model_names = ['傻瓜基线', '逻辑回归', '决策树', '随机森林']
    test_aucs = [results[m]['metrics']['auc'] for m in model_order]
    colors = ['gray', 'blue', 'green', 'darkgreen']

    bars = axes[0].bar(model_names, test_aucs, color=colors, alpha=0.75)
    axes[0].set_ylabel('AUC')
    axes[0].set_title('测试集 AUC 对比')
    axes[0].set_ylim(0, 1)
    axes[0].axhline(y=0.5, color='red', linestyle='--', label='随机猜测')
    axes[0].legend()
    for bar, auc in zip(bars, test_aucs):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{auc:.3f}', ha='center', va='bottom')

    diff_scores = results['significance']['diff_scores']
    ci_low, ci_high = results['significance']['diff_ci']
    axes[1].hist(diff_scores, bins=30, color='teal', alpha=0.8)
    axes[1].axvline(0, color='red', linestyle='--', label='无提升')
    axes[1].axvline(ci_low, color='black', linestyle=':', label='95% CI')
    axes[1].axvline(ci_high, color='black', linestyle=':')
    axes[1].set_title('随机森林 - 逻辑回归\npaired bootstrap AUC 差值分布')
    axes[1].set_xlabel('AUC 差值')
    axes[1].set_ylabel('频数')
    axes[1].legend()

    plt.tight_layout()
    output_dir = Path(__file__).parent.parent / 'images'
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'baseline_comparison_auc.png', dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f'\n对比图已保存到: {output_dir / "baseline_comparison_auc.png"}')


def generate_model_selection_report(results: dict) -> str:
    """生成模型选择报告。"""
    dummy_auc = results['dummy']['metrics']['auc']
    lr_auc = results['logistic_regression']['metrics']['auc']
    rf_auc = results['random_forest']['metrics']['auc']
    significance = results['significance']
    ci_low, ci_high = significance['diff_ci']

    report = []
    report.append('## 模型选择报告\n\n')
    report.append('### 基线对比\n\n')
    report.append(f'- **傻瓜基线 AUC**: {dummy_auc:.4f}\n')
    report.append(f'- **逻辑回归基线 AUC**: {lr_auc:.4f}\n')
    report.append(f'- **随机森林 AUC**: {rf_auc:.4f}\n\n')

    report.append('### 提升量与不确定性\n\n')
    report.append(f'- 随机森林相对逻辑回归的平均 AUC 提升量: {significance["diff_mean"]:.4f}\n')
    report.append(f'- paired bootstrap 95% CI: [{ci_low:.4f}, {ci_high:.4f}]\n')
    if significance['supports_improvement']:
        report.append('- 结论: 差值 CI 完全大于 0，支持随机森林有稳定提升。\n\n')
    elif significance['supports_baseline']:
        report.append('- 结论: 差值 CI 完全小于 0，说明逻辑回归在当前数据上更稳。\n\n')
    else:
        report.append('- 结论: 差值 CI 包含 0，当前证据不足以断言随机森林稳定优于逻辑回归。\n\n')

    report.append('### 复杂度 vs 提升量权衡\n\n')
    improvement_pct = (rf_auc - lr_auc) / lr_auc * 100
    if rf_auc <= lr_auc:
        report.append(f'- 当前这份数据上，随机森林比逻辑回归低 {abs(improvement_pct):.1f}%。\n')
        report.append('- 既然 paired bootstrap 也没有支持随机森林更优，优先保留逻辑回归更稳妥。\n\n')
    elif improvement_pct < 2:
        report.append(f'- 随机森林比逻辑回归提升 {improvement_pct:.1f}%，提升量较小。\n')
        report.append('- 如果业务最关心预测力，选随机森林；如果需要可解释性，选逻辑回归。\n\n')
    elif improvement_pct < 5:
        report.append(f'- 随机森林比逻辑回归提升 {improvement_pct:.1f}%，提升量中等。\n')
        report.append('- 建议结合 paired bootstrap 证据和业务解释需求一起决策。\n\n')
    else:
        report.append(f'- 随机森林比逻辑回归提升 {improvement_pct:.1f}%，提升量较大。\n')
        report.append('- 若 paired bootstrap 也支持提升，可以优先考虑随机森林。\n\n')

    report.append('### 场景推荐\n\n')
    report.append('| 场景 | 优先级 | 推荐模型 |\n')
    report.append('|------|--------|----------|\n')
    report.append('| **需要向业务方解释** | 可解释性 | 逻辑回归 或 决策树 |\n')
    report.append('| **追求最高预测力** | AUC | 随机森林 |\n')
    report.append('| **希望稳妥保守** | 证据强度 | 先看 paired bootstrap CI |\n')

    return ''.join(report)


def main() -> None:
    """主函数"""
    print('=' * 60)
    print('基线对比：傻瓜基线、逻辑回归、单特征树、决策树、随机森林')
    print('=' * 60)

    X, y = load_churn_data()
    results = train_all_baselines(X, y)
    print_comparison_table(results)

    print('\n' + '=' * 60)
    print('提升量检验（paired bootstrap AUC 差值 CI）')
    print('=' * 60)
    significance = test_significant_difference(
        results['logistic_regression']['model'],
        results['random_forest']['model'],
        results['X_test'],
        results['y_test'],
        '逻辑回归',
        '随机森林',
    )
    results['significance'] = significance

    visualize_auc_distributions(results)
    report = generate_model_selection_report(results)
    print('\n' + report)

    output_dir = Path(__file__).parent.parent.parent.parent / 'output'
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / 'baseline_comparison_report.md'
    report_path.write_text(report, encoding='utf-8')
    print(f'报告已保存到: {report_path}')

    print('\n' + '=' * 60)
    print('总结')
    print('=' * 60)
    print("""
    基线对比核心要点：
    1. 永远先看傻瓜基线和逻辑回归，确认复杂模型是否真的带来净提升。
    2. 单特征树最好直接按单特征树自己的交叉验证 AUC 来选，而不是偷看未标准化系数。
    3. 判断“随机森林是否优于逻辑回归”，应看 paired bootstrap 的 AUC 差值 CI 是否包含 0。
    4. 模型选择不只看 AUC 点估计，还要看提升量、证据强度和可解释性成本。
    """)


if __name__ == '__main__':
    main()

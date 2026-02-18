"""
Week 11: Bootstrap 显著性检验（可选的高级方法）

这是第 5 节"提升量是否显著"中提到的更严谨的统计检验方法。
如果你需要正式的 p 值，可以用这个脚本。

适用场景：
- 需要向业务方证明"提升量是统计显著的"
- 需要在论文/报告中给出正式的统计检验结果
- 交叉验证的置信区间重叠，但你想更精确地判断

方法：
1. 用 Bootstrap 估计每个模型的 AUC 分布
2. 用 Mann-Whitney U 检验判断两个分布是否有显著差异
"""

import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def bootstrap_auc(model, X, y, n_bootstrap=1000, random_state=42):
    """
    通过 Bootstrap 估计模型 AUC 的分布

    参数:
        model: 训练好的模型（带 predict_proba 方法）
        X: 特征数据
        y: 真实标签
        n_bootstrap: Bootstrap 重复次数
        random_state: 随机种子

    返回:
        auc_scores: AUC 分数的数组
    """
    np.random.seed(random_state)
    auc_scores = []
    n = len(X)

    for _ in range(n_bootstrap):
        # 有放回抽样
        idx = np.random.choice(n, n, replace=True)
        X_boot = X.iloc[idx] if hasattr(X, 'iloc') else X[idx]
        y_boot = y.iloc[idx] if hasattr(y, 'iloc') else y[idx]

        # 预测并计算 AUC
        y_prob = model.predict_proba(X_boot)[:, 1]
        auc_scores.append(roc_auc_score(y_boot, y_prob))

    return np.array(auc_scores)


def compare_models_with_bootstrap(model_a, model_b, X, y,
                                   name_a="Model A", name_b="Model B",
                                   n_bootstrap=1000, alpha=0.05):
    """
    用 Bootstrap + Mann-Whitney U 检验比较两个模型

    参数:
        model_a, model_b: 训练好的模型
        X, y: 测试数据
        name_a, name_b: 模型名称
        n_bootstrap: Bootstrap 次数
        alpha: 显著性水平

    返回:
        dict: 包含检验结果的字典
    """
    # Bootstrap 估计 AUC 分布
    auc_a = bootstrap_auc(model_a, X, y, n_bootstrap)
    auc_b = bootstrap_auc(model_b, X, y, n_bootstrap)

    # Mann-Whitney U 检验（双侧检验）
    stat, p_value = mannwhitneyu(auc_a, auc_b, alternative='two-sided')

    # 判断显著性
    is_significant = p_value < alpha

    # 计算置信区间
    ci_a = (np.percentile(auc_a, 2.5), np.percentile(auc_a, 97.5))
    ci_b = (np.percentile(auc_b, 2.5), np.percentile(auc_b, 97.5))

    return {
        f'{name_a}_mean': auc_a.mean(),
        f'{name_a}_std': auc_a.std(),
        f'{name_a}_ci': ci_a,
        f'{name_b}_mean': auc_b.mean(),
        f'{name_b}_std': auc_b.std(),
        f'{name_b}_ci': ci_b,
        'difference': auc_b.mean() - auc_a.mean(),
        'p_value': p_value,
        'is_significant': is_significant
    }


def print_comparison_result(result, name_a="Model A", name_b="Model B"):
    """格式化打印比较结果"""
    print(f"\n{'='*50}")
    print("Bootstrap 显著性检验结果")
    print(f"{'='*50}")

    print(f"\n{name_a}:")
    print(f"  AUC = {result[f'{name_a}_mean']:.4f} ± {result[f'{name_a}_std']:.4f}")
    print(f"  95% CI: [{result[f'{name_a}_ci'][0]:.4f}, {result[f'{name_a}_ci'][1]:.4f}]")

    print(f"\n{name_b}:")
    print(f"  AUC = {result[f'{name_b}_mean']:.4f} ± {result[f'{name_b}_std']:.4f}")
    print(f"  95% CI: [{result[f'{name_b}_ci'][0]:.4f}, {result[f'{name_b}_ci'][1]:.4f}]")

    print(f"\n差异:")
    print(f"  {name_b} - {name_a} = {result['difference']:.4f}")

    print(f"\nMann-Whitney U 检验:")
    print(f"  p 值 = {result['p_value']:.4f}")
    print(f"  结论: {'显著差异（p < 0.05）' if result['is_significant'] else '无显著差异'}")

    return result


if __name__ == "__main__":
    # 示例：用模拟数据演示
    print("生成模拟数据...")
    X, y = make_classification(
        n_samples=1000, n_features=10, n_informative=5,
        n_redundant=2, random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 训练两个模型
    print("训练逻辑回归...")
    lr = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=1000, random_state=42))
    ])
    lr.fit(X_train, y_train)

    print("训练随机森林...")
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)

    # Bootstrap 检验
    print("\n执行 Bootstrap 检验（1000 次重采样）...")
    result = compare_models_with_bootstrap(
        lr, rf, X_test, y_test,
        name_a="Logistic Regression",
        name_b="Random Forest",
        n_bootstrap=1000
    )

    print_comparison_result(result, "Logistic Regression", "Random Forest")

    print("\n" + "="*50)
    print("解读：")
    print("="*50)
    if result['is_significant']:
        print(f"✓ 随机森林的 AUC 显著高于逻辑回归")
        print(f"  提升量 {result['difference']:.4f} 是统计显著的")
    else:
        print(f"✗ 提升量不显著，可能是随机波动")
        print(f"  建议使用更简单的逻辑回归")

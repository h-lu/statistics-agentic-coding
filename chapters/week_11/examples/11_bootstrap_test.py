"""
Week 11: paired bootstrap 比较两个模型的 AUC 提升量

运行方式：python3 chapters/week_11/examples/11_bootstrap_test.py
预期输出：逻辑回归与随机森林的 AUC 差值均值、95% CI、结论解释
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_churn_data() -> tuple[pd.DataFrame, pd.Series]:
    """加载 Week 10/11 共享的 churn 数据。"""
    data_path = Path(__file__).resolve().parents[3] / 'data' / 'customer_churn.csv'
    df = pd.read_csv(data_path)
    X_raw = df.drop(columns=['is_churned'])
    X = pd.get_dummies(X_raw, columns=['contract_type'], drop_first=False)
    y = df['is_churned']
    return X, y


def make_logistic_pipeline() -> Pipeline:
    """构造逻辑回归基线。"""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=1000, random_state=42)),
    ])


def paired_bootstrap_auc_difference(model_a, model_b, X, y, n_bootstrap=2000, random_state=42):
    """在同一批 bootstrap 索引上估计 AUC 差值分布。"""
    rng = np.random.default_rng(random_state)
    y_true = y.to_numpy() if hasattr(y, 'to_numpy') else np.asarray(y)
    prob_a = model_a.predict_proba(X)[:, 1]
    prob_b = model_b.predict_proba(X)[:, 1]
    n = len(y_true)

    auc_a = []
    auc_b = []
    diff = []

    while len(diff) < n_bootstrap:
        idx = rng.integers(0, n, size=n)
        y_boot = y_true[idx]
        if len(np.unique(y_boot)) < 2:
            continue
        auc_a_boot = roc_auc_score(y_boot, prob_a[idx])
        auc_b_boot = roc_auc_score(y_boot, prob_b[idx])
        auc_a.append(auc_a_boot)
        auc_b.append(auc_b_boot)
        diff.append(auc_b_boot - auc_a_boot)

    diff = np.array(diff)
    ci = np.percentile(diff, [2.5, 97.5])
    return {
        'auc_a_mean': float(np.mean(auc_a)),
        'auc_b_mean': float(np.mean(auc_b)),
        'diff_mean': float(np.mean(diff)),
        'diff_ci': (float(ci[0]), float(ci[1])),
        'supports_improvement': bool(ci[0] > 0),
        'supports_baseline': bool(ci[1] < 0),
        'supports_baseline': bool(ci[1] < 0),
    }


def print_comparison_result(result, name_a='Model A', name_b='Model B'):
    """格式化打印比较结果。"""
    ci_low, ci_high = result['diff_ci']
    print(f"\n{'=' * 50}")
    print('paired bootstrap AUC 差值结果')
    print(f"{'=' * 50}")
    print(f"\n{name_a}: bootstrap AUC 均值 = {result['auc_a_mean']:.4f}")
    print(f"{name_b}: bootstrap AUC 均值 = {result['auc_b_mean']:.4f}")
    print(f"\n{name_b} - {name_a} 的平均提升量 = {result['diff_mean']:.4f}")
    print(f"95% CI = [{ci_low:.4f}, {ci_high:.4f}]")
    if result['supports_improvement']:
        print('结论: 差值 CI 完全大于 0，支持随机森林优于逻辑回归。')
    elif result['supports_baseline']:
        print('结论: 差值 CI 完全小于 0，支持逻辑回归优于随机森林。')
    else:
        print('结论: 差值 CI 包含 0，当前证据不足以断言提升稳定存在。')


if __name__ == '__main__':
    print('加载共享 churn 数据...')
    X, y = load_churn_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print('训练逻辑回归...')
    lr = make_logistic_pipeline()
    lr.fit(X_train, y_train)

    print('训练随机森林...')
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=5, max_features='sqrt',
        min_samples_split=20, min_samples_leaf=10,
        random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)

    print('\n执行 paired bootstrap 检验（2000 次重采样）...')
    result = paired_bootstrap_auc_difference(
        lr, rf, X_test, y_test, n_bootstrap=2000
    )
    print_comparison_result(result, 'Logistic Regression', 'Random Forest')

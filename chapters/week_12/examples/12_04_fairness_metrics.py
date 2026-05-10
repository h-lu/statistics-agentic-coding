"""
示例：公平性指标——区分 Demographic Parity、Equal Opportunity 与 Equalized Odds

本例演示：
1. 用同一组预测比较多种公平性定义
2. 通过后处理阈值让统计均等更接近
3. 观察准确率与公平性指标如何一起变化

运行方式：python3 chapters/week_12/examples/12_04_fairness_metrics.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

STARTER_CODE = Path(__file__).resolve().parents[1] / "starter_code"
if str(STARTER_CODE) not in sys.path:
    sys.path.insert(0, str(STARTER_CODE))

import solution


def generate_fairness_scenario(n_samples: int = 2200, random_state: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(random_state)

    group = rng.binomial(1, 0.42, n_samples)
    days = rng.poisson(30, n_samples)
    count = rng.poisson(5, n_samples)
    spend = rng.gamma(2.2, 42.0, n_samples)
    vip = rng.binomial(1, 0.28, n_samples)
    support_calls = rng.poisson(1.4, n_samples)

    base_logit = (
        -2.3
        + 0.08 * np.maximum(days - 18, 0)
        - 0.17 * count
        - 0.012 * spend
        - 0.8 * vip
        + 0.24 * support_calls
    )
    group_effect = 0.55 * group
    logit = base_logit + group_effect
    prob = 1.0 / (1.0 + np.exp(-logit))
    churn = rng.binomial(1, prob)

    X = pd.DataFrame(
        {
            "days_since_last_purchase": days,
            "purchase_count": count,
            "avg_spend": spend.round(2),
            "vip_status": vip,
            "support_calls": support_calls,
            "group": group,
        }
    )
    return X, pd.Series(churn, name="churn")


def apply_demographic_parity_threshold(y_prob: np.ndarray, group: np.ndarray, target_rate: float) -> np.ndarray:
    y_pred = np.zeros_like(y_prob, dtype=int)
    for group_value in np.unique(group):
        mask = group == group_value
        threshold = float(np.quantile(y_prob[mask], max(0.0, min(1.0, 1.0 - target_rate))))
        y_pred[mask] = (y_prob[mask] >= threshold).astype(int)
    return y_pred


def plot_tradeoff(original_metrics: dict, adjusted_metrics: dict, original_accuracy: float, adjusted_accuracy: float, output_path: Path) -> None:
    solution.setup_chinese_font()

    labels = ["DP diff", "TPR diff", "FPR diff"]
    original_values = [
        original_metrics["demographic_parity"]["difference"],
        original_metrics["equalized_odds"]["tpr_diff"],
        original_metrics["equalized_odds"]["fpr_diff"],
    ]
    adjusted_values = [
        adjusted_metrics["demographic_parity"]["difference"],
        adjusted_metrics["equalized_odds"]["tpr_diff"],
        adjusted_metrics["equalized_odds"]["fpr_diff"],
    ]

    x = np.arange(len(labels))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].bar(x - width / 2, original_values, width, label="Baseline", color="coral", alpha=0.8)
    axes[0].bar(x + width / 2, adjusted_values, width, label="DP-adjusted", color="steelblue", alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Difference")
    axes[0].set_title("Fairness metrics (smaller is closer to parity)")
    axes[0].legend()

    axes[1].scatter(
        [max(original_values[1:]), max(adjusted_values[1:])],
        [original_accuracy, adjusted_accuracy],
        s=180,
        c=["coral", "steelblue"],
        edgecolors="black",
    )
    axes[1].annotate("Baseline", (max(original_values[1:]), original_accuracy), xytext=(8, 0), textcoords="offset points")
    axes[1].annotate("DP-adjusted", (max(adjusted_values[1:]), adjusted_accuracy), xytext=(8, 0), textcoords="offset points")
    axes[1].set_xlabel("Equalized Odds gap (max of TPR/FPR diff)")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy vs fairness")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    print("=" * 60)
    print("Week 12 fairness metrics demo")
    print("=" * 60)

    output_dir = Path(__file__).resolve().parents[3] / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y = generate_fairness_scenario()
    X_train, X_test, y_train, y_test = train_test_split(
        X.drop(columns="group"), y, test_size=0.3, random_state=42, stratify=y
    )
    group_test = X.loc[X_test.index, "group"].to_numpy()

    model = RandomForestClassifier(
        n_estimators=140,
        max_depth=6,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    baseline_threshold = float(np.quantile(y_prob, 0.68))
    y_pred = (y_prob >= baseline_threshold).astype(int)

    original_metrics = solution.compute_all_fairness_metrics(y_test, y_pred, group_test, y_prob=y_prob)
    original_accuracy = accuracy_score(y_test, y_pred)

    target_positive_rate = float(y_pred.mean())
    y_pred_fair = apply_demographic_parity_threshold(y_prob, group_test, target_positive_rate)
    adjusted_metrics = solution.compute_all_fairness_metrics(y_test, y_pred_fair, group_test, y_prob=y_prob)
    adjusted_accuracy = accuracy_score(y_test, y_pred_fair)

    print(f"\nAUC: {roc_auc_score(y_test, y_prob):.4f}")
    print(f"原始准确率: {original_accuracy:.4f}")
    print(f"后处理准确率: {adjusted_accuracy:.4f}")
    print(f"准确率变化: {adjusted_accuracy - original_accuracy:+.4f}")

    print("\n原始模型:")
    print(f"  Demographic parity diff: {original_metrics['demographic_parity']['difference']:.3f}")
    print(f"  Equal opportunity diff (TPR only): {original_metrics['equal_opportunity']:.3f}")
    print(f"  Equalized odds TPR diff: {original_metrics['equalized_odds']['tpr_diff']:.3f}")
    print(f"  Equalized odds FPR diff: {original_metrics['equalized_odds']['fpr_diff']:.3f}")

    print("\nDP 后处理后:")
    print(f"  Demographic parity diff: {adjusted_metrics['demographic_parity']['difference']:.3f}")
    print(f"  Equal opportunity diff (TPR only): {adjusted_metrics['equal_opportunity']:.3f}")
    print(f"  Equalized odds TPR diff: {adjusted_metrics['equalized_odds']['tpr_diff']:.3f}")
    print(f"  Equalized odds FPR diff: {adjusted_metrics['equalized_odds']['fpr_diff']:.3f}")

    plot_path = output_dir / "fairness_tradeoff.png"
    plot_tradeoff(original_metrics, adjusted_metrics, original_accuracy, adjusted_accuracy, plot_path)

    print("\n说明：")
    print("  - 如果只比较 TPR，你讨论的是 Equal Opportunity。")
    print("  - 如果同时比较 TPR 和 FPR，你讨论的是 Equalized Odds。")
    print("  - 本例只演示一种后处理思路，准确率变化幅度不是固定常数。")
    print(f"\n已生成文件:\n  - {plot_path}")


if __name__ == "__main__":
    main()

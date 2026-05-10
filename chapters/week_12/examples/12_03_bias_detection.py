"""
示例：偏见检测——按群体拆开混淆矩阵，而不是只看整体准确率

本例演示：
1. 构造带有历史标签偏差的客户流失数据
2. 训练模型并按地区/性别分组评估
3. 用混淆矩阵与 TPR/FPR 差异稳定展示“可讨论的不公平结果”

运行方式：python3 chapters/week_12/examples/12_03_bias_detection.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

STARTER_CODE = Path(__file__).resolve().parents[1] / "starter_code"
if str(STARTER_CODE) not in sys.path:
    sys.path.insert(0, str(STARTER_CODE))

import solution


def generate_biased_data(n_samples: int = 2400, random_state: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(random_state)

    region = rng.choice(["A", "B", "C"], size=n_samples, p=[0.45, 0.35, 0.20])
    gender = rng.choice(["female", "male"], size=n_samples, p=[0.5, 0.5])
    days = rng.poisson(30, n_samples)
    count = rng.poisson(4, n_samples)
    spend = rng.gamma(2.5, 40.0, n_samples)
    vip = rng.binomial(1, 0.25, n_samples)
    complaints = rng.poisson(1.2, n_samples)

    true_logit = (
        -2.4
        + 0.09 * np.maximum(days - 18, 0)
        - 0.22 * count
        - 0.014 * spend
        - 0.85 * vip
        + 0.28 * complaints
    )

    # 历史标签偏差：region B 的客户更容易被标记为“高流失风险”
    historical_bias = np.where(region == "B", 0.9, 0.0) + np.where(gender == "female", 0.15, 0.0)
    observed_logit = true_logit + historical_bias
    observed_prob = 1.0 / (1.0 + np.exp(-observed_logit))
    churn = rng.binomial(1, observed_prob)

    X = pd.DataFrame(
        {
            "days_since_last_purchase": days,
            "purchase_count": count,
            "avg_spend": spend.round(2),
            "vip_status": vip,
            "complaints_last_90d": complaints,
            "region": region,
            "gender": gender,
        }
    )
    return X, pd.Series(churn, name="churn")


def main() -> None:
    print("=" * 60)
    print("Week 12 bias detection demo")
    print("=" * 60)

    output_dir = Path(__file__).resolve().parents[3] / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y = generate_biased_data()
    X_encoded = pd.get_dummies(X, columns=["region", "gender"], drop_first=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.3, random_state=42, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=140,
        max_depth=6,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    threshold = float(np.quantile(y_prob, 0.70))
    y_pred = (y_prob >= threshold).astype(int)

    region_test = X.loc[X_test.index, "region"].to_numpy()
    gender_test = X.loc[X_test.index, "gender"].to_numpy()

    region_results = solution.evaluate_by_group(y_test, y_pred, region_test)
    gender_results = solution.evaluate_by_group(y_test, y_pred, gender_test)

    print("\n按地区分组评估：")
    print(region_results.to_string(index=False))
    print("\n按性别分组评估：")
    print(gender_results.to_string(index=False))

    region_bias = solution.detect_prediction_bias(y_test, y_pred, region_test)
    gender_bias = solution.detect_prediction_bias(y_test, y_pred, gender_test)

    print("\n地区偏见摘要：")
    print(
        f"  positive_rate_diff={region_bias['positive_rate_diff']:.3f}, "
        f"TPR diff={region_bias['equalized_odds']['tpr_diff']:.3f}, "
        f"FPR diff={region_bias['equalized_odds']['fpr_diff']:.3f}"
    )
    print("\n性别偏见摘要：")
    print(
        f"  positive_rate_diff={gender_bias['positive_rate_diff']:.3f}, "
        f"TPR diff={gender_bias['equalized_odds']['tpr_diff']:.3f}, "
        f"FPR diff={gender_bias['equalized_odds']['fpr_diff']:.3f}"
    )

    if region_bias["bias_detected"]:
        print("\n结论：地区维度存在稳定可讨论的分组差异，值得回头追查数据来源和标签流程。")
    else:
        print("\n结论：本次地区差异不明显，但这不代表模型天然公平。")

    metrics_path = output_dir / "fairness_comparison.png"
    cm_path = output_dir / "group_confusion_matrices.png"
    solution.plot_group_metrics(y_test, y_pred, region_test, metrics_path)
    solution.plot_group_confusion_matrices(y_test, y_pred, region_test, cm_path)

    print("\n已生成文件:")
    print(f"  - {metrics_path}")
    print(f"  - {cm_path}")


if __name__ == "__main__":
    main()

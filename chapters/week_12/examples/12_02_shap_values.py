"""
示例：局部解释——从全局特征重要性到单样本贡献分解

本例演示：
1. 训练一个客户流失预测模型
2. 生成全局贡献汇总图
3. 生成单样本瀑布图
4. 明确说明解释值对应的输出空间

运行方式：python3 chapters/week_12/examples/12_02_shap_values.py
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


def generate_synthetic_data(n_samples: int = 420, random_state: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(random_state)

    days = rng.poisson(28, n_samples)
    count = rng.poisson(5, n_samples)
    spend = rng.gamma(shape=2.0, scale=45.0, size=n_samples)
    vip = rng.binomial(1, 0.28, n_samples)
    service_tickets = rng.poisson(1.5, n_samples)
    age = rng.integers(20, 68, n_samples)

    logit = (
        -2.2
        + 0.08 * np.maximum(days - 20, 0)
        - 0.18 * count
        - 0.015 * spend
        - 0.8 * vip
        + 0.22 * service_tickets
        + 0.01 * (age - 40)
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    y = rng.binomial(1, prob)

    X = pd.DataFrame(
        {
            "days_since_last_purchase": days,
            "purchase_count": count,
            "avg_spend": spend.round(2),
            "vip_status": vip,
            "service_tickets": service_tickets,
            "age": age,
        }
    )
    return X, pd.Series(y, name="churn")


def main() -> None:
    print("=" * 60)
    print("Week 12 SHAP / local explanation demo")
    print("=" * 60)

    output_dir = Path(__file__).resolve().parents[3] / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y = generate_synthetic_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=5,
        min_samples_leaf=8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    explanation_slice = X_test.iloc[:80].copy()

    summary_path = output_dir / "shap_summary_plot.png"
    waterfall_path = output_dir / "shap_waterfall_plot.png"

    solution.plot_shap_summary(
        model,
        X_train,
        explanation_slice,
        feature_names=explanation_slice.columns.tolist(),
        output_path=summary_path,
    )
    solution.plot_shap_waterfall(
        model,
        explanation_slice,
        sample_idx=0,
        feature_names=explanation_slice.columns.tolist(),
        output_path=waterfall_path,
        X_train=X_train,
    )

    explanation = solution.explain_single_prediction(
        model,
        explanation_slice,
        sample_idx=0,
        feature_names=explanation_slice.columns.tolist(),
        X_train=X_train,
    )

    print("\n解释输出空间:")
    if explanation["output_space"] == "probability":
        print("  当前环境未安装 shap，示例使用的是概率空间下的确定性近似归因。")
        print("  它能说明“哪个特征把预测往上推/往下拉”，但不应冒充真实 SHAP。")
    else:
        print("  当前环境使用了解释器原生输出空间；常见情况可能是模型输出或 log-odds。")

    print("\n单样本解释:")
    print(solution.generate_explanation_text(
        model,
        explanation_slice,
        sample_idx=0,
        feature_names=explanation_slice.columns.tolist(),
        X_train=X_train,
    ))

    print("\n已生成文件:")
    print(f"  - {summary_path}")
    print(f"  - {waterfall_path}")


if __name__ == "__main__":
    main()

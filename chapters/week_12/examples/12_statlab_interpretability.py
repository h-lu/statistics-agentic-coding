"""
示例：StatLab 超级线——客户流失模型的可解释性与公平性报告

本例演示：
1. 延续客户流失场景，而不是切到无关数据集
2. 生成特征重要性、局部解释和公平性摘要
3. 输出面向非技术读者的 Markdown 报告

运行方式：python3 chapters/week_12/examples/12_statlab_interpretability.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

STARTER_CODE = Path(__file__).resolve().parents[1] / "starter_code"
if str(STARTER_CODE) not in sys.path:
    sys.path.insert(0, str(STARTER_CODE))

import solution


def generate_statlab_data(n_samples: int = 900, random_state: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(random_state)

    region = rng.choice(["north", "south", "west"], size=n_samples, p=[0.42, 0.33, 0.25])
    gender = rng.choice(["female", "male"], size=n_samples)
    days = rng.poisson(29, n_samples)
    count = rng.poisson(5, n_samples)
    spend = rng.gamma(2.1, 48.0, n_samples)
    vip = rng.binomial(1, 0.27, n_samples)
    complaints = rng.poisson(1.3, n_samples)
    discount_usage = rng.binomial(1, 0.45, n_samples)

    logit = (
        -2.35
        + 0.08 * np.maximum(days - 18, 0)
        - 0.16 * count
        - 0.013 * spend
        - 0.75 * vip
        + 0.25 * complaints
        + 0.22 * discount_usage
        + np.where(region == "south", 0.40, 0.0)
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    churn = rng.binomial(1, prob)

    X = pd.DataFrame(
        {
            "days_since_last_purchase": days,
            "purchase_count": count,
            "avg_spend": spend.round(2),
            "vip_status": vip,
            "complaints_last_90d": complaints,
            "discount_usage": discount_usage,
            "region": region,
            "gender": gender,
        }
    )
    return X, pd.Series(churn, name="churn")


def generate_non_technical_summary(metrics: dict, local_text: str, fairness_report: str) -> str:
    return "\n".join(
        [
            "## 面向业务方的结论",
            "",
            f"- 该模型的 AUC 为 {metrics['auc']:.2f}，表示它把高风险客户排在低风险客户前面的区分能力较强。",
            f"- 召回率为 {metrics['recall']:.0%}，意味着在真实会流失的客户里，模型能提前抓到约这么多。",
            "- 局部解释不直接等于“概率增加了多少个百分点”；要先看解释值位于概率空间还是模型输出空间。",
            "- 公平性部分需要同时看预测正率、TPR、FPR，而不是只看整体准确率是否相同。",
            "",
            "### 一个样本的解释",
            local_text,
            "",
            "### 公平性摘要",
            fairness_report,
        ]
    )


def main() -> None:
    print("=" * 60)
    print("Week 12 StatLab interpretability demo")
    print("=" * 60)

    output_dir = Path(__file__).resolve().parents[3] / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y = generate_statlab_data()
    sensitive_region = X["region"].copy()

    X_encoded = pd.get_dummies(X, columns=["region", "gender"], drop_first=False)
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.3, random_state=42, stratify=y
    )
    region_test = sensitive_region.loc[X_test.index].to_numpy()

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=6,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    threshold = float(np.quantile(y_prob, 0.68))
    y_pred = (y_prob >= threshold).astype(int)
    explanation_slice = X_test.iloc[:90].copy()

    feature_importance = solution.compute_feature_importance(
        model, feature_names=X_test.columns.tolist()
    )
    summary_path = output_dir / "shap_feature_importance.png"
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

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "auc": roc_auc_score(y_test, y_prob),
    }

    fairness_df = solution.evaluate_by_group(y_test, y_pred, region_test)
    fairness_report = solution.generate_fairness_report(
        y_test, y_pred, region_test, attr_name="region", y_prob=y_prob
    )
    local_text = solution.generate_explanation_text(
        model,
        explanation_slice,
        sample_idx=0,
        feature_names=explanation_slice.columns.tolist(),
        X_train=X_train,
    )

    report = [
        "# 客户流失模型：可解释性与公平性报告",
        "",
        "## 模型性能摘要",
        f"- Accuracy: {metrics['accuracy']:.3f}",
        f"- Precision: {metrics['precision']:.3f}",
        f"- Recall: {metrics['recall']:.3f}",
        f"- AUC: {metrics['auc']:.3f}",
        "",
        "## 主要特征",
    ]
    report.extend(
        f"- {row.feature}: importance={row.importance:.3f}"
        for row in feature_importance.head(8).itertuples()
    )
    report.extend(
        [
            "",
            f"![Global Contributions]({summary_path.name})",
            "",
            f"![Local Waterfall]({waterfall_path.name})",
            "",
            generate_non_technical_summary(metrics, local_text, fairness_report),
            "",
            "## 模型边界",
            "- 本报告使用的是合成的客户流失场景，用于教学示范，不应直接照搬到真实业务。",
            "- 若环境未安装 shap，局部解释会退化为确定性近似归因；它能帮助教学，但不等于正式审计。",
            "- 在高风险业务里，公平性评估需要结合数据来源、阈值策略和人工复核流程一起看。",
        ]
    )

    report_path = output_dir / "interpretability_ethics_report.md"
    report_path.write_text("\n".join(report), encoding="utf-8")

    print(f"\n报告已保存到: {report_path}")
    print(f"全局贡献图: {summary_path}")
    print(f"局部瀑布图: {waterfall_path}")
    print("\n公平性摘要:")
    print(fairness_df.to_string(index=False))


if __name__ == "__main__":
    main()

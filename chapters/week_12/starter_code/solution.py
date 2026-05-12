"""
Week 12 作业参考实现

本文件提供一套可运行、可测试的最小实现：
1. 特征重要性与逻辑回归系数
2. 局部/全局解释接口（优先使用真实 SHAP；缺库时退化为确定性的近似归因）
3. 分组偏见检测与公平性评估
4. 面向非技术读者的解释报告

说明：
- 如果环境中安装了 `shap`，相关函数会优先返回真实 SHAP 结果。
- 如果未安装 `shap`，不会再返回随机 mock；而是使用基于参考样本的确定性近似贡献，
  仅用于教学和测试环境中的接口闭环，不应与真实 SHAP 混为一谈。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    import shap  # type: ignore
except ImportError:  # pragma: no cover - exercised in current environment
    shap = None


def setup_chinese_font() -> str:
    """配置中文字体。"""
    font_candidates = [
        Path("/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/google-droid-fonts/DroidSansFallback.ttf"),
        Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
        Path("/usr/share/fonts/truetype/droid/DroidSansFallback.ttf"),
    ]
    for font_path in font_candidates:
        if font_path.exists():
            fm.fontManager.addfont(str(font_path))
            font_name = fm.FontProperties(fname=str(font_path)).get_name()
            plt.rcParams["font.family"] = font_name
            plt.rcParams["font.sans-serif"] = [font_name]
            plt.rcParams["axes.unicode_minus"] = False
            return font_name
    chinese_fonts = [
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "PingFang SC",
        "Microsoft YaHei",
    ]
    available = [f.name for f in fm.fontManager.ttflist]
    for font in chinese_fonts:
        if font in available:
            plt.rcParams["font.family"] = font
            plt.rcParams["font.sans-serif"] = [font]
            plt.rcParams["axes.unicode_minus"] = False
            return font
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    return "DejaVu Sans"


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return 1.0 / (1.0 + np.exp(-values))


def _to_numpy_1d(values, *, name: str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.size == 0:
        raise ValueError(f"{name} cannot be empty")
    return arr.reshape(-1)


def _to_2d_numeric_array(values, *, name: str = "X") -> np.ndarray:
    if isinstance(values, pd.DataFrame):
        arr = values.to_numpy(dtype=float)
    elif isinstance(values, pd.Series):
        arr = values.to_frame().to_numpy(dtype=float)
    else:
        arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.size == 0:
        raise ValueError(f"{name} cannot be empty")
    return arr


def _as_dataframe(X, *, feature_names: list[str] | None = None) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X.copy()
    array = _to_2d_numeric_array(X, name="X")
    if feature_names is None:
        feature_names = [f"feature_{idx}" for idx in range(array.shape[1])]
    return pd.DataFrame(array, columns=feature_names)


def _resolve_feature_names(
    feature_names: list[str] | tuple[str, ...] | np.ndarray | pd.Index | None,
    n_features: int,
) -> list[str]:
    if feature_names is None:
        return [f"feature_{idx}" for idx in range(n_features)]
    names = list(feature_names)
    if len(names) != n_features:
        raise ValueError(
            f"feature_names length {len(names)} does not match n_features {n_features}"
        )
    return names


def _validate_same_length(**arrays) -> int:
    lengths = {name: len(np.asarray(values)) for name, values in arrays.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Input lengths must match, got: {lengths}")
    return next(iter(lengths.values()))


def _predict_positive_score(model, X) -> np.ndarray:
    X_arr = _to_2d_numeric_array(X)
    if hasattr(model, "feature_names_in_"):
        model_input = pd.DataFrame(X_arr, columns=list(model.feature_names_in_))
    else:
        model_input = X_arr
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(model_input)
        if probs.ndim == 2 and probs.shape[1] > 1:
            return probs[:, 1]
        return probs.reshape(-1)
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(model_input), dtype=float).reshape(-1)
        return _sigmoid(scores)
    preds = np.asarray(model.predict(model_input), dtype=float).reshape(-1)
    return preds


def _extract_expected_value(explainer) -> float:
    expected_value = getattr(explainer, "expected_value", 0.0)
    if isinstance(expected_value, list):
        return float(expected_value[1] if len(expected_value) > 1 else expected_value[0])
    if isinstance(expected_value, np.ndarray):
        if expected_value.ndim == 0:
            return float(expected_value.item())
        flat = expected_value.reshape(-1)
        return float(flat[1] if flat.size > 1 else flat[0])
    return float(expected_value)


def _normalize_shap_output(raw_values) -> np.ndarray:
    if isinstance(raw_values, list):
        return np.asarray(raw_values[1] if len(raw_values) > 1 else raw_values[0], dtype=float)

    values = np.asarray(raw_values, dtype=float)
    if values.ndim == 3:
        if values.shape[0] == 2:
            return values[1]
        if values.shape[-1] == 2:
            return values[:, :, 1]
    return values


def _approximate_local_contributions(model, background, X) -> np.ndarray:
    """
    在没有 SHAP 依赖时提供确定性的近似归因。

    做法：
    - 用训练集均值作为参考样本；
    - 分别只替换一个特征，计算预测概率变化；
    - 再按总变化缩放，使贡献和约等于 `prediction - baseline`。
    """
    X_arr = _to_2d_numeric_array(X, name="X_test")
    background_arr = _to_2d_numeric_array(background, name="X_train")
    reference = background_arr.mean(axis=0)
    baseline = float(_predict_positive_score(model, reference.reshape(1, -1))[0])
    predictions = _predict_positive_score(model, X_arr)

    contributions = np.zeros_like(X_arr, dtype=float)
    for row_idx, sample in enumerate(X_arr):
        deltas = np.zeros(X_arr.shape[1], dtype=float)
        for col_idx in range(X_arr.shape[1]):
            modified = reference.copy()
            modified[col_idx] = sample[col_idx]
            deltas[col_idx] = (
                float(_predict_positive_score(model, modified.reshape(1, -1))[0]) - baseline
            )

        target_shift = float(predictions[row_idx] - baseline)
        total_delta = float(deltas.sum())
        if np.isclose(total_delta, 0.0):
            magnitudes = np.abs(sample - reference)
            if np.isclose(magnitudes.sum(), 0.0):
                scaled = np.zeros_like(deltas)
            else:
                scaled = magnitudes / magnitudes.sum() * target_shift
        else:
            scaled = deltas * (target_shift / total_delta)
        contributions[row_idx] = scaled

    return contributions


@dataclass
class ApproximateShapExplainer:
    """`shap` 缺失时的最小解释器接口。"""

    model: object
    background: np.ndarray
    output_space: str = "probability"

    def __post_init__(self) -> None:
        background_arr = _to_2d_numeric_array(self.background, name="background")
        self.reference = background_arr.mean(axis=0)
        self.expected_value = float(
            _predict_positive_score(self.model, self.reference.reshape(1, -1))[0]
        )

    def shap_values(self, X):
        return _approximate_local_contributions(self.model, self.background, X)

    def __call__(self, X):
        return self.shap_values(X)


# ============================================================================
# 第一部分：特征重要性
# ============================================================================

def compute_feature_importance_from_data(X: pd.DataFrame | np.ndarray, y) -> dict:
    """
    从原始数据训练模型并计算逻辑回归系数与随机森林特征重要性。
    """
    X_df = _as_dataframe(X)
    y_arr = _to_numpy_1d(y, name="y")
    if len(X_df) != len(y_arr):
        raise ValueError("X and y must have the same number of rows")

    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_arr, test_size=0.3, random_state=42, stratify=y_arr
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(X_train_scaled, y_train)

    coef_df = get_lr_coefficients(log_reg, feature_names=X_train.columns)
    coef_df["abs_coef"] = coef_df["coefficient"].abs()
    coef_df = coef_df.sort_values("abs_coef", ascending=False).reset_index(drop=True)

    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    importance_df = compute_feature_importance(rf, feature_names=X_train.columns)

    return {
        "log_reg_coefficients": coef_df,
        "rf_importance": importance_df,
        "holdout_metrics": {
            "log_reg_accuracy": accuracy_score(y_test, log_reg.predict(X_test_scaled)),
            "rf_accuracy": accuracy_score(y_test, rf.predict(X_test)),
        },
    }


def compute_feature_importance(
    model_or_X,
    feature_names_or_y=None,
    *,
    feature_names=None,
    y=None,
):
    """
    计算特征重要性，支持两种用法：

    1. `compute_feature_importance(model, feature_names=None)`
    2. `compute_feature_importance(X, y)`
    """
    if y is not None:
        return compute_feature_importance_from_data(model_or_X, y)

    if isinstance(model_or_X, (pd.DataFrame, np.ndarray)) and not hasattr(
        model_or_X, "feature_importances_"
    ):
        return compute_feature_importance_from_data(model_or_X, feature_names_or_y)

    model = model_or_X
    if feature_names is None and feature_names_or_y is not None and not isinstance(
        feature_names_or_y, (pd.Series, np.ndarray, list, tuple, pd.Index)
    ):
        feature_names = None
    elif feature_names is None:
        feature_names = feature_names_or_y

    if hasattr(model, "feature_importances_"):
        importance = np.asarray(model.feature_importances_, dtype=float).reshape(-1)
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
        if coef.ndim == 2:
            coef = coef[0]
        importance = np.abs(coef).reshape(-1)
        total = importance.sum()
        if total > 0:
            importance = importance / total
    else:
        raise ValueError("Model must provide feature_importances_ or coef_")

    if feature_names is None:
        return importance

    names = _resolve_feature_names(feature_names, len(importance))
    return (
        pd.DataFrame({"feature": names, "importance": importance})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def get_top_features(model, feature_names, k: int = 5):
    importance_df = compute_feature_importance(model, feature_names=feature_names)
    return importance_df.head(k).copy()


def plot_feature_importance(model, feature_names, output_path: str | Path):
    importance_df = compute_feature_importance(model, feature_names=feature_names)
    setup_chinese_font()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(
        importance_df["feature"].iloc[::-1],
        importance_df["importance"].iloc[::-1],
        color="steelblue",
        alpha=0.85,
    )
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def compute_lr_feature_importance(model):
    return compute_feature_importance(model)


def get_lr_coefficients(model, feature_names=None) -> pd.DataFrame:
    if not hasattr(model, "coef_"):
        raise ValueError("Model must provide coef_")
    coef = np.asarray(model.coef_, dtype=float)
    if coef.ndim == 2:
        coef = coef[0]
    names = _resolve_feature_names(feature_names, len(coef))
    return pd.DataFrame({"feature": names, "coefficient": coef})


# ============================================================================
# 第二部分：SHAP / 近似局部解释
# ============================================================================

def create_shap_explainer(model, X_train):
    X_train_arr = _to_2d_numeric_array(X_train, name="X_train")
    if shap is None:
        return ApproximateShapExplainer(model=model, background=X_train_arr)

    try:
        return shap.TreeExplainer(model)
    except Exception:
        background = shap.sample(X_train_arr, min(len(X_train_arr), 100))
        return shap.KernelExplainer(lambda arr: _predict_positive_score(model, arr), background)


def compute_shap_values(model, X_train, X_test):
    explainer = create_shap_explainer(model, X_train)
    X_test_arr = _to_2d_numeric_array(X_test, name="X_test")

    if isinstance(explainer, ApproximateShapExplainer):
        return explainer.shap_values(X_test_arr)

    try:
        raw_values = explainer.shap_values(X_test_arr)
    except Exception:
        raw_values = explainer(X_test_arr)

    values = raw_values.values if hasattr(raw_values, "values") else raw_values
    return _normalize_shap_output(values)


def get_base_value(model, X_train):
    explainer = create_shap_explainer(model, X_train)
    if isinstance(explainer, ApproximateShapExplainer):
        return explainer.expected_value
    return _extract_expected_value(explainer)


def explain_single_prediction(model, X_test, sample_idx, feature_names, X_train=None):
    X_test_arr = _to_2d_numeric_array(X_test, name="X_test")
    background = X_train if X_train is not None else X_test_arr
    shap_values = compute_shap_values(model, background, X_test_arr)
    base_value = get_base_value(model, background)
    names = _resolve_feature_names(feature_names, X_test_arr.shape[1])

    sample_values = shap_values[sample_idx]
    prediction = float(_predict_positive_score(model, X_test_arr[[sample_idx]])[0])
    contributions = pd.DataFrame(
        {
            "feature": names,
            "value": X_test_arr[sample_idx],
            "contribution": sample_values,
        }
    ).assign(abs_contribution=lambda df: df["contribution"].abs())
    contributions = contributions.sort_values("abs_contribution", ascending=False).reset_index(
        drop=True
    )

    return {
        "sample_idx": sample_idx,
        "base_value": float(base_value),
        "prediction": prediction,
        "output_space": "probability" if shap is None else "model_output",
        "contributions": contributions,
    }


def get_feature_contributions(model, X_test, sample_idx, X_train=None, feature_names=None):
    X_test_arr = _to_2d_numeric_array(X_test, name="X_test")
    background = X_train if X_train is not None else X_test_arr
    shap_values = compute_shap_values(model, background, X_test_arr)
    names = _resolve_feature_names(feature_names, X_test_arr.shape[1])
    return pd.DataFrame(
        {
            "feature": names,
            "value": X_test_arr[sample_idx],
            "contribution": shap_values[sample_idx],
        }
    )


def compute_shap_summary(model, X_train, X_test, feature_names=None):
    X_test_arr = _to_2d_numeric_array(X_test, name="X_test")
    shap_values = compute_shap_values(model, X_train, X_test_arr)
    names = _resolve_feature_names(feature_names, X_test_arr.shape[1])
    summary = pd.DataFrame(
        {
            "feature": names,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        }
    )
    summary["importance"] = summary["mean_abs_shap"]
    return summary.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def plot_shap_summary(model, X_train, X_test, feature_names, output_path: str | Path):
    summary = compute_shap_summary(model, X_train, X_test, feature_names)
    setup_chinese_font()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(
        summary["feature"].iloc[::-1],
        summary["mean_abs_shap"].iloc[::-1],
        color="darkorange",
        alpha=0.85,
    )
    label = "Mean |contribution|" if shap is None else "Mean |SHAP value|"
    ax.set_xlabel(label)
    ax.set_title("Global Feature Contributions")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def plot_shap_waterfall(model, X_test, sample_idx, feature_names, output_path: str | Path, X_train=None):
    explanation = explain_single_prediction(
        model, X_test, sample_idx, feature_names=feature_names, X_train=X_train
    )
    contributions = explanation["contributions"].head(8).iloc[::-1]

    setup_chinese_font()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    colors = ["firebrick" if value > 0 else "seagreen" for value in contributions["contribution"]]
    ax.barh(contributions["feature"], contributions["contribution"], color=colors, alpha=0.85)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Contribution to prediction")
    ax.set_title(f"Local explanation for sample {sample_idx}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def generate_explanation_text(model, X_test, sample_idx, feature_names, X_train=None):
    explanation = explain_single_prediction(
        model, X_test, sample_idx, feature_names=feature_names, X_train=X_train
    )
    top_factors = explanation["contributions"].head(5)

    lines = [
        f"样本 #{sample_idx} 的预测解释",
        f"基线输出: {explanation['base_value']:.3f}",
        f"当前样本预测值: {explanation['prediction']:.3f}",
    ]
    if explanation["output_space"] == "probability":
        lines.append("说明：当前环境未安装 shap，下面的贡献值是概率空间下的确定性近似归因。")
    else:
        lines.append("说明：贡献值来自模型解释器；在不同解释器下，输出空间可能是 log-odds 或模型输出。")
    lines.append("")
    lines.append("主要影响因素:")
    for _, row in top_factors.iterrows():
        direction = "提高" if row["contribution"] > 0 else "降低"
        lines.append(
            f"- {row['feature']}: {direction}预测值 {abs(row['contribution']):.3f}"
        )
    return "\n".join(lines)


# ============================================================================
# 第三部分：偏见检测与公平性评估
# ============================================================================

def _group_rates(y_true, y_pred, sensitive_attr, *, min_group_size: int = 1) -> pd.DataFrame:
    y_true_arr = _to_numpy_1d(y_true, name="y_true")
    y_pred_arr = _to_numpy_1d(y_pred, name="y_pred")
    sensitive_arr = _to_numpy_1d(sensitive_attr, name="sensitive_attr")
    _validate_same_length(y_true=y_true_arr, y_pred=y_pred_arr, sensitive_attr=sensitive_arr)

    df = pd.DataFrame(
        {
            "y_true": y_true_arr.astype(int),
            "y_pred": y_pred_arr.astype(int),
            "group": sensitive_arr,
        }
    )
    results: list[dict] = []
    for group_value, group_df in df.groupby("group", dropna=False):
        if len(group_df) < min_group_size:
            continue
        cm = confusion_matrix(group_df["y_true"], group_df["y_pred"], labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        results.append(
            {
                "group": group_value,
                "count": int(len(group_df)),
                "tp": int(tp),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "accuracy": accuracy_score(group_df["y_true"], group_df["y_pred"]),
                "true_positive_rate": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
                "false_positive_rate": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
                "positive_rate": float(group_df["y_pred"].mean()),
            }
        )
    return pd.DataFrame(results)


def evaluate_group_fairness(y_true, y_pred, sensitive_groups, min_group_size: int = 10) -> pd.DataFrame:
    return _group_rates(y_true, y_pred, sensitive_groups, min_group_size=min_group_size)


def evaluate_by_group(y_true, y_pred, sensitive_attr, min_group_size: int = 1) -> pd.DataFrame:
    return _group_rates(y_true, y_pred, sensitive_attr, min_group_size=min_group_size)


def group_confusion_matrices(y_true, y_pred, sensitive_attr) -> dict:
    y_true_arr = _to_numpy_1d(y_true, name="y_true")
    y_pred_arr = _to_numpy_1d(y_pred, name="y_pred")
    sensitive_arr = _to_numpy_1d(sensitive_attr, name="sensitive_attr")
    _validate_same_length(y_true=y_true_arr, y_pred=y_pred_arr, sensitive_attr=sensitive_arr)

    result = {}
    for group in np.unique(sensitive_arr):
        mask = sensitive_arr == group
        result[group] = confusion_matrix(y_true_arr[mask], y_pred_arr[mask], labels=[0, 1])
    return result


def check_fairness_warnings(fairness_df: pd.DataFrame) -> list[str]:
    if fairness_df is None or fairness_df.empty or len(fairness_df) < 2:
        return ["警告: 需要至少 2 个分组才能进行公平性评估"]

    warnings: list[str] = []
    tpr_diff = float(
        fairness_df["true_positive_rate"].max() - fairness_df["true_positive_rate"].min()
    )
    fpr_diff = float(
        fairness_df["false_positive_rate"].max() - fairness_df["false_positive_rate"].min()
    )
    if tpr_diff > 0.1:
        warnings.append(f"警告: 真阳性率差异过大 ({tpr_diff:.3f})")
    if fpr_diff > 0.1:
        warnings.append(f"警告: 假阳性率差异过大 ({fpr_diff:.3f})")
    if not warnings:
        warnings.append("分组差异在可接受范围内")
    return warnings


def demographic_parity_difference(y_pred, sensitive_features) -> float:
    y_pred_arr = _to_numpy_1d(y_pred, name="y_pred")
    sensitive_arr = _to_numpy_1d(sensitive_features, name="sensitive_features")
    _validate_same_length(y_pred=y_pred_arr, sensitive_features=sensitive_arr)

    rates = pd.DataFrame({"y_pred": y_pred_arr, "group": sensitive_arr}).groupby("group")[
        "y_pred"
    ].mean()
    return float(rates.max() - rates.min())


def demographic_parity_ratio(y_pred, sensitive_features) -> float:
    y_pred_arr = _to_numpy_1d(y_pred, name="y_pred")
    sensitive_arr = _to_numpy_1d(sensitive_features, name="sensitive_features")
    _validate_same_length(y_pred=y_pred_arr, sensitive_features=sensitive_arr)

    rates = pd.DataFrame({"y_pred": y_pred_arr, "group": sensitive_arr}).groupby("group")[
        "y_pred"
    ].mean()
    min_rate = float(rates.min())
    max_rate = float(rates.max())
    if np.isclose(max_rate, 0.0):
        return 1.0
    return min_rate / max_rate


def equalized_odds_tpr_difference(y_true, y_pred, sensitive_attr) -> float:
    group_df = evaluate_by_group(y_true, y_pred, sensitive_attr)
    return float(group_df["true_positive_rate"].max() - group_df["true_positive_rate"].min())


def equalized_odds_fpr_difference(y_true, y_pred, sensitive_attr) -> float:
    group_df = evaluate_by_group(y_true, y_pred, sensitive_attr)
    return float(group_df["false_positive_rate"].max() - group_df["false_positive_rate"].min())


def equalized_odds_difference(y_true, y_pred, sensitive_attr):
    return {
        "tpr_diff": equalized_odds_tpr_difference(y_true, y_pred, sensitive_attr),
        "fpr_diff": equalized_odds_fpr_difference(y_true, y_pred, sensitive_attr),
    }


def equal_opportunity_difference(y_true, y_pred, sensitive_attr) -> float:
    """真正的 Equal Opportunity：只看 TPR。"""
    return equalized_odds_tpr_difference(y_true, y_pred, sensitive_attr)


def calibration_by_group(y_true, y_prob, sensitive_attr, n_bins: int = 5) -> pd.DataFrame:
    y_true_arr = _to_numpy_1d(y_true, name="y_true").astype(int)
    y_prob_arr = _to_numpy_1d(y_prob, name="y_prob").astype(float)
    sensitive_arr = _to_numpy_1d(sensitive_attr, name="sensitive_attr")
    _validate_same_length(y_true=y_true_arr, y_prob=y_prob_arr, sensitive_attr=sensitive_arr)

    if np.any((y_prob_arr < 0) | (y_prob_arr > 1)):
        raise ValueError("Probabilities must be in [0, 1]")

    rows: list[dict] = []
    for group in np.unique(sensitive_arr):
        mask = sensitive_arr == group
        true_rate = float(y_true_arr[mask].mean()) if mask.any() else np.nan
        pred_rate = float(y_prob_arr[mask].mean()) if mask.any() else np.nan
        rows.append(
            {
                "group": group,
                "count": int(mask.sum()),
                "avg_predicted_prob": pred_rate,
                "true_rate": true_rate,
                "calibration_error": abs(pred_rate - true_rate),
            }
        )
    return pd.DataFrame(rows)


def calibration_difference(y_true, y_prob, sensitive_attr) -> float:
    calibration_df = calibration_by_group(y_true, y_prob, sensitive_attr)
    return float(
        calibration_df["calibration_error"].max() - calibration_df["calibration_error"].min()
    )


def plot_calibration_curve(y_true, y_prob, sensitive_attr, output_path: str | Path):
    y_true_arr = _to_numpy_1d(y_true, name="y_true").astype(int)
    y_prob_arr = _to_numpy_1d(y_prob, name="y_prob").astype(float)
    sensitive_arr = _to_numpy_1d(sensitive_attr, name="sensitive_attr")
    _validate_same_length(y_true=y_true_arr, y_prob=y_prob_arr, sensitive_attr=sensitive_arr)

    setup_chinese_font()
    fig, ax = plt.subplots(figsize=(6, 5))
    for group in np.unique(sensitive_arr):
        mask = sensitive_arr == group
        if mask.sum() < 2:
            continue
        frac_pos, mean_pred = calibration_curve(
            y_true_arr[mask], y_prob_arr[mask], n_bins=5, strategy="uniform"
        )
        ax.plot(mean_pred, frac_pos, marker="o", label=f"group={group}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration by group")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def compute_all_fairness_metrics(y_true, y_pred, sensitive_attr, y_prob=None) -> dict:
    metrics = {
        "demographic_parity": {
            "difference": demographic_parity_difference(y_pred, sensitive_attr),
            "ratio": demographic_parity_ratio(y_pred, sensitive_attr),
        },
        "equalized_odds": equalized_odds_difference(y_true, y_pred, sensitive_attr),
        "equal_opportunity": equal_opportunity_difference(y_true, y_pred, sensitive_attr),
    }
    if y_prob is not None:
        metrics["calibration"] = {
            "difference": calibration_difference(y_true, y_prob, sensitive_attr),
            "by_group": calibration_by_group(y_true, y_prob, sensitive_attr).to_dict("records"),
        }
    return metrics


def compute_fairness_accuracy_tradeoff(y_true, y_pred, sensitive_attr, y_prob=None) -> dict:
    metrics = compute_all_fairness_metrics(y_true, y_pred, sensitive_attr, y_prob=y_prob)
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    return metrics


def compute_pareto_frontier(y_true, y_probs, sensitive_attr, thresholds=None):
    y_true_arr = _to_numpy_1d(y_true, name="y_true").astype(int)
    y_prob_arr = _to_numpy_1d(y_probs, name="y_probs").astype(float)
    sensitive_arr = _to_numpy_1d(sensitive_attr, name="sensitive_attr")
    _validate_same_length(y_true=y_true_arr, y_probs=y_prob_arr, sensitive_attr=sensitive_arr)

    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 17)

    rows = []
    for threshold in thresholds:
        y_pred = (y_prob_arr >= threshold).astype(int)
        eo = equalized_odds_difference(y_true_arr, y_pred, sensitive_arr)
        rows.append(
            {
                "threshold": float(threshold),
                "accuracy": float(accuracy_score(y_true_arr, y_pred)),
                "demographic_parity_diff": demographic_parity_difference(y_pred, sensitive_arr),
                "equalized_odds_score": max(eo["tpr_diff"], eo["fpr_diff"]),
            }
        )
    frontier = pd.DataFrame(rows)
    return frontier.sort_values(
        ["equalized_odds_score", "demographic_parity_diff", "accuracy"],
        ascending=[True, True, False],
    ).reset_index(drop=True)


def check_fairness_threshold(y_true, y_pred, sensitive_attr, threshold: float = 0.1, thresholds=None):
    metrics = compute_all_fairness_metrics(y_true, y_pred, sensitive_attr)
    if thresholds is not None:
        dp_limit = thresholds.get("demographic_parity", threshold)
        eo_limit = thresholds.get("equalized_odds", threshold)
    else:
        dp_limit = eo_limit = threshold

    eo = metrics["equalized_odds"]
    return bool(
        metrics["demographic_parity"]["difference"] <= dp_limit
        and eo["tpr_diff"] <= eo_limit
        and eo["fpr_diff"] <= eo_limit
    )


def detect_prediction_bias(y_true, y_pred, sensitive_attr) -> dict:
    group_df = evaluate_by_group(y_true, y_pred, sensitive_attr)
    report = {
        "positive_rate_diff": demographic_parity_difference(y_pred, sensitive_attr),
        "demographic_parity_ratio": demographic_parity_ratio(y_pred, sensitive_attr),
        "equalized_odds": equalized_odds_difference(y_true, y_pred, sensitive_attr),
        "group_metrics": group_df.to_dict("records"),
    }
    report["bias_detected"] = bool(
        report["positive_rate_diff"] > 0.1
        or report["equalized_odds"]["tpr_diff"] > 0.1
        or report["equalized_odds"]["fpr_diff"] > 0.1
    )
    return report


def detect_outcome_bias(y_true, sensitive_attr) -> dict:
    y_true_arr = _to_numpy_1d(y_true, name="y_true").astype(int)
    sensitive_arr = _to_numpy_1d(sensitive_attr, name="sensitive_attr")
    _validate_same_length(y_true=y_true_arr, sensitive_attr=sensitive_arr)

    rates = (
        pd.DataFrame({"y_true": y_true_arr, "group": sensitive_arr})
        .groupby("group")["y_true"]
        .mean()
    )
    return {
        "group_rates": rates.to_dict(),
        "outcome_diff": float(rates.max() - rates.min()),
    }


def detect_disparate_impact(y_pred, sensitive_attr) -> float:
    return float(demographic_parity_ratio(y_pred, sensitive_attr))


def identify_data_bias(y_true, sensitive_attr, threshold: float = 0.1) -> dict:
    outcome = detect_outcome_bias(y_true, sensitive_attr)
    outcome["has_bias"] = bool(outcome["outcome_diff"] > threshold)
    return outcome


def identify_algorithmic_bias(y_true, y_pred, sensitive_attr, threshold: float = 0.1) -> dict:
    prediction_bias = detect_prediction_bias(y_true, y_pred, sensitive_attr)
    has_bias = bool(
        prediction_bias["positive_rate_diff"] > threshold
        or prediction_bias["equalized_odds"]["tpr_diff"] > threshold
        or prediction_bias["equalized_odds"]["fpr_diff"] > threshold
    )
    prediction_bias["has_bias"] = has_bias
    return prediction_bias


def plot_group_metrics(y_true, y_pred, sensitive_attr, output_path: str | Path):
    group_df = evaluate_by_group(y_true, y_pred, sensitive_attr)
    setup_chinese_font()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    metrics = ["accuracy", "true_positive_rate", "false_positive_rate"]
    titles = ["Accuracy", "TPR", "FPR"]
    for ax, metric, title in zip(axes, metrics, titles):
        ax.bar(group_df["group"].astype(str), group_df[metric], color="steelblue", alpha=0.85)
        ax.set_ylim(0, 1)
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def plot_group_confusion_matrices(y_true, y_pred, sensitive_attr, output_path: str | Path):
    cms = group_confusion_matrices(y_true, y_pred, sensitive_attr)
    setup_chinese_font()

    fig, axes = plt.subplots(1, len(cms), figsize=(4.5 * len(cms), 4), constrained_layout=True)
    if len(cms) == 1:
        axes = [axes]
    for ax, (group, cm) in zip(axes, cms.items()):
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"group={group}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for row in range(2):
            for col in range(2):
                ax.text(col, row, int(cm[row, col]), ha="center", va="center")
    fig.colorbar(im, ax=axes)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def check_bias_threshold(y_true, y_pred, sensitive_attr, threshold: float = 0.1) -> bool:
    bias_report = detect_prediction_bias(y_true, y_pred, sensitive_attr)
    eo = bias_report["equalized_odds"]
    max_bias = max(
        bias_report["positive_rate_diff"],
        eo["tpr_diff"],
        eo["fpr_diff"],
    )
    return bool(max_bias > threshold)


def evaluate_intersectional_groups(y_true, y_pred, sensitive_attrs: dict[str, np.ndarray]):
    combined = pd.DataFrame({name: _to_numpy_1d(values, name=name) for name, values in sensitive_attrs.items()})
    combined_group = combined.astype(str).agg(" | ".join, axis=1).to_numpy()
    return evaluate_by_group(y_true, y_pred, combined_group)


def detect_intersectional_bias(y_true, y_pred, sensitive_attrs: dict[str, np.ndarray]) -> dict:
    group_df = evaluate_intersectional_groups(y_true, y_pred, sensitive_attrs)
    if group_df.empty:
        return {"bias_detected": False, "group_metrics": []}
    return {
        "bias_detected": bool(
            (group_df["true_positive_rate"].max() - group_df["true_positive_rate"].min()) > 0.1
            or (group_df["false_positive_rate"].max() - group_df["false_positive_rate"].min()) > 0.1
        ),
        "group_metrics": group_df.to_dict("records"),
    }


def evaluate_multiple_sensitive_attrs(y_true, y_pred, sensitive_attrs: dict[str, np.ndarray]) -> dict:
    return {
        attr_name: compute_all_fairness_metrics(y_true, y_pred, attr_values)
        for attr_name, attr_values in sensitive_attrs.items()
    }


def get_fairness_improvement_suggestions(y_true, y_pred, sensitive_attr):
    metrics = compute_all_fairness_metrics(y_true, y_pred, sensitive_attr)
    suggestions = [
        "检查训练样本是否失衡；必要时对少数群体做重采样（resample / oversample）或重新加权（reweight）。",
        "在特征工程阶段审计敏感属性的代理变量，必要时做预处理约束。",
        "对不同群体单独调阈值或做后处理校准，明确是在追求 demographic parity、equal opportunity 还是 equalized odds。",
        "在部署后持续监控分组 TPR、FPR 和预测正率，而不是只看整体准确率。",
    ]
    eo = metrics["equalized_odds"]
    if eo["tpr_diff"] > 0.1:
        suggestions.append("当前 TPR 差异较大，优先检查是否对某些群体漏报过多。")
    if eo["fpr_diff"] > 0.1:
        suggestions.append("当前 FPR 差异较大，优先检查是否对某些群体误报过多。")
    return suggestions


# ============================================================================
# 第四部分：报告生成
# ============================================================================

def generate_fairness_report(y_true, y_pred, sensitive_attr, attr_name: str = "group", y_prob=None) -> str:
    metrics = compute_all_fairness_metrics(y_true, y_pred, sensitive_attr, y_prob=y_prob)
    bias_report = detect_prediction_bias(y_true, y_pred, sensitive_attr)
    suggestions = get_fairness_improvement_suggestions(y_true, y_pred, sensitive_attr)

    lines = [
        f"# Fairness Report for {attr_name}",
        "",
        "## Summary",
        f"- Demographic parity difference: {metrics['demographic_parity']['difference']:.3f}",
        f"- Demographic parity ratio: {metrics['demographic_parity']['ratio']:.3f}",
        f"- Equalized odds TPR difference: {metrics['equalized_odds']['tpr_diff']:.3f}",
        f"- Equalized odds FPR difference: {metrics['equalized_odds']['fpr_diff']:.3f}",
        f"- Equal opportunity (TPR only) difference: {metrics['equal_opportunity']:.3f}",
        "",
        "## Interpretation",
        "- `Demographic parity` 看的是不同群体拿到正向结果的比例是否接近。",
        "- `Equalized odds` 同时比较 TPR 和 FPR；如果同时谈这两项，就不应误称为机会均等。",
        "- `Equal opportunity` 只比较 TPR，适合强调不要漏掉真正正例的场景。",
        "",
        "## Bias Check",
        f"- Bias detected: {'yes' if bias_report['bias_detected'] else 'no'}",
        "",
        "## Recommendations",
    ]
    lines.extend(f"- {item}" for item in suggestions)
    return "\n".join(lines)


def generate_bias_report(y_true, y_pred, sensitive_attr, attr_name: str = "group") -> str:
    group_df = evaluate_by_group(y_true, y_pred, sensitive_attr)
    bias_report = detect_prediction_bias(y_true, y_pred, sensitive_attr)

    lines = [
        f"# Bias Report for {attr_name}",
        "",
        "## Group Metrics",
    ]
    for _, row in group_df.iterrows():
        lines.append(
            f"- {row['group']}: accuracy={row['accuracy']:.3f}, "
            f"TPR={row['true_positive_rate']:.3f}, FPR={row['false_positive_rate']:.3f}"
        )
    lines.extend(
        [
            "",
            "## Bias Summary",
            f"- Prediction-rate difference: {bias_report['positive_rate_diff']:.3f}",
            f"- Equalized odds TPR difference: {bias_report['equalized_odds']['tpr_diff']:.3f}",
            f"- Equalized odds FPR difference: {bias_report['equalized_odds']['fpr_diff']:.3f}",
            "",
            "## Mitigation Suggestions",
        ]
    )
    lines.extend(f"- {item}" for item in get_fairness_improvement_suggestions(y_true, y_pred, sensitive_attr))
    return "\n".join(lines)


def generate_explanation_report(model_metrics: dict, feature_importance: pd.DataFrame, fairness_df: pd.DataFrame) -> str:
    auc = float(model_metrics.get("auc", 0.0))
    recall = float(model_metrics.get("recall", 0.0))
    precision = float(model_metrics.get("precision", 0.0))

    lines = [
        "# 模型解释报告",
        "",
        "## 模型性能",
        f"- AUC = {auc:.2f}：表示模型把正例排在负例前面的排序能力较强，不等于“{auc:.0%} 都预测对了”。",
        f"- 召回率 = {recall:.0%}：在真实正例中，模型抓到了多少。",
        f"- 精确率 = {precision:.0%}：在模型判为正例的样本中，有多少真的为正。",
        "",
        "## 主要影响因素",
    ]

    for _, row in feature_importance.head(5).iterrows():
        lines.append(f"- {row['feature']}")

    lines.extend(["", "## 公平性说明"])
    if fairness_df is not None and not fairness_df.empty:
        for _, row in fairness_df.iterrows():
            lines.append(
                f"- {row['group']} 组：准确率 {row['accuracy']:.1%}，TPR {row['true_positive_rate']:.1%}，FPR {row['false_positive_rate']:.1%}"
            )
        for warning in check_fairness_warnings(fairness_df):
            lines.append(f"- {warning}")
    else:
        lines.append("- 当前没有足够分组信息，无法评估公平性。")

    lines.extend(
        [
            "",
            "## 行动建议",
            "- 该模型适合辅助排序或预警，不应替代人工判断。",
            "- 报告给非技术读者时，应说明解释值对应的输出空间；不要把 SHAP 值直接说成“概率百分点”。",
            "- 如需做显著性检验，应写清原假设、检验方法和 p 值含义；p < 0.05 不等于“95% 把握结论为真”。",
        ]
    )
    return "\n".join(lines)


# ============================================================================
# 演示主函数
# ============================================================================

def main() -> None:
    print("=" * 60)
    print("Week 12 作业参考实现演示")
    print("=" * 60)

    output_dir = Path(__file__).parent.parent.parent.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    n = 1000
    X = pd.DataFrame(
        {
            "feature_1": np.random.randn(n),
            "feature_2": np.random.randn(n),
            "feature_3": np.random.randn(n),
        }
    )
    logit = -1 + 0.8 * X["feature_1"] - 0.4 * X["feature_2"]
    prob = _sigmoid(logit.to_numpy())
    y = np.random.binomial(1, prob)
    X["group"] = np.random.choice(["A", "B", "C"], n, p=[0.5, 0.3, 0.2])

    importance_results = compute_feature_importance_from_data(X.drop(columns="group"), y)
    print("\n随机森林特征重要性:")
    print(importance_results["rf_importance"].head())

    X_train, X_test, y_train, y_test = train_test_split(
        X.drop(columns="group"),
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )
    model = RandomForestClassifier(n_estimators=80, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    shap_summary = compute_shap_summary(model, X_train, X_test, feature_names=X_test.columns)
    print("\n全局贡献摘要:")
    print(shap_summary.head())

    fairness_df = evaluate_by_group(
        y_test,
        model.predict(X_test),
        X.loc[X_test.index, "group"].to_numpy(),
    )
    report = generate_explanation_report(
        {
            "auc": roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]),
            "precision": precision_score(y_test, model.predict(X_test), zero_division=0),
            "recall": recall_score(y_test, model.predict(X_test), zero_division=0),
        },
        importance_results["rf_importance"],
        fairness_df,
    )
    report_path = output_dir / "solution_explanation_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"\n报告已保存到: {report_path}")


if __name__ == "__main__":
    main()

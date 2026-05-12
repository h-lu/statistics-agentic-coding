"""
示例：Pipeline 防止数据泄漏——正确使用交叉验证

输入：
  - 带缺失值的客户流失特征表
  - 训练/测试划分前后的预处理流程

输出：
  - 有无 Pipeline 的交叉验证 AUC 对比
  - 预处理泄漏与目标泄漏的可视化对比图

核心概念：
  - 数据泄漏与交叉验证污染
  - Pipeline / ColumnTransformer 的作用
  - 为什么任何 fit 都不该偷看验证集

常见错误：
  - 在划分数据前就 fit 填补器或标准化器
  - 在全量数据上做特征工程后再交叉验证
  - 把高分数误读为模型真的更强

运行方式：python3 chapters/week_10/examples/05_pipeline_data_leakage.py
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

STARTER_CODE_DIR = Path(__file__).resolve().parents[1] / 'starter_code'
if str(STARTER_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(STARTER_CODE_DIR))

from week_10 import load_customer_churn_data


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


def create_churn_data_with_missing(seed: int = 42) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """加载 churn 主线数据，并注入少量缺失值模拟真实场景。"""
    rng = np.random.default_rng(seed)
    df = load_customer_churn_data().copy()

    for col in ['avg_spend', 'days_since_last_purchase', 'membership_days']:
        mask = rng.random(len(df)) < 0.12
        df.loc[mask, col] = np.nan

    X = df.drop(columns=['is_churned'])
    y = df['is_churned']
    numeric_features = [
        'purchase_count',
        'avg_spend',
        'days_since_last_purchase',
        'membership_days',
        'support_tickets',
    ]
    categorical_features = ['contract_type']
    return X, y, numeric_features, categorical_features


def build_pipeline(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    """构建无泄漏的预处理 + 逻辑回归流水线。"""
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    return Pipeline([
        ('preprocessor', preprocessor),
        ('model', LogisticRegression(max_iter=1000, random_state=42))
    ])


def bad_preprocessing_before_split() -> dict:
    """错误示范：在划分和交叉验证之前先 fit 预处理。"""
    print("=" * 60)
    print("错误示范：在划分之前做预处理（泄漏了验证集统计量）")
    print("=" * 60)

    X, y, numeric_features, categorical_features = create_churn_data_with_missing()
    print("\n原始数据缺失值情况：")
    print(X.isnull().sum())

    X_filled = X.copy()
    for col in numeric_features:
        X_filled[col] = X_filled[col].fillna(X_filled[col].median())
    for col in categorical_features:
        X_filled[col] = X_filled[col].fillna(X_filled[col].mode()[0])

    X_encoded = pd.get_dummies(X_filled, columns=categorical_features, drop_first=False)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_encoded),
        columns=X_encoded.columns,
        index=X_encoded.index,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_prob)
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='roc_auc')

    print(f"\n测试集 AUC: {test_auc:.4f}")
    print(f"交叉验证 AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print("\n问题分析：")
    print("  - 缺失值填充先看了全部样本的中位数。")
    print("  - 标准化先看了全部样本的均值和方差。")
    print("  - 这类泄漏通常让验证分数略微偏高，看起来比真实情况更稳。")

    return {'test_auc': test_auc, 'cv_mean': cv_scores.mean(), 'cv_std': cv_scores.std()}


def good_pipeline_approach() -> dict:
    """正确做法：把预处理放进 Pipeline。"""
    print("\n" + "=" * 60)
    print("正确做法：使用 Pipeline 防止预处理泄漏")
    print("=" * 60)

    X, y, numeric_features, categorical_features = create_churn_data_with_missing()
    pipeline = build_pipeline(numeric_features, categorical_features)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    pipeline.fit(X_train, y_train)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_prob)
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='roc_auc')

    print(f"\n测试集 AUC: {test_auc:.4f}")
    print(f"交叉验证 AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print("\nPipeline 的作用：")
    print("  - 每一折先在训练子集上 fit 预处理。")
    print("  - 再把同一套参数应用到验证子集。")
    print("  - 评估因此更诚实，不会偷看验证集统计量。")

    return {'test_auc': test_auc, 'cv_mean': cv_scores.mean(), 'cv_std': cv_scores.std()}


def demonstrate_target_leakage() -> dict:
    """演示目标泄漏：异常虚高的 AUC 才是更典型的泄漏信号。"""
    print("\n" + "=" * 60)
    print("目标泄漏示范：分数异常虚高才要优先怀疑泄漏")
    print("=" * 60)

    X, y, numeric_features, categorical_features = create_churn_data_with_missing()
    rng = np.random.default_rng(42)
    X_leaky = X.copy()

    # 这是典型坏特征：它几乎直接复述了标签，现实预测时拿不到。
    X_leaky['retention_call_outcome'] = np.where(
        y.to_numpy() == 1,
        'cancelled_after_call',
        rng.choice(['stayed', 'renewed_contract'], size=len(X_leaky)),
    )

    pipeline_clean = build_pipeline(numeric_features, categorical_features)
    clean_scores = cross_val_score(pipeline_clean, X, y, cv=5, scoring='roc_auc')

    leaky_pipeline = build_pipeline(
        numeric_features,
        categorical_features + ['retention_call_outcome'],
    )
    leaky_scores = cross_val_score(leaky_pipeline, X_leaky, y, cv=5, scoring='roc_auc')

    print(f"\n正常特征的交叉验证 AUC: {clean_scores.mean():.4f} (+/- {clean_scores.std():.4f})")
    print(f"带泄漏特征的交叉验证 AUC: {leaky_scores.mean():.4f} (+/- {leaky_scores.std():.4f})")
    print("\n解读：")
    print("  - Pipeline 只能防预处理泄漏，防不了目标泄漏。")
    print("  - 如果验证 AUC 突然接近 1.0，要先审查特征是不是偷带了未来信息或标签信息。")
    print("  - AUC 接近 0.5 更常见的解释是模型弱、特征弱或任务本身难。")

    return {'clean_auc': clean_scores.mean(), 'leaky_auc': leaky_scores.mean()}


def plot_leakage_comparison(pre_bad: dict, pre_good: dict, target_leakage: dict) -> None:
    """可视化两类泄漏对分数的影响。"""
    font = setup_chinese_font()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(
        ['先预处理再 CV', 'Pipeline + CV'],
        [pre_bad['cv_mean'], pre_good['cv_mean']],
        color=['#d95f02', '#1b9e77'],
        alpha=0.8,
    )
    axes[0].set_ylabel('平均 AUC')
    axes[0].set_title('预处理泄漏 vs 正确 Pipeline')
    axes[0].grid(True, alpha=0.3, axis='y')

    axes[1].bar(
        ['正常特征', '带目标泄漏特征'],
        [target_leakage['clean_auc'], target_leakage['leaky_auc']],
        color=['#7570b3', '#e7298a'],
        alpha=0.8,
    )
    axes[1].set_ylabel('平均 AUC')
    axes[1].set_title('目标泄漏会把分数顶得异常高')
    axes[1].grid(True, alpha=0.3, axis='y')

    for ax in axes:
        for patch in ax.patches:
            height = patch.get_height()
            ax.text(
                patch.get_x() + patch.get_width() / 2,
                height + 0.01,
                f'{height:.4f}',
                ha='center',
                va='bottom',
                fontsize=10,
            )

    plt.tight_layout()
    output_dir = Path(__file__).parent.parent / 'images'
    output_dir.mkdir(exist_ok=True)
    plt.savefig(
        output_dir / 'pipeline_leakage_comparison.png',
        dpi=150,
        bbox_inches='tight',
        facecolor='white',
        edgecolor='none',
    )
    plt.close()
    print(f"\n图片已保存到: {output_dir / 'pipeline_leakage_comparison.png'}")


def demonstrate_data_leakage_checklist() -> None:
    """给出修正后的泄漏诊断清单。"""
    print("\n" + "=" * 60)
    print("数据泄漏检查清单")
    print("=" * 60)

    X, y, numeric_features, categorical_features = create_churn_data_with_missing()
    pipeline = build_pipeline(numeric_features, categorical_features)
    model_scores = cross_val_score(pipeline, X, y, cv=5, scoring='roc_auc')

    dummy = DummyClassifier(strategy='most_frequent')
    dummy_scores = cross_val_score(dummy, X, y, cv=5, scoring='roc_auc')

    print("""
数据泄漏常见场景与防御：

| 场景 | 错误做法 | 正确做法 |
|------|---------|---------|
| 预处理 | 划分前先填充/标准化 | 放进 Pipeline |
| 特征选择 | 用全量数据挑特征 | 在交叉验证折内完成 |
| 业务特征 | 用未来事件或标签衍生列 | 审查特征定义，确认预测时可获得 |

如何检查数据泄漏？
1. 先看验证分数是否异常虚高，比如 AUC 突然接近 1.0。
2. 把可疑特征删掉、把预处理移进 Pipeline，看分数是否明显回落。
3. 检查时间边界：预测时拿不到的信息，一律不能做特征。
4. 用 dummy baseline 看“模型是否有信号”，但不要把“接近 baseline”误诊成泄漏。
""")

    print(f"你的模型 AUC: {model_scores.mean():.4f} (+/- {model_scores.std():.4f})")
    print(f"Dummy baseline AUC: {dummy_scores.mean():.4f} (+/- {dummy_scores.std():.4f})")
    print("说明：dummy baseline 主要用于判断模型有没有学到有效信号。")


def main() -> None:
    """主函数"""
    pre_bad = bad_preprocessing_before_split()
    pre_good = good_pipeline_approach()
    target_leakage = demonstrate_target_leakage()
    plot_leakage_comparison(pre_bad, pre_good, target_leakage)
    demonstrate_data_leakage_checklist()

    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print("""
数据泄漏与防御核心要点：

1. 预处理泄漏：
   - 在划分之前 fit 填充器或标准化器，会让验证分数轻微虚高。
   - Pipeline 可以修复这类问题。

2. 目标泄漏：
   - 特征里混入了标签信息或未来信息，验证分数会异常漂亮。
   - 这类问题不是靠 Pipeline 自动修，而是靠特征审计修。

3. 正确诊断方式：
   - AUC 接近 baseline，通常说明模型弱或特征弱。
   - AUC 异常接近 1.0，才更值得优先怀疑泄漏。
""")


if __name__ == "__main__":
    main()

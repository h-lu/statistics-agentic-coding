"""
示例：公平性指标——如何量化"不公平"？

本例演示：
1. 检测模型在不同群体上的性能差异（AUC、准确率）
2. 计算差异影响比（Disparate Impact Ratio）
3. 计算平等机会差异（Equal Opportunity）
4. 计算均等几率（Equalized Odds）

运行方式：python3 chapters/week_12/examples/02_fairness_metrics.py
预期输出：
- 不同群体的性能指标对比表
- 公平性指标（差异影响比、平等机会、均等几率）
- 阈值调整对公平性的影响

依赖安装：
pip install pandas numpy scikit-learn matplotlib
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, recall_score,
    precision_score, confusion_matrix
)

OUTPUT_DIR = Path(__file__).parent.parent.parent / "output" / "week_12"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 设置随机种子保证可复现
np.random.seed(42)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def generate_biased_data(n_samples: int = 2000) -> pd.DataFrame:
    """
    生成带有偏见模式的模拟数据

    场景：信用评分
    - 数据中存在历史偏见：女性申请人的违约率被高估
    - 特征包括收入、信用历史、债务收入比、信用查询次数
    - 敏感属性：性别（0=男性，1=女性）
    """
    np.random.seed(42)

    # 生成样本
    n_male = int(n_samples * 0.6)
    n_female = n_samples - n_male

    # 男性样本
    male_income = np.random.lognormal(10.2, 0.5, n_male)
    male_credit_age = np.random.uniform(2, 20, n_male)
    male_debt_ratio = np.random.beta(2, 5, n_male) * 0.6
    male_inquiries = np.random.poisson(1.5, n_male)

    # 女性样本（故意引入偏见：收入略低，债务略高）
    female_income = np.random.lognormal(10.0, 0.5, n_female)  # 收入略低
    female_credit_age = np.random.uniform(0, 18, n_female)
    female_debt_ratio = np.random.beta(2.2, 5, n_female) * 0.7  # 债务略高
    female_inquiries = np.random.poisson(2.0, n_female)  # 查询略多

    # 合并数据
    income = np.concatenate([male_income, female_income])
    credit_age = np.concatenate([male_credit_age, female_credit_age])
    debt_ratio = np.concatenate([male_debt_ratio, female_debt_ratio])
    inquiries = np.concatenate([male_inquiries, female_inquiries])
    gender = np.concatenate([np.zeros(n_male), np.ones(n_female)])

    # 生成目标（引入历史偏见）
    # 男性：真实的违约模式
    male_logit = (
        -4.0
        + 0.5 * np.log(male_income / 10000)
        - 0.1 * male_credit_age
        + 3.0 * male_debt_ratio
        + 0.3 * male_inquiries
    )
    male_prob = 1 / (1 + np.exp(-male_logit))
    male_default = np.random.binomial(1, male_prob)

    # 女性：额外的"偏见项"（+0.3 logit）
    female_logit = (
        -4.0
        + 0.5 * np.log(female_income / 10000)
        - 0.1 * female_credit_age
        + 3.0 * female_debt_ratio
        + 0.3 * female_inquiries
        + 0.3  # 偏见项：女性被"额外"惩罚
    )
    female_prob = 1 / (1 + np.exp(-female_logit))
    female_default = np.random.binomial(1, female_prob)

    default = np.concatenate([male_default, female_default])

    df = pd.DataFrame({
        'income': income,
        'credit_history_age': credit_age,
        'debt_to_income': debt_ratio,
        'credit_inquiries': inquiries,
        'gender': gender,  # 0=男性，1=女性
        'default': default
    })

    # 打乱顺序
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    return df


def disparate_impact_ratio(y_pred, group_a_mask, group_b_mask):
    """
    计算差异影响比

    定义：群体 A 的通过率 / 群体 B 的通过率
    通过率 = 预测为正类（通过）的比例

    80% 规则：如果差异影响比 < 0.8，可能存在歧视
    """
    pass_rate_a = y_pred[group_a_mask].mean()
    pass_rate_b = y_pred[group_b_mask].mean()

    if pass_rate_b == 0:
        return None

    return pass_rate_a / pass_rate_b


def equal_opportunity_difference(y_true, y_pred, group_a_mask, group_b_mask):
    """
    计算平等机会差异

    定义：两个群体的召回率（TPR）之差
    召回率 = 真正为正类的样本中被正确预测的比例

    完全平等：差异 = 0
    """
    # 真正违约的样本
    true_positive_a = (y_true[group_a_mask] == 1) & (y_pred[group_a_mask] == 1)
    true_positive_b = (y_true[group_b_mask] == 1) & (y_pred[group_b_mask] == 1)

    actual_positive_a = (y_true[group_a_mask] == 1)
    actual_positive_b = (y_true[group_b_mask] == 1)

    if actual_positive_a.sum() == 0 or actual_positive_b.sum() == 0:
        return None

    tpr_a = true_positive_a.sum() / actual_positive_a.sum()
    tpr_b = true_positive_b.sum() / actual_positive_b.sum()

    return tpr_a - tpr_b


def equalized_odds(y_true, y_pred, group_a_mask, group_b_mask):
    """
    检查均等几率（Equalized Odds）

    定义：两个群体的召回率（TPR）和假阳性率（FPR）都相等

    返回：(TPR 相近?, FPR 相近?, TPR 差异, FPR 差异)
    """
    # 计算混淆矩阵
    try:
        tn_a, fp_a, fn_a, tp_a = confusion_matrix(
            y_true[group_a_mask], y_pred[group_a_mask]
        ).ravel()
        tn_b, fp_b, fn_b, tp_b = confusion_matrix(
            y_true[group_b_mask], y_pred[group_b_mask]
        ).ravel()
    except ValueError:
        # 某个群体可能没有正类或负类样本
        return None, None, None, None

    # TPR（召回率）
    tpr_a = tp_a / (tp_a + fn_a) if (tp_a + fn_a) > 0 else 0
    tpr_b = tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else 0

    # FPR（假阳性率）
    fpr_a = fp_a / (fp_a + tn_a) if (fp_a + tn_a) > 0 else 0
    fpr_b = fp_b / (fp_b + tn_b) if (fp_b + tn_b) > 0 else 0

    # 检查是否相近（差异 < 0.05）
    tpr_close = abs(tpr_a - tpr_b) < 0.05
    fpr_close = abs(fpr_a - fpr_b) < 0.05

    return tpr_close, fpr_close, tpr_a - tpr_b, fpr_a - fpr_b


def evaluate_fairness():
    """完整的公平性评估流程"""
    print("=" * 70)
    print("公平性指标示例：量化模型的不公平程度")
    print("=" * 70)

    # 生成数据
    print("\n生成带有历史偏见的模拟数据...")
    df = generate_biased_data(n_samples=2000)
    print(f"数据形状: {df.shape}")
    print(f"\n性别分布：")
    print(df['gender'].value_counts().rename({0: '男性', 1: '女性'}))
    print(f"\n按性别的违约率：")
    print(df.groupby('gender')['default'].mean().rename({0: '男性', 1: '女性'}))

    # 准备特征和目标
    feature_cols = ['income', 'credit_history_age', 'debt_to_income', 'credit_inquiries']
    # 注意：性别不在训练特征中（模拟"删除敏感变量"的常见做法）

    X = df[feature_cols]
    y = df['default']

    # 划分数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 训练模型（不使用性别特征）
    print("\n训练随机森林（不包含性别特征）...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    # 预测
    y_pred_proba = rf.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # ========== 1. 按性别分组的性能差异 ==========
    print("\n" + "-" * 70)
    print("1. 按性别分组的性能差异")
    print("-" * 70)

    male_mask = X_test.index.isin(df[df['gender'] == 0].index)
    female_mask = X_test.index.isin(df[df['gender'] == 1].index)

    # 需要重新获取测试集中的性别
    # 简化处理：使用原始索引
    test_indices = y_test.index
    gender_test = df.loc[test_indices, 'gender']
    male_mask = gender_test == 0
    female_mask = gender_test == 1

    # AUC
    male_auc = roc_auc_score(y_test[male_mask], y_pred_proba[male_mask])
    female_auc = roc_auc_score(y_test[female_mask], y_pred_proba[female_mask])

    print(f"\nAUC（区分能力）：")
    print(f"  男性:  {male_auc:.3f}")
    print(f"  女性:  {female_auc:.3f}")
    print(f"  差异:  {abs(male_auc - female_auc):.3f}")

    if abs(male_auc - female_auc) > 0.05:
        print(f"  ⚠️  警告：AUC 差异 > 0.05，模型在不同群体上的性能存在显著差异")

    # 准确率
    male_acc = accuracy_score(y_test[male_mask], y_pred[male_mask])
    female_acc = accuracy_score(y_test[female_mask], y_pred[female_mask])

    print(f"\n准确率：")
    print(f"  男性:  {male_acc:.3f}")
    print(f"  女性:  {female_acc:.3f}")
    print(f"  差异:  {abs(male_acc - female_acc):.3f}")

    # ========== 2. 差异影响比 ==========
    print("\n" + "-" * 70)
    print("2. 差异影响比（Disparate Impact Ratio）")
    print("-" * 70)
    print("\n定义：女性通过率 / 男性通过率")
    print("      （通过率 = 预测为'不违约'的比例）")
    print("      80% 规则：如果 < 0.8，可能存在歧视")

    # 通过率（预测为不违约的比例）
    male_pass_rate = (1 - y_pred[male_mask]).mean()
    female_pass_rate = (1 - y_pred[female_mask]).mean()

    print(f"\n通过率（预测为不违约）：")
    print(f"  男性:  {male_pass_rate:.3f}")
    print(f"  女性:  {female_pass_rate:.3f}")

    di_ratio = female_pass_rate / male_pass_rate if male_pass_rate > 0 else None
    print(f"\n差异影响比（女性/男性）: {di_ratio:.3f}")

    if di_ratio < 0.8:
        print(f"  ⚠️  警告：差异影响比 < 0.8，不符合 80% 规则，可能存在法律风险")
    elif di_ratio < 0.9:
        print(f"  ⚠️  注意：差异影响比 < 0.9，存在一定的公平性问题")
    else:
        print(f"  ✓ 差异影响比 >= 0.9，公平性尚可")

    # ========== 3. 平等机会差异 ==========
    print("\n" + "-" * 70)
    print("3. 平等机会差异（Equal Opportunity Difference）")
    print("-" * 70)
    print("\n定义：不同群体的召回率（TPR）之差")
    print("      召回率 = 真正违约的人中被识别的比例")

    male_tpr = recall_score(y_test[male_mask], y_pred[male_mask], zero_division=0)
    female_tpr = recall_score(y_test[female_mask], y_pred[female_mask], zero_division=0)

    print(f"\n召回率（真正违约的人中被识别的比例）：")
    print(f"  男性:  {male_tpr:.3f}")
    print(f"  女性:  {female_tpr:.3f}")
    print(f"  差异:  {abs(male_tpr - female_tpr):.3f}")

    if abs(male_tpr - female_tpr) > 0.1:
        print(f"  ⚠️  警告：召回率差异 > 0.1，模型在识别真正高风险客户时存在群体差异")

    # ========== 4. 均等几率 ==========
    print("\n" + "-" * 70)
    print("4. 均等几率（Equalized Odds）")
    print("-" * 70)
    print("\n定义：要求不同群体的 TPR 和 FPR 都相等")

    tpr_close, fpr_close, tpr_diff, fpr_diff = equalized_odds(
        y_test.values, y_pred, male_mask, female_mask
    )

    if tpr_close is not None:
        # 计算假阳性率
        tn_m, fp_m, fn_m, tp_m = confusion_matrix(
            y_test[male_mask], y_pred[male_mask]
        ).ravel()
        tn_f, fp_f, fn_f, tp_f = confusion_matrix(
            y_test[female_mask], y_pred[female_mask]
        ).ravel()

        male_fpr = fp_m / (fp_m + tn_m) if (fp_m + tn_m) > 0 else 0
        female_fpr = fp_f / (fp_f + tn_f) if (fp_f + tn_f) > 0 else 0

        print(f"\n假阳性率（实际不违约但被预测为违约的比例）：")
        print(f"  男性:  {male_fpr:.3f}")
        print(f"  女性:  {female_fpr:.3f}")
        print(f"  差异:  {abs(male_fpr - female_fpr):.3f}")

        print(f"\n均等几率是否满足：")
        print(f"  TPR 相近（差异 < 0.05）: {tpr_close}")
        print(f"  FPR 相近（差异 < 0.05）: {fpr_close}")

        if not (tpr_close and fpr_close):
            print(f"  ⚠️  警告：均等几率不满足，模型在不同群体上的错误模式不一致")

    # ========== 5. 阈值调整的影响 ==========
    print("\n" + "-" * 70)
    print("5. 阈值调整对公平性的影响")
    print("-" * 70)
    print("\n尝试不同阈值，观察公平性指标的变化...")

    thresholds = np.linspace(0.3, 0.7, 9)
    results = []

    for thresh in thresholds:
        y_pred_t = (y_pred_proba >= thresh).astype(int)

        # 通过率
        male_pass = (1 - y_pred_t[male_mask]).mean()
        female_pass = (1 - y_pred_t[female_mask]).mean()
        di = female_pass / male_pass if male_pass > 0 else None

        # AUC
        male_acc_t = accuracy_score(y_test[male_mask], y_pred_t[male_mask])
        female_acc_t = accuracy_score(y_test[female_mask], y_pred_t[female_mask])

        # 召回率
        male_tpr_t = recall_score(y_test[male_mask], y_pred_t[male_mask], zero_division=0)
        female_tpr_t = recall_score(y_test[female_mask], y_pred_t[female_mask], zero_division=0)

        results.append({
            'threshold': thresh,
            'male_acc': male_acc_t,
            'female_acc': female_acc_t,
            'di_ratio': di,
            'male_tpr': male_tpr_t,
            'female_tpr': female_tpr_t,
        })

    results_df = pd.DataFrame(results)

    print("\n阈值 vs 公平性指标：")
    print(results_df[['threshold', 'di_ratio', 'male_acc', 'female_acc',
                      'male_tpr', 'female_tpr']].round(3))

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：差异影响比 vs 阈值
    ax1 = axes[0]
    ax1.plot(results_df['threshold'], results_df['di_ratio'], 'o-', linewidth=2, markersize=8)
    ax1.axhline(y=0.8, color='r', linestyle='--', label='80% 规则')
    ax1.axhline(y=0.9, color='orange', linestyle='--', alpha=0.7, label='90% 目标')
    ax1.set_xlabel('决策阈值', fontsize=12)
    ax1.set_ylabel('差异影响比（女性/男性）', fontsize=12)
    ax1.set_title('阈值对公平性的影响', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 右图：准确率 vs 阈值
    ax2 = axes[1]
    ax2.plot(results_df['threshold'], results_df['male_acc'], 'o-', label='男性', linewidth=2, markersize=8)
    ax2.plot(results_df['threshold'], results_df['female_acc'], 's-', label='女性', linewidth=2, markersize=8)
    ax2.set_xlabel('决策阈值', fontsize=12)
    ax2.set_ylabel('准确率', fontsize=12)
    ax2.set_title('不同群体的准确率对比', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fairness_threshold_analysis.png', dpi=150, bbox_inches='tight')
    print("\n✅ 已保存: fairness_threshold_analysis.png")
    plt.close()

    # ========== 总结 ==========
    print("\n" + "=" * 70)
    print("总结：公平性指标对比")
    print("=" * 70)
    print(f"""
本次评估的模型公平性指标：

1. 性能差异：
   - 男性 AUC: {male_auc:.3f}
   - 女性 AUC: {female_auc:.3f}
   - 差异: {abs(male_auc - female_auc):.3f} {'⚠️ 超过 0.05' if abs(male_auc - female_auc) > 0.05 else '✓ 可接受'}

2. 差异影响比：{di_ratio:.3f}
   - 80% 规则: {'❌ 不符合（< 0.8）' if di_ratio < 0.8 else '✓ 符合'}

3. 平等机会差异：{abs(male_tpr - female_tpr):.3f}
   - 女性 TPR: {female_tpr:.3f}（识别真正高风险女性的能力）
   - 男性 TPR: {male_tpr:.3f}（识别真正高风险男性的能力）

4. 均等几率：{'✓ 满足' if (tpr_close and fpr_close) else '❌ 不满足'}

公平性 trade-off：
- 降低阈值可以提高整体通过率，但可能扩大群体差异
- 提高阈值可以减少假阳性，但可能增加假阴性
- 需要根据业务场景选择：优先"不冤枉好人"还是"不漏掉风险"

改进建议：
1. 收集更多女性样本（减少数据偏见）
2. 检查代理变量（是否有特征间接编码性别）
3. 使用公平性约束的模型（如 Fairlearn）
4. 对不同群体使用不同阈值（但可能有法律风险）
    """)


if __name__ == "__main__":
    evaluate_fairness()

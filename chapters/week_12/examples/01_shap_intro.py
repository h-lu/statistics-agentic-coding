"""
示例：从特征重要性到 SHAP——理解全局与局部解释

本例演示：
1. 特征重要性的局限（只知道整体上哪些特征重要）
2. SHAP 全局解释（summary_plot）
3. SHAP 局部解释（force_plot，解释单个预测）

运行方式：python3 chapters/week_12/examples/01_shap_intro.py
预期输出：
- 控制台打印特征重要性与 SHAP 值对比
- 生成 SHAP summary plot 图（shap_summary.png）
- 生成 SHAP force plot 图（shap_force_sample.png）

依赖安装：
pip install shap pandas numpy scikit-learn matplotlib
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

OUTPUT_DIR = Path(__file__).parent.parent.parent / "output" / "week_12"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 设置随机种子保证可复现
np.random.seed(42)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def generate_credit_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    生成模拟信用评分数据

    特征：
    - income: 月收入
    - credit_history_age: 信用历史年限
    - debt_to_income: 债务收入比
    - credit_inquiries: 近6个月信用查询次数
    - employment_length: 工作年限

    目标：
    - default: 是否违约（1=违约，0=不违约）
    """
    np.random.seed(42)

    # 生成特征
    income = np.random.lognormal(10, 0.5, n_samples)  # 对数正态分布
    credit_history_age = np.random.uniform(0, 20, n_samples)
    debt_to_income = np.random.beta(2, 5, n_samples) * 0.8  # 大部分在 0-0.5
    credit_inquiries = np.random.poisson(2, n_samples)
    employment_length = np.random.exponential(5, n_samples)
    employment_length = np.clip(employment_length, 0, 30)

    # 生成目标（基于真实模式 + 噪声）
    # 违约概率与收入负相关，与债务收入比、信用查询次数正相关
    logit = (
        -4.0  # 基准
        + 0.5 * np.log(income / 10000)  # 收入越高，违约概率越低
        - 0.1 * credit_history_age  # 信用历史越长，违约概率越低
        + 3.0 * debt_to_income  # 债务收入比越高，违约概率越高
        + 0.3 * credit_inquiries  # 查询次数越多，违约概率越高
        - 0.05 * employment_length  # 工作年限越长，违约概率越低
    )

    # 添加噪声
    logit += np.random.normal(0, 0.5, n_samples)

    # Sigmoid 变换
    default_prob = 1 / (1 + np.exp(-logit))
    default = np.random.binomial(1, default_prob)

    df = pd.DataFrame({
        'income': income,
        'credit_history_age': credit_history_age,
        'debt_to_income': debt_to_income,
        'credit_inquiries': credit_inquiries,
        'employment_length': employment_length,
        'default': default
    })

    return df


def feature_importance_vs_shap():
    """
    对比特征重要性与 SHAP 值

    特征重要性：全局的，告诉你"整体上哪些特征被模型用得最多"
    SHAP 值：局部的，告诉你"对某个样本，每个特征贡献了多少"
    """
    print("=" * 70)
    print("示例 1：特征重要性与 SHAP 值对比")
    print("=" * 70)

    # 生成数据
    print("\n生成模拟信用评分数据...")
    df = generate_credit_data(n_samples=1000)
    print(f"数据形状: {df.shape}")
    print(f"违约率: {df['default'].mean():.2%}")

    # 准备特征和目标
    feature_cols = ['income', 'credit_history_age', 'debt_to_income',
                    'credit_inquiries', 'employment_length']
    X = df[feature_cols]
    y = df['default']

    # 划分数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 训练随机森林
    print("\n训练随机森林...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    # 计算准确率
    train_acc = rf.score(X_train, y_train)
    test_acc = rf.score(X_test, y_test)
    print(f"训练集准确率: {train_acc:.3f}")
    print(f"测试集准确率: {test_acc:.3f}")

    # ========== 特征重要性 ==========
    print("\n" + "-" * 70)
    print("1. 特征重要性（基于不纯度）")
    print("-" * 70)

    importances = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n特征重要性排序：")
    for _, row in importances.iterrows():
        print(f"  {row['feature']:20s}: {row['importance']:.4f}")

    print("\n⚠️  特征重要性的局限：")
    print("  - 只告诉你'整体上哪些特征重要'")
    print("  - 无法解释'对某个样本，每个特征贡献了多少'")
    print("  - 对相关特征有偏（相关特征会'稀释'重要性）")

    # ========== SHAP 全局解释 ==========
    print("\n" + "-" * 70)
    print("2. SHAP 全局解释（summary_plot）")
    print("-" * 70)

    try:
        import shap
    except ImportError:
        print("\n⚠️  需要安装 shap 库：")
        print("   pip install shap")
        return

    # 初始化 SHAP 解释器（对树模型使用 TreeExplainer，更快）
    explainer = shap.TreeExplainer(rf)

    # 计算 SHAP 值（测试集）
    print("计算 SHAP 值（这可能需要几秒钟）...")
    shap_values = explainer.shap_values(X_test)

    # 对于二分类，shap_values 可能是列表或数组
    if isinstance(shap_values, list):
        # [shap_values_class0, shap_values_class1]
        shap_values_positive = shap_values[1]
    else:
        # 如果是数组，直接使用
        shap_values_positive = shap_values

    print("\n生成 SHAP summary plot...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_positive, X_test, show=False)
    plt.title('SHAP 全局解释：每个特征的值如何影响预测')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'shap_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ 已保存: shap_summary.png")

    # 计算 SHAP 重要性（绝对值的均值）
    shap_importance = pd.DataFrame({
        'feature': feature_cols,
        'shap_importance': np.abs(shap_values_positive).mean(axis=0)
    }).sort_values('shap_importance', ascending=False)

    print("\nSHAP 重要性排序（绝对值的均值）：")
    for _, row in shap_importance.iterrows():
        print(f"  {row['feature']:20s}: {row['shap_importance']:.4f}")

    # ========== SHAP 局部解释 ==========
    print("\n" + "-" * 70)
    print("3. SHAP 局部解释（force_plot，单个样本）")
    print("-" * 70)

    # 选择一个被拒的样本（预测概率 > 0.7）
    y_pred_proba = rf.predict_proba(X_test)[:, 1]
    rejected_mask = y_pred_proba > 0.7

    if rejected_mask.sum() > 0:
        sample_idx = X_test[rejected_mask].index[0]
        sample = X_test.loc[sample_idx]
        sample_proba = y_pred_proba[X_test.index.get_loc(sample_idx)]

        print(f"\n选择样本 {sample_idx}:")
        print(f"  预测违约概率: {sample_proba:.2%}")
        print(f"\n样本特征值：")
        for feat in feature_cols:
            print(f"    {feat:20s}: {sample[feat]:.2f}")

        # 获取该样本的 SHAP 值
        sample_shap_idx = X_test.index.get_loc(sample_idx)

        if isinstance(shap_values, list):
            sample_shap = shap_values[1][sample_shap_idx]
            base_value = explainer.expected_value[1]
        else:
            sample_shap = shap_values[sample_shap_idx]
            base_value = explainer.expected_value

        print(f"\nSHAP 值分解：")
        print(f"  基准值（平均违约概率）: {base_value:.3f}")
        print(f"  特征贡献：")

        # 按绝对值排序
        contrib_df = pd.DataFrame({
            'feature': feature_cols,
            'shap_value': sample_shap,
            'feature_value': sample[feature_cols].values
        }).assign(abs_shap=lambda x: x['shap_value'].abs()).sort_values('abs_shap', ascending=False)

        for _, row in contrib_df.iterrows():
            direction = "↑" if row['shap_value'] > 0 else "↓"
            print(f"    {row['feature']:20s}: {row['shap_value']:+.4f}  {direction}  (值={row['feature_value']:.2f})")

        print(f"\n  预测 = 基准值 + 贡献之和 = {base_value:.3f} + {sample_shap.sum():.3f} = {base_value + sample_shap.sum():.3f}")

        # 生成 force plot
        print("\n生成 SHAP force plot...")
        plt.figure(figsize=(14, 4))
        shap.force_plot(
            base_value,
            sample_shap,
            sample,
            matplotlib=True,
            show=False,
            link='logit'
        )
        plt.title(f'SHAP 局部解释：为什么这个样本被预测为高风险？')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'shap_force_sample.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✅ 已保存: shap_force_sample.png")

        # ========== 向客户解释 ==========
        print("\n" + "-" * 70)
        print("4. 如何向客户解释这个预测？")
        print("-" * 70)

        print("\n❌ 坏的解释（技术术语堆砌）：")
        print("  '您的 SHAP 值显示债务收入比贡献了 +0.35，")
        print("   信用查询次数贡献了 +0.18，收入贡献了 -0.12。'")

        print("\n✅ 好的解释（客户能懂）：")
        # 找出正向贡献最大的2个特征
        top_positive = contrib_df[contrib_df['shap_value'] > 0].head(2)
        top_negative = contrib_df[contrib_df['shap_value'] < 0].head(1)

        reasons = []
        for _, row in top_positive.iterrows():
            if row['feature'] == 'debt_to_income':
                reasons.append(f"您的债务收入比（{row['feature_value']:.1%}）较高")
            elif row['feature'] == 'credit_inquiries':
                reasons.append(f"您近6个月有{int(row['feature_value'])}次信用查询")
            elif row['feature'] == 'income':
                reasons.append(f"您的月收入（{row['feature_value']:.0f}元）较低")
            else:
                reasons.append(f"您的{row['feature']}（{row['feature_value']:.2f}）")

        explanation = "您的申请被拒主要因为："
        if len(reasons) == 1:
            explanation += reasons[0] + "。"
        elif len(reasons) == 2:
            explanation += reasons[0] + "；" + reasons[1] + "。"
        else:
            explanation += "、".join(reasons[:-1]) + "以及" + reasons[-1] + "。"

        print(f"  '{explanation}'")
        print(f"\n  这些因素把您的通过概率从平均的{(1-base_value):.1%}降低到了{(1-sample_proba):.1%}。")

        if not top_negative.empty:
            row = top_negative.iloc[0]
            if row['feature'] == 'credit_history_age':
                print(f"  好消息是，您的信用历史较长（{row['feature_value']:.1f}年），这有助于降低风险。")

    else:
        print("\n没有找到高概率违约的样本")

    # ========== 总结 ==========
    print("\n" + "=" * 70)
    print("总结：特征重要性 vs SHAP")
    print("=" * 70)
    print("""
特征重要性（全局）：
  ✓ 回答"整体上哪些特征最重要"
  ✗ 无法解释单个预测
  ✗ 受相关特征影响

SHAP 值（全局 + 局部）：
  ✓ 全局：summary_plot 看哪些特征重要
  ✓ 局部：force_plot 解释单个预测
  ✓ 可加性：所有贡献加起来等于预测值
  ✓ 基于博弈论，有坚实的数学基础

向客户解释时：
  ❌ 不要说"SHAP 值"或"边际贡献"
  ✅ 要说具体原因（债务收入比高、信用查询多等）
  ✅ 要说明这些因素如何影响通过概率
    """)


if __name__ == "__main__":
    feature_importance_vs_shap()

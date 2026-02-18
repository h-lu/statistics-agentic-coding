"""
示例：SHAP 值——从全局可解释性到局部可解释性

本例演示：
1. 如何使用 SHAP 值解释单个预测
2. SHAP 汇总图：全局 + 局部信息
3. 为什么同一个特征对不同样本的贡献可以不同

运行方式：python3 chapters/week_12/examples/12_02_shap_values.py
预期输出：stdout 输出 SHAP 值解释 + 保存图表到 output/

注意：需要安装 shap 库（pip install shap）
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

# 尝试导入 shap，如果没有安装则提供提示
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("警告: shap 库未安装，请运行: pip install shap")


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


def generate_synthetic_data(n_samples: int = 1000, random_state: int = 42) -> tuple:
    """
    生成合成流失预测数据（包含非线性关系）
    """
    rng = np.random.default_rng(random_state)

    # 生成特征
    days = rng.poisson(30, n_samples)
    count = rng.poisson(5, n_samples)
    spend = rng.exponential(100, n_samples)
    vip = rng.binomial(1, 0.3, n_samples)
    age = rng.integers(18, 70, n_samples)

    # 生成目标（包含非线性关系）
    # 购买天数和流失概率的关系：30天后风险急剧上升
    days_effect = np.where(days > 30, (days - 30) * 0.1, 0)

    # 购买次数和流失概率的关系：10次以上风险很低
    count_effect = np.where(count > 10, -1.5, -0.1 * count)

    # VIP 的保护效应
    vip_effect = -1.5 * vip

    logit = -2 + days_effect + count_effect + vip_effect + 0.01 * (age - 40)
    prob = 1 / (1 + np.exp(-logit))
    churn = rng.binomial(1, prob)

    X = pd.DataFrame({
        'days_since_last_purchase': days,
        'purchase_count': count,
        'avg_spend': spend,
        'vip_status': vip,
        'age': age
    })
    y = pd.Series(churn, name='churn')

    return X, y


def train_random_forest(X: pd.DataFrame, y: pd.Series) -> tuple:
    """训练随机森林模型"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    return rf, X_train, X_test, y_train, y_test


def explain_single_prediction(
    explainer: shap.TreeExplainer,
    shap_values: np.ndarray,
    X_test: pd.DataFrame,
    sample_idx: int,
    expected_value: float
) -> dict:
    """
    解释单个预测，返回业务语言描述
    """
    sample = X_test.iloc[sample_idx]
    shap_vals = shap_values[sample_idx]

    # 基线 + SHAP 值 = 预测值
    prediction = expected_value + shap_vals.sum()

    # 找出贡献最大的 3 个特征
    top_idx = np.argsort(np.abs(shap_vals))[-3:][::-1]

    explanations = []
    for idx in top_idx:
        feature = X_test.columns[idx]
        contribution = shap_vals[idx]
        # 确保 contribution 是标量
        if isinstance(contribution, np.ndarray):
            contribution = float(contribution.flat[0])
        value = sample[feature]
        # 确保 value 是标量
        if isinstance(value, np.ndarray):
            value = float(value.flat[0])

        direction = "增加" if contribution > 0 else "降低"
        explanations.append({
            'feature': feature,
            'value': value,
            'contribution': contribution,
            'direction': direction
        })

    return {
        'sample_idx': sample_idx,
        'prediction': prediction,
        'expected_value': expected_value,
        'shap_sum': shap_vals.sum(),
        'top_factors': explanations
    }


def format_shap_explanation(explanation: dict) -> str:
    """将 SHAP 解释格式化为业务语言"""
    lines = []
    lines.append("=" * 60)
    lines.append(f"样本 #{explanation['sample_idx']} 的预测解释")
    lines.append("=" * 60)

    prob = 1 / (1 + np.exp(-explanation['prediction']))
    baseline_prob = 1 / (1 + np.exp(-explanation['expected_value']))

    lines.append(f"\n基线流失概率: {baseline_prob:.1%}")
    lines.append(f"样本预测流失概率: {prob:.1%}")
    lines.append(f"SHAP 调整: {explanation['shap_sum']:.3f} (log-odds)")

    lines.append("\n主要影响因素:")
    for i, factor in enumerate(explanation['top_factors'], 1):
        lines.append(f"\n{i}. {factor['feature']} = {factor['value']:.2f}")
        lines.append(f"   -> {factor['direction']}流失风险 (SHAP 值: {factor['contribution']:.3f})")

    return "\n".join(lines)


def demonstrate_shap_value_variation(
    explainer: shap.TreeExplainer,
    shap_values: np.ndarray,
    X_test: pd.DataFrame,
    output_dir: Path
) -> None:
    """
    演示同一个特征对不同样本的贡献可以不同
    """
    font = setup_chinese_font()

    # 选择 days_since_last_purchase 这个特征
    feature_idx = X_test.columns.get_loc('days_since_last_purchase')
    feature_name = 'days_since_last_purchase'
    feature_shap = shap_values[:, feature_idx]

    # 绘制 SHAP 值 vs 特征值的关系图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. SHAP 值 vs 特征值散点图
    axes[0].scatter(X_test[feature_name], feature_shap, alpha=0.5, s=20)
    axes[0].axhline(y=0, color='red', linestyle='--', linewidth=1)
    axes[0].set_xlabel(f'{feature_name}')
    axes[0].set_ylabel(f'SHAP 值（对流失风险的贡献）')
    axes[0].set_title(f'SHAP 值 vs {feature_name}')
    axes[0].grid(True, alpha=0.3)

    # 添加说明
    axes[0].text(0.05, 0.95,
                '正 SHAP 值：增加流失风险\n负 SHAP 值：降低流失风险',
                transform=axes[0].transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 2. 按 VIP 状态分组的 SHAP 值分布
    vip_mask = X_test['vip_status'] == 1
    non_vip_mask = X_test['vip_status'] == 0

    axes[1].hist(feature_shap[non_vip_mask], bins=30, alpha=0.5, label='非会员', color='blue')
    axes[1].hist(feature_shap[vip_mask], bins=30, alpha=0.5, label='会员', color='green')
    axes[1].axvline(x=0, color='red', linestyle='--', linewidth=1)
    axes[1].set_xlabel(f'SHAP 值（{feature_name} 的贡献）')
    axes[1].set_ylabel('频数')
    axes[1].set_title(f'{feature_name} 的 SHAP 值分布（按 VIP 状态分组）')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图片
    plt.savefig(output_dir / 'shap_value_variation.png',
                dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

    print(f"SHAP 值变化图已保存到: {output_dir / 'shap_value_variation.png'}")


def plot_shap_summary(shap_values: np.ndarray, X_test: pd.DataFrame, output_dir: Path) -> None:
    """绘制 SHAP 汇总图"""
    font = setup_chinese_font()

    fig, ax = plt.subplots(figsize=(10, 6))

    # 手动绘制简化的 SHAP 汇总图
    feature_importance = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(feature_importance)

    for i, idx in enumerate(sorted_idx):
        feature = X_test.columns[idx]
        values = shap_values[:, idx]

        # 使用特征值决定颜色（红色=高值，蓝色=低值）
        feature_values = X_test.iloc[:, idx]
        normalized = (feature_values - feature_values.min()) / (feature_values.max() - feature_values.min() + 1e-10)

        # 绘制散点
        colors = plt.cm.RdBu_r(normalized)
        ax.scatter(values, [i] * len(values), c=colors, alpha=0.5, s=10)

        # 添加特征名称
        ax.text(-0.5, i, feature, va='center', ha='right', fontsize=10)

    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('SHAP 值（对流失风险的贡献）')
    ax.set_yticks([])
    ax.set_title('SHAP 汇总图：每个特征对不同样本的贡献分布')

    # 添加图例
    ax.text(0.02, 0.98,
            '红色 = 高特征值\n蓝色 = 低特征值\n右 = 增加流失风险\n左 = 降低流失风险',
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
            fontsize=9)

    plt.tight_layout()

    # 保存图片
    plt.savefig(output_dir / 'shap_summary.png',
                dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

    print(f"SHAP 汇总图已保存到: {output_dir / 'shap_summary.png'}")


def main() -> None:
    """主函数"""
    if not SHAP_AVAILABLE:
        print("\n请先安装 shap 库:")
        print("  pip install shap")
        return

    print("=" * 60)
    print("SHAP 值：局部可解释性演示")
    print("=" * 60)

    # 创建输出目录
    output_dir = Path(__file__).parent.parent.parent.parent / 'output'
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 生成数据并训练模型
    print("\n1. 生成合成数据并训练随机森林...")
    X, y = generate_synthetic_data()
    rf, X_train, X_test, y_train, y_test = train_random_forest(X, y)

    print(f"   训练集: {X_train.shape[0]} 行")
    print(f"   测试集: {X_test.shape[0]} 行")

    # 2. 计算 SHAP 值
    print("\n2. 计算 SHAP 值...")
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_test)

    # shap_values 可能是嵌套列表（二分类），取正类的 SHAP 值
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    expected_value = explainer.expected_value
    # expected_value 可能是 array（二分类），取正类的值
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]
    expected_value = float(expected_value)  # 确保是标量

    print(f"   基线 log-odds: {expected_value:.3f}")
    print(f"   基线流失概率: {1 / (1 + np.exp(-expected_value)):.1%}")

    # 3. 解释单个预测
    print("\n3. 解释单个预测...")

    # 选择一个被预测为流失的样本（使用稳健方法）
    y_prob = rf.predict_proba(X_test)[:, 1]

    # 找概率最高的样本作为"高流失风险"示例
    churn_sample_idx = np.argmax(y_prob)
    churn_prob = y_prob[churn_sample_idx]

    explanation = explain_single_prediction(
        explainer, shap_values, X_test, churn_sample_idx, expected_value
    )
    print(f"   选择流失概率最高的样本 (idx={churn_sample_idx}, prob={churn_prob:.1%})")
    print(format_shap_explanation(explanation))

    # 找概率最低的样本作为"低流失风险"示例
    safe_sample_idx = np.argmin(y_prob)
    safe_prob = y_prob[safe_sample_idx]

    explanation2 = explain_single_prediction(
        explainer, shap_values, X_test, safe_sample_idx, expected_value
    )
    print(f"\n   选择流失概率最低的样本 (idx={safe_sample_idx}, prob={safe_prob:.1%})")
    print(format_shap_explanation(explanation2))

    # 4. 演示 SHAP 值的变化
    print("\n4. 演示同一个特征对不同样本的贡献可以不同...")
    demonstrate_shap_value_variation(explainer, shap_values, X_test, output_dir)

    # 5. 绘制 SHAP 汇总图
    print("\n5. 绘制 SHAP 汇总图...")
    plot_shap_summary(shap_values, X_test, output_dir)

    # 6. 总结
    print("\n" + "=" * 60)
    print("总结：SHAP 值的核心概念")
    print("=" * 60)
    print("""
1. 局部可解释性
   - 特征重要性：全局的、平均的（"模型整体上看什么"）
   - SHAP 值：局部的、单样本的（"为什么这个样本被这样预测"）

2. SHAP 值的含义
   - 基线（expected_value）：所有样本的平均预测
   - SHAP 值：每个特征对"偏离基线"的贡献
   - 正 SHAP 值：增加流失风险
   - 负 SHAP 值：降低流失风险

3. 非线性关系的体现
   - 同一个特征对不同样本的贡献可以不同
   - 例如：days_since_last_purchase 对某些样本贡献很大，对某些很小
   - 因为特征之间存在交互作用（如 VIP 状态会调节购买天数的影响）

4. 业务语言翻译
   - 不要说 "SHAP 值是 0.31"
   - 要说 "最近 45 天未购买，这是流失风险的主要来源（贡献 +31% 概率）"
    """)

    print("\n生成的文件:")
    print(f"  1. {output_dir / 'shap_value_variation.png'}")
    print(f"  2. {output_dir / 'shap_summary.png'}")


if __name__ == "__main__":
    main()

"""
示例：特征重要性的陷阱——从回归系数到随机森林特征重要性

本例演示：
1. 逻辑回归系数：有方向的特征重要性
2. 随机森林特征重要性：只有强度，没有方向
3. 相关特征的"分票"陷阱

运行方式：python3 chapters/week_12/examples/12_01_feature_importance.py
预期输出：stdout 输出特征重要性对比 + 保存图表到 output/
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


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
    生成合成流失预测数据

    特征：
    - days_since_last_purchase: 最近购买天数
    - purchase_count: 购买次数
    - avg_spend: 平均消费金额
    - vip_status: 是否会员（0/1）
    - age: 年龄
    - gender: 性别（0=女, 1=男）

    目标：
    - churn: 是否流失（0/1）
    """
    rng = np.random.default_rng(random_state)

    # 生成特征
    days = rng.poisson(30, n_samples)  # 平均30天
    count = rng.poisson(5, n_samples)  # 平均5次
    spend = rng.exponential(100, n_samples)  # 平均100元
    vip = rng.binomial(1, 0.3, n_samples)  # 30%会员
    age = rng.integers(18, 70, n_samples)
    gender = rng.binomial(1, 0.5, n_samples)

    # 生成目标（与特征相关）
    # 购买天数越多、购买次数越少、非会员 -> 更容易流失
    logit = (
        -2.0
        + 0.05 * days  # 购买天数增加，流失概率增加
        - 0.2 * count  # 购买次数增加，流失概率减少
        - 0.01 * spend  # 消费增加，流失概率略减
        - 1.0 * vip  # 会员流失概率更低
        + 0.01 * (age - 40)  # 年龄对流失影响较小
        + 0.2 * gender  # 性别对流失影响较小
    )
    prob = 1 / (1 + np.exp(-logit))
    churn = rng.binomial(1, prob)

    X = pd.DataFrame({
        'days_since_last_purchase': days,
        'purchase_count': count,
        'avg_spend': spend,
        'vip_status': vip,
        'age': age,
        'gender': gender
    })
    y = pd.Series(churn, name='churn')

    return X, y


def compare_feature_importance(X: pd.DataFrame, y: pd.Series) -> dict:
    """
    对比逻辑回归系数和随机森林特征重要性
    """
    # 划分数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    results = {}

    # 1. 逻辑回归（需要标准化）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(X_train_scaled, y_train)

    # 逻辑回归系数表
    coef_df = pd.DataFrame({
        'feature': X_train.columns,
        'coefficient': log_reg.coef_[0],
        'abs_coef': np.abs(log_reg.coef_[0])
    }).sort_values('abs_coef', ascending=False)

    results['logistic_regression'] = {
        'model': log_reg,
        'coefficients': coef_df,
        'feature_names': X_train.columns.tolist()
    }

    # 2. 随机森林
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    # 随机森林特征重要性
    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    results['random_forest'] = {
        'model': rf,
        'importance': importance_df,
        'feature_names': X_train.columns.tolist()
    }

    return results


def demonstrate_correlation_dilution() -> pd.DataFrame:
    """
    演示相关特征的"分票"陷阱
    """
    rng = np.random.default_rng(42)

    n = 1000

    # 生成相关特征
    # purchase_count 和 total_spend 高度相关
    purchase_count = rng.poisson(5, n)
    total_spend = purchase_count * 100 + rng.normal(0, 20, n)  # 高度相关
    avg_spend = total_spend / (purchase_count + 1)

    # 其他特征
    days = rng.poisson(30, n)

    # 目标：主要受 purchase_count 影响
    logit = -2 + 0.3 * purchase_count + 0.05 * days
    prob = 1 / (1 + np.exp(-logit))
    y = rng.binomial(1, prob)

    # 场景1：原始数据
    X1 = pd.DataFrame({
        'purchase_count': purchase_count,
        'total_spend': total_spend,
        'avg_spend': avg_spend,
        'days_since_last_purchase': days
    })

    rf1 = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
    rf1.fit(X1, y)

    importance1 = pd.DataFrame({
        'feature': X1.columns,
        'importance': rf1.feature_importances_
    }).sort_values('importance', ascending=False)

    # 场景2：添加 purchase_count 的副本（模拟相关特征）
    X2 = X1.copy()
    for i in range(5):
        X2[f'purchase_count_copy_{i}'] = purchase_count + rng.normal(0, 0.1, n)

    rf2 = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
    rf2.fit(X2, y)

    importance2 = pd.DataFrame({
        'feature': X2.columns,
        'importance': rf2.feature_importances_
    }).sort_values('importance', ascending=False)

    return {
        'original': importance1,
        'diluted': importance2
    }


def plot_feature_importance_comparison(
    log_reg_coef: pd.DataFrame,
    rf_importance: pd.DataFrame,
    output_dir: Path
) -> None:
    """绘制特征重要性对比图"""
    font = setup_chinese_font()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. 逻辑回归系数（有正负方向）
    coef_sorted = log_reg_coef.sort_values('coefficient')
    colors = ['red' if c < 0 else 'green' for c in coef_sorted['coefficient']]

    axes[0].barh(coef_sorted['feature'], coef_sorted['coefficient'], color=colors, alpha=0.7)
    axes[0].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    axes[0].set_xlabel('系数值')
    axes[0].set_title('逻辑回归系数\n（绿色=增加流失概率，红色=降低流失概率）')
    axes[0].invert_yaxis()

    # 添加数值标签
    for i, (idx, row) in enumerate(coef_sorted.iterrows()):
        axes[0].text(row['coefficient'], i,
                    f"{row['coefficient']:.3f}",
                    va='center', ha='left' if row['coefficient'] > 0 else 'right',
                    fontsize=9)

    # 2. 随机森林特征重要性（只有强度）
    rf_sorted = rf_importance.sort_values('importance')

    axes[1].barh(rf_sorted['feature'], rf_sorted['importance'],
                color='steelblue', alpha=0.7)
    axes[1].set_xlabel('特征重要性')
    axes[1].set_title('随机森林特征重要性\n（只有强度，没有方向）')
    axes[1].invert_yaxis()

    # 添加数值标签
    for i, (idx, row) in enumerate(rf_sorted.iterrows()):
        axes[1].text(row['importance'], i,
                    f"{row['importance']:.3f}",
                    va='center', ha='left', fontsize=9)

    plt.tight_layout()

    # 保存图片
    plt.savefig(output_dir / 'feature_importance_comparison.png',
                dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

    print(f"特征重要性对比图已保存到: {output_dir / 'feature_importance_comparison.png'}")


def plot_correlation_dilution(dilution_results: dict, output_dir: Path) -> None:
    """绘制相关特征分票陷阱图"""
    font = setup_chinese_font()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. 原始数据
    importance1 = dilution_results['original']
    axes[0].barh(range(len(importance1)), importance1['importance'],
                color='steelblue', alpha=0.7)
    axes[0].set_yticks(range(len(importance1)))
    axes[0].set_yticklabels(importance1['feature'])
    axes[0].set_xlabel('特征重要性')
    axes[0].set_title('原始数据：purchase_count 重要性最高')
    axes[0].invert_yaxis()

    # 2. 添加副本后的数据
    importance2 = dilution_results['diluted'].head(8)
    axes[1].barh(range(len(importance2)), importance2['importance'],
                color='coral', alpha=0.7)
    axes[1].set_yticks(range(len(importance2)))
    axes[1].set_yticklabels(importance2['feature'])
    axes[1].set_xlabel('特征重要性')
    axes[1].set_title('添加相关副本：重要性被"稀释"')
    axes[1].invert_yaxis()

    plt.tight_layout()

    # 保存图片
    plt.savefig(output_dir / 'correlation_dilution.png',
                dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

    print(f"相关特征分票陷阱图已保存到: {output_dir / 'correlation_dilution.png'}")


def main() -> None:
    """主函数"""
    print("=" * 60)
    print("特征重要性的陷阱演示")
    print("=" * 60)

    # 创建输出目录
    output_dir = Path(__file__).parent.parent.parent.parent / 'output'
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 生成合成数据
    print("\n1. 生成合成数据...")
    X, y = generate_synthetic_data()
    print(f"   数据规模: {X.shape[0]} 行, {X.shape[1]} 列")
    print(f"   流失率: {y.mean():.2%}")

    # 2. 对比特征重要性
    print("\n2. 对比逻辑回归和随机森林的特征重要性...")
    results = compare_feature_importance(X, y)

    # 输出逻辑回归系数
    print("\n" + "=" * 60)
    print("逻辑回归系数（有方向）")
    print("=" * 60)
    coef_df = results['logistic_regression']['coefficients']
    print(coef_df.to_string(index=False))
    print("\n解读:")
    print("  - 正系数：特征值增加 -> 流失概率增加")
    print("  - 负系数：特征值增加 -> 流失概率降低")
    print("  - 绝对值：系数绝对值越大 -> 影响越强")

    # 输出随机森林特征重要性
    print("\n" + "=" * 60)
    print("随机森林特征重要性（只有强度）")
    print("=" * 60)
    importance_df = results['random_forest']['importance']
    print(importance_df.to_string(index=False))
    print("\n解读:")
    print("  - 重要性值越大 -> 特征对预测的贡献越大")
    print("  - 无法看出方向：不知道特征值增加会增加还是降低流失概率")

    # 3. 演示相关特征分票陷阱
    print("\n3. 演示相关特征的'分票'陷阱...")
    dilution_results = demonstrate_correlation_dilution()

    print("\n原始数据中的特征重要性:")
    print(dilution_results['original'].to_string(index=False))

    print("\n添加相关副本后的特征重要性（Top 8）:")
    print(dilution_results['diluted'].head(8).to_string(index=False))

    # 4. 绘制图表
    print("\n4. 生成图表...")
    plot_feature_importance_comparison(
        results['logistic_regression']['coefficients'],
        results['random_forest']['importance'],
        output_dir
    )
    plot_correlation_dilution(dilution_results, output_dir)

    # 5. 总结
    print("\n" + "=" * 60)
    print("总结：特征重要性的三个陷阱")
    print("=" * 60)
    print("""
1. 陷阱一：有强度，没方向
   - 随机森林特征重要性只告诉你"什么重要"，不告诉你"往哪个方向推"
   - 逻辑回归系数有正负，可以解读方向

2. 陷阱二：相关特征会"分票"
   - 如果两个特征高度相关，它们会互相"稀释"重要性
   - 单个副本看起来不重要，但加起来其实很强

3. 陷阱三：是全局的，不是局部的
   - 特征重要性是平均重要性，无法解释单个预测
   - 同一个特征对不同样本的贡献可能不同

建议：
- 快速了解"模型看了什么" -> 用特征重要性
- 解释"为什么这个样本被预测为流失" -> 用 SHAP 值（下一节）
- 想知道"方向和强度" -> 用逻辑回归系数
    """)

    print("\n生成的文件:")
    print(f"  1. {output_dir / 'feature_importance_comparison.png'}")
    print(f"  2. {output_dir / 'correlation_dilution.png'}")


if __name__ == "__main__":
    main()

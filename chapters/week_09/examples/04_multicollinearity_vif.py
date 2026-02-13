"""
示例：多重共线性检测——VIF 计算

本例演示多重共线性问题以及如何用方差膨胀因子(VIF)检测。
展示共线性导致的系数不稳定现象。

运行方式：python3 chapters/week_09/examples/04_multicollinearity_vif.py
预期输出：
- 好模型（低 VIF）和坏模型（高 VIF）的对比
- VIF 计算表
- 相关矩阵热力图（保存为 correlation_heatmap.png）
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

OUTPUT_DIR = Path(__file__).parent.parent.parent / "output" / "week_09"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)


def generate_data_with_multicollinearity(n_samples: int = 100) -> pd.DataFrame:
    """
    生成具有多重共线性的数据

    参数:
        n_samples: 样本数量

    返回:
        包含相关特征的 DataFrame
    """
    # 基础面积
    area_base = np.random.uniform(50, 150, n_samples)

    # 卧室数：与面积高度相关
    n_bedrooms = (area_base / 20).astype(int) + np.random.randint(-1, 2, n_samples)
    n_bedrooms = np.maximum(n_bedrooms, 1)

    # 客厅数：与卧室数高度相关
    n_living_rooms = (n_bedrooms / 2).astype(int) + np.random.randint(0, 2, n_samples)
    n_living_rooms = np.maximum(n_living_rooms, 1)

    # 卫生间数：与卧室数高度相关
    n_bathrooms = n_bedrooms + np.random.randint(-1, 2, n_samples)
    n_bathrooms = np.maximum(n_bathrooms, 1)

    # 总房间数（卧室+客厅+卫生间，冗余变量！）
    total_rooms = n_bedrooms + n_living_rooms + n_bathrooms

    # 房龄：独立变量
    age_years = np.random.randint(0, 31, n_samples)

    # 房价：真实关系
    noise = np.random.normal(0, 10, n_samples)
    price_wan = (20 + 0.8 * area_base - 0.3 * age_years +
                 3 * n_bedrooms + noise)

    return pd.DataFrame({
        'area_sqm': area_base,
        'n_bedrooms': n_bedrooms,
        'n_living_rooms': n_living_rooms,
        'n_bathrooms': n_bathrooms,
        'total_rooms': total_rooms,
        'age_years': age_years,
        'price_wan': price_wan
    })


def calculate_vif(X: pd.DataFrame) -> pd.DataFrame:
    """
    计算方差膨胀因子(VIF)

    VIF = 1 / (1 - R²_i)
    其中 R²_i 是第 i 个变量对其他变量回归的 R²

    参数:
        X: 自变量 DataFrame

    返回:
        包含变量名和 VIF 值的 DataFrame
    """
    vif_data = pd.DataFrame()
    vif_data["变量"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                        for i in range(X.shape[1])]

    return vif_data


def plot_correlation_heatmap(df: pd.DataFrame, features: list) -> None:
    """
    画相关矩阵热力图

    参数:
        df: 数据
        features: 要画相关性的特征列表
    """
    corr = df[features].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('特征相关矩阵（高相关预示多重共线性）', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'correlation_heatmap.png', dpi=150, bbox_inches='tight')
    print("✅ 相关矩阵热力图已保存为 correlation_heatmap.png")
    plt.close()


def fit_and_print_model(y: pd.Series, X: pd.DataFrame,
                        model_name: str) -> sm.regression.linear_model.RegressionResults:
    """
    拟合模型并打印结果

    参数:
        y: 因变量
        X: 自变量
        model_name: 模型名称

    返回:
        拟合的模型
    """
    X_sm = sm.add_constant(X)
    model = sm.OLS(y, X_sm).fit()

    print(f"\n{'=' * 70}")
    print(f"{model_name}")
    print('=' * 70)

    # 打印系数
    coef_df = pd.DataFrame({
        '系数': model.params,
        '标准误': model.bse,
        't值': model.tvalues,
        'p值': model.pvalues,
    })
    print(coef_df.round(3))

    print(f"\nR² = {model.rsquared:.3f}")
    print(f"调整 R² = {model.rsquared_adj:.3f}")

    return model


def main() -> None:
    """主函数：演示多重共线性问题"""
    print("=" * 70)
    print("示例4: 多重共线性检测——VIF 计算")
    print("=" * 70)

    # 1. 生成数据
    df = generate_data_with_multicollinearity(n_samples=100)
    print(f"\n📊 数据概览 (前5行):")
    print(df.head())

    # 2. 查看相关性
    room_features = ['n_bedrooms', 'n_living_rooms', 'n_bathrooms', 'total_rooms']
    print(f"\n🔗 房间类特征的相关性:")
    print(df[room_features].corr().round(3))

    # 3. 画相关热力图
    plot_correlation_heatmap(df, room_features)

    # ========================================
    # 场景1: 坏模型——包含冗余变量
    # ========================================
    print(f"\n{'='*70}")
    print("场景1: 坏模型——包含 total_rooms（冗余变量）")
    print('='*70)

    X_bad = df[['area_sqm', 'age_years', 'n_bedrooms',
                'n_living_rooms', 'n_bathrooms', 'total_rooms']]
    vif_bad = calculate_vif(X_bad)

    print("\n⚠️  VIF 表（坏模型）:")
    print(vif_bad)

    high_vif = vif_bad[vif_bad['VIF'] >= 10]
    if len(high_vif) > 0:
        print(f"\n🚨 严重共线性变量 (VIF ≥ 10):")
        print(high_vif)
        print(f"\n问题: 这些变量的方差被膨胀了，系数估计极不稳定!")

    # 拟合坏模型
    model_bad = fit_and_print_model(
        df['price_wan'],
        X_bad,
        "坏模型: price ~ area + age + bedrooms + living + bath + total"
    )

    # ========================================
    # 场景2: 好模型——删除冗余变量
    # ========================================
    print(f"\n{'='*70}")
    print("场景2: 好模型——删除 total_rooms 和高相关变量")
    print('='*70)

    X_good = df[['area_sqm', 'age_years', 'n_bedrooms']]
    vif_good = calculate_vif(X_good)

    print("\n✅ VIF 表（好模型）:")
    print(vif_good)

    all_low_vif = (vif_good['VIF'] < 5).all()
    if all_low_vif:
        print(f"\n✓ 所有 VIF < 5，共线性问题不严重")

    # 拟合好模型
    model_good = fit_and_print_model(
        df['price_wan'],
        X_good,
        "好模型: price ~ area + age + bedrooms"
    )

    # ========================================
    # 对比结果
    # ========================================
    print(f"\n{'='*70}")
    print("对比: 坏模型 vs 好模型")
    print('='*70)

    comparison = pd.DataFrame({
        '坏模型系数': model_bad.params,
        '好模型系数': model_good.params.reindex(model_bad.params.index, fill_value='N/A')
    })
    print(comparison)

    print(f"\n💡 关键观察:")
    print(f"  1. 坏模型中 n_bathrooms 的系数可能变成负数（不合理）")
    print(f"  2. 坏模型的标准误更大（系数不确定性强）")
    print(f"  3. 好模型的 R² 略低，但系数更稳定、可解释")

    # ========================================
    # 处理策略总结
    # ========================================
    print(f"\n{'='*70}")
    print("多重共线性处理策略")
    print('='*70)
    print("""
    1. 删除冗余变量:
       - 删除 VIF 最大的变量
       - 合并相关变量（如卧室+客厅+卫生间 → 总房间数）

    2. 使用正则化 (Week 12 会深入):
       - Ridge 回归: 系数收缩但不归零
       - LASSO 回归: 系数可能归零（自动选择变量）

    3. 主成分分析 (PCA):
       - 将相关变量转换为主成分
       - 牺牲可解释性换取稳定性

    4. 领域知识驱动:
       - 结合业务选择最重要的变量
       - 避免"为了用而用"
    """)

    print("\n" + "=" * 70)
    print("✅ 示例4完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()

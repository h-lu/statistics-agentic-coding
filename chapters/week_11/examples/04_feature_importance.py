"""
示例：特征重要性分析——内置重要性 vs 置换重要性

本例演示：
1. 基于不纯度的特征重要性（内置）
2. 置换重要性（permutation importance）
3. 相关特征对重要性的影响
4. 特征重要性可视化与正确解释

运行方式：python3 chapters/week_11/examples/04_feature_importance.py
预期输出：
- 两种特征重要性对比图（保存为 feature_importance_types.png）
- 相关特征影响演示图（保存为 correlation_effect.png）
- 控制台输出特征重要性排名与解释
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

OUTPUT_DIR = Path(__file__).parent.parent.parent / "output" / "week_11"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 设置随机种子保证可复现
np.random.seed(42)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def generate_data_with_correlation(n_samples: int = 500) -> pd.DataFrame:
    """
    生成包含相关特征的数据

    设计：
    - area_sqm: 面积（主要特征）
    - area_sqft: 面积（平方英尺，与 area_sqm 高度相关）
    - bedrooms: 卧室数（与面积相关）
    - bathrooms: 浴室数（与卧室数相关）
    - age_years: 房龄（独立特征）
    - noise_feature: 噪声特征（无信息）

    目标：房价（主要由面积决定）
    """
    np.random.seed(42)

    # 核心特征
    area_sqm = np.random.uniform(30, 200, n_samples)
    area_sqft = area_sqm * 10.764  # 高度相关（转换关系）
    bedrooms = np.random.randint(1, 6, n_samples)
    bathrooms = np.round(bedrooms * 0.7 + np.random.uniform(-0.5, 0.5, n_samples))
    bathrooms = np.clip(bathrooms, 1, 4).astype(int)

    # 独立特征
    age_years = np.random.uniform(0, 30, n_samples)
    noise_feature = np.random.uniform(0, 100, n_samples)  # 噪声

    # 目标变量：主要由面积决定
    base_price = 50 + 5 * area_sqm - 0.02 * area_sqm**2
    base_price += 10 * bedrooms
    base_price -= 2 * age_years
    # noise_feature 不参与目标生成

    price_noise = np.random.normal(0, 15, n_samples)
    price = base_price + price_noise

    return pd.DataFrame({
        'area_sqm': area_sqm,
        'area_sqft': area_sqft,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'age_years': age_years,
        'noise_feature': noise_feature,
        'price': price
    })


def compute_builtin_importance(rf: RandomForestRegressor, feature_names: list) -> pd.DataFrame:
    """计算基于不纯度的特征重要性（内置）"""
    importances = rf.feature_importances_
    return pd.DataFrame({
        'feature': feature_names,
        'builtin_importance': importances
    }).sort_values('builtin_importance', ascending=False)


def compute_permutation_importance(rf: RandomForestRegressor, X_test, y_test, feature_names: list, n_repeats: int = 30) -> pd.DataFrame:
    """计算置换重要性"""
    perm_result = permutation_importance(
        rf, X_test, y_test,
        n_repeats=n_repeats,
        random_state=42,
        n_jobs=-1
    )

    return pd.DataFrame({
        'feature': feature_names,
        'permutation_importance': perm_result.importances_mean,
        'permutation_std': perm_result.importances_std
    }).sort_values('permutation_importance', ascending=False)


def compare_importance_methods(df: pd.DataFrame) -> None:
    """对比两种特征重要性方法"""
    X = df.drop('price', axis=1)
    y = df['price']
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 训练随机森林
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=7,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    # 计算两种重要性
    builtin_importance = compute_builtin_importance(rf, feature_names)
    perm_importance = compute_permutation_importance(rf, X_test, y_test, feature_names)

    # 打印结果
    print("\n" + "=" * 60)
    print("特征重要性对比：内置 vs 置换")
    print("=" * 60)

    comparison = pd.merge(
        builtin_importance,
        perm_importance,
        on='feature',
        how='outer'
    ).sort_values('permutation_importance', ascending=False)

    print(f"\n{'特征':<20} {'内置重要性':>15} {'置换重要性':>15} {'重要性差异':>15}")
    print("-" * 60)
    for _, row in comparison.iterrows():
        feature = row['feature']
        builtin = row.get('builtin_importance', 0)
        perm = row.get('permutation_importance', 0)
        diff = perm - builtin
        print(f"{feature:<20} {builtin:>15.4f} {perm:>15.4f} {diff:>+15.4f}")

    # 可视化对比
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 内置重要性
    builtin_sorted = builtin_importance.sort_values('builtin_importance')
    axes[0].barh(builtin_sorted['feature'], builtin_sorted['builtin_importance'],
                  alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('重要性', fontsize=12)
    axes[0].set_title('基于不纯度的特征重要性（内置）', fontsize=14)
    axes[0].grid(True, alpha=0.3, axis='x')

    # 置换重要性
    perm_sorted = perm_importance.sort_values('permutation_importance')
    axes[1].barh(perm_sorted['feature'], perm_sorted['permutation_importance'],
                  alpha=0.7, color='steelblue', edgecolor='black')
    axes[1].set_xlabel('重要性（性能下降）', fontsize=12)
    axes[1].set_title('置换重要性（Permutation）', fontsize=14)
    axes[1].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'feature_importance_types.png', dpi=150, bbox_inches='tight')
    print("✅ 特征重要性对比图已保存为 feature_importance_types.png")
    plt.close()

    return comparison, rf


def demonstrate_correlation_effect(df: pd.DataFrame) -> None:
    """演示相关特征对重要性的影响"""
    X = df.drop('price', axis=1)
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 计算特征相关矩阵
    corr_matrix = X_train.corr()

    print("\n" + "=" * 60)
    print("特征相关性矩阵")
    print("=" * 60)
    print(corr_matrix.round(3))

    # 训练两个模型：全特征 vs 去除相关特征
    feature_names_all = X.columns.tolist()

    # 模型1：所有特征
    rf_all = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_all.fit(X_train, y_train)
    importance_all = pd.DataFrame({
        'feature': feature_names_all,
        'importance': rf_all.feature_importances_
    })

    # 模型2：去除 area_sqft（与 area_sqm 高度相关）
    features_reduced = [f for f in feature_names_all if f != 'area_sqft']
    X_train_reduced = X_train[features_reduced]
    X_test_reduced = X_test[features_reduced]

    rf_reduced = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_reduced.fit(X_train_reduced, y_train)
    importance_reduced = pd.DataFrame({
        'feature': features_reduced,
        'importance': rf_reduced.feature_importances_
    })

    # 对比 area_sqm 的重要性变化
    area_importance_all = importance_all[importance_all['feature'] == 'area_sqm']['importance'].values[0]
    area_importance_reduced = importance_reduced[importance_reduced['feature'] == 'area_sqm']['importance'].values[0]

    print("\n" + "=" * 60)
    print("相关特征对重要性的影响")
    print("=" * 60)
    print(f"\narea_sqm 的重要性:")
    print(f"  包含 area_sqft 时: {area_importance_all:.4f}")
    print(f"  去除 area_sqft 后: {area_importance_reduced:.4f}")
    print(f"  重要性增加: {(area_importance_reduced - area_importance_all):.4f} ({(area_importance_reduced / area_importance_all - 1) * 100:.1f}%)")

    print(f"\narea_sqft 的重要性:")
    area_sqft_importance = importance_all[importance_all['feature'] == 'area_sqft']['importance'].values[0]
    print(f"  包含时: {area_sqft_importance:.4f}")
    print(f"  说明: 与 area_sqm 高度相关（r = {corr_matrix.loc['area_sqm', 'area_sqft']:.3f}）")
    print(f"  结果: 重要性被'稀释'，因为模型可以选择任一个")

    # 性能对比
    r2_all = rf_all.score(X_test, y_test)
    r2_reduced = rf_reduced.score(X_test_reduced, y_test)

    print(f"\n模型性能:")
    print(f"  所有特征 R²: {r2_all:.3f}")
    print(f"  去除相关特征 R²: {r2_reduced:.3f}")
    print(f"  性能损失: {(r2_all - r2_reduced):.4f}")

    # 可视化相关性
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 相关性热图
    im = axes[0].imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[0].set_xticks(range(len(corr_matrix)))
    axes[0].set_yticks(range(len(corr_matrix)))
    axes[0].set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
    axes[0].set_yticklabels(corr_matrix.columns)
    axes[0].set_title('特征相关性矩阵', fontsize=14)

    # 添加数值标注
    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix)):
            text = axes[0].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                               ha="center", va="center", color="black", fontsize=10)

    plt.colorbar(im, ax=axes[0], label='相关系数')

    # 重要性对比（全特征 vs 去除相关）
    importance_comparison = pd.merge(
        importance_all,
        importance_reduced,
        on='feature',
        how='outer',
        suffixes=('_all', '_reduced')
    ).sort_values('importance_all', ascending=False)

    x = range(len(importance_comparison))
    width = 0.35

    axes[1].barh([i - width/2 for i in x], importance_comparison['importance_all'],
                  width, label='所有特征', alpha=0.7)
    axes[1].barh([i + width/2 for i in x], importance_comparison['importance_reduced'],
                  width, label='去除相关特征', alpha=0.7)
    axes[1].set_yticks(x)
    axes[1].set_yticklabels(importance_comparison['feature'])
    axes[1].set_xlabel('特征重要性', fontsize=12)
    axes[1].set_title('去除相关特征前后的重要性对比', fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'correlation_effect.png', dpi=150, bbox_inches='tight')
    print("✅ 相关性效果图已保存为 correlation_effect.png")
    plt.close()


def demonstrate_high_cardinality_trap() -> None:
    """
    演示高基数类别特征的陷阱

    高基数特征（如用户ID）可能被误认为"重要"
    """
    np.random.seed(42)

    n_samples = 1000

    # 生成数据
    X = pd.DataFrame({
        'useful_feature': np.random.randn(n_samples),
        'high_cardinality_id': np.arange(n_samples),  # 每个样本唯一
        'random_feature': np.random.randn(n_samples)
    })

    # 目标只由 useful_feature 决定
    y = 2 * X['useful_feature'] + np.random.randn(n_samples) * 0.5

    # 训练随机森林
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n" + "=" * 60)
    print("高基数特征陷阱演示")
    print("=" * 60)
    print("\n场景：用户ID（每个样本唯一） vs 真正有用的特征")
    print(importance.to_string(index=False))

    print(f"\n解读:")
    print(f"  - high_cardinality_id 被认为非常重要（实际上无信息）")
    print(f"  - 原因：决策树可以通过记住每个ID来'作弊'")
    print(f"  - 解决方法：删除高基数特征、使用目标编码、或使用置换重要性")


def explain_feature_importance_correctly() -> None:
    """正确解释特征重要性"""
    print("\n" + "=" * 60)
    print("特征重要性的正确解释")
    print("=" * 60)
    print("""
1. 基于不纯度的特征重要性（内置）
   - 定义：每个特征在分裂时对不纯度/MSE 的平均减少量
   - 优点：计算快速，无需额外训练
   - 缺点：
     * 偏向高基数特征（类别多的特征）
     * 相关特征会"稀释"重要性
     * 不能反映特征删除后的实际影响

2. 置换重要性
   - 定义：随机打乱某特征值，观察模型性能下降
   - 优点：
     * 更真实地反映特征重要性
     * 不受高基数影响
     * 可以用于任何模型（不只是树模型）
   - 缺点：计算较慢（需要多次预测）

3. 特征重要性的正确解释
   ✓ 正确："该特征在模型中被频繁使用"
   ✓ 正确："删除该特征会降低模型性能"
   ✗ 错误："该特征导致目标变化"（因果）
   ✗ 错误："该特征比其他特征更重要"（绝对排序）

4. 常见陷阱
   - 相关性陷阱：高度相关的特征会互相"稀释"重要性
   - 高基数陷阱：唯一ID等特征被误认为"重要"
   - 因果陷阱：特征重要性 ≠ 因果关系
   - 稳定性陷阱：不同数据集、不同模型可能得到不同排序

5. 实践建议
   - 优先使用置换重要性（更可靠）
   - 检查特征相关性，避免误读
   - 删除高基数无信息特征
   - 多次运行观察稳定性
   - 结合领域知识解释
    """)


def main() -> None:
    """主函数"""
    print("=" * 60)
    print("示例4: 特征重要性分析")
    print("=" * 60)

    # 1. 生成包含相关特征的数据
    print("\n步骤1: 生成包含相关特征的数据...")
    df = generate_data_with_correlation(n_samples=500)
    print(f"数据形状: {df.shape}")
    print(f"\n前5行:")
    print(df.head())

    # 2. 对比两种特征重要性方法
    print("\n步骤2: 对比内置重要性 vs 置换重要性...")
    comparison, rf = compare_importance_methods(df)

    # 3. 演示相关性的影响
    print("\n步骤3: 演示相关特征的影响...")
    demonstrate_correlation_effect(df)

    # 4. 演示高基数陷阱
    print("\n步骤4: 演示高基数特征陷阱...")
    demonstrate_high_cardinality_trap()

    # 5. 正确解释特征重要性
    print("\n步骤5: 特征重要性的正确解释...")
    explain_feature_importance_correctly()

    # 总结
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print("""
关键要点：
1. 内置特征重要性有偏向（高基数、相关特征）
2. 置换重要性更可靠（但计算更慢）
3. 相关特征会"稀释"重要性
4. 高基数特征（如ID）可能被误认为重要
5. 特征重要性 ≠ 因果关系

实践建议：
- 优先使用置换重要性
- 检查特征相关性矩阵
- 删除无信息的高基数特征
- 结合领域知识解释
- 多次运行观察稳定性
    """)

    print("\n" + "=" * 60)
    print("✅ 示例4完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()

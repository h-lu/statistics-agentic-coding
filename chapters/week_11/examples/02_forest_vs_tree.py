"""
示例：随机森林 vs 单棵决策树——方差降低的演示

本例演示：
1. 单棵决策树的预测不稳定性（高方差）
2. 随机森林通过 Bagging 降低方差的原理
3. 单树 vs 随机森林的性能对比
4. 特征重要性比较

运行方式：python3 chapters/week_11/examples/02_forest_vs_tree.py
预期输出：
- 预测不稳定性对比图（保存为 prediction_stability.png）
- 单树 vs 森林性能对比表
- 特征重要性对比图（保存为 feature_importance_comparison.png）
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_model import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

OUTPUT_DIR = Path(__file__).parent.parent.parent / "output" / "week_11"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 设置随机种子保证可复现
np.random.seed(42)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def generate_regression_data(n_samples: int = 500, noise: float = 20.0) -> pd.DataFrame:
    """
    生成回归数据（模拟房价预测场景）

    特征：
    - area_sqm: 面积
    - bedrooms: 卧室数
    - bathrooms: 浴室数
    - age_years: 房龄
    - distance_km: 距离市中心距离

    目标：
    - price: 房价（非线性关系 + 交互作用）
    """
    np.random.seed(42)

    area = np.random.uniform(30, 200, n_samples)
    bedrooms = np.random.randint(1, 6, n_samples)
    bathrooms = np.random.randint(1, 4, n_samples)
    age = np.random.uniform(0, 30, n_samples)
    distance = np.random.uniform(1, 50, n_samples)

    # 非线性关系 + 交互作用
    base_price = 50 + 5 * area - 0.02 * area**2  # 面积有边际递减
    base_price += 20 * bedrooms  # 卧室数线性贡献
    base_price += 15 * bathrooms * (area / 100)  # 浴室与面积有交互
    base_price -= 2 * age  # 房龄负面影响
    base_price -= 3 * distance  # 距离负面影响

    # 添加噪声
    price_noise = np.random.normal(0, noise, n_samples)
    price = base_price + price_noise

    return pd.DataFrame({
        'area_sqm': area,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'age_years': age,
        'distance_km': distance,
        'price': price
    })


def compare_single_tree_variance(df: pd.DataFrame, n_trials: int = 10) -> None:
    """
    演示单棵决策树的预测不稳定性（高方差）

    用不同的随机种子训练多棵树，观察预测差异
    """
    X = df.drop('price', axis=1)
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 固定一个测试样本
    sample_idx = 0
    X_sample = X_test.iloc[[sample_idx]]

    predictions = []
    for seed in range(n_trials):
        tree = DecisionTreeRegressor(max_depth=5, random_state=seed)
        tree.fit(X_train, y_train)
        pred = tree.predict(X_sample)[0]
        predictions.append(pred)

    # 分析预测方差
    pred_mean = np.mean(predictions)
    pred_std = np.std(predictions)
    true_value = y_test.iloc[sample_idx]

    print("\n" + "=" * 60)
    print("单棵决策树的预测不稳定性（10 次不同随机种子）")
    print("=" * 60)
    print(f"真实值: {true_value:.1f}")
    print(f"预测均值: {pred_mean:.1f}")
    print(f"预测标准差: {pred_std:.1f}")
    print(f"变异系数: {(pred_std / pred_mean * 100):.1f}%")
    print(f"\n各次预测: {[f'{p:.1f}' for p in predictions]}")

    # 画图展示预测分布
    plt.figure(figsize=(10, 6))
    plt.bar(range(n_trials), predictions, alpha=0.7, edgecolor='black')
    plt.axhline(y=true_value, color='red', linestyle='--', linewidth=2, label=f'真实值 ({true_value:.1f})')
    plt.axhline(y=pred_mean, color='green', linestyle='-', linewidth=2, label=f'预测均值 ({pred_mean:.1f})')
    plt.xlabel('随机种子（不同训练样本）', fontsize=12)
    plt.ylabel('预测房价（万元）', fontsize=12)
    plt.title('单棵决策树的预测不稳定性（高方差）', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'prediction_stability.png', dpi=150, bbox_inches='tight')
    print("✅ 预测不稳定性图已保存为 prediction_stability.png")
    plt.close()


def compare_tree_vs_forest(df: pd.DataFrame) -> None:
    """对比单棵决策树 vs 随机森林的性能"""
    X = df.drop('price', axis=1)
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 单棵决策树
    tree = DecisionTreeRegressor(max_depth=7, random_state=42)
    tree.fit(X_train, y_train)

    tree_train_r2 = tree.score(X_train, y_train)
    tree_test_r2 = tree.score(X_test, y_test)
    tree_test_mse = mean_squared_error(y_test, tree.predict(X_test))

    # 随机森林
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=7,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    rf_train_r2 = rf.score(X_train, y_train)
    rf_test_r2 = rf.score(X_test, y_test)
    rf_test_mse = mean_squared_error(y_test, rf.predict(X_test))

    # 打印对比结果
    print("\n" + "=" * 60)
    print("单棵决策树 vs 随机森林：性能对比")
    print("=" * 60)
    print(f"{'指标':<20} {'决策树':>15} {'随机森林':>15} {'改进':>15}")
    print("-" * 60)
    print(f"{'训练集 R²':<20} {tree_train_r2:>15.3f} {rf_train_r2:>15.3f} {(rf_train_r2 - tree_train_r2):>+15.3f}")
    print(f"{'测试集 R²':<20} {tree_test_r2:>15.3f} {rf_test_r2:>15.3f} {(rf_test_r2 - tree_test_r2):>+15.3f}")
    print(f"{'测试集 MSE':<20} {tree_test_mse:>15.1f} {rf_test_mse:>15.1f} {(tree_test_mse - rf_test_mse):>+15.1f}")

    # 方差降低分析
    print("\n" + "=" * 60)
    print("Bagging 的方差降低效果")
    print("=" * 60)
    print(f"训练集过拟合程度:")
    print(f"  决策树: {tree_train_r2 - tree_test_r2:.3f}")
    print(f"  随机森林: {rf_train_r2 - rf_test_r2:.3f}")
    print(f"  方差降低: {(tree_train_r2 - tree_test_r2) - (rf_train_r2 - rf_test_r2):.3f}")

    return tree, rf


def compare_feature_importance(tree: DecisionTreeRegressor, rf: RandomForestRegressor, feature_names: list) -> None:
    """对比单树 vs 森林的特征重要性"""
    tree_importance = tree.feature_importances_
    rf_importance = rf.feature_importances_

    # 创建对比图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 决策树特征重要性
    idx = np.argsort(tree_importance)
    axes[0].barh(range(len(idx)), tree_importance[idx], alpha=0.7, edgecolor='black')
    axes[0].set_yticks(range(len(idx)))
    axes[0].set_yticklabels([feature_names[i] for i in idx])
    axes[0].set_xlabel('重要性', fontsize=12)
    axes[0].set_title('单棵决策树：特征重要性', fontsize=14)
    axes[0].grid(True, alpha=0.3, axis='x')

    # 随机森林特征重要性
    idx = np.argsort(rf_importance)
    axes[1].barh(range(len(idx)), rf_importance[idx], alpha=0.7, color='steelblue', edgecolor='black')
    axes[1].set_yticks(range(len(idx)))
    axes[1].set_yticklabels([feature_names[i] for i in idx])
    axes[1].set_xlabel('重要性', fontsize=12)
    axes[1].set_title('随机森林：特征重要性', fontsize=14)
    axes[1].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'feature_importance_comparison.png', dpi=150, bbox_inches='tight')
    print("✅ 特征重要性对比图已保存为 feature_importance_comparison.png")
    plt.close()

    # 打印特征重要性排序
    print("\n" + "=" * 60)
    print("特征重要性排序对比")
    print("=" * 60)

    tree_df = pd.DataFrame({
        '特征': feature_names,
        '决策树': tree_importance
    }).sort_values('决策树', ascending=False)

    rf_df = pd.DataFrame({
        '特征': feature_names,
        '随机森林': rf_importance
    }).sort_values('随机森林', ascending=False)

    comparison = pd.merge(tree_df, rf_df, on='特征', how='outer')
    print(comparison.to_string(index=False))


def demonstrate_bagging_effect(df: pd.DataFrame) -> None:
    """演示 Bagging 的平均化效果"""
    X = df.drop('price', axis=1)
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 训练多棵树
    n_trees = [1, 5, 10, 25, 50, 100]
    test_r2_scores = []
    test_std_scores = []

    for n in n_trees:
        rf = RandomForestRegressor(
            n_estimators=n,
            max_depth=7,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)

        # 整体预测
        y_pred = rf.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        test_r2_scores.append(r2)

        # 计算单树预测的标准差（评估方差）
        tree_preds = np.array([tree.predict(X_test) for tree in rf.estimators_])
        tree_std = np.std(tree_preds, axis=0).mean()
        test_std_scores.append(tree_std)

    # 画图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].plot(n_trees, test_r2_scores, 'o-', linewidth=2, markersize=8)
    axes[0].set_xlabel('树的数量 (n_estimators)', fontsize=12)
    axes[0].set_ylabel('测试集 R²', fontsize=12)
    axes[0].set_title('树的数量 vs 性能', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xscale('log')

    axes[1].plot(n_trees, test_std_scores, 's-', linewidth=2, markersize=8, color='orange')
    axes[1].set_xlabel('树的数量 (n_estimators)', fontsize=12)
    axes[1].set_ylabel('单树预测标准差', fontsize=12)
    axes[1].set_title('树的数量 vs 预测方差', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xscale('log')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'bagging_effect.png', dpi=150, bbox_inches='tight')
    print("✅ Bagging 效果图已保存为 bagging_effect.png")
    plt.close()


def main() -> None:
    """主函数"""
    print("=" * 60)
    print("示例2: 随机森林 vs 单棵决策树")
    print("=" * 60)

    # 1. 生成数据
    print("\n步骤1: 生成回归数据...")
    df = generate_regression_data(n_samples=500, noise=20)
    print(f"数据形状: {df.shape}")
    print(f"\n数据前5行:")
    print(df.head())

    # 2. 演示单树的不稳定性
    print("\n步骤2: 演示单棵决策树的预测不稳定性...")
    compare_single_tree_variance(df, n_trials=10)

    # 3. 对比单树 vs 森林
    print("\n步骤3: 对比单棵决策树 vs 随机森林...")
    tree, rf = compare_tree_vs_forest(df)

    # 4. 对比特征重要性
    print("\n步骤4: 对比特征重要性...")
    feature_names = df.drop('price', axis=1).columns.tolist()
    compare_feature_importance(tree, rf, feature_names)

    # 5. 演示 Bagging 效果
    print("\n步骤5: 演示 Bagging 的平均化效果...")
    demonstrate_bagging_effect(df)

    # 总结
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print("""
随机森林的核心优势：
1. 降低方差：多棵树投票/平均，预测更稳定
2. 减少过拟合：Bootstrap 采样 + 特征随机性
3. 性能提升：R² 通常比单棵树高 5-15%
4. 特征重要性更可靠：不依赖单次分裂的随机性

Bagging 的两个随机性来源：
1. Bootstrap 采样：每棵树看到不同的训练子样本
2. 特征随机性：每次分裂只考虑部分特征

何时使用随机森林？
- 需要稳健的预测性能
- 特征重要性分析
- 数据有非线性、交互作用
- 不需要极致的可解释性

何时使用单棵决策树？
- 需要"如果-那么"规则解释
- 快速原型验证
- 计算资源有限
    """)

    print("\n" + "=" * 60)
    print("✅ 示例2完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
示例：决策树与随机森林的超参数调优

本例演示：
1. 使用 GridSearchCV 进行网格搜索（穷举所有参数组合）
2. 使用 RandomizedSearchCV 进行随机搜索（随机采样参数组合）
3. 验证曲线：展示单个超参数对性能的影响
4. 嵌套交叉验证：防止过拟合验证集

运行方式：python3 chapters/week_11/examples/03_hyperparameter_tuning.py
预期输出：
- 网格搜索结果（最佳参数、最佳分数）
- 随机搜索对比结果
- 验证曲线图（保存为 validation_curve.png）
- 控制台输出超参数选择建议
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    train_test_split, GridSearchCV, RandomizedSearchCV,
    validation_curve, cross_val_score
)
from sklearn.metrics import r2_score
from scipy.stats import randint

OUTPUT_DIR = Path(__file__).parent.parent.parent / "output" / "week_11"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 设置随机种子保证可复现
np.random.seed(42)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def generate_regression_data(n_samples: int = 500) -> pd.DataFrame:
    """生成回归数据（与示例2相同）"""
    np.random.seed(42)

    area = np.random.uniform(30, 200, n_samples)
    bedrooms = np.random.randint(1, 6, n_samples)
    bathrooms = np.random.randint(1, 4, n_samples)
    age = np.random.uniform(0, 30, n_samples)
    distance = np.random.uniform(1, 50, n_samples)

    base_price = 50 + 5 * area - 0.02 * area**2
    base_price += 20 * bedrooms
    base_price += 15 * bathrooms * (area / 100)
    base_price -= 2 * age
    base_price -= 3 * distance

    price_noise = np.random.normal(0, 20, n_samples)
    price = base_price + price_noise

    return pd.DataFrame({
        'area_sqm': area,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'age_years': age,
        'distance_km': distance,
        'price': price
    })


def grid_search_example(df: pd.DataFrame) -> None:
    """演示网格搜索（GridSearchCV）"""
    X = df.drop('price', axis=1)
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("\n" + "=" * 60)
    print("网格搜索（GridSearchCV）：穷举所有参数组合")
    print("=" * 60)

    # 定义参数网格
    param_grid = {
        'max_depth': [3, 5, 7, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 5, 10]
    }

    print(f"\n参数网格:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")

    # 计算总组合数
    n_combinations = 1
    for values in param_grid.values():
        n_combinations *= len(values)
    print(f"\n总组合数: {n_combinations}")
    print(f"5-fold CV 总拟合次数: {n_combinations * 5}")

    # 网格搜索
    grid_search = GridSearchCV(
        DecisionTreeRegressor(random_state=42),
        param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        return_train_score=True
    )

    grid_search.fit(X_train, y_train)

    # 结果
    print(f"\n最佳参数: {grid_search.best_params_}")
    print(f"最佳 CV R²: {grid_search.best_score_:.3f}")

    # 测试集性能
    best_tree = grid_search.best_estimator_
    test_r2 = best_tree.score(X_test, y_test)
    print(f"测试集 R²: {test_r2:.3f}")

    # 查看所有结果
    results = pd.DataFrame(grid_search.cv_results_)
    print(f"\nTop 5 参数组合:")
    top_results = results.nlargest(5, 'mean_test_score')[['param_max_depth', 'param_min_samples_split', 'param_min_samples_leaf', 'mean_test_score']]
    print(top_results.to_string(index=False))

    return grid_search


def randomized_search_example(df: pd.DataFrame) -> None:
    """演示随机搜索（RandomizedSearchCV）"""
    X = df.drop('price', axis=1)
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("\n" + "=" * 60)
    print("随机搜索（RandomizedSearchCV）：随机采样参数组合")
    print("=" * 60)

    # 定义参数分布（不是网格）
    param_dist = {
        'n_estimators': randint(50, 300),
        'max_depth': randint(3, 15),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'max_features': ['sqrt', 'log2', 0.5, 0.8]
    }

    print(f"\n参数分布:")
    for param, dist in param_dist.items():
        print(f"  {param}: {dist}")

    # 随机搜索
    random_search = RandomizedSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        param_distributions=param_dist,
        n_iter=50,  # 只采样 50 个组合
        cv=5,
        scoring='r2',
        random_state=42,
        n_jobs=-1,
        return_train_score=True
    )

    random_search.fit(X_train, y_train)

    # 结果
    print(f"\n最佳参数: {random_search.best_params_}")
    print(f"最佳 CV R²: {random_search.best_score_:.3f}")

    # 测试集性能
    best_rf = random_search.best_estimator_
    test_r2 = best_rf.score(X_test, y_test)
    print(f"测试集 R²: {test_r2:.3f}")

    # 对比网格搜索和随机搜索的效率
    print(f"\n效率对比:")
    print(f"  随机搜索迭代次数: 50")
    print(f"  5-fold CV 总拟合次数: 50 * 5 = 250")
    print(f"  如果用网格搜索，假设每个参数 10 个值：10^5 * 5 = 500,000+ 次")

    return random_search


def plot_validation_curve(df: pd.DataFrame) -> None:
    """画验证曲线：展示单个超参数对性能的影响"""
    X = df.drop('price', axis=1)
    y = df['price']

    param_name = 'max_depth'
    param_range = [1, 2, 3, 5, 7, 10, 15, None]

    # 计算验证曲线
    train_scores, val_scores = validation_curve(
        DecisionTreeRegressor(random_state=42),
        X, y,
        param_name=param_name,
        param_range=param_range,
        cv=5,
        scoring='r2',
        n_jobs=-1
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)

    # 画图
    plt.figure(figsize=(10, 6))

    param_labels = [str(p) if p is not None else 'None' for p in param_range]
    plt.plot(param_labels, train_scores_mean, 'o-', linewidth=2,
             label='训练集', markersize=8)
    plt.fill_between(param_labels,
                     train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std,
                     alpha=0.2)

    plt.plot(param_labels, val_scores_mean, 's-', linewidth=2,
             label='验证集', markersize=8)
    plt.fill_between(param_labels,
                     val_scores_mean - val_scores_std,
                     val_scores_mean + val_scores_std,
                     alpha=0.2)

    plt.xlabel('max_depth (树的最大深度)', fontsize=12)
    plt.ylabel('R² 分数', fontsize=12)
    plt.title('验证曲线：max_depth 对性能的影响', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'validation_curve.png', dpi=150, bbox_inches='tight')
    print("✅ 验证曲线图已保存为 validation_curve.png")
    plt.close()

    # 找到最佳深度
    best_idx = np.argmax(val_scores_mean)
    best_depth = param_range[best_idx]
    best_score = val_scores_mean[best_idx]
    print(f"\n最佳 max_depth: {best_depth} (CV R² = {best_score:.3f})")


def compare_grid_vs_random(df: pd.DataFrame) -> None:
    """对比网格搜索和随机搜索"""
    X = df.drop('price', axis=1)
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("\n" + "=" * 60)
    print("网格搜索 vs 随机搜索：效率与性能对比")
    print("=" * 60)

    # 网格搜索（小范围）
    param_grid = {
        'max_depth': [5, 7, 10],
        'min_samples_leaf': [1, 5, 10]
    }

    from time import time
    start = time()
    grid_search = GridSearchCV(
        RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
        param_grid,
        cv=3,
        scoring='r2',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    grid_time = time() - start

    # 随机搜索（大范围，更少迭代）
    param_dist = {
        'n_estimators': randint(50, 200),
        'max_depth': randint(3, 15),
        'min_samples_leaf': randint(1, 10),
        'max_features': ['sqrt', 'log2', 0.5, 0.8]
    }

    start = time()
    random_search = RandomizedSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        param_distributions=param_dist,
        n_iter=30,
        cv=3,
        scoring='r2',
        random_state=42,
        n_jobs=-1
    )
    random_search.fit(X_train, y_train)
    random_time = time() - start

    # 对比结果
    grid_test_r2 = grid_search.best_estimator_.score(X_test, y_test)
    random_test_r2 = random_search.best_estimator_.score(X_test, y_test)

    print(f"\n网格搜索:")
    print(f"  时间: {grid_time:.1f} 秒")
    print(f"  最佳参数: {grid_search.best_params_}")
    print(f"  测试集 R²: {grid_test_r2:.3f}")

    print(f"\n随机搜索:")
    print(f"  时间: {random_time:.1f} 秒")
    print(f"  最佳参数: {random_search.best_params_}")
    print(f"  测试集 R²: {random_test_r2:.3f}")

    print(f"\n结论:")
    print(f"  性能差异: {abs(random_test_r2 - grid_test_r2):.4f}")
    print(f"  时间差异: {random_time - grid_time:+.1f} 秒")


def main() -> None:
    """主函数"""
    print("=" * 60)
    print("示例3: 超参数调优——网格搜索 vs 随机搜索")
    print("=" * 60)

    # 1. 生成数据
    print("\n步骤1: 生成回归数据...")
    df = generate_regression_data(n_samples=500)
    print(f"数据形状: {df.shape}")

    # 2. 网格搜索示例
    print("\n步骤2: 网格搜索...")
    grid_search_example(df)

    # 3. 随机搜索示例
    print("\n步骤3: 随机搜索...")
    randomized_search_example(df)

    # 4. 验证曲线
    print("\n步骤4: 画验证曲线...")
    plot_validation_curve(df)

    # 5. 对比网格搜索 vs 随机搜索
    print("\n步骤5: 对比网格搜索 vs 随机搜索...")
    compare_grid_vs_random(df)

    # 总结
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print("""
超参数调优的关键要点：

1. 网格搜索（GridSearchCV）
   - 优点：保证找到搜索范围内的最优参数
   - 缺点：计算成本高（组合爆炸）
   - 适用：参数少、搜索范围小

2. 随机搜索（RandomizedSearchCV）
   - 优点：效率高，更适合高维搜索空间
   - 缺点：可能错过最优参数
   - 适用：参数多、搜索范围大

3. 超参数选择建议
   - max_depth: 从 3-10 开始，None 容易过拟合
   - min_samples_leaf: 1-10，控制过拟合
   - min_samples_split: 2-20，防止过细分裂
   - n_estimators: 100-300，更多树更稳定（但收益递减）
   - max_features: 'sqrt'（分类）或 1/3（回归）

4. 防止过拟合验证集
   - 使用嵌套交叉验证
   - 保留测试集只在最终评估时使用
   - 不要在测试集上反复调参数

5. 调优策略
   - 先用随机搜索快速探索
   - 再用网格搜索精细调优
   - 用验证曲线观察单个参数的影响
    """)

    print("\n" + "=" * 60)
    print("✅ 示例3完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()

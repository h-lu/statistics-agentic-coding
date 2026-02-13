"""
示例：决策树可视化与深度对过拟合的影响

本例演示：
1. 如何用决策树拟合回归/分类数据
2. 决策树结构的可视化（plot_tree）
3. 不同 max_depth 对过拟合的影响
4. 导出决策树规则为可读文本

运行方式：python3 chapters/week_11/examples/01_decision_tree_viz.py
预期输出：
- 决策树可视化图（保存为 decision_tree_depth_3.png）
- 深度对性能影响的图表（保存为 depth_vs_performance.png）
- 控制台输出树规则文本
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

OUTPUT_DIR = Path(__file__).parent.parent.parent / "output" / "week_11"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 设置随机种子保证可复现
np.random.seed(42)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def generate_nonlinear_data(n_samples: int = 300) -> pd.DataFrame:
    """
    生成非线性关系数据（模拟房价 vs 面积的非线性关系）

    真实关系：y = f(x) + 噪声
    其中 f(x) 是非线性函数（有边际递减效应）
    """
    np.random.seed(42)
    x = np.random.uniform(20, 150, n_samples)

    # 真实关系：非线性（边际递减）
    # 小户型时每平米单价高，大户型时单价趋于平稳
    true_y = 200 + 8 * x - 0.02 * x**2

    # 添加噪声
    noise = np.random.normal(0, 30, n_samples)
    y = true_y + noise

    return pd.DataFrame({'area_sqm': x, 'price': y})


def plot_data_with_linear_fit(df: pd.DataFrame) -> None:
    """画数据散点图和线性拟合（展示线性模型的局限）"""
    from sklearn.linear_model import LinearRegression

    X = df[['area_sqm']]
    y = df['price']

    # 线性回归
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    y_pred_linear = lin_reg.predict(X)

    # 画图
    plt.figure(figsize=(10, 6))
    plt.scatter(df['area_sqm'], df['price'], alpha=0.5, s=40, label='真实数据')
    plt.plot(df['area_sqm'], y_pred_linear, 'r-', linewidth=2, label='线性回归拟合')

    plt.xlabel('面积 (㎡)', fontsize=12)
    plt.ylabel('价格 (万元)', fontsize=12)
    plt.title('线性回归 vs 非线性数据', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'linear_vs_nonlinear_data.png', dpi=150, bbox_inches='tight')
    print("✅ 线性 vs 非线性图已保存为 linear_vs_nonlinear_data.png")
    plt.close()


def fit_and_visualize_tree(df: pd.DataFrame, max_depth: int = 3) -> DecisionTreeRegressor:
    """拟合决策树并可视化"""
    X = df[['area_sqm']]
    y = df['price']

    # 划分数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 拟合决策树
    tree = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    tree.fit(X_train, y_train)

    # 计算性能
    train_r2 = tree.score(X_train, y_train)
    test_r2 = tree.score(X_test, y_test)

    print(f"\n深度={max_depth} 的决策树:")
    print(f"  训练集 R²: {train_r2:.3f}")
    print(f"  测试集 R²: {test_r2:.3f}")
    print(f"  过拟合程度: {train_r2 - test_r2:.3f}")

    # 可视化树结构
    plt.figure(figsize=(16, 10))
    plot_tree(
        tree,
        feature_names=['面积(㎡)'],
        filled=True,
        rounded=True,
        fontsize=10,
        precision=1,
        impurity=False  # 不显示不纯度（MSE）
    )
    plt.title(f'决策树结构可视化 (max_depth={max_depth})', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'decision_tree_depth_{max_depth}.png', dpi=150, bbox_inches='tight')
    print(f"✅ 决策树可视化图已保存为 decision_tree_depth_{max_depth}.png")
    plt.close()

    return tree


def export_tree_rules(tree: DecisionTreeRegressor, feature_names: list) -> None:
    """导出决策树规则为文本"""
    rules = export_text(
        tree,
        feature_names=feature_names,
        decimals=1
    )

    print("\n" + "=" * 60)
    print("决策树规则（如果-那么）:")
    print("=" * 60)
    print(rules)


def analyze_depth_effect(df: pd.DataFrame) -> None:
    """分析不同深度对过拟合的影响"""
    X = df[['area_sqm']]
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    depths = [1, 2, 3, 5, 7, 10, None]
    train_r2_scores = []
    test_r2_scores = []

    for depth in depths:
        if depth is None:
            tree = DecisionTreeRegressor(random_state=42)
            depth_label = "None (无限制)"
        else:
            tree = DecisionTreeRegressor(max_depth=depth, random_state=42)
            depth_label = str(depth)

        tree.fit(X_train, y_train)
        train_r2 = tree.score(X_train, y_train)
        test_r2 = tree.score(X_test, y_test)

        train_r2_scores.append(train_r2)
        test_r2_scores.append(test_r2)

        print(f"深度={depth_label:>10}: 训练R²={train_r2:.3f}, 测试R²={test_r2:.3f}")

    # 画图
    plt.figure(figsize=(10, 6))
    depth_labels = ['1', '2', '3', '5', '7', '10', 'None']

    plt.plot(range(len(depth_labels)), train_r2_scores,
             'o-', linewidth=2, label='训练集 R²', markersize=8)
    plt.plot(range(len(depth_labels)), test_r2_scores,
             's-', linewidth=2, label='测试集 R²', markersize=8)

    plt.xticks(range(len(depth_labels)), depth_labels)
    plt.xlabel('max_depth (树的最大深度)', fontsize=12)
    plt.ylabel('R² 分数', fontsize=12)
    plt.title('决策树深度对过拟合的影响', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'depth_vs_performance.png', dpi=150, bbox_inches='tight')
    print("✅ 深度影响图已保存为 depth_vs_performance.png")
    plt.close()

    # 找到最佳深度
    best_idx = int(np.argmax(test_r2_scores))
    print(f"\n最佳深度: {depth_labels[best_idx]} (测试集 R² = {test_r2_scores[best_idx]:.3f})")


def main() -> None:
    """主函数"""
    print("=" * 60)
    print("示例1: 决策树可视化与过拟合分析")
    print("=" * 60)

    # 1. 生成非线性数据
    print("\n步骤1: 生成非线性数据...")
    df = generate_nonlinear_data(n_samples=300)
    print(f"数据形状: {df.shape}")
    print(f"\n前5行:")
    print(df.head())

    # 2. 展示线性回归的局限
    print("\n步骤2: 对比线性回归...")
    plot_data_with_linear_fit(df)

    # 3. 拟合并可视化决策树（深度=3）
    print("\n步骤3: 拟合决策树 (max_depth=3)...")
    tree = fit_and_visualize_tree(df, max_depth=3)

    # 4. 导出规则
    print("\n步骤4: 导出决策树规则...")
    export_tree_rules(tree, ['面积(㎡)'])

    # 5. 分析深度影响
    print("\n步骤5: 分析不同深度的影响...")
    analyze_depth_effect(df)

    # 总结
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print("""
决策树的关键特点：
1. 可解释性强：每个节点都是"如果-那么"规则
2. 能捕捉非线性：不受线性假设限制
3. 容易过拟合：深度越大，训练集R²越接近1，但测试集性能下降

控制过拟合的方法：
- max_depth: 限制树的最大深度
- min_samples_split: 节点分裂所需的最小样本数
- min_samples_leaf: 叶子节点的最小样本数
- ccp_alpha: 剪枝参数（事后剪枝）

实际使用建议：
- 从 max_depth=3 或 5 开始
- 用交叉验证选择最佳深度
- 优先考虑随机森林（降低方差）
    """)

    print("\n" + "=" * 60)
    print("✅ 示例1完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()

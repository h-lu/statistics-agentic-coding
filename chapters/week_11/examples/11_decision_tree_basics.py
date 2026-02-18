"""
示例：决策树基础——从根节点到叶节点，树结构可视化与过拟合演示

运行方式：python3 chapters/week_11/examples/11_decision_tree_basics.py
预期输出：决策树可视化图（保存到 images/）、训练/测试 AUC 对比、过拟合演示
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import seaborn as sns

# 配置中文字体
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


def load_titanic_data() -> tuple:
    """加载并准备泰坦尼克数据集"""
    print("=" * 60)
    print("加载泰坦尼克数据集")
    print("=" * 60)

    titanic = sns.load_dataset("titanic")

    # 选择特征
    feature_cols = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
    X = titanic[feature_cols].copy()
    y = titanic['survived']

    # 简化：删除缺失值（仅用于示例，生产环境应做更完善的缺失值处理）
    X_clean = X.dropna()
    y_clean = y.loc[X_clean.index]

    # 将分类型变量转为数值（为了简化，这里手动编码）
    X_clean = X_clean.copy()
    X_clean['sex'] = X_clean['sex'].map({'male': 0, 'female': 1})
    X_clean['embarked'] = X_clean['embarked'].map({'C': 0, 'Q': 1, 'S': 2})
    X_clean['pclass'] = X_clean['pclass'].astype(str)

    # 对 pclass 进行 one-hot 编码
    X_clean = pd.get_dummies(X_clean, columns=['pclass'], drop_first=False)

    print(f"数据集规模: {X_clean.shape[0]} 行, {X_clean.shape[1]} 列")
    print(f"特征列表: {list(X_clean.columns)}")
    print(f"目标变量: survived (0=未生存, 1=生存)")
    print(f"类别分布: 0: {(y_clean==0).sum()}, 1: {(y_clean==1).sum()}")

    return X_clean, y_clean


def train_and_visualize_tree(X: pd.DataFrame, y: pd.Series, max_depth: int = 3) -> dict:
    """
    训练决策树并可视化

    参数:
    - X: 特征 DataFrame
    - y: 目标变量
    - max_depth: 树的最大深度

    返回:
    - dict: 包含模型和评估指标的字典
    """
    print("\n" + "=" * 60)
    print(f"训练决策树 (max_depth={max_depth})")
    print("=" * 60)

    # 划分数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 训练决策树
    tree = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42
    )
    tree.fit(X_train, y_train)

    # 预测
    y_pred_train = tree.predict(X_train)
    y_pred_test = tree.predict(X_test)
    y_prob_train = tree.predict_proba(X_train)[:, 1]
    y_prob_test = tree.predict_proba(X_test)[:, 1]

    # 评估
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    train_auc = roc_auc_score(y_train, y_prob_train)
    test_auc = roc_auc_score(y_test, y_prob_test)

    print(f"\n训练集: 准确率={train_acc:.4f}, AUC={train_auc:.4f}")
    print(f"测试集: 准确率={test_acc:.4f}, AUC={test_auc:.4f}")
    print(f"泛化差距: AUC 差距={train_auc - test_auc:.4f}")

    # 可视化树结构
    font = setup_chinese_font()
    fig, ax = plt.subplots(figsize=(20, 10))

    plot_tree(tree,
              feature_names=X.columns,
              class_names=['未生存', '生存'],
              filled=True,
              rounded=True,
              fontsize=10,
              ax=ax)

    ax.set_title(f'决策树结构 (max_depth={max_depth})', fontsize=14)

    plt.tight_layout()

    # 保存图片
    output_dir = Path(__file__).parent.parent / 'images'
    output_dir.mkdir(exist_ok=True)
    img_path = output_dir / f'decision_tree_depth_{max_depth}.png'
    plt.savefig(img_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"\n树结构图已保存到: {img_path}")

    # 打印树规则
    print("\n决策树规则解读:")
    print_tree_rules(tree, X.columns)

    return {
        'model': tree,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'train_auc': train_auc,
        'test_auc': test_auc
    }


def print_tree_rules(tree: DecisionTreeClassifier, feature_names: list) -> None:
    """打印决策树的决策规则（文本形式）"""
    n_nodes = tree.tree_.node_count
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right
    feature = tree.tree_.feature
    threshold = tree.tree_.threshold

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)

    # 遍历树计算节点深度
    stack = [(0, 0)]
    while len(stack) > 0:
        node_id, depth = stack.pop()
        node_depth[node_id] = depth

        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True

    print("\n节点信息:")
    for i in range(n_nodes):
        if is_leaves[i]:
            print(f"节点 {i} (叶节点): 样本数={tree.tree_.n_node_samples[i]}, "
                  f"值={tree.tree_.value[i][0]}")
        else:
            feat_name = feature_names[feature[i]]
            thresh = threshold[i]
            print(f"节点 {i} (内部节点): 如果 {feat_name} < {thresh:.2f}, "
                  f"样本数={tree.tree_.n_node_samples[i]}")


def demonstrate_overfitting() -> None:
    """
    演示决策树的过拟合：对比不同深度的树
    """
    print("\n" + "=" * 60)
    print("演示过拟合：不同深度的决策树对比")
    print("=" * 60)

    X, y = load_titanic_data()

    depths = [2, 3, 5, 10, None]  # None 表示不限制深度
    results = []

    for depth in depths:
        depth_label = "无限制" if depth is None else str(depth)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        # 训练树
        if depth is None:
            tree = DecisionTreeClassifier(random_state=42)
        else:
            tree = DecisionTreeClassifier(
                max_depth=depth,
                min_samples_split=2 if depth <= 3 else 20,
                min_samples_leaf=1 if depth <= 3 else 10,
                random_state=42
            )

        tree.fit(X_train, y_train)

        # 评估
        train_auc = roc_auc_score(y_train, tree.predict_proba(X_train)[:, 1])
        test_auc = roc_auc_score(y_test, tree.predict_proba(X_test)[:, 1])

        results.append({
            'depth': depth_label,
            'n_nodes': tree.tree_.node_count,
            'train_auc': train_auc,
            'test_auc': test_auc,
            'gap': train_auc - test_auc
        })

    # 打印结果表
    print(f"\n{'深度':<10} | {'节点数':<10} | {'训练 AUC':<12} | {'测试 AUC':<12} | {'差距':<10}")
    print("-" * 70)
    for r in results:
        print(f"{r['depth']:<10} | {r['n_nodes']:<10} | "
              f"{r['train_auc']:<12.4f} | {r['test_auc']:<12.4f} | "
              f"{r['gap']:<10.4f}")

    # 可视化过拟合曲线
    font = setup_chinese_font()
    fig, ax = plt.subplots(figsize=(10, 6))

    depths_plot = [r['depth'] for r in results]
    train_aucs = [r['train_auc'] for r in results]
    test_aucs = [r['test_auc'] for r in results]

    x_pos = np.arange(len(depths_plot))
    width = 0.35

    ax.bar(x_pos - width/2, train_aucs, width, label='训练集 AUC', color='blue', alpha=0.7)
    ax.bar(x_pos + width/2, test_aucs, width, label='测试集 AUC', color='green', alpha=0.7)

    ax.set_xlabel('树的最大深度')
    ax.set_ylabel('AUC')
    ax.set_title('决策树过拟合演示：训练集 vs 测试集 AUC')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(depths_plot)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0.5, 1.0)

    # 添加注释
    for i, r in enumerate(results):
        if r['gap'] > 0.1:
            ax.annotate(f"过拟合!\n差距={r['gap']:.2f}",
                       xy=(i + width/2, r['test_auc']),
                       xytext=(i + width/2, r['train_auc'] - 0.05),
                       arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                       fontsize=9, ha='center', color='red')

    plt.tight_layout()

    # 保存图片
    output_dir = Path(__file__).parent.parent / 'images'
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'decision_tree_overfitting.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"\n过拟合演示图已保存到: {output_dir / 'decision_tree_overfitting.png'}")


def demonstrate_pruning() -> None:
    """
    演示剪枝：预剪枝 vs 后剪枝
    """
    print("\n" + "=" * 60)
    print("演示剪枝：控制过拟合的两种方法")
    print("=" * 60)

    X, y = load_titanic_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 1. 预剪枝（Pre-pruning）：提前停止生长
    print("\n【预剪枝】通过限制参数提前停止树的生长")
    pre_pruned = DecisionTreeClassifier(
        max_depth=4,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42
    )
    pre_pruned.fit(X_train, y_train)

    train_auc_pre = roc_auc_score(y_train, pre_pruned.predict_proba(X_train)[:, 1])
    test_auc_pre = roc_auc_score(y_test, pre_pruned.predict_proba(X_test)[:, 1])

    print(f"  训练集 AUC: {train_auc_pre:.4f}")
    print(f"  测试集 AUC: {test_auc_pre:.4f}")
    print(f"  节点数: {pre_pruned.tree_.node_count}")

    # 2. 后剪枝（Post-pruning）：CCP 剪枝
    print("\n【后剪枝】使用成本复杂度剪枝（CCP）")

    # 计算剪枝路径
    full_tree = DecisionTreeClassifier(random_state=42)
    full_tree.fit(X_train, y_train)

    path = full_tree.cost_complexity_pruning_path(X_train, y_train)
    alphas = path.ccp_alphas

    print(f"  可用的 alpha 值数量: {len(alphas)}")
    print(f"  alpha 范围: [{alphas.min():.6f}, {alphas.max():.6f}]")

    # 用交叉验证选择最佳 alpha
    from sklearn.model_selection import cross_val_score

    cv_scores = []
    for alpha in alphas:
        tree = DecisionTreeClassifier(ccp_alpha=alpha, random_state=42)
        scores = cross_val_score(tree, X_train, y_train, cv=5, scoring='roc_auc')
        cv_scores.append(scores.mean())

    # 选择最佳 alpha
    best_idx = np.argmax(cv_scores)
    best_alpha = alphas[best_idx]

    print(f"\n  最佳 alpha: {best_alpha:.6f}")
    print(f"  对应的交叉验证 AUC: {cv_scores[best_idx]:.4f}")

    # 用最佳 alpha 训练最终模型
    post_pruned = DecisionTreeClassifier(ccp_alpha=best_alpha, random_state=42)
    post_pruned.fit(X_train, y_train)

    train_auc_post = roc_auc_score(y_train, post_pruned.predict_proba(X_train)[:, 1])
    test_auc_post = roc_auc_score(y_test, post_pruned.predict_proba(X_test)[:, 1])

    print(f"\n  训练集 AUC: {train_auc_post:.4f}")
    print(f"  测试集 AUC: {test_auc_post:.4f}")
    print(f"  节点数: {post_pruned.tree_.node_count}")

    # 对比表
    print("\n" + "=" * 60)
    print("剪枝方法对比")
    print("=" * 60)
    print(f"{'方法':<15} | {'节点数':<10} | {'训练 AUC':<12} | {'测试 AUC':<12}")
    print("-" * 60)
    print(f"{'预剪枝':<15} | {pre_pruned.tree_.node_count:<10} | "
          f"{train_auc_pre:<12.4f} | {test_auc_pre:<12.4f}")
    print(f"{'后剪枝 (CCP)':<15} | {post_pruned.tree_.node_count:<10} | "
          f"{train_auc_post:<12.4f} | {test_auc_post:<12.4f}")


def main() -> None:
    """主函数"""
    print("=" * 60)
    print("决策树基础：树结构可视化与过拟合演示")
    print("=" * 60)

    # 1. 加载数据并训练一棵简单的决策树
    X, y = load_titanic_data()
    results = train_and_visualize_tree(X, y, max_depth=3)

    # 2. 演示过拟合
    demonstrate_overfitting()

    # 3. 演示剪枝
    demonstrate_pruning()

    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print("""
    决策树核心要点：
    1. 树结构：根节点 -> 内部节点 -> 叶节点，每个节点问一个特征问题
    2. 分裂标准：基尼不纯度或信息增益，目标是让数据"更纯净"
    3. 可解释性：可以画出树结构，直观展示决策规则
    4. 过拟合风险：树太深会记住训练数据，需要通过剪枝控制
    5. 预剪枝：限制深度、最小样本数等参数
    6. 后剪枝：CCP 剪枝，通过交叉验证选择最佳剪枝强度

    常见错误：
    - 不限制深度，导致严重过拟合
    - 只看训练集性能，忽略泛化能力
    - 忽略树的可解释性优势
    """)


if __name__ == "__main__":
    main()

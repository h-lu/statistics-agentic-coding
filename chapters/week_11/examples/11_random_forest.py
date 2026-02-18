"""
示例：随机森林——Bagging 与特征随机性，群体智慧降低方差

运行方式：python3 chapters/week_11/examples/11_random_forest.py
预期输出：单树 vs 随机森林对比、特征重要性、Bagging 原理演示
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
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

    # 简化：删除缺失值
    X_clean = X.dropna()
    y_clean = y.loc[X_clean.index]

    # 编码分类型变量
    X_clean = X_clean.copy()
    X_clean['sex'] = X_clean['sex'].map({'male': 0, 'female': 1})
    X_clean['embarked'] = X_clean['embarked'].map({'C': 0, 'Q': 1, 'S': 2})
    X_clean = pd.get_dummies(X_clean, columns=['pclass'], drop_first=False)

    print(f"数据集规模: {X_clean.shape[0]} 行, {X_clean.shape[1]} 列")

    return X_clean, y_clean


def compare_single_tree_vs_random_forest() -> dict:
    """
    对比单棵决策树和随机森林

    返回:
    - dict: 包含各模型评估结果的字典
    """
    print("\n" + "=" * 60)
    print("对比：单棵决策树 vs 随机森林")
    print("=" * 60)

    X, y = load_titanic_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    results = {}

    # 1. 单棵决策树
    print("\n【单棵决策树】")
    single_tree = DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42
    )
    single_tree.fit(X_train, y_train)

    train_auc = roc_auc_score(y_train, single_tree.predict_proba(X_train)[:, 1])
    test_auc = roc_auc_score(y_test, single_tree.predict_proba(X_test)[:, 1])

    print(f"  训练集 AUC: {train_auc:.4f}")
    print(f"  测试集 AUC: {test_auc:.4f}")
    print(f"  节点数: {single_tree.tree_.node_count}")

    results['single_tree'] = {
        'model': single_tree,
        'train_auc': train_auc,
        'test_auc': test_auc,
        'n_nodes': single_tree.tree_.node_count
    }

    # 2. Bagging
    print("\n【Bagging（Bootstrap Aggregating）】")
    bagging = BaggingClassifier(
        DecisionTreeClassifier(max_depth=5, min_samples_split=20, min_samples_leaf=10),
        n_estimators=100,
        max_samples=0.8,
        max_features=1.0,  # 使用所有特征
        random_state=42,
        n_jobs=-1
    )
    bagging.fit(X_train, y_train)

    train_auc = roc_auc_score(y_train, bagging.predict_proba(X_train)[:, 1])
    test_auc = roc_auc_score(y_test, bagging.predict_proba(X_test)[:, 1])

    print(f"  训练集 AUC: {train_auc:.4f}")
    print(f"  测试集 AUC: {test_auc:.4f}")
    print(f"  基估计器数量: {len(bagging.estimators_)}")

    results['bagging'] = {
        'model': bagging,
        'train_auc': train_auc,
        'test_auc': test_auc,
        'n_estimators': len(bagging.estimators_)
    }

    # 3. 随机森林
    print("\n【随机森林（Bagging + 特征随机性）】")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        max_features='sqrt',  # 每次分裂只考虑 sqrt(n_features) 个特征
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    train_auc = roc_auc_score(y_train, rf.predict_proba(X_train)[:, 1])
    test_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])

    print(f"  训练集 AUC: {train_auc:.4f}")
    print(f"  测试集 AUC: {test_auc:.4f}")
    print(f"  树的数量: {len(rf.estimators_)}")

    results['random_forest'] = {
        'model': rf,
        'train_auc': train_auc,
        'test_auc': test_auc,
        'n_estimators': len(rf.estimators_),
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test
    }

    # 打印对比表
    print("\n" + "=" * 60)
    print("模型对比")
    print("=" * 60)
    print(f"{'模型':<20} | {'训练 AUC':<12} | {'测试 AUC':<12} | {'泛化差距':<12}")
    print("-" * 70)
    for name, res in [('单棵决策树', results['single_tree']),
                      ('Bagging', results['bagging']),
                      ('随机森林', results['random_forest'])]:
        gap = res['train_auc'] - res['test_auc']
        print(f"{name:<20} | {res['train_auc']:<12.4f} | "
              f"{res['test_auc']:<12.4f} | {gap:<12.4f}")

    return results


def visualize_feature_importance(results: dict) -> None:
    """
    可视化随机森林的特征重要性
    """
    print("\n" + "=" * 60)
    print("特征重要性（随机森林）")
    print("=" * 60)

    rf = results['random_forest']['model']
    X_train = results['random_forest']['X_train']

    # 获取特征重要性
    importances = rf.feature_importances_
    feature_names = X_train.columns

    # 创建 DataFrame 并排序
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    print("\n特征重要性排名:")
    print(f"{'排名':<6} | {'特征':<20} | {'重要性':<12}")
    print("-" * 50)
    for idx, row in importance_df.iterrows():
        rank = importance_df.index.get_loc(idx) + 1
        print(f"{rank:<6} | {row['feature']:<20} | {row['importance']:<12.4f}")

    # 可视化
    font = setup_chinese_font()
    fig, ax = plt.subplots(figsize=(10, 6))

    top_n = 15
    top_features = importance_df.head(top_n)

    bars = ax.barh(range(len(top_features)), top_features['importance'].values, color='steelblue')
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'].values)
    ax.set_xlabel('特征重要性')
    ax.set_title(f'随机森林特征重要性（Top {top_n}）')
    ax.invert_yaxis()

    # 添加数值标签
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2,
                f'{width:.3f}', ha='left', va='center', fontsize=9)

    plt.tight_layout()

    # 保存图片
    output_dir = Path(__file__).parent.parent / 'images'
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'random_forest_feature_importance.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"\n特征重要性图已保存到: {output_dir / 'random_forest_feature_importance.png'}")


def demonstrate_bootstrap_mechanism() -> None:
    """
    演示 Bootstrap 机制：Bagging 的核心
    """
    print("\n" + "=" * 60)
    print("Bootstrap 机制演示")
    print("=" * 60)

    X, y = load_titanic_data()

    n_samples = len(X)
    n_bootstrap = 5

    print(f"\n原始数据集大小: {n_samples}")
    print(f"Bootstrap 样本数: {n_bootstrap}")

    for i in range(n_bootstrap):
        # Bootstrap 采样（有放回）
        np.random.seed(42 + i)
        bootstrap_idx = np.random.choice(n_samples, size=n_samples, replace=True)

        # 计算 OOB（Out-of-Bag）样本
        oob_idx = np.setdiff1d(np.arange(n_samples), bootstrap_idx)

        unique_count = len(np.unique(bootstrap_idx))
        oob_count = len(oob_idx)

        print(f"\nBootstrap 样本 {i+1}:")
        print(f"  唯一样本数: {unique_count} ({unique_count/n_samples*100:.1f}%)")
        print(f"  OOB 样本数: {oob_count} ({oob_count/n_samples*100:.1f}%)")

        print("\n  说明:")
        print("  - Bootstrap 样本中，约 63.2% 的原始样本会至少出现一次")
        print("  - 约 36.8% 的样本不会出现（OOB 样本）")
        print("  - 每个 Bootstrap 样本都略有不同，这降低了模型之间的相关性")


def visualize_bagging_vs_random_forest() -> None:
    """
    可视化 Bagging vs 随机森林的差异
    """
    print("\n" + "=" * 60)
    print("Bagging vs 随机森林：特征随机性的作用")
    print("=" * 60)

    X, y = load_titanic_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    n_estimators_list = [10, 50, 100, 200]
    bagging_scores = []
    rf_scores = []

    for n_est in n_estimators_list:
        # Bagging（只在样本层面随机）
        bagging = BaggingClassifier(
            DecisionTreeClassifier(max_depth=5, min_samples_split=20, min_samples_leaf=10),
            n_estimators=n_est,
            max_samples=0.8,
            max_features=1.0,  # 使用所有特征
            random_state=42,
            n_jobs=-1
        )
        bagging.fit(X_train, y_train)
        bagging_auc = roc_auc_score(y_test, bagging.predict_proba(X_test)[:, 1])
        bagging_scores.append(bagging_auc)

        # 随机森林（样本 + 特征层面都随机）
        rf = RandomForestClassifier(
            n_estimators=n_est,
            max_depth=5,
            max_features='sqrt',
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
        rf_scores.append(rf_auc)

        print(f"n_estimators={n_est:3d}: Bagging AUC={bagging_auc:.4f}, "
              f"随机森林 AUC={rf_auc:.4f}")

    # 可视化
    font = setup_chinese_font()
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(n_estimators_list, bagging_scores, 'o-', label='Bagging', linewidth=2, markersize=8)
    ax.plot(n_estimators_list, rf_scores, 's-', label='随机森林', linewidth=2, markersize=8)

    ax.set_xlabel('树的数量 (n_estimators)')
    ax.set_ylabel('测试集 AUC')
    ax.set_title('Bagging vs 随机森林：特征随机性的作用')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 添加注释
    for i, n in enumerate(n_estimators_list):
        ax.annotate(f"{rf_scores[i]:.4f}",
                   xy=(n, rf_scores[i]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9)

    plt.tight_layout()

    # 保存图片
    output_dir = Path(__file__).parent.parent / 'images'
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'bagging_vs_random_forest.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"\n对比图已保存到: {output_dir / 'bagging_vs_random_forest.png'}")


def main() -> None:
    """主函数"""
    print("=" * 60)
    print("随机森林：Bagging 与特征随机性")
    print("=" * 60)

    # 1. 对比单棵树、Bagging、随机森林
    results = compare_single_tree_vs_random_forest()

    # 2. 可视化特征重要性
    visualize_feature_importance(results)

    # 3. 演示 Bootstrap 机制
    demonstrate_bootstrap_mechanism()

    # 4. 可视化 Bagging vs 随机森林
    visualize_bagging_vs_random_forest()

    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print("""
    随机森林核心要点：
    1. Bagging（Bootstrap Aggregating）：
       - 从原始数据中有放回地抽取多个样本集
       - 在每个样本集上训练一个模型
       - 聚合预测（分类用投票，回归用平均）

    2. 随机森林的额外改进：
       - 在每个节点分裂时，只考虑随机选取的一部分特征
       - 这进一步降低了树之间的相关性
       - 限制特征反而效果更好（反直觉但正确）

    3. 优势：
       - 预测力强：通常比单棵树好很多
       - 不易过拟合：多棵树投票，降低方差
       - 稳定性好：数据的小变化不会导致预测的大变化
       - 自动特征选择：可以输出特征重要性

    4. 代价：
       - 可解释性下降：不能像单棵树那样画出整个森林
       - 训练慢：需要训练多棵树（但可以并行）
       - 预测慢：需要对多棵树求和/投票
       - 模型大：100 棵树可能占很多内存

    与 Week 08 Bootstrap 的连接：
    - Bagging 本质上是 Bootstrap 的应用
    - Bootstrap 原本用于估计统计量的分布
    - Bagging 用于降低模型的方差
    """)


if __name__ == "__main__":
    main()

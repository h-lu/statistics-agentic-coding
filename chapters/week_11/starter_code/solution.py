"""
Week 11 作业参考答案：树模型与基线对比

注意：这是基础层 + 进阶层的参考实现，供学习使用。
请先自己尝试完成作业，遇到困难时再参考此文件。

运行方式：python3 chapters/week_11/starter_code/solution.py
预期输出：决策树和随机森林的评估结果、模型对比表
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report)
import time

# 配置中文字体
def setup_chinese_font() -> str:
    """配置中文字体"""
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


# ============================================================
# 任务 1：训练决策树（基础层）
# ============================================================

def task1_train_decision_tree(X_train, X_test, y_train, y_test):
    """
    任务 1：训练第一棵决策树
    """
    print("=" * 60)
    print("任务 1：训练决策树")
    print("=" * 60)

    # 训练决策树（限制深度为 3）
    tree = DecisionTreeClassifier(
        max_depth=3,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42
    )
    tree.fit(X_train, y_train)

    # 预测
    y_pred = tree.predict(X_test)
    y_prob = tree.predict_proba(X_test)[:, 1]

    # 评估
    print("\n决策树评估指标：")
    print(f"  准确率: {accuracy_score(y_test, y_pred):.4f}")
    print(f"  精确率: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  召回率: {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  F1 分数: {f1_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  AUC: {roc_auc_score(y_test, y_prob):.4f}")

    print("\n分类报告：")
    print(classification_report(y_test, y_pred, target_names=['Not Churn', 'Churn']))

    # 可视化决策树
    font = setup_chinese_font()
    fig, ax = plt.subplots(figsize=(16, 8))

    plot_tree(tree,
              feature_names=X_train.columns,
              class_names=['Not Churn', 'Churn'],
              filled=True,
              rounded=True,
              fontsize=10,
              ax=ax)

    ax.set_title('决策树结构 (max_depth=3)', fontsize=14)

    # 保存图片
    output_dir = Path(__file__).parent.parent / 'output'
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'decision_tree_homework.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"\n决策树图已保存到: {output_dir / 'decision_tree_homework.png'}")

    return tree, y_prob


# ============================================================
# 任务 2：解读决策树（基础层）
# ============================================================

def task2_interpret_tree(tree, feature_names):
    """
    任务 2：解读决策树规则
    """
    print("\n" + "=" * 60)
    print("任务 2：解读决策树")
    print("=" * 60)

    # 获取树结构信息
    n_nodes = tree.tree_.node_count
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right
    feature = tree.tree_.feature
    threshold = tree.tree_.threshold

    print("\n决策树规则解读：")
    print("-" * 60)

    # 根节点信息
    root_feat_idx = feature[0]
    root_threshold = threshold[0]
    root_feat_name = feature_names[root_feat_idx]

    print(f"\n【根节点】")
    print(f"  分裂特征: {root_feat_name}")
    print(f"  阈值: {root_threshold:.2f}")
    print(f"  规则: 如果 {root_feat_name} < {root_threshold:.2f}，走左子树；否则走右子树")

    # 找出样本数最多的叶节点（通常是流失率最高或最低的）
    print(f"\n【叶节点分析】")
    node_samples = tree.tree_.n_node_samples
    node_values = tree.tree_.value

    for i in range(n_nodes):
        if children_left[i] == children_right[i]:  # 叶节点
            churn_ratio = node_values[i][0][1] / node_samples[i]
            print(f"  叶节点 {i}: 样本数={node_samples[i]}, "
                  f"流失率={churn_ratio:.2%}")

    print("\n【可解释性优势】")
    print("  - 决策树可以用 if-else 规则直观展示，业务方不需要懂统计学就能理解")
    print("  - 每个叶节点代表一类相似客户，流失率就是这类客户的历史比例")
    print("  - 逻辑回归的系数需要解释'对数几率'的变化，不如决策树直观")


# ============================================================
# 任务 3：训练随机森林（进阶层）
# ============================================================

def task3_train_random_forest(X_train, X_test, y_train, y_test, tree_auc):
    """
    任务 3：训练随机森林并对比
    """
    print("\n" + "=" * 60)
    print("任务 3：训练随机森林")
    print("=" * 60)

    # 训练随机森林
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )

    start_time = time.time()
    rf.fit(X_train, y_train)
    train_time = time.time() - start_time

    # 预测
    y_pred_rf = rf.predict(X_test)
    y_prob_rf = rf.predict_proba(X_test)[:, 1]

    # 评估
    print("\n随机森林评估指标：")
    print(f"  准确率: {accuracy_score(y_test, y_pred_rf):.4f}")
    print(f"  精确率: {precision_score(y_test, y_pred_rf, zero_division=0):.4f}")
    print(f"  召回率: {recall_score(y_test, y_pred_rf, zero_division=0):.4f}")
    print(f"  F1 分数: {f1_score(y_test, y_pred_rf, zero_division=0):.4f}")
    print(f"  AUC: {roc_auc_score(y_test, y_prob_rf):.4f}")
    print(f"  训练时间: {train_time:.4f} 秒")

    # 对比
    rf_auc = roc_auc_score(y_test, y_prob_rf)
    improvement = rf_auc - tree_auc
    improvement_pct = (improvement / tree_auc) * 100

    print(f"\n【随机森林 vs 决策树】")
    print(f"  决策树 AUC: {tree_auc:.4f}")
    print(f"  随机森林 AUC: {rf_auc:.4f}")
    print(f"  提升量: +{improvement:.4f} (+{improvement_pct:.1f}%)")

    print("\n【可解释性差异】")
    print("  - 决策树：可以画出完整的树结构，直观展示每条决策路径")
    print("  - 随机森林：100 棵树无法全部可视化，只能通过特征重要性间接解释")
    print("  - 随机森林的代价是可解释性下降，但预测力提升、稳定性更好")

    return rf, y_prob_rf


# ============================================================
# 任务 4：特征重要性（进阶层）
# ============================================================

def task4_feature_importance(rf, feature_names):
    """
    任务 4：提取并可视化特征重要性
    """
    print("\n" + "=" * 60)
    print("任务 4：特征重要性")
    print("=" * 60)

    # 获取特征重要性
    importances = rf.feature_importances_

    # 创建 DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).reset_index(drop=True)

    print("\n特征重要性排名：")
    print(importiance_df)

    print("\n【业务解释】")
    for i, row in importance_df.head(3).iterrows():
        print(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")

    print("\n  最重要特征是'距上次购买天数'，这与业务直觉一致：")
    print("  很久没购买的客户更可能流失。")

    # 可视化
    font = setup_chinese_font()
    fig, ax = plt.subplots(figsize=(10, 6))

    top_features = importance_df
    bars = ax.barh(range(len(top_features)), top_features['importance'].values, color='steelblue')
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'].values)
    ax.set_xlabel('特征重要性')
    ax.set_title('随机森林特征重要性')
    ax.invert_yaxis()

    # 添加数值标签
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2,
                f'{width:.3f}', ha='left', va='center', fontsize=9)

    plt.tight_layout()

    # 保存图片
    output_dir = Path(__file__).parent.parent / 'output'
    plt.savefig(output_dir / 'feature_importance_homework.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"\n特征重要性图已保存到: {output_dir / 'feature_importance_homework.png'}")


# ============================================================
# 任务 5：基线对比（挑战层）
# ============================================================

def task5_baseline_comparison(X_train, X_test, y_train, y_test):
    """
    任务 5：基线对比框架
    """
    print("\n" + "=" * 60)
    print("任务 5：基线对比")
    print("=" * 60)

    # 定义模型
    models = {
        'dummy': DummyClassifier(strategy='most_frequent'),
        'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
        'decision_tree': DecisionTreeClassifier(
            max_depth=3,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        )
    }

    # 训练并对比
    results = {}
    for name, model in models.items():
        start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start

        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)

        results[name] = {'auc': auc, 'train_time': train_time}

    # 打印对比表
    print("\n模型对比表：")
    print("| 模型 | AUC | 训练时间 |")
    print("|------|-----|---------|")

    name_display = {
        'dummy': '傻瓜基线',
        'logistic_regression': '逻辑回归',
        'decision_tree': '决策树',
        'random_forest': '随机森林'
    }

    for name in ['dummy', 'logistic_regression', 'decision_tree', 'random_forest']:
        res = results[name]
        print(f"| {name_display[name]} | {res['auc']:.4f} | {res['train_time']:.4f}s |")

    # 分析
    dummy_auc = results['dummy']['auc']
    lr_auc = results['logistic_regression']['auc']
    rf_auc = results['random_forest']['auc']
    lr_time = results['logistic_regression']['train_time']
    rf_time = results['random_forest']['train_time']

    improvement = rf_auc - lr_auc
    improvement_pct = (improvement / lr_auc) * 100
    time_ratio = rf_time / lr_time

    print("\n【分析】")
    print(f"  傻瓜基线 AUC: {dummy_auc:.4f}（随机猜测水平）")
    print(f"  逻辑回归 AUC: {lr_auc:.4f}（比基线提升 {(lr_auc - dummy_auc):.4f}）")
    print(f"  随机森林 AUC: {rf_auc:.4f}（比逻辑回归提升 {improvement:.4f}，即 {improvement_pct:.1f}%）")
    print(f"  训练时间对比: 随机森林比逻辑回归慢 {time_ratio:.1f} 倍")

    print("\n【模型选择建议】")
    if improvement_pct < 2:
        print(f"  - 随机森林只比逻辑回归提升 {improvement_pct:.1f}%，提升量较小")
        print(f"  - 但训练时间慢了 {time_ratio:.1f} 倍")
        print(f"  - 建议：如果需要向业务方解释规则，选逻辑回归；如果追求最高预测力，选随机森林")
    elif improvement_pct < 5:
        print(f"  - 随机森林比逻辑回归提升 {improvement_pct:.1f}%，提升量中等")
        print(f"  - 建议根据业务场景选择：预测力 vs 可解释性")
    else:
        print(f"  - 随机森林比逻辑回归提升 {improvement_pct:.1f}%，提升量显著")
        print(f"  - 建议选择随机森林")

    return results


# ============================================================
# 主函数
# ============================================================

def generate_sample_data():
    """
    生成示例数据（用于演示）
    实际作业中应该使用 data/customer_churn.csv
    """
    np.random.seed(42)
    n_samples = 1000

    # 生成特征
    purchase_count = np.random.poisson(5, n_samples)
    avg_spend = np.random.gamma(10, 10, n_samples)
    days_since_last_purchase = np.random.exponential(30, n_samples)
    membership_days = np.random.randint(30, 365, n_samples)

    # 生成目标变量（流失率与特征相关）
    logit = -3 + 0.1 * purchase_count - 0.02 * avg_spend + 0.05 * days_since_last_purchase - 0.005 * membership_days
    prob = 1 / (1 + np.exp(-logit))
    is_churned = (np.random.random(n_samples) < prob).astype(int)

    # 创建 DataFrame
    df = pd.DataFrame({
        'purchase_count': purchase_count,
        'avg_spend': avg_spend,
        'days_since_last_purchase': days_since_last_purchase,
        'membership_days': membership_days,
        'is_churned': is_churned
    })

    return df


def main():
    """主函数"""
    print("=" * 60)
    print("Week 11 作业参考答案")
    print("=" * 60)

    # 生成示例数据（实际作业中应该加载真实数据）
    print("\n生成示例数据...")
    df = generate_sample_data()

    # 准备特征和目标
    X = df[['purchase_count', 'avg_spend', 'days_since_last_purchase', 'membership_days']]
    y = df['is_churned']

    # 划分数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"数据集规模: {X.shape[0]} 行")
    print(f"训练集: {X_train.shape[0]} 行, 测试集: {X_test.shape[0]} 行")
    print(f"正类占比: {y.mean():.2%}")

    # 任务 1：训练决策树
    tree, tree_prob = task1_train_decision_tree(X_train, X_test, y_train, y_test)
    tree_auc = roc_auc_score(y_test, tree_prob)

    # 任务 2：解读决策树
    task2_interpret_tree(tree, X.columns)

    # 任务 3：训练随机森林
    rf, rf_prob = task3_train_random_forest(X_train, X_test, y_train, y_test, tree_auc)

    # 任务 4：特征重要性
    task4_feature_importance(rf, X.columns)

    # 任务 5：基线对比
    task5_baseline_comparison(X_train, X_test, y_train, y_test)

    print("\n" + "=" * 60)
    print("所有任务完成！")
    print("=" * 60)
    print("\n总结：")
    print("  - 决策树提供了可解释的决策规则")
    print("  - 随机森林通过'群体智慧'提升了预测力")
    print("  - 基线对比帮助我们评估'更复杂的模型是否值得'")
    print("  - 模型选择需要权衡：提升量 vs 复杂度 vs 可解释性")


# ============================================================
# 导出函数（供测试使用）
# ============================================================

def train_decision_tree(X_train, y_train, max_depth=3, min_samples_split=20,
                       min_samples_leaf=10, max_leaf_nodes=None, random_state=42):
    """
    Train decision tree classifier (训练决策树分类器)

    Parameters control overfitting through pruning and depth limits.
    Use max_depth to prevent overfitting on small datasets.

    参数:
    ------
    X_train : array-like
        训练特征
    y_train : array-like
        训练标签
    max_depth : int
        树的最大深度，用于控制过拟合
    min_samples_split : int
        分裂内部节点所需的最小样本数
    min_samples_leaf : int
        叶节点的最小样本数
    max_leaf_nodes : int
        最大叶节点数
    random_state : int
        随机种子

    返回:
    ------
    tree : DecisionTreeClassifier
        训练好的决策树模型
    """
    # 输入验证
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)

    if len(X_train) < 2:
        raise ValueError(f"Need at least 2 samples to train, got {len(X_train)}")

    if len(np.unique(y_train)) < 2:
        raise ValueError(f"Need at least 2 classes to train, got {len(np.unique(y_train))}")

    if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
        raise ValueError("X_train contains NaN or Inf values")

    tree = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_leaf_nodes=max_leaf_nodes,
        random_state=random_state
    )
    tree.fit(X_train, y_train)
    return tree


def fit_decision_tree(X_train, y_train, **kwargs):
    """fit_decision_tree 的别名"""
    return train_decision_tree(X_train, y_train, **kwargs)


def predict_tree(tree, X):
    """使用决策树预测"""
    return tree.predict(X), tree.predict_proba(X)[:, 1]


def train_random_forest(X_train, y_train, n_estimators=100, max_depth=5, random_state=42):
    """
    Train random forest classifier (训练随机森林分类器)

    An ensemble of decision trees that provides better prediction accuracy
    and stability than single trees, with reduced overfitting risk.
    """
    # 输入验证
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)

    if len(X_train) < 2:
        raise ValueError(f"Need at least 2 samples to train, got {len(X_train)}")

    if len(np.unique(y_train)) < 2:
        raise ValueError(f"Need at least 2 classes to train, got {len(np.unique(y_train))}")

    if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
        raise ValueError("X_train contains NaN or Inf values")

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=random_state,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    return rf


def fit_random_forest(X_train, y_train, **kwargs):
    """fit_random_forest 的别名"""
    return train_random_forest(X_train, y_train, **kwargs)


def predict_random_forest(rf, X):
    """使用随机森林预测"""
    return rf.predict(X), rf.predict_proba(X)[:, 1]


def get_feature_importance(model, feature_names=None):
    """
    获取特征重要性

    参数:
    ------
    model : 决策树或随机森林模型
    feature_names : list, optional
        特征名称列表。如果为 None，返回重要性数组

    返回:
    ------
    如果提供 feature_names，返回 DataFrame
    否则返回 numpy 数组
    """
    importances = model.feature_importances_

    if feature_names is None:
        return importances

    return pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)


def plot_feature_importance(model, feature_names, output_path=None):
    """绘制特征重要性图"""
    importance_df = get_feature_importance(model, feature_names)

    setup_chinese_font()
    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.barh(range(len(importance_df)), importance_df['importance'].values, color='steelblue')
    ax.set_yticks(range(len(importance_df)))
    ax.set_yticklabels(importance_df['feature'].values)
    ax.set_xlabel('特征重要性')
    ax.set_title('特征重要性')
    ax.invert_yaxis()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return importance_df


def train_dummy_baseline(X_train, y_train, strategy='most_frequent'):
    """
    Train dummy baseline classifier (训练傻瓜基线模型)

    A baseline model that predicts using simple rules (most frequent class).
    Provides a lower bound for model performance comparison.
    """
    dummy = DummyClassifier(strategy=strategy)
    dummy.fit(X_train, y_train)
    return dummy


def train_logistic_baseline(X_train, y_train, random_state=42):
    """训练逻辑回归基线模型"""
    lr = LogisticRegression(max_iter=1000, random_state=random_state)
    lr.fit(X_train, y_train)
    return lr


def compare_with_baselines(*args, test_size=0.3, random_state=42):
    """
    Baseline comparison: train multiple models and compare (基线对比)

    Compares dummy baseline, logistic regression, decision tree, and random forest.
    Supports two calling modes:
    1. compare_with_baselines(X, y) - splits data internally
    2. compare_with_baselines(X_train, y_train, X_test, y_test) - uses pre-split data

    支持两种调用方式:
    1. compare_with_baselines(X, y) - 内部划分数据集
    2. compare_with_baselines(X_train, y_train, X_test, y_test) - 使用已划分的数据
    """
    if len(args) == 2:
        # (X, y) - 内部划分数据集
        X, y = args
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    elif len(args) == 4:
        # (X_train, y_train, X_test, y_test)
        X_train, y_train, X_test, y_test = args
    else:
        raise TypeError(f"compare_with_baselines() takes 2 or 4 arguments ({len(args)} given)")

    models = {
        'dummy': train_dummy_baseline(X_train, y_train),
        'logistic_regression': train_logistic_baseline(X_train, y_train),
        'decision_tree': train_decision_tree(X_train, y_train),
        'random_forest': train_random_forest(X_train, y_train)
    }

    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        results[name] = {
            'model': model,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc': roc_auc_score(y_test, y_prob)
        }

    return results


def baseline_comparison(X, y, test_size=0.3, random_state=42):
    """
    Complete baseline comparison pipeline (完整的基线对比流程)

    Splits data and compares multiple models: dummy baseline,
    logistic regression, decision tree, and random forest.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return compare_with_baselines(X_train, y_train, X_test, y_test)


def detect_overfitting(*args):
    """
    检测过拟合

    支持两种调用方式:
    1. detect_overfitting(model, X_train, X_test, y_train, y_test)
    2. detect_overfitting(X_train, y_train, X_test, y_test) - 内部训练默认模型
    """
    from sklearn.metrics import accuracy_score

    if len(args) == 5:
        # (model, X_train, X_test, y_train, y_test)
        model, X_train, X_test, y_train, y_test = args
    elif len(args) == 4:
        # (X_train, y_train, X_test, y_test) - 训练默认模型
        X_train, y_train, X_test, y_test = args
        model = train_decision_tree(X_train, y_train, random_state=42)
    else:
        raise TypeError(f"detect_overfitting() takes 4 or 5 arguments ({len(args)} given)")

    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    gap = train_acc - test_acc

    return {
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'gap': gap,
        'is_overfitting': gap > 0.1  # 如果差距超过 10%，认为过拟合
    }


def check_overfitting(*args):
    """check_overfitting 的别名"""
    return detect_overfitting(*args)


def tree_models_comparison(X, y, test_size=0.3, random_state=42):
    """树模型对比（用于 StatLab）"""
    return baseline_comparison(X, y, test_size, random_state)


def format_model_comparison_report(results):
    """格式化模型对比报告"""
    lines = ["## 模型对比与选择\n"]
    lines.append("| 模型 | AUC |")
    lines.append("|------|-----|")

    name_map = {
        'dummy': 'Dummy (Baseline)',
        'logistic_regression': 'Logistic Regression',
        'decision_tree': 'Decision Tree',
        'random_forest': 'Random Forest'
    }

    for name, res in results.items():
        display_name = name_map.get(name, name)
        lines.append(f"| {display_name} | {res['auc']:.4f} |")

    return "\n".join(lines)


def train_single_feature_tree(X_train, y_train, feature_idx=0, max_depth=2, random_state=42):
    """
    训练单特征决策树基线

    使用单个最重要的特征训练简单的决策树，作为最简单的树基线。
    这有助于展示"即使只用一个特征，树模型也比随机猜测好"。
    """
    # 选择单个特征
    X_single = X_train[:, [feature_idx]] if X_train.ndim > 1 else X_train.reshape(-1, 1)

    tree = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=random_state
    )
    tree.fit(X_single, y_train)
    return tree


def calculate_improvement(model_metric, baseline_metric):
    """
    计算相对提升量

    返回 (新模型 - 基线) / 基线
    如果基线为 0，返回绝对差值
    """
    if baseline_metric == 0:
        return model_metric
    return (model_metric - baseline_metric) / baseline_metric


def select_best_model(X_train, y_train, X_test, y_test, metric='auc'):
    """
    选择最佳模型

    在多个候选模型中选择性能最佳的模型。
    默认使用 AUC 作为评估指标。
    """
    results = compare_with_baselines(X_train, y_train, X_test, y_test)

    best_model_name = None
    best_score = -float('inf')

    for name, res in results.items():
        score = res.get(metric, res.get('auc', 0))
        if score > best_score:
            best_score = score
            best_model_name = name

    return {
        'model_name': best_model_name,
        'model': results[best_model_name]['model'],
        'score': best_score,
        'all_results': results
    }


def format_model_selection_reasoning(results):
    """
    格式化模型选择理由

    生成可读的模型选择理由说明，包括：
    - 各模型性能对比
    - 提升量分析
    - 模型选择建议
    """
    lines = ["## 模型选择理由\n"]

    # 提取各模型 AUC
    dummy_auc = results.get('dummy', {}).get('auc', results.get('dummy_auc', 0.5))
    logistic_auc = results.get('logistic_regression', {}).get('auc', results.get('logistic_auc', 0.7))
    tree_auc = results.get('decision_tree', {}).get('auc', results.get('tree_auc', 0.75))
    forest_auc = results.get('random_forest', {}).get('auc', results.get('forest_auc', 0.8))

    lines.append(f"### 性能对比")
    lines.append(f"- **Dummy (Baseline)**: AUC = {dummy_auc:.4f} - 随机猜测水平")
    lines.append(f"- **Logistic Regression**: AUC = {logistic_auc:.4f} - 相对基线提升 +{calculate_improvement(logistic_auc, dummy_auc)*100:.1f}%")
    lines.append(f"- **Decision Tree**: AUC = {tree_auc:.4f} - 相对基线提升 +{calculate_improvement(tree_auc, dummy_auc)*100:.1f}%")
    lines.append(f"- **Random Forest**: AUC = {forest_auc:.4f} - 相对基线提升 +{calculate_improvement(forest_auc, dummy_auc)*100:.1f}%")

    lines.append(f"\n### 选择建议")

    # 分析提升量
    rf_vs_lr_improvement = calculate_improvement(forest_auc, logistic_auc)

    if rf_vs_lr_improvement < 0.02:
        lines.append(f"- 随机森林相对逻辑回归的提升较小 ({rf_vs_lr_improvement*100:.1f}%)")
        lines.append(f"- 建议根据场景选择：需要可解释性选逻辑回归，追求最高预测力选随机森林")
    elif rf_vs_lr_improvement < 0.05:
        lines.append(f"- 随机森林相对逻辑回归有中等提升 ({rf_vs_lr_improvement*100:.1f}%)")
        lines.append(f"- 建议权衡预测力与可解释性")
    else:
        lines.append(f"- 随机森林显著优于逻辑回归 ({rf_vs_lr_improvement*100:.1f}% 提升)")
        lines.append(f"- 建议选择随机森林")

    return "\n".join(lines)


if __name__ == "__main__":
    main()

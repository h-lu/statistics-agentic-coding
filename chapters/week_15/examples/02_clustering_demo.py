"""
示例：K-means 聚类——肘部法则 + 轮廓系数

运行方式：python3 chapters/week_15/examples/02_clustering_demo.py
预期输出：最优 K 值选择、聚类结果、簇特征分析
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from pathlib import Path


def generate_synthetic_clusters(n_samples: int = 1000, n_centers: int = 5,
                                n_features: int = 10, random_seed: int = 42) -> tuple:
    """
    生成合成聚类数据

    参数:
        n_samples: 总样本数
        n_centers: 真实簇数
        n_features: 特征维度
        random_seed: 随机种子

    返回:
        X: 特征矩阵
        y_true: 真实标签（用于评估，实际聚类时不可用）
    """
    np.random.seed(random_seed)
    X, y_true = make_blobs(
        n_samples=n_samples,
        centers=n_centers,
        n_features=n_features,
        cluster_std=1.5,
        random_state=random_seed
    )
    return X, y_true


def find_optimal_k(X: np.ndarray, k_range: range = range(2, 11),
                   plot_dir: Path = None) -> dict:
    """
    使用肘部法则和轮廓系数选择最优 K 值

    参数:
        X: 特征矩阵
        k_range: 尝试的 K 值范围
        plot_dir: 图片保存目录

    返回:
        包含 WCSS、轮廓分数和建议的字典
    """
    wcss = []  # Within-Cluster Sum of Squares
    silhouette_scores = []

    for k in k_range:
        # K-means 聚类
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        # 记录 WCSS（肘部法则）
        wcss.append(kmeans.inertia_)

        # 记录轮廓系数（需要至少 2 个簇）
        if k > 1 and k < X.shape[0]:
            score = silhouette_score(X, labels)
            silhouette_scores.append(score)

    # 找出最优 K 值
    # 方法1：轮廓系数最大
    k_silhouette = np.argmax(silhouette_scores) + min(k_range)

    # 方法2：肘部法则（WCSS 下降速度变缓的点）
    # 计算二阶差分，找拐点
    wcss_diff = np.diff(wcss)
    wcss_diff2 = np.diff(wcss_diff)
    k_elbow = np.argmin(wcss_diff2) + min(k_range) + 1

    # 可视化
    if plot_dir is not None:
        _plot_k_selection(k_range, wcss, silhouette_scores,
                         k_silhouette, k_elbow, plot_dir)

    return {
        'wcss': wcss,
        'silhouette_scores': silhouette_scores,
        'k_silhouette': k_silhouette,
        'k_elbow': k_elbow,
        'k_range': list(k_range)
    }


def _plot_k_selection(k_range, wcss, silhouette_scores,
                     k_silhouette, k_elbow, plot_dir):
    """绘制 K 值选择图表"""
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 肘部法则
    ax1.plot(list(k_range), wcss, 'bo-', linewidth=2, markersize=8)
    ax1.axvline(x=k_elbow, color='red', linestyle='--',
                label=f'肘部 K={k_elbow}', linewidth=2)
    ax1.set_xlabel('簇的数量 K')
    ax1.set_ylabel('簇内平方和（WCSS）')
    ax1.set_title('肘部法则：选择最优 K 值')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 轮廓系数
    ax2.plot(list(k_range)[1:], silhouette_scores, 'gs-',
             linewidth=2, markersize=8)
    ax2.axvline(x=k_silhouette, color='red', linestyle='--',
                label=f'最大轮廓系数 K={k_silhouette}', linewidth=2)
    ax2.set_xlabel('簇的数量 K')
    ax2.set_ylabel('轮廓系数')
    ax2.set_title('轮廓系数：选择最优 K 值')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_dir / 'kmeans_k_selection.png', dpi=150)
    print(f"✅ 图表已保存: {plot_dir / 'kmeans_k_selection.png'}")


def perform_kmeans_clustering(X: np.ndarray, n_clusters: int,
                              plot_dir: Path = None) -> tuple:
    """
    执行 K-means 聚类

    参数:
        X: 特征矩阵
        n_clusters: 簇数
        plot_dir: 图片保存目录

    返回:
        labels: 聚类标签
        kmeans: KMeans 模型
        silhouette_avg: 轮廓系数
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    # 计算轮廓系数
    silhouette_avg = silhouette_score(X, labels)

    # 可视化（如果是 2D 数据）
    if plot_dir is not None and X.shape[1] >= 2:
        _plot_clustering_result(X, labels, kmeans, plot_dir)

    return labels, kmeans, silhouette_avg


def _plot_clustering_result(X, labels, kmeans, plot_dir):
    """绘制聚类结果"""
    plot_dir = Path(plot_dir)

    fig, ax = plt.subplots(figsize=(10, 8))

    # 使用前 2 个特征绘制
    scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis',
                       alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

    # 绘制簇中心
    centers = kmeans.cluster_centers_
    ax.scatter(centers[:, 0], centers[:, 1], c='red', marker='X',
               s=200, linewidths=3, edgecolors='black', label='簇中心')

    ax.set_xlabel('特征 1')
    ax.set_ylabel('特征 2')
    ax.set_title(f'K-means 聚类结果 (K={kmeans.n_clusters})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 添加颜色条
    plt.colorbar(scatter, ax=ax, label='簇编号')

    plt.tight_layout()
    plt.savefig(plot_dir / 'kmeans_clustering_result.png', dpi=150)
    print(f"✅ 图表已保存: {plot_dir / 'kmeans_clustering_result.png'}")


def analyze_cluster_characteristics(X: np.ndarray, labels: np.ndarray,
                                    feature_names: list = None) -> pd.DataFrame:
    """
    分析各簇的特征均值

    参数:
        X: 特征矩阵
        labels: 聚类标签
        feature_names: 特征名称列表

    返回:
        各簇特征摘要 DataFrame
    """
    if feature_names is None:
        feature_names = [f'特征_{i}' for i in range(X.shape[1])]

    df = pd.DataFrame(X, columns=feature_names)
    df['cluster'] = labels

    # 计算各簇的统计摘要
    cluster_summary = df.groupby('cluster').agg(['mean', 'std'])
    cluster_counts = df['cluster'].value_counts().sort_index()

    return cluster_summary, cluster_counts


def main() -> None:
    print("=" * 60)
    print("K-means 聚类示例")
    print("=" * 60)

    # 生成合成数据
    print("\n生成合成聚类数据...")
    X, y_true = generate_synthetic_clusters(
        n_samples=1000, n_centers=5, n_features=10, random_seed=42
    )
    print(f"数据形状: {X.shape}")
    print(f"真实簇数: 5")

    # 选择最优 K 值
    print("\n" + "=" * 60)
    print("选择最优 K 值（肘部法则 + 轮廓系数）")
    print("=" * 60)

    report_dir = Path("chapters/week_15/report")
    k_results = find_optimal_k(X, k_range=range(2, 11), plot_dir=report_dir)

    print(f"\n肘部法则建议: K = {k_results['k_elbow']}")
    print(f"轮廓系数建议: K = {k_results['k_silhouette']}")

    # 选择最终 K 值（这里选择轮廓系数建议的值）
    k_final = k_results['k_silhouette']
    print(f"\n最终选择: K = {k_final}")

    # 执行聚类
    print("\n" + "=" * 60)
    print(f"执行 K-means 聚类 (K={k_final})")
    print("=" * 60)

    labels, kmeans, silhouette_avg = perform_kmeans_clustering(
        X, n_clusters=k_final, plot_dir=report_dir
    )

    print(f"轮廓系数: {silhouette_avg:.3f}")
    print(f"各簇样本数: {np.bincount(labels)}")

    # 分析簇特征
    print("\n" + "=" * 60)
    print("各簇特征摘要")
    print("=" * 60)

    cluster_summary, cluster_counts = analyze_cluster_characteristics(X, labels)
    print("\n各簇样本数:")
    print(cluster_counts)

    print("\n各簇特征均值（前 5 个特征）:")
    print(cluster_summary.iloc[:, :5])

    # 坏例子：K 值选择不当
    print("\n" + "=" * 60)
    print("坏例子：K 值选择不当（K=2）")
    print("=" * 60)

    kmeans_bad = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels_bad = kmeans_bad.fit_predict(X)
    silhouette_bad = silhouette_score(X, labels_bad)

    print(f"K=2 时的轮廓系数: {silhouette_bad:.3f}")
    print(f"K={k_final} 时的轮廓系数: {silhouette_avg:.3f}")
    print("\n问题：K 值太小会把不同的簇强行合并，导致轮廓系数下降")

    # 坏例子：K 值太大
    print("\n" + "=" * 60)
    print("坏例子：K 值太大（K=15）")
    print("=" * 60)

    kmeans_bad2 = KMeans(n_clusters=15, random_state=42, n_init=10)
    labels_bad2 = kmeans_bad2.fit_predict(X)
    silhouette_bad2 = silhouette_score(X, labels_bad2)

    print(f"K=15 时的轮廓系数: {silhouette_bad2:.3f}")
    print(f"K={k_final} 时的轮廓系数: {silhouette_avg:.3f}")
    print("\n问题：K 值太大会导致过拟合，把一个簇拆成多个")

    print("\n" + "=" * 60)
    print("聚类分析完成")
    print("=" * 60)
    print("\n关键结论:")
    print("1. 肘部法则和轮廓系数是选择 K 值的常用方法")
    print("2. 轮廓系数接近 1 表示聚类效果好，接近 0 表示边界模糊")
    print("3. K 值太小会强行合并，太大会过拟合")
    print("4. 最终 K 值应结合业务解释性综合考虑")


if __name__ == "__main__":
    main()

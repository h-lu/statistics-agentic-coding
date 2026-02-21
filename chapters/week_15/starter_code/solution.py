"""
Week 15 作业参考实现

本文件提供基础作业任务的参考实现，供学生在遇到困难时参考。
请先尝试自己完成，再看这个解决方案。

参考实现涵盖：
- 任务 1：维度灾难计算验证
- 任务 2：PCA 降维分析与可视化
- 任务 3：K-means 聚类分析与评估

运行方式：python3 chapters/week_15/starter_code/solution.py

注意：这只是基础参考实现，不包含进阶/挑战部分。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ===== 图表中文字体配置 =====
def setup_chinese_font() -> str:
    """配置中文字体，返回使用的字体名称"""
    import matplotlib.font_manager as fm
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
# 任务 1：维度灾难——高维数据的陷阱
# ============================================================

def task1_curse_of_dimensionality():
    """
    任务 1：计算不同维度下，随机点之间的平均距离

    目的：验证维度灾难现象——高维空间中距离趋同
    """
    print("=" * 60)
    print("任务 1：维度灾难验证")
    print("=" * 60)

    def average_distance(dim: int, n_samples: int = 1000) -> float:
        """计算给定维度下，随机点之间的平均距离"""
        from scipy.spatial.distance import pdist
        points = np.random.rand(n_samples, dim)
        distances = pdist(points)
        return np.mean(distances)

    # 测试不同维度
    dimensions = [1, 2, 5, 10, 20, 50, 100]
    avg_distances = [average_distance(d) for d in dimensions]

    print("\n不同维度下的平均距离：")
    for d, dist in zip(dimensions, avg_distances):
        print(f"  {d:3d} 维: {dist:.4f}")

    # 计算变异系数
    def distance_cv(dim: int, n_samples: int = 1000) -> float:
        """计算距离的变异系数（标准差/均值）

        变异系数 = 标准差 / 均值
        - 当所有距离几乎相等时，标准差相对于均值很小，CV 趋于 0
        - 维度越高，距离差异越小，CV 越小
        - 低维空间：CV 大（距离有区分度）
        - 高维空间：CV 小（距离趋同，度量失效）
        """
        from scipy.spatial.distance import pdist
        points = np.random.rand(n_samples, dim)
        distances = pdist(points)
        return distances.std() / distances.mean()

    cvs = [distance_cv(d) for d in dimensions]

    print("\n距离变异系数（越小表示距离越趋同）：")
    for d, cv in zip(dimensions, cvs):
        print(f"  {d:3d} 维: {cv:.4f}")

    print("\n结论：")
    print("  - 维度增加 → 平均距离增大（数据变得更稀疏）")
    print("  - 维度增加 → 变异系数减小（距离趋同，最近邻不再'近'）")

    return dimensions, avg_distances, cvs


# ============================================================
# 任务 2：PCA 降维——从 50 维到 2 维
# ============================================================

def task2_pca_analysis():
    """
    任务 2：PCA 降维分析

    目的：
    - 标准化数据
    - 查看累积解释方差
    - 选择主成分数量
    - 解释主成分载荷
    """
    print("\n" + "=" * 60)
    print("任务 2：PCA 降维分析")
    print("=" * 60)

    # 生成模拟数据
    print("\n生成模拟电商数据（50 个特征）...")
    X, y = make_classification(
        n_samples=500,
        n_features=50,
        n_informative=5,
        n_redundant=20,
        random_state=42
    )

    feature_names = [f'feature_{i}' for i in range(50)]

    # 步骤 1：标准化（PCA 对尺度敏感！）
    print("\n步骤 1：标准化数据")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"  原始范围：[{X.min():.2f}, {X.max():.2f}]")
    print(f"  标准化后：均值≈0，标准差≈1")

    # 步骤 2：PCA（保留所有主成分）
    print("\n步骤 2：拟合 PCA 模型")
    pca = PCA(random_state=42)
    pca.fit(X_scaled)

    # 步骤 3：查看累积解释方差
    print("\n步骤 3：累积方差解释")
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)

    print(f"  前 3 个主成分解释：{cumulative_var[2]:.2%}")
    print(f"  前 10 个主成分解释：{cumulative_var[9]:.2%}")

    # 找到解释 80% 方差需要的主成分数
    n_80 = np.argmax(cumulative_var >= 0.8) + 1
    n_90 = np.argmax(cumulative_var >= 0.9) + 1
    print(f"  解释 80% 方差需要：{n_80} 个主成分")
    print(f"  解释 90% 方差需要：{n_90} 个主成分")

    # 步骤 4：查看主成分载荷
    print("\n步骤 4：主成分载荷分析")
    print("\n第一主成分载荷（前 10 个特征）：")

    loadings_pc1 = pd.DataFrame({
        'feature': feature_names,
        'loading': pca.components_[0]
    }).sort_values('loading', key=abs, ascending=False)

    print(loadings_pc1.head(10).to_string(index=False))

    print("\n第二主成分载荷（前 10 个特征）：")
    loadings_pc2 = pd.DataFrame({
        'feature': feature_names,
        'loading': pca.components_[1]
    }).sort_values('loading', key=abs, ascending=False)

    print(loadings_pc2.head(10).to_string(index=False))

    # 步骤 5：降维到 2D（用于可视化）
    print("\n步骤 5：降维到 2D")
    pca_2d = PCA(n_components=2, random_state=42)
    X_2d = pca_2d.fit_transform(X_scaled)

    print(f"  PC1 解释方差：{pca_2d.explained_variance_ratio_[0]:.2%}")
    print(f"  PC2 解释方差：{pca_2d.explained_variance_ratio_[1]:.2%}")

    # 可视化
    setup_chinese_font()
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)

    # 累积方差图
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 51), cumulative_var * 100, 'o-', linewidth=2)
    plt.axhline(y=80, color='r', linestyle='--', label='80% 阈值')
    plt.axhline(y=90, color='g', linestyle='--', label='90% 阈值')
    plt.xlabel('主成分数量')
    plt.ylabel('累积解释方差比例 (%)')
    plt.title('PCA：累积方差解释')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'task2_cumulative_variance.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n图片已保存：{output_dir / 'task2_cumulative_variance.png'}")

    # 2D 散点图
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='viridis', alpha=0.6)
    plt.xlabel(f'PC1（{pca_2d.explained_variance_ratio_[0]:.1%} 方差）')
    plt.ylabel(f'PC2（{pca_2d.explained_variance_ratio_[1]:.1%} 方差）')
    plt.title('PCA 降维：50 维 → 2 维')
    plt.colorbar(scatter, label='类别')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'task2_pca_2d.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"图片已保存：{output_dir / 'task2_pca_2d.png'}")

    return pca, X_scaled, X_2d, cumulative_var


# ============================================================
# 任务 3：K-means 聚类——发现隐藏分组
# ============================================================

def task3_kmeans_clustering(X_scaled):
    """
    任务 3：K-means 聚类分析

    目的：
    - 尝试不同 K 值
    - 用轮廓系数选择最优 K
    - 解释聚类结果
    """
    print("\n" + "=" * 60)
    print("任务 3：K-means 聚类分析")
    print("=" * 60)

    # 步骤 1：尝试不同 K 值
    print("\n步骤 1：尝试不同 K 值")
    K_range = range(2, 11)
    silhouette_scores = []
    inertias = []

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)

        sil_score = silhouette_score(X_scaled, labels)
        silhouette_scores.append(sil_score)
        inertias.append(kmeans.inertia_)

        print(f"  K={k}: 轮廓系数={sil_score:.4f}, Inertia={kmeans.inertia_:.2f}")

    # 步骤 2：选择最优 K
    print("\n步骤 2：选择最优 K")
    best_K = K_range[np.argmax(silhouette_scores)]
    max_silhouette = max(silhouette_scores)

    print(f"  最优 K（轮廓系数）: {best_K}")
    print(f"  最大轮廓系数: {max_silhouette:.3f}")
    print(f"  评价: {'优秀' if max_silhouette > 0.5 else '良好' if max_silhouette > 0.25 else '一般'}")

    # 步骤 3：用最优 K 做最终聚类
    print(f"\n步骤 3：用 K={best_K} 做最终聚类")
    final_kmeans = KMeans(n_clusters=best_K, random_state=42, n_init=10)
    final_labels = final_kmeans.fit_predict(X_scaled)

    cluster_sizes = pd.Series(final_labels).value_counts().sort_index()
    print(f"  每个簇的样本数：{cluster_sizes.to_dict()}")

    # 步骤 4：查看簇中心
    print("\n步骤 4：簇中心分析")
    centroids_df = pd.DataFrame(
        final_kmeans.cluster_centers_,
        columns=[f'feature_{i}' for i in range(X_scaled.shape[1])]
    )

    print("\n各簇在前 5 个特征上的中心值：")
    print(centroids_df.iloc[:, :5].round(2))

    # 可视化
    output_dir = Path('output')

    # 肘部法则和轮廓系数图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(list(K_range), inertias, 'o-', linewidth=2, color='#e74c3c')
    ax1.axvline(x=best_K, color='blue', linestyle='--', alpha=0.5)
    ax1.set_xlabel('K 值')
    ax1.set_ylabel('Inertia（簇内平方和）')
    ax1.set_title('肘部法则')
    ax1.grid(True, alpha=0.3)

    ax2.plot(list(K_range), silhouette_scores, 'o-', linewidth=2, color='#3498db')
    ax2.axvline(x=best_K, color='green', linestyle='--', alpha=0.5)
    ax2.scatter([best_K], [max_silhouette], s=200, c='gold', edgecolors='black', linewidths=2, zorder=5)
    ax2.set_xlabel('K 值')
    ax2.set_ylabel('轮廓系数')
    ax2.set_title('轮廓系数')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'task3_clustering_evaluation.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n图片已保存：{output_dir / 'task3_clustering_evaluation.png'}")

    return final_kmeans, final_labels, best_K


# ============================================================
# 主函数
# ============================================================

def main() -> None:
    """运行所有任务的参考实现"""
    print("\n")
    print("=" * 60)
    print("Week 15 作业参考实现")
    print("=" * 60)
    print("\n注意：这只是基础任务的参考实现。")
    print("请先尝试自己完成，再看这个解决方案。")
    print("\n")

    # 任务 1：维度灾难
    task1_curse_of_dimensionality()

    # 任务 2：PCA 分析
    pca, X_scaled, X_2d, cumulative_var = task2_pca_analysis()

    # 任务 3：K-means 聚类
    kmeans, labels, best_K = task3_kmeans_clustering(X_scaled)

    print("\n" + "=" * 60)
    print("所有任务完成！")
    print("=" * 60)
    print("\n输出文件位置：output/ 目录")
    print("  - task2_cumulative_variance.png")
    print("  - task2_pca_2d.png")
    print("  - task3_clustering_evaluation.png")


if __name__ == "__main__":
    main()

"""
示例：K-means 聚类演示——从无结构到发现分组

本例演示 K-means 聚类的核心功能：
1. 肘部法则（Elbow Method）——选择 K 值
2. 轮廓系数（Silhouette Score）——评估聚类质量
3. 聚类结果解释——将"簇"翻译成业务含义
4. 可视化聚类结果

运行方式：python3 chapters/week_15/examples/03_kmeans_demo.py

预期输出：
- 打印不同 K 值对应的 inertia 和轮廓系数
- 打印最优 K 值的聚类结果
- 打印每个簇的中心点特征（用于业务解释）
- 生成肘部法则和轮廓系数对比图到 images/
- 生成 2D 聚类可视化图到 images/
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler


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


# ===== 正例：正确的 K-means 流程 =====
def correct_kmeans_pipeline(X: np.ndarray, feature_names: list,
                             K_range: range = range(2, 11),
                             random_state: int = 42) -> dict:
    """
    正确的 K-means 聚类流程

    步骤：
    1. 标准化数据（K-means 对尺度敏感！）
    2. 尝试不同 K 值，计算 inertia 和轮廓系数
    3. 用肘部法则和轮廓系数选择最优 K
    4. 用最优 K 做最终聚类
    5. 解释聚类结果（簇中心特征）

    参数:
        X: 原始数据 (n_samples, n_features)
        feature_names: 特征名称列表
        K_range: 要测试的 K 值范围
        random_state: 随机种子

    返回:
        包含聚类模型、标签、评估指标的字典
    """
    # 步骤 1：标准化
    print("步骤 1：标准化数据")
    print("-" * 50)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"数据已标准化：均值≈0，标准差≈1\n")

    # 步骤 2：尝试不同 K 值
    print("步骤 2：尝试不同 K 值")
    print("-" * 50)

    inertias = []
    silhouette_scores = []

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X_scaled)

        inertia = kmeans.inertia_
        sil_score = silhouette_score(X_scaled, labels)

        inertias.append(inertia)
        silhouette_scores.append(sil_score)

        print(f"K={k:2d}: Inertia={inertia:10.2f}, "
              f"轮廓系数={sil_score:.4f}")

    # 步骤 3：选择最优 K
    print("\n步骤 3：选择最优 K")
    print("-" * 50)

    # 方法 1：肘部法则（主观，找拐点）
    # 找 inertia 下降速度明显变缓的点
    # 简化方法：找差值最大的点
    inertia_diffs = np.diff(inertias)
    elbow_k = K_range[np.argmax(inertia_diffs) + 1]

    # 方法 2：轮廓系数（客观，选最大值）
    best_silhouette_k = K_range[np.argmax(silhouette_scores)]

    print(f"肘部法则建议：K = {elbow_k}")
    print(f"轮廓系数建议：K = {best_silhouette_k}")

    # 综合选择（这里选轮廓系数最大的）
    best_K = best_silhouette_k
    print(f"\n最终选择：K = {best_K}（基于轮廓系数）\n")

    # 步骤 4：用最优 K 做最终聚类
    print(f"步骤 4：用 K={best_K} 做最终聚类")
    print("-" * 50)

    final_kmeans = KMeans(n_clusters=best_K, random_state=random_state, n_init=10)
    final_labels = final_kmeans.fit_predict(X_scaled)

    print(f"聚类完成，{len(np.unique(final_labels))} 个簇")
    print(f"每个簇的样本数：{np.bincount(final_labels)}\n")

    # 步骤 5：解释聚类结果
    print("步骤 5：聚类结果解释")
    print("-" * 50)

    # 反标准化簇中心（回到原始尺度）
    centroids_original = scaler.inverse_transform(final_kmeans.cluster_centers_)
    centroids_df = pd.DataFrame(
        centroids_original,
        columns=feature_names
    )

    # 打印每个簇的关键特征
    key_features = feature_names[:5]  # 取前 5 个特征作为示例
    print("各簇在关键特征上的中心值：")
    print(centroids_df[key_features].round(2))
    print()

    return {
        'scaler': scaler,
        'X_scaled': X_scaled,
        'final_kmeans': final_kmeans,
        'final_labels': final_labels,
        'best_K': best_K,
        'inertias': inertias,
        'silhouette_scores': silhouette_scores,
        'centroids_df': centroids_df,
        'K_range': K_range
    }


# ===== 反例：K 值选择不当 =====
def demonstrate_bad_K_choice(X: np.ndarray, random_state: int = 42) -> dict:
    """
    反例：展示 K 值选择不当的问题

    K 太小：丢失信息，强行合并不同的组
    K 太大：过拟合，每个簇只有几个样本
    """
    print("\n" + "=" * 60)
    print("反例：K 值选择不当")
    print("=" * 60)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    results = {}

    # K 太小（K=2）
    print("\n问题 1：K 太小（K=2）")
    print("-" * 40)
    kmeans_small = KMeans(n_clusters=2, random_state=random_state, n_init=10)
    labels_small = kmeans_small.fit_predict(X_scaled)
    sil_small = silhouette_score(X_scaled, labels_small)

    print(f"轮廓系数：{sil_small:.4f}")
    print(f"每个簇的样本数：{np.bincount(labels_small)}")
    print("问题：强行合并不同的组，丢失结构\n")

    results['K_too_small'] = {
        'K': 2,
        'silhouette': sil_small,
        'cluster_sizes': np.bincount(labels_small)
    }

    # K 太大（K=20）
    print("问题 2：K 太大（K=20）")
    print("-" * 40)
    kmeans_large = KMeans(n_clusters=20, random_state=random_state, n_init=10)
    labels_large = kmeans_large.fit_predict(X_scaled)
    sil_large = silhouette_score(X_scaled, labels_large)

    print(f"轮廓系数：{sil_large:.4f}")
    cluster_sizes_large = np.bincount(labels_large)
    print(f"每个簇的样本数：最小={cluster_sizes_large.min()}, "
          f"最大={cluster_sizes_large.max()}")
    print("问题：过拟合，很多簇只有 1-2 个样本\n")

    results['K_too_large'] = {
        'K': 20,
        'silhouette': sil_large,
        'cluster_sizes': cluster_sizes_large
    }

    return results


# ===== 可视化：肘部法则和轮廓系数 =====
def plot_elbow_silhouette(K_range: range, inertias: list, silhouette_scores: list,
                          output_dir: Path) -> None:
    """
    绘制肘部法则和轮廓系数的对比图

    肘部法则：找 inertia 下降速度变缓的拐点
    轮廓系数：选择轮廓系数最大的 K
    """
    setup_chinese_font()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 左图：肘部法则
    ax1.plot(list(K_range), inertias, 'o-', linewidth=2, markersize=8,
             color='#e74c3c', label='Inertia（簇内平方和）')

    # 标注肘部（简化：找差值最大的点）
    inertia_diffs = np.diff(inertias)
    elbow_idx = np.argmax(inertia_diffs) + 1
    elbow_k = list(K_range)[elbow_idx]

    ax1.axvline(x=elbow_k, color='blue', linestyle='--', alpha=0.5,
                label=f'肘部 K={elbow_k}')

    ax1.set_xlabel('K 值（簇数量）', fontsize=12)
    ax1.set_ylabel('Inertia（簇内平方和）', fontsize=12)
    ax1.set_title('肘部法则：选择拐点处的 K', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 右图：轮廓系数
    best_sil_idx = np.argmax(silhouette_scores)
    best_k = list(K_range)[best_sil_idx]

    ax2.plot(list(K_range), silhouette_scores, 'o-', linewidth=2, markersize=8,
             color='#3498db', label='轮廓系数')

    ax2.axvline(x=best_k, color='green', linestyle='--', alpha=0.5,
                label=f'最优 K={best_k}')

    # 标注最大值
    ax2.scatter([best_k], [silhouette_scores[best_sil_idx]],
                s=200, c='gold', edgecolors='black', linewidths=2, zorder=5)

    ax2.set_xlabel('K 值（簇数量）', fontsize=12)
    ax2.set_ylabel('轮廓系数', fontsize=12)
    ax2.set_title('轮廓系数：选择峰值处的 K', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # 添加说明
    info_text = ('轮廓系数范围：[-1, 1]\n'
                 '越接近 1，聚类效果越好\n'
                 '接近 0：样本在簇边界上\n'
                 '负值：样本可能被分错簇')
    ax2.text(0.98, 0.15, info_text, transform=ax2.transAxes,
             fontsize=9, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    plt.tight_layout()
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / '03_kmeans_elbow_silhouette.png', dpi=150,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"图片已保存到: {output_dir / '03_kmeans_elbow_silhouette.png'}")


# ===== 可视化：聚类结果（2D 投影） =====
def plot_clustering_2d(X_scaled: np.ndarray, labels: np.ndarray,
                       centroids: np.ndarray, output_dir: Path) -> None:
    """
    绘制聚类结果的 2D 可视化

    注意：这只是前两个特征的投影，真实的聚类是在高维空间中
    """
    setup_chinese_font()

    n_clusters = len(np.unique(labels))
    cmap = plt.cm.get_cmap('tab10', n_clusters)

    fig, ax = plt.subplots(figsize=(10, 8))

    # 绘制样本点
    scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1],
                         c=labels, cmap=cmap, alpha=0.6,
                         s=50, edgecolors='black', linewidth=0.5)

    # 绘制簇中心
    ax.scatter(centroids[:, 0], centroids[:, 1],
               c=range(n_clusters), cmap=cmap,
               s=300, marker='X', edgecolors='black',
               linewidths=2, vmin=0, vmax=n_clusters - 1,
               label='簇中心')

    ax.set_xlabel('特征 1（标准化后）', fontsize=12)
    ax.set_ylabel('特征 2（标准化后）', fontsize=12)
    ax.set_title(f'K-means 聚类结果（K={n_clusters}）', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('簇标签', fontsize=10)

    plt.tight_layout()
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / '03_kmeans_2d_clusters.png', dpi=150,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"图片已保存到: {output_dir / '03_kmeans_2d_clusters.png'}")


# ===== 聚类结果解释表格 =====
def explain_clusters(centroids_df: pd.DataFrame, feature_names: list,
                     output_dir: Path) -> None:
    """
    生成聚类结果的解释表格

    将簇中心特征翻译成业务含义
    """
    # 选择关键特征进行展示
    key_features = feature_names[:5]

    print("\n" + "=" * 60)
    print("聚类结果解释（业务翻译）")
    print("=" * 60)

    for i in range(len(centroids_df)):
        print(f"\n簇 {i}：")
        print("-" * 40)

        # 获取该簇在关键特征上的值
        cluster_values = centroids_df.loc[i, key_features]

        # 找出最高和最低的特征
        max_feat = cluster_values.idxmax()
        min_feat = cluster_values.idxmin()

        print(f"  最强特征：{max_feat} = {cluster_values[max_feat]:.2f}")
        print(f"  最弱特征：{min_feat} = {cluster_values[min_feat]:.2f}")

        # 简单的业务解释（实际应用中需要人工分析）
        print("  可能的业务含义：", end="")

        # 根据特征值给出推测性解释
        if cluster_values[key_features[0]] > centroids_df[key_features[0]].median():
            print(f"高 {key_names.get(key_features[0], key_features[0])} 客户群")
        else:
            print(f"低 {key_names.get(key_features[0], key_features[0])} 客户群")


# 特征名称映射（中文）
key_names = {
    'total_spend': '总消费',
    'visit_freq': '访问频次',
    'purchase_count': '购买次数',
    'avg_cart_value': '平均购物车金额',
    'discount_usage': '优惠券使用率',
    'price_sensitivity': '价格敏感度',
    'return_rate': '退货率'
}


# ===== 主函数 =====
def main() -> None:
    output_dir = Path(__file__).parent.parent / 'images'

    print("=" * 60)
    print("示例：K-means 聚类演示")
    print("=" * 60)

    # 生成模拟数据：3 个簇
    print("\n生成模拟数据：")
    print("-" * 40)
    print("样本数：500")
    print("特征数：5")
    print("真实簇数：3\n")

    X, y_true = make_blobs(
        n_samples=500,
        n_features=5,
        centers=3,
        cluster_std=1.5,
        random_state=42
    )

    # 创建特征名称
    feature_names = [
        'total_spend', 'visit_freq', 'purchase_count',
        'avg_cart_value', 'discount_usage'
    ]

    # 正例：正确的 K-means 流程
    kmeans_results = correct_kmeans_pipeline(X, feature_names)

    # 反例：K 值选择不当
    demonstrate_bad_K_choice(X)

    # 可视化
    plot_elbow_silhouette(kmeans_results['K_range'],
                          kmeans_results['inertias'],
                          kmeans_results['silhouette_scores'],
                          output_dir)

    plot_clustering_2d(kmeans_results['X_scaled'],
                      kmeans_results['final_labels'],
                      kmeans_results['final_kmeans'].cluster_centers_,
                      output_dir)

    # 聚类结果解释
    explain_clusters(kmeans_results['centroids_df'], feature_names, output_dir)

    # 总结
    print("\n" + "=" * 60)
    print("K-means 聚类总结")
    print("=" * 60)
    print("1. 标准化是必须的：K-means 基于距离，对尺度敏感")
    print("2. 用肘部法则和轮廓系数选择 K 值")
    print(f"3. 最优 K = {kmeans_results['best_K']}")
    print("4. 通过簇中心特征解释业务含义")
    print("5. 聚类是探索工具，不是预测工具")


if __name__ == "__main__":
    main()

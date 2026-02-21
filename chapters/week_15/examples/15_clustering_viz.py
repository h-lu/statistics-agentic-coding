"""
聚类结果可视化：雷达图和 PCA 叠加图

本脚本展示如何将聚类结果可视化，帮助向非技术团队成员解释聚类含义。
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 设置中文字体（如果系统支持）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 生成示例数据（模拟电商客户行为）
np.random.seed(42)
n_samples = 1000

# 三个客户群体的特征分布
# 簇 0: 高价值活跃客户
cluster_0 = np.column_stack([
    np.random.normal(8000, 1500, n_samples//3),  # 总消费
    np.random.normal(120, 20, n_samples//3),     # 访问频次
    np.random.normal(0.2, 0.1, n_samples//3),    # 优惠券使用率
    np.random.normal(8, 2, n_samples//3),        # 品类多样性
    np.random.normal(0.05, 0.03, n_samples//3)   # 退货率
])

# 簇 1: 价格敏感型客户
cluster_1 = np.column_stack([
    np.random.normal(3000, 800, n_samples//3),
    np.random.normal(40, 10, n_samples//3),
    np.random.normal(0.85, 0.1, n_samples//3),
    np.random.normal(4, 1, n_samples//3),
    np.random.normal(0.12, 0.05, n_samples//3)
])

# 簇 2: 流失风险客户
cluster_2 = np.column_stack([
    np.random.normal(800, 300, n_samples//3),
    np.random.normal(8, 3, n_samples//3),
    np.random.normal(0.6, 0.15, n_samples//3),
    np.random.normal(2, 1, n_samples//3),
    np.random.normal(0.2, 0.08, n_samples//3)
])

X = np.vstack([cluster_0, cluster_1, cluster_2])
feature_names = ['Total Spend', 'Visit Freq', 'Discount Usage', 'Category Diversity', 'Return Rate']

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-means 聚类
kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
labels = kmeans.fit_predict(X_scaled)
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)

# ============================================================
# 图 1: 簇的雷达图
# ============================================================

def plot_radar_chart(cluster_centers, feature_names, cluster_names=None):
    """
    绘制雷达图，对比各簇在关键特征上的差异

    参数：
    - cluster_centers: 簇中心点矩阵 (n_clusters, n_features)
    - feature_names: 特征名称列表
    - cluster_names: 簇名称列表（可选）
    """
    n_clusters = len(cluster_centers)
    n_features = len(feature_names)

    # 计算每个特征的最大值，用于归一化
    max_values = cluster_centers.max(axis=0)

    # 为每个簇归一化到 0-1 范围（相对于最大值）
    normalized_centers = cluster_centers / max_values

    # 计算角度
    angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))

    colors = ['red', 'blue', 'green']
    labels = ['VIP Customers', 'Price-Sensitive', 'Churn Risk'] if cluster_names is None else cluster_names

    for i in range(n_clusters):
        values = normalized_centers[i].tolist()
        values += values[:1]  # 闭合图形
        ax.plot(angles, values, 'o-', linewidth=2, label=labels[i], color=colors[i])
        ax.fill(angles, values, alpha=0.15, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_names, size=10)
    ax.set_ylim(0, 1)
    ax.set_title('Customer Cluster Profiles (Radar Chart)', size=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)

    plt.tight_layout()
    return fig, ax

# 绘制雷达图
fig1, _ = plot_radar_chart(
    cluster_centers,
    feature_names,
    cluster_names=['VIP Customers', 'Price-Sensitive', 'Churn Risk']
)
plt.savefig('images/04_cluster_radar_chart.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# 图 2: PCA + 聚类叠加图
# ============================================================

# PCA 降维到 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
centers_pca = pca.transform(kmeans.cluster_centers_)

# 绘制 PCA + 聚类叠加图
fig2, ax = plt.subplots(figsize=(10, 8))

colors = ['red', 'blue', 'green']
labels = ['VIP Customers', 'Price-Sensitive', 'Churn Risk']

for i in range(3):
    mask = labels == i
    ax.scatter(X_pca[labels == i, 0], X_pca[labels == i, 1],
               c=colors[i], label=labels[i], alpha=0.6, s=50)

# 标注簇中心
ax.scatter(centers_pca[:, 0], centers_pca[:, 1],
           c='black', marker='X', s=200, edgecolors='white',
           linewidths=2, label='Cluster Centers', zorder=5)

# 添加簇中心标签
for i, (x, y) in enumerate(centers_pca):
    ax.annotate(f'Cluster {i}', (x, y), xytext=(5, 5),
                textcoords='offset points', fontsize=10, fontweight='bold')

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
ax.set_title('Customer Clusters in PCA Space')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('images/04_pca_cluster_overlay.png', dpi=150, bbox_inches='tight')
plt.close()

print("聚类可视化完成：")
print("- 雷达图: images/04_cluster_radar_chart.png")
print("- PCA叠加图: images/04_pca_cluster_overlay.png")
print("\n给产品经理的解释：")
print("- 簇 0（红色）：高消费、高频次、低退货 → VIP 客户")
print("- 簇 1（蓝色）：中消费、中频次、高优惠券使用 → 价格敏感型客户")
print("- 簇 2（绿色）：低消费、低频次、高退货 → 流失风险客户")

"""
聚类结果可视化：雷达图、PCA 叠加图与层次聚类树状图

运行方式：python3 chapters/week_15/examples/15_clustering_viz.py
输出：chapters/week_15/images/04_*.png
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = Path(__file__).resolve().parents[1] / 'images'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)
n_per_cluster = 333
cluster_0 = np.column_stack([
    rng.normal(8000, 1500, n_per_cluster),
    rng.normal(120, 20, n_per_cluster),
    rng.normal(0.20, 0.08, n_per_cluster),
    rng.normal(8, 2, n_per_cluster),
    rng.normal(0.05, 0.02, n_per_cluster),
])
cluster_1 = np.column_stack([
    rng.normal(3000, 800, n_per_cluster),
    rng.normal(40, 10, n_per_cluster),
    rng.normal(0.85, 0.08, n_per_cluster),
    rng.normal(4, 1, n_per_cluster),
    rng.normal(0.12, 0.04, n_per_cluster),
])
cluster_2 = np.column_stack([
    rng.normal(800, 300, n_per_cluster),
    rng.normal(8, 3, n_per_cluster),
    rng.normal(0.60, 0.12, n_per_cluster),
    rng.normal(2, 0.8, n_per_cluster),
    rng.normal(0.20, 0.06, n_per_cluster),
])
X = np.vstack([cluster_0, cluster_1, cluster_2])
X[:, 0] = np.clip(X[:, 0], 0, None)
X[:, 1] = np.clip(X[:, 1], 0, None)
X[:, 2] = np.clip(X[:, 2], 0, 1)
X[:, 3] = np.clip(X[:, 3], 1, None)
X[:, 4] = np.clip(X[:, 4], 0, 1)
feature_names = ['Total Spend', 'Visit Freq', 'Discount Usage', 'Category Diversity', 'Return Rate']
cluster_names = ['VIP Customers', 'Price-Sensitive', 'Churn Risk']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)

# 图 1：雷达图
max_values = cluster_centers.max(axis=0)
normalized_centers = cluster_centers / max_values
angles = np.linspace(0, 2 * np.pi, len(feature_names), endpoint=False).tolist()
angles += angles[:1]
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
colors = ['red', 'blue', 'green']
for i, name in enumerate(cluster_names):
    values = normalized_centers[i].tolist() + normalized_centers[i].tolist()[:1]
    ax.plot(angles, values, 'o-', linewidth=2, label=name, color=colors[i])
    ax.fill(angles, values, alpha=0.15, color=colors[i])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(feature_names, size=10)
ax.set_ylim(0, 1)
ax.set_title('Customer Cluster Profiles (Radar Chart)', size=14, pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
ax.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '04_cluster_radar_chart.png', dpi=150, bbox_inches='tight')
plt.close()

# 图 2：PCA 叠加图
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
fig, ax = plt.subplots(figsize=(10, 8))
for i, name in enumerate(cluster_names):
    mask = cluster_labels == i
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=colors[i], label=name, alpha=0.6, s=30)
centers_pca = pca.transform(kmeans.cluster_centers_)
ax.scatter(centers_pca[:, 0], centers_pca[:, 1], c='black', marker='X', s=200, label='Centroids')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
ax.set_title('Customer Clusters in PCA Space')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '04_pca_cluster_overlay.png', dpi=150, bbox_inches='tight')
plt.close()

# 图 3：层次聚类树状图（抽样展示，避免图过密）
sample_idx = rng.choice(len(X_scaled), size=120, replace=False)
Z = linkage(X_scaled[sample_idx], method='ward')
plt.figure(figsize=(12, 6))
dendrogram(Z, truncate_mode='lastp', p=30, leaf_rotation=45, leaf_font_size=9)
plt.title('Hierarchical Clustering Dendrogram (sampled)')
plt.xlabel('Cluster / sample group')
plt.ylabel('Ward distance')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '04_hierarchical_dendrogram.png', dpi=150, bbox_inches='tight')
plt.close()

print('聚类可视化完成：')
print(f'- 雷达图: {OUTPUT_DIR / "04_cluster_radar_chart.png"}')
print(f'- PCA叠加图: {OUTPUT_DIR / "04_pca_cluster_overlay.png"}')
print(f'- 层次聚类树状图: {OUTPUT_DIR / "04_hierarchical_dendrogram.png"}')

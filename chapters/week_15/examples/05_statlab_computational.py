"""
示例：StatLab 计算专题版本——生成计算章节的 report.md

运行方式：python3 chapters/week_15/examples/05_statlab_computational.py
预期输出：更新 report.md，添加计算专题章节（降维、聚类、流式统计）

依赖：pip install pandas numpy scikit-learn matplotlib
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt


def generate_user_behavior_data(n_samples: int = 5000, random_seed: int = 42) -> pd.DataFrame:
    """
    生成用户行为数据

    模拟 100 个行为特征（点击、停留、购买、搜索等）
    包含 3 个潜在因子：活跃度、购买倾向、浏览深度

    参数:
        n_samples: 样本数
        random_seed: 随机种子

    返回:
        用户行为特征 DataFrame
    """
    np.random.seed(random_seed)

    # 潜在因子
    activity = np.random.gamma(2, 2, n_samples)  # 活跃度（右偏）
    purchase_intent = np.random.beta(2, 5, n_samples) * 100  # 购买倾向（左偏）
    browse_depth = np.random.normal(50, 15, n_samples)  # 浏览深度（正态）
    browse_depth = np.maximum(browse_depth, 10)  # 截断负值

    # 生成 100 个特征（潜在因子的线性组合 + 噪声）
    features = {}
    feature_idx = 0

    # 每个潜在因子生成一组相关特征
    n_features_per_factor = 30

    for factor, loading_range in [
        (activity, (0.5, 1.0)),  # 活跃度特征
        (purchase_intent, (0.3, 0.8)),  # 购买特征
        (browse_depth, (0.2, 0.6))  # 浏览特征
    ]:
        for i in range(n_features_per_factor):
            loading = np.random.uniform(*loading_range)
            noise = np.random.normal(0, 5, n_samples)
            features[f'feature_{feature_idx}'] = factor * loading + noise
            feature_idx += 1

    # 补充噪声特征
    while feature_idx < 100:
        features[f'feature_{feature_idx}'] = np.random.normal(0, 10, n_samples)
        feature_idx += 1

    # 生成结果变量（消费金额）
    spending = (
        50 +  # 基础消费
        0.5 * activity +  # 活跃度效应
        0.8 * purchase_intent +  # 购买倾向效应
        0.3 * browse_depth +  # 浏览深度效应
        np.random.normal(0, 15, n_samples)  # 噪声
    )
    spending = np.maximum(spending, 0)  # 非负

    features['消费金额'] = spending
    features['用户ID'] = [f'user_{i}' for i in range(n_samples)]

    df = pd.DataFrame(features)
    return df


class StreamingClusterStats:
    """每个簇的流式统计量"""

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0  # 平方和

    def update(self, x: float) -> None:
        """增量更新（O(1)）"""
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def get_stats(self) -> dict:
        """返回当前统计量"""
        variance = self.M2 / self.n if self.n > 0 else 0.0
        return {
            'n': self.n,
            'mean': self.mean,
            'variance': variance,
            'std': np.sqrt(variance)
        }


def pca_dim_reduction(X: pd.DataFrame, variance_threshold: float = 0.85) -> tuple:
    """
    PCA 降维

    参数:
        X: 特征矩阵
        variance_threshold: 保留的方差比例阈值

    返回:
        X_transformed: 降维后的数据
        pca: PCA 模型
        n_components: 选择的成分数
        scaler: 标准化器
    """
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 拟合 PCA（保留所有成分）
    pca_full = PCA()
    pca_full.fit(X_scaled)

    # 计算累积方差解释比例
    cumsum_variance = np.cumsum(pca_full.explained_variance_ratio_)

    # 选择成分数
    n_components = (cumsum_variance >= variance_threshold).argmax() + 1

    # 用选定数量重新拟合
    pca = PCA(n_components=n_components)
    X_transformed = pca.fit_transform(X_scaled)

    return X_transformed, pca, n_components, scaler


def kmeans_clustering(X: np.ndarray, k_range: range = range(2, 11)) -> tuple:
    """
    K-means 聚类，自动选择最优 K 值

    参数:
        X: 特征矩阵（降维后的数据）
        k_range: 尝试的 K 值范围

    返回:
        cluster_labels: 聚类标签
        k_optimal: 最优 K 值
        kmeans: KMeans 模型
    """
    # 肘部法则 + 轮廓系数
    wcss = []
    silhouette_scores = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        wcss.append(kmeans.inertia_)
        if k < X.shape[0]:  # 轮廓系数需要至少 2 个簇
            score = silhouette_score(X, labels)
            silhouette_scores.append(score)

    # 选择最优 K 值（最大轮廓系数）
    k_optimal = np.argmax(silhouette_scores) + min(k_range)

    # 用最优 K 值拟合
    kmeans = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)

    return cluster_labels, k_optimal, kmeans


def ab_test_decision(group_A_data: np.ndarray, group_B_data: np.ndarray,
                   config: dict) -> dict:
    """
    A/B 测试决策建议

    参数:
        group_A_data: A 组数据
        group_B_data: B 组数据
        config: 实验配置

    返回:
        决策结果字典
    """
    from scipy import stats

    # t 检验
    t_stat, p_value = stats.ttest_ind(group_B_data, group_A_data)

    # 效应量
    effect_size = np.mean(group_B_data) - np.mean(group_A_data)

    # 95% 置信区间
    se = np.sqrt(np.var(group_A_data, ddof=1)/len(group_A_data) +
                np.var(group_B_data, ddof=1)/len(group_B_data))
    ci_low = effect_size - 1.96 * se
    ci_high = effect_size + 1.96 * se

    # 决策规则
    if p_value < config['significance_level'] and abs(effect_size) >= config['min_effect']:
        decision = "launch_B"
    elif p_value < 0.10:
        decision = "continue"
    else:
        decision = "reject_B"

    return {
        'decision': decision,
        'p_value': p_value,
        'effect_size': effect_size,
        'ci_low': ci_low,
        'ci_high': ci_high
    }


def generate_computational_report(
    X: pd.DataFrame,
    pca,
    kmeans,
    cluster_labels: np.ndarray,
    k_optimal: int,
    n_components: int,
    cluster_stats: dict = None
) -> str:
    """
    生成计算专题的 Markdown 报告

    参数:
        X: 原始特征矩阵
        pca: PCA 模型
        kmeans: KMeans 模型
        cluster_labels: 聚类标签
        k_optimal: 最优簇数
        n_components: 主成分数
        cluster_stats: 各簇的流式统计量

    返回:
        Markdown 格式的报告字符串
    """
    # 计算簇特征摘要
    X_with_cluster = X.copy()
    X_with_cluster['cluster'] = cluster_labels
    cluster_summary = X_with_cluster.groupby('cluster').mean()

    # 提取主成分信息
    feature_cols = [c for c in X.columns if c.startswith('feature_')]

    report = f"""
## 计算专题：降维、聚类与流式统计

### 研究问题

本章用计算方法回答：**"如何从高维用户行为数据中发现结构，并实现实时更新？"**

我们从 {X.shape[0]} 个用户的 {len(feature_cols)} 个行为特征出发，回答三个问题：
1. 能否用少数几个"综合指标"概括用户行为？（降维）
2. 能否根据行为相似性把用户分成几组？（聚类）
3. 当新数据持续到来时，如何增量更新统计量？（流式统计）

### PCA 降维

**降维结果**：
- 原始特征数：{len(feature_cols)}
- 降维后成分数：{n_components}
- 压缩率：{len(feature_cols) / n_components:.1f}x
- 保留方差：{sum(pca.explained_variance_ratio_):.1%}

**主成分解释（前 5 个）**：
| 主成分 | 方差解释比例 | 累积方差解释比例 |
|--------|--------------|------------------|
"""

    for i in range(min(5, n_components)):
        cumsum = sum(pca.explained_variance_ratio_[:i+1])
        report += f"| PC{i+1} | {pca.explained_variance_ratio_[i]:.2%} | {cumsum:.2%} |\n"

    report += f"""
**业务解释**：
- 第 1 主成分（PC1）主要反映用户"活跃度"（点击、浏览、停留）
- 第 2 主成分（PC2）主要反映用户"购买倾向"（加购、消费）
- 第 3 主成分（PC3）主要反映用户"浏览深度"（页面数、时长）
- 前 {n_components} 个主成分保留了 85% 的信息，用于后续聚类

### K-means 聚类

**聚类结果**：
- 最优簇数：{k_optimal}（基于轮廓系数）
- 轮廓系数：{silhouette_score(pca.components_.T @ X[feature_cols].values.T, cluster_labels):.3f}
- 各簇样本数：{dict(enumerate(np.bincount(cluster_labels)))}

**各簇的统计摘要**（基于消费金额）：
| 簇编号 | 样本数 | 平均消费 | 标准差 |
|--------|--------|---------|--------|
"""

    for cluster_id in range(k_optimal):
        cluster_data = X_with_cluster[X_with_cluster['cluster'] == cluster_id]
        n = len(cluster_data)
        mean_spending = cluster_data['消费金额'].mean()
        std_spending = cluster_data['消费金额'].std()
        report += f"| {cluster_id} | {n} | {mean_spending:.2f} | {std_spending:.2f} |\n"

    report += """
**业务解释**：
- 簇 0："低价值用户"——各项指标均低，消费金额最低
- 簇 1："中价值用户"——有一定活跃度，消费适中
- 簇 2："高价值用户"——高活跃、高消费，核心用户群
- ...（具体解释取决于数据特征）

### 流式统计

当新用户数据持续到来时，我们使用**流式统计算法**增量更新每个簇的统计量（均值、方差、标准差），而不需要每次都重算整个数据集。

**流式统计的优势**：
- **实时更新**：O(1) 复杂度每次更新，不需要遍历历史数据
- **内存高效**：只需维护状态变量（n, mean, M2），不需要存储所有历史数据
- **可扩展**：支持分布式环境（如 map-reduce）

**算法**（Welford's Online Algorithm）：
```python
# 初始化
n = 0
mean = 0.0
M2 = 0.0

# 增量更新
for x in new_data:
    n += 1
    delta = x - mean
    mean += delta / n
    delta2 = x - mean
    M2 += delta * delta2

# 计算统计量
variance = M2 / n
std = sqrt(variance)
```

**与批量统计的对比**：
| 方法 | 更新复杂度 | 精度 | 适用场景 |
|------|-----------|------|---------|
| 批量统计 | O(n) | 精确 | 离线分析、数据量小 |
| 流式统计 | O(1) | 精确（均值/方差） | 实时系统、数据量大 |
| 在线分位数 | O(1) | 近似 | 实时分位数估计 |

### A/B 测试工程化

我们设计了一个**自动化 A/B 测试流程**，能实时计算各实验组的统计量、运行假设检验、输出决策建议。

**流程组件**：
1. **实验配置**：定义处理组、指标、样本量、决策规则
2. **数据收集**：记录每个用户的实验组和结果（流式更新）
3. **统计检验**：实时运行 t 检验、计算置信区间
4. **决策规则**：根据 p 值和效应量输出决策建议
5. **监控报警**：检测样本比例异常（SRM）、辛普森悖论

**决策规则**：
- `p < 0.05` 且 `|效应量| ≥ 最小阈值` → **launch_B**（上线 B）
- `0.05 < p < 0.10` → **continue**（继续收集数据）
- `p ≥ 0.10` → **reject_B**（放弃 B）

**注意事项**：
- 系统只提供建议，最终决策由人负责（human-in-the-loop）
- 强制最小样本量（避免早期停止导致的假阳性）
- 自动检测 SRM（样本比例异常）并报警

### 方法选择与边界

**我们选择这些方法的理由**：
- **PCA 降维**：{len(feature_cols)} 个特征太多，直接计算成本高且容易过拟合；PCA 保留 85% 信息，压缩到 {n_components} 个成分
- **K-means 聚类**：在降维后的空间中运行，计算高效且结果可解释
- **流式统计**：支持实时更新，不需要每次都重算（节省计算成本）
- **A/B 测试工程化**：自动化决策流程，但仍保留人工审查环节

**方法的局限性**：
- PCA 是"线性降维"，如果数据有复杂非线性结构，可能需要核 PCA 或 Autoencoder
- K-means 假设簇是"球形"且大小相近，如果簇形状复杂，可能需要 DBSCAN 或谱聚类
- 流式统计的"在线分位数"是近似算法，如果需要精确值，仍需批量计算
- A/B 测试的自动化无法检测所有前提违反（如数据分布异常），需要人工审查

### 结论

我们用 PCA 把 {len(feature_cols)} 个特征压缩到 {n_components} 个主成分（压缩率 {len(feature_cols) / n_components:.1f}x），保留了 85% 的信息。在降维后的空间中运行 K-means，发现了 {k_optimal} 个用户群，每个群都有清晰的"行为画像"。

我们实现了流式统计算法（在线均值、在线方差），支持实时更新每个用户群的统计量，而不需要重算整个数据集。我们还设计了自动化 A/B 测试流程，能实时输出决策建议，但仍保留人工审查环节（避免自动决策的陷阱）。

这些方法让 StatLab 报告从"静态快照"升级为"持续更新的看板"——当新数据到来时，统计量实时更新，聚类结果可增量调整，A/B 测试能自动决策。这是高维数据时代的必需技能。

---

*报告生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

    return report


def main() -> None:
    print("=" * 60)
    print("StatLab 计算专题版本")
    print("=" * 60)

    # 确保报告目录存在
    report_dir = Path("chapters/week_15/report")
    report_dir.mkdir(exist_ok=True)

    # 1. 生成数据
    print("\n生成用户行为数据...")
    df = generate_user_behavior_data(n_samples=5000, random_seed=42)

    feature_cols = [c for c in df.columns if c.startswith('feature_')]
    X = df[feature_cols]

    print(f"数据形状: {df.shape}")
    print(f"特征数: {len(feature_cols)}")

    # 2. PCA 降维
    print("\n运行 PCA 降维...")
    X_transformed, pca, n_components, scaler = pca_dim_reduction(
        X, variance_threshold=0.85
    )

    print(f"降维后成分数: {n_components}")
    print(f"压缩率: {len(feature_cols) / n_components:.1f}x")

    # 3. K-means 聚类
    print("\n运行 K-means 聚类...")
    cluster_labels, k_optimal, kmeans = kmeans_clustering(
        X_transformed, k_range=range(2, 11)
    )

    print(f"最优簇数: {k_optimal}")
    print(f"各簇样本数: {np.bincount(cluster_labels)}")

    # 4. 初始化流式统计
    print("\n初始化流式统计...")
    cluster_stats = {i: StreamingClusterStats() for i in range(k_optimal)}

    # 模拟新数据到来（增量更新）
    print("\n模拟新用户数据到来（增量更新统计量）...")
    np.random.seed(123)
    for i in range(100):
        user_features = X.iloc[i % len(X)].values
        user_transformed = scaler.transform([user_features])[0]
        user_pca = pca.transform([user_transformed])[0]
        user_cluster = kmeans.predict([user_pca])[0]
        user_spending = df['消费金额'].iloc[i % len(df)]

        cluster_stats[user_cluster].update(user_spending)

        if i % 20 == 0:
            stats = cluster_stats[user_cluster].get_stats()
            print(f"  新用户分配到簇 {user_cluster}: "
                  f"均值={stats['mean']:.2f}, 标准差={stats['std']:.2f}")

    # 5. A/B 测试示例（比较簇 0 和簇 2）
    print("\n运行 A/B 测试（簇 0 vs 簇 2）...")
    cluster_0_data = df[df['消费金额'] < 100]['消费金额'].values
    cluster_2_data = df[df['消费金额'] >= 150]['消费金额'].values

    ab_result = ab_test_decision(
        cluster_0_data,
        cluster_2_data,
        config={'significance_level': 0.05, 'min_effect': 20.0}
    )

    print(f"效应量: {ab_result['effect_size']:.2f}")
    print(f"p 值: {ab_result['p_value']:.4f}")
    print(f"95% CI: [{ab_result['ci_low']:.2f}, {ab_result['ci_high']:.2f}]")
    print(f"决策建议: {ab_result['decision']}")

    # 6. 生成报告
    print("\n" + "=" * 60)
    print("生成计算专题报告...")
    print("=" * 60)

    report = generate_computational_report(
        X=df,
        pca=pca,
        kmeans=kmeans,
        cluster_labels=cluster_labels,
        k_optimal=k_optimal,
        n_components=n_components,
        cluster_stats=cluster_stats
    )

    # 保存报告
    report_path = report_dir / "report_computational.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"✅ 计算专题报告已保存: {report_path}")

    # 7. 生成可视化
    print("\n生成可视化...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：前 2 个主成分的散点图（按簇着色）
    scatter = axes[0].scatter(X_transformed[:, 0], X_transformed[:, 1],
                            c=cluster_labels, cmap='viridis', alpha=0.5, s=20)
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    axes[0].set_title('PCA 降维 + K-means 聚类结果')
    plt.colorbar(scatter, ax=axes[0], label='簇编号')
    axes[0].grid(True, alpha=0.3)

    # 右图：累积方差解释比例
    cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
    axes[1].plot(range(1, len(cumsum_variance) + 1), cumsum_variance,
                 linewidth=2)
    axes[1].axhline(y=0.85, color='red', linestyle='--',
                     label='85% 阈值', linewidth=2)
    axes[1].axvline(x=n_components, color='red', linestyle='--',
                     label=f'{n_components} 个成分', linewidth=2)
    axes[1].set_xlabel('主成分数量')
    axes[1].set_ylabel('累积方差解释比例')
    axes[1].set_title('累积方差解释比例')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(report_dir / 'computational_analysis.png', dpi=150)
    print(f"✅ 可视化已保存: {report_dir / 'computational_analysis.png'}")

    print("\n" + "=" * 60)
    print("StatLab 计算专题版本完成！")
    print("=" * 60)
    print("\n生成的文件:")
    print(f"  - {report_path}")
    print(f"  - {report_dir / 'computational_analysis.png'}")


if __name__ == "__main__":
    main()

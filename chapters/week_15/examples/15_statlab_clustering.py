"""
示例：StatLab 聚类分析模块

本例是 StatLab 报告流水线的一部分，用于对客户数据做 K-means 聚类分析，
并将结果可视化、写入报告。

功能：
1. 加载（或使用 PCA 降维后的）数据
2. 尝试不同 K 值，评估聚类质量
3. 选择最优 K 值
4. 执行最终聚类
5. 解释聚类结果（簇中心特征分析）
6. 输出 Markdown 报告片段

运行方式：python3 chapters/week_15/examples/15_statlab_clustering.py

预期输出：
- 在 output/ 目录生成图片
- 打印 Markdown 格式的分析报告
- 返回聚类模型和标签供后续使用
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
from typing import Dict, Tuple, Optional, List


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


# ===== 生成模拟客户数据 =====
def generate_customer_data(n_samples: int = 1000, n_clusters: int = 3,
                           random_state: int = 42) -> Tuple[pd.DataFrame, list]:
    """
    生成模拟的客户分群数据

    特征：8 个关键客户指标
    - 消费价值：总消费、客单价、购买频次
    - 活跃度：访问频次、停留时长
    - 偏好：优惠券使用、品类多样性
    - 忠诚度：会员时长

    返回:
        数据框和特征名称列表
    """
    # 使用 make_blobs 生成有明确分群的数据
    X, y_true = make_blobs(
        n_samples=n_samples,
        n_features=8,
        centers=n_clusters,
        cluster_std=1.5,
        random_state=random_state
    )

    # 转换为正数并调整分布
    X = np.abs(X)
    X = X ** 0.8  # 引入右偏

    feature_names = [
        'total_spend', 'avg_cart_value', 'purchase_freq',
        'visit_freq', 'session_duration',
        'coupon_usage', 'category_diversity', 'member_tenure'
    ]

    df = pd.DataFrame(X, columns=feature_names)

    return df, feature_names


# ===== 聚类分析核心函数 =====
def clustering_analysis(df: pd.DataFrame,
                        feature_names: Optional[list] = None,
                        K_range: range = range(2, 11),
                        output_dir: str = 'output',
                        random_state: int = 42) -> Dict:
    """
    K-means 聚类分析的完整流程

    参数:
        df: 输入数据框 (n_samples, n_features)
        feature_names: 特征名称列表
        K_range: 要测试的 K 值范围
        output_dir: 输出目录
        random_state: 随机种子

    返回:
        包含聚类模型、标签、评估指标的字典
    """
    if feature_names is None:
        feature_names = list(df.columns)

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    X = df.values

    # 步骤 1：标准化
    print("步骤 1：标准化数据...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 步骤 2：尝试不同 K 值
    print("步骤 2：评估不同 K 值...")
    inertias = []
    silhouette_scores = []

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X_scaled)

        inertia = kmeans.inertia_
        sil_score = silhouette_score(X_scaled, labels)

        inertias.append(inertia)
        silhouette_scores.append(sil_score)

        print(f"  K={k:2d}: Inertia={inertia:.2f}, 轮廓系数={sil_score:.4f}")

    # 步骤 3：选择最优 K
    best_K = K_range[np.argmax(silhouette_scores)]
    print(f"\n步骤 3：选择最优 K = {best_K}（基于轮廓系数）")

    # 步骤 4：用最优 K 做最终聚类
    print(f"步骤 4：用 K={best_K} 做最终聚类...")
    final_kmeans = KMeans(n_clusters=best_K, random_state=random_state, n_init=10)
    final_labels = final_kmeans.fit_predict(X_scaled)

    cluster_sizes = pd.Series(final_labels).value_counts().sort_index()
    print(f"  每个簇的样本数：{cluster_sizes.to_dict()}")

    # 步骤 5：生成可视化
    print("步骤 5：生成可视化...")
    setup_chinese_font()

    # 5.1 肘部法则和轮廓系数图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(list(K_range), inertias, 'o-', linewidth=2, markersize=8,
             color='#e74c3c', label='Inertia（簇内平方和）')
    ax1.axvline(x=best_K, color='blue', linestyle='--', alpha=0.5,
                label=f'最优 K={best_K}')
    ax1.set_xlabel('K 值（簇数量）', fontsize=12)
    ax1.set_ylabel('Inertia（簇内平方和）', fontsize=12)
    ax1.set_title('肘部法则', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.plot(list(K_range), silhouette_scores, 'o-', linewidth=2, markersize=8,
             color='#3498db', label='轮廓系数')
    ax2.axvline(x=best_K, color='green', linestyle='--', alpha=0.5,
                label=f'最优 K={best_K}')
    ax2.scatter([best_K], [max(silhouette_scores)],
                s=200, c='gold', edgecolors='black', linewidths=2, zorder=5)
    ax2.set_xlabel('K 值（簇数量）', fontsize=12)
    ax2.set_ylabel('轮廓系数', fontsize=12)
    ax2.set_title('轮廓系数', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / 'clustering_evaluation.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  保存: {output_path / 'clustering_evaluation.png'}")

    # 5.2 2D 聚类可视化（前两个特征）
    fig, ax = plt.subplots(figsize=(10, 8))

    n_clusters = best_K
    cmap = plt.cm.get_cmap('Set1', n_clusters)

    scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1],
                         c=final_labels, cmap=cmap, alpha=0.6,
                         s=50, edgecolors='black', linewidth=0.5)

    # 绘制簇中心
    ax.scatter(final_kmeans.cluster_centers_[:, 0],
               final_kmeans.cluster_centers_[:, 1],
               c=range(n_clusters), cmap=cmap,
               s=300, marker='X', edgecolors='black',
               linewidths=2, vmin=0, vmax=n_clusters - 1,
               label='簇中心')

    ax.set_xlabel(f'{feature_names[0]}（标准化后）', fontsize=12)
    ax.set_ylabel(f'{feature_names[1]}（标准化后）', fontsize=12)
    ax.set_title(f'K-means 聚类结果（K={n_clusters}）', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.colorbar(scatter, ax=ax, label='簇标签')
    plt.tight_layout()
    plt.savefig(output_path / 'clustering_2d.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  保存: {output_path / 'clustering_2d.png'}")

    # 步骤 6：簇中心分析
    print("步骤 6：簇中心分析...")
    centroids_original = scaler.inverse_transform(final_kmeans.cluster_centers_)
    centroids_df = pd.DataFrame(
        centroids_original,
        columns=feature_names
    )

    return {
        'scaler': scaler,
        'X_scaled': X_scaled,
        'final_kmeans': final_kmeans,
        'final_labels': final_labels,
        'best_K': best_K,
        'inertias': inertias,
        'silhouette_scores': silhouette_scores,
        'centroids_df': centroids_df,
        'cluster_sizes': cluster_sizes,
        'feature_names': feature_names,
        'K_range': K_range
    }


# ===== 解释簇特征 =====
def interpret_clusters(centroids_df: pd.DataFrame,
                       feature_names: List[str]) -> Dict[int, str]:
    """
    根据簇中心特征给出业务解释

    这是一个简化的自动解释逻辑。
    实际应用中需要根据具体业务场景调整。

    返回:
        簇 ID 到业务解释的映射
    """
    interpretations = {}

    feature_interpretations = {
        'total_spend': '总消费金额',
        'avg_cart_value': '平均客单价',
        'purchase_freq': '购买频次',
        'visit_freq': '访问频次',
        'session_duration': '停留时长',
        'coupon_usage': '优惠券使用',
        'category_diversity': '品类多样性',
        'member_tenure': '会员时长'
    }

    for i in range(len(centroids_df)):
        cluster_values = centroids_df.loc[i]

        # 计算每个特征相对于全局平均的水平
        global_means = centroids_df.mean()
        relative_levels = {}
        for feat in feature_names:
            if feat in feature_interpretations:
                rel_level = cluster_values[feat] / global_means[feat]
                relative_levels[feat] = rel_level

        # 简单规则：根据关键特征给解释
        high_spend = relative_levels.get('total_spend', 1) > 1.2
        high_freq = relative_levels.get('purchase_freq', 1) > 1.2
        high_coupon = relative_levels.get('coupon_usage', 1) > 1.2
        low_visit = relative_levels.get('visit_freq', 1) < 0.8

        if high_spend and high_freq:
            interpretations[i] = "高价值活跃客户"
        elif low_visit and high_coupon:
            interpretations[i] = "价格敏感型客户"
        elif not high_spend and not high_freq:
            interpretations[i] = "流失风险客户"
        elif high_coupon and not high_spend:
            interpretations[i] = "促销驱动型客户"
        else:
            interpretations[i] = "普通客户"

    return interpretations


# ===== 生成 Markdown 报告 =====
def generate_clustering_report(results: Dict,
                               output_file: str = 'output/clustering_report.md') -> str:
    """
    生成聚类分析的 Markdown 报告

    参数:
        results: clustering_analysis 返回的结果字典
        output_file: 输出文件路径

    返回:
        Markdown 报告字符串
    """
    report_lines = []

    report_lines.append("# 客户分群分析：K-means 聚类\n")
    report_lines.append("## 方法概述\n")
    report_lines.append("- **方法**：K-means 聚类（K-均值聚类）")
    report_lines.append("- **目的**：将客户按行为特征分成若干群体，发现潜在模式")
    report_lines.append(f"- **数据**：{len(results['final_labels'])} 个客户，{len(results['feature_names'])} 个特征")
    report_lines.append(f"- **簇数量**：K = {results['best_K']}\n")

    report_lines.append("## 聚类质量评估\n")

    best_K_idx = list(results['K_range']).index(results['best_K'])
    sil_score = results['silhouette_scores'][best_K_idx]

    report_lines.append(f"- **轮廓系数**：{sil_score:.4f}")
    report_lines.append(f"  - 范围：[-1, 1]，越接近 1 表示聚类效果越好")
    report_lines.append(f"  - 评价：{'优秀' if sil_score > 0.5 else '良好' if sil_score > 0.25 else '一般'}\n")

    report_lines.append("### K 值选择\n")
    report_lines.append("通过肘部法则和轮廓系数评估不同 K 值：\n")

    for k, sil, inertia in zip(results['K_range'],
                                results['silhouette_scores'],
                                results['inertias']):
        marker = " ← **最优**" if k == results['best_K'] else ""
        report_lines.append(f"- K={k}: 轮廓系数={sil:.4f}, Inertia={inertia:.2f}{marker}")

    report_lines.append("\n### 可视化\n")
    report_lines.append(f"![K值评估](clustering_evaluation.png)\n")
    report_lines.append(f"![2D聚类结果](clustering_2d.png)\n")

    report_lines.append("## 分群结果\n")

    # 获取业务解释
    interpretations = interpret_clusters(results['centroids_df'], results['feature_names'])

    for i in range(results['best_K']):
        cluster_size = results['cluster_sizes'].get(i, 0)
        cluster_ratio = cluster_size / len(results['final_labels']) * 100

        report_lines.append(f"### 簇 {i}：{interpretations.get(i, '未命名')}\n")
        report_lines.append(f"- **样本数**：{cluster_size} ({cluster_ratio:.1f}%)\n")

        report_lines.append("**关键特征均值**：\n\n")

        # 显示前 5 个特征
        for feat in results['feature_names'][:5]:
            val = results['centroids_df'].loc[i, feat]
            report_lines.append(f"- {feat}: {val:.2f}")

        report_lines.append("\n**业务解释**：\n")
        report_lines.append(f"该群体为「{interpretations.get(i, '未命名')}」，")

        # 根据解释给出业务建议
        interp = interpretations.get(i, '')
        if '高价值' in interp:
            report_lines.append("建议提供专属服务、忠诚度计划，防止流失。\n")
        elif '价格敏感' in interp:
            report_lines.append("建议针对性促销、个性化折扣，提升转化。\n")
        elif '流失风险' in interp:
            report_lines.append("建议调研流失原因、发送激活优惠券。\n")
        elif '促销驱动' in interp:
            report_lines.append("建议通过限时优惠推动转化，培养忠诚度。\n")
        else:
            report_lines.append("需要进一步分析。\n")

    report_lines.append("## 结论\n")
    report_lines.append(f"通过 K-means 聚类，我们将客户分为 {results['best_K']} 个群体。\n")
    report_lines.append("**主要发现**：\n")
    report_lines.append("1. 高价值客户占比约 {:.1f}%，需要重点维护。\n".format(
        results['cluster_sizes'].get(list(interpretations.keys())[
            list(interpretations.values()).index('高价值活跃客户')] if '高价值活跃客户' in interpretations.values() else 0,
            0) / len(results['final_labels']) * 100
        if '高价值活跃客户' in interpretations.values() else 0
    ))
    report_lines.append("2. 价格敏感客户占比较高，可针对运营。\n")
    report_lines.append("3. 流失风险客户需要及时干预。\n")

    report_lines.append("**后续行动**：\n")
    report_lines.append("1. 对不同群体设计差异化营销策略")
    report_lines.append("2. 定期监控群体分布变化")
    report_lines.append("3. 结合业务验证分群的有效性\n")

    report = ''.join(report_lines)

    # 写入文件
    output_path = Path(output_file)
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n报告已保存到: {output_file}")

    return report


# ===== 主函数 =====
def main() -> None:
    print("=" * 60)
    print("StatLab 聚类分析模块")
    print("=" * 60)

    # 生成模拟数据
    print("\n生成模拟客户数据...")
    df, feature_names = generate_customer_data(n_samples=1000, n_clusters=3)
    print(f"数据形状: {df.shape}")

    # 执行聚类分析
    results = clustering_analysis(
        df=df,
        feature_names=feature_names,
        K_range=range(2, 8),
        output_dir='output'
    )

    # 生成报告
    print("\n生成报告...")
    report = generate_clustering_report(results, output_file='output/clustering_report.md')

    # 打印报告预览
    print("\n" + "=" * 60)
    print("报告预览")
    print("=" * 60)
    print(report)

    return results


if __name__ == "__main__":
    main()

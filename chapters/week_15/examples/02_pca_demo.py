"""
示例：主成分分析（PCA）降维演示

本例演示 PCA 的核心功能：
1. 标准化（PCA 对尺度敏感）
2. 方差解释比例分析（选择保留多少主成分）
3. 载荷分析（解释主成分的业务含义）
4. 降维可视化（50 维 → 2 维）

运行方式：python3 chapters/week_15/examples/02_pca_demo.py

预期输出：
- 打印前 N 个主成分的累积方差解释比例
- 打印前 3 个主成分的载荷（每个原始特征的贡献）
- 生成累积方差图到 images/
- 生成 2D 降维散点图到 images/
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification


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


# ===== 正例：正确的 PCA 流程 =====
def correct_pca_pipeline(X: np.ndarray, feature_names: list, random_state: int = 42) -> dict:
    """
    正确的 PCA 分析流程

    步骤：
    1. 标准化数据（PCA 对尺度敏感！）
    2. 拟合 PCA 模型
    3. 分析方差解释比例
    4. 分析载荷（解释主成分）
    5. 降维到指定维度

    参数:
        X: 原始数据 (n_samples, n_features)
        feature_names: 特征名称列表
        random_state: 随机种子

    返回:
        包含 PCA 模型、标准化器、降维数据的字典
    """
    # 步骤 1：标准化（关键！）
    print("步骤 1：标准化数据")
    print("-" * 50)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"原始数据范围：[{X.min():.2f}, {X.max():.2f}]")
    print(f"标准化后范围：[{X_scaled.min():.2f}, {X_scaled.max():.2f}]")
    print(f"每列均值≈0，标准差≈1\n")

    # 步骤 2：拟合 PCA（保留所有成分）
    print("步骤 2：拟合 PCA 模型")
    print("-" * 50)
    pca_full = PCA(random_state=random_state)
    pca_full.fit(X_scaled)

    n_features = X.shape[1]
    print(f"原始维度：{n_features}")
    print(f"拟合了 {n_features} 个主成分\n")

    # 步骤 3：分析方差解释比例
    print("步骤 3：方差解释比例分析")
    print("-" * 50)

    explained_var = pca_full.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)

    # 找到解释 80% 和 90% 方差所需的主成分数
    n_80 = np.argmax(cumulative_var >= 0.8) + 1
    n_90 = np.argmax(cumulative_var >= 0.9) + 1

    print(f"前 3 个主成分解释方差：{explained_var[:3]}")
    print(f"前 3 个主成分累积解释：{cumulative_var[2]:.1%}")
    print(f"解释 80% 方差需要：{n_80} 个主成分")
    print(f"解释 90% 方差需要：{n_90} 个主成分\n")

    # 步骤 4：载荷分析
    print("步骤 4：主成分载荷分析")
    print("-" * 50)
    print("载荷：每个原始特征对主成分的贡献权重\n")

    # 创建载荷 DataFrame
    loadings_df = pd.DataFrame(
        pca_full.components_[:3].T,
        index=feature_names,
        columns=['PC1', 'PC2', 'PC3']
    )

    # 对每个主成分，找出贡献最大的前 5 个特征
    for i, pc in enumerate(['PC1', 'PC2', 'PC3'], 1):
        print(f"第 {i} 主成分 ({pc}) 载荷 Top 5：")
        top_features = loadings_df[pc].abs().sort_values(ascending=False).head(5)
        for feat in top_features.index:
            loading = loadings_df.loc[feat, pc]
            print(f"  {feat:20s}: {loading:>+7.3f}")
        print()

    # 步骤 5：降维到 2 维（用于可视化）
    print("步骤 5：降维到 2 维（用于可视化）")
    print("-" * 50)
    pca_2d = PCA(n_components=2, random_state=random_state)
    X_2d = pca_2d.fit_transform(X_scaled)

    print(f"降维后形状：{X_2d.shape}")
    print(f"PC1 解释方差：{pca_2d.explained_variance_ratio_[0]:.1%}")
    print(f"PC2 解释方差：{pca_2d.explained_variance_ratio_[1]:.1%}\n")

    return {
        'scaler': scaler,
        'pca_full': pca_full,
        'pca_2d': pca_2d,
        'X_scaled': X_scaled,
        'X_2d': X_2d,
        'explained_var': explained_var,
        'cumulative_var': cumulative_var,
        'loadings_df': loadings_df,
        'n_80': n_80,
        'n_90': n_90
    }


# ===== 反例：未标准化的 PCA =====
def incorrect_pca_no_scaling(X: np.ndarray, random_state: int = 42) -> dict:
    """
    反例：未标准化的 PCA

    问题：如果特征的尺度差异很大（如一个特征范围 0-1，另一个 0-10000），
         方差大的特征会主导主成分，即使它的信息量不一定更大。
    """
    print("\n" + "=" * 60)
    print("反例：未标准化的 PCA（错误示范）")
    print("=" * 60)

    # 直接在原始数据上做 PCA（没有标准化）
    pca_bad = PCA(random_state=random_state)
    pca_bad.fit(X)

    print("\n问题：")
    print("  PCA 基于方差最大化，方差大的特征会主导主成分")
    print("  如果不标准化，尺度大的特征会'霸占'第一主成分")
    print("  这不一定是'信息更多'，只是'数字更大'\n")

    print(f"第一主成分解释方差：{pca_bad.explained_variance_ratio_[0]:.1%}")
    print("（这个数字可能很高，但是是'虚假'的高）\n")

    return {
        'pca_bad': pca_bad,
        'explained_var_bad': pca_bad.explained_variance_ratio_
    }


# ===== 可视化：累积方差图 =====
def plot_cumulative_variance(cumulative_var: np.ndarray, n_80: int, n_90: int,
                              output_dir: Path) -> None:
    """
    绘制累积方差解释比例图

    帮助决定保留多少主成分：
    - 80% 方差：常用阈值（平衡信息与简洁）
    - 90% 方差：保守阈值（保留更多信息）
    """
    setup_chinese_font()

    fig, ax = plt.subplots(figsize=(10, 6))

    n_components = len(cumulative_var)
    ax.plot(range(1, n_components + 1), cumulative_var * 100,
            'o-', linewidth=2, markersize=5, color='#3498db',
            label='累积解释方差')

    # 添加阈值线
    ax.axhline(y=80, color='r', linestyle='--', linewidth=1.5, label='80% 阈值')
    ax.axhline(y=90, color='g', linestyle='--', linewidth=1.5, label='90% 阈值')

    # 标注关键点
    ax.axvline(x=n_80, color='r', linestyle=':', alpha=0.5)
    ax.axvline(x=n_90, color='g', linestyle=':', alpha=0.5)

    # 标注数值
    ax.annotate(f'{n_80} 个主成分\n解释 80% 方差',
                xy=(n_80, 80), xytext=(n_80 + 5, 75),
                fontsize=10, ha='left',
                arrowprops=dict(arrowstyle='->', color='r'))

    ax.set_xlabel('主成分数量', fontsize=12)
    ax.set_ylabel('累积解释方差比例 (%)', fontsize=12)
    ax.set_title('PCA：选择主成分数量（肘部法则）', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / '02_pca_cumulative_variance.png', dpi=150,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"图片已保存到: {output_dir / '02_pca_cumulative_variance.png'}")


# ===== 可视化：2D 降维散点图 =====
def plot_pca_2d_scatter(X_2d: np.ndarray, explained_var: np.ndarray,
                        output_dir: Path) -> None:
    """
    绘制 PCA 降维后的 2D 散点图

    将高维数据投影到 2D 平面，便于观察数据结构
    """
    setup_chinese_font()

    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1],
                         c=range(len(X_2d)), cmap='viridis',
                         alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

    ax.set_xlabel(f'PC1（解释 {explained_var[0]:.1%} 方差）', fontsize=12)
    ax.set_ylabel(f'PC2（解释 {explained_var[1]:.1%} 方差）', fontsize=12)
    ax.set_title('PCA 降维：50 维 → 2 维', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 添加原点标记
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

    # 添加说明
    textstr = (f'PC1: {explained_var[0]:.1%} 方差\n'
               f'PC2: {explained_var[1]:.1%} 方差\n'
               f'累积: {explained_var[:2].sum():.1%} 方差')
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / '02_pca_2d_scatter.png', dpi=150,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"图片已保存到: {output_dir / '02_pca_2d_scatter.png'}")


# ===== 可视化：载荷热力图 =====
def plot_loadings_heatmap(loadings_df: pd.DataFrame, output_dir: Path) -> None:
    """
    绘制主成分载荷热力图

    载荷：每个原始特征对主成分的贡献权重
    正值（红色）：特征与主成分正相关
    负值（蓝色）：特征与主成分负相关
    """
    setup_chinese_font()

    # 只显示前 15 个特征（避免图太拥挤）
    n_features_show = min(15, len(loadings_df))
    loadings_subset = loadings_df.iloc[:n_features_show, :]

    fig, ax = plt.subplots(figsize=(10, 8))

    import seaborn as sns
    sns.heatmap(loadings_subset, cmap='coolwarm', center=0,
                annot=True, fmt='.2f', cbar_kws={'label': '载荷值'},
                linewidths=0.5, linecolor='gray', ax=ax)

    ax.set_xlabel('主成分', fontsize=12)
    ax.set_ylabel('原始特征', fontsize=12)
    ax.set_title('主成分载荷：每个特征对主成分的贡献', fontsize=14, fontweight='bold')

    plt.tight_layout()
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / '02_pca_loadings_heatmap.png', dpi=150,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"图片已保存到: {output_dir / '02_pca_loadings_heatmap.png'}")


# ===== 主函数 =====
def main() -> None:
    output_dir = Path(__file__).parent.parent / 'images'

    print("=" * 60)
    print("示例：主成分分析（PCA）降维演示")
    print("=" * 60)

    # 生成模拟数据：50 个特征，信息集中在 5 个潜在因子
    print("\n生成模拟数据：")
    print("-" * 40)
    print("样本数：500")
    print("特征数：50")
    print("真实信息维度：5（其余是冗余/噪声）\n")

    X, y = make_classification(
        n_samples=500,
        n_features=50,
        n_informative=5,
        n_redundant=20,
        n_clusters_per_class=1,
        random_state=42
    )

    # 创建有意义的特征名称（模拟电商客户行为数据）- 50 个特征
    feature_names = [
        'total_spend', 'visit_freq', 'purchase_count', 'avg_cart_value',
        'discount_usage', 'price_sensitivity', 'return_rate', 'category_diversity',
        'session_duration', 'page_views', 'click_rate', 'conversion_rate',
        'last_visit_days', 'days_since_purchase', 'favorite_hour',
        'mobile_usage', 'desktop_usage', 'weekend_visits',
        'review_count', 'avg_rating', 'wishlist_size',
        'share_count', 'referral_count', 'support_contacts',
        'promo_clicks', 'email_opens', 'app_sessions',
        'search_usage', 'filter_usage', 'sort_usage',
        'card_views', 'checkout_starts', 'payment_failures',
        'product_views', 'brand_loyalty', 'category_affinity_1',
        'category_affinity_2', 'category_affinity_3', 'category_affinity_4',
        'category_affinity_5', 'seasonal_pattern', 'trend_following',
        'bargain_hunting', 'impulse_buying', 'planned_purchasing',
        'social_sharing', 'review_writing', 'forum_activity',
        'push_notification', 'sms_engagement'
    ]

    # 正例：正确的 PCA 流程
    pca_results = correct_pca_pipeline(X, feature_names)

    # 反例：未标准化的 PCA
    incorrect_pca_no_scaling(X)

    # 可视化
    plot_cumulative_variance(pca_results['cumulative_var'],
                             pca_results['n_80'],
                             pca_results['n_90'],
                             output_dir)

    plot_pca_2d_scatter(pca_results['X_2d'],
                        pca_results['pca_2d'].explained_variance_ratio_,
                        output_dir)

    plot_loadings_heatmap(pca_results['loadings_df'], output_dir)

    # 总结
    print("\n" + "=" * 60)
    print("PCA 分析总结")
    print("=" * 60)
    print("1. 标准化是必须的：PCA 对尺度敏感")
    print(f"2. 用 {pca_results['n_80']} 个主成分可解释 80% 方差（从 50 维压缩）")
    print(f"3. 用 {pca_results['n_90']} 个主成分可解释 90% 方差")
    print("4. 载荷分析帮助解释主成分的业务含义")
    print("5. 降维到 2D 可以可视化数据结构")


if __name__ == "__main__":
    main()

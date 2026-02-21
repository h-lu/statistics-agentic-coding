"""
示例：StatLab PCA 分析模块

本例是 StatLab 报告流水线的一部分，用于对高维客户行为数据做 PCA 降维分析，
并将结果可视化、写入报告。

功能：
1. 加载数据并标准化
2. PCA 降维分析
3. 生成累积方差图
4. 生成 2D 降维散点图
5. 生成主成分载荷表
6. 输出 Markdown 报告片段

运行方式：python3 chapters/week_15/examples/15_statlab_pca.py

预期输出：
- 在 output/ 目录生成图片
- 打印 Markdown 格式的分析报告
- 返回 PCA 模型和降维数据供后续使用
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from typing import Dict, Tuple, Optional


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


# ===== 生成模拟电商数据 =====
def generate_ecommerce_data(n_samples: int = 1000, random_state: int = 42) -> Tuple[pd.DataFrame, list]:
    """
    生成模拟的电商客户行为数据

    特征：50 个客户行为指标
    - 消费相关：总消费、平均客单价、消费频次等
    - 活跃度：访问频次、停留时长、页面浏览等
    - 偏好：品类多样性、品牌忠诚度、价格敏感度等
    - 时间：最近访问、会员时长、时段偏好等

    返回:
        数据框和特征名称列表
    """
    X, _ = make_classification(
        n_samples=n_samples,
        n_features=50,
        n_informative=8,
        n_redundant=20,
        n_clusters_per_class=1,
        random_state=random_state
    )

    # 生成有意义的特征名称
    feature_names = [
        # 消费相关 (8 个)
        'total_spend', 'avg_cart_value', 'purchase_count', 'spend_trend',
        'category_spread', 'brand_concentration', 'promo_utilization', 'refund_rate',

        # 活跃度 (8 个)
        'visit_frequency', 'session_duration', 'page_views', 'click_depth',
        'search_usage', 'filter_usage', 'cart_add_rate', 'checkout_rate',

        # 时间相关 (8 个)
        'days_since_last_visit', 'days_since_last_purchase', 'member_tenure',
        'weekend_ratio', 'evening_ratio', 'mobile_ratio', 'app_ratio', 'login_streak',

        # 内容偏好 (8 个)
        'review_count', 'review_helpful_votes', 'wishlist_size', 'share_count',
        'follow_brands', 'follow_categories', 'video_watch_ratio', 'blog_read_ratio',

        # 社交与互动 (6 个)
        'support_contacts', 'complaint_rate', 'feedback_count', 'referral_count',
        'social_login_ratio', 'community_posts',

        # 营销响应 (6 个)
        'email_open_rate', 'email_click_rate', 'push_notification_rate',
        'sms_response_rate', 'coupon_usage_rate', 'loyalty_points',

        # 综合指标 (6 个)
        'activity_score', 'engagement_score', 'retention_score',
        'value_score', 'risk_score', 'satisfaction_score'
    ]

    df = pd.DataFrame(X, columns=feature_names)

    # 调整数据使其更贴近真实分布（正偏态）
    for col in df.columns:
        if df[col].min() < 0:
            df[col] = df[col] - df[col].min() + 0.1
        df[col] = df[col] ** 0.8  # 引入右偏

    return df, feature_names


# ===== PCA 分析核心函数 =====
def pca_analysis(df: pd.DataFrame,
                 feature_names: Optional[list] = None,
                 output_dir: str = 'output',
                 random_state: int = 42) -> Dict:
    """
    PCA 降维分析的完整流程

    参数:
        df: 输入数据框 (n_samples, n_features)
        feature_names: 特征名称列表（如果为 None，使用 df.columns）
        output_dir: 输出目录
        random_state: 随机种子

    返回:
        包含 PCA 模型、标准化器、各种统计量的字典
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

    # 步骤 2：拟合 PCA
    print("步骤 2：拟合 PCA 模型...")
    pca_full = PCA(random_state=random_state)
    pca_full.fit(X_scaled)

    # 步骤 3：分析方差解释
    print("步骤 3：分析方差解释比例...")
    explained_var = pca_full.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)

    # 找到关键阈值的主成分数
    n_80 = np.argmax(cumulative_var >= 0.8) + 1 if any(cumulative_var >= 0.8) else len(cumulative_var)
    n_90 = np.argmax(cumulative_var >= 0.9) + 1 if any(cumulative_var >= 0.9) else len(cumulative_var)

    print(f"  前 3 个主成分解释方差：{explained_var[:3]}")
    print(f"  解释 80% 方差需要 {n_80} 个主成分")
    print(f"  解释 90% 方差需要 {n_90} 个主成分")

    # 步骤 4：生成可视化
    print("步骤 4：生成可视化...")
    setup_chinese_font()

    # 4.1 累积方差图
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(cumulative_var) + 1), cumulative_var * 100,
            'o-', linewidth=2, markersize=4, color='#3498db')
    ax.axhline(y=80, color='r', linestyle='--', linewidth=1.5, label='80% 阈值')
    ax.axhline(y=90, color='g', linestyle='--', linewidth=1.5, label='90% 阈值')
    ax.axvline(x=n_80, color='r', linestyle=':', alpha=0.5)
    ax.axvline(x=n_90, color='g', linestyle=':', alpha=0.5)
    ax.set_xlabel('主成分数量', fontsize=12)
    ax.set_ylabel('累积解释方差比例 (%)', fontsize=12)
    ax.set_title('PCA：累积方差解释', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(output_path / 'pca_cumulative_variance.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  保存: {output_path / 'pca_cumulative_variance.png'}")

    # 4.2 方差解释条形图（前 10 个主成分）
    fig, ax = plt.subplots(figsize=(10, 6))
    n_show = min(10, len(explained_var))
    ax.bar(range(1, n_show + 1), explained_var[:n_show] * 100,
           color='#3498db', alpha=0.7, edgecolor='black')
    ax.set_xlabel('主成分', fontsize=12)
    ax.set_ylabel('解释方差比例 (%)', fontsize=12)
    ax.set_title(f'前 {n_show} 个主成分的方差解释', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path / 'pca_explained_variance.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  保存: {output_path / 'pca_explained_variance.png'}")

    # 步骤 5：降维到 2D（用于可视化）
    print("步骤 5：降维到 2D...")
    pca_2d = PCA(n_components=2, random_state=random_state)
    X_2d = pca_2d.fit_transform(X_scaled)

    # 5.1 2D 散点图
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1],
                         c=range(len(X_2d)), cmap='viridis',
                         alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    ax.set_xlabel(f'PC1（{pca_2d.explained_variance_ratio_[0]:.1%} 方差）', fontsize=12)
    ax.set_ylabel(f'PC2（{pca_2d.explained_variance_ratio_[1]:.1%} 方差）', fontsize=12)
    ax.set_title('客户行为降维可视化（50 维 → 2 维）', fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='样本索引')
    plt.tight_layout()
    plt.savefig(output_path / 'pca_2d_scatter.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  保存: {output_path / 'pca_2d_scatter.png'}")

    # 步骤 6：载荷分析
    print("步骤 6：载荷分析...")
    loadings_df = pd.DataFrame(
        pca_full.components_[:5].T,
        index=feature_names,
        columns=[f'PC{i+1}' for i in range(5)]
    )

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
        'n_90': n_90,
        'feature_names': feature_names
    }


# ===== 生成 Markdown 报告 =====
def generate_pca_report(results: Dict, output_file: str = 'output/pca_report.md') -> str:
    """
    生成 PCA 分析的 Markdown 报告

    参数:
        results: pca_analysis 返回的结果字典
        output_file: 输出文件路径

    返回:
        Markdown 报告字符串
    """
    report_lines = []

    report_lines.append("# 降维分析：PCA 主成分分析\n")
    report_lines.append("## 方法概述\n")
    report_lines.append("- **方法**：主成分分析（Principal Component Analysis, PCA）")
    report_lines.append("- **目的**：将 50 个高维特征压缩为少数几个主成分，保留主要信息")
    report_lines.append(f"- **数据**：{len(results['X_scaled'])} 个样本，{len(results['feature_names'])} 个特征\n")

    report_lines.append("## 方差解释\n")
    report_lines.append("### 累积方差解释\n")
    report_lines.append(f"- **前 3 个主成分** 解释方差：")
    report_lines.append(f"  - PC1: {results['explained_var'][0]:.2%}")
    report_lines.append(f"  - PC2: {results['explained_var'][1]:.2%}")
    report_lines.append(f"  - PC3: {results['explained_var'][2]:.2%}")
    report_lines.append(f"  - 累积: {results['cumulative_var'][2]:.2%}\n")
    report_lines.append(f"- **解释 80% 方差** 需要 {results['n_80']} 个主成分")
    report_lines.append(f"- **解释 90% 方差** 需要 {results['n_90']} 个主成分\n")

    report_lines.append("### 可视化\n")
    report_lines.append(f"![累积方差解释](pca_cumulative_variance.png)\n")
    report_lines.append(f"![主成分方差解释](pca_explained_variance.png)\n")
    report_lines.append(f"![2D 降维可视化](pca_2d_scatter.png)\n")

    report_lines.append("## 主成分解释\n")
    report_lines.append("### 第一主成分 (PC1)\n")
    report_lines.append("**载荷最高的 5 个特征**：\n\n")

    loadings_df = results['loadings_df']
    pc1_top = loadings_df['PC1'].abs().sort_values(ascending=False).head(5)
    for feat in pc1_top.index:
        loading = loadings_df.loc[feat, 'PC1']
        report_lines.append(f"- {feat}: {loading:+.3f}")

    report_lines.append("\n**解释**：第一主成分主要反映...")
    report_lines.append("（需要根据载荷特征人工解释）\n")

    report_lines.append("### 第二主成分 (PC2)\n")
    report_lines.append("**载荷最高的 5 个特征**：\n\n")

    pc2_top = loadings_df['PC2'].abs().sort_values(ascending=False).head(5)
    for feat in pc2_top.index:
        loading = loadings_df.loc[feat, 'PC2']
        report_lines.append(f"- {feat}: {loading:+.3f}")

    report_lines.append("\n**解释**：第二主成分主要反映...")
    report_lines.append("（需要根据载荷特征人工解释）\n")

    report_lines.append("## 结论\n")
    report_lines.append(f"通过 PCA 降维，我们用 {results['n_80']} 个主成分")
    report_lines.append(f"保留了 80% 的原始信息（从 {len(results['feature_names'])} 维压缩）。\n")
    report_lines.append("这有助于：\n")
    report_lines.append("1. **降低计算复杂度**：高维模型训练更快")
    report_lines.append("2. **去除冗余信息**：特征间的相关性被消除")
    report_lines.append("3. **可视化数据结构**：2D/3D 散点图揭示潜在模式\n")

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
    print("StatLab PCA 分析模块")
    print("=" * 60)

    # 生成模拟数据
    print("\n生成模拟电商数据...")
    df, feature_names = generate_ecommerce_data(n_samples=1000)
    print(f"数据形状: {df.shape}")

    # 执行 PCA 分析
    results = pca_analysis(
        df=df,
        feature_names=feature_names,
        output_dir='output'
    )

    # 生成报告
    print("\n生成报告...")
    report = generate_pca_report(results, output_file='output/pca_report.md')

    # 打印报告预览
    print("\n" + "=" * 60)
    print("报告预览")
    print("=" * 60)
    print(report)

    return results


if __name__ == "__main__":
    main()

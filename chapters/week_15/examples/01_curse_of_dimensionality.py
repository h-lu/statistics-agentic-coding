"""
示例：维度灾难演示——高维空间中距离度量失效

本例演示维度灾难的核心现象：
1. 高维空间中数据变得极其稀疏
2. 点对之间的距离趋同（最近邻和最远邻差异变小）
3. 这导致基于距离的方法（如 KNN、K-means）在高维中失效

运行方式：python3 chapters/week_15/examples/01_curse_of_dimensionality.py

预期输出：
- 打印不同维度下的平均距离和距离变异系数
- 生成一张维度 vs 平均距离的折线图到 images/
- 生成一张维度 vs 距离变异系数的折线图到 images/
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.distance import pdist, cdist


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


# ===== 正例：计算维度灾难的指标 =====
def compute_dimension_statistics(dim: int, n_samples: int = 1000, random_state: int = 42) -> dict:
    """
    计算给定维度下的距离统计量

    参数:
        dim: 数据维度
        n_samples: 样本数量
        random_state: 随机种子

    返回:
        包含平均距离、最小/最大距离、变异系数的字典
    """
    rng = np.random.RandomState(random_state)
    points = rng.rand(n_samples, dim)

    # 计算所有点对之间的距离
    distances = pdist(points, metric='euclidean')

    # 变异系数 = 标准差 / 均值，衡量距离的相对离散程度
    # 变异系数越小，说明距离越趋同（最近邻和最远邻差异越小）
    cv = distances.std() / distances.mean() if distances.mean() > 0 else 0

    return {
        '维度': dim,
        '平均距离': distances.mean(),
        '最小距离': distances.min(),
        '最大距离': distances.max(),
        '距离标准差': distances.std(),
        '变异系数': cv,
        '距离范围': distances.max() - distances.min()
    }


# ===== 反例：高维中最近邻失去意义 =====
def demonstrate_nearest_neighbor_failure(n_samples: int = 100, random_state: int = 42) -> dict:
    """
    反例：展示高维中"最近邻"和"随机点"的距离差异变小

    在低维中，最近邻的距离明显小于随机点的距离
    在高维中，这个差异会消失——最近邻不再"近"
    """
    rng = np.random.RandomState(random_state)

    results = {}

    for dim in [1, 2, 5, 10, 20, 50, 100]:
        points = rng.rand(n_samples, dim)

        # 取第一个点作为查询点
        query_point = points[0:1, :]  # shape: (1, dim)

        # 计算查询点到其他所有点的距离
        distances = cdist(query_point, points[1:], metric='euclidean')[0]  # shape: (n_samples-1,)

        # 最近邻距离
        nearest_dist = distances.min()

        # 平均距离（作为"随机点"的参照）
        avg_dist = distances.mean()

        # 相对差异：最近邻比平均近多少？
        # 在低维中，这个值应该很小（最近邻明显更近）
        # 在高维中，这个值会接近 1（最近邻和平均点差不多远）
        relative_ratio = nearest_dist / avg_dist if avg_dist > 0 else 0

        results[dim] = {
            '最近邻距离': nearest_dist,
            '平均距离': avg_dist,
            '相对比率': relative_ratio
        }

    return results


# ===== 可视化：维度灾难 =====
def plot_curse_of_dimensionality(dimensions: list, avg_distances: list,
                                   cvs: list, output_dir: Path) -> None:
    """
    绘制维度灾难的两种表现

    参数:
        dimensions: 维度列表
        avg_distances: 各维度对应的平均距离
        cvs: 各维度对应的距离变异系数
        output_dir: 图片输出目录
    """
    setup_chinese_font()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 左图：平均距离随维度增加
    ax1.plot(dimensions, avg_distances, 'o-', linewidth=2, markersize=8,
             color='#e74c3c', label='平均距离')
    ax1.set_xlabel('维度', fontsize=12)
    ax1.set_ylabel('点对之间的平均距离', fontsize=12)
    ax1.set_title('维度灾难：维度越高，平均距离越大', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # 添加说明文字
    ax1.text(0.98, 0.95, '单位超立方体中\n随机点变得更"分散"',
             transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 右图：变异系数随维度增加（关键指标）
    ax2.plot(dimensions, cvs, 'o-', linewidth=2, markersize=8,
             color='#3498db', label='变异系数')
    ax2.set_xlabel('维度', fontsize=12)
    ax2.set_ylabel('距离变异系数 (标准差/均值)', fontsize=12)
    ax2.set_title('距离趋同：变异系数随维度下降', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 添加阈值线（经验值：变异系数 < 0.1 时距离基本失去区分度）
    ax2.axhline(y=0.1, color='r', linestyle='--', label='0.1 阈值（区分度失效）')
    ax2.legend()

    # 添加说明文字
    ax2.text(0.98, 0.95, '变异系数越小，\n最近邻和最远邻\n的差异越小',
             transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.tight_layout()
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / '01_curse_of_dimensionality.png', dpi=150,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"图片已保存到: {output_dir / '01_curse_of_dimensionality.png'}")


# ===== 可视化：最近邻失效 =====
def plot_nearest_neighbor_failure(results: dict, output_dir: Path) -> None:
    """
    绘制最近邻相对比率随维度的变化

    参数:
        results: 各维度的最近邻统计结果
        output_dir: 图片输出目录
    """
    setup_chinese_font()

    dimensions = list(results.keys())
    ratios = [results[d]['相对比率'] for d in dimensions]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(dimensions, ratios, 'o-', linewidth=2, markersize=8,
            color='#9b59b6', label='最近邻距离 / 平均距离')

    # 添加参考线（比率为 1 时表示最近邻和平均点一样远）
    ax.axhline(y=1.0, color='r', linestyle='--', label='比率 = 1（最近邻不再"近"）')

    ax.set_xlabel('维度', fontsize=12)
    ax.set_ylabel('最近邻距离 / 平均距离', fontsize=12)
    ax.set_title('维度灾难：高维中"最近邻"失去意义', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 添加关键洞察的标注
    for dim, ratio in zip(dimensions, ratios):
        if dim >= 20:
            ax.annotate(f'K={dim}: {ratio:.2f}',
                       xy=(dim, ratio),
                       xytext=(dim, ratio + 0.05),
                       fontsize=9,
                       ha='center',
                       arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5))

    plt.tight_layout()
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / '01_nearest_neighbor_failure.png', dpi=150,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"图片已保存到: {output_dir / '01_nearest_neighbor_failure.png'}")


# ===== 主函数 =====
def main() -> None:
    output_dir = Path(__file__).parent.parent / 'images'

    print("=" * 60)
    print("示例 1：维度灾难——高维空间的反直觉性质")
    print("=" * 60)

    # 测试不同维度
    dimensions = [1, 2, 5, 10, 20, 50, 100, 200]
    avg_distances = []
    cvs = []

    print("\n维度灾难的量化指标：")
    print("-" * 70)
    print(f"{'维度':>6} {'平均距离':>12} {'最小距离':>12} {'最大距离':>12} {'变异系数':>10}")
    print("-" * 70)

    for dim in dimensions:
        stats = compute_dimension_statistics(dim)
        avg_distances.append(stats['平均距离'])
        cvs.append(stats['变异系数'])

        print(f"{stats['维度']:>6} {stats['平均距离']:>12.4f} "
              f"{stats['最小距离']:>12.4f} {stats['最大距离']:>12.4f} "
              f"{stats['变异系数']:>10.4f}")

    print("\n关键洞察：")
    print("  1. 维度增加 → 平均距离增大（数据变得更稀疏）")
    print("  2. 维度增加 → 变异系数减小（距离趋同，最近邻不再'近'）")
    print(f"  3. 在 100 维时，变异系数约为 {cvs[-1]:.3f}，距离基本失去区分度")

    # 可视化维度灾难
    plot_curse_of_dimensionality(dimensions, avg_distances, cvs, output_dir)

    # 反例：最近邻失效
    print("\n" + "=" * 60)
    print("示例 2（反例）：高维中最近邻失去意义")
    print("=" * 60)

    nn_results = demonstrate_nearest_neighbor_failure()

    print("\n最近邻距离 vs 平均距离的比率：")
    print("-" * 60)
    print(f"{'维度':>6} {'最近邻距离':>14} {'平均距离':>14} {'比率':>10}")
    print("-" * 60)

    for dim, stats in nn_results.items():
        print(f"{dim:>6} {stats['最近邻距离']:>14.4f} "
              f"{stats['平均距离']:>14.4f} {stats['相对比率']:>10.4f}")

    print("\n关键洞察：")
    print("  在 1 维时，最近邻距离只有平均距离的 ~20%")
    print("  在 100 维时，最近邻距离接近平均距离的 80% 以上")
    print("  → 高维中'最近邻'不再比'随机点'近多少！")

    # 可视化最近邻失效
    plot_nearest_neighbor_failure(nn_results, output_dir)

    print("\n" + "=" * 60)
    print("结论：维度灾难的三个现实影响")
    print("=" * 60)
    print("1. 数据稀疏：需要指数级增长的样本才能'填满'高维空间")
    print("2. 距离失效：基于距离的方法（KNN、K-means）在高维中效果差")
    print("3. 过拟合风险：特征数 >> 样本数时，模型会记住噪声")
    print("\n应对方法：")
    print("  - 特征选择：从 p 个特征中选 k 个（k < p）")
    print("  - 降维：用 PCA 等方法将 p 维压缩到 k 维")
    print("  - 正则化：对模型添加约束，防止过拟合")


if __name__ == "__main__":
    main()

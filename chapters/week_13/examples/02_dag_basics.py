"""
示例：因果图（DAG）基础——混杂、碰撞、链式结构

本例演示因果图的三种基本结构：
1. 混杂（Confounding）：X 和 Y 的共同原因 Z，需要控制
2. 碰撞（Collision）：X 和 Y 的共同结果 Z，绝对不要控制
3. 链式（Chain/Mediation）：X 通过 M 影响 Y，取决于研究问题

运行方式：python3 chapters/week_13/examples/02_dag_basics.py
预期输出：stdout 输出三种结构的演示 + 保存因果图到 images/
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 尝试导入 networkx
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    print("警告: networkx 未安装，DAG 可视化将被跳过。请运行: pip install networkx")


def setup_chinese_font() -> str:
    """配置中文字体，返回使用的字体名称"""
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


# ============================================================================
# 结构1：混杂（Confounding）
# ============================================================================

def demonstrate_confounding(n_samples: int = 10000, random_state: int = 42) -> dict:
    """
    结构1：混杂

    定义：X 和 Y 的共同原因 Z
    示例：高价值客户（Z）→ 收到优惠券（X），高价值客户（Z）→ 低流失率（Y）

    问题：如果不控制 Z，X 和 Y 的相关可能是虚假的
    解决：控制混杂变量 Z（如果可观测）
    """
    print("=" * 60)
    print("结构1：混杂（Confounding）")
    print("=" * 60)

    rng = np.random.default_rng(random_state)

    # 混杂变量：高价值客户
    Z = rng.binomial(1, 0.3, n_samples)  # 30% 是高价值客户

    # Z 同时影响 X 和 Y
    # 高价值客户更容易收到优惠券
    X = rng.binomial(1, np.where(Z == 1, 0.8, 0.2))

    # 高价值客户流失率更低（优惠券 X 没有因果效应）
    Y = rng.binomial(1, np.where(Z == 1, 0.10, 0.30))

    df = pd.DataFrame({'Z': Z, 'X': X, 'Y': Y})

    # 计算关联（未控制 Z）
    association_biased = df[df['X'] == 1]['Y'].mean() - df[df['X'] == 0]['Y'].mean()

    # 计算关联（控制 Z）
    # 在 Z=0 和 Z=1 的子群体中分别计算，然后加权平均
    effect_z0 = df[df['Z'] == 0].groupby('X')['Y'].mean().diff().dropna().values[0]
    effect_z1 = df[df['Z'] == 1].groupby('X')['Y'].mean().diff().dropna().values[0]

    n_z0 = (df['Z'] == 0).sum()
    n_z1 = (df['Z'] == 1).sum()
    effect_adjusted = (effect_z0 * n_z0 + effect_z1 * n_z1) / (n_z0 + n_z1)

    print(f"\n因果结构：Z → X, Z → Y（混杂）")
    print(f"示例：高价值客户 → 收到优惠券，高价值客户 → 低流失率")

    print(f"\n观察到的关联（未控制 Z）：{association_biased:.3f}")
    print(f"  收到优惠券的客户流失率更低？是的")
    print(f"  但这不是因果效应！")

    print(f"\n真实的因果效应（控制 Z）：{effect_adjusted:.3f}")
    print(f"  优惠券没有因果效应（接近 0）")

    print(f"\n结论：")
    print(f"  - 混杂会导致虚假相关")
    print(f"  - 需要控制混杂变量 Z")
    print(f"  - 控制后，关联 = 因果（如果没有其他混杂）")

    return {
        'structure': 'confounding',
        'biased_association': association_biased,
        'true_effect': effect_adjusted,
        'lesson': '控制混杂变量，否则虚假相关'
    }


def draw_confounding_dag(output_dir: Path) -> None:
    """绘制混杂结构的 DAG"""
    if not NETWORKX_AVAILABLE:
        return

    G = nx.DiGraph()

    # 添加节点和边
    G.add_edges_from([
        ('Z (混杂)', 'X (Treatment)'),
        ('Z (混杂)', 'Y (Outcome)')
    ])

    fig, ax = plt.subplots(figsize=(8, 5))
    pos = nx.spring_layout(G, seed=42)

    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='lightcoral', ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=12, ax=ax)
    nx.draw_networkx_edges(G, pos, arrowsize=20, arrowstyle='->', ax=ax)

    ax.set_title('混杂结构（Confounding）\n需要控制 Z', fontsize=14, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / 'dag_confounding.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print(f"\nDAG 已保存: {output_dir / 'dag_confounding.png'}")


# ============================================================================
# 结构2：碰撞（Collider）
# ============================================================================

def demonstrate_collider(n_samples: int = 10000, random_state: int = 42) -> dict:
    """
    结构2：碰撞

    定义：X 和 Y 的共同结果 Z
    示例：优惠券（X）→ 客户满意（Z），高消费（Y）→ 客户满意（Z）

    问题：如果控制 Z，X 和 Y 会产生虚假关联（原本独立的 X 和 Y 变相关了）
    解决：绝对不要控制碰撞变量！
    """
    print("\n" + "=" * 60)
    print("结构2：碰撞（Collider）")
    print("=" * 60)

    rng = np.random.default_rng(random_state)

    # X 和 Y 独立
    X = rng.binomial(1, 0.5, n_samples)  # 是否收到优惠券
    Y = rng.binomial(1, 0.5, n_samples)  # 是否高消费

    # Z 是 X 和 Y 的碰撞变量（共同结果）
    # 如果 X=1 或 Y=1，Z 更可能是 1
    Z_prob = 0.2 + 0.4 * X + 0.4 * Y
    Z = rng.binomial(1, Z_prob)

    df = pd.DataFrame({'X': X, 'Y': Y, 'Z': Z})

    # 计算关联（未控制 Z）
    corr_uncond = df[['X', 'Y']].corr().iloc[0, 1]

    # 计算关联（控制 Z）
    corr_cond_z0 = df[df['Z'] == 0][['X', 'Y']].corr().iloc[0, 1] if (df['Z'] == 0).sum() > 10 else 0
    corr_cond_z1 = df[df['Z'] == 1][['X', 'Y']].corr().iloc[0, 1] if (df['Z'] == 1).sum() > 10 else 0

    print(f"\n因果结构：X → Z ← Y（碰撞）")
    print(f"示例：优惠券 → 客户满意 ← 高消费")

    print(f"\nX 和 Y 的相关性：")
    print(f"  未控制 Z：{corr_uncond:.3f}（接近 0，独立）")
    print(f"  控制 Z=0：{corr_cond_z0:.3f}")
    print(f"  控制 Z=1：{corr_cond_z1:.3f}（出现负相关！）")

    print(f"\n为什么控制 Z 会产生虚假相关？")
    print(f"  如果客户满意（Z=1）但没收到优惠券（X=0），那一定是高消费（Y=1）")
    print(f"  如果客户满意（Z=1）但低消费（Y=0），那一定收到了优惠券（X=1）")
    print(f"  这就导致了 X 和 Y 在控制 Z 后出现负相关")

    print(f"\n结论：")
    print(f"  - 碰撞变量会引入虚假相关（如果控制）")
    print(f"  - 绝对不要控制碰撞变量！")
    print(f"  - '控制一切'是错误策略")

    return {
        'structure': 'collider',
        'correlation_uncontrolled': corr_uncond,
        'correlation_controlled': corr_cond_z1,
        'lesson': '不要控制碰撞变量，否则虚假相关'
    }


def draw_collider_dag(output_dir: Path) -> None:
    """绘制碰撞结构的 DAG"""
    if not NETWORKX_AVAILABLE:
        return

    G = nx.DiGraph()

    # 添加节点和边
    G.add_edges_from([
        ('X (Treatment)', 'Z (碰撞变量)'),
        ('Y (Outcome)', 'Z (碰撞变量)')
    ])

    fig, ax = plt.subplots(figsize=(8, 5))
    pos = nx.spring_layout(G, seed=42)

    # 绘制节点（碰撞变量用不同颜色）
    colors = ['lightblue', 'lightblue', 'lightgreen']
    nodes = list(G.nodes())
    nx.draw_networkx_nodes(G, pos, node_size=2000, nodelist=[nodes[2]],
                           node_color='lightgreen', ax=ax)
    nx.draw_networkx_nodes(G, pos, node_size=2000, nodelist=[nodes[0], nodes[1]],
                           node_color='lightblue', ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=12, ax=ax)
    nx.draw_networkx_edges(G, pos, arrowsize=20, arrowstyle='->', ax=ax)

    ax.set_title('碰撞结构（Collider）\n绝对不要控制 Z！', fontsize=14, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / 'dag_collider.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print(f"DAG 已保存: {output_dir / 'dag_collider.png'}")


# ============================================================================
# 结构3：链式/中介（Chain/Mediation）
# ============================================================================

def demonstrate_mediation(n_samples: int = 10000, random_state: int = 42) -> dict:
    """
    结构3：链式/中介

    定义：X 通过 M 影响 Y
    示例：优惠券（X）→ 购买次数增加（M）→ 流失率降低（Y）

    问题：如果控制 M，X 对 Y 的直接效应会消失（因为 X 是通过 M 起作用的）
    取决于研究问题：
      - 想知道总效应：不要控制 M
      - 想知道直接效应：控制 M
      - 想知道间接效应：总效应 - 直接效应
    """
    print("\n" + "=" * 60)
    print("结构3：链式/中介（Chain/Mediation）")
    print("=" * 60)

    rng = np.random.default_rng(random_state)

    # X 影响 M，M 影响 Y
    X = rng.binomial(1, 0.5, n_samples)  # 是否收到优惠券
    M_prob = 0.3 + 0.4 * X  # 优惠券增加购买次数
    M = rng.binomial(1, M_prob)  # 是否增加购买次数

    Y_prob = 0.30 - 0.15 * M  # 购买次数增加降低流失率
    Y = rng.binomial(1, Y_prob)  # 是否流失

    df = pd.DataFrame({'X': X, 'M': M, 'Y': Y})

    # 计算总效应（未控制 M）
    total_effect = df[df['X'] == 1]['Y'].mean() - df[df['X'] == 0]['Y'].mean()

    # 计算直接效应（控制 M）
    direct_effect_z0 = df[df['M'] == 0].groupby('X')['Y'].mean().diff().dropna().values[0] if (df['M'] == 0).sum() > 10 else 0
    direct_effect_z1 = df[df['M'] == 1].groupby('X')['Y'].mean().diff().dropna().values[0] if (df['M'] == 1).sum() > 10 else 0

    n_m0 = (df['M'] == 0).sum()
    n_m1 = (df['M'] == 1).sum()
    direct_effect = (direct_effect_z0 * n_m0 + direct_effect_z1 * n_m1) / (n_m0 + n_m1)

    indirect_effect = total_effect - direct_effect

    print(f"\n因果结构：X → M → Y（中介）")
    print(f"示例：优惠券 → 购买次数增加 → 流失率降低")

    print(f"\n因果效应分解：")
    print(f"  总效应（未控制 M）：{total_effect:.3f}")
    print(f"  直接效应（控制 M）：{direct_effect:.3f}")
    print(f"  间接效应（通过 M）：{indirect_effect:.3f}")

    print(f"\n结论：")
    print(f"  - 优惠券通过增加购买次数降低流失率")
    print(f"  - 想知道总效应：不要控制 M")
    print(f"  - 想知道直接效应：控制 M")
    print(f"  - 想知道间接效应：总效应 - 直接效应")

    return {
        'structure': 'mediation',
        'total_effect': total_effect,
        'direct_effect': direct_effect,
        'indirect_effect': indirect_effect,
        'lesson': '是否控制中介取决于研究问题'
    }


def draw_mediation_dag(output_dir: Path) -> None:
    """绘制中介结构的 DAG"""
    if not NETWORKX_AVAILABLE:
        return

    G = nx.DiGraph()

    # 添加节点和边
    G.add_edges_from([
        ('X (Treatment)', 'M (中介)'),
        ('M (中介)', 'Y (Outcome)')
    ])

    fig, ax = plt.subplots(figsize=(8, 5))
    pos = nx.spring_layout(G, seed=42)

    # 绘制节点（中介变量用不同颜色）
    nodes = list(G.nodes())
    nx.draw_networkx_nodes(G, pos, node_size=2000, nodelist=[nodes[1]],
                           node_color='lightyellow', ax=ax)
    nx.draw_networkx_nodes(G, pos, node_size=2000, nodelist=[nodes[0], nodes[2]],
                           node_color='lightblue', ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=12, ax=ax)
    nx.draw_networkx_edges(G, pos, arrowsize=20, arrowstyle='->', ax=ax)

    ax.set_title('中介结构（Mediation）\nX 通过 M 影响 Y', fontsize=14, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / 'dag_mediation.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print(f"DAG 已保存: {output_dir / 'dag_mediation.png'}")


# ============================================================================
# 坏例子：控制错误的变量
# ============================================================================

def bad_example_wrong_control(n_samples: int = 10000, random_state: int = 42) -> dict:
    """
    坏例子：控制了碰撞变量

    场景：优惠券（X）和流失率（Y）原本独立
    但如果你控制了客户满意（Z，碰撞变量），X 和 Y 会产生虚假相关
    """
    print("\n" + "=" * 60)
    print("坏例子：控制了碰撞变量")
    print("=" * 60)

    rng = np.random.default_rng(random_state)

    # X 和 Y 独立（优惠券对流失率没有因果效应）
    X = rng.binomial(1, 0.5, n_samples)
    Y = rng.binomial(1, 0.3, n_samples)

    # Z 是碰撞变量
    Z_prob = 0.2 + 0.3 * X + 0.3 * (1 - Y)  # X=1 或 Y=0 时 Z 更可能=1
    Z = rng.binomial(1, Z_prob)

    df = pd.DataFrame({'X': X, 'Y': Y, 'Z': Z})

    # 未控制 Z 的关联
    effect_uncontrolled = df.groupby('X')['Y'].mean().diff().dropna().values[0]

    # 控制 Z 的关联（错误！）
    effect_z0 = df[df['Z'] == 0].groupby('X')['Y'].mean().diff().dropna().values[0] if (df['Z'] == 0).sum() > 10 else 0
    effect_z1 = df[df['Z'] == 1].groupby('X')['Y'].mean().diff().dropna().values[0] if (df['Z'] == 1).sum() > 10 else 0

    print(f"\n场景：优惠券（X）对流失率（Y）没有因果效应")
    print(f"但客户满意（Z）是碰撞变量：X → Z ← Y")

    print(f"\n未控制 Z 的关联：{effect_uncontrolled:.3f}（接近 0，正确）")
    print(f"控制 Z=0 后的关联：{effect_z0:.3f}")
    print(f"控制 Z=1 后的关联：{effect_z1:.3f}（虚假相关！）")

    print(f"\n错误结论：控制 Z 后，优惠券看起来'增加'了流失率")
    print(f"真相：优惠券没有因果效应")

    print(f"\n教训：")
    print(f"  - 不是所有变量都应该控制")
    print(f"  - 控制碰撞变量会引入虚假相关")
    print(f"  - 需要用因果图判断该控制什么")

    return {
        'true_effect': 0.0,
        'biased_effect': effect_z1,
        'lesson': '控制碰撞变量会破坏因果识别'
    }


# ============================================================================
# 主函数
# ============================================================================

def main() -> None:
    """运行所有演示"""
    font = setup_chinese_font()

    print("\n" + "=" * 60)
    print("因果图（DAG）三种基本结构")
    print("=" * 60)

    # 创建输出目录
    output_dir = Path(__file__).parent.parent / 'images'
    output_dir.mkdir(parents=True, exist_ok=True)

    # 结构1：混杂
    confounding_results = demonstrate_confounding()
    draw_confounding_dag(output_dir)

    # 结构2：碰撞
    collider_results = demonstrate_collider()
    draw_collider_dag(output_dir)

    # 结构3：中介
    mediation_results = demonstrate_mediation()
    draw_mediation_dag(output_dir)

    # 坏例子
    bad_results = bad_example_wrong_control()

    # 总结
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)

    print("\n因果图的三种基本结构：")
    print("  1. 混杂（Confounding）：需要控制")
    print("  2. 碰撞（Collider）：绝对不要控制")
    print("  3. 链式/中介（Chain）：取决于研究问题")

    print("\n关键教训：")
    print("  - '控制一切'是错误策略")
    print("  - 需要用因果图判断该控制什么，不该控制什么")
    print("  - 因果图必须基于领域知识，不是从数据中发现的")

    print(f"\n生成的图表：")
    print(f"  - {output_dir / 'dag_confounding.png'}")
    print(f"  - {output_dir / 'dag_collider.png'}")
    print(f"  - {output_dir / 'dag_mediation.png'}")


if __name__ == "__main__":
    main()

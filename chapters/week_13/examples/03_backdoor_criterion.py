"""
示例：d-分离与后门准则——如何识别因果路径

本例演示：
1. d-分离：判断两个变量是否在给定条件下独立
2. 后门准则：判断控制哪些变量可以识别因果效应
3. 用因果图分析"优惠券 → 流失率"的因果路径

运行方式：python3 chapters/week_13/examples/03_backdoor_criterion.py
预期输出：stdout 输出 d-分离和后门准则的演示
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple


# ============================================================================
# d-分离：判断独立性
# ============================================================================

def explain_d_separation() -> None:
    """
    d-分离（d-separation）：判断两个变量是否在给定条件下独立

    直觉：d-分离 = "因果路径被阻断"

    三种基本结构的 d-分离规则：
    1. 链式结构（A → B → C）：控制 B，A 和 C 独立
    2. 混杂结构（A ← B → C）：控制 B，A 和 C 独立
    3. 碰撞结构（A → B ← C）：控制 B，A 和 C 产生关联（路径被"打开"）
    """
    print("=" * 60)
    print("d-分离：判断独立性")
    print("=" * 60)

    print("\n什么是 d-分离？")
    print("  d-separation = '因果路径被阻断'")
    print("  如果 X 和 Y 被 d-分离，它们在给定条件下独立")

    print("\n三种基本结构的 d-分离规则：")

    print("\n1. 链式结构（A → B → C）：")
    print("   未控制 B：A 和 C 相关（路径通）")
    print("   控制 B：A 和 C 独立（路径被阻断）")

    print("\n2. 混杂结构（A ← B → C）：")
    print("   未控制 B：A 和 C 相关（虚假相关，路径通）")
    print("   控制 B：A 和 C 独立（路径被阻断）")

    print("\n3. 碰撞结构（A → B ← C）：")
    print("   未控制 B：A 和 C 独立（路径本来就是阻断的）")
    print("   控制 B：A 和 C 产生关联（路径被'打开'了！）")

    print("\n关键直觉：")
    print("  - 控制中介变量（B）会阻断路径")
    print("  - 控制混杂变量（B）会阻断路径")
    print("  - 控制碰撞变量（B）会打开路径（产生虚假相关）")


# ============================================================================
# 后门准则：识别因果效应
# ============================================================================

def explain_backdoor_criterion() -> None:
    """
    后门准则（Backdoor Criterion）：判断控制哪些变量可以识别因果效应

    定义：对于 X → Y，如果满足以下条件，则控制变量集 Z 可以识别因果效应：
    1. Z 没有包含 X 的后代（不是中介或碰撞变量）
    2. 控制 Z 后，X 和 Y 之间的所有"后门路径"都被阻断（d-分离）

    后门路径：X ← ... → Y（从 X 出发，箭头指向 X 的路径）
    前门路径：X → ... → Y（从 X 出发，箭头背离 X 的路径）
    """
    print("\n" + "=" * 60)
    print("后门准则：识别因果效应")
    print("=" * 60)

    print("\n什么是后门准则？")
    print("  用于判断：控制哪些变量可以识别因果效应")

    print("\n两个条件：")
    print("  1. Z 没有包含 X 的后代（不是中介或碰撞变量）")
    print("  2. 控制 Z 后，X 和 Y 之间的所有后门路径都被阻断")

    print("\n什么是后门路径？")
    print("  后门路径：X ← ... → Y（箭头指向 X）")
    print("  前门路径：X → ... → Y（箭头背离 X）")

    print("\n为什么要阻断后门路径？")
    print("  后门路径是混杂路径，会导致虚假相关")
    print("  只有阻断所有后门路径，X 和 Y 的关联才是因果效应")

    print("\n示例：优惠券（X）→ 流失率（Y）")
    print("  后门路径：X ← 高价值客户 → Y")
    print("  需要控制：高价值客户")
    print("  前门路径：X → 购买次数 → Y（这是因果机制，不要阻断）")


# ============================================================================
# 实践：分析"优惠券 → 流失率"的因果路径
# ============================================================================

class CausalPathAnalyzer:
    """
    简化的因果路径分析器

    这是一个教学演示，实际应用应使用 dowhy 或类似库
    """

    def __init__(self):
        # 定义因果图（边的列表）
        self.edges = []

    def add_edge(self, from_node: str, to_node: str) -> None:
        """添加有向边"""
        self.edges.append((from_node, to_node))

    def get_parents(self, node: str) -> List[str]:
        """获取父节点（指向该节点的节点）"""
        return [src for src, dst in self.edges if dst == node]

    def get_children(self, node: str) -> List[str]:
        """获取子节点（该节点指向的节点）"""
        return [dst for src, dst in self.edges if src == node]

    def get_ancestors(self, node: str) -> Set[str]:
        """获取祖先节点（递归）"""
        ancestors = set()
        queue = self.get_parents(node)

        while queue:
            current = queue.pop(0)
            if current not in ancestors:
                ancestors.add(current)
                queue.extend(self.get_parents(current))

        return ancestors

    def get_descendants(self, node: str) -> Set[str]:
        """获取后代节点（递归）"""
        descendants = set()
        queue = self.get_children(node)

        while queue:
            current = queue.pop(0)
            if current not in descendants:
                descendants.add(current)
                queue.extend(self.get_children(current))

        return descendants

    def is_collider(self, node: str, path: List[str]) -> bool:
        """
        判断节点在路径中是否是碰撞变量

        碰撞变量：两个箭头指向的节点（... → node ← ...）
        """
        if len(path) < 3:
            return False

        idx = path.index(node)
        if idx == 0 or idx == len(path) - 1:
            return False

        # 检查 node 的两边是否都是箭头指向它
        prev_node = path[idx - 1]
        next_node = path[idx + 1]

        arrow_from_prev = (prev_node, node) in self.edges
        arrow_from_next = (next_node, node) in self.edges

        return arrow_from_prev and arrow_from_next

    def find_backdoor_paths(self, treatment: str, outcome: str) -> List[List[str]]:
        """
        找到所有后门路径（简化版，只找长度 <= 3 的路径）

        后门路径：从 treatment 出发，第一个箭头指向 treatment
        """
        backdoor_paths = []

        # 获取 treatment 的所有父节点（混杂）
        parents = self.get_parents(treatment)

        for parent in parents:
            # 检查 parent 是否能到达 outcome
            if parent == outcome:
                backdoor_paths.append([treatment, parent, outcome])
            else:
                # 检查 parent 的子节点
                for child in self.get_children(parent):
                    if child == outcome:
                        backdoor_paths.append([treatment, parent, outcome])
                    # 检查是否有更长的路径（简化版只处理三层）

        return backdoor_paths

    def identify_confounders(self, treatment: str, outcome: str) -> Dict[str, List[str]]:
        """
        识别需要控制的混杂变量

        返回：
        - confounders: 需要控制的混杂变量
        - colliders: 绝对不要控制的碰撞变量
        - mediators: 中介变量（是否控制取决于研究问题）
        """
        # 找后门路径
        backdoor_paths = self.find_backdoor_paths(treatment, outcome)

        # 需要控制的混杂变量
        confounders = set()
        for path in backdoor_paths:
            if len(path) >= 2:
                confounders.add(path[1])  # 中间节点

        # 后代变量（包括中介和碰撞）
        descendants = self.get_descendants(treatment)

        # 排除 outcome 本身
        descendants.discard(outcome)

        # 简化：假设所有后代都是中介（实际需要更复杂的判断）
        mediators = descendants

        return {
            'confounders': list(confounders),
            'mediators': list(mediators),
            'backdoor_paths': backdoor_paths
        }


def analyze_coupon_churn_example() -> None:
    """
    分析"优惠券 → 流失率"的因果图

    因果图：
    - 高价值客户 → 优惠券
    - 高价值客户 → 流失率
    - 优惠券 → 购买次数
    - 购买次数 → 流失率
    - VIP 身份 → 流失率
    """
    print("\n" + "=" * 60)
    print("实践：分析'优惠券 → 流失率'的因果路径")
    print("=" * 60)

    # 创建因果图
    analyzer = CausalPathAnalyzer()

    # 添加边（基于领域知识）
    analyzer.add_edge('high_value_customer', 'coupon')
    analyzer.add_edge('high_value_customer', 'churn')
    analyzer.add_edge('coupon', 'purchase_count')
    analyzer.add_edge('purchase_count', 'churn')
    analyzer.add_edge('vip_status', 'churn')

    print("\n因果图（基于领域知识）：")
    print("  high_value_customer → coupon")
    print("  high_value_customer → churn")
    print("  coupon → purchase_count → churn")
    print("  vip_status → churn")

    # 找后门路径
    print("\n分析：coupon → churn")
    backdoor_paths = analyzer.find_backdoor_paths('coupon', 'churn')
    print(f"\n后门路径（混杂路径）：")
    for i, path in enumerate(backdoor_paths, 1):
        print(f"  路径 {i}: {' → '.join(path)}")

    # 识别混杂变量
    identification = analyzer.identify_confounders('coupon', 'churn')

    print(f"\n需要控制的混杂变量：")
    for confounder in identification['confounders']:
        print(f"  - {confounder}")

    print(f"\n中介变量（是否控制取决于研究问题）：")
    for mediator in identification['mediators']:
        print(f"  - {mediator}")

    print("\n结论：")
    print("  1. 需要控制 high_value_customer（混杂）")
    print("  2. purchase_count 是中介变量")
    print("     - 想知道总效应：不要控制")
    print("     - 想知道直接效应：控制")
    print("  3. vip_status 是混杂变量，建议控制")

    print("\n问题：high_value_customer 可能不可观测！")
    print("  解决方案：")
    print("    - 用代理变量（如 purchase_count, vip_status）")
    print("    - 但代理不完美，剩余混杂仍存在")
    print("    - 最优方法：随机对照试验（RCT）")


# ============================================================================
# 坏例子：控制碰撞变量
# ============================================================================

def bad_example_controlling_collider(n_samples: int = 10000, random_state: int = 42) -> None:
    """
    坏例子：控制了碰撞变量

    场景：你想分析 X 对 Y 的因果效应，但你控制了碰撞变量 Z
    结果：X 和 Y 产生虚假相关
    """
    print("\n" + "=" * 60)
    print("坏例子：控制了碰撞变量")
    print("=" * 60)

    rng = np.random.default_rng(random_state)

    # X 和 Y 独立
    X = rng.binomial(1, 0.5, n_samples)
    Y = rng.binomial(1, 0.3, n_samples)

    # Z 是碰撞变量（X 和 Y 的共同结果）
    Z_prob = 0.2 + 0.3 * X + 0.3 * (1 - Y)
    Z = rng.binomial(1, Z_prob)

    df = pd.DataFrame({'X': X, 'Y': Y, 'Z': Z})

    # 未控制 Z 的关联（正确）
    effect_uncontrolled = df.groupby('X')['Y'].mean().diff().dropna().values[0]

    # 控制 Z 的关联（错误！）
    if (df['Z'] == 1).sum() > 10:
        effect_controlled = df[df['Z'] == 1].groupby('X')['Y'].mean().diff().dropna().values[0]
    else:
        effect_controlled = 0

    print(f"\n场景：X 对 Y 没有因果效应（它们独立）")
    print(f"但 Z 是碰撞变量：X → Z ← Y")

    print(f"\n结果：")
    print(f"  未控制 Z 的关联：{effect_uncontrolled:.3f}（接近 0，正确）")
    print(f"  控制 Z 后的关联：{effect_controlled:.3f}（虚假相关！）")

    print(f"\n错误：控制 Z 后，X 看起来对 Y 有影响")
    print(f"真相：X 和 Y 独立，Z 是碰撞变量")

    print(f"\n教训：")
    print(f"  - 控制碰撞变量会打开虚假路径")
    print(f"  - 需要用 d-分离判断该控制什么")
    print(f"  - '控制一切'是错误策略")


# ============================================================================
# 主函数
# ============================================================================

def main() -> None:
    """运行所有演示"""
    print("\n" + "=" * 60)
    print("d-分离与后门准则演示")
    print("=" * 60)

    # 1. 解释 d-分离
    explain_d_separation()

    # 2. 解释后门准则
    explain_backdoor_criterion()

    # 3. 实践：分析优惠券案例
    analyze_coupon_churn_example()

    # 4. 坏例子
    bad_example_controlling_collider()

    # 总结
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)

    print("\n关键概念：")
    print("  1. d-分离：判断因果路径是否被阻断")
    print("     - 控制中介/混杂：阻断路径")
    print("     - 控制碰撞：打开路径（虚假相关）")

    print("\n  2. 后门准则：识别需要控制的混杂")
    print("     - 后门路径：混杂路径（需要阻断）")
    print("     - 前门路径：因果机制（不要阻断）")

    print("\n  3. 实践步骤：")
    print("     a. 画因果图（基于领域知识）")
    print("     b. 找所有后门路径")
    print("     c. 识别需要控制的混杂")
    print("     d. 不要控制碰撞变量！")
    print("     e. 是否控制中介取决于研究问题")

    print("\n推荐工具：")
    print("  - dowhy: Python 因果推断库")
    print("  - causal-learn: 因果发现")
    print("  - networkx: DAG 可视化")


if __name__ == "__main__":
    main()

"""
Week 13 作业参考实现

本文件提供作业的基础部分参考实现，用于学习参考。
当你在作业中遇到困难时，可以查看此文件，但建议先独立完成。

包含内容：
1. 因果推断三层级的判断和区分
2. 因果图（DAG）的基本结构和识别
3. d-分离和后门准则的应用
4. RCT 假设的检查
5. 倾向得分匹配（PSM）的实现

注意：这只是基础实现，不包含进阶部分和挑战部分。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors


# ============================================================================
# 第一部分：因果推断三层级
# ============================================================================

def classify_causal_question(question: str) -> str:
    """
    判断因果问题属于哪个层级

    参数:
    - question: 因果问题

    返回:
    - str: 'association', 'intervention', 或 'counterfactual'
    """
    question_lower = question.lower()

    # 关键词判断
    association_keywords = ['相关', '关联', 'correlation', 'associated', 'related']
    intervention_keywords = ['如果', '会怎样', 'if', 'would', 'effect', 'impact']
    counterfactual_keywords = ['如果当时', '如果不', 'would have', 'had been', 'counterfactual']

    for keyword in counterfactual_keywords:
        if keyword in question_lower:
            return 'counterfactual'

    for keyword in intervention_keywords:
        if keyword in question_lower:
            return 'intervention'

    for keyword in association_keywords:
        if keyword in question_lower:
            return 'association'

    return 'unknown'


def demonstrate_causal_ladder() -> Dict[str, Dict[str, str]]:
    """
    演示因果推断三层级的区别

    返回:
    - dict: 每个层级的示例和说明
    """
    return {
        'association': {
            'question': '使用优惠券的客户流失率是否更低？',
            'notation': 'P(churn|coupon)',
            'method': '相关系数、回归系数、卡方检验',
            'limitation': '相关不等于因果，可能存在混杂'
        },
        'intervention': {
            'question': '如果发放优惠券，流失率会怎样变化？',
            'notation': 'P(churn|do(coupon))',
            'method': 'RCT、DID、IV、PSM',
            'limitation': '需要因果识别策略, 不能直接从观察数据得出'
        },
        'counterfactual': {
            'question': '如果当时没发放优惠券，这个客户会流失吗？',
            'notation': 'P(churn_coupon=0|coupon=1, churn=0)',
            'method': '结构因果模型（SCM）',
            'limitation': '需要估计个体潜在结果，无法完全验证'
        }
    }


# ============================================================================
# 第二部分：因果图（DAG）基本结构
# ============================================================================

class SimpleDAG:
    """简化的 DAG 类，用于教学演示"""

    def __init__(self):
        self.edges: List[Tuple[str, str]] = []
        self.nodes: Set[str] = set()

    def add_edge(self, from_node: str, to_node: str) -> None:
        """添加有向边"""
        self.edges.append((from_node, to_node))
        self.nodes.update([from_node, to_node])

    def get_parents(self, node: str) -> List[str]:
        """获取父节点（指向该节点的节点）"""
        return [src for src, dst in self.edges if dst == node]

    def get_children(self, node: str) -> List[str]:
        """获取子节点（该节点指向的节点）"""
        return [dst for src, dst in self.edges if src == node]

    def is_descendant(self, node: str, ancestor: str) -> bool:
        """判断 node 是否是 ancestor 的后代"""
        visited = set()
        queue = [ancestor]

        while queue:
            current = queue.pop(0)
            if current == node:
                return True
            if current in visited:
                continue
            visited.add(current)
            queue.extend(self.get_children(current))

        return False


def identify_structure(dag: SimpleDAG, node_x: str, node_y: str, node_z: str) -> str:
    """
    识别三个节点之间的因果结构

    参数:
    - dag: 因果图
    - node_x, node_y, node_z: 三个节点

    返回:
    - str: 'confounding', 'collider', 'chain', 或 'other'
    """
    # 获取边的关系
    x_to_z = (node_x, node_z) in dag.edges
    z_to_x = (node_z, node_x) in dag.edges
    y_to_z = (node_y, node_z) in dag.edges
    z_to_y = (node_z, node_y) in dag.edges

    x_to_y = (node_x, node_y) in dag.edges
    y_to_x = (node_y, node_x) in dag.edges

    # 混杂：X ← Z → Y
    if z_to_x and z_to_y:
        return 'confounding'

    # 碰撞：X → Z ← Y
    if x_to_z and y_to_z:
        return 'collider'

    # 链式：X → Z → Y 或 X ← Z ← Y
    if (x_to_z and z_to_y) or (z_to_x and y_to_z):
        return 'chain'

    return 'other'


def get_control_advice(structure: str) -> str:
    """
    根据结构类型给出控制建议

    参数:
    - structure: 结构类型

    返回:
    - str: 控制建议
    """
    advice = {
        'confounding': '需要控制该变量（混杂）',
        'collider': '绝对不要控制该变量（碰撞）',
        'chain': '取决于研究问题（中介变量）',
        'other': '需要进一步分析'
    }
    return advice.get(structure, '未知结构')


# ============================================================================
# 第三部分：d-分离和后门准则
# ============================================================================

def find_backdoor_paths(
    dag: SimpleDAG,
    treatment: str,
    outcome: str
) -> List[List[str]]:
    """
    找到所有后门路径（简化版，只找长度 <= 3 的路径）

    参数:
    - dag: 因果图
    - treatment: 处理变量
    - outcome: 结果变量

    返回:
    - list: 后门路径列表
    """
    backdoor_paths = []

    # 获取 treatment 的所有父节点（混杂）
    parents = dag.get_parents(treatment)

    for parent in parents:
        # 检查 parent 是否能到达 outcome
        if parent == outcome:
            backdoor_paths.append([treatment, parent, outcome])
        else:
            # 检查 parent 的子节点
            for child in dag.get_children(parent):
                if child == outcome:
                    backdoor_paths.append([treatment, parent, outcome])

    return backdoor_paths


def identify_confounders(
    dag: SimpleDAG,
    treatment: str,
    outcome: str
) -> List[str]:
    """
    识别需要控制的混杂变量

    参数:
    - dag: 因果图
    - treatment: 处理变量
    - outcome: 结果变量

    返回:
    - list: 需要控制的混杂变量列表
    """
    backdoor_paths = find_backdoor_paths(dag, treatment, outcome)
    confounders = set()

    for path in backdoor_paths:
        if len(path) >= 2:
            confounders.add(path[1])  # 中间节点

    return list(confounders)


# ============================================================================
# 第四部分：RCT 假设检查
# ============================================================================

def check_balance_assumption(
    df: pd.DataFrame,
    treatment_col: str,
    covariate_cols: List[str]
) -> pd.DataFrame:
    """
    检查 RCT 的随机化假设（基线平衡性）

    参数:
    - df: 数据
    - treatment_col: 处理变量列名
    - covariate_cols: 协变量列名列表

    返回:
    - DataFrame: 平衡性检验结果
    """
    from scipy import stats

    results = []

    for covariate in covariate_cols:
        treated = df[df[treatment_col] == 1][covariate]
        control = df[df[treatment_col] == 0][covariate]

        # t 检验
        t_stat, p_value = stats.ttest_ind(treated, control)

        # 标准化均值差异（Standardized Mean Difference）
        pooled_std = np.sqrt((treated.std()**2 + control.std()**2) / 2)
        smd = (treated.mean() - control.mean()) / pooled_std if pooled_std > 0 else 0

        results.append({
            'covariate': covariate,
            'treated_mean': treated.mean(),
            'control_mean': control.mean(),
            'smd': smd,
            'p_value': p_value,
            'balanced': abs(smd) < 0.1 and p_value > 0.05
        })

    return pd.DataFrame(results)


# ============================================================================
# 第五部分：倾向得分匹配（PSM）
# ============================================================================

def propensity_score_matching(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    covariate_cols: List[str],
    random_state: int = 42
) -> Dict:
    """
    倾向得分匹配（PSM）：估计因果效应

    参数:
    - df: 数据
    - treatment_col: 处理变量列名
    - outcome_col: 结果变量列名
    - covariate_cols: 协变量列名列表
    - random_state: 随机种子

    返回:
    - dict: 包含 ATE、匹配后结果等
    """
    # 第一步：估计倾向得分
    ps_model = LogisticRegression(max_iter=1000, random_state=random_state)
    ps_model.fit(df[covariate_cols], df[treatment_col])
    df['propensity_score'] = ps_model.predict_proba(df[covariate_cols])[:, 1]

    # 第二步：匹配（1:1 最近邻匹配）
    treated = df[df[treatment_col] == 1].copy()
    control = df[df[treatment_col] == 0].copy()

    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(control[['propensity_score']])
    distances, indices = nn.kneighbors(treated[['propensity_score']])

    matched_control = control.iloc[indices.flatten()].copy()

    # 第三步：计算平均处理效应（ATE）
    treated_outcome = treated[outcome_col].mean()
    matched_control_outcome = matched_control[outcome_col].mean()
    ate = treated_outcome - matched_control_outcome

    return {
        'ATE': ate,
        'treated_outcome': treated_outcome,
        'control_outcome': matched_control_outcome,
        'n_matched': len(treated),
        'propensity_score_model': ps_model,
        'matched_indices': indices.flatten()
    }


def check_matching_balance(
    df: pd.DataFrame,
    treatment_col: str,
    covariate_cols: List[str],
    matched_indices: np.ndarray
) -> pd.DataFrame:
    """
    检查匹配后的平衡性

    参数:
    - df: 数据
    - treatment_col: 处理变量列名
    - covariate_cols: 协变量列名列表
    - matched_indices: 匹配的对照组索引

    返回:
    - DataFrame: 平衡性检验结果
    """
    treated = df[df[treatment_col] == 1]
    matched_control = df.iloc[matched_indices]

    results = []

    for covariate in covariate_cols:
        treated_mean = treated[covariate].mean()
        control_mean = matched_control[covariate].mean()

        # 标准化均值差异
        treated_std = treated[covariate].std()
        control_std = matched_control[covariate].std()
        pooled_std = np.sqrt((treated_std**2 + control_std**2) / 2)
        smd = (treated_mean - control_mean) / pooled_std if pooled_std > 0 else 0

        results.append({
            'covariate': covariate,
            'smd': smd,
            'balanced': abs(smd) < 0.1
        })

    return pd.DataFrame(results)


# ============================================================================
# 演示主函数
# ============================================================================

def main() -> None:
    """
    演示如何使用上述函数

    这是一个完整的示例，展示从因果层级判断到 PSM 的流程
    """
    print("=" * 60)
    print("Week 13 作业参考实现演示")
    print("=" * 60)

    # 1. 因果推断三层级
    print("\n1. 因果推断三层级")
    print("-" * 40)

    questions = [
        "使用优惠券的客户流失率是否更低？",
        "如果发放优惠券，流失率会怎样变化？",
        "如果当时没发放优惠券，这个客户会流失吗？"
    ]

    for question in questions:
        level = classify_causal_question(question)
        print(f"\n问题: {question}")
        print(f"  层级: {level}")

    # 2. 因果图结构识别
    print("\n2. 因果图结构识别")
    print("-" * 40)

    dag = SimpleDAG()
    dag.add_edge('high_value', 'coupon')
    dag.add_edge('high_value', 'churn')
    dag.add_edge('coupon', 'purchase_count')
    dag.add_edge('purchase_count', 'churn')

    # 识别 high_value 的结构类型
    structure = identify_structure(dag, 'coupon', 'churn', 'high_value')
    advice = get_control_advice(structure)

    print(f"\n结构: coupon ← high_value → churn")
    print(f"  类型: {structure}")
    print(f"  建议: {advice}")

    # 识别 purchase_count 的结构类型
    structure = identify_structure(dag, 'coupon', 'churn', 'purchase_count')
    advice = get_control_advice(structure)

    print(f"\n结构: coupon → purchase_count → churn")
    print(f"  类型: {structure}")
    print(f"  建议: {advice}")

    # 3. 后门路径识别
    print("\n3. 后门路径识别")
    print("-" * 40)

    backdoor_paths = find_backdoor_paths(dag, 'coupon', 'churn')
    confounders = identify_confounders(dag, 'coupon', 'churn')

    print(f"\n后门路径: {backdoor_paths}")
    print(f"需要控制的混杂变量: {confounders}")

    # 4. 生成模拟数据
    print("\n4. 生成模拟数据")
    print("-" * 40)

    np.random.seed(42)
    n = 1000

    high_value = np.random.binomial(1, 0.3, n)
    coupon = np.random.binomial(1, np.where(high_value == 1, 0.7, 0.2))
    purchase_count = np.random.poisson(5, n) + coupon * 2
    churn_prob = 0.35 - 0.15 * high_value - 0.1 * (purchase_count > 5).astype(int) - 0.3 * coupon
    churn_prob = np.clip(churn_prob, 0, 1)
    churn = np.random.binomial(1, churn_prob)

    df = pd.DataFrame({
        'high_value': high_value,
        'coupon': coupon,
        'purchase_count': purchase_count,
        'churn': churn
    })

    print(f"\n数据规模: {df.shape[0]} 行")
    print(f"流失率: {df['churn'].mean():.2%}")
    print(f"优惠券占比: {df['coupon'].mean():.2%}")

    # 5. RCT 假设检查（演示）
    print("\n5. RCT 假设检查（演示）")
    print("-" * 40)

    # 假设数据是 RCT（虽然实际不是）
    balance_results = check_balance_assumption(
        df, 'coupon', ['high_value', 'purchase_count']
    )

    print("\n平衡性检验结果:")
    print(balance_results)

    # 6. 倾向得分匹配
    print("\n6. 倾向得分匹配")
    print("-" * 40)

    psm_results = propensity_score_matching(
        df,
        treatment_col='coupon',
        outcome_col='churn',
        covariate_cols=['high_value', 'purchase_count']
    )

    print(f"\nPSM 结果:")
    print(f"  实验组流失率: {psm_results['treated_outcome']:.3f}")
    print(f"  对照组流失率（匹配后）: {psm_results['control_outcome']:.3f}")
    print(f"  ATE: {psm_results['ATE']:.3f}")

    # 检查匹配后平衡性
    balance_after = check_matching_balance(
        df, 'coupon', ['high_value', 'purchase_count'],
        psm_results['matched_indices']
    )

    print("\n匹配后平衡性:")
    print(balance_after)

    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)
    print("\n你可以参考此文件完成作业，但建议先独立尝试。")


if __name__ == "__main__":
    main()

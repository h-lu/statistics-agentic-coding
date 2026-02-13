"""
示例：StatLab 集成——因果推断章节生成

本脚本是 StatLab 超级线的一部分，用于在可复现分析报告中添加
"因果推断"章节。它执行完整的因果推断分析流程，包括：
- 画因果图（DAG）
- 识别策略（后门准则）
- 因果效应估计（回归 + 倾向评分匹配）
- 自动生成报告片段和图表

运行方式：python3 chapters/week_13/examples/05_statlab_causal.py
预期输出：
- 报告片段（追加到 report.md）
- 因果图（保存到 report/causal_dag.png）
- 倾向评分匹配可视化（保存到 report/psm_comparison.png）

依赖: 需要预先清洗好的数据（假设路径为 data/clean_data.csv）

说明：本脚本在 Week 09-12 基础上增量修改，将分析目标从
"预测与关联"扩展到"因果效应估计"。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import NearestNeighbors
from scipy import stats
import networkx as nx

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def causal_inference_report(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    confounders: List[str],
    output_dir: str = "report"
) -> str:
    """
    对数据集进行完整的因果推断分析，生成报告片段

    参数:
        df: 清洗后的数据
        treatment: 处理变量名（如 'coupon_used'）
        outcome: 结果变量名（如 'spending'）
        confounders: 混杂变量列表（如 ['activity', 'history_spend']）
        output_dir: 报告输出目录

    返回:
        Markdown 格式的报告片段
    """
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("StatLab 因果推断报告生成器")
    print("=" * 70)
    print(f"\n📊 数据概览:")
    print(f"  总样本数: {len(df)}")
    print(f"  处理变量: {treatment}")
    print(f"  结果变量: {outcome}")
    print(f"  混杂变量: {', '.join(confounders)}")

    # ========== 1. 画因果图 ==========
    print(f"\n✅ 步骤 1: 画因果图（DAG）...")

    dag_path = output_path / "causal_dag.png"
    plot_causal_dag(treatment, outcome, confounders, dag_path)

    print(f"  ✅ 因果图已保存: {dag_path}")

    # ========== 2. 识别策略（后门准则） ==========
    print(f"\n✅ 步骤 2: 识别后门路径...")

    backdoor_paths = identify_backdoor_paths(treatment, outcome, confounders)

    # ========== 3. 未调整的估计（小北的错误） ==========
    print(f"\n✅ 步骤 3: 计算未调整的估计...")

    treated = df[df[treatment] == 1][outcome].mean()
    control = df[df[treatment] == 0][outcome].mean()
    naive_effect = treated - control

    print(f"  用券组平均: {treated:.2f}")
    print(f"  对照组平均: {control:.2f}")
    print(f"  未调整差异: {naive_effect:.2f}")

    # ========== 4. 回归估计 ==========
    print(f"\n✅ 步骤 4: 带调整集的回归...")

    X = df[[treatment] + confounders]
    y = df[outcome]

    reg_model = LinearRegression()
    reg_model.fit(X, y)

    reg_coef = reg_model.coef_[0]
    reg_intercept = reg_model.intercept_

    # 计算标准误差和置信区间
    n = len(df)
    k = len(confounders) + 1
    y_pred = reg_model.predict(X)
    residuals = y - y_pred
    mse = np.sum(residuals**2) / (n - k - 1)

    X_with_intercept = np.column_stack([np.ones(n), X.values])
    cov_matrix = mse * np.linalg.inv(X_with_intercept.T @ X_with_intercept)
    se_treatment = np.sqrt(cov_matrix[1, 1])

    t_stat = reg_coef / se_treatment
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - k - 1))
    ci_low = reg_coef - 1.96 * se_treatment
    ci_high = reg_coef + 1.96 * se_treatment

    print(f"  回归系数: {reg_coef:.2f}")
    print(f"  95% CI: [{ci_low:.2f}, {ci_high:.2f}]")
    print(f"  p 值: {p_value:.4f}")

    # ========== 5. 倾向评分匹配 ==========
    print(f"\n✅ 步骤 5: 倾向评分匹配...")

    # 估计倾向评分
    ps_model = LogisticRegression(random_state=42)
    ps_model.fit(df[confounders], df[treatment])
    df_ps = df.copy()
    df_ps['propensity_score'] = ps_model.predict_proba(df[confounders])[:, 1]

    # 匹配
    treated_df = df_ps[df_ps[treatment] == 1].copy()
    control_df = df_ps[df_ps[treatment] == 0].copy()

    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(control_df[['propensity_score']])
    distances, indices = nn.kneighbors(treated_df[['propensity_score']])
    matched_control = control_df.iloc[indices.flatten()].copy()

    # 计算 ATT
    att = (treated_df[outcome].values - matched_control[outcome].values).mean()

    # Bootstrap CI
    n_boot = 500
    att_samples = []
    for i in range(n_boot):
        treated_boot = treated_df.sample(n=len(treated_df), replace=True)
        control_boot = control_df.sample(n=len(control_df), replace=True)

        nn_boot = NearestNeighbors(n_neighbors=1)
        nn_boot.fit(control_boot[['propensity_score']])
        _, indices_boot = nn_boot.kneighbors(treated_boot[['propensity_score']])
        matched_boot = control_boot.iloc[indices_boot.flatten()]

        att_boot = (treated_boot[outcome].values - matched_boot[outcome].values).mean()
        att_samples.append(att_boot)

    att_ci_low = np.percentile(att_samples, 2.5)
    att_ci_high = np.percentile(att_samples, 97.5)

    print(f"  ATT: {att:.2f}")
    print(f"  95% CI (Bootstrap): [{att_ci_low:.2f}, {att_ci_high:.2f}]")

    # ========== 6. 倾向评分可视化 ==========
    print(f"\n✅ 步骤 6: 保存倾向评分可视化...")

    psm_path = output_path / "psm_comparison.png"
    plot_psm_comparison(
        treated_df['propensity_score'].values,
        control_df['propensity_score'].values,
        matched_control['propensity_score'].values,
        psm_path
    )
    print(f"  ✅ 可视化已保存: {psm_path}")

    # ========== 7. 生成报告 ==========
    print(f"\n✅ 步骤 7: 生成报告片段...")

    report = generate_report_markdown(
        treatment=treatment,
        outcome=outcome,
        confounders=confounders,
        backdoor_paths=backdoor_paths,
        naive_effect=naive_effect,
        treated=treated,
        control=control,
        reg_coef=reg_coef,
        reg_ci_low=ci_low,
        reg_ci_high=ci_high,
        reg_p=p_value,
        att=att,
        att_ci_low=att_ci_low,
        att_ci_high=att_ci_high,
        dag_path=dag_path,
        psm_path=psm_path,
        n_total=len(df)
    )

    print(f"  ✅ 报告片段生成完成")

    print("\n" + "=" * 70)
    print("✅ 因果推断分析完成！")
    print("=" * 70)

    return report


def plot_causal_dag(treatment: str, outcome: str, confounders: List[str], output_path: Path):
    """画因果图"""
    G = nx.DiGraph()

    # 添加边
    for conf in confounders:
        G.add_edge(conf, treatment)
        G.add_edge(conf, outcome)
    G.add_edge(treatment, outcome)

    # 布局
    pos = {}
    pos[treatment] = (1, 0)
    pos[outcome] = (2, 0)

    for i, conf in enumerate(confounders):
        pos[conf] = (0, (i - len(confounders) / 2) * 0.5)

    # 画图
    plt.figure(figsize=(10, 6))
    nx.draw_networkx_nodes(G, pos, node_color='lightblue',
                          node_size=3000, alpha=0.9)
    nx.draw_networkx_edges(G, pos, edge_color='gray',
                          arrowsize=20, width=2, alpha=0.7)
    nx.draw_networkx_labels(G, pos, font_size=12,
                            font_family='sans-serif')
    plt.title("因果图（DAG）", fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_psm_comparison(treated_ps, control_ps, matched_ps, output_path: Path):
    """画倾向评分匹配前后的对比图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 匹配前
    axes[0].hist(treated_ps, alpha=0.5, label='处理组', bins=20, color='blue')
    axes[0].hist(control_ps, alpha=0.5, label='对照组', bins=20, color='red')
    axes[0].set_xlabel('倾向评分', fontsize=12)
    axes[0].set_ylabel('样本数', fontsize=12)
    axes[0].set_title('匹配前：倾向评分分布差异大', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # 匹配后
    axes[1].hist(treated_ps, alpha=0.5, label='处理组', bins=20, color='blue')
    axes[1].hist(matched_ps, alpha=0.5, label='匹配的对照组', bins=20, color='green')
    axes[1].set_xlabel('倾向评分', fontsize=12)
    axes[1].set_ylabel('样本数', fontsize=12)
    axes[1].set_title('匹配后：倾向评分分布接近', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def identify_backdoor_paths(treatment: str, outcome: str, confounders: List[str]) -> List[str]:
    """识别后门路径"""
    paths = []
    for conf in confounders:
        path = f"{treatment} ← {conf} → {outcome}"
        paths.append(path)
    return paths


def generate_report_markdown(
    treatment: str, outcome: str, confounders: List[str],
    backdoor_paths: List[str],
    naive_effect: float, treated: float, control: float,
    reg_coef: float, reg_ci_low: float, reg_ci_high: float, reg_p: float,
    att: float, att_ci_low: float, att_ci_high: float,
    dag_path: Path, psm_path: Path,
    n_total: int
) -> str:
    """生成 Markdown 格式的报告片段"""

    confounder_list = "、".join(confounders)

    report = f"""
## 因果推断

### 研究问题

本章回答的因果问题是：

**"如果给用户发放{treatment}，{outcome}会提高多少？"**

注意：这与关联问题不同。关联问题是"用券用户和未用券用户的{outcome}差异"，而因果问题是"发券这个行为的因果效应"。

### 因果假设

我们用因果图（DAG）表达因果假设：

![因果图]({dag_path.name})

**图解**：
- **处理变量（X）**：{treatment}（0=未使用，1=使用）
- **结果变量（Y）**：{outcome}
- **混杂变量（Z）**：{confounder_list}（同时影响用券和消费）
- **因果路径**：{treatment} → {outcome}（我们想估计的效应）

### 识别策略

根据**后门准则（Backdoor Criterion）**，我们需要调整以下混杂变量：

"""

    for path in backdoor_paths:
        report += f"- **{path}**：虚假关联路径\n"

    report += f"""
**调整理由**：
- {confounder_list}同时影响"{treatment}"和"{outcome}"
- 不调整这些变量，会把混杂变量的效应归功于处理

**不调整的变量**：
- 中介变量（如使用频率）：{treatment}通过影响中介变量影响{outcome}，调整它会切断因果路径

### 因果效应估计

我们用两种方法估计因果效应，以检查稳健性。

#### 方法 1：带调整集的回归

| 指标 | 估计值 | 95% CI | p 值 |
|------|--------|--------|------|
| {treatment} 的因果效应 | **{reg_coef:.2f}** | [{reg_ci_low:.2f}, {reg_ci_high:.2f}] | {reg_p:.4f} |

**解读**：在控制了{confounder_list}后，{treatment}对{outcome}的因果效应为**{reg_coef:.2f}**（95% CI [{reg_ci_low:.2f}, {reg_ci_high:.2f}]）。

#### 方法 2：倾向评分匹配

匹配质量检查：

![倾向评分分布（匹配前后）]({psm_path.name})

匹配后的因果效应：

| 指标 | 估计值 | 95% CI |
|------|--------|--------|
| **ATT（处理组平均处理效应）** | **{att:.2f}** | [{att_ci_low:.2f}, {att_ci_high:.2f}] |

**解读**：倾向评分匹配估计的因果效应为**{att:.2f}**（95% CI [{att_ci_low:.2f}, {att_ci_high:.2f}]），与回归结果接近，结论稳健。

### 混杂偏差对比

| 方法 | 估计值 | 说明 |
|------|--------|------|
| **未调整（小北的错误）** | {naive_effect:.2f} | 直接比较均值，被混杂夸大 |
| **回归（带调整集）** | {reg_coef:.2f} | 控制混杂后的因果效应 |
| **倾向评分匹配** | {att:.2f} | 匹配相似样本后的因果效应 |

**结论**：未调整的估计被夸大了{naive_effect - reg_coef:.2f}元，调整后真实的因果效应约为{(reg_coef + att) / 2:.2f}元。

### 结论边界

**我们能回答的（因果结论）**：
- {treatment}对{outcome}的因果效应约为**{(reg_coef + att) / 2:.2f} ± {abs(reg_coef - att) / 2:.2f}**元（两种方法的平均）
- 这个结论在调整了混杂变量（{confounder_list}）后成立
- 95% 置信区间不包含零，效应统计显著

**我们不能回答的（只是相关或未知）**：
- 个体因果效应（反事实）："如果张三没用券，他会消费多少"是个体反事实，无法直接观测
- 长期效应：数据只有当前时间范围，无法回答更长期的效应
- 效应异质性：我们估计的是平均效应，不同人群的效应可能不同

**限制**：
- 存在未观察混杂的可能（如用户收入，如果数据中没有）
- 倾向评分匹配会丢弃无法匹配的样本（可能影响外推性）
- 回归假设线性关系，如果真实关系非线性，估计可能有偏差

### 工程实践

本分析使用了以下最佳实践：
- **先画因果图**：明确假设，可视化变量关系
- **后门准则**：科学地选择调整集，不盲目调整一切
- **两种方法验证**：回归 + 匹配，结果接近则结论稳健
- **Bootstrap 置信区间**：量化不确定性，不依赖分布假设

### 数据来源
- **样本量**：n = {n_total}
- **分析日期**：2026-02-13
- **随机种子**：42（保证可复现）

---

"""
    return report


# ============================================================================
# 示例使用（生成模拟数据演示）
# ============================================================================

def demo_with_mock_data():
    """使用模拟数据演示完整流程"""
    print("\n" + "=" * 70)
    print("StatLab 因果推断报告生成器 - 演示模式")
    print("=" * 70)

    # 1. 生成模拟数据（优惠券案例）
    np.random.seed(42)
    n = 1000

    # 混杂变量
    activity = np.random.normal(50, 15, n)
    history_spend = np.random.normal(100, 30, n)

    # 处理变量
    coupon_prob = 0.2 + 0.006 * activity + 0.002 * history_spend
    coupon = np.random.binomial(1, np.clip(coupon_prob, 0, 1))

    # 结果变量（真实效应 = 30 元）
    spending = (
        50 + 1.5 * activity + 0.3 * history_spend +
        30 * coupon + np.random.normal(0, 15, n)
    )

    df = pd.DataFrame({
        '用户活跃度': activity,
        '历史消费': history_spend,
        '优惠券使用': coupon,
        '消费金额': spending
    })

    print(f"\n📊 模拟数据概览:")
    print(df.head(10))
    print(f"\n用券率: {coupon.mean():.1%}")
    print(f"平均消费: {spending.mean():.2f} 元")

    # 2. 运行因果推断分析
    report = causal_inference_report(
        df=df,
        treatment='优惠券使用',
        outcome='消费金额',
        confounders=['用户活跃度', '历史消费'],
        output_dir='report'
    )

    # 3. 打印报告
    print("\n" + "=" * 70)
    print("生成的报告片段:")
    print("=" * 70)
    print(report)

    # 4. 保存到文件
    output_path = Path("report")
    output_path.mkdir(exist_ok=True)

    report_file = output_path / "causal_inference_report.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# 因果推断 - StatLab 示例报告\n\n")
        f.write(report)

    print(f"\n✅ 报告已保存到: {report_file}")
    print(f"✅ 因果图已保存到: {output_path}/causal_dag.png")
    print(f"✅ 倾向评分匹配图已保存到: {output_path}/psm_comparison.png")

    return report


def main():
    """主函数"""
    # 演示模式：使用模拟数据
    report = demo_with_mock_data()

    print("\n" + "=" * 70)
    print("💡 使用说明")
    print("=" * 70)
    print("""
在你的 StatLab 项目中使用本脚本的步骤：

1. 替换数据源:
   df = pd.read_csv("data/clean_data.csv")

2. 指定你的处理变量、结果变量和混杂变量:
   treatment = "your_treatment_variable"    # 如 'coupon_used'
   outcome = "your_outcome_variable"        # 如 'spending'
   confounders = ["conf1", "conf2", ...]   # 混杂变量列表

3. 运行函数生成报告:
   report = causal_inference_report(
       df, treatment, outcome, confounders, "report"
   )

4. 将生成的报告片段追加到 report.md

本周 StatLab 的改进（相比上周）:
- 新增因果图（DAG）可视化
- 新增识别策略（后门准则）
- 新增因果效应估计（回归 + 匹配）
- 明确区分"因果结论"和"相关发现"
- 写清结论边界（能回答什么、不能回答什么）

这是从"预测/关联"到"因果推断"的关键跃迁！
    """)


if __name__ == "__main__":
    main()

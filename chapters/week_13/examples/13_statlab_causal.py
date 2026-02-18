"""
示例：StatLab 因果推断报告——从相关性到因果性

本示例是 Week 13 的 StatLab 超级线代码，在 Week 12 可解释性与伦理报告的基础上增量添加：
- 画因果图（明确分析假设）
- 识别因果路径和混杂因素
- 区分"相关性发现"和"因果性结论"
- 生成因果推断报告模块

与上周对比：
- Week 12: 模型可解释性（SHAP 值）+ 公平性评估 + 伦理风险清单
- Week 13: 因果图 + 因果路径识别 + 相关/因果区分 + 因果推断报告

运行方式：python3 chapters/week_13/examples/13_statlab_causal.py
预期输出：生成完整的因果推断报告（Markdown + 图表），输出到 output/
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

# 尝试导入 networkx
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    print("警告: networkx 未安装，DAG 可视化将被跳过。请运行: pip install networkx")


RANDOM_STATE = 42


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
# 第一部分：生成模拟数据（复用流失预测场景）
# ============================================================================

def generate_churn_data(n_samples: int = 5000, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """
    生成流失预测数据（复用 Week 12 的数据结构）

    新增：coupon 变量（是否收到优惠券）

    特征：
    - days_since_last_purchase: 最近购买天数
    - purchase_count: 购买次数
    - avg_spend: 平均消费金额
    - vip_status: 是否会员（0/1）
    - age: 年龄
    - coupon: 是否收到优惠券（0/1）

    目标：
    - churn: 是否流失（0/1）

    因果结构（用于演示）：
    - high_value (未观测) → coupon
    - high_value (未观测) → churn
    - coupon → purchase_count
    - purchase_count → churn
    - vip_status → churn
    """
    rng = np.random.default_rng(random_state)

    # 未观测的混杂变量：高价值客户
    high_value = rng.binomial(1, 0.3, n_samples)

    # 高价值客户更容易收到优惠券
    coupon_prob = np.where(high_value == 1, 0.7, 0.2)
    coupon = rng.binomial(1, coupon_prob)

    # 生成其他特征
    days = rng.poisson(30, n_samples)
    count = rng.poisson(5, n_samples)
    spend = rng.exponential(100, n_samples)
    vip = rng.binomial(1, 0.3, n_samples)
    age = rng.integers(18, 70, n_samples)

    # 优惠券影响购买次数（因果机制）
    count = count + coupon * 2

    # 生成目标（流失率）
    # 高价值客户和 VIP 流失率低，购买次数多流失率低
    # 优惠券有小幅负因果效应（降低流失率）
    logit = (
        -1.5
        - 1.0 * high_value  # 高价值客户流失率低
        - 0.8 * vip  # VIP 流失率低
        - 0.15 * count  # 购买次数多流失率低
        + 0.04 * days  # 最近购买天数多流失率高
        - 0.3 * coupon  # 优惠券降低流失率（因果效应）
    )
    prob = 1 / (1 + np.exp(-logit))
    churn = rng.binomial(1, prob)

    df = pd.DataFrame({
        'days_since_last_purchase': days,
        'purchase_count': count,
        'avg_spend': spend,
        'vip_status': vip,
        'age': age,
        'coupon': coupon,
        'churn': churn
    })

    return df


# ============================================================================
# 第二部分：画因果图
# ============================================================================

def draw_causal_dag(output_dir: Path) -> Path:
    """
    画因果图：优惠券对流失率的影响

    基于领域知识的假设：
    - high_value_customer (未观测) → coupon
    - high_value_customer (未观测) → churn
    - coupon → purchase_count
    - purchase_count → churn
    - vip_status → churn
    - days_since_last_purchase → churn
    """
    if not NETWORKX_AVAILABLE:
        print("  跳过 DAG 可视化（networkx 未安装）")
        return None

    font = setup_chinese_font()

    G = nx.DiGraph()

    # 添加节点和边（基于领域知识）
    edges = [
        ('高价值客户\n(未观测)', '优惠券'),
        ('高价值客户\n(未观测)', '流失率'),
        ('优惠券', '购买次数'),
        ('购买次数', '流失率'),
        ('VIP身份', '流失率'),
        ('最近购买天数', '流失率'),
    ]

    G.add_edges_from(edges)

    # 绘制
    fig, ax = plt.subplots(figsize=(10, 7))
    pos = nx.spring_layout(G, seed=42, k=1.5)

    # 分别绘制节点（未观测变量用不同颜色）
    unobserved = ['高价值客户\n(未观测)']
    observed = [n for n in G.nodes() if n not in unobserved]

    nx.draw_networkx_nodes(G, pos, nodelist=unobserved, node_size=3000,
                           node_color='lightcoral', ax=ax, label='未观测')
    nx.draw_networkx_nodes(G, pos, nodelist=observed, node_size=3000,
                           node_color='lightblue', ax=ax, label='观测')

    nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)
    nx.draw_networkx_edges(G, pos, arrowsize=20, arrowstyle='->',
                           edge_color='gray', width=2, ax=ax)

    ax.legend(fontsize=10, loc='upper left')
    ax.set_title('因果图：优惠券对流失率的影响\n（基于领域知识的假设）',
                 fontsize=14, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()
    output_path = output_dir / 'causal_dag.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print(f"  因果图已保存: {output_path}")
    return output_path


# ============================================================================
# 第三部分：识别因果路径
# ============================================================================

def identify_backdoor_paths() -> dict:
    """
    识别后门路径（需要控制的混杂路径）

    基于上面的 DAG 手动分析：
    - 后门路径1：优惠券 ← 高价值客户 → 流失率（需要控制高价值客户）
    - 前门路径：优惠券 → 购买次数 → 流失率（因果机制，不阻断）

    问题：高价值客户可能不可观测
    解决：用代理变量（如 purchase_count, vip_status）
    """
    backdoor_paths = {
        'coupon -> churn': [
            {
                'path': 'coupon <- 高价值客户 -> churn',
                'type': '混杂',
                'need_to_control': '高价值客户',
                'observable': False,
                'proxy_variables': ['purchase_count', 'vip_status']
            }
        ]
    }

    frontdoor_paths = {
        'coupon -> churn': [
            {
                'path': 'coupon -> purchase_count -> churn',
                'type': '中介',
                'control_decision': '取决于研究问题（总效应 vs 直接效应）'
            }
        ]
    }

    return {
        'backdoor_paths': backdoor_paths,
        'frontdoor_paths': frontdoor_paths,
        'recommendation': '控制高价值客户的代理变量（purchase_count, vip_status）'
    }


# ============================================================================
# 第四部分：区分相关和因果
# ============================================================================

def analyze_correlation_vs_causation(df: pd.DataFrame) -> dict:
    """
    分析"优惠券和流失率"的关系：区分相关和因果
    """
    results = {}

    # 1. 相关性发现（观察数据）
    print("\n相关性发现（观察数据）：")

    churn_with_coupon = df[df['coupon'] == 1]['churn'].mean()
    churn_without_coupon = df[df['coupon'] == 0]['churn'].mean()
    correlation = churn_without_coupon - churn_with_coupon

    print(f"  收到优惠券的客户流失率: {churn_with_coupon:.3f}")
    print(f"  未收到优惠券的客户流失率: {churn_without_coupon:.3f}")
    print(f"  相关性差异: {correlation:.3f}")

    # 卡方检验
    contingency = pd.crosstab(df['coupon'], df['churn'])
    chi2, p_value, _, _ = chi2_contingency(contingency)

    print(f"  卡方检验 p 值: {p_value:.4f}")

    results['correlation'] = {
        'churn_with_coupon': churn_with_coupon,
        'churn_without_coupon': churn_without_coupon,
        'difference': correlation,
        'p_value': p_value,
        'significant': p_value < 0.05
    }

    # 2. 因果性估计（倾向得分匹配）
    print("\n因果性估计（倾向得分匹配）：")

    covariates = ['purchase_count', 'vip_status', 'days_since_last_purchase', 'age']

    # 估计倾向得分
    ps_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    ps_model.fit(df[covariates], df['coupon'])
    df['propensity_score'] = ps_model.predict_proba(df[covariates])[:, 1]

    # 匹配
    treated = df[df['coupon'] == 1].copy()
    control = df[df['coupon'] == 0].copy()

    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(control[['propensity_score']])
    distances, indices = nn.kneighbors(treated[['propensity_score']])

    matched_control = control.iloc[indices.flatten()]

    # 计算匹配后的因果效应
    treated_outcome = treated['churn'].mean()
    matched_control_outcome = matched_control['churn'].mean()
    causal_effect = treated_outcome - matched_control_outcome

    print(f"  匹配后实验组流失率: {treated_outcome:.3f}")
    print(f"  匹配后对照组流失率: {matched_control_outcome:.3f}")
    print(f"  匹配后的因果效应（ATE）: {causal_effect:.3f}")

    results['causation_psm'] = {
        'treated_outcome': treated_outcome,
        'control_outcome': matched_control_outcome,
        'causal_effect': causal_effect,
        'n_matched': len(treated),
        'method': '倾向得分匹配（PSM）'
    }

    # 3. 回归调整（控制混杂）
    print("\n因果性估计（回归调整）：")

    X = df[['coupon'] + covariates]
    y = df['churn']

    log_reg = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    log_reg.fit(X, y)

    # 提取优惠券的系数（因果效应估计）
    coef_idx = list(X.columns).index('coupon')
    causal_coef = log_reg.coef_[0][coef_idx]

    # 转换为概率变化（在基线处）
    baseline_prob = y.mean()
    prob_change = baseline_prob * (1 - baseline_prob) * causal_coef

    print(f"  回归系数（log-odds）: {causal_coef:.3f}")
    print(f"  转换为概率变化: {prob_change:.3f}")

    results['causation_regression'] = {
        'coefficient': causal_coef,
        'prob_change': prob_change,
        'method': '逻辑回归（控制混杂）'
    }

    return results


# ============================================================================
# 第五部分：生成因果推断报告
# ============================================================================

def generate_causal_inference_report(
    df: pd.DataFrame,
    dag_path: Path,
    identification: dict,
    analysis_results: dict,
    output_file: Path
) -> str:
    """
    生成因果推断报告：区分相关和因果
    """
    md = ["# 因果推断分析报告\n\n"]
    md.append(f"**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    # 1. 因果图
    md.append("## 1. 因果图（假设）\n\n")
    if dag_path:
        md.append(f"![]({dag_path.name})\n\n")

    md.append("**因果假设说明**:\n\n")
    md.append("- 箭头表示基于领域知识的因果关系假设\n")
    md.append("- **高价值客户（未观测）** 是混杂变量：\n")
    md.append("  - 同时影响'是否收到优惠券'和'流失率'\n")
    md.append("  - 如果不控制，会导致虚假相关\n")
    md.append("- **购买次数** 是中介变量：\n")
    md.append("  - 优惠券 → 购买次数 → 流失率\n")
    md.append("  - 这是因果机制的一部分\n\n")

    # 2. 因果路径分析
    md.append("## 2. 因果路径分析\n\n")

    md.append("### 后门路径（混杂路径）\n\n")
    for treatment, paths in identification['backdoor_paths'].items():
        for i, path in enumerate(paths, 1):
            md.append(f"**路径 {i}**: {path['path']}\n\n")
            md.append(f"- 类型：{path['type']}\n")
            md.append(f"- 需要控制：{path['need_to_control']}\n")
            if not path['observable']:
                md.append(f"- **警告**: {path['need_to_control']} 可能不可观测！\n")
                md.append(f"- 代理变量：{', '.join(path['proxy_variables'])}\n")
            md.append("\n")

    md.append("### 前门路径（因果机制）\n\n")
    for treatment, paths in identification['frontdoor_paths'].items():
        for i, path in enumerate(paths, 1):
            md.append(f"**路径 {i}**: {path['path']}\n\n")
            md.append(f"- 类型：{path['type']}\n")
            md.append(f"- {path['control_decision']}\n\n")

    # 3. 相关性发现
    md.append("## 3. 相关性发现（观察数据）\n\n")

    corr = analysis_results['correlation']
    md.append("| 指标 | 值 |\n")
    md.append("|------|-----|\n")
    md.append(f"| 收到优惠券的流失率 | {corr['churn_with_coupon']:.3f} |\n")
    md.append(f"| 未收到优惠券的流失率 | {corr['churn_without_coupon']:.3f} |\n")
    md.append(f"| 差异 | {corr['difference']:.3f} |\n")
    md.append(f"| 卡方检验 p 值 | {corr['p_value']:.4f} |\n")
    md.append(f"| 显著性 | {'是' if corr['significant'] else '否'} |\n\n")

    md.append("**结论**: 优惠券和流失率在观察数据中显著相关。\n\n")

    # 4. 因果性结论
    md.append("## 4. 因果性结论（需谨慎）\n\n")

    # PSM 结果
    psm = analysis_results['causation_psm']
    md.append(f"### 方法 1: 倾向得分匹配（PSM）\n\n")
    md.append(f"- **平均处理效应（ATE）**: {psm['causal_effect']:.3f}\n")
    md.append(f"- 实验组流失率（匹配后）: {psm['treated_outcome']:.3f}\n")
    md.append(f"- 对照组流失率（匹配后）: {psm['control_outcome']:.3f}\n")
    md.append(f"- 匹配样本数: {psm['n_matched']}\n\n")

    # 回归结果
    reg = analysis_results['causation_regression']
    md.append(f"### 方法 2: 逻辑回归（控制混杂）\n\n")
    md.append(f"- 回归系数（log-odds）: {reg['coefficient']:.3f}\n")
    md.append(f"- 转换为概率变化: {reg['prob_change']:.3f}\n\n")

    md.append("**局限性**:\n\n")
    md.append("- PSM 和回归调整只能控制**观测变量**\n")
    md.append("- **未观测混杂**（如'购买意愿'）仍可能存在\n")
    md.append("- 代理变量不能完全替代未观测混杂\n")
    md.append("- **最优方法**: 随机对照试验（RCT/A/B 测试）\n\n")

    # 5. 能回答什么，不能回答什么
    md.append("## 5. 我们能回答什么，不能回答什么\n\n")

    md.append("| 问题 | 能回答吗？ | 原因 |\n")
    md.append("|------|-----------|------|\n")
    md.append(f"| 优惠券和流失率相关吗？ | ✅ 能 | 观察数据可以回答 |\n")
    md.append(f"| 发放优惠券会降低流失率吗？ | ⚠️ 部分能 | 需要更强的因果识别策略（如 RCT）|\n")
    md.append(f"| 如果不发放优惠券，流失率会怎样？ | ❌ 不能 | 需要反事实推断（更高层级）|\n\n")

    # 6. 建议的因果识别策略
    md.append("## 6. 建议的因果识别策略\n\n")

    md.append("**等级 1: 随机对照试验（RCT）——金标准**\n\n")
    md.append("- 随机分配客户是否收到优惠券\n")
    md.append("- 比较实验组和对照组的流失率\n")
    md.append("- 优点：切断所有混杂路径（观测和未观测）\n")
    md.append("- 缺点：成本较高，可能存在伦理问题\n\n")

    md.append("**等级 2: 准实验（Quasi-Experiment）——次优选择**\n\n")
    md.append("- 利用自然实验（如政策变化、地理差异）\n")
    md.append("- 双重差分（DID）、断点回归（RDD）、工具变量（IV）\n")
    md.append("- 优点：比纯观察研究可靠\n")
    md.append("- 缺点：需要更强的假设\n\n")

    md.append("**等级 3: 观察研究 + 敏感性分析**\n\n")
    md.append("- 倾向得分匹配（PSM）、分层调整\n")
    md.append("- 必须报告局限性：未观测混杂可能存在\n")
    md.append("- 敏感性分析：如果存在未观测混杂，结论会如何变化？\n\n")

    return "".join(md)


# ============================================================================
# 主函数
# ============================================================================

def main() -> None:
    """主函数：运行 StatLab 因果推断报告生成流水线"""
    print("=" * 60)
    print("StatLab 因果推断报告生成")
    print("=" * 60)

    # 创建输出目录
    output_dir = Path(__file__).parent.parent.parent.parent / 'output'
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 生成数据
    print("\n1. 生成模拟流失数据...")
    df = generate_churn_data()
    print(f"   数据规模: {df.shape[0]} 行, {df.shape[1]} 列")
    print(f"   流失率: {df['churn'].mean():.2%}")
    print(f"   优惠券占比: {df['coupon'].mean():.2%}")

    # 2. 画因果图
    print("\n2. 画因果图（明确假设）...")
    dag_path = draw_causal_dag(output_dir)

    # 3. 识别因果路径
    print("\n3. 识别因果路径...")
    identification = identify_backdoor_paths()
    print(f"   后门路径数: {len(identification['backdoor_paths']['coupon -> churn'])}")
    print(f"   前门路径数: {len(identification['frontdoor_paths']['coupon -> churn'])}")

    # 4. 区分相关和因果
    print("\n4. 区分相关和因果...")
    analysis_results = analyze_correlation_vs_causation(df)

    # 5. 生成报告
    print("\n5. 生成因果推断报告...")
    report_file = output_dir / 'causal_inference_report.md'
    report = generate_causal_inference_report(
        df, dag_path, identification, analysis_results, report_file
    )
    report_file.write_text(report, encoding='utf-8')

    print(f"   报告已保存: {report_file}")

    # 6. 打印摘要
    print("\n" + "=" * 60)
    print("报告摘要")
    print("=" * 60)

    print("\n相关性发现:")
    print(f"  优惠券和流失率的相关性差异: {analysis_results['correlation']['difference']:.3f}")
    print(f"  卡方检验 p 值: {analysis_results['correlation']['p_value']:.4f}")

    print("\n因果性估计:")
    print(f"  PSM 估计的因果效应: {analysis_results['causation_psm']['causal_effect']:.3f}")
    print(f"  回归估计的概率变化: {analysis_results['causation_regression']['prob_change']:.3f}")

    print("\n本周 StatLab 进展:")
    print("  - 添加了因果图模块（明确分析假设）")
    print("  - 实现了因果路径识别（后门/前门路径）")
    print("  - 区分了'相关性发现'和'因果性结论'")
    print("  - 生成了因果推断报告（包含局限性说明）")

    print("\n与 Week 12 的对比:")
    print("  - 上周: 模型可解释性（SHAP）+ 公平性评估 + 伦理风险")
    print("  - 本周: 因果图 + 因果路径识别 + 相关/因果区分")
    print("  - 新增: 从'能预测'到'能回答因果问题'")

    print("\n生成的文件:")
    if dag_path:
        print(f"  1. {dag_path} - 因果图")
    print(f"  2. {report_file} - 因果推断报告")

    print("\n" + "=" * 60)
    print("StatLab 因果推断报告生成完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()

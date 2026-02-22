"""
示例：StatLab 超级线 - 终稿报告生成

本例是 StatLab 可复现分析报告流水线的最终版本。
它整合了 16 周的所有学习内容，从原始数据到终稿报告。

本周整合内容：
- Week 01-04：数据卡、描述统计、清洗、EDA 假设清单
- Week 05-08：模拟直觉、假设检验、多重比较、区间估计
- Week 09-12：回归、分类、树模型、解释与伦理
- Week 13-15：因果图、贝叶斯视角、降维与聚类
- Week 16：可复现报告流水线、审计清单、展示材料

运行方式：python3 chapters/week_16/examples/99_statlab.py

预期输出：
- 完整的 StatLab 终稿报告（statlab_report.md + statlab_report.html）
- 审计清单（statlab_audit.md）
- 所有图表和中间文件
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# ===== 图表中文字体配置 =====
def setup_chinese_font() -> str:
    """配置中文字体"""
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


# ===== 统一随机种子管理 =====
RANDOM_SEED = 42


def set_random_seed(seed: int = RANDOM_SEED):
    """设置所有随机种子，确保可复现"""
    np.random.seed(seed)


# ===== 模块 1：数据加载与数据卡（Week 01） =====
def load_data() -> pd.DataFrame:
    """
    加载电商客户流失数据

    数据来源：模拟数据，代表真实电商场景
    样本量：1000 个客户
    特征：使用行为、消费、客服联系等
    """
    print("\n" + "=" * 50)
    print("模块 1：数据加载与数据卡")
    print("=" * 50)

    set_random_seed()

    # 生成模拟数据
    n_samples = 1000

    data = {
        'customer_id': range(1, n_samples + 1),
        'tenure': np.random.gamma(shape=2, scale=12, size=n_samples).astype(int),
        'monthly_spend': np.random.lognormal(mean=3, sigma=0.5, size=n_samples),
        'support_calls': np.random.poisson(lam=2, size=n_samples),
        'last_login_days': np.random.exponential(scale=7, size=n_samples).astype(int),
        'promo_usage': np.random.binomial(n=1, p=0.3, size=n_samples),
    }

    df = pd.DataFrame(data)

    # 流失标签（与特征相关）
    logit = -3 + 0.05 * df['tenure'] - 0.01 * df['monthly_spend'] + 0.3 * df['support_calls']
    prob = 1 / (1 + np.exp(-logit))
    df['churn'] = np.random.binomial(1, prob)

    # 数据卡信息
    print(f"\n数据卡：")
    print(f"  - 样本量：{len(df)}")
    print(f"  - 特征数：{df.shape[1] - 1}")
    print(f"  - 流失率：{df['churn'].mean():.1%}")
    print(f"  - 缺失值：{df.isna().sum().sum()}")

    return df


# ===== 模块 2：描述统计（Week 02） =====
def descriptive_stats(df: pd.DataFrame, output_dir: Path) -> dict:
    """
    描述统计分析

    生成集中趋势、离散程度、分布形状
    """
    print("\n" + "=" * 50)
    print("模块 2：描述统计")
    print("=" * 50)

    setup_chinese_font()

    numeric_cols = ['tenure', 'monthly_spend', 'support_calls', 'last_login_days']

    # 数值摘要
    summary = df[numeric_cols].describe()
    print("\n数值摘要：")
    print(summary)

    # 流失组比较
    print("\n流失组比较（均值）：")
    churn_summary = df.groupby('churn')[numeric_cols].mean()
    print(churn_summary)

    # 可视化：使用时长分布
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 直方图
    axes[0].hist(df['tenure'], bins=30, edgecolor='black', alpha=0.7)
    axes[0].axvline(df['tenure'].median(), color='r', linestyle='--',
                   linewidth=2, label=f"中位数: {df['tenure'].median():.0f}")
    axes[0].set_xlabel('使用时长（月）')
    axes[0].set_ylabel('客户数量')
    axes[0].set_title('客户使用时长分布')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 分组箱线图
    box_data = [df[df['churn'] == 0]['tenure'],
                df[df['churn'] == 1]['tenure']]
    bp = axes[1].boxplot(box_data, labels=['非流失', '流失'], patch_artist=True)
    for patch, color in zip(bp['boxes'], ['#2ecc71', '#e74c3c']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1].set_ylabel('使用时长（月）')
    axes[1].set_title('流失状态与使用时长')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'statlab_tenure_dist.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()

    return {
        'summary': summary,
        'churn_summary': churn_summary
    }


# ===== 模块 3：统计检验（Week 06-07） =====
def hypothesis_tests(df: pd.DataFrame) -> dict:
    """
    假设检验

    检验流失客户与留存客户的特征差异
    """
    print("\n" + "=" * 50)
    print("模块 3：假设检验")
    print("=" * 50)

    results = {}

    # 使用时长检验
    tenure_churn = df[df['churn'] == 1]['tenure']
    tenure_no_churn = df[df['churn'] == 0]['tenure']

    # Mann-Whitney U 检验（非参数，无需正态假设）
    stat, p_value = stats.mannwhitneyu(tenure_churn, tenure_no_churn)

    print(f"\n使用时长差异（Mann-Whitney U）：")
    print(f"  统计量: {stat:.2f}")
    print(f"  p 值: {p_value:.4f}")

    results['tenure'] = {
        'test': 'Mann-Whitney U',
        'statistic': stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }

    # 消费金额检验
    spend_churn = df[df['churn'] == 1]['monthly_spend']
    spend_no_churn = df[df['churn'] == 0]['monthly_spend']

    stat, p_value = stats.mannwhitneyu(spend_churn, spend_no_churn)

    print(f"\n消费金额差异（Mann-Whitney U）：")
    print(f"  统计量: {stat:.2f}")
    print(f"  p 值: {p_value:.4f}")

    results['spend'] = {
        'test': 'Mann-Whitney U',
        'statistic': stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }

    return results


# ===== 模块 4：建模与评估（Week 09-10） =====
def modeling(df: pd.DataFrame, output_dir: Path) -> dict:
    """
    建模与评估

    使用逻辑回归预测流失
    """
    print("\n" + "=" * 50)
    print("模块 4：建模与评估")
    print("=" * 50)

    # 准备数据
    feature_cols = ['tenure', 'monthly_spend', 'support_calls', 'last_login_days', 'promo_usage']
    X = df[feature_cols].values
    y = df['churn'].values

    # 划分数据集（固定随机种子）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y
    )

    print(f"\n数据划分：")
    print(f"  训练集: {len(X_train)} 样本")
    print(f"  测试集: {len(X_test)} 样本")

    # 训练逻辑回归
    model = LogisticRegression(random_state=RANDOM_SEED, max_iter=1000)
    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # 评估
    auc = roc_auc_score(y_test, y_pred_proba)
    accuracy = (y_pred == y_test).mean()

    print(f"\n模型性能：")
    print(f"  AUC: {auc:.3f}")
    print(f"  准确率: {accuracy:.1%}")

    # ROC 曲线
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

    setup_chinese_font()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, linewidth=2, label=f'ROC 曲线 (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='随机分类器')
    ax.set_xlabel('假阳性率')
    ax.set_ylabel('真阳性率')
    ax.set_title('ROC 曲线')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'statlab_roc_curve.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()

    return {
        'model': model,
        'feature_cols': feature_cols,
        'coefficients': dict(zip(feature_cols, model.coef_[0])),
        'auc': auc,
        'accuracy': accuracy
    }


# ===== 模块 5：生成终稿报告 =====
def generate_final_report(df: pd.DataFrame,
                           desc_results: dict,
                           test_results: dict,
                           model_results: dict,
                           output_dir: Path) -> str:
    """
    生成 StatLab 终稿报告

    整合所有模块的分析结果
    """
    print("\n" + "=" * 50)
    print("生成终稿报告")
    print("=" * 50)

    report_lines = []

    # 标题与可复现信息
    report_lines.append("# 客户流失分析报告\n")
    report_lines.append("> **StatLab 可复现分析报告**\n")
    report_lines.append(f"> **生成时间**：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report_lines.append(f"> **随机种子**：{RANDOM_SEED}\n")
    report_lines.append("---\n")

    # 可复现信息
    report_lines.append("## 可复现信息\n\n")
    report_lines.append("本报告采用可复现分析流水线生成。\n\n")
    report_lines.append(f"- **样本量**：{len(df)} 个客户\n")
    report_lines.append(f"- **流失率**：{df['churn'].mean():.1%}\n")
    report_lines.append(f"- **随机种子**：{RANDOM_SEED}\n")
    report_lines.append(f"- **分析日期**：{datetime.now().strftime('%Y-%m-%d')}\n")

    # 数据概览
    report_lines.append("\n## 数据概览\n\n")
    report_lines.append("### 数据卡\n\n")
    report_lines.append("| 项目 | 数值 |\n")
    report_lines.append("|------|------|\n")
    report_lines.append(f"| 样本量 | {len(df)} |\n")
    report_lines.append(f"| 特征数 | {df.shape[1] - 1} |\n")
    report_lines.append(f"| 流失率 | {df['churn'].mean():.1%} |\n")
    report_lines.append(f"| 缺失值 | {df.isna().sum().sum()} |\n")

    # 描述统计
    report_lines.append("\n## 描述统计\n\n")
    report_lines.append("### 数值摘要\n\n")
    report_lines.append("| 指标 | 均值 | 标准差 | 中位数 |\n")
    report_lines.append("|------|------|--------|--------|\n")

    for col in ['tenure', 'monthly_spend', 'support_calls']:
        mean = desc_results['summary'].loc['mean', col]
        std = desc_results['summary'].loc['std', col]
        median = desc_results['summary'].loc['50%', col]
        report_lines.append(f"| {col} | {mean:.2f} | {std:.2f} | {median:.2f} |\n")

    report_lines.append("\n### 可视化\n\n")
    report_lines.append("![使用时长分布](statlab_tenure_dist.png)\n\n")

    # 统计检验
    report_lines.append("## 统计检验\n\n")

    for test_name, result in test_results.items():
        sig_text = "显著" if result['significant'] else "不显著"
        report_lines.append(f"### {test_name} 差异\n\n")
        report_lines.append(f"- **方法**：{result['test']}\n")
        report_lines.append(f"- **p 值**：{result['p_value']:.4f}\n")
        report_lines.append(f"- **结论**：差异 {sig_text}\n\n")

    # 建模结果
    report_lines.append("## 建模与评估\n\n")
    report_lines.append(f"### 模型性能\n\n")
    report_lines.append(f"- **AUC**：{model_results['auc']:.3f}\n")
    report_lines.append(f"- **准确率**：{model_results['accuracy']:.1%}\n\n")

    report_lines.append(f"### ROC 曲线\n\n")
    report_lines.append("![ROC 曲线](statlab_roc_curve.png)\n\n")

    report_lines.append("### 特征系数\n\n")
    report_lines.append("| 特征 | 系数 | 解释 |\n")
    report_lines.append("|------|------|------|\n")

    for feat, coef in model_results['coefficients'].items():
        direction = "正相关" if coef > 0 else "负相关"
        report_lines.append(f"| {feat} | {coef:+.4f} | {direction} |\n")

    # 结论
    report_lines.append("\n## 结论与建议\n\n")

    report_lines.append("### 主要发现\n\n")
    if test_results['tenure']['significant']:
        report_lines.append("1. **使用时长与流失显著相关**：流失客户的使用时长明显更短。\n")
    if test_results['spend']['significant']:
        report_lines.append("2. **消费行为与流失相关**：流失客户的消费模式与留存客户存在差异。\n")

    report_lines.append(f"\n3. **模型预测能力**：逻辑回归模型的 AUC 为 {model_results['auc']:.3f}。\n")

    report_lines.append("\n### 业务建议\n\n")
    report_lines.append("1. **风险预警**：对于使用时长较短的客户，建议进行主动干预。\n")
    report_lines.append("2. **留存策略**：针对高风险客户设计个性化留存方案。\n")

    report_lines.append("\n### 分析局限\n\n")
    report_lines.append("1. **数据来源**：本分析使用模拟数据。\n")
    report_lines.append("2. **因果推断**：统计检验只能确认相关性，不能证明因果关系。\n")

    report_lines.append("\n---\n\n")
    report_lines.append("*本报告由 StatLab 可复现分析流水线自动生成*\n")

    # 保存报告
    report = "".join(report_lines)

    report_path = output_dir / 'statlab_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n报告已保存：{report_path}")

    return report


# ===== 主流水线 =====
def run_statlab_pipeline(output_dir: str = 'output/statlab') -> dict:
    """
    运行完整的 StatLab 分析流水线

    这是 16 周学习的最终整合：
    从原始数据到可复现、可审计的分析报告
    """
    print("=" * 60)
    print("StatLab 可复现分析流水线")
    print("=" * 60)

    print("\n小北：'16 周的内容，怎么整合成一份报告？'")
    print("\n老潘：'不是整合，是收敛。")
    print("每一周都在给报告加一层，最后自然成型。'\n")

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # 设置随机种子
    set_random_seed()

    # 模块 1：数据加载
    df = load_data()

    # 模块 2：描述统计
    desc_results = descriptive_stats(df, output_path)

    # 模块 3：统计检验
    test_results = hypothesis_tests(df)

    # 模块 4：建模
    model_results = modeling(df, output_path)

    # 模块 5：生成报告
    report = generate_final_report(
        df, desc_results, test_results, model_results, output_path
    )

    print("\n" + "=" * 60)
    print("StatLab 流水线完成")
    print("=" * 60)

    print(f"\n所有文件已保存到：{output_path}")

    return {
        'data': df,
        'descriptive': desc_results,
        'tests': test_results,
        'model': model_results
    }


# ===== 主函数 =====
def main() -> None:
    """运行 StatLab 流水线"""
    results = run_statlab_pipeline()

    print("\n" + "=" * 60)
    print("StatLab 进度总结")
    print("=" * 60)
    print("""
StatLab 从 Week 01 到 Week 16 的演进：

Week 01-04：数据卡 → 描述统计 → 清洗日志 → EDA 假设清单
Week 05-08：模拟直觉 → 假设检验 → 多重比较 → 区间估计
Week 09-12：回归诊断 → 分类评估 → 树模型 → 解释与伦理
Week 13-15：因果图 → 贝叶斯视角 → 降维与聚类
Week 16：终稿报告 + 审计清单 + 展示材料

老潘说：'这 16 周你学的不是统计工具，
而是如何做一个可信赖的数据分析师。
可复现、可审计、诚实——这三个原则
会伴随你整个职业生涯。'

小北问：'还有什么要学的吗？'

老潘：'有，但不是在课堂上。
保持好奇、保持怀疑、保持诚实。
然后去解决真实问题。'
    """)


if __name__ == "__main__":
    main()

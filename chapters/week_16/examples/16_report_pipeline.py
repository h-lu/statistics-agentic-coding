"""
示例：可复现报告流水线

本例演示如何构建一个从原始数据到终稿报告的完整分析流水线。
这是专业数据分析交付物的标准做法：任何人运行脚本都能得到相同结果。

核心原则：
1. 固定随机种子（所有随机操作可复现）
2. 明确输入输出（数据从哪来、报告到哪去）
3. 版本记录（依赖库版本、执行时间）
4. 线性执行（无交互式输入、从上到下一次跑完）

运行方式：python3 chapters/week_16/examples/16_report_pipeline.py

预期输出：
- 在 output/ 目录生成图表
- 打印分析流水线的执行步骤
- 生成结构化的分析结果供后续报告生成使用
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


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


# ===== 步骤 1：数据加载与清洗 =====
def load_and_clean_data(random_state: int = 42) -> pd.DataFrame:
    """
    数据加载与清洗

    可复现要点：
    - 固定随机种子
    - 记录数据来源
    - 记录清洗决策

    返回清洗后的数据框
    """
    print("\n" + "=" * 60)
    print("步骤 1：数据加载与清洗")
    print("=" * 60)

    # 生成模拟数据（在实际项目中这里是读取真实数据）
    np.random.seed(random_state)
    n_samples = 1000

    data = {
        'customer_id': range(1, n_samples + 1),
        'tenure': np.random.gamma(shape=2, scale=12, size=n_samples).astype(int),
        'monthly_spend': np.random.lognormal(mean=3, sigma=0.5, size=n_samples),
        'support_calls': np.random.poisson(lam=2, size=n_samples),
        'churn': np.random.binomial(n=1, p=0.2, size=n_samples)
    }
    df = pd.DataFrame(data)

    print(f"原始数据：{len(df)} 行, {df.shape[1]} 列")
    print(f"流失率：{df['churn'].mean():.1%}")

    # 清洗决策（记录在报告中）
    # - 缺失值：本次数据无缺失
    # - 异常值：保留所有真实值（不删除高消费用户）
    # - 数据类型：已确认

    print("\n清洗决策记录：")
    print("  - 缺失值处理：无缺失")
    print("  - 异常值处理：保留所有真实值")
    print("  - 数据来源：模拟电商客户数据")

    return df


# ===== 步骤 2：描述统计与可视化 =====
def compute_descriptive_stats(df: pd.DataFrame, output_dir: Path) -> dict:
    """
    描述统计与可视化

    生成：
    - 数值摘要（均值、中位数、分位数）
    - 分布图
    - 分组比较（流失 vs 非流失）
    """
    print("\n" + "=" * 60)
    print("步骤 2：描述统计与可视化")
    print("=" * 60)

    output_dir.mkdir(exist_ok=True)
    setup_chinese_font()

    stats_summary = {}

    # 2.1 数值摘要
    print("\n数值摘要：")
    numeric_cols = ['tenure', 'monthly_spend', 'support_calls']
    summary = df[numeric_cols].describe()
    print(summary)

    stats_summary['summary'] = summary

    # 2.2 流失组比较
    print("\n分组比较（流失 vs 非流失）：")
    churn_summary = df.groupby('churn')[numeric_cols].mean()
    print(churn_summary)

    stats_summary['churn_summary'] = churn_summary

    # 2.3 生成可视化
    # 2.3.1 使用时长分布
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 直方图
    axes[0].hist(df['tenure'], bins=30, edgecolor='black', alpha=0.7, color='#3498db')
    axes[0].set_xlabel('使用时长（月）', fontsize=12)
    axes[0].set_ylabel('客户数量', fontsize=12)
    axes[0].set_title('客户使用时长分布', fontsize=14, fontweight='bold')
    axes[0].axvline(df['tenure'].median(), color='r', linestyle='--',
                   linewidth=2, label=f"中位数: {df['tenure'].median():.0f}月")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 分组箱线图
    churn_labels = ['非流失', '流失']
    box_data = [df[df['churn'] == 0]['tenure'],
                df[df['churn'] == 1]['tenure']]
    bp = axes[1].boxplot(box_data, labels=churn_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], ['#2ecc71', '#e74c3c']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1].set_ylabel('使用时长（月）', fontsize=12)
    axes[1].set_title('流失状态与使用时长', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'tenure_distribution.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"保存: {output_dir / 'tenure_distribution.png'}")

    # 2.3.2 消费金额分布
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 直方图
    axes[0].hist(df['monthly_spend'], bins=30, edgecolor='black',
                 alpha=0.7, color='#3498db')
    axes[0].set_xlabel('月消费金额（元）', fontsize=12)
    axes[0].set_ylabel('客户数量', fontsize=12)
    axes[0].set_title('月消费金额分布', fontsize=14, fontweight='bold')
    axes[0].axvline(df['monthly_spend'].median(), color='r', linestyle='--',
                   linewidth=2, label=f"中位数: {df['monthly_spend'].median():.0f}元")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 分组箱线图
    box_data = [df[df['churn'] == 0]['monthly_spend'],
                df[df['churn'] == 1]['monthly_spend']]
    bp = axes[1].boxplot(box_data, labels=churn_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], ['#2ecc71', '#e74c3c']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1].set_ylabel('月消费金额（元）', fontsize=12)
    axes[1].set_title('流失状态与月消费', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'spend_distribution.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"保存: {output_dir / 'spend_distribution.png'}")

    return stats_summary


# ===== 步骤 3：统计检验 =====
def run_hypothesis_tests(df: pd.DataFrame) -> dict:
    """
    统计检验

    检验问题：
    - 流失用户与非流失用户的使用时长是否有显著差异？
    - 流失用户与非流失用户的消费金额是否有显著差异？

    方法：独立样本 t 检验
    前提：近似正态分布、方差齐性
    """
    print("\n" + "=" * 60)
    print("步骤 3：统计检验")
    print("=" * 60)

    test_results = {}

    # 3.1 使用时长差异检验
    print("\n检验 1：使用时长差异")
    print("-" * 40)

    tenure_churn = df[df['churn'] == 1]['tenure']
    tenure_no_churn = df[df['churn'] == 0]['tenure']

    # 正态性检查（Shapiro-Wilk）
    _, p_normal_churn = stats.shapiro(tenure_churn.sample(min(500, len(tenure_churn))))
    _, p_normal_no_churn = stats.shapiro(tenure_no_churn.sample(min(500, len(tenure_no_churn))))

    print(f"正态性检验（Shapiro-Wilk）：")
    print(f"  - 流失组: p = {p_normal_churn:.4f}")
    print(f"  - 非流失组: p = {p_normal_no_churn:.4f}")

    if p_normal_churn < 0.05 or p_normal_no_churn < 0.05:
        print("  → 数据不完全满足正态假设，考虑非参数检验")
        # 使用 Mann-Whitney U 检验
        stat_mw, p_mw = stats.mannwhitneyu(tenure_churn, tenure_no_churn)
        print(f"\nMann-Whitney U 检验：")
        print(f"  - 统计量: {stat_mw:.2f}")
        print(f"  - p 值: {p_mw:.4f}")
        test_results['tenure'] = {
            'test': 'Mann-Whitney U',
            'statistic': stat_mw,
            'p_value': p_mw,
            'significant': p_mw < 0.05
        }
    else:
        # 使用 t 检验
        # 方差齐性检验
        _, p_levene = stats.levene(tenure_churn, tenure_no_churn)
        equal_var = p_levene > 0.05

        t_stat, p_t = stats.ttest_ind(tenure_churn, tenure_no_churn, equal_var=equal_var)

        print(f"\n独立样本 t 检验：")
        print(f"  - t 统计量: {t_stat:.4f}")
        print(f"  - p 值: {p_t:.4f}")
        print(f"  - 方差齐性: {'满足' if equal_var else '不满足'}")
        test_results['tenure'] = {
            'test': 't-test',
            'statistic': t_stat,
            'p_value': p_t,
            'significant': p_t < 0.05
        }

    # 3.2 消费金额差异检验
    print("\n检验 2：消费金额差异")
    print("-" * 40)

    spend_churn = df[df['churn'] == 1]['monthly_spend']
    spend_no_churn = df[df['churn'] == 0]['monthly_spend']

    # 消费金额通常右偏，直接用 Mann-Whitney U
    stat_mw, p_mw = stats.mannwhitneyu(spend_churn, spend_no_churn)

    print(f"Mann-Whitney U 检验：")
    print(f"  - 统计量: {stat_mw:.2f}")
    print(f"  - p 值: {p_mw:.4f}")

    test_results['spend'] = {
        'test': 'Mann-Whitney U',
        'statistic': stat_mw,
        'p_value': p_mw,
        'significant': p_mw < 0.05
    }

    return test_results


# ===== 步骤 4：建模与评估 =====
def train_and_evaluate_model(df: pd.DataFrame, random_state: int = 42) -> dict:
    """
    建模与评估

    模型：逻辑回归预测流失
    评估：准确率、精确率、召回率、AUC
    """
    print("\n" + "=" * 60)
    print("步骤 4：建模与评估")
    print("=" * 60)

    # 准备数据
    feature_cols = ['tenure', 'monthly_spend', 'support_calls']
    X = df[feature_cols].values
    y = df['churn'].values

    # 划分训练/测试集（固定随机种子！）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state, stratify=y
    )

    print(f"\n数据划分：")
    print(f"  - 训练集: {len(X_train)} 样本")
    print(f"  - 测试集: {len(X_test)} 样本")
    print(f"  - 训练集流失率: {y_train.mean():.1%}")
    print(f"  - 测试集流失率: {y_test.mean():.1%}")

    # 训练模型
    model = LogisticRegression(random_state=random_state, max_iter=1000)
    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # 评估
    print(f"\n模型评估：")
    print(f"  - 准确率: {(y_pred == y_test).mean():.1%}")

    # AUC
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"  - AUC: {auc:.3f}")

    # 混淆矩阵
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n混淆矩阵：")
    print(f"  真阴性: {cm[0, 0]}, 假阳性: {cm[0, 1]}")
    print(f"  假阴性: {cm[1, 0]}, 真阳性: {cm[1, 1]}")

    # 特征系数
    print(f"\n特征系数：")
    for feat, coef in zip(feature_cols, model.coef_[0]):
        print(f"  - {feat}: {coef:+.4f}")

    return {
        'model': model,
        'feature_cols': feature_cols,
        'coefficients': dict(zip(feature_cols, model.coef_[0])),
        'accuracy': (y_pred == y_test).mean(),
        'auc': auc,
        'confusion_matrix': cm
    }


# ===== 步骤 5：生成版本信息 =====
def generate_reproducibility_info(random_state: int = 42) -> dict:
    """
    生成可复现性信息

    记录：
    - 执行时间
    - 随机种子
    - 依赖版本
    """
    import sys
    import sklearn

    repro_info = {
        'execution_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'random_seed': random_state,
        'python_version': sys.version.split()[0],
        'dependencies': {
            'numpy': np.__version__,
            'pandas': pd.__version__,
            'scikit-learn': sklearn.__version__
        }
    }

    print("\n" + "=" * 60)
    print("可复现性信息")
    print("=" * 60)
    print(f"执行时间: {repro_info['execution_time']}")
    print(f"随机种子: {repro_info['random_seed']}")
    print(f"Python 版本: {repro_info['python_version']}")
    print(f"依赖版本:")
    for lib, version in repro_info['dependencies'].items():
        print(f"  - {lib}: {version}")

    return repro_info


# ===== 主流水线 =====
def run_analysis_pipeline(data_path: str = None,
                          output_dir: str = 'output',
                          random_state: int = 42) -> dict:
    """
    完整的分析流水线

    这是可复现报告的核心：从原始数据到结构化结果，
    每一步都可以重现、每一步都有记录。

    参数:
        data_path: 数据文件路径（None 则生成模拟数据）
        output_dir: 输出目录
        random_state: 随机种子

    返回:
        包含所有分析结果的字典（用于生成报告）
    """
    print("=" * 60)
    print("可复现报告流水线")
    print("=" * 60)
    print("\n老潘说：'可复现报告不是跑得通，而是任何人跑一遍")
    print("都能得到相同结果。固定随机种子、记录依赖版本、")
    print("写清楚数据来源——这三点是基础。'\n")

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # 步骤 1：数据加载与清洗
    df = load_and_clean_data(random_state=random_state)

    # 步骤 2：描述统计与可视化
    desc_stats = compute_descriptive_stats(df, output_path)

    # 步骤 3：统计检验
    test_results = run_hypothesis_tests(df)

    # 步骤 4：建模与评估
    model_results = train_and_evaluate_model(df, random_state=random_state)

    # 步骤 5：可复现性信息
    repro_info = generate_reproducibility_info(random_state=random_state)

    # 汇总结果
    pipeline_results = {
        'data': {
            'n_samples': len(df),
            'n_features': df.shape[1] - 1,  # 减去 target
            'churn_rate': df['churn'].mean()
        },
        'descriptive': desc_stats,
        'tests': test_results,
        'model': model_results,
        'reproducibility': repro_info
    }

    print("\n" + "=" * 60)
    print("流水线执行完成")
    print("=" * 60)
    print(f"结果已汇总，可用于生成 report.md")
    print(f"图表已保存到: {output_path}")

    return pipeline_results


# ===== 主函数 =====
def main() -> None:
    """运行示例流水线"""
    results = run_analysis_pipeline(
        output_dir='output',
        random_state=42
    )

    # 打印关键结果预览
    print("\n" + "=" * 60)
    print("结果预览（用于报告生成）")
    print("=" * 60)
    print(f"\n数据概览:")
    print(f"  - 样本数: {results['data']['n_samples']}")
    print(f"  - 流失率: {results['data']['churn_rate']:.1%}")

    print(f"\n统计检验:")
    for test_name, test_result in results['tests'].items():
        sig = "显著" if test_result['significant'] else "不显著"
        print(f"  - {test_name}: {test_result['test']}, p={test_result['p_value']:.4f} ({sig})")

    print(f"\n模型评估:")
    print(f"  - AUC: {results['model']['auc']:.3f}")
    print(f"  - 准确率: {results['model']['accuracy']:.1%}")


if __name__ == "__main__":
    main()

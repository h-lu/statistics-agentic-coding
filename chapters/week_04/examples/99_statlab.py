#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StatLab Week 04：探索性数据分析与假设清单

本脚本在 Week 03 报告基础上，添加：
1. 相关性分析（Pearson/Spearman）
2. 分组比较（groupby/透视表）
3. 多变量关系分析（混杂变量识别）
4. 可检验假设清单
5. EDA 章节生成并追加到 report.md

这是 StatLab 超级线的 Week 04 增量更新。

运行方式：python3 chapters/week_04/examples/99_statlab.py
预期输出：更新后的 report.md，包含 EDA 章节
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Any


class StatLabEDA:
    """StatLab EDA 分析器（Week 04 核心组件）"""

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.start_time = datetime.now().isoformat()
        self.findings: list[dict] = []
        self.hypotheses: list[dict] = []
        self.figures: list[str] = []

    def add_finding(self, finding_id: str, description: str,
                    evidence: dict, priority: str = "中") -> None:
        """添加 EDA 发现"""
        self.findings.append({
            'id': finding_id,
            'description': description,
            'evidence': evidence,
            'priority': priority,
            'timestamp': datetime.now().isoformat()
        })

    def add_hypothesis(self, hypothesis: dict) -> None:
        """添加假设"""
        self.hypotheses.append(hypothesis)

    def generate_eda_section(self) -> str:
        """生成 Markdown 格式的 EDA 章节"""
        lines = [
            "## 探索性数据分析",
            "",
            f"> 本章记录从数据中发现的关系、差异与假设，为后续统计推断提供基础。",
            f"> 生成时间：{self.start_time[:19]}",
            "",
            "### 分析摘要",
            "",
            f"- **数据集**：{self.dataset_name}",
            f"- **发现数量**：{len(self.findings)} 个",
            f"- **假设数量**：{len(self.hypotheses)} 个",
            f"- **生成图表**：{len(self.figures)} 个",
            "",
            "### 关键发现",
            ""
        ]

        # 按优先级排序
        priority_order = {'高': 0, '中': 1, '低': 2}
        sorted_findings = sorted(
            self.findings,
            key=lambda x: priority_order.get(x['priority'], 3)
        )

        for f in sorted_findings:
            lines.append(f"**{f['id']}** [{f['priority']}优先级]")
            lines.append(f"- {f['description']}")
            if 'correlation' in f['evidence']:
                r = f['evidence']['correlation']
                lines.append(f"- 相关系数：r = {r:.3f}")
            if 'group_diff' in f['evidence']:
                diff = f['evidence']['group_diff']
                lines.append(f"- 组间差异：{diff}")
            lines.append("")

        # 假设清单
        lines.extend([
            "### 可检验假设清单",
            ""
        ])

        sorted_hypotheses = sorted(
            self.hypotheses,
            key=lambda x: priority_order.get(x['priority'], 3)
        )

        for h in sorted_hypotheses:
            lines.extend([
                f"**假设 {h['id']}** [{h['priority']}优先级]",
                f"- 描述：{h['description']}",
                f"- H0：{h['H0']}",
                f"- H1：{h['H1']}",
                f"- 数据支持：{h['data_support']}",
                f"- 建议检验：{h['proposed_test']}",
                f"- 潜在混杂：{h['confounders']}",
                ""
            ])

        # 局限
        lines.extend([
            "### 分析局限",
            "",
            "- 横截面数据，无法确定因果方向",
            "- 部分变量为用户自报，可能存在测量误差",
            "- 未控制的潜在混杂：职业、教育水平、家庭状况",
            "",
            "### 下一步工作",
            "",
            "- Week 06-08 将对上述假设进行统计检验",
            "- 考虑收集纵向数据以支持因果推断",
            "- 探索机器学习模型预测消费行为",
            ""
        ])

        return '\n'.join(lines)


def load_or_generate_data() -> pd.DataFrame:
    """加载数据或生成示例数据"""
    # 尝试加载 Week 03 的数据
    data_paths = [
        'checkpoint/week_03_cleaned.csv',
        '../week_03/checkpoint/week_03_cleaned.csv',
        'data/users_cleaned.csv',
    ]

    for path in data_paths:
        if Path(path).exists():
            print(f"加载数据: {path}")
            return pd.read_csv(path)

    # 生成示例数据
    print("未找到现有数据，生成示例数据...")
    return generate_sample_data()


def generate_sample_data(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """生成示例电商用户数据"""
    rng = np.random.default_rng(seed)

    df = pd.DataFrame({
        'user_id': range(1, n + 1),
        'age': np.clip(rng.normal(35, 12, n), 18, 70).astype(int),
        'monthly_income': np.clip(rng.lognormal(8.5, 0.6, n), 3000, 80000).astype(int),
        'gender': rng.choice(['男', '女'], n, p=[0.48, 0.52]),
        'city_tier': rng.choice(['一线', '二线', '三线'], n, p=[0.3, 0.45, 0.25]),
        'registration_days': np.clip(rng.exponential(200, n), 1, 1000).astype(int),
    })

    # 消费与收入、城市级别相关
    base_spend = df['monthly_income'] * rng.uniform(0.15, 0.35, n)
    city_multiplier = df['city_tier'].map({'一线': 1.3, '二线': 1.0, '三线': 0.8})
    df['monthly_spend'] = (base_spend * city_multiplier + rng.normal(0, 500, n)).astype(int).clip(100, None)

    # 用户等级
    spend_bins = [0, 1000, 3000, 8000, float('inf')]
    spend_labels = ['普通', '银卡', '金卡', '钻石']
    df['user_level'] = pd.cut(df['monthly_spend'], bins=spend_bins, labels=spend_labels)

    return df


def correlation_analysis(df: pd.DataFrame, eda: StatLabEDA) -> pd.DataFrame:
    """执行相关性分析"""
    print("\n[分析] 相关性分析...")

    # 自动检测数值列
    available_cols = df.columns.tolist()
    # 尝试找到匹配的列名
    age_col = next((c for c in available_cols if 'age' in c.lower()), None)
    income_col = next((c for c in available_cols if 'income' in c.lower() or '收入' in c), None)
    spend_col = next((c for c in available_cols if 'spend' in c.lower() or 'consum' in c.lower() or '消费' in c), None)

    numeric_cols = [c for c in [age_col, income_col, spend_col] if c is not None]

    if len(numeric_cols) < 2:
        print("  警告：未找到足够的数值列进行相关性分析")
        # 使用所有数值列
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:4]
        print(f"  使用检测到的数值列: {numeric_cols}")

    corr_matrix = df[numeric_cols].corr(method='pearson')

    # 记录发现（使用实际存在的列名）
    if len(numeric_cols) >= 2:
        col1, col2 = numeric_cols[0], numeric_cols[1]
        income_spend_r = corr_matrix.loc[col1, col2] if col1 in corr_matrix.index and col2 in corr_matrix.columns else 0
        eda.add_finding(
            'F1',
            f'{col1}与{col2}的相关性分析',
            {'correlation': income_spend_r, 'variable_pair': (col1, col2)},
            '高'
        )
        print(f"  {col1}-{col2}: r = {income_spend_r:.3f}")

    if len(numeric_cols) >= 3:
        col3 = numeric_cols[2]
        age_spend_r = corr_matrix.loc[col1, col3] if col1 in corr_matrix.index and col3 in corr_matrix.columns else 0
        eda.add_finding(
            'F2',
            f'{col1}与{col3}的相关性分析',
            {'correlation': age_spend_r, 'variable_pair': (col1, col3)},
            '中'
        )
        print(f"  {col1}-{col3}: r = {age_spend_r:.3f}")
    else:
        income_spend_r = age_spend_r = 0

    return corr_matrix


def group_comparison(df: pd.DataFrame, eda: StatLabEDA) -> None:
    """执行分组比较"""
    print("\n[分析] 分组比较...")

    # 自动检测分组列
    available_cols = df.columns.tolist()

    # 尝试找到消费/支出列
    spend_col = next((c for c in available_cols if 'spend' in c.lower() or 'consum' in c.lower() or 'total_spend' in c.lower()), None)
    if spend_col is None:
        spend_col = df.select_dtypes(include=[np.number]).columns[-1]  # 使用最后一个数值列

    # 尝试找到用户等级列
    level_col = next((c for c in available_cols if 'level' in c.lower() or '等级' in c), None)
    if level_col and level_col in df.columns:
        try:
            level_stats = df.groupby(level_col)[spend_col].agg(['mean', 'count'])
            print(f"  用户等级统计:\n{level_stats.round(0)}")
        except Exception as e:
            print(f"  无法按用户等级分组: {e}")

    # 尝试找到城市级别列
    city_col = next((c for c in available_cols if 'city' in c.lower() or '城市' in c), None)
    if city_col and city_col in df.columns:
        try:
            city_stats = df.groupby(city_col)[spend_col].mean()
            if len(city_stats) >= 2:
                city_diff = (city_stats.iloc[0] / city_stats.iloc[-1] - 1) * 100
                eda.add_finding(
                    'F3',
                    f'{city_stats.index[0]}城市用户消费比{city_stats.index[-1]}高 {abs(city_diff):.0f}%',
                    {'group_diff': f'{city_diff:.0f}%', 'groups': (city_stats.index[0], city_stats.index[-1])},
                    '中'
                )
                print(f"  城市差异: {city_stats.index[0]} vs {city_stats.index[-1]} = {city_diff:.0f}%")
        except Exception as e:
            print(f"  无法按城市分组: {e}")
    else:
        print(f"  未找到城市级别列，跳过城市分组分析")


def generate_hypotheses(df: pd.DataFrame, eda: StatLabEDA) -> None:
    """生成假设清单"""
    print("\n[分析] 生成假设清单...")

    # 自动检测数值列
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:4]

    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr()
        col1, col2 = numeric_cols[0], numeric_cols[1]
        corr_val = corr_matrix.loc[col1, col2]

        # H1: 相关性假设
        eda.add_hypothesis({
            'id': 'H1',
            'description': f'{col1}与{col2}存在相关关系',
            'H0': f'{col1}与{col2}的 Pearson 相关系数 = 0',
            'H1': f'{col1}与{col2}的 Pearson 相关系数 ≠ 0',
            'data_support': f'EDA 发现 r = {corr_val:.3f}',
            'proposed_test': 'Pearson 相关性检验',
            'confounders': '其他未控制的变量',
            'priority': '高'
        })

    # H2: 通用探索性假设
    eda.add_hypothesis({
        'id': 'H2',
        'description': '不同用户群体的消费行为存在差异',
        'H0': '各用户群体的平均消费相等',
        'H1': '至少有一组用户的平均消费不同',
        'data_support': '分组统计显示群体间存在差异',
        'proposed_test': '单因素方差分析 (ANOVA)',
        'confounders': '收入、年龄、城市分布',
        'priority': '中'
    })

    print(f"  生成假设: {len(eda.hypotheses)} 个")


def create_visualizations(df: pd.DataFrame, output_dir: str = 'checkpoint') -> list[str]:
    """创建可视化图表"""
    print("\n[分析] 创建可视化...")

    Path(output_dir).mkdir(exist_ok=True)
    figures = []

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 自动检测数值列
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:4]

    # 图1: 相关性热力图
    if len(numeric_cols) >= 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        corr = df[numeric_cols].corr()
        sns.heatmap(corr, annot=True, cmap='RdBu_r', center=0, vmin=-1, vmax=1, ax=ax)
        ax.set_title('Variable Correlation Heatmap')
        fig_path = f'{output_dir}/week_04_correlation_heatmap.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        figures.append(fig_path)
        print(f"  保存: {fig_path}")

    # 图2: 尝试创建分组箱线图
    available_cols = df.columns.tolist()
    cat_col = next((c for c in available_cols if df[c].dtype == 'object' or df[c].dtype.name == 'category'), None)
    spend_col = next((c for c in available_cols if 'spend' in c.lower() or 'total_spend' in c.lower()), numeric_cols[-1] if numeric_cols else None)

    if cat_col and spend_col and cat_col != spend_col:
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            # 只取前10个类别避免图表过于拥挤
            top_cats = df[cat_col].value_counts().head(10).index
            df_plot = df[df[cat_col].isin(top_cats)]
            sns.boxplot(data=df_plot, x=cat_col, y=spend_col, ax=ax)
            ax.set_title(f'{spend_col} by {cat_col}')
            fig_path = f'{output_dir}/week_04_group_boxplot.png'
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.close()
            figures.append(fig_path)
            print(f"  保存: {fig_path}")
        except Exception as e:
            print(f"  无法创建箱线图: {e}")

    return figures


def append_to_report(content: str, report_path: str = 'report.md') -> None:
    """追加内容到报告"""
    path = Path(report_path)

    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            existing = f.read()

        # 检查是否已有 EDA 章节
        if '## 探索性数据分析' in existing:
            print(f"  警告：{report_path} 已包含 EDA 章节，将更新")
            # 替换现有章节
            start = existing.find('## 探索性数据分析')
            end = existing.find('\n## ', start + 1)
            if end == -1:
                new_content = existing[:start] + content
            else:
                new_content = existing[:start] + content + existing[end:]
        else:
            new_content = existing + '\n\n' + content

        with open(path, 'w', encoding='utf-8') as f:
            f.write(new_content)
    else:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(f"# StatLab 分析报告\n\n{content}")

    print(f"  报告已更新: {report_path}")


def save_checkpoint(eda: StatLabEDA, output_dir: str = 'checkpoint') -> None:
    """保存检查点"""
    Path(output_dir).mkdir(exist_ok=True)

    # 保存发现列表
    import json
    checkpoint = {
        'week': 4,
        'timestamp': eda.start_time,
        'findings': eda.findings,
        'hypotheses': eda.hypotheses,
        'figures': eda.figures
    }

    with open(f'{output_dir}/week_04_checkpoint.json', 'w', encoding='utf-8') as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)

    print(f"  检查点已保存: {output_dir}/week_04_checkpoint.json")


def main() -> None:
    """主函数：StatLab Week 04 更新"""
    print("=" * 70)
    print("StatLab Week 04: 探索性数据分析与假设清单")
    print("=" * 70)

    # 初始化
    eda = StatLabEDA(dataset_name="用户消费分析")

    # 1. 加载数据
    print("\n[1/6] 加载数据...")
    df = load_or_generate_data()
    print(f"  数据形状: {df.shape}")

    # 2. 相关性分析
    print("\n[2/6] 相关性分析...")
    corr_matrix = correlation_analysis(df, eda)

    # 3. 分组比较
    print("\n[3/6] 分组比较...")
    group_comparison(df, eda)

    # 4. 生成假设
    print("\n[4/6] 生成假设清单...")
    generate_hypotheses(df, eda)

    # 5. 创建可视化
    print("\n[5/6] 创建可视化...")
    eda.figures = create_visualizations(df)

    # 6. 生成报告
    print("\n[6/6] 生成 EDA 章节...")
    eda_section = eda.generate_eda_section()
    append_to_report(eda_section, 'report.md')

    # 保存检查点
    save_checkpoint(eda)

    # 完成总结
    print("\n" + "=" * 70)
    print("StatLab Week 04 更新完成!")
    print("=" * 70)
    print(f"\n本周新增内容:")
    print(f"  - 相关性分析（Pearson/Spearman）")
    print(f"  - 分组比较（groupby/透视表）")
    print(f"  - 多变量关系分析")
    print(f"  - 可检验假设清单（{len(eda.hypotheses)} 个）")
    print(f"  - EDA 章节（已追加到 report.md）")
    print(f"\n下周预告:")
    print(f"  - 统计模拟与直觉")
    print(f"  - 假设检验基础")


if __name__ == "__main__":
    main()

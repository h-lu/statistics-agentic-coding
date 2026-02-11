#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StatLab Week 02 更新脚本

本脚本在 Week 01 数据卡基础上，补充描述统计章节。
这是增量更新的示例：不从头重写，而是追加内容。

运行方式：python 02_statlab_update.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def setup_style():
    """设置绘图风格"""
    sns.set_theme(style="whitegrid")
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False


def add_descriptive_statistics(data_path, report_path):
    """在 report.md 中追加描述统计章节"""
    print("=" * 60)
    print("StatLab Week 02：添加描述统计章节")
    print("=" * 60)

    # 读取数据
    df = pd.read_csv(data_path)

    print(f"\n读取数据：{data_path}")
    print(f"数据形状：{df.shape}")

    # 创建输出目录
    figures_dir = Path('figures')
    figures_dir.mkdir(exist_ok=True)

    # ========== 1. 生成统计摘要 ==========
    print("\n[1/4] 生成统计摘要表...")

    # 选择数值列
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) == 0:
        print("  警告：没有数值型列")
        return

    # 计算统计摘要
    summary_data = []
    for col in numeric_cols[:6]:  # 最多处理 6 列
        col_data = df[col].dropna()
        summary_data.append({
            '字段': col,
            '样本数': col_data.count(),
            '均值': col_data.mean(),
            '中位数': col_data.median(),
            '标准差': col_data.std(),
            '最小值': col_data.min(),
            '最大值': col_data.max(),
            'Q1': col_data.quantile(0.25),
            'Q3': col_data.quantile(0.75),
            'IQR': col_data.quantile(0.75) - col_data.quantile(0.25),
        })

    summary_df = pd.DataFrame(summary_data)

    # ========== 2. 生成分布图 ==========
    print("[2/4] 生成分布图...")

    figures_created = []

    for col in numeric_cols[:3]:  # 最多画 3 列的图
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # 左图：直方图 + KDE
        sns.histplot(df[col].dropna(), bins=30, kde=True, ax=axes[0], color='steelblue')
        axes[0].set_title(f'{col} 分布（直方图）', fontsize=11)
        axes[0].set_xlabel(col)
        axes[0].set_ylabel('频数')

        # 右图：箱线图
        sns.boxplot(y=df[col], ax=axes[1], color='lightblue')
        axes[1].set_title(f'{col} 分布（箱线图）', fontsize=11)
        axes[1].set_ylabel(col)

        # 添加统计标注
        mean_val = df[col].mean()
        median_val = df[col].median()
        axes[0].axvline(mean_val, color='red', linestyle='--', alpha=0.7, label=f'均值={mean_val:.1f}')
        axes[0].axvline(median_val, color='green', linestyle='--', alpha=0.7, label=f'中位数={median_val:.1f}')
        axes[0].legend(fontsize=9)

        plt.tight_layout()

        # 保存图片
        fig_filename = f'dist_{col.replace(" ", "_").replace("/", "_")}.png'
        fig_path = figures_dir / fig_filename
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()

        figures_created.append({
            'column': col,
            'filename': fig_filename,
            'path': str(fig_path),
            'mean': mean_val,
            'median': median_val
        })

        print(f"  已保存：{fig_filename}")

    # ========== 3. 追加到 report.md ==========
    print("[3/4] 追加内容到 report.md...")

    # 准备追加内容
    append_content = f"""

## 描述统计

### 核心指标摘要

| 字段 | 样本数 | 均值 | 中位数 | 标准差 | 最小值 | 最大值 | IQR |
|------|--------|------|--------|--------|--------|--------|-----|
"""

    # 添加表格行
    for _, row in summary_df.iterrows():
        append_content += f"| {row['字段']} | {row['样本数']:.0f} | {row['均值']:.2f} | {row['中位数']:.2f} | {row['标准差']:.2f} | {row['最小值']:.2f} | {row['最大值']:.2f} | {row['IQR']:.2f} |\n"

    append_content += "\n### 分布图\n\n"

    # 添加图片
    for fig_info in figures_created:
        append_content += f"**{fig_info['column']} 的分布**\n\n"
        append_content += f"![{fig_info['column']}](figures/{fig_info['filename']})\n\n"
        append_content += f"说明：{fig_info['column']} 的均值是 {fig_info['mean']:.2f}，"
        append_content += f"中位数是 {fig_info['median']:.2f}。"
        # 判断偏态
        if fig_info['mean'] > fig_info['median']:
            append_content += " 均值高于中位数，说明分布右偏（有高值拉高均值）。"
        elif fig_info['mean'] < fig_info['median']:
            append_content += " 均值低于中位数，说明分布左偏。"
        else:
            append_content += " 均值与中位数接近，分布较为对称。"
        append_content += "\n\n"

    # 追加到文件
    with open(report_path, 'a', encoding='utf-8') as f:
        f.write(append_content)

    print(f"  已追加到：{report_path}")

    return {
        'summary': summary_df,
        'figures': figures_created
    }


def create_checkpoint(df, checkpoint_dir='checkpoint'):
    """创建检查点，保存当前分析状态"""
    print("\n[4/4] 创建检查点...")

    Path(checkpoint_dir).mkdir(exist_ok=True)

    # 保存关键信息
    checkpoint_info = {
        'week': '02',
        'data_shape': df.shape,
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist(),
        'missing_values': df.isna().sum().to_dict(),
    }

    import json
    checkpoint_path = Path(checkpoint_dir) / 'week_02_checkpoint.json'
    with open(checkpoint_path, 'w', encoding='utf-8') as f:
        json.dump(checkpoint_info, f, ensure_ascii=False, indent=2)

    print(f"  检查点已保存：{checkpoint_path}")

    return checkpoint_path


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='StatLab Week 02 更新脚本')
    parser.add_argument('--data', default='data/users.csv',
                       help='数据文件路径（默认：data/users.csv）')
    parser.add_argument('--report', default='report.md',
                       help='报告文件路径（默认：report.md）')

    args = parser.parse_args()

    setup_style()

    # 执行更新
    result = add_descriptive_statistics(args.data, args.report)

    # 创建检查点（如果有数据）
    try:
        df = pd.read_csv(args.data)
        create_checkpoint(df)
    except:
        print("\n  跳过检查点创建（数据文件不存在）")

    print("\n" + "=" * 60)
    print("StatLab Week 02 更新完成！")
    print("=" * 60)
    print("\n下周 StatLab 将基于此报告继续添加：")
    print("  • 缺失值处理与异常值处理")
    print("  • 清洗日志")
    print("  • EDA 假设清单")


if __name__ == "__main__":
    main()

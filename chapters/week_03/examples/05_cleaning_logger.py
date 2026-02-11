#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
示例：清洗日志生成与报告追加

本例演示：
1. 记录清洗决策的数据结构
2. 生成 Markdown 格式的清洗日志
3. 追加到 report.md 的函数
4. 可复现分析的关键实践

运行方式：python3 chapters/week_03/examples/05_cleaning_logger.py
预期输出：清洗日志 Markdown 内容，以及追加到报告的操作
"""
from __future__ import annotations

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Any


class CleaningLogger:
    """
    清洗日志记录器

    用于记录数据清洗过程中的所有决策和操作，
    确保分析的可复现性和可审计性。
    """

    def __init__(self, dataset_name: str = "unnamed"):
        self.dataset_name = dataset_name
        self.entries: list[dict] = []
        self.start_time = datetime.now().isoformat()

    def log_missing_handling(
        self,
        column: str,
        original_missing: int,
        strategy: str,
        rationale: str,
        alternatives: list[str] | None = None
    ) -> None:
        """
        记录缺失值处理决策

        参数：
            column: 处理的列名
            original_missing: 原始缺失数量
            strategy: 采用的策略（如 'median_imputation', 'drop_rows'）
            rationale: 选择该策略的理由
            alternatives: 考虑过但未采用的策略
        """
        entry = {
            'type': 'missing_handling',
            'column': column,
            'original_missing': original_missing,
            'strategy': strategy,
            'rationale': rationale,
            'alternatives': alternatives or [],
            'timestamp': datetime.now().isoformat()
        }
        self.entries.append(entry)

    def log_outlier_handling(
        self,
        column: str,
        detection_method: str,
        outliers_found: int,
        handling_strategy: str,
        rationale: str,
        examples: list[Any] | None = None
    ) -> None:
        """
        记录异常值处理决策

        参数：
            column: 处理的列名
            detection_method: 检测方法（如 'IQR', 'Z-score'）
            outliers_found: 发现的异常值数量
            handling_strategy: 处理策略（如 'winsorize', 'remove', 'keep'）
            rationale: 选择该策略的理由
            examples: 异常值示例
        """
        entry = {
            'type': 'outlier_handling',
            'column': column,
            'detection_method': detection_method,
            'outliers_found': outliers_found,
            'handling_strategy': handling_strategy,
            'rationale': rationale,
            'examples': examples or [],
            'timestamp': datetime.now().isoformat()
        }
        self.entries.append(entry)

    def log_transformation(
        self,
        column: str,
        transformation: str,
        rationale: str,
        parameters: dict | None = None
    ) -> None:
        """
        记录特征变换决策

        参数：
            column: 处理的列名
            transformation: 变换类型（如 'StandardScaler', 'log'）
            rationale: 变换理由
            parameters: 变换参数
        """
        entry = {
            'type': 'transformation',
            'column': column,
            'transformation': transformation,
            'rationale': rationale,
            'parameters': parameters or {},
            'timestamp': datetime.now().isoformat()
        }
        self.entries.append(entry)

    def log_custom(self, entry_type: str, details: dict) -> None:
        """记录自定义类型的日志条目"""
        entry = {
            'type': entry_type,
            'timestamp': datetime.now().isoformat(),
            **details
        }
        self.entries.append(entry)

    def to_dict(self) -> dict:
        """将日志导出为字典"""
        return {
            'dataset_name': self.dataset_name,
            'start_time': self.start_time,
            'end_time': datetime.now().isoformat(),
            'entry_count': len(self.entries),
            'entries': self.entries
        }

    def to_json(self, filepath: str | None = None) -> str:
        """
        将日志导出为 JSON 字符串或保存到文件

        参数：
            filepath: 如果提供，保存到文件；否则返回字符串
        """
        data = self.to_dict()
        json_str = json.dumps(data, ensure_ascii=False, indent=2)

        if filepath:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(json_str)
            print(f"日志已保存: {filepath}")

        return json_str

    def to_markdown(self) -> str:
        """
        将日志生成为 Markdown 格式的清洗报告
        """
        lines = []
        lines.append("## 数据清洗日志\n")
        lines.append(f"**数据集**: {self.dataset_name}  ")
        lines.append(f"**记录时间**: {self.start_time}  ")
        lines.append(f"**条目数**: {len(self.entries)}\n")

        # 按类型分组
        missing_entries = [e for e in self.entries if e['type'] == 'missing_handling']
        outlier_entries = [e for e in self.entries if e['type'] == 'outlier_handling']
        transform_entries = [e for e in self.entries if e['type'] == 'transformation']
        other_entries = [e for e in self.entries if e['type'] not in ['missing_handling', 'outlier_handling', 'transformation']]

        # 缺失值处理
        if missing_entries:
            lines.append("### 缺失值处理\n")
            lines.append("| 字段 | 缺失数 | 策略 | 理由 |")
            lines.append("|------|--------|------|------|")
            for e in missing_entries:
                rationale_short = e['rationale'][:50] + '...' if len(e['rationale']) > 50 else e['rationale']
                lines.append(f"| {e['column']} | {e['original_missing']} | {e['strategy']} | {rationale_short} |")
            lines.append("")

        # 异常值处理
        if outlier_entries:
            lines.append("### 异常值处理\n")
            lines.append("| 字段 | 检测方法 | 发现数 | 处理策略 | 理由 |")
            lines.append("|------|----------|--------|----------|------|")
            for e in outlier_entries:
                rationale_short = e['rationale'][:40] + '...' if len(e['rationale']) > 40 else e['rationale']
                lines.append(f"| {e['column']} | {e['detection_method']} | {e['outliers_found']} | {e['handling_strategy']} | {rationale_short} |")
            lines.append("")

        # 特征变换
        if transform_entries:
            lines.append("### 特征变换\n")
            lines.append("| 字段 | 变换 | 理由 |")
            lines.append("|------|------|------|")
            for e in transform_entries:
                rationale_short = e['rationale'][:50] + '...' if len(e['rationale']) > 50 else e['rationale']
                lines.append(f"| {e['column']} | {e['transformation']} | {rationale_short} |")
            lines.append("")

        # 其他条目
        for e in other_entries:
            lines.append(f"### {e['type']}\n")
            for key, value in e.items():
                if key not in ['type', 'timestamp']:
                    lines.append(f"- **{key}**: {value}")
            lines.append("")

        # 详细理由附录
        lines.append("### 详细决策理由\n")
        for e in self.entries:
            if 'rationale' in e:
                lines.append(f"**{e['column']} - {e.get('strategy', e.get('transformation', '处理'))}**: {e['rationale']}")
                if e.get('alternatives'):
                    lines.append(f"  - 考虑过但未采用的策略: {', '.join(e['alternatives'])}")
                lines.append("")

        return '\n'.join(lines)


def append_to_report(report_path: str, content: str, section_title: str | None = None) -> None:
    """
    追加内容到报告文件

    参数：
        report_path: 报告文件路径
        content: 要追加的内容
        section_title: 可选的章节标题
    """
    path = Path(report_path)

    # 确保目录存在
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'a', encoding='utf-8') as f:
        if section_title:
            f.write(f"\n\n## {section_title}\n\n")
        f.write(content)

    print(f"内容已追加到: {report_path}")


def generate_sample_cleaning_log() -> CleaningLogger:
    """
    生成一个示例清洗日志

    模拟 Week 03 的 StatLab 数据清洗过程
    """
    logger = CleaningLogger(dataset_name="用户消费分析数据集")

    # 记录缺失值处理
    logger.log_missing_handling(
        column='age',
        original_missing=25,
        strategy='median_imputation',
        rationale='缺失率 5%，MCAR 机制，中位数对异常值稳健',
        alternatives=['删除法（损失样本）', '均值填充（受异常值影响）']
    )

    logger.log_missing_handling(
        column='income',
        original_missing=120,
        strategy='group_median_imputation',
        rationale='缺失率 24%，MAR 机制（与老用户相关），按城市分组填充保留地域差异',
        alternatives=['删除法（损失过多样本）', '简单中位数填充（忽略城市差异）']
    )

    # 记录异常值处理
    logger.log_outlier_handling(
        column='total_spend',
        detection_method='IQR (1.5x)',
        outliers_found=18,
        handling_strategy='categorize_and_keep',
        rationale='异常值包含 VIP 高消费用户（真实）和数据错误（负数），分类处理而非统一删除',
        examples=[-5000.0, 120000.0, 98000.0]
    )

    # 记录特征变换
    logger.log_transformation(
        column='age',
        transformation='StandardScaler',
        rationale='用于聚类分析，需要消除量纲影响',
        parameters={'with_mean': True, 'with_std': True}
    )

    logger.log_transformation(
        column='income',
        transformation='MinMaxScaler',
        rationale='用于神经网络输入，需要固定到 [0,1] 范围',
        parameters={'feature_range': [0, 1]}
    )

    logger.log_transformation(
        column='city',
        transformation='OneHotEncoder',
        rationale='名义分类变量，用于线性模型',
        parameters={'sparse_output': False, 'drop': None}
    )

    return logger


def main() -> None:
    """主函数"""
    print("=" * 70)
    print("清洗日志生成与报告追加")
    print("=" * 70)

    # 生成示例日志
    logger = generate_sample_cleaning_log()

    print("\n【日志条目统计】")
    print(f"数据集: {logger.dataset_name}")
    print(f"条目数: {len(logger.entries)}")

    # 按类型统计
    type_counts = {}
    for e in logger.entries:
        t = e['type']
        type_counts[t] = type_counts.get(t, 0) + 1
    print("条目类型分布:")
    for t, count in type_counts.items():
        print(f"  - {t}: {count}")

    # 生成 Markdown
    print("\n" + "=" * 70)
    print("生成的 Markdown 清洗日志")
    print("=" * 70)

    markdown_content = logger.to_markdown()
    print(markdown_content[:2000])  # 打印前 2000 字符
    print("\n... [内容截断，完整内容见生成的文件] ...")

    # 保存到文件
    output_dir = Path('chapters/week_03/examples/output')
    output_dir.mkdir(exist_ok=True)

    # 保存 JSON 版本
    json_path = output_dir / 'cleaning_log.json'
    logger.to_json(str(json_path))

    # 保存 Markdown 版本
    md_path = output_dir / 'cleaning_log.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    print(f"Markdown 日志已保存: {md_path}")

    # 演示追加到 report.md
    print("\n" + "=" * 70)
    print("追加到报告演示")
    print("=" * 70)

    report_path = output_dir / 'report.md'

    # 创建报告头部（如果不存在）
    if not report_path.exists():
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# StatLab 分析报告\n\n")
            f.write("## 数据卡\n\n[Week 01 内容...]\n\n")
            f.write("## 描述统计\n\n[Week 02 内容...]\n")

    # 追加清洗日志
    append_to_report(str(report_path), markdown_content)

    print(f"\n报告已更新: {report_path}")

    # 读取并显示报告内容
    print("\n【报告当前内容预览】")
    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()
        print(content[:1500])
        print("\n...")

    print("\n" + "=" * 70)
    print("老潘的点评：")
    print("=" * 70)
    print("'清洗日志不是写给自己看的，是写给三个月后复查的人看的。'")
    print("'包括你自己——三个月后你会忘记当初为什么这样处理。'")
    print("'每一个决策都要有理由，哪怕理由是\"当时时间不够\"，也要诚实记录。'")


if __name__ == "__main__":
    main()

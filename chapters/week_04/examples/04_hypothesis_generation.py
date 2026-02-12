#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
示例：从 EDA 发现到可检验假设

本例演示：
1. 从 EDA 发现中提炼假设
2. 将假设形式化为 H0/H1
3. 创建假设清单模板
4. 为每个假设标记数据支持和潜在混杂

运行方式：python3 chapters/week_04/examples/04_hypothesis_generation.py
预期输出：stdout 输出格式化的假设清单
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Any


def generate_eda_findings() -> list[dict]:
    """
    生成模拟的 EDA 发现

    这些发现来自对电商用户数据的探索性分析
    """
    findings = [
        {
            'id': 'F1',
            'description': '收入与月消费呈中度正相关（Pearson r = 0.52）',
            'variable_pair': ('monthly_income', 'monthly_spend'),
            'correlation': 0.52,
            'p_value': 0.001,
            'sample_size': 285,
            'business_relevance': '高：收入是消费的重要预测因子'
        },
        {
            'id': 'F2',
            'description': '一线城市用户平均消费比三线城市高 48%',
            'group_comparison': ('一线', '三线'),
            'metric': 'monthly_spend',
            'difference_pct': 48,
            'sample_sizes': {'一线': 95, '三线': 78},
            'business_relevance': '中：可能反映生活成本差异'
        },
        {
            'id': 'F3',
            'description': '女性用户整体消费比男性高 12%，但控制收入后差异降至 3%',
            'group_comparison': ('女', '男'),
            'metric': 'monthly_spend',
            'overall_diff_pct': 12,
            'controlled_diff_pct': 3,
            'confounder': 'monthly_income',
            'business_relevance': '中：性别差异可能被收入混杂'
        },
        {
            'id': 'F4',
            'description': '钻石用户平均年龄（42岁）比普通用户（31岁）大 11岁',
            'group_comparison': ('钻石', '普通'),
            'metric': 'age',
            'difference': 11,
            'business_relevance': '高：年龄-等级-消费存在关联网络'
        },
        {
            'id': 'F5',
            'description': '注册天数与消费无显著相关（r = 0.08）',
            'variable_pair': ('registration_days', 'monthly_spend'),
            'correlation': 0.08,
            'p_value': 0.18,
            'business_relevance': '低：用户忠诚度不直接转化为消费'
        }
    ]
    return findings


def finding_to_hypothesis(finding: dict) -> dict | None:
    """
    将单个 EDA 发现转换为可检验假设

    规则：
    - 相关性发现 → 相关性检验假设
    - 分组差异发现 → 均值差异检验假设
    - 控制后差异消失的发现 → 混杂变量假设
    """
    h_id = finding['id'].replace('F', 'H')

    # 发现 F1：收入-消费相关
    if finding['id'] == 'F1':
        return {
            'id': h_id,
            'description': '用户月收入与月消费金额存在正相关关系',
            'H0': '收入与消费的 Pearson 相关系数 = 0（无线性相关）',
            'H1': '收入与消费的 Pearson 相关系数 > 0（存在正相关）',
            'data_support': f"EDA 发现 r = {finding['correlation']:.2f}, p < 0.001, n = {finding['sample_size']}",
            'proposed_test': 'Pearson 相关性检验（单尾）',
            'confounders': ['年龄', '城市级别', '职业类型'],
            'priority': '高',
            'effect_size_measure': 'r（相关系数）',
            'estimated_effect': f"r ≈ {finding['correlation']:.2f}（中度相关）"
        }

    # 发现 F2：城市级别差异
    elif finding['id'] == 'F2':
        return {
            'id': h_id,
            'description': '不同城市级别用户的平均消费存在差异',
            'H0': '一线 = 二线 = 三线城市的平均消费（无差异）',
            'H1': '至少有一组城市的平均消费不同',
            'data_support': f"EDA 发现一线 vs 三线差异 {finding['difference_pct']}%",
            'proposed_test': '单因素方差分析（ANOVA）',
            'confounders': ['收入分布', '用户等级构成', '生活成本'],
            'priority': '中',
            'effect_size_measure': 'eta-squared（η²）',
            'estimated_effect': '中等效应（预计 η² ≈ 0.06）'
        }

    # 发现 F3：性别差异（被收入混杂）
    elif finding['id'] == 'F3':
        return {
            'id': h_id,
            'description': '控制收入后，性别对消费无显著影响',
            'H0': '相同收入层内，男女消费差异 = 0',
            'H1': '相同收入层内，男女消费差异 ≠ 0',
            'data_support': f"整体差异 {finding['overall_diff_pct']}%，控制{finding['confounder']}后降至 {finding['controlled_diff_pct']}%",
            'proposed_test': '协方差分析（ANCOVA）或分层 t 检验',
            'confounders': ['年龄', '职业类型', '家庭状况'],
            'priority': '中',
            'effect_size_measure': 'Cohen\'s d',
            'estimated_effect': f"控制后 d ≈ {finding['controlled_diff_pct']/100*2:.2f}（小效应）"
        }

    # 发现 F4：年龄-等级关系
    elif finding['id'] == 'F4':
        return {
            'id': h_id,
            'description': '用户等级与年龄正相关（高等级用户年龄更大）',
            'H0': '各等级用户的平均年龄相等',
            'H1': '钻石用户的平均年龄显著高于普通用户',
            'data_support': f"EDA 发现钻石用户平均 {finding['difference']} 岁大于普通用户",
            'proposed_test': '独立样本 t 检验（钻石 vs 普通）',
            'confounders': ['收入', '注册时长', '职业阶段'],
            'priority': '高',
            'effect_size_measure': 'Cohen\'s d',
            'estimated_effect': '大效应（预计 d > 0.8）'
        }

    # 发现 F5：无显著相关
    elif finding['id'] == 'F5':
        return {
            'id': h_id,
            'description': '用户注册天数与消费无显著相关',
            'H0': '注册天数与消费的 Pearson 相关系数 = 0',
            'H1': '注册天数与消费的 Pearson 相关系数 ≠ 0',
            'data_support': f"EDA 发现 r = {finding['correlation']:.2f}, p = {finding['p_value']:.2f}（不显著）",
            'proposed_test': 'Pearson 相关性检验（确认性）',
            'confounders': ['用户活跃度', '注册后行为模式'],
            'priority': '低',
            'effect_size_measure': 'r',
            'estimated_effect': f"r ≈ {finding['correlation']:.2f}（可忽略）"
        }

    return None


def generate_hypothesis_list(findings: list[dict]) -> list[dict]:
    """将多个 EDA 发现转换为假设清单"""
    hypotheses = []
    for finding in findings:
        h = finding_to_hypothesis(finding)
        if h:
            hypotheses.append(h)
    return hypotheses


def format_hypothesis(h: dict) -> str:
    """格式化单个假设为可读文本"""
    lines = [
        f"\n{'='*70}",
        f"假设 {h['id']} [{h['priority']}优先级]",
        f"{'='*70}",
        f"描述：{h['description']}",
        f"",
        f"零假设 (H0)：{h['H0']}",
        f"备择假设 (H1)：{h['H1']}",
        f"",
        f"数据支持：{h['data_support']}",
        f"建议检验：{h['proposed_test']}",
        f"效应量指标：{h['effect_size_measure']}",
        f"预估效应：{h['estimated_effect']}",
        f"潜在混杂：{', '.join(h['confounders'])}",
    ]
    return '\n'.join(lines)


def print_hypothesis_report(hypotheses: list[dict]) -> None:
    """打印完整的假设清单报告"""
    print("\n" + "=" * 70)
    print("可检验假设清单")
    print("=" * 70)
    print(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"假设数量：{len(hypotheses)} 个")

    # 按优先级排序
    priority_order = {'高': 0, '中': 1, '低': 2}
    sorted_hypotheses = sorted(hypotheses, key=lambda x: priority_order.get(x['priority'], 3))

    for h in sorted_hypotheses:
        print(format_hypothesis(h))

    # 汇总统计
    print("\n" + "=" * 70)
    print("假设清单汇总")
    print("=" * 70)

    priority_counts = pd.Series([h['priority'] for h in hypotheses]).value_counts()
    print(f"\n优先级分布：")
    for p, c in priority_counts.items():
        print(f"  {p}优先级: {c} 个")

    test_types = {}
    for h in hypotheses:
        test = h['proposed_test']
        test_types[test] = test_types.get(test, 0) + 1

    print(f"\n建议检验方法分布：")
    for test, count in test_types.items():
        print(f"  {test}: {count} 个")


def generate_hypothesis_template() -> str:
    """
    生成假设清单模板（供用户填写）
    """
    template = """
# 可检验假设清单模板

## 使用说明
1. 每个假设必须有数据支持（来自 EDA 发现）
2. H0 和 H1 必须互斥且穷尽
3. 必须标记潜在混杂变量
4. 按业务价值区分优先级

## 假设条目模板

### 假设 [ID] [优先级]
- **描述**：[一句话描述研究问题]
- **H0（零假设）**：[默认状态，无效应/无差异]
- **H1（备择假设）**：[你想证明的效应/差异]
- **数据支持**：[EDA 发现的具体数据，如"r=0.52, p<0.001"]
- **建议检验**：[统计检验方法，如"Pearson 相关性检验"]
- **效应量指标**：[如何衡量效应大小，如"Cohen's d"]
- **潜在混杂**：[可能影响关系的第三方变量]
- **样本量**：[可用于检验的样本量]
- **备注**：[其他需要考虑的因素]

## 好假设的标准

1. **可检验**：能用数据证伪
   - ✅ "收入与消费正相关"
   - ❌ "用户喜欢我们的产品"

2. **具体明确**：变量和操作清晰
   - ✅ "月收入与月消费的 Pearson r > 0"
   - ❌ "收入影响消费"

3. **有数据支持**：基于 EDA 发现，非拍脑袋
   - ✅ "EDA 发现 r=0.52, n=285"
   - ❌ "我觉得应该是这样"

4. **考虑混杂**：标记潜在混淆因素
   - ✅ "潜在混杂：年龄、城市级别"
   - ❌ [无]

## 优先级划分指南

- **高**：对业务决策有直接影响，效应明显，数据支持强
- **中**：有业务价值，但效应较小或混杂较多
- **低**：探索性假设，或效应可能可忽略

## 常见错误

1. **H0 和 H1 不对称**
   - ❌ H0: r = 0, H1: r = 0.5（H1 太具体）
   - ✅ H0: r = 0, H1: r ≠ 0

2. **忽视方向性**
   - 如果有明确方向预期，使用单尾检验
   - ✅ H1: r > 0（单尾）
   - ✅ H1: r ≠ 0（双尾，无明确方向）

3. **遗漏效应量**
   - 统计显著 ≠ 实际重要
   - 必须报告效应量（r, d, η² 等）
"""
    return template


def export_hypothesis_to_markdown(hypotheses: list[dict], output_path: str = 'hypothesis_list.md') -> None:
    """将假设清单导出为 Markdown 文件"""
    lines = [
        "# 可检验假设清单",
        "",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"假设数量：{len(hypotheses)} 个",
        "",
        "---",
        ""
    ]

    # 按优先级排序
    priority_order = {'高': 0, '中': 1, '低': 2}
    sorted_hypotheses = sorted(hypotheses, key=lambda x: priority_order.get(x['priority'], 3))

    for h in sorted_hypotheses:
        lines.extend([
            f"## 假设 {h['id']} [{h['priority']}优先级]",
            "",
            f"**描述**：{h['description']}",
            "",
            "**零假设 (H0)**：",
            f"{h['H0']}",
            "",
            "**备择假设 (H1)**：",
            f"{h['H1']}",
            "",
            "**数据支持**：",
            f"{h['data_support']}",
            "",
            "**建议检验**：",
            f"{h['proposed_test']}",
            "",
            "**效应量指标**：",
            f"{h['effect_size_measure']}",
            "",
            "**预估效应**：",
            f"{h['estimated_effect']}",
            "",
            "**潜在混杂**：",
            f"{', '.join(h['confounders'])}",
            "",
            "---",
            ""
        ])

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"\n假设清单已导出: {output_path}")


def main() -> None:
    """主函数"""
    print("=" * 70)
    print("假设生成示例：从 EDA 发现到可检验假设")
    print("=" * 70)

    # 1. 生成 EDA 发现
    print("\n[1/3] 生成 EDA 发现...")
    findings = generate_eda_findings()
    print(f"  发现数量: {len(findings)}")
    for f in findings:
        print(f"    {f['id']}: {f['description'][:50]}...")

    # 2. 转换为假设
    print("\n[2/3] 将发现转换为假设...")
    hypotheses = generate_hypothesis_list(findings)
    print(f"  生成假设: {len(hypotheses)} 个")

    # 3. 打印报告
    print("\n[3/3] 生成假设清单报告...")
    print_hypothesis_report(hypotheses)

    # 4. 导出 Markdown
    export_hypothesis_to_markdown(hypotheses, 'hypothesis_list.md')

    # 5. 输出模板
    print("\n" + "=" * 70)
    print("假设清单模板（已保存到 hypothesis_template.md）")
    print("=" * 70)
    template = generate_hypothesis_template()
    with open('hypothesis_template.md', 'w', encoding='utf-8') as f:
        f.write(template)
    print("模板内容预览（前 30 行）：")
    print('\n'.join(template.split('\n')[:30]))
    print("...")

    # 角色对话
    print("\n" + "=" * 70)
    print("老潘的提醒：")
    print("=" * 70)
    print("'假设清单比代码更重要。'")
    print("'代码可以重写，但假设错了，整个分析方向就错了。'")
    print("'每个假设都要能追溯到具体的 EDA 发现，不能拍脑袋。'")

    print("\n" + "=" * 70)
    print("关键结论")
    print("=" * 70)
    print("1. EDA 发现只是现象，假设是待检验的命题")
    print("2. H0 和 H1 必须形式化、可证伪")
    print("3. 每个假设必须标记数据支持和潜在混杂")
    print("4. 按业务价值区分优先级，不要试图检验所有发现")
    print("5. 假设清单是 Week 06-08 统计检验的'待办事项'")


if __name__ == "__main__":
    main()

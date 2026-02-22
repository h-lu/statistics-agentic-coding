"""
示例：Markdown 报告生成器

本例演示如何用 Python 动态生成 Markdown 格式的分析报告。
核心思想：报告由脚本生成，而不是手工复制粘贴。

优势：
1. 数据更新后，重新运行脚本即可更新报告
2. 版本控制友好（Markdown 是纯文本）
3. 可以转换成 HTML、PDF 等多种格式
4. 数值和图表通过代码插入，不会出错

运行方式：python3 chapters/week_16/examples/16_markdown_generator.py

预期输出：
- 在 output/ 目录生成 report.md
- 打印报告预览
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import json


# ===== 简单模板引擎（f-string 方式）=====
def generate_markdown_with_fstring(results: Dict[str, Any],
                                    output_path: str = 'output/report.md') -> str:
    """
    使用 f-string 生成 Markdown 报告

    优点：
    - 简单直接，不需要额外库
    - Python 原生支持
    - 易于调试

    缺点：
    - 复杂模板维护困难
    - 不支持模板继承
    """
    print("\n使用 f-string 生成 Markdown...")

    # 从结果中提取数据
    data_info = results['data']
    test_results = results['tests']
    model_results = results['model']
    repro_info = results['reproducibility']

    # 构建 Markdown 内容
    markdown = f"""# 客户流失分析报告

> **报告生成时间**：{repro_info['execution_time']}
> **随机种子**：{repro_info['random_seed']}

---

## 可复现信息

本项目采用可复现报告流水线，任何人运行脚本都能得到相同结果。

- **数据来源**：模拟电商客户数据
- **样本数量**：{data_info['n_samples']} 个客户
- **流失率**：{data_info['churn_rate']:.1%}
- **随机种子**：{repro_info['random_seed']}
- **Python 版本**：{repro_info['python_version']}

### 依赖版本

"""

    # 添加依赖版本表格
    for lib, version in repro_info['dependencies'].items():
        markdown += f"- {lib}: {version}\n"

    markdown += "\n---\n\n"

    # 添加数据概览
    markdown += "## 数据概览\n\n"
    markdown += f"本分析包含 **{data_info['n_samples']}** 个客户样本，"
    markdown += f"其中 **{data_info['churn_rate']:.1%}** 的客户发生流失。\n\n"

    # 添加描述统计
    markdown += "## 描述统计\n\n"
    markdown += "### 数值摘要\n\n"

    desc_summary = results['descriptive']['summary']
    markdown += "| 指标 | 均值 | 标准差 | 最小值 | 中位数 | 最大值 |\n"
    markdown += "|------|------|--------|--------|--------|--------|\n"

    for idx in desc_summary.index:
        mean = desc_summary.loc[idx, 'mean']
        std = desc_summary.loc[idx, 'std']
        min_val = desc_summary.loc[idx, 'min']
        median = desc_summary.loc[idx, '50%']
        max_val = desc_summary.loc[idx, 'max']
        markdown += f"| {idx} | {mean:.2f} | {std:.2f} | {min_val:.2f} | {median:.2f} | {max_val:.2f} |\n"

    markdown += "\n"

    # 添加可视化
    markdown += "### 数据分布\n\n"
    markdown += "![使用时长分布](tenure_distribution.png)\n\n"
    markdown += "![消费金额分布](spend_distribution.png)\n\n"

    # 添加统计检验
    markdown += "## 统计检验\n\n"
    markdown += "我们使用统计检验来验证流失客户与留存客户在关键指标上的差异。\n\n"

    for i, (test_name, test_result) in enumerate(test_results.items(), 1):
        sig_text = "显著" if test_result['significant'] else "不显著"
        sig_mark = "✓" if test_result['significant'] else "✗"

        markdown += f"### 检验 {i}：{test_name} 差异\n\n"
        markdown += f"- **检验方法**：{test_result['test']}\n"
        markdown += f"- **统计量**：{test_result['statistic']:.4f}\n"
        markdown += f"- **p 值**：{test_result['p_value']:.4f}\n"
        markdown += f"- **结论**：差异 {sig_text} {sig_mark}\n\n"

    # 添加建模结果
    markdown += "## 建模与评估\n\n"
    markdown += f"我们使用逻辑回归模型预测客户流失。模型在测试集上的表现：\n\n"
    markdown += f"- **AUC**：{model_results['auc']:.3f}\n"
    markdown += f"- **准确率**：{model_results['accuracy']:.1%}\n\n"

    markdown += "### 特征系数\n\n"
    markdown += "| 特征 | 系数 | 解释 |\n"
    markdown += "|------|------|------|\n"

    for feat, coef in model_results['coefficients'].items():
        direction = "正相关（增加流失风险）" if coef > 0 else "负相关（降低流失风险）"
        markdown += f"| {feat} | {coef:+.4f} | {direction} |\n"

    markdown += "\n"

    # 添加结论
    markdown += "## 结论与建议\n\n"

    # 根据检验结果动态生成结论
    tenure_sig = test_results['tenure']['significant']
    spend_sig = test_results['spend']['significant']

    markdown += "### 主要发现\n\n"
    if tenure_sig:
        markdown += "1. **使用时长与流失显著相关**：流失客户的使用时长明显短于留存客户。\n"
    else:
        markdown += "1. **使用时长与流失无显著差异**：数据不支持使用时长预测流失。\n"

    if spend_sig:
        markdown += "2. **消费行为与流失显著相关**：流失客户的消费模式与留存客户存在差异。\n"
    else:
        markdown += "2. **消费行为与流失无显著差异**：需要进一步探索其他因素。\n"

    markdown += f"""
3. **模型预测能力**：逻辑回归模型的 AUC 为 {model_results['auc']:.3f}，
   表明模型具有一定的预测能力。

### 业务建议

1. **风险预警**：对于使用时长较短、消费模式异常的客户，建议进行主动干预。
2. **留存策略**：针对高风险客户设计个性化留存方案。
3. **持续监测**：定期更新模型，监测流失率变化趋势。

---

### 分析局限

1. **数据来源**：本分析使用模拟数据，实际应用中需验证结论的泛化性。
2. **因果推断**：统计检验只能确认相关性，不能证明因果关系。
3. **模型假设**：逻辑回归假设线性关系，实际关系可能更复杂。

---

*本报告由可复现分析流水线自动生成*
"""

    # 写入文件
    output_file = Path(output_path)
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(markdown)

    print(f"报告已保存到: {output_file}")

    return markdown


# ===== 使用 Jinja2 模板（更灵活的方式）=====
def generate_markdown_with_jinja2(results: Dict[str, Any],
                                   template_path: str = None,
                                   output_path: str = 'output/report_jinja.md') -> str:
    """
    使用 Jinja2 模板引擎生成 Markdown 报告

    优点：
    - 模板与代码分离
    - 支持条件渲染、循环
    - 支持模板继承
    - 更易维护大型报告

    需要安装：pip install jinja2
    """
    try:
        from jinja2 import Template
    except ImportError:
        print("Jinja2 未安装，使用 pip install jinja2 安装")
        return None

    print("\n使用 Jinja2 生成 Markdown...")

    # 定义模板（实际项目中可以放在单独的 .md 文件中）
    template_str = """# 客户流失分析报告

> **报告生成时间**：{{ repro_info.execution_time }}
> **随机种子**：{{ repro_info.random_seed }}

---

## 可复现信息

- **数据来源**：模拟电商客户数据
- **样本数量**：{{ data_info.n_samples }} 个客户
- **流失率**：{{ "%.1f"|format(data_info.churn_rate * 100) }}%
- **随机种子**：{{ repro_info.random_seed }}

### 依赖版本

{% for lib, version in repro_info.dependencies.items() -%}
- {{ lib }}: {{ version }}
{% endfor %}

---

## 数据概览

本分析包含 **{{ data_info.n_samples }}** 个客户样本，
其中 **{{ "%.1f"|format(data_info.churn_rate * 100) }}%** 的客户发生流失。

---

## 统计检验

{% for test_name, test_result in test_results.items() %}
### 检验 {{ loop.index }}：{{ test_name }} 差异

- **检验方法**：{{ test_result.test }}
- **统计量**：{{ "%.4f"|format(test_result.statistic) }}
- **p 值**：{{ "%.4f"|format(test_result.p_value) }}
- **结论**：差异 {% if test_result.significant %}显著 ✓{% else %}不显著 ✗{% endif %}

{% endfor %}

---

## 建模与评估

- **AUC**：{{ "%.3f"|format(model_results.auc) }}
- **准确率**：{{ "%.1f"|format(model_results.accuracy * 100) }}%

### 特征系数

| 特征 | 系数 | 解释 |
|------|------|------|
{% for feat, coef in model_results.coefficients.items() -%}
| {{ feat }} | {{ "%+.4f"|format(coef) }} | {% if coef > 0 %}正相关（增加流失风险）{% else %}负相关（降低流失风险）{% endif %} |
{% endfor %}

---

*本报告由可复现分析流水线自动生成*
"""

    # 渲染模板
    template = Template(template_str)
    markdown = template.render(
        data_info=type('obj', (object,), results['data'])(),
        test_results=results['tests'],
        model_results=type('obj', (object,), results['model'])(),
        repro_info=type('obj', (object,), results['reproducibility'])()
    )

    # 写入文件
    output_file = Path(output_path)
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(markdown)

    print(f"报告已保存到: {output_file}")

    return markdown


# ===== 生成表格辅助函数 =====
def generate_dataframe_table(df: pd.DataFrame,
                              title: str = "数据表",
                              float_format: str = "%.2f") -> str:
    """
    将 pandas DataFrame 转换为 Markdown 表格

    参数:
        df: 输入数据框
        title: 表格标题
        float_format: 浮点数格式

    返回:
        Markdown 表格字符串
    """
    markdown = f"### {title}\n\n"

    # 表头
    markdown += "| " + " | ".join(str(col) for col in df.columns) + " |\n"
    markdown += "| " + " | ".join("---" for _ in df.columns) + " |\n"

    # 数据行
    for idx, row in df.iterrows():
        row_str = []
        for val in row:
            if isinstance(val, float):
                row_str.append(float_format % val)
            else:
                row_str.append(str(val))
        markdown += "| " + " | ".join(row_str) + " |\n"

    markdown += "\n"
    return markdown


# ===== 生成图表引用 =====
def generate_figure_caption(fig_path: str,
                             caption: str,
                             width: str = "100%") -> str:
    """
    生成带标题的图片引用

    参数:
        fig_path: 图片相对路径
        caption: 图片标题
        width: 图片宽度（HTML 中有效）

    返回:
        Markdown 图片引用字符串
    """
    return f"""
![{caption}]({fig_path})

*{caption}*
"""


# ===== 生成置信区间表示 =====
def format_confidence_interval(mean: float,
                                ci_low: float,
                                ci_high: float,
                                precision: int = 2) -> str:
    """
    格式化置信区间为 Markdown

    示例输出：3.45 [95% CI: 2.10, 4.80]
    """
    return f"{mean:.{precision}f} \\[95% CI: {ci_low:.{precision}f}, {ci_high:.{precision}f}\\]"


# ===== 主函数 =====
def main() -> None:
    """演示报告生成流程"""
    print("=" * 60)
    print("Markdown 报告生成器")
    print("=" * 60)

    # 模拟分析结果（实际项目中从 16_report_pipeline.py 获取）
    mock_results = {
        'data': {
            'n_samples': 1000,
            'n_features': 3,
            'churn_rate': 0.2
        },
        'descriptive': {
            'summary': pd.DataFrame(
                [[24.5, 15.3, 1, 22, 72],
                 [85.2, 45.6, 10.5, 78.3, 250.0],
                 [2.1, 1.8, 0, 2, 8]],
                index=['tenure', 'monthly_spend', 'support_calls'],
                columns=['mean', 'std', 'min', '50%', 'max']
            )
        },
        'tests': {
            'tenure': {
                'test': 'Mann-Whitney U',
                'statistic': 95234.5,
                'p_value': 0.0001,
                'significant': True
            },
            'spend': {
                'test': 'Mann-Whitney U',
                'statistic': 112456.0,
                'p_value': 0.0032,
                'significant': True
            }
        },
        'model': {
            'coefficients': {
                'tenure': -0.0523,
                'monthly_spend': 0.0021,
                'support_calls': 0.2345
            },
            'auc': 0.782,
            'accuracy': 0.81
        },
        'reproducibility': {
            'execution_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'random_seed': 42,
            'python_version': '3.11.0',
            'dependencies': {
                'numpy': '1.24.0',
                'pandas': '2.0.0',
                'scikit-learn': '1.3.0'
            }
        }
    }

    # 方式 1：使用 f-string 生成报告
    print("\n方式 1：f-string 模板")
    print("-" * 40)
    markdown_fstring = generate_markdown_with_fstring(
        mock_results,
        output_path='output/report_fstring.md'
    )

    # 方式 2：使用 Jinja2 生成报告
    print("\n方式 2：Jinja2 模板")
    print("-" * 40)
    markdown_jinja = generate_markdown_with_jinja2(
        mock_results,
        output_path='output/report_jinja.md'
    )

    # 演示辅助函数
    print("\n辅助函数演示")
    print("-" * 40)

    # 表格生成
    sample_df = pd.DataFrame({
        '指标': ['AUC', '准确率', '精确率', '召回率'],
        '值': [0.78, 0.81, 0.75, 0.68]
    })
    table_md = generate_dataframe_table(sample_df, "模型性能指标")
    print("生成的表格：")
    print(table_md)

    # 置信区间格式化
    ci_str = format_confidence_interval(3.45, 2.10, 4.80)
    print(f"置信区间格式：{ci_str}")

    print("\n" + "=" * 60)
    print("报告生成完成")
    print("=" * 60)
    print("\n老潘说：'用脚本生成报告，手工编辑只会带来不一致。")
    print("数据更新后，重新跑一遍脚本，报告自动更新——这才是工程化。'")


if __name__ == "__main__":
    main()

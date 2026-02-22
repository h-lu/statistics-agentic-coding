"""
Solution Template for Week 16: From Analysis to Delivery

本模块提供报告生成、Markdown 转换、审计清单等功能的实现模板。

注意：这是一个模板文件，学生需要根据作业要求实现具体功能。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from pathlib import Path


# =============================================================================
# 数据加载功能
# =============================================================================

def load_data(file_path: str) -> pd.DataFrame:
    """
    加载数据文件

    Args:
        file_path: 数据文件路径（支持 .csv, .xlsx 等）

    Returns:
        包含数据的 DataFrame

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 文件格式不支持
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    # 根据扩展名选择读取方法
    if path.suffix == '.csv':
        return pd.read_csv(file_path)
    elif path.suffix in ['.xlsx', '.xls']:
        return pd.read_excel(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {path.suffix}")


# =============================================================================
# 描述统计功能
# =============================================================================

def compute_descriptive(data: pd.DataFrame) -> Dict[str, Any]:
    """
    计算描述统计量

    Args:
        data: 输入数据

    Returns:
        包含描述统计量的字典

    Raises:
        TypeError: 如果输入不是 DataFrame
    """
    if data is None:
        raise TypeError("输入数据不能为 None")

    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"输入必须是 pandas DataFrame，得到的是 {type(data)}")

    if len(data) == 0:
        return {}

    numeric_cols = data.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) == 0:
        return {'message': '没有数值列'}

    stats = {}

    for col in numeric_cols:
        col_data = data[col].dropna()
        stats[col] = {
            'count': len(col_data),
            'mean': float(col_data.mean()),
            'std': float(col_data.std()),
            'min': float(col_data.min()),
            'max': float(col_data.max()),
            'median': float(col_data.median()),
            'q25': float(col_data.quantile(0.25)),
            'q75': float(col_data.quantile(0.75))
        }

    return stats


# =============================================================================
# Markdown 生成功能
# =============================================================================

def generate_heading(text: str, level: int = 1) -> str:
    """
    生成 Markdown 标题

    Args:
        text: 标题文本
        level: 标题级别（1-6）

    Returns:
        Markdown 格式的标题
    """
    if level < 1 or level > 6:
        level = 1

    return '#' * level + ' ' + text


def generate_bold(text: str) -> str:
    """
    生成粗体文本

    Args:
        text: 要加粗的文本

    Returns:
        Markdown 格式的粗体文本
    """
    return f'**{text}**'


def generate_list(items: List[str], ordered: bool = False) -> str:
    """
    生成 Markdown 列表

    Args:
        items: 列表项
        ordered: 是否为有序列表

    Returns:
        Markdown 格式的列表
    """
    if ordered:
        return '\n'.join(f'{i+1}. {item}' for i, item in enumerate(items))
    else:
        return '\n'.join(f'- {item}' for item in items)


def generate_table(data: Union[pd.DataFrame, Dict], align: Optional[List[str]] = None) -> str:
    """
    生成 Markdown 表格

    Args:
        data: 数据（DataFrame 或字典）
        align: 列对齐方式（可选）

    Returns:
        Markdown 格式的表格
    """
    if isinstance(data, dict):
        # 如果是标量字典，需要提供索引
        try:
            df = pd.DataFrame(data)
        except ValueError:
            # 如果所有值都是标量，使用索引 [0]
            df = pd.DataFrame(data, index=[0])
    else:
        df = data.copy()

    if df.empty:
        return "| 空表 |\n| --- |"

    # 构建 Markdown 表格
    lines = []

    # 表头
    header = '| ' + ' | '.join(str(col) for col in df.columns) + ' |'
    lines.append(header)

    # 分隔符
    if align:
        sep = '| ' + ' | '.join(':---' if a == 'left' else
                                 '---:' if a == 'right' else
                                 ':---:' if a == 'center' else '---'
                                 for a in align) + ' |'
    else:
        sep = '| ' + ' | '.join('---' for _ in df.columns) + ' |'
    lines.append(sep)

    # 数据行
    for _, row in df.iterrows():
        row_str = '| ' + ' | '.join(str(val) if pd.notna(val) else '' for val in row) + ' |'
        lines.append(row_str)

    return '\n'.join(lines)


def generate_image(path: str, alt: str, title: Optional[str] = None) -> str:
    """
    生成 Markdown 图片链接

    Args:
        path: 图片路径
        alt: 替代文本
        title: 图片标题（可选）

    Returns:
        Markdown 格式的图片链接
    """
    if title:
        return f'![{alt}]({path} "{title}")'
    return f'![{alt}]({path})'


def generate_code_block(code: Union[str, List[str]], language: str = '') -> str:
    """
    生成 Markdown 代码块

    Args:
        code: 代码内容
        language: 语言标记

    Returns:
        Markdown 格式的代码块
    """
    if isinstance(code, list):
        code = '\n'.join(code)

    return f'```{language}\n{code}\n```'


# =============================================================================
# 报告生成功能
# =============================================================================

def render_markdown(content: Dict[str, Any]) -> str:
    """
    渲染 Markdown 报告

    Args:
        content: 报告内容字典

    Returns:
        Markdown 格式的报告
    """
    lines = []

    # 标题
    if 'title' in content:
        lines.append(generate_heading(content['title']))
        lines.append('')

    # 可复现信息
    lines.append(generate_heading('可复现信息', level=2))
    lines.append(generate_list([
        f"生成时间：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "随机种子：42",
        "依赖：pandas, numpy"
    ]))
    lines.append('')

    # 统计信息
    if 'statistics' in content:
        stats = content['statistics']
        lines.append(generate_heading('描述统计', level=2))

        if isinstance(stats, dict):
            for key, value in stats.items():
                lines.append(f"- {key}: {value}")

        lines.append('')

    return '\n'.join(lines)


def generate_report(data: pd.DataFrame, output_path: str, **kwargs) -> Optional[str]:
    """
    生成分析报告

    Args:
        data: 输入数据
        output_path: 输出文件路径
        **kwargs: 额外参数

    Returns:
        输出文件路径（成功时）或 None
    """
    # 计算描述统计
    stats = compute_descriptive(data)

    # 渲染报告
    content = {
        'title': kwargs.get('title', '分析报告'),
        'statistics': stats
    }

    markdown = render_markdown(content)

    # 写入文件
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(markdown)

    return output_path  # 返回文件路径而不是内容


# =============================================================================
# 审计清单功能
# =============================================================================

def audit_data_source(report_content: str) -> bool:
    """
    审计：检查数据来源说明

    Args:
        report_content: 报告内容

    Returns:
        是否包含数据来源说明
    """
    if not report_content or not isinstance(report_content, str):
        return False
    keywords = ['数据来源', 'source', 'dataset', 'kaggle', 'uci']
    return any(keyword in report_content.lower() for keyword in keywords)


def audit_random_seed(report_content: str) -> bool:
    """
    审计：检查随机种子说明

    Args:
        report_content: 报告内容

    Returns:
        是否说明随机种子
    """
    keywords = ['seed', '种子', 'random', 'np.random.seed']
    return any(keyword in report_content.lower() for keyword in keywords)


def audit_dependencies(report_content: str) -> bool:
    """
    审计：检查依赖版本记录

    Args:
        report_content: 报告内容

    Returns:
        是否记录依赖版本
    """
    keywords = ['version', '版本', 'pandas', 'numpy', 'scikit-learn']
    return any(keyword in report_content.lower() for keyword in keywords)


def audit_assumptions(report_content: str) -> bool:
    """
    审计：检查统计假设验证

    Args:
        report_content: 报告内容

    Returns:
        是否验证了假设
    """
    keywords = ['正态性', '方差齐性', '残差', 'assumption', 'shapiro', 'levene']
    return any(keyword in report_content.lower() for keyword in keywords)


def audit_confidence_intervals(report_content: str) -> bool:
    """
    审计：检查置信区间报告

    Args:
        report_content: 报告内容

    Returns:
        是否报告置信区间
    """
    keywords = ['置信区间', 'ci', 'confidence interval', '95%', '99%']
    return any(keyword in report_content.lower() for keyword in keywords)


def audit_uncertainty(report_content: str) -> bool:
    """
    审计：检查不确定性表达

    Args:
        report_content: 报告内容

    Returns:
        是否表达不确定性
    """
    keywords = ['不确定性', 'uncertainty', '置信区间', '误差', 'error', 'ci']
    return any(keyword in report_content.lower() for keyword in keywords)


def audit_causal_claims(report_content: str) -> bool:
    """
    审计：检查因果声明（过度自信）

    Args:
        report_content: 报告内容

    Returns:
        是否正确处理因果声明（True=正确，False=有问题）
    """
    # 检查过度自信的语言
    overconfident = ['证明', 'proves', '必然', 'certainly', '确定', '100%']
    has_overconfident = any(keyword in report_content.lower() for keyword in overconfident)

    # 检查谨慎的语言
    cautious = ['支持', 'suggests', '相关', 'associated', '可能', 'may']
    has_cautious = any(keyword in report_content.lower() for keyword in cautious)

    # 如果有谨慎语言或没有过度自信语言，则通过
    return has_cautious or not has_overconfident


def audit_sample_size(report_content: str) -> bool:
    """
    审计：检查样本量报告

    Args:
        report_content: 报告内容

    Returns:
        是否说明样本量
    """
    keywords = ['样本量', 'n=', 'sample size', '样本数', 'n =']
    return any(keyword in report_content.lower() for keyword in keywords)


def audit_missing_data(report_content: str) -> bool:
    """
    审计：检查缺失值说明

    Args:
        report_content: 报告内容

    Returns:
        是否说明缺失值处理
    """
    keywords = ['缺失', 'missing', 'na', 'nan', '插补', 'impute']
    return any(keyword in report_content.lower() for keyword in keywords)


def audit_report(report_content: str) -> Dict[str, bool]:
    """
    综合审计报告

    Args:
        report_content: 报告内容

    Returns:
        各项审计结果的字典
    """
    if report_content is None or not isinstance(report_content, str) or len(report_content.strip()) == 0:
        return {}

    return {
        'data_source': audit_data_source(report_content),
        'random_seed': audit_random_seed(report_content),
        'dependencies': audit_dependencies(report_content),
        'assumptions': audit_assumptions(report_content),
        'confidence_intervals': audit_confidence_intervals(report_content),
        'uncertainty': audit_uncertainty(report_content),
        'causal_claims': audit_causal_claims(report_content),
        'sample_size': audit_sample_size(report_content),
        'missing_data': audit_missing_data(report_content)
    }


def calculate_audit_score(report_content: str) -> float:
    """
    计算审计得分

    Args:
        report_content: 报告内容

    Returns:
        审计得分（0-100）
    """
    audit_results = audit_report(report_content)

    if not audit_results:
        return 0.0

    passed = sum(1 for v in audit_results.values() if v)
    total = len(audit_results)

    return (passed / total) * 100 if total > 0 else 0.0


def generate_audit_checklist(report_content: str) -> str:
    """
    生成审计清单

    Args:
        report_content: 报告内容

    Returns:
        Markdown 格式的审计清单
    """
    audit_results = audit_report(report_content)

    lines = []
    lines.append("## 审计清单")
    lines.append("")

    categories = {
        '可复现性': ['data_source', 'random_seed', 'dependencies'],
        '统计假设': ['assumptions', 'confidence_intervals'],
        '诚实性': ['uncertainty', 'causal_claims', 'sample_size', 'missing_data']
    }

    labels = {
        'data_source': '数据来源明确',
        'random_seed': '随机种子固定',
        'dependencies': '依赖版本记录',
        'assumptions': '假设验证',
        'confidence_intervals': '置信区间报告',
        'uncertainty': '不确定性表达',
        'causal_claims': '因果声明谨慎',
        'sample_size': '样本量说明',
        'missing_data': '缺失值处理说明'
    }

    for category, checks in categories.items():
        lines.append(f"### {category}")
        for check in checks:
            if check in audit_results:
                status = 'x' if audit_results[check] else ' '
                label = labels.get(check, check)
                lines.append(f"- [{status}] {label}")
        lines.append("")

    return '\n'.join(lines)


# =============================================================================
# 辅助功能
# =============================================================================

def convert_to_html(markdown_content: str) -> str:
    """
    将 Markdown 转换为 HTML（简化版）

    注意：这是一个简化实现，实际应用应使用 markdown 库

    Args:
        markdown_content: Markdown 内容

    Returns:
        HTML 内容
    """
    try:
        import markdown
        return markdown.markdown(markdown_content)
    except ImportError:
        # 简化的转换（仅处理一级标题）
        html = markdown_content
        lines = html.split('\n')
        if lines and lines[0].startswith('# '):
            lines[0] = '<h1>' + lines[0][2:] + '</h1>'
        return '\n'.join(lines)


# =============================================================================
# 端到端分析功能（占位符）
# =============================================================================

def run_customer_churn_analysis(data: pd.DataFrame, output_dir: str,
                                 random_seed: int = 42) -> Dict[str, Any]:
    """
    运行客户流失分析

    Args:
        data: 客户数据
        output_dir: 输出目录
        random_seed: 随机种子

    Returns:
        分析结果字典
    """
    np.random.seed(random_seed)

    # 计算基本统计
    stats = compute_descriptive(data)

    # 生成报告
    report_path = Path(output_dir) / "churn_report.md"
    generate_report(data, str(report_path), title='客户流失分析报告')

    return {
        'statistics': stats,
        'churn_rate': data['churn'].mean() if 'churn' in data.columns else None
    }


def run_ab_test_analysis(data: pd.DataFrame, group_col: str,
                         metric_col: str, output_dir: str) -> Dict[str, Any]:
    """
    运行 A/B 测试分析

    Args:
        data: A/B 测试数据
        group_col: 分组列名
        metric_col: 指标列名
        output_dir: 输出目录

    Returns:
        分析结果字典
    """
    groups = data.groupby(group_col)[metric_col].agg(['mean', 'count', 'std'])

    result = {
        'control_rate': groups.loc['control', 'mean'] if 'control' in groups.index else None,
        'treatment_rate': groups.loc['treatment', 'mean'] if 'treatment' in groups.index else None,
        'lift': None,
        'p_value': None,
        'ci_lower': None,
        'ci_upper': None
    }

    if result['control_rate'] and result['treatment_rate']:
        result['lift'] = (result['treatment_rate'] - result['control_rate']) / result['control_rate']

        # 简单的两比例 z-test 计算 p 值
        from scipy import stats as scipy_stats

        control_count = groups.loc['control', 'count']
        treatment_count = groups.loc['treatment', 'count']

        control_conv = groups.loc['control', 'mean']
        treatment_conv = groups.loc['treatment', 'mean']

        # 合并比例（用于标准误计算）
        pooled_p = (control_conv * control_count + treatment_conv * treatment_count) / (control_count + treatment_count)

        # 标准误
        se = np.sqrt(pooled_p * (1 - pooled_p) * (1/control_count + 1/treatment_count))

        # z 统计量
        if se > 0:
            z = (treatment_conv - control_conv) / se
            # 双尾检验的 p 值
            result['p_value'] = float(2 * (1 - scipy_stats.norm.cdf(abs(z))))

            # 95% 置信区间
            diff = treatment_conv - control_conv
            margin = 1.96 * np.sqrt(control_conv * (1 - control_conv) / control_count +
                                    treatment_conv * (1 - treatment_conv) / treatment_count)
            result['ci_lower'] = float(diff - margin)
            result['ci_upper'] = float(diff + margin)

    return result


def generate_full_report(data: pd.DataFrame, output_path: str,
                         include_metadata: bool = True) -> str:
    """
    生成完整报告

    Args:
        data: 输入数据
        output_path: 输出路径
        include_metadata: 是否包含元数据

    Returns:
        报告文件路径
    """
    return generate_report(data, output_path, title='完整分析报告')

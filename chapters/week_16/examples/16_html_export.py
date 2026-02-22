"""
示例：HTML 报告导出

本例演示如何将 Markdown 报告转换为 HTML 格式。
HTML 版本的优势：
1. 可交互（可嵌入交互式图表）
2. 易于分享（可部署到网页）
3. 专业外观（可用 CSS 自定义样式）

运行方式：python3 chapters/week_16/examples/16_html_export.py

预期输出：
- 在 output/ 目录生成 report.html
- 演示多种转换方式
"""
from __future__ import annotations

import subprocess
from pathlib import Path
import sys


# ===== 方式 1：使用 Pandoc 命令行 =====
def convert_with_pandoc(markdown_path: str,
                        html_path: str,
                        css_path: str = None,
                        standalone: bool = True) -> bool:
    """
    使用 Pandoc 将 Markdown 转换为 HTML

    Pandoc 是最通用的文档转换工具，支持：
    - Markdown → HTML/PDF/Word/幻灯片
    - 语法高亮、数学公式、表格
    - 自定义 CSS 样式

    安装：apt-get install pandoc  或  brew install pandoc

    参数:
        markdown_path: Markdown 文件路径
        html_path: 输出 HTML 文件路径
        css_path: 可选的 CSS 样式文件路径
        standalone: 是否生成完整的 HTML（包含 <head> 等）

    返回:
        转换是否成功
    """
    print("\n方式 1：使用 Pandoc 转换")
    print("-" * 40)

    # 检查 pandoc 是否安装
    try:
        result = subprocess.run(['pandoc', '--version'],
                                capture_output=True,
                                text=True)
        if result.returncode != 0:
            print("Pandoc 未安装，请安装后重试")
            print("  Ubuntu: sudo apt-get install pandoc")
            print("  macOS: brew install pandoc")
            return False
        print(f"Pandoc 版本: {result.stdout.split()[1]}")
    except FileNotFoundError:
        print("Pandoc 未安装，请安装后重试")
        return False

    # 构建 pandoc 命令
    cmd = ['pandoc', markdown_path, '-o', html_path]

    if standalone:
        cmd.append('--standalone')

    if css_path:
        cmd.extend(['--css', css_path])

    # 添加其他选项
    cmd.extend([
        '--toc',  # 生成目录
        '--toc-depth=2',  # 目录深度
        '--highlight-style=pygments',  # 代码高亮样式
        '--metadata=lang:zh-CN'  # 指定语言
    ])

    print(f"执行命令: {' '.join(cmd)}")

    # 执行转换
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"转换成功: {html_path}")
            return True
        else:
            print(f"转换失败: {result.stderr}")
            return False
    except Exception as e:
        print(f"转换出错: {e}")
        return False


# ===== 方式 2：使用 Python markdown 库 =====
def convert_with_python_markdown(markdown_path: str,
                                  html_path: str,
                                  css_style: str = None) -> bool:
    """
    使用 Python 的 markdown 库转换

    优点：
    - 不需要外部工具
    - 可以扩展自定义功能
    - 适合集成到 Python 脚本中

    需要安装：pip install markdown

    参数:
        markdown_path: Markdown 文件路径
        html_path: 输出 HTML 文件路径
        css_style: 内联 CSS 样式字符串

    返回:
        转换是否成功
    """
    print("\n方式 2：使用 Python markdown 库")
    print("-" * 40)

    try:
        import markdown
    except ImportError:
        print("markdown 库未安装")
        print("请运行: pip install markdown")
        return False

    # 读取 Markdown 文件
    md_path = Path(markdown_path)
    if not md_path.exists():
        print(f"Markdown 文件不存在: {markdown_path}")
        return False

    with open(md_path, 'r', encoding='utf-8') as f:
        md_content = f.read()

    # 转换为 HTML
    md = markdown.Markdown(extensions=[
        'tables',  # 支持表格
        'fenced_code',  # 支持代码块
        'toc',  # 支持目录
        'nl2br',  # 换行转 <br>
        'sane_lists'  # 更好的列表支持
    ])

    html_body = md.convert(md_content)

    # 生成完整 HTML
    if css_style is None:
        # 默认样式
        css_style = """
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                max-width: 800px;
                margin: 40px auto;
                padding: 0 20px;
                line-height: 1.6;
                color: #333;
            }
            h1, h2, h3 {
                color: #2c3e50;
                margin-top: 30px;
            }
            table {
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
            }
            th, td {
                border: 1px solid #ddd;
                padding: 8px 12px;
                text-align: left;
            }
            th {
                background-color: #f2f2f2;
            }
            code {
                background-color: #f4f4f4;
                padding: 2px 6px;
                border-radius: 3px;
            }
            pre {
                background-color: #f4f4f4;
                padding: 15px;
                border-radius: 5px;
                overflow-x: auto;
            }
            blockquote {
                border-left: 4px solid #ddd;
                margin: 0;
                padding-left: 20px;
                color: #666;
            }
        </style>
        """

    html_template = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>分析报告</title>
    {css_style}
</head>
<body>
{html_body}
</body>
</html>
"""

    # 写入 HTML 文件
    html_path_obj = Path(html_path)
    html_path_obj.parent.mkdir(exist_ok=True)

    with open(html_path_obj, 'w', encoding='utf-8') as f:
        f.write(html_template)

    print(f"转换成功: {html_path}")
    return True


# ===== 方式 3：使用 WeasyPrint 生成带样式的 PDF =====
def convert_to_pdf_with_weasyprint(markdown_path: str,
                                    pdf_path: str) -> bool:
    """
    将 Markdown 转换为 PDF（使用 WeasyPrint）

    优点：
    - 生成高质量 PDF
    - 支持 CSS 样式
    - 适合打印和分享

    需要安装：pip install weasyprint

    参数:
        markdown_path: Markdown 文件路径
        pdf_path: 输出 PDF 文件路径

    返回:
        转换是否成功
    """
    print("\n方式 3：使用 WeasyPrint 生成 PDF")
    print("-" * 40)

    try:
        from weasyprint import HTML, CSS
    except ImportError:
        print("WeasyPrint 未安装")
        print("请运行: pip install weasyprint")
        return False

    # 先用 markdown 库转换
    try:
        import markdown
    except ImportError:
        print("markdown 库未安装")
        print("请运行: pip install markdown")
        return False

    # 读取 Markdown
    with open(markdown_path, 'r', encoding='utf-8') as f:
        md_content = f.read()

    # 转换为 HTML
    md = markdown.Markdown(extensions=['tables', 'fenced_code'])
    html_body = md.convert(md_content)

    # 完整 HTML（带打印样式）
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        @page {{
            size: A4;
            margin: 2cm;
        }}
        body {{
            font-family: "DejaVu Sans", sans-serif;
            font-size: 11pt;
            line-height: 1.5;
        }}
        h1 {{ page-break-before: always; }}
        h1:first-of-type {{ page-break-before: auto; }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 1em 0;
        }}
        th, td {{
            border: 1px solid #ccc;
            padding: 0.5em;
        }}
        th {{
            background-color: #eee;
        }}
    </style>
</head>
<body>
{html_body}
</body>
</html>
"""

    # 生成 PDF
    try:
        HTML(string=html_content).write_pdf(pdf_path)
        print(f"PDF 生成成功: {pdf_path}")
        return True
    except Exception as e:
        print(f"PDF 生成失败: {e}")
        return False


# ===== 创建 CSS 样式文件 =====
def create_sample_css(css_path: str = 'output/report.css') -> str:
    """
    创建示例 CSS 样式文件

    可用于 Pandoc 转换时自定义样式
    """
    css_content = """
/* 分析报告样式 */

:root {
    --primary-color: #3498db;
    --secondary-color: #2c3e50;
    --accent-color: #e74c3c;
    --text-color: #333;
    --light-bg: #f8f9fa;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 "Helvetica Neue", Arial, "Noto Sans SC", sans-serif;
    max-width: 900px;
    margin: 0 auto;
    padding: 40px 20px;
    line-height: 1.8;
    color: var(--text-color);
    background-color: #fff;
}

/* 标题 */
h1, h2, h3, h4 {
    color: var(--secondary-color);
    margin-top: 1.5em;
    margin-bottom: 0.8em;
    font-weight: 600;
}

h1 {
    font-size: 2.5em;
    border-bottom: 3px solid var(--primary-color);
    padding-bottom: 0.3em;
}

h2 {
    font-size: 1.8em;
    border-bottom: 1px solid #ddd;
    padding-bottom: 0.2em;
}

/* 引用块 */
blockquote {
    border-left: 4px solid var(--primary-color);
    margin: 1.5em 0;
    padding: 0.5em 1.5em;
    background-color: var(--light-bg);
    color: #555;
}

/* 代码 */
code {
    background-color: #f4f4f4;
    padding: 0.2em 0.4em;
    border-radius: 3px;
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
    font-size: 0.9em;
}

pre {
    background-color: #2c3e50;
    color: #ecf0f1;
    padding: 1.5em;
    border-radius: 5px;
    overflow-x: auto;
}

pre code {
    background: none;
    padding: 0;
    color: inherit;
}

/* 表格 */
table {
    border-collapse: collapse;
    width: 100%;
    margin: 1.5em 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

th, td {
    border: 1px solid #ddd;
    padding: 12px 15px;
    text-align: left;
}

th {
    background-color: var(--primary-color);
    color: white;
    font-weight: 500;
}

tr:nth-child(even) {
    background-color: var(--light-bg);
}

tr:hover {
    background-color: #e8f4f8;
}

/* 链接 */
a {
    color: var(--primary-color);
    text-decoration: none;
    border-bottom: 1px solid transparent;
    transition: border-bottom 0.3s;
}

a:hover {
    border-bottom-color: var(--primary-color);
}

/* 图片 */
img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 1.5em auto;
    border-radius: 4px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

/* 列表 */
ul, ol {
    padding-left: 2em;
}

li {
    margin: 0.5em 0;
}

/* 分隔线 */
hr {
    border: none;
    border-top: 2px solid #ddd;
    margin: 3em 0;
}

/* 打印样式 */
@media print {
    body {
        max-width: 100%;
        padding: 0;
    }
    h1 {
        page-break-before: always;
    }
    h1:first-of-type {
        page-break-before: auto;
    }
    a {
        color: #000;
        text-decoration: underline;
    }
    pre, blockquote {
        page-break-inside: avoid;
    }
}
"""

    css_file = Path(css_path)
    css_file.parent.mkdir(exist_ok=True)
    with open(css_file, 'w', encoding='utf-8') as f:
        f.write(css_content)

    print(f"CSS 样式文件已创建: {css_path}")
    return css_path


# ===== 演示转换流程 =====
def demo_conversion_workflow():
    """
    演示完整的报告转换工作流
    """
    print("=" * 60)
    print("HTML 报告导出演示")
    print("=" * 60)

    # 首先创建一个示例 Markdown 文件
    sample_md = """# 客户流失分析报告

> **报告生成时间**：2026-02-21
> **随机种子**：42

## 数据概览

本分析包含 1000 个客户样本，其中 20% 的客户发生流失。

### 描述统计

| 指标 | 均值 | 标准差 |
|------|------|--------|
| 使用时长 | 24.5 | 15.3 |
| 月消费 | 85.2 | 45.6 |
| 客服联系 | 2.1 | 1.8 |

## 统计检验

- **使用时长差异**：Mann-Whitney U, p < 0.001（显著）
- **消费金额差异**：Mann-Whitney U, p = 0.003（显著）

## 结论

使用时长和消费行为与客户流失显著相关，建议针对高风险客户进行主动干预。

---

*本报告由可复现分析流水线自动生成*
"""

    # 写入示例 Markdown
    md_path = Path('output/sample_report.md')
    md_path.parent.mkdir(exist_ok=True)
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(sample_md)
    print(f"示例 Markdown 已创建: {md_path}")

    # 创建 CSS 样式
    css_path = create_sample_css()

    # 方式 1：Pandoc
    html_path_pandoc = 'output/report_pandoc.html'
    success_pandoc = convert_with_pandoc(
        str(md_path),
        html_path_pandoc,
        css_path=str(css_path)
    )

    # 方式 2：Python markdown
    html_path_python = 'output/report_python.html'
    success_python = convert_with_python_markdown(
        str(md_path),
        html_path_python
    )

    # 总结
    print("\n" + "=" * 60)
    print("转换完成")
    print("=" * 60)

    if success_pandoc:
        print(f"✓ Pandoc 转换成功: {html_path_pandoc}")
    else:
        print("✗ Pandoc 转换失败（可能未安装 Pandoc）")

    if success_python:
        print(f"✓ Python 转换成功: {html_path_python}")

    print("\n老潘说：'在公司里我们用 Pandoc 做文档转换，")
    print("因为它最可靠、格式最一致。但对于简单的 HTML，")
    print("Python 的 markdown 库也够用了。'")


# ===== 主函数 =====
def main() -> None:
    """运行 HTML 导出演示"""
    demo_conversion_workflow()

    print("\n" + "=" * 60)
    print("使用建议")
    print("=" * 60)
    print("""
1. **开发阶段**：用 Python markdown 库（快速迭代）
2. **正式交付**：用 Pandoc（格式稳定、功能完整）
3. **PDF 交付**：用 WeasyPrint 或 Pandoc → PDF
4. **在线分享**：部署 HTML 到 GitHub Pages 或内部服务器

安装命令：
  pip install markdown weasyprint
  sudo apt-get install pandoc  (Ubuntu)
  brew install pandoc           (macOS)
    """)


if __name__ == "__main__":
    main()

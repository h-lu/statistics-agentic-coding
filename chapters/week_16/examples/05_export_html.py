"""
示例：从 Markdown 导出 HTML 展示版

本例演示如何将 report.md 转换为 report.html，
支持 pandoc 和 markdown 库两种方式。

运行方式：python3 chapters/week_16/examples/05_export_html.py
预期输出：生成 report/report.html（使用 pandoc 或 markdown 库）
"""
from __future__ import annotations

import subprocess
from pathlib import Path


def check_pandoc_installed() -> bool:
    """
    检查 pandoc 是否已安装

    返回：
        bool: pandoc 是否可用
    """
    try:
        result = subprocess.run(
            ["pandoc", "--version"],
            check=True,
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            # 提取版本号
            version_line = result.stdout.split('\n')[0]
            print(f"✓ 检测到 pandoc: {version_line}")
            return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    return False


def export_with_pandoc(markdown_path: str, output_path: str,
                      css_path: str = None) -> bool:
    """
    使用 pandoc 导出 Markdown 为 HTML

    参数：
        markdown_path: Markdown 文件路径
        output_path: 输出 HTML 文件路径
        css_path: 自定义 CSS 样式文件路径（可选）

    返回：
        bool: 是否成功导出
    """
    cmd = [
        "pandoc",
        markdown_path,
        "-o", output_path,
        "--standalone",  # 独立 HTML（包含 CSS）
        "--toc",  # 目录
        "--toc-depth=2",  # 目录深度
        "--highlight-style=pygments",  # 代码高亮
    ]

    # 添加自定义 CSS（如果提供）
    if css_path and Path(css_path).exists():
        cmd.extend(["--css", css_path])
        print(f"✓ 使用自定义样式: {css_path}")

    print(f"执行命令: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
        print(f"✅ HTML 已导出（使用 pandoc）: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ pandoc 导出失败: {e}")
        return False
    except FileNotFoundError:
        print(f"❌ 未找到 pandoc: {e}")
        return False


def export_with_markdown_library(markdown_path: str, output_path: str) -> bool:
    """
    使用 markdown 库导出 Markdown 为 HTML

    参数：
        markdown_path: Markdown 文件路径
        output_path: 输出 HTML 文件路径

    返回：
        bool: 是否成功导出
    """
    try:
        import markdown
    except ImportError:
        print("❌ 未安装 markdown 库")
        print("建议安装: pip install markdown")
        return False

    # 读取 Markdown 内容
    md_file = Path(markdown_path)
    if not md_file.exists():
        print(f"❌ Markdown 文件不存在: {markdown_path}")
        return False

    try:
        with open(md_file, "r", encoding="utf-8") as f:
            md_content = f.read()
    except Exception as e:
        print(f"❌ 读取 Markdown 失败: {e}")
        return False

    # 转换为 HTML
    try:
        html_content = markdown.markdown(
            md_content,
            extensions=['tables', 'fenced_code', 'codehilite', 'toc']
        )
    except Exception as e:
        print(f"❌ Markdown 转换失败: {e}")
        return False

    # 生成完整的 HTML 文档
    full_html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>StatLab 终稿报告</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/github-markdown-css@5.6.1/github-markdown.min.css">
    <style>
        body {{
            max-width: 900px;
            margin: 2rem auto;
            padding: 0 1rem;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
            line-height: 1.6;
        }}
        .markdown-body {{
            box-sizing: border-box;
            min-width: 200px;
            max-width: 980px;
            margin: 0 auto;
            padding: 45px;
        }}
        .markdown-body img {{
            max-width: 100%;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
        }}
        table th, table td {{
            border: 1px solid #ddd;
            padding: 8px;
        }}
        table th {{
            background-color: #f2f2f2;
            text-align: left;
        }}
    </style>
</head>
<body>
    <article class="markdown-body">
        {html_content}
    </article>
</body>
</html>
"""

    # 保存 HTML 文件
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(full_html)
        print(f"✅ HTML 已导出（使用 markdown 库）: {output_path}")
        return True
    except Exception as e:
        print(f"❌ 保存 HTML 失败: {e}")
        return False


def export_to_html(markdown_path: str = "report/report.md",
                   output_path: str = "report/report.html",
                   css_path: str = None) -> bool:
    """
    导出 Markdown 为 HTML（自动选择可用方法）

    参数：
        markdown_path: Markdown 文件路径
        output_path: 输出 HTML 文件路径
        css_path: 自定义 CSS 样式文件路径（可选）

    返回：
        bool: 是否成功导出
    """
    # 检查 Markdown 文件是否存在
    md_file = Path(markdown_path)
    if not md_file.exists():
        print(f"❌ Markdown 文件不存在: {markdown_path}")
        return False

    print("\n" + "=" * 50)
    print("Markdown 到 HTML 导出工具")
    print("=" * 50)
    print(f"\n输入文件: {markdown_path}")
    print(f"输出文件: {output_path}\n")

    # 优先使用 pandoc（功能更强大）
    print("步骤 1: 尝试使用 pandoc...")
    if check_pandoc_installed():
        print("\n步骤 2: 使用 pandoc 导出...")
        return export_with_pandoc(markdown_path, output_path, css_path)

    # 备选：使用 markdown 库
    print("\n步骤 2: pandoc 不可用，尝试使用 markdown 库...")
    print("建议: 安装 pandoc 以获得更好的支持（数学公式、代码高亮等）")
    print("      https://pandoc.org/installing.html\n")

    return export_with_markdown_library(markdown_path, output_path)


def generate_custom_css(output_path: str = "report/style.css"):
    """
    生成自定义 CSS 样式文件

    参数：
        output_path: CSS 文件路径

    返回：
        bool: 是否成功生成
    """
    css_content = """/* StatLab 报告自定义样式 */

body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    line-height: 1.6;
    color: #333;
    max-width: 900px;
    margin: 0 auto;
    padding: 2rem;
}

h1, h2, h3, h4, h5, h6 {
    margin-top: 1.5em;
    margin-bottom: 0.5em;
    font-weight: 600;
}

h1 {
    font-size: 2.5em;
    border-bottom: 2px solid #eee;
    padding-bottom: 0.3em;
}

h2 {
    font-size: 2em;
    border-bottom: 1px solid #eee;
    padding-bottom: 0.3em;
}

code {
    background-color: #f6f8fa;
    padding: 0.2em 0.4em;
    border-radius: 3px;
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
    font-size: 85%;
}

pre {
    background-color: #f6f8fa;
    padding: 16px;
    border-radius: 6px;
    overflow: auto;
}

pre code {
    background-color: transparent;
    padding: 0;
}

table {
    border-collapse: collapse;
    width: 100%;
    margin: 1em 0;
}

table th, table td {
    border: 1px solid #ddd;
    padding: 8px 12px;
    text-align: left;
}

table th {
    background-color: #f2f2f2;
    font-weight: 600;
}

blockquote {
    border-left: 4px solid #ddd;
    padding-left: 1em;
    margin-left: 0;
    color: #666;
}

img {
    max-width: 100%;
    height: auto;
}
"""

    css_file = Path(output_path)
    css_file.parent.mkdir(parents=True, exist_ok=True)

    with open(css_file, "w", encoding="utf-8") as f:
        f.write(css_content)

    print(f"✅ CSS 样式已生成: {output_path}")
    return True


def print_conversion_tips():
    """打印转换工具建议"""
    print("\n" + "=" * 50)
    print("HTML 导出工具对比：")
    print("=" * 50)
    print("""
pandoc（推荐）:
  优点:
    - 支持数学公式（LaTeX）
    - 更好的代码高亮
    - 支持更多 Markdown 扩展
    - 可导出为 PDF、PPTX 等多种格式
  安装: https://pandoc.org/installing.html

markdown 库（备选）:
  优点:
    - 纯 Python 实现
    - 无需安装额外软件
  缺点:
    - 数学公式支持有限
    - 代码高亮较弱
  安装: pip install markdown
    """)


def main():
    """执行完整的 HTML 导出流程"""
    print("\n" + "=" * 50)
    print("StatLab HTML 导出工具")
    print("=" * 50)

    # 1. 生成自定义 CSS
    print("\n步骤 1: 生成自定义 CSS 样式...")
    generate_custom_css()

    # 2. 导出 HTML
    print("\n步骤 2: 导出 Markdown 为 HTML...")
    success = export_to_html(
        markdown_path="report/report.md",
        output_path="report/report.html",
        css_path="report/style.css"
    )

    # 3. 打印转换建议
    print_conversion_tips()

    # 4. 总结
    print("\n" + "=" * 50)
    if success:
        print("✅ HTML 导出完成")
        print("=" * 50)
        print("\n建议：")
        print("  1. 在浏览器中打开 report/report.html")
        print("  2. 检查样式和内容是否正确")
        print("  3. 如需修改样式，编辑 report/style.css")
        print("  4. 如需导出为 PDF，使用 pandoc:")
        print("     pandoc report/report.md -o report.pdf --pdf-engine=xelatex")
    else:
        print("❌ HTML 导出失败")
        print("=" * 50)
        print("\n可能的原因：")
        print("  1. Markdown 文件不存在")
        print("  2. 未安装 pandoc 或 markdown 库")
        print("\n解决方案：")
        print("  - 安装 pandoc: https://pandoc.org/installing.html")
        print("  - 或安装 markdown 库: pip install markdown")


if __name__ == "__main__":
    main()

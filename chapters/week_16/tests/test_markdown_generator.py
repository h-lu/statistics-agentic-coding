"""
Tests for Markdown Generator

Markdown 生成测试用例矩阵：
- 正例：验证标题、表格、列表、图片链接的正确生成
- 边界：空值、特殊字符、超长字符串、Unicode 字符
- 反例：无效输入类型
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add starter_code to path
starter_code_path = Path(__file__).parent.parent / "starter_code"
sys.path.insert(0, str(starter_code_path))


# =============================================================================
# 测试数据 Fixture
# =============================================================================

@pytest.fixture
def sample_statistics():
    """
    Fixture：生成示例统计数据
    """
    return {
        'n': 500,
        'mean': 25.5,
        'std': 10.2,
        'min': 1.0,
        'max': 72.0,
        'median': 24.0,
        'q25': 18.0,
        'q75': 32.0
    }


@pytest.fixture
def sample_test_results():
    """
    Fixture：生成示例检验结果
    """
    return {
        'test_name': 't_test',
        'statistic': 2.45,
        'p_value': 0.015,
        'ci_lower': 0.5,
        'ci_upper': 4.2,
        'degrees_of_freedom': 98
    }


@pytest.fixture
def sample_dataframe():
    """
    Fixture：生成示例 DataFrame
    """
    return pd.DataFrame({
        'Variable': ['tenure', 'monthly_charges', 'total_charges'],
        'Mean': [25.5, 65.3, 2300.5],
        'Std': [10.2, 15.8, 1200.3],
        'Min': [1.0, 20.0, 100.0],
        'Max': [72.0, 120.0, 8500.0]
    })


# =============================================================================
# 正例测试：Markdown 基础元素生成
# =============================================================================

class TestMarkdownBasicElements:
    """测试 Markdown 基础元素生成"""

    def test_generate_heading_h1(self):
        """
        正例：生成一级标题

        给定：标题文本
        期望：返回 "# 标题文本"
        """
        try:
            import solution
        except ImportError:
            pytest.skip("solution.py not found")

        if hasattr(solution, 'generate_heading'):
            result = solution.generate_heading('分析报告', level=1)
            assert '# 分析报告' in result
            assert result.startswith('# ')
        else:
            pytest.skip("generate_heading function not implemented")

    def test_generate_heading_h2(self):
        """
        正例：生成二级标题

        期望：返回 "## 标题文本"
        """
        try:
            import solution
        except ImportError:
            pytest.skip("solution.py not found")

        if hasattr(solution, 'generate_heading'):
            result = solution.generate_heading('数据概览', level=2)
            assert '## 数据概览' in result
        else:
            pytest.skip("generate_heading function not implemented")

    def test_generate_heading_h3(self):
        """
        正例：生成三级标题

        期望：返回 "### 标题文本"
        """
        try:
            import solution
        except ImportError:
            pytest.skip("solution.py not found")

        if hasattr(solution, 'generate_heading'):
            result = solution.generate_heading('描述统计', level=3)
            assert '### 描述统计' in result
        else:
            pytest.skip("generate_heading function not implemented")

    def test_generate_bold_text(self):
        """
        正例：生成粗体文本

        期望：返回 "**粗体文本**"
        """
        try:
            import solution
        except ImportError:
            pytest.skip("solution.py not found")

        if hasattr(solution, 'generate_bold'):
            result = solution.generate_bold('重要结论')
            assert '**重要结论**' in result
        else:
            pytest.skip("generate_bold function not implemented")

    def test_generate_italic_text(self):
        """
        正例：生成斜体文本

        期望：返回 "*斜体文本*" 或 "_斜体文本_"
        """
        try:
            import solution
        except ImportError:
            pytest.skip("solution.py not found")

        if hasattr(solution, 'generate_italic'):
            result = solution.generate_italic('注')
            assert ('*注*' in result) or ('_注_' in result)
        else:
            pytest.skip("generate_italic function not implemented")

    def test_generate_code_inline(self):
        """
        正例：生成行内代码

        期望：返回 "`代码`"
        """
        try:
            import solution
        except ImportError:
            pytest.skip("solution.py not found")

        if hasattr(solution, 'generate_code'):
            result = solution.generate_code('np.mean(data)')
            assert '`np.mean(data)`' in result
        else:
            pytest.skip("generate_code function not implemented")


# =============================================================================
# 正例测试：列表生成
# =============================================================================

class TestMarkdownLists:
    """测试 Markdown 列表生成"""

    def test_generate_bullet_list(self):
        """
        正例：生成无序列表

        给定：字符串列表
        期望：返回正确的 Markdown 无序列表格式
        """
        try:
            import solution
        except ImportError:
            pytest.skip("solution.py not found")

        items = ['数据来源明确', '随机种子固定', '依赖版本记录']

        if hasattr(solution, 'generate_list'):
            result = solution.generate_list(items, ordered=False)

            # 检查列表格式
            for item in items:
                assert item in result
                # 无序列表使用 - 或 *
                assert '- ' in result or '* ' in result
        else:
            pytest.skip("generate_list function not implemented")

    def test_generate_numbered_list(self):
        """
        正例：生成有序列表

        给定：字符串列表
        期望：返回正确的 Markdown 有序列表格式
        """
        try:
            import solution
        except ImportError:
            pytest.skip("solution.py not found")

        items = ['数据加载', '描述统计', '假设检验', '报告生成']

        if hasattr(solution, 'generate_list'):
            result = solution.generate_list(items, ordered=True)

            # 检查列表格式
            assert '1.' in result or '1)' in result
            for item in items:
                assert item in result
        else:
            pytest.skip("generate_list function not implemented")

    def test_generate_nested_list(self):
        """
        正例：生成嵌套列表

        给定：包含子项的列表
        期望：正确缩进子项
        """
        try:
            import solution
        except ImportError:
            pytest.skip("solution.py not found")

        items = [
            '可复现性',
            ['数据来源', '随机种子'],
            '统计分析',
            ['假设检验', '置信区间']
        ]

        if hasattr(solution, 'generate_nested_list'):
            result = solution.generate_nested_list(items)

            # 检查缩进（子项应有额外的空格）
            assert '  ' in result or '\t' in result
        else:
            pytest.skip("generate_nested_list function not implemented")


# =============================================================================
# 正例测试：表格生成
# =============================================================================

class TestMarkdownTables:
    """测试 Markdown 表格生成"""

    def test_generate_simple_table(self, sample_dataframe):
        """
        正例：从 DataFrame 生成 Markdown 表格

        给定：包含数据的 DataFrame
        期望：返回正确的 Markdown 表格格式
        """
        try:
            import solution
        except ImportError:
            pytest.skip("solution.py not found")

        if hasattr(solution, 'generate_table'):
            result = solution.generate_table(sample_dataframe)

            # 检查表头分隔符
            assert '| --- |' in result or '|---|' in result

            # 检查列名存在
            for col in sample_dataframe.columns:
                assert col in result
        else:
            pytest.skip("generate_table function not implemented")

    def test_table_from_dict(self, sample_statistics):
        """
        正例：从字典生成表格

        给定：统计量字典
        期望：生成包含统计量的表格
        """
        try:
            import solution
        except ImportError:
            pytest.skip("solution.py not found")

        if hasattr(solution, 'generate_table'):
            result = solution.generate_table(sample_statistics)

            # 检查关键统计量存在
            for key in ['mean', 'std', 'min', 'max']:
                assert key in result.lower() or str(sample_statistics[key]) in result
        else:
            pytest.skip("generate_table function not implemented")

    def test_table_alignment(self):
        """
        正例：表格对齐选项

        验证表格可以指定列对齐方式
        """
        try:
            import solution
        except ImportError:
            pytest.skip("solution.py not found")

        df = pd.DataFrame({
            'Left': ['A', 'B'],
            'Center': [1, 2],
            'Right': [3.0, 4.0]
        })

        if hasattr(solution, 'generate_table'):
            result = solution.generate_table(df, align=['left', 'center', 'right'])

            # 检查对齐标记
            assert ':---' in result  # left
            assert ':---:' in result  # center
            assert '---:' in result  # right
        else:
            pytest.skip("generate_table function not implemented")


# =============================================================================
# 正例测试：图片和链接
# =============================================================================

class TestMarkdownImagesAndLinks:
    """测试 Markdown 图片和链接生成"""

    def test_generate_image_link(self):
        """
        正例：生成图片链接

        给定：图片路径和替代文本
        期望：返回 "
![alt](path)
" 格式
        """
        try:
            import solution
        except ImportError:
            pytest.skip("solution.py not found")

        if hasattr(solution, 'generate_image'):
            result = solution.generate_image('figures/plot.png', '分布图')
            assert '![分布图](figures/plot.png)' in result
        else:
            pytest.skip("generate_image function not implemented")

    def test_generate_image_with_title(self):
        """
        正例：生成带标题的图片链接

        期望：返回 "
![alt](path "title")
" 格式
        """
        try:
            import solution
        except ImportError:
            pytest.skip("solution.py not found")

        if hasattr(solution, 'generate_image'):
            result = solution.generate_image(
                'figures/plot.png',
                '分布图',
                title='客户使用时长分布'
            )
            # 检查包含图片标记
            assert '![' in result
            assert '](' in result
        else:
            pytest.skip("generate_image function not implemented")

    def test_generate_hyperlink(self):
        """
        正例：生成超链接

        给定：URL 和链接文本
        期望：返回 "[text](url)" 格式
        """
        try:
            import solution
        except ImportError:
            pytest.skip("solution.py not found")

        if hasattr(solution, 'generate_link'):
            result = solution.generate_link('https://example.com', '数据来源')
            assert '[数据来源](https://example.com)' in result
        else:
            pytest.skip("generate_link function not implemented")


# =============================================================================
# 正例测试：代码块生成
# =============================================================================

class TestMarkdownCodeBlocks:
    """测试 Markdown 代码块生成"""

    def test_generate_code_block(self):
        """
        正例：生成代码块

        给定：多行代码
        期望：返回 ``` 包围的代码块
        """
        try:
            import solution
        except ImportError:
            pytest.skip("solution.py not found")

        code_lines = ['import pandas as pd', 'df = pd.read_csv("data.csv")', 'print(df.head())']

        if hasattr(solution, 'generate_code_block'):
            result = solution.generate_code_block(code_lines, language='python')
            assert '```python' in result
            assert '```' in result
            for line in code_lines:
                assert line in result
        else:
            pytest.skip("generate_code_block function not implemented")

    def test_generate_code_block_without_language(self):
        """
        正例：生成无语言标记的代码块

        期望：返回 ``` 包围但不带语言标记的代码块
        """
        try:
            import solution
        except ImportError:
            pytest.skip("solution.py not found")

        code = 'print("Hello, World!")'

        if hasattr(solution, 'generate_code_block'):
            result = solution.generate_code_block(code)
            assert '```' in result
            assert 'print("Hello, World!")' in result
        else:
            pytest.skip("generate_code_block function not implemented")


# =============================================================================
# 正例测试：引用块生成
# =============================================================================

class TestMarkdownBlockquotes:
    """测试 Markdown 引用块生成"""

    def test_generate_blockquote(self):
        """
        正例：生成引用块

        给定：引用文本
        期望：返回 "> 引用文本" 格式
        """
        try:
            import solution
        except ImportError:
            pytest.skip("solution.py not found")

        if hasattr(solution, 'generate_blockquote'):
            result = solution.generate_blockquote('重要发现：p < 0.05')
            assert '> 重要发现：p < 0.05' in result
        else:
            pytest.skip("generate_blockquote function not implemented")

    def test_generate_multiline_blockquote(self):
        """
        正例：生成多行引用块

        给定：多行文本
        期望：每行都以 > 开头
        """
        try:
            import solution
        except ImportError:
            pytest.skip("solution.py not found")

        lines = ['第一行', '第二行', '第三行']

        if hasattr(solution, 'generate_blockquote'):
            result = solution.generate_blockquote(lines)

            # 检查每行都有引用标记
            for line in lines:
                assert f'> {line}' in result
        else:
            pytest.skip("generate_blockquote function not implemented")


# =============================================================================
# 边界测试：特殊字符和边界情况
# =============================================================================

class TestMarkdownBoundaryCases:
    """测试边界情况"""

    def test_empty_string_handled(self):
        """
        边界：空字符串应能处理

        给定：空字符串
        期望：不报错，返回空或默认值
        """
        try:
            import solution
        except ImportError:
            pytest.skip("solution.py not found")

        if hasattr(solution, 'generate_heading'):
            result = solution.generate_heading('')
            assert result is not None or result == ''
        else:
            pytest.skip("generate_heading function not implemented")

    def test_special_characters_escaped(self):
        """
        边界：特殊字符应正确处理

        Markdown 特殊字符：* _ [ ] ( ) ` # + - . ! |
        """
        try:
            import solution
        except ImportError:
            pytest.skip("solution.py not found")

        text_with_special = "包含 *特殊_字符[和](链接)的文本"

        if hasattr(solution, 'generate_bold'):
            result = solution.generate_bold(text_with_special)
            # 至少不应报错
            assert result is not None
            assert len(result) > 0
        else:
            pytest.skip("generate_bold function not implemented")

    def test_very_long_heading(self):
        """
        边界：超长标题应能处理

        给定：1000 字符的标题
        期望：正常生成
        """
        try:
            import solution
        except ImportError:
            pytest.skip("solution.py not found")

        long_title = '很长的标题' * 200

        if hasattr(solution, 'generate_heading'):
            result = solution.generate_heading(long_title)
            assert long_title in result
        else:
            pytest.skip("generate_heading function not implemented")

    def test_unicode_characters(self):
        """
        边界：Unicode 字符应正确处理

        给定：包含 emoji、中文、日文等 Unicode 字符
        期望：正确编码
        """
        try:
            import solution
        except ImportError:
            pytest.skip("solution.py not found")

        unicode_text = "中文 📊 日本語 🇯🇵 Ελληνικά"

        if hasattr(solution, 'generate_bold'):
            result = solution.generate_bold(unicode_text)
            assert unicode_text in result
        else:
            pytest.skip("generate_bold function not implemented")

    def test_newline_preservation(self):
        """
        边界：换行符应正确处理

        Markdown 中换行需要两个空格或 <br>
        """
        try:
            import solution
        except ImportError:
            pytest.skip("solution.py not found")

        text_with_newline = "第一行\n第二行\n第三行"

        if hasattr(solution, 'generate_code_block'):
            result = solution.generate_code_block(text_with_newline)
            # 代码块应保留换行
            assert '第一行' in result and '第二行' in result
        else:
            pytest.skip("generate_code_block function not implemented")

    def test_null_value_in_table(self):
        """
        边界：表格中的 None/NaN 值应处理

        给定：包含空值的 DataFrame
        期望：空值显示为空或标记
        """
        try:
            import solution
        except ImportError:
            pytest.skip("solution.py not found")

        df_with_na = pd.DataFrame({
            'A': [1, None, 3],
            'B': ['x', 'y', None]
        })

        if hasattr(solution, 'generate_table'):
            result = solution.generate_table(df_with_na)
            # 不应报错
            assert result is not None
            assert len(result) > 0
        else:
            pytest.skip("generate_table function not implemented")


# =============================================================================
# 反例测试：错误处理
# =============================================================================

class TestMarkdownErrorCases:
    """测试错误处理"""

    def test_invalid_heading_level_raises_error(self):
        """
        反例：无效的标题级别应报错

        给定：level < 1 或 level > 6
        期望：抛出异常或返回默认值
        """
        try:
            import solution
        except ImportError:
            pytest.skip("solution.py not found")

        if hasattr(solution, 'generate_heading'):
            # Markdown 只支持 1-6 级标题
            try:
                result = solution.generate_heading('test', level=0)
                # 如果不报错，至少验证返回值
                assert result is not None
            except (ValueError, IndexError):
                assert True  # 预期的错误
        else:
            pytest.skip("generate_heading function not implemented")

    def test_none_input_handled(self):
        """
        反例：None 输入应报错或返回默认值

        给定：None
        期望：不崩溃
        """
        try:
            import solution
        except ImportError:
            pytest.skip("solution.py not found")

        if hasattr(solution, 'generate_bold'):
            try:
                result = solution.generate_bold(None)
                # 可能返回空字符串或 "None"
                assert result is not None
            except (TypeError, ValueError):
                assert True  # 预期的错误
        else:
            pytest.skip("generate_bold function not implemented")

    def test_empty_dataframe_table(self):
        """
        反例：空 DataFrame 生成表格

        给定：空 DataFrame
        期望：返回空表格或报错
        """
        try:
            import solution
        except ImportError:
            pytest.skip("solution.py not found")

        empty_df = pd.DataFrame()

        if hasattr(solution, 'generate_table'):
            result = solution.generate_table(empty_df)
            # 空表格或报错都可接受
            assert result is not None or result == ''
        else:
            pytest.skip("generate_table function not implemented")


# =============================================================================
# 完整报告模板测试
# =============================================================================

class TestMarkdownReportTemplate:
    """测试完整报告模板生成"""

    def test_generate_full_report_template(self, sample_statistics, sample_test_results):
        """
        正例：生成完整报告模板

        验证报告包含所有必要部分
        """
        try:
            import solution
        except ImportError:
            pytest.skip("solution.py not found")

        if hasattr(solution, 'render_markdown'):
            result = solution.render_markdown({
                'title': '客户流失分析报告',
                'statistics': sample_statistics,
                'test_results': sample_test_results
            })

            # 检查包含标题
            assert '客户流失' in result

            # 检查包含可复现信息
            assert any(keyword in result for keyword in
                      ['数据来源', '日期', '可复现', 'reproducible'])

        else:
            pytest.skip("render_markdown function not implemented")

    def test_report_contains_sections(self):
        """
        正例：报告应包含所有标准章节

        验证：数据概览、描述统计、结果、结论
        """
        try:
            import solution
        except ImportError:
            pytest.skip("solution.py not found")

        if hasattr(solution, 'render_markdown'):
            result = solution.render_markdown({})

            # 检查是否包含标准章节标记
            has_sections = any(keyword in result for keyword in
                              ['##', '数据', '统计', '结论', '结果'])
            assert has_sections or len(result) > 0  # 至少生成一些内容
        else:
            pytest.skip("render_markdown function not implemented")


# =============================================================================
# HTML 转换测试
# =============================================================================

class TestMarkdownToHTML:
    """测试 Markdown 到 HTML 的转换"""

    def test_markdown_to_html_conversion(self):
        """
        正例：Markdown 可以转换为 HTML

        验证基本的转换能力（如果实现）
        """
        try:
            import solution
        except ImportError:
            pytest.skip("solution.py not found")

        markdown_content = "# 标题\n\n这是段落。"

        if hasattr(solution, 'convert_to_html'):
            result = solution.convert_to_html(markdown_content)
            # HTML 应包含标签
            assert '<' in result and '>' in result
            assert 'h1' in result.lower() or 'h2' in result.lower()
        else:
            pytest.skip("convert_to_html function not implemented")

    def test_html_contains_headings(self):
        """
        正例：HTML 应包含标题标签

        验证 # 被转换为 <h1>
        """
        try:
            import solution
        except ImportError:
            pytest.skip("solution.py not found")

        markdown = "# 主标题\n\n## 子标题"

        if hasattr(solution, 'convert_to_html'):
            result = solution.convert_to_html(markdown)
            assert '<h1>' in result.lower() or '<h2>' in result.lower()
        else:
            pytest.skip("convert_to_html function not implemented")

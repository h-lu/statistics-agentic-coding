#!/usr/bin/env python3
"""
build_site.py - Docusaurus 站点内容生成脚本

从 chapters/ 目录自动生成 Docusaurus 站点内容，包括：
- 解析 TOC.md 生成 sidebars.ts
- 为每个 week_XX 生成完整的页面结构
- 处理 YAML 文件（ANCHORS.yml, TERMS.yml）的渲染
- 处理代码文件的聚合和展示

用法:
    python scripts/build_site.py [options]

选项:
    --site-dir SITE_DIR    站点目录 (默认: site)
    --chapters-dir DIR     chapters 目录 (默认: chapters)
    --shared-dir DIR       shared 目录 (默认: shared)
    --verbose              详细输出
"""

import argparse
import logging
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


# =============================================================================
# 数据类定义
# =============================================================================

@dataclass
class WeekInfo:
    """每周信息"""
    number: int
    title: str
    phase: str
    phase_label: str
    has_chapter: bool = False
    has_assignment: bool = False
    has_rubric: bool = False
    has_anchors: bool = False
    has_terms: bool = False
    has_examples: bool = False
    has_starter_code: bool = False
    has_tests: bool = False


@dataclass
class PhaseInfo:
    """阶段信息"""
    name: str
    label: str
    weeks: list[WeekInfo] = field(default_factory=list)


@dataclass
class AnchorEntry:
    """锚点条目"""
    id: str
    claim: str
    evidence: str
    verification: str
    status: str = "pending"


@dataclass
class TermEntry:
    """术语条目"""
    chinese: str
    english: str
    definition: str


# =============================================================================
# 日志配置
# =============================================================================

def setup_logging(verbose: bool = False) -> logging.Logger:
    """配置日志输出"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    return logging.getLogger(__name__)


# =============================================================================
# TOC 解析器
# =============================================================================

class TOCParser:
    """TOC.md 解析器"""
    
    # 阶段映射
    PHASE_MAP = {
        '阶段一': ('phase-1', '思维奠基'),
        '阶段二': ('phase-2', '系统化工程'),
        '阶段三': ('phase-3', 'AI 时代的工程'),
        '阶段四': ('phase-4', '综合实战'),
    }
    
    def __init__(self, chapters_dir: Path, logger: logging.Logger):
        self.chapters_dir = chapters_dir
        self.logger = logger
        self.phases: list[PhaseInfo] = []
        
    def parse(self) -> list[PhaseInfo]:
        """解析 TOC.md 文件"""
        toc_path = self.chapters_dir / 'TOC.md'
        
        if not toc_path.exists():
            self.logger.warning(f"TOC.md 不存在: {toc_path}")
            return self._create_default_phases()
        
        self.logger.info(f"解析 TOC.md: {toc_path}")
        content = toc_path.read_text(encoding='utf-8')
        
        current_phase: Optional[PhaseInfo] = None
        
        for line in content.split('\n'):
            line = line.strip()
            
            # 解析阶段标题
            if line.startswith('## '):
                phase_name = line[3:].strip()
                current_phase = self._create_phase(phase_name)
                if current_phase:
                    self.phases.append(current_phase)
                    self.logger.debug(f"发现阶段: {current_phase.label}")
            
            # 解析周条目 - 支持列表格式 (- **Week**) 和表格格式 (| 01 | ...)
            elif current_phase and (line.startswith('- **Week') or (line.startswith('|') and re.match(r'^\|\s*\d+\s*\|', line))):
                week_info = self._parse_week_line(line, current_phase)
                if week_info:
                    current_phase.weeks.append(week_info)
                    self.logger.debug(f"  发现周: Week {week_info.number:02d} - {week_info.title}")
        
        return self.phases
    
    def _create_phase(self, phase_name: str) -> Optional[PhaseInfo]:
        """创建阶段信息"""
        for phase_key, (phase_id, phase_label) in self.PHASE_MAP.items():
            if phase_key in phase_name:
                return PhaseInfo(
                    name=phase_name,
                    label=f"{phase_key}：{phase_label}"
                )
        return None
    
    def _parse_week_line(self, line: str, phase: PhaseInfo) -> Optional[WeekInfo]:
        """解析周条目行 - 支持列表格式和表格格式"""
        # 尝试匹配列表格式: - **Week 01**: 标题
        list_pattern = r'- \*\*Week\s*(\d+)\*\*:\s*(.+)'
        # 尝试匹配表格格式: | 01 | [标题](week_01/CHAPTER.md) | ...
        table_pattern = r'^\|\s*(\d+)\s*\|\s*\[([^\]]+)\]\([^)]+\)'
        
        match = re.match(list_pattern, line)
        if match:
            week_num = int(match.group(1))
            week_title = match.group(2).strip()
        else:
            match = re.match(table_pattern, line)
            if not match:
                return None
            week_num = int(match.group(1))
            week_title = match.group(2).strip()
        
        # 检查周目录中的文件
        week_dir = self.chapters_dir / f'week_{week_num:02d}'
        
        week_info = WeekInfo(
            number=week_num,
            title=week_title,
            phase=phase.name,
            phase_label=phase.label,
            has_chapter=(week_dir / 'CHAPTER.md').exists(),
            has_assignment=(week_dir / 'ASSIGNMENT.md').exists(),
            has_rubric=(week_dir / 'RUBRIC.md').exists(),
            has_anchors=(week_dir / 'ANCHORS.yml').exists(),
            has_terms=(week_dir / 'TERMS.yml').exists(),
            has_examples=(week_dir / 'examples').exists() and any((week_dir / 'examples').iterdir()),
            has_starter_code=(week_dir / 'starter_code').exists() and any((week_dir / 'starter_code').iterdir()),
            has_tests=(week_dir / 'tests').exists() and any((week_dir / 'tests').iterdir()),
        )
        
        return week_info
    
    def _create_default_phases(self) -> list[PhaseInfo]:
        """创建默认阶段（当 TOC.md 不存在时）"""
        self.logger.warning("使用默认阶段配置")
        phases = []
        
        for phase_key, (phase_id, phase_label) in self.PHASE_MAP.items():
            phase = PhaseInfo(
                name=f"{phase_key}：{phase_label}",
                label=f"{phase_key}：{phase_label}"
            )
            
            # 为每个阶段添加周
            week_ranges = {
                '阶段一': range(1, 5),
                '阶段二': range(5, 9),
                '阶段三': range(9, 13),
                '阶段四': range(13, 17),
            }
            
            for week_num in week_ranges.get(phase_key, range(1, 5)):
                week_dir = self.chapters_dir / f'week_{week_num:02d}'
                if week_dir.exists():
                    week_info = WeekInfo(
                        number=week_num,
                        title=f"Week {week_num:02d}",
                        phase=phase.name,
                        phase_label=phase.label,
                        has_chapter=(week_dir / 'CHAPTER.md').exists(),
                        has_assignment=(week_dir / 'ASSIGNMENT.md').exists(),
                        has_rubric=(week_dir / 'RUBRIC.md').exists(),
                        has_anchors=(week_dir / 'ANCHORS.yml').exists(),
                        has_terms=(week_dir / 'TERMS.yml').exists(),
                        has_examples=(week_dir / 'examples').exists() and any((week_dir / 'examples').iterdir()),
                        has_starter_code=(week_dir / 'starter_code').exists() and any((week_dir / 'starter_code').iterdir()),
                        has_tests=(week_dir / 'tests').exists() and any((week_dir / 'tests').iterdir()),
                    )
                    phase.weeks.append(week_info)
            
            phases.append(phase)
        
        return phases


# =============================================================================
# YAML 解析器
# =============================================================================

class YAMLParser:
    """YAML 文件解析器"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def parse_anchors(self, file_path: Path) -> list[AnchorEntry]:
        """解析 ANCHORS.yml 文件"""
        if not file_path.exists():
            return []
        
        try:
            content = yaml.safe_load(file_path.read_text(encoding='utf-8'))
            anchors = []
            
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        anchors.append(AnchorEntry(
                            id=str(item.get('id', '')),
                            claim=str(item.get('claim', '')),
                            evidence=str(item.get('evidence', '')),
                            verification=str(item.get('verification', '')),
                            status=str(item.get('status', 'pending'))
                        ))
            
            return anchors
        except Exception as e:
            self.logger.warning(f"解析 ANCHORS.yml 失败: {file_path} - {e}")
            return []
    
    def parse_terms(self, file_path: Path) -> list[TermEntry]:
        """解析 TERMS.yml 或 glossary.yml 文件"""
        if not file_path.exists():
            return []
        
        try:
            content = yaml.safe_load(file_path.read_text(encoding='utf-8'))
            terms = []
            
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        # 支持多种键名格式
                        chinese = item.get('chinese') or item.get('cn') or item.get('term_zh') or ''
                        english = item.get('english') or item.get('en') or item.get('term_en') or ''
                        definition = item.get('definition') or item.get('desc') or item.get('definition_zh') or ''
                        terms.append(TermEntry(
                            chinese=str(chinese),
                            english=str(english),
                            definition=str(definition)
                        ))
            elif isinstance(content, dict):
                # 处理 glossary.yml 格式
                for key, value in content.items():
                    if isinstance(value, dict):
                        chinese = value.get('chinese') or value.get('cn') or value.get('term_zh') or key
                        english = value.get('english') or value.get('en') or value.get('term_en') or ''
                        definition = value.get('definition') or value.get('desc') or value.get('definition_zh') or ''
                        terms.append(TermEntry(
                            chinese=str(chinese),
                            english=str(english),
                            definition=str(definition)
                        ))
            
            return terms
        except Exception as e:
            self.logger.warning(f"解析 TERMS.yml 失败: {file_path} - {e}")
            return []


# =============================================================================
# 代码文件收集器
# =============================================================================

class CodeCollector:
    """代码文件收集器"""
    
    CODE_EXTENSIONS = {'.java', '.py', '.js', '.ts', '.jsx', '.tsx', '.md', '.txt', '.yml', '.yaml', '.json', '.xml'}
    # 排除的文件扩展名（编译产物、二进制文件等）
    EXCLUDED_EXTENSIONS = {'.class', '.jar', '.war', '.nar', '.ear', '.zip', '.tar.gz', '.rar'}
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def collect_files(self, directory: Path) -> dict[str, list[dict]]:
        """收集目录中的代码文件"""
        result = {
            'examples': [],
            'starter_code': [],
            'tests': []
        }
        
        if not directory.exists():
            return result
        
        for subdir_name in ['examples', 'starter_code']:
            subdir = directory / subdir_name
            if subdir.exists():
                result[subdir_name] = self._collect_directory_files(subdir)

        # Tests 目录：优先检查顶层 tests/，否则检查 starter_code/src/test/
        tests_dir = directory / 'tests'
        if not tests_dir.exists():
            tests_dir = directory / 'starter_code' / 'src' / 'test'
        if tests_dir.exists():
            result['tests'] = self._collect_directory_files(tests_dir)

        return result
    
    def _collect_directory_files(self, directory: Path) -> list[dict]:
        """递归收集目录中的文件"""
        files = []

        try:
            for item in sorted(directory.rglob('*')):
                if item.is_file():
                    # 跳过隐藏文件
                    if item.name.startswith('.'):
                        continue
                    # 跳过排除的扩展名（编译产物、二进制文件等）
                    if item.suffix.lower() in self.EXCLUDED_EXTENSIONS:
                        continue

                    rel_path = item.relative_to(directory)
                    content = self._read_file_content(item)

                    files.append({
                        'path': str(rel_path),
                        'name': item.name,
                        'content': content,
                        'language': self._detect_language(item)
                    })
        except Exception as e:
            self.logger.warning(f"收集文件失败: {directory} - {e}")

        return files
    
    def _read_file_content(self, file_path: Path) -> str:
        """读取文件内容"""
        try:
            # 尝试以文本方式读取
            return file_path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            # 二进制文件
            return "[Binary file - content not displayed]"
        except Exception as e:
            return f"[Error reading file: {e}]"
    
    def _detect_language(self, file_path: Path) -> str:
        """检测文件语言"""
        ext = file_path.suffix.lower()
        language_map = {
            '.java': 'java',
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'jsx',
            '.tsx': 'tsx',
            '.md': 'markdown',
            '.yml': 'yaml',
            '.yaml': 'yaml',
            '.json': 'json',
            '.xml': 'xml',
            '.html': 'html',
            '.css': 'css',
            '.sh': 'bash',
            '.sql': 'sql',
        }
        return language_map.get(ext, 'text')


# =============================================================================
# 内容生成器
# =============================================================================

class ContentGenerator:
    """MDX 内容生成器"""
    
    def __init__(self, chapters_dir: Path, shared_dir: Path, logger: logging.Logger):
        self.chapters_dir = chapters_dir
        self.shared_dir = shared_dir
        self.logger = logger
        self.yaml_parser = YAMLParser(logger)
        self.code_collector = CodeCollector(logger)
    
    # -------------------------------------------------------------------------
    # 每周页面生成
    # -------------------------------------------------------------------------
    
    def generate_week_index(self, week: WeekInfo) -> str:
        """生成周主页 index.mdx"""
        lines = [
            '---',
            f'title: "Week {week.number:02d}: {week.title}"',
            f'sidebar_position: {week.number}',
            f'slug: /weeks/{week.number:02d}',
            f'tags: ["week-{week.number:02d}", "{self._get_phase_tag(week.phase_label)}"]',
            '---',
            '',
            f'# Week {week.number:02d}: {week.title}',
            '',
            f'**阶段**：{week.phase_label}',
            '',
            '## 本周内容',
            '',
        ]
        
        # 添加链接列表
        links = [
            ('讲义', './chapter.mdx', week.has_chapter),
            ('作业', './assignment.mdx', week.has_assignment),
            ('评分标准', './rubric.mdx', week.has_rubric),
            ('代码', './code.mdx', week.has_examples or week.has_starter_code or week.has_tests),
            ('锚点', './anchors.mdx', week.has_anchors),
            ('术语', './terms.mdx', week.has_terms),
        ]

        for label, link, exists in links:
            if exists:
                lines.append(f'- [{label}]({link})')
            else:
                lines.append(f'- ~~{label}~~ *(未提供)*')
        
        lines.append('')
        return '\n'.join(lines)
    
    def generate_chapter(self, week: WeekInfo) -> str:
        """生成讲义页面 chapter.mdx"""
        chapter_path = self.chapters_dir / f'week_{week.number:02d}' / 'CHAPTER.md'
        
        lines = [
            '---',
            f'title: "Week {week.number:02d}: {week.title} - 讲义"',
            f'sidebar_position: 1',
            f'tags: ["week-{week.number:02d}", "chapter"]',
            '---',
            '',
        ]
        
        if chapter_path.exists():
            content = chapter_path.read_text(encoding='utf-8')
            # 移除原有的 frontmatter
            content = self._escape_mdx_content(self._remove_frontmatter(content))
            lines.append(content)
        else:
            lines.extend([
                f'# Week {week.number:02d}: {week.title}',
                '',
                '> ⚠️ 本周讲义内容正在准备中...',
                '',
            ])
        
        return '\n'.join(lines)
    
    def generate_assignment(self, week: WeekInfo) -> str:
        """生成作业页面 assignment.mdx"""
        assignment_path = self.chapters_dir / f'week_{week.number:02d}' / 'ASSIGNMENT.md'
        
        lines = [
            '---',
            f'title: "Week {week.number:02d}: {week.title} - 作业"',
            f'sidebar_position: 2',
            f'tags: ["week-{week.number:02d}", "assignment"]',
            '---',
            '',
        ]
        
        if assignment_path.exists():
            content = assignment_path.read_text(encoding='utf-8')
            content = self._escape_mdx_content(self._remove_frontmatter(content))
            lines.append(content)
        else:
            lines.extend([
                f'# Week {week.number:02d}: {week.title} - 作业',
                '',
                '> ⚠️ 本周作业内容正在准备中...',
                '',
                '请稍后查看更新。',
                '',
            ])
        
        return '\n'.join(lines)
    
    def generate_rubric(self, week: WeekInfo) -> str:
        """生成评分标准页面 rubric.mdx"""
        rubric_path = self.chapters_dir / f'week_{week.number:02d}' / 'RUBRIC.md'
        
        lines = [
            '---',
            f'title: "Week {week.number:02d}: {week.title} - 评分标准"',
            f'sidebar_position: 3',
            f'tags: ["week-{week.number:02d}", "rubric"]',
            '---',
            '',
        ]
        
        if rubric_path.exists():
            content = rubric_path.read_text(encoding='utf-8')
            content = self._escape_mdx_content(self._remove_frontmatter(content))
            lines.append(content)
        else:
            lines.extend([
                f'# Week {week.number:02d}: {week.title} - 评分标准',
                '',
                '> ⚠️ 本周评分标准正在准备中...',
                '',
                '请稍后查看更新。',
                '',
            ])
        
        return '\n'.join(lines)
    
    def generate_code(self, week: WeekInfo) -> str:
        """生成代码页面 code.mdx"""
        week_dir = self.chapters_dir / f'week_{week.number:02d}'
        code_files = self.code_collector.collect_files(week_dir)
        
        lines = [
            '---',
            f'title: "Week {week.number:02d}: {week.title} - 代码"',
            f'sidebar_position: 4',
            f'tags: ["week-{week.number:02d}", "code"]',
            '---',
            '',
            f'# Week {week.number:02d}: {week.title} - 代码',
            '',
            'import Tabs from "@theme/Tabs";',
            'import TabItem from "@theme/TabItem";',
            '',
        ]
        
        has_any_code = any(code_files.values())
        
        if not has_any_code:
            lines.extend([
                '> ⚠️ 本周暂无代码示例',
                '',
            ])
            return '\n'.join(lines)
        
        lines.extend([
            '<Tabs>',
            '',
        ])
        
        # Examples Tab
        lines.extend(self._generate_code_tab('Examples', code_files['examples'], 'examples'))
        
        # Starter Code Tab
        lines.extend(self._generate_code_tab('Starter Code', code_files['starter_code'], 'starter_code'))
        
        # Tests Tab
        lines.extend(self._generate_code_tab('Tests', code_files['tests'], 'tests'))
        
        lines.extend([
            '</Tabs>',
            '',
        ])
        
        return '\n'.join(lines)
    
    def _generate_code_tab(self, label: str, files: list[dict], tab_value: str) -> list[str]:
        """生成代码 Tab 内容"""
        lines = [
            f'<TabItem value="{tab_value}" label="{label}">',
            '',
        ]
        
        if not files:
            lines.extend([
                f'_{label} 暂无内容_',
                '',
            ])
        else:
            # 文件树
            lines.extend([
                '## 文件列表',
                '',
            ])
            
            for file_info in files:
                lines.append(f'- `{file_info["path"]}`')
            
            lines.append('')
            
            # 代码内容
            lines.append('## 代码内容')
            lines.append('')
            
            for file_info in files:
                # 对代码块内容中的 MDX 敏感字符进行转义
                content = self._escape_mdx_content(file_info['content'])
                
                # 如果内容包含 ```，使用更长的围栏避免嵌套问题
                if '```' in content:
                    fence = '````'
                else:
                    fence = '```'
                
                lines.extend([
                    f'### {file_info["path"]}',
                    '',
                    f'{fence}{file_info["language"]}',
                    content,
                    fence,
                    '',
                ])
        
        lines.extend([
            '</TabItem>',
            '',
        ])
        
        return lines
    
    def generate_anchors(self, week: WeekInfo) -> str:
        """生成锚点页面 anchors.mdx"""
        anchors_path = self.chapters_dir / f'week_{week.number:02d}' / 'ANCHORS.yml'
        anchors = self.yaml_parser.parse_anchors(anchors_path)
        
        lines = [
            '---',
            f'title: "Week {week.number:02d}: {week.title} - 锚点"',
            f'sidebar_position: 5',
            f'tags: ["week-{week.number:02d}", "anchors"]',
            '---',
            '',
            f'# Week {week.number:02d}: {week.title} - 锚点',
            '',
        ]
        
        if not anchors:
            lines.extend([
                '> ⚠️ 本周暂无锚点记录',
                '',
            ])
            return '\n'.join(lines)
        
        lines.extend([
            '| ID | 主张 | 证据 | 验证方式 | 状态 |',
            '|------|------|------|----------|------|',
        ])
        
        for anchor in anchors:
            # 截断长文本并转义 MDX
            claim = self._escape_inline_text(self._truncate_text(anchor.claim, 50))
            evidence = self._escape_inline_text(self._truncate_text(anchor.evidence, 50))
            verification = self._escape_inline_text(self._truncate_text(anchor.verification, 30))
            
            # 状态徽章
            status_badge = self._get_status_badge(anchor.status)
            
            lines.append(f'| `{anchor.id}` | {claim} | {evidence} | {verification} | {status_badge} |')
        
        lines.append('')
        
        # 添加详细列表
        lines.extend([
            '## 详细内容',
            '',
        ])
        
        for anchor in anchors:
            claim = self._escape_inline_text(anchor.claim)
            evidence = self._escape_inline_text(anchor.evidence)
            verification = self._escape_inline_text(anchor.verification)
            
            lines.extend([
                f'### {anchor.id}',
                '',
                f'**主张**: {claim}',
                '',
                f'**证据**: {evidence}',
                '',
                f'**验证方式**: {verification}',
                '',
                f'**状态**: {anchor.status}',
                '',
            ])
        
        return '\n'.join(lines)
    
    def generate_terms(self, week: WeekInfo) -> str:
        """生成术语页面 terms.mdx"""
        terms_path = self.chapters_dir / f'week_{week.number:02d}' / 'TERMS.yml'
        terms = self.yaml_parser.parse_terms(terms_path)
        
        lines = [
            '---',
            f'title: "Week {week.number:02d}: {week.title} - 术语"',
            f'sidebar_position: 6',
            f'tags: ["week-{week.number:02d}", "terms"]',
            '---',
            '',
            f'# Week {week.number:02d}: {week.title} - 术语',
            '',
        ]
        
        if not terms:
            lines.extend([
                '> ⚠️ 本周暂无术语定义',
                '',
            ])
            return '\n'.join(lines)
        
        lines.extend([
            '| 中文 | 英文 | 定义 |',
            '|------|------|------|',
        ])
        
        for term in terms:
            definition = self._escape_inline_text(self._truncate_text(term.definition, 80))
            lines.append(f'| {term.chinese} | {term.english} | {definition} |')
        
        lines.append('')

        return '\n'.join(lines)
    
    # -------------------------------------------------------------------------
    # 全局页面生成
    # -------------------------------------------------------------------------
    
    def generate_index(self, repo_root: Path) -> str:
        """生成首页 index.mdx"""
        readme_path = repo_root / 'README.md'
        
        lines = [
            '---',
            'title: "首页"',
            'sidebar_position: 0',
            'slug: /',
            '---',
            '',
        ]
        
        if readme_path.exists():
            content = readme_path.read_text(encoding='utf-8')
            content = self._escape_mdx_content(self._remove_frontmatter(content))
            lines.append(content)
        else:
            lines.extend([
                '# Java 软件工程',
                '',
                '欢迎来到 Java 软件工程课程文档站点！',
                '',
                '## 课程结构',
                '',
                '- **阶段一：思维奠基**（Week 01-04）',
                '- **阶段二：系统化工程**（Week 05-08）',
                '- **阶段三：AI 时代的工程**（Week 09-12）',
                '- **阶段四：综合实战**（Week 13-16）',
                '',
            ])
        
        return '\n'.join(lines)
    
    def generate_syllabus(self) -> str:
        """生成教学大纲页面 syllabus.mdx"""
        syllabus_path = self.chapters_dir / 'SYLLABUS.md'
        
        lines = [
            '---',
            'title: "教学大纲"',
            'sidebar_position: 100',
            '---',
            '',
        ]
        
        if syllabus_path.exists():
            content = syllabus_path.read_text(encoding='utf-8')
            content = self._escape_mdx_content(self._remove_frontmatter(content))
            lines.append(content)
        else:
            lines.extend([
                '# 教学大纲',
                '',
                '> ⚠️ 教学大纲正在准备中...',
                '',
            ])
        
        return '\n'.join(lines)
    
    def generate_campusflow(self) -> str:
        """生成 CampusFlow 项目页面 campusflow.mdx"""
        project_path = self.shared_dir / 'book_project.md'
        
        lines = [
            '---',
            'title: "CampusFlow 项目"',
            'sidebar_position: 101',
            '---',
            '',
        ]
        
        if project_path.exists():
            content = project_path.read_text(encoding='utf-8')
            content = self._escape_mdx_content(self._remove_frontmatter(content))
            lines.append(content)
        else:
            lines.extend([
                '# CampusFlow 项目',
                '',
                '> ⚠️ 项目文档正在准备中...',
                '',
            ])
        
        return '\n'.join(lines)
    
    def generate_glossary(self) -> str:
        """生成术语表页面 glossary.mdx"""
        glossary_path = self.shared_dir / 'glossary.yml'
        terms = self.yaml_parser.parse_terms(glossary_path)
        
        lines = [
            '---',
            'title: "术语表"',
            'sidebar_position: 102',
            '---',
            '',
            '# 术语表',
            '',
        ]
        
        if not terms:
            lines.extend([
                '> ⚠️ 术语表正在准备中...',
                '',
            ])
            return '\n'.join(lines)
        
        # 按首字母分组
        from collections import defaultdict
        grouped = defaultdict(list)
        
        for term in terms:
            first_char = term.chinese[0].upper() if term.chinese else '#'
            grouped[first_char].append(term)
        
        # 字母导航（纯文本，不使用锚点链接避免 MDX 问题）
        lines.append('## 快速导航')
        lines.append('')

        sorted_chars = sorted(grouped.keys())
        nav_items = []
        for char in sorted_chars:
            nav_items.append(f'**{char}**')
        lines.append(' · '.join(nav_items))
        lines.append('')
        
        # 各组内容
        for char in sorted(grouped.keys()):
            lines.extend([
                f'## {char}',
                '',
                '| 中文 | 英文 | 定义 |',
                '|------|------|------|',
            ])
            
            for term in sorted(grouped[char], key=lambda x: x.chinese):
                definition = self._truncate_text(term.definition, 100)
                # Wrap content with MDX special chars (<, >, {, }) in backticks to avoid JSX parsing errors
                import re
                # Pattern to match content like ArrayList<String>, /users/{id}, <>
                definition = re.sub(r'([\w/]+\{[^}]+\})', r'`\1`', definition)  # /users/{id}
                definition = re.sub(r'([\w]+<[^>]+>)', r'`\1`', definition)      # ArrayList<String>
                definition = re.sub(r'(<>)', r'`\1`', definition)                # <>
                lines.append(f'| {term.chinese} | {term.english} | {definition} |')
            
            lines.append('')
        
        return '\n'.join(lines)
    
    def generate_style_guide(self) -> str:
        """生成风格指南页面 style-guide.mdx"""
        style_path = self.shared_dir / 'style_guide.md'
        
        lines = [
            '---',
            'title: "风格指南"',
            'sidebar_position: 103',
            '---',
            '',
        ]
        
        if style_path.exists():
            content = style_path.read_text(encoding='utf-8')
            content = self._escape_mdx_content(self._remove_frontmatter(content))
            lines.append(content)
        else:
            lines.extend([
                '# 风格指南',
                '',
                '> ⚠️ 风格指南正在准备中...',
                '',
            ])
        
        return '\n'.join(lines)
    
    # -------------------------------------------------------------------------
    # 辅助方法
    # -------------------------------------------------------------------------
    
    def _remove_frontmatter(self, content: str) -> str:
        """移除 Markdown 文件的 frontmatter"""
        if content.startswith('---'):
            parts = content.split('---', 2)
            if len(parts) >= 3:
                return parts[2].strip()
        return content
    
    def _escape_mdx_content(self, content: str) -> str:
        """转义 MDX 内容中可能被误认为 JSX 标签的语法"""
        import re
        import uuid

        # 使用唯一的占位符前缀避免冲突
        placeholder_prefix = f"__MDX_PROTECTED_{uuid.uuid4().hex[:8]}__"
        placeholders = {}
        placeholder_id = 0

        def make_placeholder():
            nonlocal placeholder_id
            ph = f"{placeholder_prefix}_{placeholder_id}_"
            placeholder_id += 1
            return ph

        # 步骤0: 首先保护代码块（```...```），避免代码块内的内容被转义
        # 同时对代码块内的 URL 进行特殊处理以避免 MDX 解析问题
        def protect_code_blocks(text):
            result = []
            i = 0
            while i < len(text):
                # 查找代码块开始
                if text[i:i+3] == '```':
                    # 找到代码块结束
                    end_idx = text.find('\n```', i + 3)
                    if end_idx == -1:
                        # 没有找到结束，保持原样
                        result.append(text[i:])
                        break
                    # 包含结束的 ```
                    end_idx = end_idx + 4
                    code_block = text[i:end_idx]
                    placeholder = make_placeholder()
                    # 存储原始代码块用于恢复
                    placeholders[placeholder] = code_block
                    result.append(placeholder)
                    i = end_idx
                else:
                    result.append(text[i])
                    i += 1
            return ''.join(result)

        content = protect_code_blocks(content)
        self.logger.debug(f"After code block protection: {len(placeholders)} placeholders")

        # 保护合法的 MDX 组件和 HTML 标签（这些不应该被转义）
        protected_tags = ['Tabs', 'TabItem', 'details', 'summary', 'br', 'code', 'pre', 'kbd', 'sub', 'sup']

        # 步骤0.5: 保护内联代码（`...`），避免内联代码内的内容被转义
        # 匹配单个反引号包围的内容（不是代码块）
        inline_code_pattern = r'(?<!`)`(?!`)([^`\n]+)(?<!`)`(?!`)'
        matches = list(re.finditer(inline_code_pattern, content))
        for match in reversed(matches):
            placeholder = make_placeholder()
            placeholders[placeholder] = match.group(0)
            content = content[:match.start()] + placeholder + content[match.end():]

        # 步骤1: 保护合法的 MDX 组件，用占位符替换
        # 使用列表存储替换信息，避免在迭代时修改字符串
        for tag in protected_tags:
            # 匹配 <Tag ...>...</Tag>（包括跨行的）
            pattern = rf'<{tag}\b[^>]*>.*?</{tag}>'
            matches = list(re.finditer(pattern, content, re.DOTALL))
            for match in reversed(matches):  # 从后往前替换，避免位置偏移
                placeholder = make_placeholder()
                placeholders[placeholder] = match.group(0)
                content = content[:match.start()] + placeholder + content[match.end():]
            
            # 匹配自闭合 <Tag ... /> 或 <Tag>
            pattern = rf'<{tag}\b[^>]*/>'
            matches = list(re.finditer(pattern, content))
            for match in reversed(matches):
                placeholder = make_placeholder()
                placeholders[placeholder] = match.group(0)
                content = content[:match.start()] + placeholder + content[match.end():]
            
            # 匹配 <Tag>（自闭合但没有斜杠，如 <br>, <hr>）
            pattern = rf'<{tag}\b[^<>]*>'
            matches = list(re.finditer(pattern, content))
            for match in reversed(matches):
                placeholder = make_placeholder()
                placeholders[placeholder] = match.group(0)
                content = content[:match.start()] + placeholder + content[match.end():]
        
        # 步骤2: 保护已知的合法 HTML 标签（这些不应该被转义）
        html_tags = {'div', 'span', 'p', 'a', 'img', 'table', 'tr', 'td', 'th', 'thead', 'tbody',
                     'ul', 'ol', 'li', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'strong', 'em',
                     'b', 'i', 'u', 's', 'del', 'ins', 'iframe', 'video', 'audio', 'source',
                     'input', 'form', 'button', 'label', 'select', 'option', 'hr', 'small'}
        
        for tag in html_tags:
            # 匹配 <tag ...>...</tag>
            pattern = rf'<{tag}\b[^>]*>.*?</{tag}>'
            matches = list(re.finditer(pattern, content, re.DOTALL))
            for match in reversed(matches):
                placeholder = make_placeholder()
                placeholders[placeholder] = match.group(0)
                content = content[:match.start()] + placeholder + content[match.end():]
            
            # 匹配 <tag ... />
            pattern = rf'<{tag}\b[^>]*/>'
            matches = list(re.finditer(pattern, content))
            for match in reversed(matches):
                placeholder = make_placeholder()
                placeholders[placeholder] = match.group(0)
                content = content[:match.start()] + placeholder + content[match.end():]
            
            # 匹配 <tag>（自闭合但没有斜杠）
            pattern = rf'<{tag}\b[^<>]*>'
            matches = list(re.finditer(pattern, content))
            for match in reversed(matches):
                placeholder = make_placeholder()
                placeholders[placeholder] = match.group(0)
                content = content[:match.start()] + placeholder + content[match.end():]
        
        # 步骤3: 转义看起来像 JSX/泛型的标签
        # 注意：此时所有合法标签已经被占位符保护
        
        # 3.1 转义空尖括号 <>（菱形操作符）
        content = re.sub(r'(?<!&)<>', '&lt;&gt;', content)
        
        # 3.2 转义泛型类型参数 <T, K>、<String, Integer> 等
        # 匹配 <字母,字母> 或 <字母, 字母> 等形式
        def escape_generic(match):
            inner = match.group(1)
            # 如果内部包含逗号，且看起来像类型参数，则转义
            return f'&lt;{inner}&gt;'
        
        # 匹配 <Type1, Type2> 或 <T, K extends Something> 等形式
        content = re.sub(r'<([A-Za-z][A-Za-z0-9_]*\s*,\s*[A-Za-z][^>]*)>', escape_generic, content)
        
        # 3.3 转义单个泛型参数 <String>、<Integer>、<6> 等
        def escape_single_tag(match):
            tag = match.group(1)
            # 任何看起来像标签的内容都转义
            return f'&lt;{tag}&gt;'
        
        # 匹配 <Word> 形式的标签（包括 <6> 这种数字形式）
        content = re.sub(r'<([A-Za-z0-9_]+)>', escape_single_tag, content)
        
        # 3.4 转义比较表达式中的 <数字（如 <60 分）
        # 匹配 <数字 后跟非字母数字字符的情况
        def escape_comparison(match):
            return f'&lt;{match.group(1)}'
        
        content = re.sub(r'<(\d+)(?=[^\d>])', escape_comparison, content)
        
        # 3.4b 转义比较运算符 <= 和 >=
        content = re.sub(r'<=', '&lt;=', content)
        content = re.sub(r'>=', '&gt;=', content)
        
        # 3.5 转义闭合标签 </Word>
        content = re.sub(r'</([A-Za-z0-9_]+)>', lambda m: f'&lt;/{m.group(1)}&gt;', content)
        
        # 3.6 转义 ${...} 格式的 shell 变量/占位符，避免被当作 JSX 表达式
        # 所有 ${...} 格式都转义（它们通常是 shell 变量，不是 JS 表达式）
        # 使用 HTML 实体转义 { 和 }
        def escape_shell_var(match):
            var_content = match.group(1)
            # 转义 $ { } 为 HTML 实体
            return f'&#36;&#123;{var_content}&#125;'
        
        content = re.sub(r'\$\{([^}]+)\}', escape_shell_var, content)
        
        # 3.7 转义 REST API 路径参数 {id}, {taskId} 等
        # 这些会被误认为 JSX 表达式
        # 匹配 {word} 格式，但排除纯数字（如 {1} 可能是合法对象字面量）
        def escape_path_param(match):
            inner = match.group(1)
            # 如果是纯数字，不转义（可能是对象字面量）
            if inner.isdigit():
                return match.group(0)
            # 其他情况转义
            return f'\\{{{inner}}}'
        
        # 匹配简单的 {identifier} 格式（不包括冒号，避免匹配 {PORT:8080} 这种 shell 变量默认值语法）
        content = re.sub(r'\{([a-zA-Z_][a-zA-Z0-9_]*)\}', escape_path_param, content)
        
        # 步骤4: 恢复保护的标签
        # 从后往前恢复，避免嵌套问题
        for placeholder in sorted(placeholders.keys(), reverse=True):
            original = placeholders[placeholder]
            content = content.replace(placeholder, original)

        # 步骤4.5: 对代码块内的 URL 进行转义，避免 MDX 的 URL 解析问题
        # Docusaurus/MDX 在某些情况下会尝试解析代码块内的 URL，导致错误
        # 使用零宽断言来匹配代码块内的 http:// 和 https://
        def escape_urls_in_code_blocks(text):
            result = []
            i = 0
            while i < len(text):
                if text[i:i+3] == '```':
                    # 找到代码块结束
                    end_idx = text.find('\n```', i + 3)
                    if end_idx == -1:
                        result.append(text[i:])
                        break
                    end_idx = end_idx + 4
                    code_block = text[i:end_idx]
                    # 转义 URL 中的冒号
                    code_block = code_block.replace('http://', 'http&#58;//')
                    code_block = code_block.replace('https://', 'https&#58;//')
                    result.append(code_block)
                    i = end_idx
                else:
                    result.append(text[i])
                    i += 1
            return ''.join(result)

        content = escape_urls_in_code_blocks(content)

        # 步骤5: 确保自闭合标签使用正确的格式（<br /> 而不是 <br>）
        # 这在 MDX 表格中特别重要
        self_closing_tags = ['br', 'hr', 'img', 'input', 'meta', 'link', 'area', 'base', 'col', 
                            'embed', 'param', 'source', 'track', 'wbr']
        for tag in self_closing_tags:
            # 将 <tag> 转换为 <tag />，但避免重复转换 <tag />
            content = re.sub(rf'<{tag}\b([^<>]*[^/])?>', rf'<{tag}\1 />', content)
            content = re.sub(rf'<{tag}\b\s*/\s*>', rf'<{tag} />', content)  # 标准化已有斜杠的格式
        
        return content
    
    def _get_phase_tag(self, phase_label: str) -> str:
        """获取阶段标签"""
        if '阶段一' in phase_label:
            return 'phase-1'
        elif '阶段二' in phase_label:
            return 'phase-2'
        elif '阶段三' in phase_label:
            return 'phase-3'
        elif '阶段四' in phase_label:
            return 'phase-4'
        return 'phase-unknown'
    
    def _truncate_text(self, text: str, max_length: int) -> str:
        """截断长文本"""
        if len(text) <= max_length:
            return text
        return text[:max_length - 3] + '...'
    
    def _get_status_badge(self, status: str) -> str:
        """获取状态徽章"""
        status_map = {
            'verified': '✅ 已验证',
            'pending': '⏳ 待验证',
            'failed': '❌ 失败',
            'draft': '📝 草稿',
        }
        return status_map.get(status.lower(), status)
    
    def _escape_inline_text(self, text: str) -> str:
        """转义内联文本中的 MDX 敏感字符（用于表格等内联环境）"""
        import re
        import uuid
        
        # 合法的 HTML 标签（这些不应该被转义）
        html_tags = {'br', 'hr', 'img', 'input', 'meta', 'link', 'area', 'base', 'col', 
                     'embed', 'param', 'source', 'track', 'wbr'}
        
        # 使用占位符保护合法标签
        placeholder_prefix = f"__HTML_{uuid.uuid4().hex[:6]}__"
        placeholders = {}
        placeholder_id = 0
        
        def make_placeholder():
            nonlocal placeholder_id
            ph = f"{placeholder_prefix}_{placeholder_id}_"
            placeholder_id += 1
            return ph
        
        # 保护合法的自闭合 HTML 标签 <tag> 或 <tag />
        for tag in html_tags:
            # 匹配 <tag> 或 <tag />
            pattern = rf'<{tag}\s*/?>'
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            for match in reversed(matches):
                placeholder = make_placeholder()
                placeholders[placeholder] = match.group(0)
                text = text[:match.start()] + placeholder + text[match.end():]
        
        # 转义空尖括号 <>
        text = re.sub(r'<>', '&lt;&gt;', text)
        
        # 转义泛型类型 <String>, <Integer> 等
        def escape_tag(match):
            tag = match.group(1)
            return f'&lt;{tag}&gt;'
        
        text = re.sub(r'<([A-Za-z0-9_]+)>', escape_tag, text)
        
        # 转义比较表达式 <60 等
        text = re.sub(r'<(\d+)(?=[^\d>])', lambda m: f'&lt;{m.group(1)}', text)
        
        # 转义 REST API 路径参数 {id}, {taskId} 等
        def escape_path_param(match):
            inner = match.group(1)
            if inner.isdigit():
                return match.group(0)
            return f'\\{{{inner}}}'
        
        text = re.sub(r'\{([a-zA-Z_][a-zA-Z0-9_]*)\}', escape_path_param, text)
        
        # 恢复保护的 HTML 标签
        for placeholder in sorted(placeholders.keys(), reverse=True):
            text = text.replace(placeholder, placeholders[placeholder])
        
        return text


# =============================================================================
# Sidebars 生成器
# =============================================================================

class SidebarsGenerator:
    """sidebars.ts 生成器"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def generate(self, phases: list[PhaseInfo]) -> str:
        """生成 sidebars.ts 内容"""
        lines = [
            '/**',
            ' * Copyright (c) CampusFlow Authors.',
            ' *',
            ' * This source code is licensed under the MIT license.',
            ' */',
            '',
            'import type {SidebarsConfig} from \'@docusaurus/plugin-content-docs\';',
            '',
            'const sidebars: SidebarsConfig = {',
            '  tutorialSidebar: [',
        ]
        
        # 添加各阶段
        for phase in phases:
            lines.extend(self._generate_phase_section(phase))
        
        # 添加全局页面
        lines.extend([
            '    // 全局页面',
            '    { type: \'doc\', id: \'syllabus\', label: \'教学大纲\' },',
            '    { type: \'doc\', id: \'campusflow\', label: \'CampusFlow\' },',
            '    { type: \'doc\', id: \'glossary\', label: \'术语表\' },',
            '    { type: \'doc\', id: \'style-guide\', label: \'风格指南\' },',
            '  ],',
            '};',
            '',
            'export default sidebars;',
            '',
        ])
        
        return '\n'.join(lines)
    
    def _generate_phase_section(self, phase: PhaseInfo) -> list[str]:
        """生成阶段部分"""
        lines = [
            f'    // {phase.label}',
            '    {',
            '      type: \'category\',',
            f'      label: \'{phase.label}\',',
            '      collapsed: false,',
            '      items: [',
        ]
        
        for week in phase.weeks:
            lines.extend(self._generate_week_section(week))
        
        lines.extend([
            '      ],',
            '    },',
            '',
        ])
        
        return lines
    
    def _generate_week_section(self, week: WeekInfo) -> list[str]:
        """生成周部分"""
        week_id = f'{week.number:02d}'
        
        lines = [
            '        {',
            '          type: \'category\',',
            f'          label: \'Week {week_id}: {week.title}\',',
            '          collapsed: true,',
            '          items: [',
            f'            {{ type: \'doc\', id: \'weeks/{week_id}/index\', label: \'周主页\' }},',
        ]
        
        # 根据存在的文件添加链接
        if week.has_chapter:
            lines.append(f'            {{ type: \'doc\', id: \'weeks/{week_id}/chapter\', label: \'讲义\' }},')
        
        if week.has_assignment:
            lines.append(f'            {{ type: \'doc\', id: \'weeks/{week_id}/assignment\', label: \'作业\' }},')
        
        if week.has_rubric:
            lines.append(f'            {{ type: \'doc\', id: \'weeks/{week_id}/rubric\', label: \'评分标准\' }},')
        
        # 代码页面（只要有代码目录就添加）
        if week.has_examples or week.has_starter_code or week.has_tests:
            lines.append(f'            {{ type: \'doc\', id: \'weeks/{week_id}/code\', label: \'代码\' }},')
        
        if week.has_anchors:
            lines.append(f'            {{ type: \'doc\', id: \'weeks/{week_id}/anchors\', label: \'锚点\' }},')
        
        if week.has_terms:
            lines.append(f'            {{ type: \'doc\', id: \'weeks/{week_id}/terms\', label: \'术语\' }},')

        lines.extend([
            '          ],',
            '        },',
        ])
        
        return lines


# =============================================================================
# 站点构建器
# =============================================================================

class SiteBuilder:
    """Docusaurus 站点构建器"""
    
    def __init__(
        self,
        site_dir: Path,
        chapters_dir: Path,
        shared_dir: Path,
        repo_root: Path,
        logger: logging.Logger
    ):
        self.site_dir = site_dir
        self.chapters_dir = chapters_dir
        self.shared_dir = shared_dir
        self.repo_root = repo_root
        self.logger = logger
        
        self.docs_dir = site_dir / 'docs'
        self.toc_parser = TOCParser(chapters_dir, logger)
        self.content_generator = ContentGenerator(chapters_dir, shared_dir, logger)
        self.sidebars_generator = SidebarsGenerator(logger)
    
    def build(self) -> bool:
        """构建站点"""
        self.logger.info("=" * 60)
        self.logger.info("开始构建 Docusaurus 站点")
        self.logger.info("=" * 60)
        
        try:
            # 1. 解析 TOC
            self.logger.info("\n[1/4] 解析 TOC.md...")
            phases = self.toc_parser.parse()
            self.logger.info(f"  发现 {len(phases)} 个阶段")
            
            # 2. 创建目录结构
            self.logger.info("\n[2/4] 创建目录结构...")
            self._create_directory_structure(phases)
            
            # 3. 生成每周页面
            self.logger.info("\n[3/4] 生成每周页面...")
            self._generate_week_pages(phases)
            
            # 4. 生成全局页面和 sidebars
            self.logger.info("\n[4/4] 生成全局页面和 sidebars...")
            self._generate_global_pages()
            self._generate_sidebars(phases)
            
            self.logger.info("\n" + "=" * 60)
            self.logger.info("站点构建完成！")
            self.logger.info(f"输出目录: {self.site_dir.absolute()}")
            self.logger.info("=" * 60)
            
            return True
            
        except Exception as e:
            self.logger.error(f"构建失败: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return False
    
    def _create_directory_structure(self, phases: list[PhaseInfo]) -> None:
        """创建目录结构"""
        # 创建 docs 目录
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"  创建目录: {self.docs_dir}")
        
        # 为每周创建目录
        for phase in phases:
            for week in phase.weeks:
                week_dir = self.docs_dir / 'weeks' / f'{week.number:02d}'
                week_dir.mkdir(parents=True, exist_ok=True)
                self.logger.debug(f"  创建目录: {week_dir}")
    
    def _generate_week_pages(self, phases: list[PhaseInfo]) -> None:
        """生成每周页面"""
        total_weeks = sum(len(phase.weeks) for phase in phases)
        processed = 0
        
        for phase in phases:
            for week in phase.weeks:
                processed += 1
                self.logger.info(f"  [{processed}/{total_weeks}] Week {week.number:02d}: {week.title}")
                
                week_dir = self.docs_dir / 'weeks' / f'{week.number:02d}'
                
                # 生成各页面
                self._write_file(week_dir / 'index.mdx', self.content_generator.generate_week_index(week))
                self._write_file(week_dir / 'chapter.mdx', self.content_generator.generate_chapter(week))
                self._write_file(week_dir / 'assignment.mdx', self.content_generator.generate_assignment(week))
                self._write_file(week_dir / 'rubric.mdx', self.content_generator.generate_rubric(week))
                self._write_file(week_dir / 'code.mdx', self.content_generator.generate_code(week))
                self._write_file(week_dir / 'anchors.mdx', self.content_generator.generate_anchors(week))
                self._write_file(week_dir / 'terms.mdx', self.content_generator.generate_terms(week))
    
    def _generate_global_pages(self) -> None:
        """生成全局页面"""
        self.logger.info("  生成全局页面...")
        
        self._write_file(self.docs_dir / 'index.mdx', self.content_generator.generate_index(self.repo_root))
        self._write_file(self.docs_dir / 'syllabus.mdx', self.content_generator.generate_syllabus())
        self._write_file(self.docs_dir / 'campusflow.mdx', self.content_generator.generate_campusflow())
        self._write_file(self.docs_dir / 'glossary.mdx', self.content_generator.generate_glossary())
        self._write_file(self.docs_dir / 'style-guide.mdx', self.content_generator.generate_style_guide())
    
    def _generate_sidebars(self, phases: list[PhaseInfo]) -> None:
        """生成 sidebars.ts (仅当文件不存在时)"""
        sidebars_path = self.site_dir / 'sidebars.ts'
        # 如果 sidebars.ts 已存在，不要覆盖它
        if sidebars_path.exists():
            self.logger.debug(f"  sidebars.ts 已存在，跳过生成")
            return
        sidebars_content = self.sidebars_generator.generate(phases)
        self._write_file(sidebars_path, sidebars_content)
    
    def _write_file(self, path: Path, content: str) -> None:
        """写入文件"""
        try:
            path.write_text(content, encoding='utf-8')
            self.logger.debug(f"  写入文件: {path}")
        except Exception as e:
            self.logger.warning(f"  写入文件失败: {path} - {e}")


# =============================================================================
# 命令行参数解析
# =============================================================================

def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='从 chapters/ 目录自动生成 Docusaurus 站点内容',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
    python scripts/build_site.py
    python scripts/build_site.py --verbose
    python scripts/build_site.py --site-dir ./my-site --chapters-dir ./content
        '''
    )
    
    parser.add_argument(
        '--site-dir',
        type=str,
        default='site',
        help='站点输出目录 (默认: site)'
    )
    
    parser.add_argument(
        '--chapters-dir',
        type=str,
        default='chapters',
        help='chapters 目录路径 (默认: chapters)'
    )
    
    parser.add_argument(
        '--shared-dir',
        type=str,
        default='shared',
        help='shared 目录路径 (默认: shared)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='启用详细输出'
    )
    
    return parser.parse_args()


# =============================================================================
# 主函数
# =============================================================================

def main() -> int:
    """主函数"""
    args = parse_args()
    
    # 设置日志
    logger = setup_logging(args.verbose)
    
    # 解析路径
    repo_root = Path.cwd()
    site_dir = repo_root / args.site_dir
    chapters_dir = repo_root / args.chapters_dir
    shared_dir = repo_root / args.shared_dir
    
    # 验证目录存在
    if not chapters_dir.exists():
        logger.error(f"chapters 目录不存在: {chapters_dir}")
        return 1
    
    if not shared_dir.exists():
        logger.warning(f"shared 目录不存在: {shared_dir}")
        shared_dir = chapters_dir  # 使用 chapters 作为 fallback
    
    # 创建站点构建器
    builder = SiteBuilder(
        site_dir=site_dir,
        chapters_dir=chapters_dir,
        shared_dir=shared_dir,
        repo_root=repo_root,
        logger=logger
    )
    
    # 构建站点
    success = builder.build()
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())

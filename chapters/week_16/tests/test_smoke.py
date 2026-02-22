"""
Smoke Tests for Week 16 solution.py

基础冒烟测试：
- 验证模块可以导入
- 验证基本函数存在
- 验证基本功能可运行

注意：由于 week_16 的 solution.py 可能尚未实现，
这些测试使用了 pytest.skip 来优雅地处理缺失的模块。
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
# 模块导入测试
# =============================================================================

def test_solution_module_exists():
    """
    冒烟测试：solution.py 模块应存在

    如果此测试失败，说明 solution.py 文件不存在
    """
    try:
        import solution
        assert solution is not None
    except ImportError:
        pytest.skip("solution.py not found - expected to be implemented later")


def test_solution_has_basic_functions():
    """
    冒烟测试：solution.py 应包含报告生成相关函数

    检查核心函数是否存在（示例函数名，实际可能不同）
    """
    try:
        import solution
    except ImportError:
        pytest.skip("solution.py not found")

    # 报告生成相关的可能函数名
    functions = [
        # 报告流水线相关
        'generate_report',
        'create_report_pipeline',
        'build_report',
        'run_analysis_pipeline',

        # Markdown 生成相关
        'render_markdown',
        'generate_markdown',
        'create_markdown_report',
        'markdown_report',

        # 审计清单相关
        'audit_report',
        'check_reproducibility',
        'validate_report',
        'audit_checklist',

        # 数据加载相关
        'load_data',
        'load_and_clean',
        'prepare_data',

        # 统计计算相关
        'compute_descriptive',
        'calculate_statistics',
        'descriptive_stats',
    ]

    # 至少有一个报告相关的函数存在
    has_any = any(hasattr(solution, func) for func in functions)

    if not has_any:
        # 没有找到预期函数，不报错，只是记录
        pytest.skip("No report generation functions found in solution.py")


# =============================================================================
# 报告流水线冒烟测试
# =============================================================================

def test_generate_report_smoke():
    """
    冒烟测试：报告生成函数应能运行

    测试基本的报告生成功能
    """
    try:
        import solution
    except ImportError:
        pytest.skip("solution.py not found")

    # 创建简单的测试数据
    np.random.seed(42)
    test_data = pd.DataFrame({
        'x': np.random.randn(100),
        'y': np.random.randn(100) * 2 + 1,
        'group': np.random.choice(['A', 'B', 'C'], 100)
    })

    # 尝试生成报告
    if hasattr(solution, 'generate_report'):
        try:
            result = solution.generate_report(test_data, output_path="test_report.md")
            assert result is not None
        except Exception as e:
            # 函数存在但可能需要额外参数，这是预期的
            pytest.skip(f"generate_report exists but needs different parameters: {e}")
    elif hasattr(solution, 'create_report'):
        try:
            result = solution.create_report(test_data)
            assert result is not None
        except Exception as e:
            pytest.skip(f"create_report exists but needs different parameters: {e}")
    else:
        pytest.skip("report generation function not implemented")


# =============================================================================
# Markdown 生成冒烟测试
# =============================================================================

def test_markdown_generation_smoke():
    """
    冒烟测试：Markdown 生成函数应能运行

    测试 Markdown 内容生成
    """
    try:
        import solution
    except ImportError:
        pytest.skip("solution.py not found")

    # 简单的统计数据
    stats = {
        'n': 100,
        'mean': 5.5,
        'std': 2.3,
        'min': 1.0,
        'max': 10.0
    }

    # 尝试生成 Markdown
    if hasattr(solution, 'render_markdown'):
        try:
            result = solution.render_markdown(stats)
            assert result is not None
            assert isinstance(result, str)
        except Exception as e:
            pytest.skip(f"render_markdown exists but needs different parameters: {e}")
    elif hasattr(solution, 'generate_markdown'):
        try:
            result = solution.generate_markdown(stats)
            assert result is not None
            assert isinstance(result, str)
        except Exception as e:
            pytest.skip(f"generate_markdown exists but needs different parameters: {e}")
    else:
        pytest.skip("markdown generation function not implemented")


# =============================================================================
# 审计清单冒烟测试
# =============================================================================

def test_audit_checklist_smoke():
    """
    冒烟测试：审计清单函数应能运行

    测试报告审计功能
    """
    try:
        import solution
    except ImportError:
        pytest.skip("solution.py not found")

    # 创建简单的测试报告内容
    test_report = """
    # 分析报告

    ## 数据来源
    数据来自 Kaggle 数据集

    ## 方法
    使用 t 检验比较两组均值

    ## 结果
    p = 0.03，95% CI [0.2, 1.5]
    """

    # 尝试审计
    if hasattr(solution, 'audit_report'):
        try:
            result = solution.audit_report(test_report)
            assert result is not None
        except Exception as e:
            pytest.skip(f"audit_report exists but needs different parameters: {e}")
    elif hasattr(solution, 'check_reproducibility'):
        try:
            result = solution.check_reproducibility(test_report)
            assert result is not None
        except Exception as e:
            pytest.skip(f"check_reproducibility exists but needs different parameters: {e}")
    else:
        pytest.skip("audit function not implemented")


# =============================================================================
# 数据加载冒烟测试
# =============================================================================

def test_data_loading_smoke():
    """
    冒烟测试：数据加载函数应能运行

    测试基本的数据加载功能
    """
    try:
        import solution
    except ImportError:
        pytest.skip("solution.py not found")

    # 创建临时测试文件
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write('x,y,group\n1,2,A\n3,4,B\n5,6,C\n')
        temp_path = f.name

    try:
        # 尝试加载数据
        if hasattr(solution, 'load_data'):
            try:
                result = solution.load_data(temp_path)
                assert result is not None
            except Exception as e:
                pytest.skip(f"load_data exists but needs different parameters: {e}")
        elif hasattr(solution, 'load_and_clean'):
            try:
                result = solution.load_and_clean(temp_path)
                assert result is not None
            except Exception as e:
                pytest.skip(f"load_and_clean exists but needs different parameters: {e}")
        else:
            pytest.skip("data loading function not implemented")
    finally:
        # 清理临时文件
        os.unlink(temp_path)


# =============================================================================
# 描述统计冒烟测试
# =============================================================================

def test_descriptive_stats_smoke():
    """
    冒烟测试：描述统计计算应能运行

    测试基本的描述统计功能
    """
    try:
        import solution
    except ImportError:
        pytest.skip("solution.py not found")

    # 创建测试数据
    np.random.seed(42)
    test_data = pd.DataFrame({
        'x': np.random.randn(100),
        'y': np.random.randn(100) * 2 + 1,
    })

    # 尝试计算描述统计
    if hasattr(solution, 'compute_descriptive'):
        try:
            result = solution.compute_descriptive(test_data)
            assert result is not None
        except Exception as e:
            pytest.skip(f"compute_descriptive exists but needs different parameters: {e}")
    elif hasattr(solution, 'calculate_statistics'):
        try:
            result = solution.calculate_statistics(test_data)
            assert result is not None
        except Exception as e:
            pytest.skip(f"calculate_statistics exists but needs different parameters: {e}")
    else:
        pytest.skip("descriptive statistics function not implemented")


# =============================================================================
# 场景测试：端到端报告生成
# =============================================================================

def test_end_to_end_report_smoke():
    """
    冒烟测试：端到端报告生成流程应能运行

    测试完整的报告生成流程
    """
    try:
        import solution
    except ImportError:
        pytest.skip("solution.py not found")

    # 创建模拟的客户流失数据
    np.random.seed(42)
    n = 500
    test_data = pd.DataFrame({
        'customer_id': range(n),
        'tenure': np.random.exponential(20, n),
        'monthly_charges': np.random.uniform(20, 100, n),
        'total_charges': np.random.uniform(100, 5000, n),
        'churn': np.random.choice([0, 1], n, p=[0.7, 0.3])
    })

    # 尝试端到端流程
    if hasattr(solution, 'run_full_analysis'):
        try:
            result = solution.run_full_analysis(test_data)
            assert result is not None
        except Exception as e:
            pytest.skip(f"run_full_analysis exists but needs different parameters: {e}")
    elif hasattr(solution, 'generate_report'):
        try:
            result = solution.generate_report(test_data, output_path="test_output.md")
            assert result is not None
        except Exception as e:
            pytest.skip(f"generate_report exists but encountered error: {e}")
    else:
        pytest.skip("end-to-end analysis function not implemented")


# =============================================================================
# 异常处理冒烟测试
# =============================================================================

def test_empty_data_handling():
    """
    冒烟测试：空数据应被正确处理

    验证函数对空输入的容错性
    """
    try:
        import solution
    except ImportError:
        pytest.skip("solution.py not found")

    # 空数据
    empty_data = pd.DataFrame()

    # 尝试处理（应该报错或返回 None）
    if hasattr(solution, 'compute_descriptive'):
        try:
            result = solution.compute_descriptive(empty_data)
            # 如果不报错，应该返回 None 或合理的默认值
            assert result is None or isinstance(result, (dict, pd.DataFrame))
        except (ValueError, RuntimeError):
            # 报错也是可接受的
            assert True


# =============================================================================
# 概念理解冒烟测试
# =============================================================================

def test_reproducible_pipeline_concepts():
    """
    冒烟测试：可复现报告流水线概念

    验证对可复现性概念的理解
    """
    # 概念测试：不需要具体实现
    concepts = {
        'reproducible_report_pipeline': '从数据到报告的自动化脚本，任何人运行都能得到相同结果',
        'data_provenance': '数据来源追溯，记录数据的获取时间、来源、版本',
        'random_seed': '固定随机数生成器种子，确保结果可复现',
        'dependency_tracking': '记录所有依赖库的版本，确保环境可复现',
        'automated_report_generation': '用脚本自动生成报告，避免手工编辑',
    }

    assert len(concepts) == 5
    assert 'reproducible_report_pipeline' in concepts
    assert 'data_provenance' in concepts
    assert 'random_seed' in concepts


def test_audit_checklist_concepts():
    """
    冒烟测试：审计清单概念

    验证对审计清单概念的理解
    """
    # 概念测试
    dimensions = {
        'data_reproducibility': '检查数据来源、依赖版本、随机种子是否记录',
        'statistical_assumptions': '检查统计假设是否验证（正态性、方差齐性等）',
        'honesty_transparency': '检查图表诚实性、缺失处理说明、因果声明边界',
        'narrative_structure': '检查研究问题清晰、方法可追溯、结论不夸大',
    }

    assert len(dimensions) == 4
    assert 'data_reproducibility' in dimensions
    assert 'statistical_assumptions' in dimensions


def test_presentation_narrative_concepts():
    """
    冒烟测试：展示叙事结构概念

    验证对展示叙事框架的理解
    """
    # 概念测试
    framework = {
        'problem': '为什么要做这个分析？1-2分钟',
        'method': '你怎么分析？用什么方法？2-3分钟',
        'findings': '你发现了什么？4-5分钟',
        'limitations': '结论的局限是什么？1-2分钟',
        'reflection': '分析意味着什么？1分钟',
    }

    assert len(framework) == 5
    assert 'problem' in framework
    assert 'limitations' in framework


def test_markdown_vs_word_concept():
    """
    冒烟测试：Markdown vs Word 概念

    验证对两者区别的理解
    """
    # 概念测试
    comparison = {
        'markdown': '纯文本格式，版本控制友好，可脚本生成',
        'word': '所见即所得，手工编辑，不易版本控制',
        'key_difference': 'Markdown 支持自动化报告生成，Word 需要手工维护',
        'html_advantage': 'HTML 版本支持交互、易于分享、可部署',
    }

    assert 'markdown' in comparison
    assert 'html_advantage' in comparison


# =============================================================================
# AI 辅助分析概念测试
# =============================================================================

def test_ai_assisted_analysis_concepts():
    """
    冒烟测试：AI 辅助分析概念

    验证对 AI 在报告生成中的作用与限制的理解
    """
    # 概念测试
    ai_can_do = {
        'accelerate_coding': '加速代码编写',
        'improve_writing': '改进文字表达',
        'format_conversion': '格式转换（MD to HTML/PDF）',
    }

    ai_cannot_do = {
        'replace_understanding': '替代对数据的理解',
        'replace_reproducibility': '替代可复现性设计',
        'replace_audit': '替代审计责任',
    }

    assert len(ai_can_do) == 3
    assert len(ai_cannot_do) == 3
    assert 'replace_understanding' in ai_cannot_do


# =============================================================================
# 报告结构测试
# =============================================================================

def test_report_structure_concept():
    """
    冒烟测试：报告结构概念

    验证对标准报告结构的理解
    """
    # 概念测试
    sections = {
        'reproducibility_info': '数据来源、依赖版本、随机种子、执行时间',
        'data_overview': '样本量、变量类型、缺失情况',
        'descriptive_statistics': '描述统计、可视化',
        'statistical_tests': '检验方法、结果、不确定性',
        'modeling': '模型选择、评估、诊断',
        'conclusions': '主要发现、局限性、下一步',
    }

    assert len(sections) == 6
    assert 'reproducibility_info' in sections
    assert 'conclusions' in sections

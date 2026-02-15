# Week 06 示例代码说明

本目录包含 Week 06 "假设检验、效应量与 AI 审查训练" 的所有示例代码。

## 示例文件列表

| 文件 | 说明 | 对应章节 |
|------|------|----------|
| `01_p_value_intuition.py` | p 值可视化演示 | 第 1 节 |
| `02_t_test_demo.py` | t 检验和比例检验演示 | 第 2 节 |
| `02_chi_square_demo.py` | 卡方检验演示 | 第 2 节 |
| `03_effect_size_demo.py` | 效应量计算演示 | 第 3 节 |
| `04_assumption_checks.py` | 前提假设检查演示 | 第 4 节 |
| `05_ai_review_demo.py` | AI 报告审查演示 | 第 5 节 |
| `06_statlab_hypothesis_test.py` | StatLab 假设检验报告生成 | StatLab 进度 |

## 运行方式

每个示例都可以独立运行：

```bash
python3 chapters/week_06/examples/01_p_value_intuition.py
python3 chapters/week_06/examples/02_t_test_demo.py
python3 chapters/week_06/examples/02_chi_square_demo.py
python3 chapters/week_06/examples/03_effect_size_demo.py
python3 chapters/week_06/examples/04_assumption_checks.py
python3 chapters/week_06/examples/05_ai_review_demo.py
python3 chapters/week_06/examples/06_statlab_hypothesis_test.py
```

## 输出文件

运行示例后，输出文件会保存到 `output/` 目录：

- `p_value_visualization.png` - p 值可视化图
- `qq_plot_comparison.png` - Q-Q 图对比
- `hypothesis_test_sections.md` - StatLab 假设检验报告片段
- `revised_hypothesis_test_report.md` - 修订后的假设检验报告

## 依赖安装

运行示例前，确保安装了必要的依赖：

```bash
pip3 install numpy scipy pandas seaborn statsmodels matplotlib
```

## StatLab 超级线

`06_statlab_hypothesis_test.py` 是 StatLab 超级线在 Week 06 的入口脚本，
在上周（不确定性量化）基础上，加入假设检验章节：p 值、效应量、前提假设检查。

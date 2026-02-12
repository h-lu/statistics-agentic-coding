# Week 07 示例代码

本目录包含 Week 07 的所有可运行示例代码，用于演示多组比较与多重比较的核心概念。

## 示例列表

| 文件 | 描述 | 运行方式 |
|------|------|----------|
| `01_f_distribution.py` | F 分布模拟与可视化 | `python3 chapters/week_07/examples/01_f_distribution.py` |
| `02_anova_example.py` | 单因素 ANOVA 完整实战 | `python3 chapters/week_07/examples/02_anova_example.py` |
| `03_posthoc_tukey.py` | Tukey HSD 事后检验 | `python3 chapters/week_07/examples/03_posthoc_tukey.py` |
| `04_chisquare_test.py` | 卡方检验（独立性检验） | `python3 chapters/week_07/examples/04_chisquare_test.py` |
| `05_ai_anova_review.py` | AI 生成的多组比较报告审查工具 | `python3 chapters/week_07/examples/05_ai_anova_review.py` |
| `99_statlab.py` | StatLab 超级线：多组比较报告生成 | `python3 chapters/week_07/examples/99_statlab.py` |

## 示例说明

### 01_f_distribution.py

演示 F 分布的模拟与可视化，帮助读者建立 ANOVA 的统计直觉。

**关键概念**：
- F 统计量的抽样分布
- 临界值与拒绝域
- 模拟分布与理论分布的对比

**输出**：
- 控制台：F 临界值
- 文件：`f_distribution_intuition.png`

### 02_anova_example.py

完整的单因素 ANOVA 分析流程。

**关键概念**：
- 前提假设检查（正态性、方差齐性）
- ANOVA 检验执行
- η² 效应量计算
- 结果可视化

**输出**：
- 控制台：假设检查结果、ANOVA 表、η² 值
- 文件：`anova_results.png`

### 03_posthoc_tukey.py

Tukey HSD 事后检验，找出具体哪些组对存在差异。

**关键概念**：
- Tukey HSD 原理
- 多重比较校正
- 均值差异与置信区间

**输出**：
- 控制台：Tukey HSD 结果表、显著差异列表
- 文件：`tukey_hsd_results.png`

### 04_chisquare_test.py

卡方检验（独立性检验），判断两个分类变量是否相关。

**关键概念**：
- 列联表
- 卡方检验执行
- Cramér's V 效应量
- 观测频数 vs 期望频数

**输出**：
- 控制台：列联表、卡方统计量、p 值、Cramér's V
- 文件：`chisquare_results.png`

### 05_ai_anova_review.py

AI 生成的多组比较报告审查工具，自动识别常见谬误。

**关键概念**：
- 6 个检查项
- 潜在问题识别
- 改进建议生成

**输出**：
- 控制台：发现的问题列表、改进建议、修订版报告

### 99_statlab.py

StatLab 超级线的 Week 07 更新，在 Week 06 报告基础上添加"多组比较结果"章节。

**关键概念**：
- 单因素 ANOVA（城市级别 vs 消费）
- Tukey HSD 事后检验
- 卡方检验（城市级别 vs 用户等级）
- 报告自动生成

**输出**：
- 控制台：分析进度、结果摘要
- 文件：更新后的 `report.md`

## 依赖要求

```bash
pip install numpy pandas scipy statsmodels matplotlib seaborn
```

## 运行建议

1. **逐个运行示例**：按编号顺序运行，理解每个步骤
2. **阅读输出**：每个示例都有详细的控制台输出
3. **查看图片**：生成的可视化图片有助于理解概念
4. **修改参数**：尝试修改代码中的参数，观察结果变化

## 与 CHAPTER.md 的对应关系

| 示例 | 对应章节 |
|------|----------|
| 01_f_distribution.py | 第 2 节：ANOVA 的核心思想——方差分解与 F 统计量 |
| 02_anova_example.py | 第 3 节：ANOVA 实战——从 F 显著到效应量 |
| 03_posthoc_tukey.py | 第 4 节：哪些组之间真的不同？——事后检验与多重比较校正 |
| 04_chisquare_test.py | 第 5 节：分类变量的关联检验——卡方检验 |
| 05_ai_anova_review.py | 第 6 节：AI 生成的多组比较报告能信吗？——审查训练 |
| 99_statlab.py | StatLab 进度：多组比较结果章节 |

## 注意事项

1. **中文字体问题**：matplotlib 可能在某些系统上无法正确显示中文，产生警告（但不影响功能）
2. **图片位置**：生成的图片文件默认保存在当前工作目录
3. **随机种子**：所有示例都设置了随机种子，确保结果可复现

## 测试

运行测试验证所有示例代码的正确性：

```bash
pytest chapters/week_07/tests/ -v
```

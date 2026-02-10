# 统计学与 Agentic 数据分析

> 一门培养"数据翻译官"的课程——既懂统计原理，又会用 AI 工具高效分析，更能判断结论是否可靠。

## 核心理念

**"AI 可以帮你跑统计检验，但不懂业务场景和数据背后的意义，只会得到错误的结论。"**

在 AI 时代，数据分析的门槛降低了，但做出正确判断的要求提高了。本课程培养三种核心能力：

1. **统计思维**：提出正确问题、选择合适方法、判断结果可靠性
2. **AI 协作**：用 AI 加速分析，但保持批判性思维
3. **数据沟通**：把数字转化为有说服力的故事

---

## 14 周课程结构

| 阶段 | 周次 | 能力目标 |
|------|------|---------|
| 数据探索基础 | 01–04 | 数据清洗、描述统计、可视化 |
| 统计推断 | 05–08 | 假设检验、置信区间、效应量 |
| 预测建模 | 09–11 | 回归、分类、模型解释 |
| 综合实战 | 12–14 | 期末数据项目 |

### Agentic 分析工作流

```
QUESTION → EXPLORE → MODEL → INTERPRET → COMMUNICATE
(提问)    → (探索)   → (建模) → (解释)   → (传达)
```

---

## 快速开始

```bash
# 1. 进入项目
cd statistics-agentic-coding

# 2. 安装依赖
pip install pandas numpy scipy statsmodels matplotlib seaborn scikit-learn jupyter

# 3. 验证环境
python scripts/validate_week.py --week week_01
```

---

## 每周学习流程

```
/team-week week_XX    # Agent Team 生成内容
/qa-week week_XX      # 质量检查
```

---

## 贯穿项目：数据探索者报告

选择一个真实数据集，完成完整分析：
- 明确研究问题
- EDA → 统计推断/预测建模 → 结果解释
- 数据可视化贯穿始终
- AI 使用记录和反思

---

## AI 协作四个阶段

| 阶段 | AI 角色 | 学生能力 |
|------|---------|----------|
| 观察期（01-03） | AI 演示代码 | 学习基础概念 |
| 识别期（04-06） | AI 生成分析 | 判断对错 |
| 协作期（07-10） | AI 执行计算 | 主导解释 |
| 主导期（11-14） | AI 辅助实现 | 独立设计方案 |

---

## 目录结构

```
chapters/
  SYLLABUS.md          # 教学大纲
  week_XX/
    CHAPTER.md         # 正文
    ASSIGNMENT.md      # 数据分析作业
    RUBRIC.md          # 评分标准
    examples/          # 代码示例
    starter_code/      # 起始代码

shared/
  book_project.md      # 贯穿项目设计
  ai_progression.md    # AI 协作指南
  characters.yml       # 角色设定

.claude/
  agents/              # Agent 配置
  skills/              # Skill 命令
```

---

## 技术栈

- **数据处理**：pandas, numpy
- **可视化**：matplotlib, seaborn, plotly
- **统计**：scipy.stats, statsmodels
- **机器学习**：scikit-learn
- **环境**：Jupyter Notebook

---

## 学习资源

- 教材：《Python 程序设计（Agentic Coding）》前置知识
- 在线：Kaggle Learn, Towards Data Science
- 实践：UCI ML Repository, 政府开放数据

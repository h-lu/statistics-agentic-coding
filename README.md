# 统计学与 Agentic 数据分析

一本面向“有 Python 基础（本书不讲 Python 入门）的数据分析初学者”的统计学教材，用 AI agent 团队协作的方式生产每一周的章包交付物。

## 这个项目是什么

16 周课程，分五个阶段：

| 阶段 | 周次 | 能力目标 |
|------|------|---------|
| 数据探索 | 01–04 | 能做 EDA、讲清数据质量与可视化 |
| 统计推断 | 05–08 | 能做检验/区间/重采样，并解释不确定性 |
| 预测建模 | 09–12 | 能做回归/分类/评估，并避免数据泄漏 |
| 高级专题 | 13–15 | 形成因果/贝叶斯/计算视角，知道边界与限制 |
| 综合实战 | 16 | 交付可复现分析报告与展示 |

每周交付一个**章包**（正文 + 示例 + 作业 + 测试 + QA），由 9 个 AI agent 协作产出，经校验脚本和 hooks 自动把关。

详细大纲见 [`chapters/SYLLABUS.md`](chapters/SYLLABUS.md)，目录见 [`chapters/TOC.md`](chapters/TOC.md)。

## 快速开始

```bash
# 1. 克隆并进入项目
git clone <repo-url> && cd statistics-agentic-coding

# 2. 一键环境搭建
make setup            # 创建 .venv 并安装依赖

# 3. 批量创建 16 周目录
make scaffold         # 从 TOC.md 读取标题，生成所有周的模板

# 4. 校验
make validate W=01    # 校验第 1 周（默认 release 模式）
make test W=01        # 跑第 1 周测试
make book-check       # 全书一致性检查
```

所有命令见 `make help`。

## 一周写作流程

在 Claude Code / Cursor 中打开本项目，使用 skill 命令：

```
/new-week 01 从数据到问题：你的第一份数据卡   # 1. 创建新周
/draft-chapter week_01               # 2. 完整写作流水线
                                     #    规划 → 写正文 → 润色 → QA → 修订回路
/polish-week week_01                 # 3. 再次深度润色
/make-assignment week_01             # 4. 生成作业 + 评分标准
/qa-week week_01                     # 5. 质量检查
/release-week week_01                # 6. 发布
/qa-book --mode fast                 # 7. 跨周一致性检查
```

或者用 agent team 并行产出：`/team-week week_01`

Gitea 协作流程见 [`shared/gitea_workflow.md`](shared/gitea_workflow.md)。

## 目录结构

```
chapters/
  SYLLABUS.md              # 16 周教学大纲（含贯穿项目与 AI 协作框架）
  TOC.md                   # 目录
  week_XX/                 # 每周一个章包
    CHAPTER.md             #   正文
    ASSIGNMENT.md          #   作业
    RUBRIC.md              #   评分标准
    QA_REPORT.md           #   质量报告（阻塞项/建议项/评分）
    ANCHORS.yml            #   可验证断言
    TERMS.yml              #   本周新术语
    examples/              #   示例代码
    starter_code/          #   作业起始代码 + solution.py
    tests/                 #   pytest 用例

shared/
  style_guide.md           # 行文风格规范
  writing_exemplars.md     # 写作范例库（好 vs 坏的具体对比）
  glossary.yml             # 全书术语表
  anchor_schema.md         # 锚点格式说明
  gitea_workflow.md        # Gitea PR 协作流程

.claude/
  agents/                  # 9 个专职 Agent（writer/polisher/qa/planner/...）
  skills/                  # 9 个 Skill 命令（/draft-chapter、/release-week、...）
  hooks/                   # 自动校验（TaskCompleted / TeammateIdle）
  settings.json            # Claude Code 项目配置

scripts/                   # 校验/构建脚本
Makefile                   # 快捷命令入口
```

## 写作质量体系

本项目对内容质量有系统性要求，不只是"知识点正确"：

- **场景驱动**：先让读者感受到需求，再引出概念
- **贯穿案例**：每章一个渐进式分析任务（数据故事/小报告），每节推进一步
- **StatLab 超级线**：全书一条可复现分析报告流水线（`report.md` / `report.html`），每章推进一次
- **禁止模板感**：不能每节都用相同结构，不能 bullet list 堆砌做小结
- **叙事质量评分**：`student-qa` agent 打 1-5 分，>= 4 分才能发布
- **修订回路**：QA 发现问题 → 回传 writer/polisher 修复 → 再次 QA

具体的好 / 坏写法对比见 [`shared/writing_exemplars.md`](shared/writing_exemplars.md)。

## Hooks 与自动校验

本项目在 `.claude/settings.json` 中启用了 agent teams（`CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1`）。

| Hook | 触发时机 | 作用 |
|------|---------|------|
| `TaskCompleted` | agent 标记任务完成时 | 跑 `validate_week.py --mode task` |
| `TeammateIdle` | teammate 空闲时 | 跑 `validate_week.py --mode idle` |

hooks 优先使用项目内 `.venv` 运行校验脚本，建议先跑 `make setup`。

## 开发约定

- 默认中文写作，关键术语可括注英文
- 重要断言必须落到 `ANCHORS.yml` 并提供可验证方式
- 新术语必须进入 `shared/glossary.yml`（校验脚本与 hooks 强制）
- 所有 task subject 以 `[week_XX]` 开头（hooks 依赖）

# Week 07 示例代码说明

本目录包含 Week 07 "多组比较与多重校正" 的所有示例代码。

## 示例文件列表

| 文件 | 说明 | 对应章节 |
|------|------|----------|
| `01_multiple_comparison_demo.py` | 多重比较问题演示 | 第 1 节 |
| `02_anova_basics.py` | ANOVA 基础演示 | 第 2 节 |
| `03_posthoc_tests.py` | 事后比较（Tukey HSD）演示 | 第 3 节 |
| `04_correction_methods.py` | 校正方法（Bonferroni/FDR）演示 | 第 4 节 |
| `05_ai_review_anova.py` | AI 报告审查演示 | 第 5 节 |
| `07_statlab_anova.py` | StatLab 多组比较报告生成 | StatLab 进度 |

## 运行方式

每个示例都可以独立运行：

```bash
# 基础示例
python3 chapters/week_07/examples/01_multiple_comparison_demo.py
python3 chapters/week_07/examples/02_anova_basics.py
python3 chapters/week_07/examples/03_posthoc_tests.py
python3 chapters/week_07/examples/04_correction_methods.py
python3 chapters/week_07/examples/05_ai_review_anova.py

# StatLab 超级线
python3 chapters/week_07/examples/07_statlab_anova.py
```

## 输出文件

运行示例后，输出文件会保存到相应目录：

### 图表文件（images/）
- `multiple_comparisons_simulation.png` - 多重比较假阳性分布图
- `anova_barplot.png` - ANOVA 条形图
- `tukey_hsd_plot.png` - Tukey HSD 事后比较结果图
- `correction_methods_comparison.png` - 校正方法对比图

### 报告文件（output/）
- `anova_sections.md` - StatLab 多组比较报告片段
- `revised_anova_report.md` - 修订后的 ANOVA 报告（AI 审查示例）

## 依赖安装

运行示例前，确保安装了必要的依赖：

```bash
pip3 install numpy scipy pandas seaborn statsmodels matplotlib
```

## 核心概念

### 多重比较问题（Family-wise Error Rate, FWER）

当你同时检验多个假设时，"至少一个假阳性"的概率会随检验次数指数增长：

```
FWER = 1 - (1 - α)^m
```

其中 α 是单个检验的显著性水平，m 是检验次数。

### ANOVA（方差分析）

通过比较"组间方差"和"组内方差"来判断多组均值是否有差异：

```
F = 组间方差 / 组内方差
```

- F ≈ 1：组间差异 ≈ 组内差异，各组均值可能相等
- F >> 1：组间差异 >> 组内差异，至少有一组均值不同

### 事后比较（Post-hoc Tests）

ANOVA 只告诉你"有差异"，不告诉你"哪一对有差异"。常用方法：

- **Tukey HSD**：自动控制 FWER，默认推荐
- **Bonferroni**：更保守，适合检验次数少时
- **Scheffé**：最保守，适合复杂比较

### 校正方法

| 方法 | 控制什么 | 特点 | 适用场景 |
|------|---------|------|---------|
| Bonferroni | FWER | 最保守 | 检验次数少，不能容忍假阳性 |
| FDR (BH) | False Discovery Rate | 更平衡 | 检验次数多，探索性研究 |

## StatLab 超级线

`07_statlab_anova.py` 是 StatLab 超级线在 Week 07 的入口脚本。

### 与上周对比

| 上周（Week 06） | 本周（Week 07） |
|------------------|------------------|
| 假设检验（p 值 + 效应量 + 假设检查） | 以上全部 + **多组比较（ANOVA + 事后比较 + 校正）** |
| 两两 t 检验 | 多组 ANOVA + Tukey HSD 校正 |
| 可能忽略多重比较问题 | 明确说明校正策略 |

### 本周改进

- 加入多组比较章节：ANOVA + 事后比较 + 校正策略
- 明确说明使用的校正方法（Tukey HSD）
- 在报告中体现 family-wise error rate 的控制

## 角色出场

- **小北**（第 1 节）：误以为"多检验几个假设没什么"，引出多重比较问题
- **阿码**（第 4 节）：追问"Bonferroni 太保守了怎么办？"，引出 FDR 方法
- **老潘**（第 5 节）：点评"这是在赌假阳性"，强调 AI 时代分析者的责任

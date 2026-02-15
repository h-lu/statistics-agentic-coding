# Week 08 示例代码

## 目录

- [01_confidence_interval_basics.py](#01_confidence_interval_basicspy) - 点估计 vs 区间估计
- [02_ci_interpretation.py](#02_ci_interpretationpy) - CI 的正确解释
- [03_bootstrap_method.py](#03_bootstrap_methodpy) - Bootstrap 方法
- [04_bootstrap_ci_methods.py](#04_bootstrap_ci_methodspy) - Bootstrap CI 方法对比
- [05_permutation_test.py](#05_permutation_testpy) - 置换检验
- [08_statlab_ci.py](#08_statlab_cipy) - StatLab CI 生成

---

## 01_confidence_interval_basics.py

**主题**：从点估计到区间估计——量化你的不确定性

**运行方式**：
```bash
python3 chapters/week_08/examples/01_confidence_interval_basics.py
```

**内容**：
- 坏例子：只报告点估计（没有不确定性）
- 好例子：报告点估计 + 置信区间
- 样本量对 CI 宽度的影响
- 标准误与 CI 的关系
- 可视化：不同样本量的 CI 对比

**输出**：
- stdout：点估计与区间估计的对比
- images/01_point_vs_interval.png：CI 可视化

**核心概念**：
- 点估计：用一个数字总结数据，但不反映不确定性
- 区间估计：点估计 ± 范围，量化"有多确定"
- CI 宽度由标准误决定：SE = SD / √n

---

## 02_ci_interpretation.py

**主题**：置信区间的正确解释——不要误读它

**运行方式**：
```bash
python3 chapters/week_08/examples/02_ci_interpretation.py
```

**内容**：
- 常见误解：真实均值有 95% 概率落在 CI 内（错！）
- 正确理解：重复抽样 100 次，约 95 个 CI 会包含真实均值
- 模拟演示：重复抽样，看覆盖率
- CI 与假设检验的关系
- 频率学派 vs 贝叶斯学派

**输出**：
- stdout：CI 的常见误解与正确理解
- images/02_ci_interpretation.png：重复抽样可视化

**核心概念**：
- 随机的是区间，不是参数
- CI 与假设检验等价：CI 包含 0 ↔ p ≥ 0.05

---

## 03_bootstrap_method.py

**主题**：Bootstrap 重采样——从"假设分布"到"让数据说话"

**运行方式**：
```bash
python3 chapters/week_08/examples/03_bootstrap_method.py
```

**内容**：
- Bootstrap 核心思想：从样本中重采样
- 坏例子：小样本非正态数据盲目用 t 公式
- 好例子：用 Bootstrap 估计 CI
- Bootstrap vs 理论公式对比
- Bootstrap 估计中位数（演示可扩展性）

**输出**：
- stdout：Bootstrap 方法演示
- images/03_bootstrap_distribution.png：Bootstrap 分布可视化

**核心概念**：
- Bootstrap：从样本中有放回地抽取很多个样本
- 优势：不依赖分布假设，适用于任何统计量
- 什么时候用：非正态数据、复杂统计量

---

## 04_bootstrap_ci_methods.py

**主题**：Bootstrap 置信区间方法——从 Percentile 到 BCa

**运行方式**：
```bash
python3 chapters/week_08/examples/04_bootstrap_ci_methods.py
```

**内容**：
- Percentile Bootstrap：最简单
- BCa Bootstrap：Bias-Corrected and Accelerated
- 偏态数据下的方法对比
- Bootstrap 什么时候失效
- 样本量选择指导

**输出**：
- stdout：不同 Bootstrap CI 方法的对比
- images/04_bootstrap_ci_methods.png：方法对比可视化

**核心概念**：
- Percentile：简单直观，适合对称分布
- BCa：最准确，自动校正偏差和加速
- 建议：正态+大样本用 Percentile，偏态+小样本用 BCa

---

## 05_permutation_test.py

**主题**：置换检验——当"零假设"是"没有差异"时

**运行方式**：
```bash
python3 chapters/week_08/examples/05_permutation_test.py
```

**内容**：
- 置换检验核心思想：打乱标签
- 坏例子：小样本非正态数据用 t 检验
- 好例子：用置换检验
- t 检验 vs 置换检验对比
- 用置换检验比较中位数

**输出**：
- stdout：置换检验演示
- images/05_permutation_test.png：置换分布可视化

**核心概念**：
- 核心思想：如果零假设成立，组别标签没意义
- 优势：不依赖分布假设，适用于小样本或非正态数据
- 限制：假设独立性、样本代表性要好

---

## 08_statlab_ci.py

**主题**：StatLab 不确定性量化报告生成

**运行方式**：
```bash
python3 chapters/week_08/examples/08_statlab_ci.py
```

**内容**：
- 单组 CI：Adelie 和 Gentoo 企鹅的喙长均值
- 组间比较：Adelie vs Gentoo、Adelie vs Chinstrap
- 使用三种方法：t 分布、Percentile Bootstrap、BCa Bootstrap
- 置换检验 p 值和效应量

**输出**：
- stdout：报告片段预览
- output/uncertainty_sections.md：完整报告片段

**与上周对比**：
- 上周：p 值 + 效应量 + 假设检查 + 多组比较
- 本周：以上全部 + 区间估计 + Bootstrap + 置换检验

---

## 运行所有示例

```bash
# 逐个运行
for file in chapters/week_08/examples/*.py; do
    python3 "$file"
done

# 或使用 pytest
pytest chapters/week_08/tests/test_examples.py -v
```

---

## 图片输出

所有图片保存在 `chapters/week_08/images/` 目录：

- 01_point_vs_interval.png - 样本量对 CI 的影响
- 02_ci_interpretation.png - 重复抽样与覆盖率
- 03_bootstrap_distribution.png - Bootstrap 分布
- 04_bootstrap_ci_methods.png - Bootstrap 方法对比
- 05_permutation_test.png - 置换分布可视化

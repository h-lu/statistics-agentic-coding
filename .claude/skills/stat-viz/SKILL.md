---
name: stat-viz
description: 创建高质量统计图表的指南。包含图表类型选择、常见错误避免、中文字体配置、代码模板。
argument-hint: "<week_id> [chart_type]"
allowed-tools: Read, Write, Edit, Grep, Glob, Bash, WebSearch
disable-model-invocation: true
---

# /stat-viz

## 用法

```
/stat-viz week_XX              # 显示本周应该生成的图表类型
/stat-viz week_XX histogram    # 生成特定类型的图表
```

## 目标

为 `chapters/week_XX/` 创建高质量的统计图表：
- 图表代码保存在 `examples/NN_chart_xxx.py`（可复现）
- 图片输出到 `images/xxx.png`（嵌入正文）
- 确保中文字体正确显示
- 避免常见的可视化错误

---

## 第一部分：图表类型选择指南

### 什么时候用什么图

| 你想展示什么 | 推荐图表 | Seaborn 函数 |
|-------------|---------|-------------|
| 单变量分布 | 直方图 + KDE | `sns.histplot(data, kde=True)` |
| 分布形状 + 离群点 | 箱线图 | `sns.boxplot(x=data)` |
| 多组分布对比 | 小提琴图 | `sns.violinplot(x=group, y=value)` |
| 两组变量关系 | 散点图 | `sns.scatterplot(x=x, y=y)` |
| 关系 + 趋势线 | 散点图 + 回归 | `sns.regplot(x=x, y=y)` |
| 多变量关系 | 相关矩阵热图 | `sns.heatmap(corr, annot=True)` |
| 分类变量分布 | 计数图 | `sns.countplot(x=cat)` |
| 时间序列趋势 | 折线图 | `sns.lineplot(x=time, y=value)` |

### 不推荐的图表类型

| 避免 | 原因 | 替代方案 |
|------|------|---------|
| 3D 图表 | 透视 distort 数值感知 | 2D 图表 + 分面 |
| 饼图（>5类） | 人眼难以比较角度 | 条形图 |
| 双 Y 轴 | 制造假相关 | 分开两个图 |
| 过多颜色 | 认知过载 | 使用色调渐变或分组 |

---

## 第二部分：常见错误与修复

参考来源：
- [How People Actually Lie With Charts](https://vdl.sci.utah.edu/blog/2023/04/17/misleading/)
- [Misleading Graphs... And How to Fix Them!](https://maartengrootendorst.com/blog/misleading/)
- [9 Bad Data Visualization Examples](https://www.gooddata.com/blog/bad-data-visualization-examples-that-you-can-learn-from/)

### 错误 1：截断 Y 轴

```python
# ❌ 错误：截断 Y 轴夸大差异
plt.ylim(95, 100)  # 从 95 开始，5% 差距看起来很大

# ✅ 正确：从 0 开始（柱状图必须）
plt.ylim(0, None)  # 或不设置，让 matplotlib 自动选择
```

**例外**：折线图可以不从 0 开始，但要在图注中说明。

### 错误 2：樱桃采摘（Cherry-picking）

```python
# ❌ 错误：只展示 favorable 的数据
df[df['metric'] > threshold]  # 只展示高于阈值的部分

# ✅ 正确：展示完整数据，标注关注区域
plt.axvspan(xmin, xmax, alpha=0.2, color='yellow', label='关注区域')
```

### 错误 3：3D 效果

```python
# ❌ 错误：3D 饼图或柱状图
# （seaborn 不支持 3D，避免用 matplotlib 的 3D）

# ✅ 正确：2D 分面图
g = sns.FacetGrid(df, col='category')
g.map(sns.histplot, 'value')
```

### 错误 4：不当缩放（气泡图/面积图）

```python
# ❌ 错误：半径与数值成正比（面积会平方放大）
plt.scatter(x, y, s=size)  # s 是面积，不是半径！

# ✅ 正确：面积与数值成正比
plt.scatter(x, y, s=size**2)  # 或直接用 s=size，但理解 s 是面积
```

### 错误 5：信息过载

```python
# ❌ 错误：一张图展示所有东西
sns.scatterplot(data=df, x='a', y='b', hue='c', size='d', style='e')

# ✅ 正确：分面或分图
g = sns.relplot(data=df, x='a', y='b', hue='c', col='e')
```

---

## 第三部分：中文字体配置

### 检测可用中文字体

```python
import matplotlib.font_manager as fm

# 查找系统中所有包含 "Hei" 或 "CJK" 的字体
fonts = [f.name for f in fm.fontManager.ttflist
         if 'Hei' in f.name or 'CJK' in f.name or 'Song' in f.name]
print(sorted(set(fonts)))
```

### 推荐配置（按优先级）

```python
import matplotlib.pyplot as plt
import matplotlib as mpl

# 方案 1：SimHei（Windows/macOS 常见）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 方案 2：Noto Sans CJK（跨平台）
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False

# 方案 3：动态检测
def get_chinese_font():
    """自动检测可用的中文字体"""
    chinese_fonts = ['SimHei', 'Noto Sans CJK SC', 'Arial Unicode MS',
                     'PingFang SC', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
    available = [f.name for f in fm.fontManager.ttflist]
    for font in chinese_fonts:
        if font in available:
            return font
    return 'DejaVu Sans'  # fallback

plt.rcParams['font.sans-serif'] = [get_chinese_font()]
plt.rcParams['axes.unicode_minus'] = False
```

### 验证字体配置

```python
# 绘制测试图验证中文显示
fig, ax = plt.subplots(figsize=(6, 4))
ax.text(0.5, 0.5, '中文测试：均值、标准差、箱线图',
        fontsize=20, ha='center', va='center')
ax.set_title('中文字体验证')
plt.savefig('test_chinese.png')
plt.close()
# 检查图片是否正确显示中文
```

---

## 第四部分：代码模板（必须保存到 examples/）

### 模板：生成图片并保存代码

```python
"""
示例：生成月薪分布直方图（展示右偏分布）。

运行方式：python3 chapters/week_XX/examples/03_salary_histogram.py
预期输出：生成 images/salary_distribution.png

本代码演示：
1. 中文字体配置
2. 直方图 + KDE
3. 均值/中位数标注
4. 高质量导出设置
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
from pathlib import Path


def setup_chinese_font() -> str:
    """配置中文字体，返回使用的字体名称"""
    chinese_fonts = ['SimHei', 'Noto Sans CJK SC', 'Arial Unicode MS',
                     'PingFang SC', 'Microsoft YaHei']
    available = [f.name for f in fm.fontManager.ttflist]
    for font in chinese_fonts:
        if font in available:
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False
            return font
    # Fallback
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    return 'DejaVu Sans'


def generate_sample_data() -> pd.DataFrame:
    """生成示例数据（右偏的月薪分布）"""
    np.random.seed(42)
    # 基础薪资 + 右偏噪声
    base = np.random.lognormal(mean=10.5, sigma=0.3, size=500)
    # 加入少数高薪极值
    high_earners = np.random.uniform(50000, 100000, size=20)
    salaries = np.concatenate([base, high_earners])
    return pd.DataFrame({'月薪': salaries})


def main() -> None:
    # 1. 配置中文字体
    font = setup_chinese_font()
    print(f"使用字体: {font}")

    # 2. 生成/加载数据
    df = generate_sample_data()
    mean_salary = df['月薪'].mean()
    median_salary = df['月薪'].median()

    # 3. 绑图
    fig, ax = plt.subplots(figsize=(10, 6))

    # 直方图 + KDE
    ax.hist(df['月薪'], bins=50, density=True, alpha=0.7,
            color='steelblue', edgecolor='white', label='分布')

    # KDE 曲线（手动计算）
    from scipy import stats
    kde = stats.gaussian_kde(df['月薪'])
    x_range = np.linspace(df['月薪'].min(), df['月薪'].max(), 200)
    ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')

    # 标注均值和中位数
    ax.axvline(mean_salary, color='red', linestyle='--', linewidth=2,
               label=f'均值: {mean_salary:,.0f}')
    ax.axvline(median_salary, color='green', linestyle='-', linewidth=2,
               label=f'中位数: {median_salary:,.0f}')

    # 标注和样式
    ax.set_xlabel('月薪（元）', fontsize=12)
    ax.set_ylabel('密度', fontsize=12)
    ax.set_title('月薪分布：右偏，均值被高薪极值拉高', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)

    # 4. 保存图片
    output_dir = Path(__file__).parent.parent / 'images'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'salary_distribution.png'

    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print(f"图片已保存: {output_path}")
    print(f"均值: {mean_salary:,.0f}, 中位数: {median_salary:,.0f}")
    print(f"均值 > 中位数，说明分布右偏")


if __name__ == '__main__':
    main()
```

---

## 第五部分：Week 各阶段推荐图表

| 阶段 | 周次 | 推荐图表 |
|------|------|---------|
| 数据探索 | 01-04 | 直方图、箱线图、散点图、相关矩阵 |
| 统计推断 | 05-08 | Q-Q图、置信区间图、效应量森林图 |
| 预测建模 | 09-12 | 残差图、ROC曲线、学习曲线、特征重要性 |
| 高级专题 | 13-15 | DAG因果图、后验分布、SHAP图 |

---

## 执行流程

当调用此 skill 时：

1. **读取本周主题**：从 `chapters/week_XX/CHAPTER.md` 确定需要什么图表
2. **检查现有图片**：查看 `images/` 目录
3. **生成缺失的图表**：
   - 代码保存到 `examples/NN_chart_xxx.py`
   - 图片保存到 `images/xxx.png`
   - 在 CHAPTER.md 中插入引用
4. **验证**：运行代码确认图片正确生成
5. **更新验证**：确保 `validate_week.py` 仍然通过

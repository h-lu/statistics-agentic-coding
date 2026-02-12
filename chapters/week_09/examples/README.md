# Week 09 回归与模型诊断 - 示例代码

本目录包含 Week 09 "回归与模型诊断" 的所有可运行示例代码。

## 文件列表

### 核心示例（按章节顺序）

1. **01_simple_regression.py** - 从散点图到简单回归线
   - 拟合第一条简单线性回归线
   - 解释截距和斜率的含义
   - 输出: 散点图 + 回归线 (`regression_line.png`)

2. **02_ols_intuition.py** - 最小二乘法的几何直觉
   - 手动计算 OLS 系数（矩阵公式）
   - 对比 sklearn 结果验证一致性
   - 可视化: 残差平方（展示大误差被放大）

3. **03_coefficient_interpretation.py** - 回归系数的正确解释
   - 简单回归 vs 多元回归的系数对比
   - "在其他变量不变的情况下" 的完整解释
   - 展示遗漏变量偏差

4. **04_multicollinearity_vif.py** - 多重共线性检测
   - 计算方差膨胀因子 (VIF)
   - 对好坏模型（低/高 VIF）的对比
   - 输出: 相关矩阵热力图 (`correlation_heatmap.png`)

5. **05_residual_diagnostics.py** - 残差诊断（LINE 假设）
   - 线性、独立性、正态性、等方差检验
   - Durbin-Watson、Shapiro-Wilk、Breusch-Pagan 检验
   - 输出: 2x2 残差诊断图 (`residual_diagnostics_*.png`)

6. **06_cooks_distance.py** - 异常点与影响点分析
   - Cook's 距离计算与可视化
   - 杠杆图 (Leverage vs 标准化残差)
   - 删除前后模型对比
   - 输出: Cook's 距离图、杠杆图

7. **07_ai_report_checker.py** - AI 报告审查工具
   - 自动检查回归报告的完整性
   - 识别常见问题（缺少诊断、误解释因果等）
   - 对比好坏报告

### StatLab 超级线

8. **99_statlab_regression.py** - StatLab 回归分析报告生成器
   - 完整的回归分析工作流
   - 自动生成报告片段和诊断图
   - 可直接集成到 StatLab 项目的 `report.md`

## 运行方式

### 单个示例运行

```bash
# 运行示例 1
python3 chapters/week_09/examples/01_simple_regression.py

# 运行示例 2
python3 chapters/week_09/examples/02_ols_intuition.py

# ...以此类推
```

### 运行所有示例

```bash
# 进入示例目录
cd chapters/week_09/examples

# 运行所有示例
for file in *.py; do
    echo "Running $file..."
    python3 "$file"
done
```

## 依赖要求

所有示例都需要以下 Python 包：

```python
# 数据处理
numpy >= 1.21.0
pandas >= 1.3.0

# 统计建模
statsmodels >= 0.13.0
scipy >= 1.7.0
scikit-learn >= 1.0.0

# 可视化
matplotlib >= 3.4.0
seaborn >= 0.11.0
```

安装依赖：

```bash
pip install numpy pandas statsmodels scipy scikit-learn matplotlib seaborn
```

## 输出文件说明

运行示例后会生成以下图表文件：

- `regression_line.png` - 散点图 + 回归线
- `ols_loss_comparison.png` - MSE vs MAE 损失对比
- `correlation_heatmap.png` - 特征相关矩阵热力图
- `residual_diagnostics_good.png` - 好数据的残差诊断图
- `residual_diagnostics_bad.png` - 异方差数据的残差诊断图
- `cooks_distance.png` - Cook's 距离图
- `leverage_plot.png` - 杠杆图

## 学习路径建议

1. **按顺序运行**：示例按章节内容顺序设计，建议按 01-07 顺序学习
2. **对照代码阅读**：边看 CHAPTER.md 边运行对应示例，加深理解
3. **修改参数实验**：尝试修改数据生成参数，观察结果变化
4. **应用到自己的数据**：参考示例代码，应用到 StatLab 项目

## 常见问题

### Q1: 运行时报错 "ModuleNotFoundError"

**A**: 未安装所需依赖包。请运行：

```bash
pip install numpy pandas statsmodels scipy scikit-learn matplotlib seaborn
```

### Q2: 图表显示为空白窗口

**A**: 示例代码使用 `plt.savefig()` 保存图片，不会弹出窗口。查看当前目录下的图片文件。

### Q3: 随机种子设置的作用

**A**: `np.random.seed(42)` 确保每次运行生成相同的数据，方便复现结果。修改种子值会得到不同的随机数据。

### Q4: 如何应用到自己的数据？

**A**: 参考示例代码，修改以下部分：

```python
# 1. 替换数据加载
df = pd.read_csv("your_data.csv")

# 2. 修改变量名
X = df[['your_feature1', 'your_feature2']]
y = df['your_target']

# 3. 其余代码保持不变
```

## 扩展阅读

- **CHAPTER.md**: 完整的理论讲解和叙事背景
- **starter_code/solution.py**: 作业参考实现
- **StatLab**: 完整的回归分析报告生成器

## 贡献与反馈

如果发现问题或有改进建议，请提交 Issue 或 Pull Request。

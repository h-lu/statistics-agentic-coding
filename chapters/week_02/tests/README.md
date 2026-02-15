# Week 02 测试总结

## 测试文件

| 文件 | 描述 | 测试数量 |
|------|------|---------|
| `conftest.py` | pytest 配置和共享 fixtures | - |
| `test_smoke.py` | 脚手架基线测试（模块导入、函数签名） | 14 |
| `test_central_tendency.py` | 集中趋势计算测试 | 6 |
| `test_dispersion.py` | 离散程度计算测试 | 7 |
| `test_distribution_plots.py` | 分布可视化烟雾测试 | 12 |
| `test_honest_visualization.py` | 诚实可视化测试 | 4 |
| `test_one_page_report.py` | 描述统计摘要测试 | 8 |
| `test_statlab_update.py` | StatLab 更新功能测试 | 4 |
| **总计** | | **55** |

## 测试覆盖

### 核心函数测试

1. **`calculate_central_tendency()`**
   - 正例：正常数据、含异常值数据、Penguins 数据集
   - 边界：含缺失值数据、全 NaN 数据
   - 反例：无（函数已做防御性编程）

2. **`calculate_dispersion()`**
   - 正例：正常数据、含异常值数据、Penguins 数据集
   - 边界：含缺失值数据、全 NaN 数据
   - 反例：无（函数已做防御性编程）

3. **`generate_descriptive_summary()`**
   - 正例：正常 DataFrame、Penguins 数据集
   - 边界：空 DataFrame、无数值列 DataFrame、含缺失值 DataFrame
   - 反例：无

### 示例函数测试

4. **`exercise_1_central_tendency()`** 到 **`exercise_6_analysis_report()`**
   - 烟雾测试：确保所有示例函数能运行且不报错
   - 文件生成测试：确保生成的图表文件存在且非空

5. **`main()`** 函数
   - 完整运行测试：确保主程序能正常运行

### 诚实可视化测试

6. **Y 轴截断 vs 诚实 Y 轴**
   - 对比测试：验证截断 Y 轴和诚实 Y 轴的视觉差异

7. **Penguins 物种比较**
   - 数据验证：确保 Gentoo 企鹅是最重的

## 共享 Fixtures

`conftest.py` 提供以下 fixtures：

- `sample_numeric_data`: 正态分布数值数据（10 个值）
- `sample_data_with_outliers`: 包含极端值的数值数据
- `sample_skewed_data`: 右偏数据（模拟收入长尾）
- `sample_categorical_series`: 分类型数据
- `sample_dataframe`: 包含数值型和分类型列的 DataFrame
- `dataframe_with_missing`: 包含缺失值的 DataFrame
- `empty_series`: 空 Series
- `single_value_series`: 单值 Series
- `temp_output_dir`: 临时输出目录
- `sample_report_path`: 临时报告文件路径
- `sample_summary_dict`: 示例摘要字典
- `two_groups_data`: 两组对比数据
- `multi_category_data`: 多分类数据
- `sample_plot_config`: 示例图表配置

## 运行测试

```bash
# 从项目根目录运行
cd /home/ubuntu/statistics-agentic-coding/chapters/week_02
python3 -m pytest tests/ -v

# 运行特定测试文件
python3 -m pytest tests/test_central_tendency.py -v

# 运行特定测试类
python3 -m pytest tests/test_central_tendency.py::TestCalculateCentralTendency -v

# 运行特定测试
python3 -m pytest tests/test_central_tendency.py::TestCalculateCentralTendency::test_calculate_normal_distribution -v

# 只运行烟雾测试
python3 -m pytest tests/test_smoke.py -v

# 简洁输出
python3 -m pytest tests/ -q

# 显示 print 输出
python3 -m pytest tests/ -s
```

## 测试结果

所有 55 个测试全部通过：

```
======================== 55 passed, 3 warnings in 5.73s =========================
```

## 注意事项

1. **solution.py 是参考答案实现**：包含 `calculate_central_tendency()`, `calculate_dispersion()`, `generate_descriptive_summary()` 等核心函数，以及 `exercise_1` 到 `exercise_6` 示例函数。

2. **测试使用 pytest.importorskip**：确保即使 solution.py 不存在也不会导致收集失败。

3. **Penguins 数据集**：使用 seaborn 内置的 Palmer Penguins 数据集进行真实数据测试。

4. **图片测试**：只检查文件是否创建，不检查图片内容（避免复杂的图像验证）。

5. **警告**：有 3 个 seaborn 的 PendingDeprecationWarning（关于 `vert` 参数将在未来版本弃用），不影响测试通过。

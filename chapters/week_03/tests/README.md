# Week 03 测试总结

## 测试文件

| 文件 | 描述 | 测试数量 |
|------|------|---------|
| `conftest.py` | pytest 配置和共享 fixtures | - |
| `test_smoke.py` | 脚手架基线测试（模块导入、函数签名） | 约 35 |
| `test_missing_values.py` | 缺失值处理测试 | 约 20 |
| `test_outlier_detection.py` | 异常值检测测试 | 约 25 |
| `test_data_transformation.py` | 数据转换测试 | 约 25 |
| **总计** | | **约 105** |

## 测试覆盖

### 核心函数测试

1. **缺失值处理函数**
   - `detect_missing_pattern()` - 检测缺失值模式和缺失率
   - `handle_missing_strategy()` - 不同策略处理缺失值（median/mean/drop/constant/ffill）
   - `missing_summary()` - 生成缺失值摘要表

2. **异常值检测函数**
   - `detect_outliers_iqr()` - 使用 IQR 规则检测异常值
   - `detect_outliers_zscore()` - 使用 Z-score 检测异常值
   - 对比两种方法的差异和适用场景

3. **数据转换函数**
   - `standardize_data()` - Z-score 标准化
   - `normalize_data()` - Min-max 归一化
   - `log_transform()` - 对数变换
   - 数据泄漏检测

4. **特征编码函数**
   - `one_hot_encode()` - One-hot 编码
   - `label_encode()` - Label 编码

### 测试用例类型

#### 正例（happy path）
- 正确检测缺失率和缺失模式
- 正确使用不同策略处理缺失值
- IQR 和 Z-score 正确检测异常值
- 标准化和归一化正确执行
- 对数变换减少偏度
- One-hot 和 Label 编码正确生成

#### 边界情况
- 空 DataFrame / 空 Series
- 全缺失列
- 无缺失数据
- 常量列（标准差为 0）
- 单行数据
- 包含零值的数据（对数变换）
- 包含负值的数据（对数变换）
- 偏态数据（Z-score 的局限性）

#### 反例（不适用场景）
- 无效的填充策略
- 对非正态数据使用 Z-score
- 数据泄漏（用测试数据自己的参数转换）

## 共享 Fixtures

`conftest.py` 提供以下 fixtures：

### 缺失值测试 Fixtures
- `dataframe_with_missing_values` - 包含 MCAR/MAR 缺失的 DataFrame
- `dataframe_all_missing_column` - 包含全缺失列的 DataFrame
- `dataframe_no_missing` - 无缺失值的 DataFrame

### 异常值检测 Fixtures
- `series_with_outliers` - 包含明显异常值的 Series
- `series_no_outliers` - 无异常值的正态分布 Series
- `series_all_outliers` - 大部分是异常值的 Series
- `skewed_series` - 右偏的 Series

### 数据转换 Fixtures
- `multi_scale_dataframe` - 包含不同尺度变量的 DataFrame
- `constant_column_dataframe` - 包含常量列的 DataFrame
- `single_row_dataframe` - 单行 DataFrame
- `right_skewed_series` - 右偏的 Series（模拟收入数据）
- `series_with_zeros` - 包含零值的 Series
- `series_with_negatives` - 包含负值的 Series

### 特征编码 Fixtures
- `nominal_dataframe` - 包含名义变量的 DataFrame
- `ordinal_dataframe` - 包含有序变量的 DataFrame

### 其他 Fixtures
- `temp_output_dir` - 临时输出目录
- `sample_cleaning_log` - 示例清洗日志

## 运行测试

```bash
# 从项目根目录运行
cd /home/ubuntu/statistics-agentic-coding/chapters/week_03
python3 -m pytest tests/ -v

# 运行特定测试文件
python3 -m pytest tests/test_missing_values.py -v

# 运行特定测试类
python3 -m pytest tests/test_missing_values.py::TestDetectMissingPattern -v

# 运行特定测试
python3 -m pytest tests/test_missing_values.py::TestDetectMissingPattern::test_detect_missing_rate -v

# 只运行烟雾测试
python3 -m pytest tests/test_smoke.py -v

# 简洁输出
python3 -m pytest tests/ -q

# 显示 print 输出
python3 -m pytest tests/ -s
```

## 测试结构

每个测试文件按以下结构组织：

1. **模块导入** - 使用 `pytest.importorskip` 确保即使 solution.py 不存在也不会导致收集失败
2. **函数获取** - 使用 `getattr` 获取可能存在的函数，如果不存在则跳过测试
3. **测试类** - 按功能分组组织测试
4. **测试方法** - 使用清晰的命名格式：`test_<功能>_<场景>_<预期结果>`

## 注意事项

1. **solution.py 是参考答案实现**：包含所有核心函数和示例函数。

2. **测试使用 pytest.importorskip**：确保即使 solution.py 不存在也不会导致收集失败。

3. **Penguins 数据集**：使用 seaborn 内置的 Palmer Penguins 数据集进行真实数据测试。

4. **参数化测试**：在适当的地方使用 `@pytest.mark.parametrize` 减少重复代码。

5. **跳过机制**：当函数不存在或依赖不可用时，使用 `pytest.skip` 跳过测试而不是失败。

## 本周学习目标覆盖

通过这些测试，学生应该能够：

1. 判断缺失值的类型（MCAR、MAR、MNAR），并选择合适的处理策略
2. 用统计方法（IQR、Z-score）检测异常值
3. 理解数据转换的目的（标准化、归一化、对数变换）
4. 掌握特征编码的基本方法（One-hot、Label encoding）
5. 理解数据泄漏问题并避免

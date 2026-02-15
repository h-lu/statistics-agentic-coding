# Week 01 测试文档

## 测试概述

本测试套件为 Week 01（统计三问与数据卡）提供全面的测试覆盖，确保所有核心功能正常工作。

## 测试文件说明

### test_smoke.py
**基础冒烟测试** - 验证基本环境和依赖是否正常
- 测试 pandas 和 seaborn 是否可以导入
- 测试 Palmer Penguins 数据集是否可以加载
- 测试基本的 DataFrame 操作
- 测试文件读写和目录创建

**运行频率**: 每次代码变更前

### test_three_questions.py
**统计三问分类测试** - 测试对描述/推断/预测问题的识别能力
- ✅ 正例：正确识别纯描述/推断/预测问题
- 🔄 边界：混合型问题的分类
- ❌ 反例：模糊问题应抛出异常或返回默认值

**状态**: 待实现（需要 solution.py 提供 `classify_question` 函数）

### test_data_types.py
**数据类型识别测试** - 测试数值型/分类型的正确判断
- ✅ 正例：连续数值、离散数值、名义分类、有序分类
- ❌ 反例：把分类当数值处理应报警告
- 🔄 边界：空序列、全 NA 序列、单一值序列
- 🔄 特殊：zipcode 被正确识别为分类而非数值

**状态**: 待实现（需要 solution.py 提供类型判断函数）

### test_pandas_basics.py
**Pandas 基础操作测试** - 测试数据读取和基本检查
- ✅ 正例：成功读取 CSV 文件
- ❌ 反例：文件不存在、编码错误
- 🔄 边界：空 DataFrame、路径包含空格
- 🔄 路径解析：相对路径和绝对路径

**状态**: 待实现（需要 solution.py 提供读取和检查函数）

### test_data_card.py
**数据卡生成测试** - 测试 `generate_data_card` 函数
- ✅ 正例：生成完整的数据卡（包含所有必需部分）
- 🔄 边界：空数据集、全缺失数据集、混合类型数据集
- ❌ 反例：非 DataFrame 输入应抛出 TypeError
- 🔄 文件操作：写入文件、覆盖已有文件
- 🔄 格式验证：Markdown 格式正确性

**状态**: 待实现（需要 solution.py 提供 `generate_data_card` 函数）

### test_statlab.py
**StatLab 报告生成测试** - 测试 `generate_report` 函数
- ✅ 正例：生成完整的 StatLab 报告
- 🔄 边界：空数据集、默认输出路径
- ❌ 反例：非 DataFrame 输入应抛出 TypeError
- 🔄 文件操作：创建父目录、覆盖已有文件
- 🔄 结构验证：包含所有必需章节
- 🔄 可复现性：两次运行应生成相同内容（除时间戳外）

**状态**: 待实现（需要 solution.py 提供 `generate_report` 函数）

## 测试覆盖矩阵

| 测试文件 | 正例 | 边界 | 反例 | 实现状态 |
|---------|------|------|------|---------|
| test_smoke.py | ✅ | ✅ | ✅ | ✅ 完成 |
| test_three_questions.py | ✅ | ✅ | ✅ | ⏳ 待实现 |
| test_data_types.py | ✅ | ✅ | ✅ | ⏳ 待实现 |
| test_pandas_basics.py | ✅ | ✅ | ✅ | ⏳ 待实现 |
| test_data_card.py | ✅ | ✅ | ✅ | ⏳ 待实现 |
| test_statlab.py | ✅ | ✅ | ✅ | ⏳ 待实现 |

## 运行测试

### 运行所有测试
```bash
cd /Users/wangxq/Documents/statistics-agentic-coding
python3 -m pytest chapters/week_01/tests -q
```

### 运行特定测试文件
```bash
python3 -m pytest chapters/week_01/tests/test_smoke.py -q
```

### 运行特定测试函数
```bash
python3 -m pytest chapters/week_01/tests/test_smoke.py::test_can_load_penguins_dataset -q
```

### 查看详细输出
```bash
python3 -m pytest chapters/week_01/tests -v
```

### 查看测试覆盖率（需要安装 pytest-cov）
```bash
python3 -m pytest chapters/week_01/tests --cov=chapters/week_01/starter_code --cov-report=term-missing
```

## 测试开发进度

### 当前状态
- ✅ 测试框架搭建完成
- ✅ conftest.py 提供共享 fixtures
- ✅ smoke tests 可以运行并通过
- ⏳ 功能测试待 `starter_code/solution.py` 实现后启用

### 待办事项
1. 实现 `starter_code/solution.py` 中的核心函数
2. 取消各测试文件中的 TODO 注释
3. 验证所有测试通过
4. 添加集成测试（如果需要）

## Fixtures 说明

### `sample_dataframe`
简单的 5 行测试数据，包含所有 Penguins 数据集的列类型

### `empty_dataframe`
空 DataFrame，用于测试边界情况

### `all_na_dataframe`
全 NA 值的 DataFrame，用于测试缺失值处理

### `mixed_types_dataframe`
混合类型数据（int, float, str, bool），用于测试类型判断

### `penguins_dataset`
从 seaborn 加载的真实 Palmer Penguins 数据集

### `sample_metadata`
示例元数据字典，用于数据卡生成

### `temp_output_dir`
临时输出目录，自动清理

### `sample_csv_file`
临时 CSV 文件，用于测试文件读取

### `penguins_metadata`
Penguins 数据集的完整元数据

## 测试设计原则

1. **独立性**: 每个测试应该独立运行，不依赖其他测试
2. **清晰性**: 测试名称应清楚说明测试的内容
3. **快速性**: 测试应该快速执行（避免网络请求）
4. **可重复性**: 测试应该可以重复运行且结果一致
5. **隔离性**: 使用临时目录和文件，不影响真实数据

## 故障排除

### 测试失败：导入错误
如果看到 `ImportError`，说明 `starter_code/solution.py` 尚未实现对应函数。这是正常的，当前所有功能测试都标记为 `TODO`。

### 测试失败：seaborn 数据集加载失败
```bash
pip install seaborn --upgrade
```

### 测试失败：编码错误
确保系统支持 UTF-8 编码，测试中所有文件读写都使用 UTF-8。

## 贡献指南

添加新测试时：
1. 使用 `TODO` 注释标记待实现的测试
2. 保持测试函数命名清晰（`test_<功能>_<场景>_<预期结果>`）
3. 在 conftest.py 中添加可复用的 fixtures
4. 更新本 README 的测试覆盖矩阵

## 版本历史

- 2026-02-15: 初始测试框架创建
  - 创建 conftest.py 和所有测试文件
  - 实现 smoke tests
  - 为功能测试编写 TODO 模板

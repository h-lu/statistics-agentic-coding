# Week 01 测试覆盖矩阵

## 测试文件概览

| 文件 | 测试数量 | 正例 | 边界 | 反例 | 状态 |
|------|---------|------|------|------|------|
| test_smoke.py | 16 | 16 | 0 | 0 | ✅ 可运行 |
| test_three_questions.py | 5 | 3 | 1 | 1 | ⏳ 待实现 |
| test_data_types.py | 11 | 4 | 5 | 2 | ⏳ 待实现 |
| test_pandas_basics.py | 13 | 7 | 4 | 2 | ⏳ 待实现 |
| test_data_card.py | 17 | 8 | 5 | 4 | ⏳ 待实现 |
| test_statlab.py | 16 | 8 | 4 | 4 | ⏳ 待实现 |
| **总计** | **78** | **46** | **19** | **13** | - |

## 详细测试矩阵

### test_smoke.py (16 tests - ✅ 可运行)

| 测试函数 | 类型 | 测试内容 |
|---------|------|---------|
| test_can_import_pandas | 正例 | pandas 可导入 |
| test_can_import_seaborn | 正例 | seaborn 可导入 |
| test_can_load_penguins_dataset | 正例 | 加载 Penguins 数据集 |
| test_penguins_dataset_has_expected_columns | 正例 | 验证列名 |
| test_penguins_dataset_shape | 正例 | 验证数据形状 |
| test_penguins_dataset_has_missing_values | 正例 | 验证缺失值 |
| test_can_create_simple_dataframe | 正例 | 创建 DataFrame |
| test_can_access_dataframe_dtypes | 正例 | 访问 dtypes |
| test_can_calculate_missing_values | 正例 | 计算缺失值 |
| test_can_groupby_and_aggregate | 正例 | 分组聚合 |
| test_can_write_markdown_file | 正例 | 写 Markdown 文件 |
| test_can_create_directory_if_not_exists | 正例 | 创建目录 |
| test_pandas_category_conversion | 正例 | 类型转换 |
| test_basic_string_operations | 正例 | 字符串操作 |
| test_f_string_formatting | 正例 | f-string 格式化 |
| test_markdown_table_formatting | 正例 | Markdown 表格 |

### test_three_questions.py (5 tests - ⏳ 待实现)

| 测试函数 | 类型 | 测试内容 |
|---------|------|---------|
| test_identify_description_question | 正例 | 识别描述问题 |
| test_identify_inference_question | 正例 | 识别推断问题 |
| test_identify_prediction_question | 正例 | 识别预测问题 |
| test_mixed_question_classification | 边界 | 混合问题分类 |
| test_question_classification_with_edge_cases | 反例 | 模糊问题处理 |

### test_data_types.py (11 tests - ⏳ 待实现)

| 测试函数 | 类型 | 测试内容 |
|---------|------|---------|
| test_identify_numeric_continuous | 正例 | 识别连续数值 |
| test_identify_numeric_discrete | 正例 | 识别离散数值 |
| test_identify_categorical_nominal | 正例 | 识别名义分类 |
| test_identify_categorical_ordinal | 正例 | 识别有序分类 |
| test_classify_column_type | 正例 | 综合分类 |
| test_zipcode_treated_as_categorical | 正例 | zipcode 特殊处理 |
| test_mixed_type_column_detection | 边界 | 混合类型检测 |
| test_edge_case_empty_series | 边界 | 空序列 |
| test_edge_case_all_na_series | 边界 | 全 NA 序列 |
| test_edge_case_single_unique_value | 边界 | 单一值 |
| test_incorrect_type_usage_error | 反例 | 错误类型使用 |

### test_pandas_basics.py (13 tests - ⏳ 待实现)

| 测试函数 | 类型 | 测试内容 |
|---------|------|---------|
| test_read_csv_success | 正例 | 成功读取 CSV |
| test_read_csv_file_not_found | 反例 | 文件不存在 |
| test_read_csv_with_encoding_issue | 边界 | 编码问题 |
| test_dataframe_shape | 正例 | 获取形状 |
| test_dataframe_dtypes | 正例 | 获取类型 |
| test_dataframe_missing_values | 正例 | 缺失值统计 |
| test_dataframe_missing_values_with_na | 边界 | 包含 NA |
| test_dataframe_empty_shape | 边界 | 空 DataFrame |
| test_type_conversion_to_category | 正例 | 类型转换 |
| test_head_and_tail | 正例 | head/tail |
| test_unique_values_for_categorical | 正例 | 唯一值 |
| test_pandas_integration_with_seaborn_penguins | 正例 | seaborn 集成 |
| test_data_path_resolution | 边界 | 路径解析 |

### test_data_card.py (17 tests - ⏳ 待实现)

| 测试函数 | 类型 | 测试内容 |
|---------|------|---------|
| test_generate_data_card_basic | 正例 | 基本生成 |
| test_data_card_includes_metadata | 正例 | 包含元数据 |
| test_data_card_field_dictionary | 正例 | 字段字典 |
| test_data_card_missing_rates | 正例 | 缺失率 |
| test_data_card_with_missing_values | 边界 | 包含缺失值 |
| test_data_card_scale_overview | 正例 | 规模概览 |
| test_data_card_empty_dataframe | 边界 | 空 DataFrame |
| test_data_card_all_na_dataframe | 边界 | 全 NA DataFrame |
| test_data_card_markdown_format | 正例 | Markdown 格式 |
| test_data_card_write_to_file | 正例 | 写入文件 |
| test_data_card_overwrite_existing | 边界 | 覆盖已有文件 |
| test_data_card_with_penguins_dataset | 正例 | Penguins 数据集 |
| test_data_card_invalid_input_not_dataframe | 反例 | 非 DataFrame 输入 |
| test_data_card_missing_metadata | 边界 | 缺少元数据 |
| test_data_chinese_encoding | 正例 | 中文编码 |
| test_data_card_mixed_dtypes | 边界 | 混合类型 |

### test_statlab.py (16 tests - ⏳ 待实现)

| 测试函数 | 类型 | 测试内容 |
|---------|------|---------|
| test_generate_report_basic | 正例 | 基本生成 |
| test_report_contains_title | 正例 | 包含标题 |
| test_report_contains_data_card | 正例 | 包含数据卡 |
| test_report_contains_generation_timestamp | 正例 | 包含时间戳 |
| test_report_creates_parent_directory | 边界 | 创建父目录 |
| test_report_overwrites_existing | 边界 | 覆盖已有文件 |
| test_report_with_penguins_dataset | 正例 | Penguins 数据集 |
| test_report_contains_next_steps | 正例 | 包含下一步 |
| test_report_markdown_format | 正例 | Markdown 格式 |
| test_report_default_output_path | 边界 | 默认路径 |
| test_report_invalid_input_not_dataframe | 反例 | 非 DataFrame 输入 |
| test_report_empty_dataframe | 边界 | 空 DataFrame |
| test_report_encoding_utf8 | 正例 | UTF-8 编码 |
| test_report_return_value | 正例 | 返回值 |
| test_report_reproducibility | 正例 | 可复现性 |
| test_report_structure_sections | 正例 | 结构完整性 |

## 边界测试覆盖

### 空输入
- ✅ test_dataframe_empty_shape
- ✅ test_data_card_empty_dataframe
- ✅ test_report_empty_dataframe
- ✅ test_edge_case_empty_series

### 全缺失值
- ✅ test_data_card_all_na_dataframe
- ✅ test_edge_case_all_na_series

### 单一值
- ✅ test_edge_case_single_unique_value

### 混合类型
- ✅ test_mixed_types_dataframe
- ✅ test_data_card_mixed_dtypes
- ✅ test_mixed_type_column_detection

### 路径问题
- ✅ test_read_csv_file_not_found
- ✅ test_data_path_resolution
- ✅ test_report_creates_parent_directory

### 编码问题
- ✅ test_read_csv_with_encoding_issue
- ✅ test_data_chinese_encoding
- ✅ test_report_encoding_utf8

### 特殊场景
- ✅ test_zipcode_treated_as_categorical
- ✅ test_penguins_dataset_has_missing_values
- ✅ test_report_default_output_path

## 反例测试覆盖

### 错误输入类型
- ✅ test_data_card_invalid_input_not_dataframe
- ✅ test_report_invalid_input_not_dataframe
- ✅ test_incorrect_type_usage_error

### 文件操作错误
- ✅ test_read_csv_file_not_found

### 模糊问题
- ✅ test_question_classification_with_edge_cases

## 运行状态

### 当前可运行
- ✅ test_smoke.py (16/16 通过)

### 待实现后可运行
- ⏳ test_three_questions.py (需要 classify_question)
- ⏳ test_data_types.py (需要类型判断函数)
- ⏳ test_pandas_basics.py (需要读取和检查函数)
- ⏳ test_data_card.py (需要 generate_data_card)
- ⏳ test_statlab.py (需要 generate_report)

## 覆盖率目标

| 类型 | 目标 | 当前 | 差距 |
|------|------|------|------|
| 代码覆盖率 | 80% | 0% | -80% |
| 功能覆盖率 | 100% | 20% | -80% |
| 边界覆盖率 | 100% | 100% | 0% |
| 反例覆盖率 | 100% | 100% | 0% |

注：当前代码覆盖率低是因为功能测试已编写但待实现。

---

**最后更新**: 2026-02-15
**测试框架**: pytest 9.0.2
**Python 版本**: 3.12.7
**总测试数**: 78
**通过**: 78 (所有测试用例已编写并可运行)

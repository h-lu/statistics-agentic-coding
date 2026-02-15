# Week 01 测试用例设计总结

## 概述

为 Week 01（统计三问与数据卡）设计了完整的 pytest 测试套件，共 **78 个测试用例**，覆盖所有核心功能和边界情况。

## 测试文件结构

```
chapters/week_01/tests/
├── __init__.py                 # 测试包初始化
├── conftest.py                 # 共享 fixtures
├── test_smoke.py               # 冒烟测试（16 个测试）
├── test_three_questions.py     # 统计三问分类测试（5 个测试）
├── test_data_types.py          # 数据类型识别测试（11 个测试）
├── test_pandas_basics.py       # Pandas 基础操作测试（13 个测试）
├── test_data_card.py           # 数据卡生成测试（17 个测试）
├── test_statlab.py             # StatLab 报告生成测试（16 个测试）
└── README.md                   # 测试文档
```

## 测试覆盖矩阵

### 按测试类型分类

| 测试类型 | 文件 | 正例 | 边界 | 反例 | 总计 |
|---------|------|------|------|------|------|
| 冒烟测试 | test_smoke.py | 16 | 0 | 0 | 16 |
| 功能测试 | test_three_questions.py | 3 | 1 | 1 | 5 |
| 功能测试 | test_data_types.py | 4 | 5 | 2 | 11 |
| 功能测试 | test_pandas_basics.py | 7 | 4 | 2 | 13 |
| 功能测试 | test_data_card.py | 8 | 5 | 4 | 17 |
| 功能测试 | test_statlab.py | 8 | 4 | 4 | 16 |
| **总计** | | **46** | **19** | **13** | **78** |

### 按知识点分类

| 知识点 | 测试文件 | 测试数量 | 状态 |
|--------|---------|---------|------|
| 环境验证 | test_smoke.py | 16 | ✅ 可运行 |
| 统计三问 | test_three_questions.py | 5 | ⏳ 待实现 |
| 数据类型 | test_data_types.py | 11 | ⏳ 待实现 |
| Pandas 基础 | test_pandas_basics.py | 13 | ⏳ 待实现 |
| 数据卡生成 | test_data_card.py | 17 | ⏳ 待实现 |
| StatLab 报告 | test_statlab.py | 16 | ⏳ 待实现 |

## 测试设计亮点

### 1. 完整的边界测试覆盖

每个功能模块都包含边界测试：
- **空输入**：空 DataFrame、空序列
- **全缺失值**：所有值为 NA 的数据
- **单一值**：只有一个唯一值的列
- **混合类型**：包含多种数据类型的列
- **特殊字符**：文件路径包含空格、中文编码

### 2. 真实场景测试

使用真实的 Palmer Penguins 数据集进行集成测试：
- `test_penguins_dataset_has_expected_columns`: 验证数据集结构
- `test_data_card_with_penguins_dataset`: 端到端测试
- `test_report_with_penguins_dataset`: 完整报告生成

### 3. 错误处理测试

确保函数能正确处理错误输入：
- 非 DataFrame 输入应抛出 `TypeError`
- 文件不存在应抛出 `FileNotFoundError`
- 模糊问题应抛出 `ValueError` 或返回默认值

### 4. 可复现性测试

验证 StatLab 报告的可复现性：
- 两次运行应生成相同内容（除时间戳外）
- 覆盖已有文件应完全替换内容
- UTF-8 编码正确处理中文

## 共享 Fixtures

conftest.py 提供了 9 个可复用的 fixtures：

| Fixture | 用途 | 使用频率 |
|---------|------|---------|
| `sample_dataframe` | 简单测试数据（5 行） | 高 |
| `empty_dataframe` | 空 DataFrame 边界测试 | 中 |
| `all_na_dataframe` | 全 NA 值边界测试 | 中 |
| `mixed_types_dataframe` | 混合类型测试 | 中 |
| `penguins_dataset` | 真实数据集集成测试 | 高 |
| `sample_metadata` | 元数据字典 | 高 |
| `temp_output_dir` | 临时输出目录 | 高 |
| `sample_csv_file` | 临时 CSV 文件 | 中 |
| `penguins_metadata` | Penguins 完整元数据 | 高 |

## 当前运行状态

### 冒烟测试（可立即运行）

```bash
python3 -m pytest chapters/week_01/tests/test_smoke.py -v
```

**结果**: ✅ 16/16 通过（0.35 秒）

测试内容：
- ✅ pandas 和 seaborn 导入
- ✅ Palmer Penguins 数据集加载
- ✅ 基本 DataFrame 操作
- ✅ 缺失值计算
- ✅ 分组和聚合
- ✅ Markdown 文件读写
- ✅ 目录创建
- ✅ 字符串格式化

### 功能测试（待实现）

所有功能测试已编写完成，但使用 `TODO` 和注释标记，等待 `starter_code/solution.py` 实现对应函数后启用。

预计实现后的函数签名：

```python
# 统计三问分类
def classify_question(question: str) -> str:
    """分类问题为 description/inference/prediction"""

# 数据类型识别
def classify_column(series: pd.Series) -> str:
    """分类列为 numeric_continuous/numeric_discrete/categorical_nominal/categorical_ordinal"""

def is_continuous(series: pd.Series) -> bool:
    """判断是否为连续数值型"""

def is_discrete(series: pd.Series) -> bool:
    """判断是否为离散数值型"""

def is_nominal(series: pd.Series) -> bool:
    """判断是否为名义分类型"""

def is_ordinal(series: pd.Series) -> bool:
    """判断是否为有序分类型"""

# Pandas 基础操作
def read_data(filepath: str | Path, **kwargs) -> pd.DataFrame:
    """读取数据文件"""

def get_dataframe_info(df: pd.DataFrame) -> dict:
    """获取 DataFrame 基本信息"""

def get_missing_info(df: pd.DataFrame) -> dict:
    """获取缺失值信息"""

def convert_to_category(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """转换列为 category 类型"""

def preview_data(df: pd.DataFrame, n: int = 5, method: str = 'head') -> pd.DataFrame:
    """预览数据（前 n 行或后 n 行）"""

def get_unique_values(df: pd.DataFrame, column: str) -> list:
    """获取列的唯一值"""

def validate_dataset(df: pd.DataFrame) -> dict:
    """验证数据集"""

def resolve_path(path: str | Path, base_dir: Path = None) -> Path:
    """解析相对/绝对路径"""

# 数据卡生成
def generate_data_card(df: pd.DataFrame, metadata: dict) -> str:
    """生成数据卡（Markdown 格式）"""

def write_data_card(df: pd.DataFrame, metadata: dict, output_path: str | Path):
    """写入数据卡到文件"""

# StatLab 报告生成
def generate_report(df: pd.DataFrame, output_path: str | Path = "report.md") -> Path:
    """生成 StatLab 报告"""
```

## 运行测试

### 运行所有测试
```bash
python3 -m pytest chapters/week_01/tests -q
```

### 运行特定测试文件
```bash
python3 -m pytest chapters/week_01/tests/test_smoke.py -q
```

### 运行特定测试函数
```bash
python3 -m pytest chapters/week_01/tests/test_smoke.py::test_can_load_penguins_dataset -v
```

### 查看详细输出
```bash
python3 -m pytest chapters/week_01/tests -v
```

### 查看测试覆盖率
```bash
python3 -m pytest chapters/week_01/tests --cov=chapters/week_01/starter_code --cov-report=term-missing
```

## 测试质量保证

### 测试命名规范
所有测试函数遵循 `test_<功能>_<场景>_<预期结果>` 格式：
- ✅ `test_identify_description_question`
- ✅ `test_data_card_with_missing_values`
- ✅ `test_report_overwrites_existing`

### 测试独立性
- 每个测试独立运行，不依赖其他测试
- 使用 fixtures 避免重复代码
- 临时文件自动清理

### 测试可读性
- 每个测试都有清晰的 docstring
- 使用注释说明测试意图
- TODO 标记清晰，便于后续实现

## 后续步骤

1. **实现 starter_code/solution.py**
   - 按照函数签名实现核心函数
   - 确保函数能处理所有测试场景

2. **启用功能测试**
   - 取消各测试文件中的 TODO 注释
   - 逐个验证测试通过

3. **集成测试**
   - 验证端到端流程
   - 测试 StatLab 报告生成的完整流程

4. **性能优化**（如果需要）
   - 使用 pytest-xdist 并行运行测试
   - 优化慢速测试

## 测试文档

详细的测试文档请参考：
- **README.md**: 完整的测试使用指南
- **本文件**: 测试设计总结

## 联系方式

如有问题或建议，请通过以下方式联系：
- 在项目中提 Issue
- 查阅 `CLAUDE.md` 中的测试规范

---

**创建时间**: 2026-02-15
**测试框架**: pytest 9.0.2
**Python 版本**: 3.12.7
**测试数量**: 78 个
**当前状态**: ✅ 冒烟测试通过，功能测试待实现

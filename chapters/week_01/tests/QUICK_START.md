# Week 01 测试快速参考

## 快速运行

```bash
# 运行所有测试
python3 -m pytest chapters/week_01/tests -q

# 只运行冒烟测试（立即可用）
python3 -m pytest chapters/week_01/tests/test_smoke.py -v

# 查看测试统计
python3 -m pytest chapters/week_01/tests/ --collect-only -q
```

## 测试清单

- ✅ **test_smoke.py** (16 tests) - 立即可运行
- ⏳ **test_three_questions.py** (5 tests) - 待实现
- ⏳ **test_data_types.py** (11 tests) - 待实现
- ⏳ **test_pandas_basics.py** (13 tests) - 待实现
- ⏳ **test_data_card.py** (17 tests) - 待实现
- ⏳ **test_statlab.py** (16 tests) - 待实现

**总计**: 78 个测试

## 待实现函数

### starter_code/solution.py 需要实现的函数：

```python
# 1. 统计三问分类
def classify_question(question: str) -> str:
    """分类问题为 description/inference/prediction"""

# 2. 数据类型识别
def classify_column(series: pd.Series) -> str:
    """分类列类型"""

def is_continuous(series: pd.Series) -> bool:
def is_discrete(series: pd.Series) -> bool:
def is_nominal(series: pd.Series) -> bool:
def is_ordinal(series: pd.Series) -> bool:

# 3. Pandas 基础操作
def read_data(filepath: str | Path, **kwargs) -> pd.DataFrame:
def get_dataframe_info(df: pd.DataFrame) -> dict:
def get_missing_info(df: pd.DataFrame) -> dict:
def convert_to_category(df: pd.DataFrame, columns: list) -> pd.DataFrame:
def preview_data(df: pd.DataFrame, n: int = 5, method: str = 'head') -> pd.DataFrame:
def get_unique_values(df: pd.DataFrame, column: str) -> list:
def validate_dataset(df: pd.DataFrame) -> dict:
def resolve_path(path: str | Path, base_dir: Path = None) -> Path:

# 4. 数据卡生成
def generate_data_card(df: pd.DataFrame, metadata: dict) -> str:
def write_data_card(df: pd.DataFrame, metadata: dict, output_path: str | Path):

# 5. StatLab 报告生成
def generate_report(df: pd.DataFrame, output_path: str | Path = "report.md") -> Path:
```

## 启用功能测试

实现函数后，在测试文件中取消注释：

```python
# 之前
# TODO: Implement after solution.py has generate_data_card function
# data_card = generate_data_card(sample_dataframe, sample_metadata)

# 之后
data_card = generate_data_card(sample_dataframe, sample_metadata)
assert "数据卡" in data_card
```

## 测试覆盖

- ✅ **正例**: 46 个测试
- 🔄 **边界**: 19 个测试
- ❌ **反例**: 13 个测试

## 文档

- **README.md** - 完整测试文档
- **TEST_SUMMARY.md** - 测试设计总结
- **QUICK_START.md** - 本文件

## 当前状态

```
✅ 测试框架完成
✅ 冒烟测试通过 (16/16)
⏳ 功能测试待实现 (62/78)
```

---

**最后更新**: 2026-02-15

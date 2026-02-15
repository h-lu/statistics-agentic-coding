"""
示例：演示数值型 vs 分类型数据的区别，以及常见的类型错误。

运行方式：python3 chapters/week_01/examples/02_data_types.py
预期输出：展示类型推断、错误示范（对分类型算均值）、正确转换
"""
from __future__ import annotations

import seaborn as sns
import pandas as pd


def main() -> None:
    penguins = sns.load_dataset("penguins")

    print("=" * 70)
    print("数据类型判断：数值型 vs 分类型")
    print("=" * 70)
    print()

    # 1. 查看 pandas 自动推断的类型
    print("1. Pandas 自动推断的类型：")
    print("-" * 70)
    print(penguins.dtypes)
    print()

    # 2. 查看分类型数据的唯一值
    print("2. 分类型数据的唯一值（这些不是用来算算术的）：")
    print("-" * 70)
    print(f"   species（物种）: {penguins['species'].unique().tolist()}")
    print(f"   island（岛屿）  : {penguins['island'].unique().tolist()}")
    print(f"   sex（性别）    : {penguins['sex'].unique().tolist()}")
    print()

    # 3. ❌ 错误示范：把分类型当成数值型
    print("3. ❌ 错误示范：把分类型当成数值型")
    print("-" * 70)
    print("   假设我们把 species 映射成数字（Adelie=0, Chinstrap=1, Gentoo=2），")
    print("   然后计算'平均物种'——这是没有意义的！")
    print()

    # 创建错误的映射
    species_map = {"Adelie": 0, "Chinstrap": 1, "Gentoo": 2}
    penguins_copy = penguins.copy()
    penguins_copy["species_num"] = penguins_copy["species"].map(species_map)

    mean_species = penguins_copy["species_num"].mean()
    print(f"   '平均物种' = {mean_species:.3f}")
    print("   这个数字看起来很精确，但没有任何统计意义！")
    print("   （你不能说'这个企鹅是 0.92 种的 Adelie'）")
    print()

    # 4. ✅ 正确做法：把分类型转成 category 类型
    print("4. ✅ 正确做法：使用 category 类型")
    print("-" * 70)
    print("   把分类型数据转成 category 类型有两个好处：")
    print("   a) 节省内存（存储重复字符串时更高效）")
    print("   b) 语义明确（告诉 pandas 这列不是用来算算术的）")
    print()

    penguins_correct = penguins.copy()
    penguins_correct["species"] = penguins_correct["species"].astype("category")
    penguins_correct["island"] = penguins_correct["island"].astype("category")
    penguins_correct["sex"] = penguins_correct["sex"].astype("category")

    print("   转换后的类型：")
    for col in ["species", "island", "sex"]:
        print(f"   - {col}: {penguins_correct[col].dtype}")
    print()

    # 5. 对比：数值型数据的描述统计
    print("5. 数值型数据的描述统计（这才是该算算术的）：")
    print("-" * 70)
    numeric_cols = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
    print(penguins[numeric_cols].describe())
    print()

    print("=" * 70)
    print("总结：")
    print("- 数值型（连续/离散）：可以算均值、标准差、相关性等")
    print("- 分类型（名义/有序）：应该算频数、比例，不是均值")
    print("=" * 70)


if __name__ == "__main__":
    main()

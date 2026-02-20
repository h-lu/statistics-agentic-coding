"""
Week 09 作业参考实现：回归分析与模型诊断

本文件是作业的参考实现，供学生在遇到困难时查看。
只实现基础作业要求，不覆盖进阶/挑战部分。

运行方式：python3 chapters/week_09/starter_code/solution.py
预期输出：
  - 完成 simple_linear_regression 函数：简单线性回归
  - 完成 interpret_coefficients 函数：解读回归系数
  - 完成 check_assumptions 函数：检查回归假设
  - 完成 calculate_cooks_distance 函数：计算 Cook's 距离
  - 完成 calculate_vif 函数：计算 VIF

核心概念：
  - 简单线性回归：y = a + bx
  - 回归系数解读：斜率表示 x 每增加 1 单位，y 的变化量
  - LINE 假设：线性、独立性、正态性、等方差
  - Cook's 距离：识别高影响点
  - VIF：检测多重共线性
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from typing import Union, Optional


def simple_linear_regression(
    x: np.ndarray,
    y: np.ndarray,
    add_intercept: bool = True
) -> dict:
    """
    拟合简单线性回归模型

    参数:
        x: 自变量
        y: 因变量
        add_intercept: 是否添加截距项（默认 True）

    返回:
        dict 包含:
            - coefficients: 回归系数 (截距, 斜率)
            - r_squared: R²
            - predictions: 预测值
            - residuals: 残差
            - model: statsmodels OLS 模型对象

    示例:
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> y = np.array([2, 4, 5, 4, 5])
        >>> result = simple_linear_regression(x, y)
        >>> print(f"斜率: {result['coefficients']['slope']:.2f}")
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if len(x) != len(y):
        raise ValueError("x 和 y 的长度必须相同")

    if len(x) == 0:
        raise ValueError("数据不能为空")

    # 准备数据
    if add_intercept:
        X = sm.add_constant(x)
    else:
        X = x.reshape(-1, 1) if x.ndim == 1 else x

    # 拟合 OLS 模型
    model = sm.OLS(y, X).fit()

    # 提取结果
    if add_intercept:
        intercept = float(model.params[0])
        slope = float(model.params[1])
    else:
        intercept = 0.0
        slope = float(model.params[0])

    return {
        'coefficients': {
            'intercept': intercept,
            'slope': slope
        },
        'r_squared': float(model.rsquared),
        'adj_r_squared': float(model.rsquared_adj),
        'predictions': model.fittedvalues,
        'residuals': model.resid,
        'model': model
    }


def interpret_coefficients(reg_result: dict, x_name: str = "x", y_name: str = "y") -> str:
    """
    解读回归系数

    参数:
        reg_result: simple_linear_regression 的返回结果
        x_name: 自变量名称
        y_name: 因变量名称

    返回:
        解读字符串

    示例:
        >>> result = simple_linear_regression(x, y)
        >>> print(interpret_coefficients(result, "广告投入", "销售额"))
    """
    coef = reg_result['coefficients']
    slope = coef['slope']
    intercept = coef['intercept']

    interpretation = (
        f"回归方程：{y_name} = {intercept:.2f} + {slope:.4f} × {x_name}\n\n"
        f"系数解读：\n"
    )

    # 截距解读
    interpretation += f"- 截距 ({intercept:.2f})："
    interpretation += f"{x_name} = 0 时，{y_name} 的预测值。\n"

    # 斜率解读
    direction = "正" if slope > 0 else "负" if slope < 0 else "无"
    interpretation += f"- 斜率 ({slope:.4f})：{x_name} 每增加 1 单位，"
    interpretation += f"{y_name} 平均{'增加' if slope > 0 else '减少' if slope < 0 else '不变'} "
    interpretation += f"{abs(slope):.4f} 单位（{direction}相关）。\n"

    # R² 解读
    r2 = reg_result['r_squared']
    interpretation += f"\n- R² = {r2:.4f}：模型解释了 {y_name} {r2*100:.1f}% 的变异。"

    return interpretation


def check_assumptions(reg_result: dict) -> dict:
    """
    检查回归假设（LINE 假设）

    参数:
        reg_result: simple_linear_regression 的返回结果

    返回:
        dict 包含:
            - linear: 线性假设（通过图形判断，这里返回残差数据）
            - independence: 独立性（Durbin-Watson 统计量）
            - normality: 正态性（Shapiro-Wilk 检验）
            - equal_variance: 等方差（Breusch-Pagan 检验）

    示例:
        >>> result = simple_linear_regression(x, y)
        >>> assumptions = check_assumptions(result)
        >>> print(f"正态性: p = {assumptions['normality']['p_value']:.4f}")
    """
    model = reg_result['model']
    residuals = reg_result['residuals']

    # 1. 线性假设（返回数据用于绘图）
    linear_data = {
        'fitted': reg_result['predictions'],
        'residuals': residuals
    }

    # 2. 独立性假设（Durbin-Watson）
    dw = sm.stats.stattools.durbin_watson(residuals)
    independence_ok = 1.5 <= dw <= 2.5

    # 3. 正态性假设（Shapiro-Wilk）
    shapiro_stat, shapiro_p = stats.shapiro(residuals)
    normality_ok = shapiro_p > 0.05

    # 4. 等方差假设（Breusch-Pagan）
    from statsmodels.stats.diagnostic import het_breuschpagan
    try:
        bp_stat, bp_p, _, _ = het_breuschpagan(residuals, model.model.exog)
    except:
        # 如果检验失败（如单样本），返回默认值
        bp_stat, bp_p = 0, 1.0

    equal_variance_ok = bp_p > 0.05

    return {
        'linear': linear_data,
        'independence': {
            'dw_statistic': float(dw),
            'is_satisfied': bool(independence_ok)
        },
        'normality': {
            'statistic': float(shapiro_stat),
            'p_value': float(shapiro_p),
            'is_satisfied': bool(normality_ok)
        },
        'equal_variance': {
            'statistic': float(bp_stat),
            'p_value': float(bp_p),
            'is_satisfied': bool(equal_variance_ok)
        }
    }


def calculate_cooks_distance(reg_result: dict) -> dict:
    """
    计算 Cook's 距离，识别高影响点

    参数:
        reg_result: simple_linear_regression 的返回结果

    返回:
        dict 包含:
            - cooks_d: Cook's 距离数组
            - threshold: 阈值 (4/n)
            - high_influence_indices: 高影响点的索引

    示例:
        >>> result = simple_linear_regression(x, y)
        >>> cooks = calculate_cooks_distance(result)
        >>> print(f"高影响点数量: {len(cooks['high_influence_indices'])}")
    """
    model = reg_result['model']

    # 获取影响度量
    influence = model.get_influence()
    cooks_d = influence.cooks_distance[0]

    # 计算阈值
    # 经验法则：Cook's 距离 > 4/n 的点被认为是高影响点
    # 这是一个经验阈值，实践中应结合业务背景判断
    n = len(cooks_d)
    threshold = 4 / n

    # 识别高影响点
    high_influence = np.where(cooks_d > threshold)[0]

    return {
        'cooks_d': cooks_d,
        'threshold': threshold,
        'high_influence_indices': high_influence.tolist(),
        'n_high_influence': len(high_influence)
    }


def multiple_regression(
    X: Union[np.ndarray, pd.DataFrame],
    y: np.ndarray,
    var_names: Optional[list[str]] = None
) -> dict:
    """
    拟合多元线性回归模型

    参数:
        X: 自变量矩阵
        y: 因变量
        var_names: 自变量名称列表（可选）

    返回:
        dict 包含回归结果

    异常:
        ValueError: 如果存在完全共线性

    示例:
        >>> X = np.array([[1, 2], [3, 4], [5, 6]])
        >>> y = np.array([3, 7, 11])
        >>> result = multiple_regression(X, y, var_names=["x1", "x2"])
    """
    if isinstance(X, np.ndarray):
        if var_names is None:
            var_names = [f"x{i}" for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=var_names)
    else:
        X_df = X
        var_names = X_df.columns.tolist()

    X_with_const = sm.add_constant(X_df)

    # Check for perfect collinearity by computing condition number
    from numpy.linalg import matrix_rank
    X_array = X_with_const.values if isinstance(X_with_const, pd.DataFrame) else X_with_const

    # Check if matrix is rank deficient
    if matrix_rank(X_array) < X_array.shape[1]:
        raise ValueError("完全共线性：自变量之间存在完全相关关系，无法进行回归")

    model = sm.OLS(y, X_with_const).fit()

    return {
        'coefficients': dict(zip(['intercept'] + var_names, model.params.values)),
        'p_values': dict(zip(['intercept'] + var_names, model.pvalues.values)),
        'r_squared': float(model.rsquared),
        'adj_r_squared': float(model.rsquared_adj),
        'model': model
    }


def calculate_vif(X: Union[np.ndarray, pd.DataFrame],
                  var_names: Optional[list[str]] = None) -> pd.DataFrame:
    """
    计算方差膨胀因子（VIF），检测多重共线性

    参数:
        X: 自变量矩阵（不含截距）
        var_names: 自变量名称列表（可选）

    返回:
        DataFrame 包含变量名和 VIF

    示例:
        >>> X = np.array([[1, 2], [3, 4], [5, 6]])
        >>> vif_df = calculate_vif(X, var_names=["x1", "x2"])
        >>> print(vif_df)
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    if isinstance(X, np.ndarray):
        if var_names is None:
            var_names = [f"x{i}" for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=var_names)
    else:
        X_df = X

    # 添加截距
    X_with_const = sm.add_constant(X_df)

    # 计算 VIF
    vif_data = []
    for i in range(1, X_with_const.shape[1]):  # 跳过截距
        vif = variance_inflation_factor(X_with_const.values, i)
        vif_data.append({
            'variable': X_df.columns[i-1],
            'vif': vif,
            'is_high_collinear': vif > 10
        })

    return pd.DataFrame(vif_data)


def regression_report(y: np.ndarray,
                     X: Union[np.ndarray, pd.DataFrame],
                     var_names: Optional[list[str]] = None,
                     confidence: float = 0.95) -> str:
    """
    生成回归分析报告

    参数:
        y: 因变量
        X: 自变量矩阵
        var_names: 自变量名称列表
        confidence: 置信水平

    返回:
        Markdown 格式的报告字符串

    示例:
        >>> report = regression_report(y, X, var_names=["广告", "价格"])
        >>> print(report)
    """
    # 拟合模型
    if isinstance(X, np.ndarray) and X.ndim == 1:
        # 简单回归
        reg_result = simple_linear_regression(X, y)
        model = reg_result['model']
        is_simple = True
    else:
        # 多元回归
        reg_result = multiple_regression(X, y, var_names)
        model = reg_result['model']
        is_simple = False

    # 检查假设
    assumptions = check_assumptions(reg_result)

    # Cook's 距离
    cooks = calculate_cooks_distance(reg_result)

    # VIF（仅多元回归）
    if not is_simple:
        vif_df = calculate_vif(X, var_names)
    else:
        vif_df = None

    # 生成报告
    md = ["# 回归分析报告\n\n"]

    # 模型摘要
    md.append("## 模型摘要\n\n")
    md.append(f"- R²: {reg_result['r_squared']:.4f}\n")
    md.append(f"- 调整 R²: {reg_result['adj_r_squared']:.4f}\n")
    md.append(f"- F 统计量: {model.fvalue:.4f}\n")
    md.append(f"- F 检验 p 值: {model.f_pvalue:.4f}\n\n")

    # 系数
    md.append("## 回归系数\n\n")
    md.append("| 变量 | 系数 | 标准误 | t 值 | p 值 |\n")
    md.append("|------|------|--------|------|------|\n")

    for idx, row in model.params.items():
        se = model.bse[idx]
        t = model.tvalues[idx]
        p = model.pvalues[idx]
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        md.append(f"| {idx} | {row:.4f} | {se:.4f} | {t:.4f} | {p:.4f} {sig} |\n")
    md.append("\n")

    # 假设检验
    md.append("## 假设检验\n\n")

    md.append("### 正态性（Shapiro-Wilk）\n")
    norm = assumptions['normality']
    md.append(f"- 统计量: {norm['statistic']:.4f}\n")
    md.append(f"- p 值: {norm['p_value']:.4f}\n")
    md.append(f"- 结论: {'✅ 满足' if norm['is_satisfied'] else '⚠️ 不满足'}\n\n")

    md.append("### 等方差（Breusch-Pagan）\n")
    homo = assumptions['equal_variance']
    md.append(f"- 统计量: {homo['statistic']:.4f}\n")
    md.append(f"- p 值: {homo['p_value']:.4f}\n")
    md.append(f"- 结论: {'✅ 满足' if homo['is_satisfied'] else '⚠️ 不满足'}\n\n")

    # 高影响点
    md.append("## 高影响点\n\n")
    md.append(f"- 阈值: {cooks['threshold']:.4f}\n")
    md.append(f"- 数量: {cooks['n_high_influence']}\n")
    if cooks['n_high_influence'] > 0:
        md.append(f"- 索引: {cooks['high_influence_indices']}\n")
    md.append("\n")

    # VIF
    if vif_df is not None:
        md.append("## 多重共线性（VIF）\n\n")
        md.append("| 变量 | VIF | 诊断 |\n")
        md.append("|------|-----|------|\n")
        for _, row in vif_df.iterrows():
            diag = '⚠️ 高' if row['is_high_collinear'] else '✅'
            md.append(f"| {row['variable']} | {row['vif']:.2f} | {diag} |\n")
        md.append("\n")

    return "".join(md)


def demonstrate_usage() -> None:
    """演示函数使用"""
    print("=" * 70)
    print("Week 09 作业参考实现演示")
    print("=" * 70)

    # 示例数据
    np.random.seed(42)
    n = 100
    x = np.random.normal(loc=50, scale=15, size=n)
    y = 10 + 0.5 * x + np.random.normal(loc=0, scale=5, size=n)

    # 1. 简单线性回归
    print("\n1. 简单线性回归")
    result = simple_linear_regression(x, y)
    coef = result['coefficients']
    print(f"   截距: {coef['intercept']:.2f}")
    print(f"   斜率: {coef['slope']:.4f}")
    print(f"   R²: {result['r_squared']:.4f}")

    # 2. 系数解读
    print("\n2. 系数解读")
    print(interpret_coefficients(result, "广告投入", "销售额"))

    # 3. 假设检查
    print("\n3. 假设检查")
    assumptions = check_assumptions(result)
    print(f"   正态性: p = {assumptions['normality']['p_value']:.4f}")
    print(f"   等方差: p = {assumptions['equal_variance']['p_value']:.4f}")
    print(f"   独立性 (DW): {assumptions['independence']['dw_statistic']:.4f}")

    # 4. Cook's 距离
    print("\n4. Cook's 距离")
    cooks = calculate_cooks_distance(result)
    print(f"   高影响点数量: {cooks['n_high_influence']}")

    # 5. 多元回归
    print("\n5. 多元回归")
    x2 = np.random.normal(loc=100, scale=20, size=n)
    X_multi = np.column_stack([x, x2])

    multi_result = multiple_regression(X_multi, y, var_names=["广告", "价格"])
    print(f"   R²: {multi_result['r_squared']:.4f}")
    print(f"   调整 R²: {multi_result['adj_r_squared']:.4f}")

    # 6. VIF
    print("\n6. VIF（多重共线性）")
    vif_df = calculate_vif(X_multi, var_names=["广告", "价格"])
    print(vif_df.to_string(index=False))

    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print("\n可用函数：")
    print("  1. simple_linear_regression: 简单线性回归")
    print("  2. interpret_coefficients: 解读回归系数")
    print("  3. check_assumptions: 检查 LINE 假设")
    print("  4. calculate_cooks_distance: 计算 Cook's 距离")
    print("  5. multiple_regression: 多元线性回归")
    print("  6. calculate_vif: 计算方差膨胀因子")
    print("  7. regression_report: 生成完整报告")
    print()


if __name__ == "__main__":
    demonstrate_usage()

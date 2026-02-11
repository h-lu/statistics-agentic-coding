#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StatLab Week 03：数据清洗与预处理流水线

本脚本在 Week 02 报告基础上，添加：
1. 缺失值分析与处理
2. 异常值检测与分类处理
3. 特征变换（标准化/编码）
4. 清洗日志生成并追加到 report.md

这是 StatLab 超级线的 Week 03 增量更新。

运行方式：python3 chapters/week_03/examples/99_statlab.py
预期输出：更新后的 report.md，包含清洗日志章节
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class StatLabCleaningLogger:
    """StatLab 清洗日志记录器（Week 03 核心组件）"""

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.entries: list[dict] = []
        self.start_time = datetime.now().isoformat()
        self.stats = {
            'initial_rows': 0,
            'final_rows': 0,
            'missing_filled': 0,
            'outliers_found': 0,
            'rows_removed': 0
        }

    def log(self, step: int, operation: str, target: str, reason: str,
            method: str, affected: int, notes: str = "") -> None:
        """记录清洗操作"""
        self.entries.append({
            'step': step,
            'operation': operation,
            'target': target,
            'reason': reason,
            'method': method,
            'rows_affected': affected,
            'notes': notes,
            'timestamp': datetime.now().isoformat()
        })

    def to_markdown(self) -> str:
        """生成 Markdown 格式的清洗日志"""
        lines = [
            "## 数据清洗与预处理\n",
            f"**数据集**: {self.dataset_name}",
            f"**处理时间**: {self.start_time[:19]}\n",
            "### 处理摘要\n",
            f"- 初始样本量: {self.stats['initial_rows']}",
            f"- 最终样本量: {self.stats['final_rows']}",
            f"- 删除样本: {self.stats['rows_removed']}",
            f"- 填充缺失值: {self.stats['missing_filled']}",
            f"- 发现异常值: {self.stats['outliers_found']}\n",
            "### 详细操作记录\n"
        ]

        for entry in self.entries:
            lines.append(f"**步骤 {entry['step']}: {entry['operation']}**")
            lines.append(f"- 目标: {entry['target']}")
            lines.append(f"- 理由: {entry['reason']}")
            lines.append(f"- 方法: {entry['method']}")
            lines.append(f"- 影响行数: {entry['rows_affected']}")
            if entry['notes']:
                lines.append(f"- 备注: {entry['notes']}")
            lines.append("")

        lines.append("### 数据质量声明\n")
        lines.append("- 所有缺失值已按上述策略处理")
        lines.append("- 异常值已分类（suspicious/VIP/normal）并分别处理")
        lines.append("- 数值特征已标准化，分类特征已编码")
        lines.append("- 清洗后的数据保存于: `data/cleaned/`\n")

        return '\n'.join(lines)


def load_or_generate_data() -> pd.DataFrame:
    """
    加载数据（如果存在）或生成示例数据

    优先尝试加载 Week 02 的数据，如果不存在则生成示例数据
    """
    data_paths = [
        'data/users.csv',
        '../week_02/data/users.csv',
        'chapters/week_02/data/users.csv'
    ]

    for path in data_paths:
        if Path(path).exists():
            print(f"加载数据: {path}")
            return pd.read_csv(path)

    # 生成示例数据
    print("未找到现有数据，生成示例数据...")
    return generate_sample_data()


def generate_sample_data(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """生成带有数据质量问题的示例数据"""
    rng = np.random.default_rng(seed)

    df = pd.DataFrame({
        'user_id': range(1, n + 1),
        'age': rng.integers(18, 80, n),
        'registration_days': np.clip(rng.exponential(100, n), 1, None).round(0).astype(int),
        'total_spend': rng.lognormal(7, 1, n),
        'city': rng.choice(['北京', '上海', '广州', '深圳'], n),
        'income': np.clip(rng.lognormal(8.5, 0.5, n) * 1000, 3000, None),
    })

    df['total_spend'] = np.clip(df['total_spend'], 0, None).round(2)
    df['income'] = df['income'].round(2)

    # 添加缺失值（MCAR 和 MAR）
    df.loc[rng.choice(df.index, 25, replace=False), 'age'] = np.nan
    mar_prob = 1 / (1 + np.exp(-(df['registration_days'] - 150) / 50))
    df.loc[rng.random(n) < mar_prob * 0.4, 'income'] = np.nan

    # 添加异常值
    df.loc[rng.choice(df.index, 3, replace=False), 'total_spend'] = -999
    df.loc[rng.choice(df.index, 5, replace=False), 'total_spend'] = df['total_spend'] * 5 + 50000

    return df


def analyze_missing(df: pd.DataFrame) -> pd.DataFrame:
    """生成缺失值概览"""
    missing = df.isna().sum()
    rate = (df.isna().mean() * 100).round(2)

    overview = pd.DataFrame({
        'missing_count': missing,
        'missing_rate_%': rate,
        'dtype': df.dtypes
    }).sort_values('missing_rate_%', ascending=False)

    return overview[overview['missing_count'] > 0]


def detect_outliers_iqr(df: pd.DataFrame, column: str, multiplier: float = 1.5) -> pd.Series:
    """使用 IQR 方法检测异常值"""
    data = df[column].dropna()
    q1, q3 = data.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - multiplier * iqr, q3 + multiplier * iqr
    return (df[column] < lower) | (df[column] > upper)


def classify_outlier(value: float) -> str:
    """根据业务规则分类异常值"""
    if value < 0:
        return 'suspicious'
    if value > 50000:
        return 'VIP'
    return 'normal'


def clean_data(df: pd.DataFrame, logger: StatLabCleaningLogger) -> pd.DataFrame:
    """
    执行完整的数据清洗流程

    步骤：
    1. 处理数据错误（负数消费）
    2. 标记 VIP 高消费用户
    3. 填充缺失值
    4. 记录所有决策
    """
    initial_rows = len(df)
    logger.stats['initial_rows'] = initial_rows
    df = df.copy()
    step = 1

    # 步骤 1: 删除负数消费（数据错误）
    negative_mask = df['total_spend'] < 0
    negative_count = negative_mask.sum()
    if negative_count > 0:
        logger.log(
            step=step, operation='删除', target='total_spend < 0',
            reason='消费金额不应为负数，判断为数据录入错误',
            method='删除包含负数消费的行',
            affected=int(negative_count),
            notes='负数记录无法确定原始值，选择删除而非修正'
        )
        df = df[~negative_mask]
        step += 1

    # 步骤 2: 标记 VIP 用户（保留但标注）
    df['outlier_category'] = df['total_spend'].apply(classify_outlier)
    vip_count = (df['outlier_category'] == 'VIP').sum()
    if vip_count > 0:
        logger.log(
            step=step, operation='标记', target='total_spend > 50000',
            reason='高消费用户为真实 VIP，不应删除',
            method='添加 outlier_category 列标记为 VIP',
            affected=int(vip_count),
            notes='VIP 用户将在后续分析中单独考虑'
        )
        logger.stats['outliers_found'] = int(vip_count)
        step += 1

    # 步骤 3: 填充 age 缺失（中位数）
    age_missing = df['age'].isna().sum()
    if age_missing > 0:
        fill_value = df['age'].median()
        logger.log(
            step=step, operation='填充', target='age',
            reason='缺失率较低（~5%），MCAR 机制，中位数稳健',
            method=f'中位数填充 ({fill_value:.0f})',
            affected=int(age_missing)
        )
        df['age'] = df['age'].fillna(fill_value)
        logger.stats['missing_filled'] += int(age_missing)
        step += 1

    # 步骤 4: 分组填充 income（按 city）
    income_missing = df['income'].isna().sum()
    if income_missing > 0:
        logger.log(
            step=step, operation='填充', target='income',
            reason='MAR 机制（与老用户相关），分组填充保留地域差异',
            method='按城市分组，使用中位数填充',
            affected=int(income_missing),
            notes='替代方案：删除法（损失 25% 样本）'
        )
        df['income'] = df.groupby('city')['income'].transform(lambda x: x.fillna(x.median()))
        logger.stats['missing_filled'] += int(income_missing)
        step += 1

    # 更新统计
    logger.stats['final_rows'] = len(df)
    logger.stats['rows_removed'] = initial_rows - len(df)

    return df


def transform_features(df: pd.DataFrame, logger: StatLabCleaningLogger) -> pd.DataFrame:
    """
    执行特征变换

    - 数值特征：StandardScaler
    - 分类特征：OneHotEncoder
    """
    df = df.copy()
    step = len(logger.entries) + 1

    # 标准化数值特征
    numeric_cols = ['age', 'income', 'total_spend']
    scaler = StandardScaler()
    df[[f'{c}_scaled' for c in numeric_cols]] = scaler.fit_transform(df[numeric_cols])

    logger.log(
        step=step, operation='变换', target=str(numeric_cols),
        reason='消除量纲影响，使不同特征可比',
        method='StandardScaler (z-score)',
        affected=len(df),
        notes='变换参数已保存，可用于新数据转换'
    )
    step += 1

    # 编码分类特征
    encoder = OneHotEncoder(sparse_output=False, drop=None)
    encoded = encoder.fit_transform(df[['city']])
    encoded_cols = [f'city_{c}' for c in encoder.categories_[0]]
    df[encoded_cols] = encoded

    logger.log(
        step=step, operation='编码', target='city',
        reason='城市为名义分类变量，需转换为数值',
        method='OneHotEncoder',
        affected=len(df),
        notes=f'生成 {len(encoded_cols)} 个二元特征'
    )

    return df


def append_to_report(content: str, report_path: str = 'report.md') -> None:
    """追加内容到报告"""
    path = Path(report_path)

    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            existing = f.read()

        # 如果已有清洗章节，在其后追加
        if '## 数据清洗' in existing:
            # 在清洗章节后插入
            insert_pos = existing.find('## 数据清洗')
            next_section = existing.find('\n## ', insert_pos + 1)
            if next_section == -1:
                new_content = existing + '\n\n' + content
            else:
                new_content = existing[:next_section] + '\n\n' + content + existing[next_section:]
        else:
            new_content = existing + '\n\n' + content

        with open(path, 'w', encoding='utf-8') as f:
            f.write(new_content)
    else:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(f"# StatLab 分析报告\n\n{content}")

    print(f"报告已更新: {report_path}")


def save_checkpoint(df: pd.DataFrame, logger: StatLabCleaningLogger, output_dir: str = 'checkpoint') -> None:
    """保存检查点"""
    Path(output_dir).mkdir(exist_ok=True)

    # 保存清洗后的数据
    data_path = Path(output_dir) / 'week_03_cleaned.csv'
    df.to_csv(data_path, index=False)

    # 保存日志 JSON
    log_path = Path(output_dir) / 'week_03_cleaning_log.json'
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump({
            'dataset_name': logger.dataset_name,
            'start_time': logger.start_time,
            'stats': logger.stats,
            'entries': logger.entries
        }, f, ensure_ascii=False, indent=2)

    print(f"检查点已保存: {output_dir}/")


def main() -> None:
    """主函数：StatLab Week 03 更新"""
    print("=" * 70)
    print("StatLab Week 03: 数据清洗与预处理")
    print("=" * 70)

    # 1. 加载数据
    print("\n[1/5] 加载数据...")
    df = load_or_generate_data()
    print(f"  数据形状: {df.shape}")

    # 2. 缺失值分析
    print("\n[2/5] 缺失值分析...")
    missing_overview = analyze_missing(df)
    if len(missing_overview) > 0:
        print(missing_overview)
    else:
        print("  未发现缺失值")

    # 3. 初始化日志并清洗数据
    print("\n[3/5] 执行数据清洗...")
    logger = StatLabCleaningLogger(dataset_name="用户消费分析")
    df_cleaned = clean_data(df, logger)
    print(f"  清洗后形状: {df_cleaned.shape}")

    # 4. 特征变换
    print("\n[4/5] 执行特征变换...")
    df_transformed = transform_features(df_cleaned, logger)
    print(f"  变换后列数: {len(df_transformed.columns)}")

    # 5. 生成报告并保存
    print("\n[5/5] 生成报告...")
    md_content = logger.to_markdown()

    # 追加到 report.md
    append_to_report(md_content, 'report.md')

    # 保存检查点
    save_checkpoint(df_transformed, logger)

    print("\n" + "=" * 70)
    print("StatLab Week 03 更新完成!")
    print("=" * 70)
    print("\n本周新增内容:")
    print("  - 缺失值分析与处理")
    print("  - 异常值检测与分类")
    print("  - 特征变换与编码")
    print("  - 清洗日志（已追加到 report.md）")
    print("\n下周预告:")
    print("  - EDA 假设清单")
    print("  - 相关性与分组比较")


if __name__ == "__main__":
    main()

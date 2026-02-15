"""
Week 04 Starter Code - EDA Narrative and Hypothesis List

Student implementation template. Students need to complete the following functions:

1. Correlation Analysis
   - calculate_correlation(x, y, method='pearson')
   - compare_correlation_methods(x, y)
   - detect_outlier_impact(x, y)

2. Group Comparison
   - groupby_aggregate(df, group_col, value_col, agg_func)
   - create_pivot_table(df, values, index, columns, aggfunc)
   - compare_group_statistics(df, group_col, value_col)

3. Hypothesis List
   - HypothesisList class
   - validate_hypothesis(hypothesis)
   - prioritize_hypotheses(hypothesis_list)
"""
from __future__ import annotations

from typing import Any

import pandas as pd


def calculate_correlation(x: pd.Series, y: pd.Series, method: str = 'pearson') -> float | None:
    """
    Calculate correlation between two variables.

    Args:
        x: First variable
        y: Second variable
        method: Correlation method ('pearson', 'spearman', 'kendall')

    Returns:
        Correlation coefficient (-1 to 1), or None if cannot compute

    Notes:
        - Pearson: Linear correlation, suitable for normal distribution data
        - Spearman: Monotonic correlation, robust to outliers
        - Kendall: Small samples or ordinal data
    """
    # TODO: Student implementation
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if len(x) == 0:
        return None
    df = pd.DataFrame({'x': x, 'y': y})
    result = df['x'].corr(df['y'], method=method)
    return result if not pd.isna(result) else None


def compare_correlation_methods(x: pd.Series, y: pd.Series) -> dict[str, float]:
    """
    Compare different correlation methods.

    Args:
        x: First variable
        y: Second variable

    Returns:
        Dictionary with pearson, spearman, kendall correlation coefficients
    """
    # TODO: Student implementation
    df = pd.DataFrame({'x': x, 'y': y})
    return {
        'pearson': df['x'].corr(df['y'], method='pearson'),
        'spearman': df['x'].corr(df['y'], method='spearman'),
        'kendall': df['x'].corr(df['y'], method='kendall')
    }


def detect_outlier_impact(x: pd.Series, y: pd.Series) -> dict[str, Any]:
    """
    Detect outlier impact on correlation coefficient.

    Args:
        x: First variable
        y: Second variable

    Returns:
        Dictionary with correlation comparison and outlier information

    Structure:
        {
            'correlation_with_outlier': float,
            'correlation_without_outlier': float,
            'outlier_indices': list[int],
            'has_outlier': bool
        }
    """
    # Create a DataFrame to work with
    df = pd.DataFrame({'x': x, 'y': y}).dropna()

    # Calculate correlation with all data (including potential outliers)
    corr_with = df['x'].corr(df['y'], method='pearson')

    # Detect outliers using IQR method on both variables
    outlier_indices = []

    for col in ['x', 'y']:
        q25 = df[col].quantile(0.25)
        q75 = df[col].quantile(0.75)
        iqr = q75 - q25

        if iqr == 0:
            # Constant column, no outliers
            continue

        lower = q25 - 1.5 * iqr
        upper = q75 + 1.5 * iqr

        # Find indices where the value is an outlier
        col_outliers = df[(df[col] < lower) | (df[col] > upper)].index
        outlier_indices.extend(col_outliers.tolist())

    # Remove duplicates
    outlier_indices = sorted(list(set(outlier_indices)))

    # Calculate correlation without outliers
    if outlier_indices:
        df_clean = df.drop(outlier_indices)
        corr_without = df_clean['x'].corr(df_clean['y'], method='pearson')
    else:
        corr_without = corr_with

    return {
        'correlation_with_outlier': corr_with,
        'correlation_without_outlier': corr_without,
        'outlier_indices': outlier_indices,
        'has_outlier': len(outlier_indices) > 0
    }


def groupby_aggregate(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    agg_func: str | list[str] = 'mean'
) -> pd.Series | pd.DataFrame:
    """
    Group by column and aggregate.

    Args:
        df: Input data
        group_col: Grouping column name
        value_col: Value column to aggregate
        agg_func: Aggregation function ('mean', 'median', 'std', 'count') or list

    Returns:
        Grouped aggregation result
    """
    # TODO: Student implementation
    return df.groupby(group_col)[value_col].agg(agg_func)


def create_pivot_table(
    df: pd.DataFrame,
    values: str,
    index: str,
    columns: str,
    aggfunc: str = 'mean'
) -> pd.DataFrame:
    """
    Create pivot table.

    Args:
        df: Input data
        values: Value column name
        index: Row index column name
        columns: Column index column name
        aggfunc: Aggregation function

    Returns:
        Pivot table DataFrame
    """
    # TODO: Student implementation
    return df.pivot_table(values=values, index=index, columns=columns, aggfunc=aggfunc)


def compare_group_statistics(
    df: pd.DataFrame,
    group_col: str,
    value_col: str
) -> dict[str, Any]:
    """
    Compare statistics between groups.

    Args:
        df: Input data
        group_col: Grouping column name
        value_col: Value column name

    Returns:
        Dictionary with group statistics and differences

    Structure:
        {
            'group_stats': dict,  # Statistics for each group
            'mean_difference': float,  # Difference between group means
            'group_means': dict  # Mean for each group
        }
    """
    # TODO: Student implementation
    grouped = df.groupby(group_col)[value_col].agg(['mean', 'median', 'std', 'count'])

    # Calculate difference between max and min means
    means = grouped['mean']
    mean_diff = means.max() - means.min()

    return {
        'group_stats': grouped.to_dict('index'),
        'mean_difference': mean_diff,
        'group_means': means.to_dict()
    }


class HypothesisList:
    """
    Hypothesis list management class.

    Used to record and organize testable hypotheses discovered during EDA.
    """

    def __init__(self):
        """Initialize empty hypothesis list."""
        self.hypotheses: list[dict[str, Any]] = []

    def add(
        self,
        observation: str,
        explanation: str,
        test_method: str,
        priority: str = 'medium'
    ) -> None:
        """
        Add a hypothesis to the list.

        Args:
            observation: Observed phenomenon
            explanation: Possible explanation
            test_method: Testing method
            priority: Priority level ('high', 'medium', 'low')
        """
        # TODO: Student implementation for validation logic
        self.hypotheses.append({
            'observation': observation,
            'explanation': explanation,
            'test_method': test_method,
            'priority': priority
        })

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert to DataFrame.

        Returns:
            DataFrame containing all hypotheses
        """
        # TODO: Student implementation
        return pd.DataFrame(self.hypotheses)


def validate_hypothesis(hypothesis: dict[str, Any]) -> bool:
    """
    Validate hypothesis format.

    Args:
        hypothesis: Hypothesis dictionary

    Returns:
        True if hypothesis contains all required fields and valid priority
    """
    # TODO: Student implementation
    required_fields = ['observation', 'explanation', 'test_method']
    has_required = all(field in hypothesis for field in required_fields)

    # Validate priority
    priority = hypothesis.get('priority', 'medium')
    valid_priority = priority in {'high', 'medium', 'low'}

    return has_required and valid_priority


def prioritize_hypotheses(hypothesis_list: HypothesisList) -> HypothesisList | pd.DataFrame:
    """
    Prioritize hypotheses by priority level.

    Args:
        hypothesis_list: HypothesisList object

    Returns:
        Sorted hypothesis list or DataFrame
    """
    # TODO: Student implementation
    priority_order = {'high': 0, 'medium': 1, 'low': 2}
    sorted_hypotheses = sorted(
        hypothesis_list.hypotheses,
        key=lambda h: priority_order.get(h.get('priority', 'medium'), 1)
    )

    result = HypothesisList()
    result.hypotheses = sorted_hypotheses
    return result

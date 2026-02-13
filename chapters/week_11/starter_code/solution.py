"""
Week 11 Starter Code Solution
Minimal implementation for validation purposes.
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

__all__ = [
    'fit_decision_tree', 'fit_random_forest',
    'get_feature_importance', 'calculate_feature_importance',
    'tune_hyperparameters', 'tune_hyperparameters_grid', 'tune_hyperparameters_random',
    'detect_overfitting', 'compare_tree_models',
    'review_tree_model_code', 'calculate_permutation_importance'
]


def fit_decision_tree(X, y, max_depth=3, random_state=42):
    """Fit a decision tree for regression or classification."""
    if len(np.unique(y)) <= 10:
        model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    else:
        model = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
    model.fit(X, y)
    return model


def fit_random_forest(X, y, n_estimators=100, random_state=42):
    """Fit a random forest for regression or classification."""
    if len(np.unique(y)) <= 10:
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    else:
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X, y)
    return model


def get_feature_importance(model, X_or_cols, method='builtin'):
    """Get feature importance (builtin or permutation)."""
    import pandas as pd

    # Convert Index to list if needed
    if isinstance(X_or_cols, pd.Index):
        cols = list(X_or_cols)
    elif hasattr(X_or_cols, 'tolist'):
        cols = X_or_cols.tolist()
    elif isinstance(X_or_cols, list):
        cols = X_or_cols
    else:
        cols = None

    if method == 'builtin':
        if cols is not None:
            # Column names provided - return dict with column names as keys
            return dict(zip(cols, model.feature_importances_))
        else:
            # X is DataFrame - use its columns
            if hasattr(model, 'feature_names_in_'):
                return dict(zip(model.feature_names_in_, model.feature_importances_))
            else:
                return dict(zip(X_or_cols.columns, model.feature_importances_))
    elif method == 'permutation':
        if cols is None and isinstance(X_or_cols, pd.DataFrame):
            # permutation importance on DataFrame
            result = permutation_importance(model, X_or_cols, y=None,
                                        random_state=42, n_repeats=10)
            return dict(zip(X_or_cols.columns, result.importances_mean))
        else:
            raise TypeError("calculate_permutation_importance() requires X to be DataFrame when y is None")
    else:
        raise ValueError(f"Unknown method: {method}")


def calculate_feature_importance(model, X_or_cols):
    """Calculate feature importance (alias for get_feature_importance)."""
    return get_feature_importance(model, X_or_cols, method='builtin')


def calculate_permutation_importance(model, X, y):
    """Calculate permutation importance."""
    if not isinstance(X, pd.DataFrame):
        raise TypeError("calculate_permutation_importance() requires X to be DataFrame")
    result = permutation_importance(model, X, y, random_state=42, n_repeats=10)
    return dict(zip(X.columns, result.importances_mean))


def tune_hyperparameters(X, y, model_type='tree', cv=5):
    """Tune hyperparameters using grid search."""
    return tune_hyperparameters_grid(X, y, model_type, cv)


def tune_hyperparameters_grid(X, y, model_type='tree', cv=5):
    """Tune hyperparameters using grid search."""
    if model_type == 'tree':
        param_grid = {'max_depth': [3, 5, 10], 'min_samples_split': [2, 5, 10]}
        base_model = DecisionTreeRegressor(random_state=42)
    else:
        param_grid = {'n_estimators': [50, 100], 'max_depth': [5, 10]}
        base_model = RandomForestRegressor(random_state=42)

    search = GridSearchCV(base_model, param_grid, cv=cv)
    search.fit(X, y)
    return search.best_params_, search.best_score_


def tune_hyperparameters_random(X, y, n_iter=20, cv=5):
    """Tune hyperparameters using randomized search."""
    param_distributions = {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 10, None]}
    if len(np.unique(y)) <= 10:
        base_model = RandomForestClassifier(random_state=42)
    else:
        base_model = RandomForestRegressor(random_state=42)

    search = RandomizedSearchCV(base_model, param_distributions, n_iter=n_iter, cv=cv, random_state=42)
    search.fit(X, y)
    return search.best_params_, search.best_score_


def detect_overfitting(model, X_train, y_train, X_test, y_test):
    """Detect overfitting by comparing train and test performance."""
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    overfitting_gap = train_score - test_score
    return {
        'train_score': train_score,
        'test_score': test_score,
        'overfitting_gap': overfitting_gap,
        'is_overfitting': overfitting_gap > 0.1
    }


def compare_tree_models(X, y):
    """Compare single decision tree vs random forest."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    tree = fit_decision_tree(X_train, y_train)
    forest = fit_random_forest(X_train, y_train)

    tree_score = tree.score(X_test, y_test)
    forest_score = forest.score(X_test, y_test)

    return {
        'tree_score': tree_score,
        'forest_score': forest_score,
        'forest_improvement': forest_score - tree_score
    }


def review_tree_model_code(code_text):
    """Review tree model code for common issues."""
    issues = []

    checks = [
        ('random_state', '是否设置random_state'),
        ('cross_validation', '是否使用交叉验证'),
        ('feature_importance', '是否检查特征重要性'),
        ('baseline_comparison', '是否与基线模型对比'),
        ('overfitting_check', '是否检查过拟合')
    ]

    code_lower = code_text.lower()

    for check_key, description in checks:
        if check_key not in code_lower:
            issues.append(description)

    return {
        'has_issues': len(issues) > 0,
        'issues': issues,
        'total_checks': len(checks)
    }

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

matplotlib.use('Agg')

starter_code_path = Path(__file__).parent.parent / 'starter_code'
if str(starter_code_path) not in sys.path:
    sys.path.insert(0, str(starter_code_path))


def load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_solution_main_smoke(monkeypatch):
    import solution

    monkeypatch.setattr(solution.plt, 'savefig', lambda *args, **kwargs: None)
    solution.main()


def test_paired_bootstrap_ci_smoke(baseline_comparison_data):
    module_path = Path(__file__).parent.parent / 'examples' / '11_baseline_comparison.py'
    baseline_example = load_module(module_path, 'week11_baseline_example')

    X_train = pd.DataFrame(baseline_comparison_data['X_train'], columns=['x1', 'x2', 'x3', 'x4', 'x5', 'x6'])
    X_test = pd.DataFrame(baseline_comparison_data['X_test'], columns=X_train.columns)
    y_train = pd.Series(baseline_comparison_data['y_train'])
    y_test = pd.Series(baseline_comparison_data['y_test'])

    logistic = baseline_example.make_logistic_pipeline()
    logistic.fit(X_train, y_train)

    forest = RandomForestClassifier(
        n_estimators=50,
        max_depth=4,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    forest.fit(X_train, y_train)

    result = baseline_example.paired_bootstrap_auc_difference(
        logistic, forest, X_test, y_test, n_bootstrap=200, random_state=42
    )

    assert 'diff_ci' in result
    assert len(result['diff_ci']) == 2
    assert result['diff_ci'][0] <= result['diff_ci'][1]
    assert isinstance(result['supports_improvement'], bool)

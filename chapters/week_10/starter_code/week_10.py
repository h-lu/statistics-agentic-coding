"""Generate the shared customer churn sample dataset used in this week.

Run from repository root:
    python3 chapters/week_10/starter_code/week_10.py
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_PATH = REPO_ROOT / 'data' / 'customer_churn.csv'
TARGET_CHURN_RATE = 0.20


def generate_customer_churn_data(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Generate the shared churn dataset with a stable ~80/20 class balance."""
    rng = np.random.default_rng(seed)
    purchase_count = rng.poisson(5, n_samples)
    avg_spend = rng.gamma(10, 10, n_samples).round(2)
    days_since_last_purchase = rng.exponential(30, n_samples).round(1)
    membership_days = rng.integers(30, 365, n_samples)
    support_tickets = rng.poisson(1.2, n_samples)
    contract_type = rng.choice(['month_to_month', 'one_year', 'two_year'], n_samples, p=[0.55, 0.30, 0.15])

    # The intercept is calibrated so the default sample lands near 80% retained / 20% churned.
    logit = (-1.45 - 0.12 * purchase_count - 0.012 * avg_spend
             + 0.045 * days_since_last_purchase - 0.004 * membership_days
             + 0.25 * support_tickets + (contract_type == 'month_to_month') * 0.7)
    prob = 1 / (1 + np.exp(-logit))
    is_churned = (rng.random(n_samples) < prob).astype(int)
    return pd.DataFrame({
        'purchase_count': purchase_count,
        'avg_spend': avg_spend,
        'days_since_last_purchase': days_since_last_purchase,
        'membership_days': membership_days,
        'support_tickets': support_tickets,
        'contract_type': contract_type,
        'is_churned': is_churned,
    })


def get_customer_churn_csv_path() -> Path:
    """Return the canonical dataset path used across the repo."""
    return DEFAULT_DATA_PATH


def save_customer_churn_data(path: Path | None = None, *, n_samples: int = 1000, seed: int = 42) -> Path:
    """Generate and save the shared churn CSV to the canonical repo path."""
    target_path = path or get_customer_churn_csv_path()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    generate_customer_churn_data(n_samples=n_samples, seed=seed).to_csv(target_path, index=False)
    return target_path


def _looks_like_expected_dataset(df: pd.DataFrame) -> bool:
    required_columns = {
        'purchase_count',
        'avg_spend',
        'days_since_last_purchase',
        'membership_days',
        'support_tickets',
        'contract_type',
        'is_churned',
    }
    if not required_columns.issubset(df.columns):
        return False
    if df.empty:
        return False
    churn_rate = float(df['is_churned'].mean())
    return abs(churn_rate - TARGET_CHURN_RATE) <= 0.03


def ensure_customer_churn_csv(path: Path | None = None, *, n_samples: int = 1000, seed: int = 42) -> Path:
    """Ensure the canonical churn CSV exists and matches this week's expected balance."""
    target_path = path or get_customer_churn_csv_path()
    if target_path.exists():
        try:
            df = pd.read_csv(target_path)
        except Exception:
            df = pd.DataFrame()
        if _looks_like_expected_dataset(df):
            return target_path
    return save_customer_churn_data(target_path, n_samples=n_samples, seed=seed)


def load_customer_churn_data(path: Path | None = None, *, n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Load the shared churn dataset, regenerating the canonical CSV if it is stale."""
    csv_path = ensure_customer_churn_csv(path, n_samples=n_samples, seed=seed)
    return pd.read_csv(csv_path)


if __name__ == '__main__':
    out = save_customer_churn_data()
    print(f'Wrote {out}')

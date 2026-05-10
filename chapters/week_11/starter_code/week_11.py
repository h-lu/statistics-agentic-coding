"""Generate the shared customer churn sample dataset used in this week.

Run from repository root:
    python3 chapters/week_11/starter_code/week_11.py
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def generate_customer_churn_data(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    purchase_count = rng.poisson(5, n_samples)
    avg_spend = rng.gamma(10, 10, n_samples).round(2)
    days_since_last_purchase = rng.exponential(30, n_samples).round(1)
    membership_days = rng.integers(30, 365, n_samples)
    age = rng.integers(18, 71, n_samples)
    support_tickets = rng.poisson(1.2, n_samples)
    contract_type = rng.choice(['month_to_month', 'one_year', 'two_year'], n_samples, p=[0.55, 0.30, 0.15])
    logit = (-2.4 - 0.12 * purchase_count - 0.012 * avg_spend
             + 0.045 * days_since_last_purchase - 0.004 * membership_days
             + 0.25 * support_tickets + (contract_type == 'month_to_month') * 0.7)
    prob = 1 / (1 + np.exp(-logit))
    is_churned = (rng.random(n_samples) < prob).astype(int)
    return pd.DataFrame({
        'purchase_count': purchase_count,
        'avg_spend': avg_spend,
        'days_since_last_purchase': days_since_last_purchase,
        'membership_days': membership_days,
        'age': age,
        'support_tickets': support_tickets,
        'contract_type': contract_type,
        'is_churned': is_churned,
    })


if __name__ == '__main__':
    out = Path('data/customer_churn.csv')
    out.parent.mkdir(exist_ok=True)
    generate_customer_churn_data().to_csv(out, index=False)
    print(f'Wrote {out}')

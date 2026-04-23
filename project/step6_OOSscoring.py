"""Step 6: Out-of-sample scoring.

For every stock-year in [oos_start, oos_end], build a composite SCORE =
sum_f ( weight_f * z_factor_f ), where weights come from the in-sample FM
t-stats of the selected factors (positive weight for positive-premium factors,
negative for negative). NaN factor values are treated as 0 (neutral).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

OOS_START = 2008
OOS_END = 2023


def score_oos(
    merged_z: pd.DataFrame,
    final_factors: pd.DataFrame,
    oos_start: int = OOS_START,
    oos_end: int = OOS_END,
) -> pd.DataFrame:
    oos = merged_z[
        (merged_z["RETURN_YEAR"] >= oos_start) & (merged_z["RETURN_YEAR"] <= oos_end)
    ].copy()
    print(
        f"Out-of-sample rows: {len(oos):,} "
        f"({oos['RETURN_YEAR'].min()}-{oos['RETURN_YEAR'].max()})"
    )

    weights = dict(zip(final_factors["Factor"], final_factors["t_stat"]))
    missing = [f for f in weights if f not in oos.columns]
    valid = [f for f in weights if f in oos.columns]
    if missing:
        print(f"  Dropping {len(missing)} factor(s) not present in OOS data: {missing}")
    print(f"  Scoring with {len(valid)} factors")

    score = np.zeros(len(oos), dtype=float)
    for f in valid:
        score += oos[f].fillna(0).to_numpy() * weights[f]
    oos["SCORE"] = score

    print(
        f"  Score range: {oos['SCORE'].min():.2f} to {oos['SCORE'].max():.2f}, "
        f"mean {oos['SCORE'].mean():.2f}"
    )
    keep_cols = ["permno", "RETURN_YEAR", "FF49", "SCORE", "RET_ANNUAL"]
    return oos[keep_cols].reset_index(drop=True)

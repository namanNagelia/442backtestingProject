"""Step 6: OOS composite SCORE = sum_f (in-sample t-stat * z-factor)."""
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

    weights = dict(zip(final_factors["Factor"], final_factors["t_stat"]))
    valid = [f for f in weights if f in oos.columns]
    print(f"OOS rows: {len(oos):,}, scoring with {len(valid)} factors")

    score = np.zeros(len(oos), dtype=float)
    for f in valid:
        score += oos[f].fillna(0).to_numpy() * weights[f]
    oos["SCORE"] = score

    keep = ["permno", "RETURN_YEAR", "FF49", "MARKET_CAP", "SCORE", "RET_ANNUAL"]
    return oos[keep].reset_index(drop=True)

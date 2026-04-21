"""Step 5: Select the final set of factors from the FM t-stats.

Two-stage filter:
  1. Keep factors with |t_stat| >= t_threshold (default 1.5).
  2. From those, keep the top_n by |t_stat| as the final model.
"""
from __future__ import annotations

import pandas as pd

T_THRESHOLD = 1.5
TOP_N = 40


def select_factors(
    fm_df: pd.DataFrame,
    t_threshold: float = T_THRESHOLD,
    top_n: int = TOP_N,
) -> pd.DataFrame:
    above = fm_df[fm_df["abs_t"] >= t_threshold].copy()
    print(f"Factors with |t| >= {t_threshold}: {len(above)}")

    final = above.nlargest(top_n, "abs_t")[
        ["Factor", "Avg_Premium", "t_stat", "abs_t", "N_Years"]
    ].reset_index(drop=True)
    print(f"Final selected factors (top {top_n} by |t|): {len(final)}")
    return final

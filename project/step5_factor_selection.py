"""Step 5: Take top N multivariate FM factors by |t|. Sign carried in t_stat weight."""
from __future__ import annotations

import pandas as pd

TOP_N = 25


def select_factors(fm_df: pd.DataFrame, top_n: int = TOP_N) -> pd.DataFrame:
    final = (
        fm_df.sort_values("abs_t", ascending=False)
        .head(top_n)[["Factor", "Avg_Premium", "t_stat", "abs_t", "N_Years"]]
        .reset_index(drop=True)
    )
    n_neg = int((final["t_stat"] < 0).sum())
    print(f"Top {top_n} by |t| (no sign filter): {n_neg} negative-premium factors")
    print(final[["Factor", "Avg_Premium", "t_stat"]].to_string(index=False))
    return final

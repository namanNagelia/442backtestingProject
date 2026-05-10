"""Step 5: Filter multivariate FM by |t| >= threshold AND sign == expected (+1).

OAP signed_predictors are normalized so every factor's expected premium is +1
(high z-score -> high return). Wrong-sign IS survivors are typically snooping
artifacts; sign filter is our primary defense.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

T_THRESHOLD = 1.5
EXPECTED_SIGN = 1


def select_factors(
    fm_df: pd.DataFrame,
    t_threshold: float = T_THRESHOLD,
    expected_sign: int = EXPECTED_SIGN,
) -> pd.DataFrame:
    above = fm_df[fm_df["abs_t"] >= t_threshold].copy()
    sign_ok = above[np.sign(above["Avg_Premium"]) == expected_sign].copy()
    dropped = above[np.sign(above["Avg_Premium"]) != expected_sign]

    final = sign_ok.sort_values("abs_t", ascending=False)[
        ["Factor", "Avg_Premium", "t_stat", "abs_t", "N_Years"]
    ].reset_index(drop=True)
    print(f"|t|>={t_threshold}: {len(above)}, sign-ok: {len(final)}, dropped wrong-sign: {len(dropped)}")
    print(final[["Factor", "Avg_Premium", "t_stat"]].to_string(index=False))
    return final

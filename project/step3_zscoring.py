"""Step 3: Z-score factors within (FF49, RETURN_YEAR), winsorize at 1/99% on IS."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

ID_COLS = [
    "permno", "yyyymm", "YEAR_x", "MONTH", "RETURN_YEAR", "PERMNO_x",
    "YEAR_y", "RET_ANNUAL", "PERMNO_y", "YEAR", "PRC_ABS", "MARKET_CAP",
    "SICCD", "FF49",
]

WINSOR_LOWER = 0.01
WINSOR_UPPER = 0.99
IN_SAMPLE_START = 1986
IN_SAMPLE_END = 2007


def zscore_by_industry_year(
    merged: pd.DataFrame,
    id_cols: Iterable[str] = ID_COLS,
    winsor_lower: float = WINSOR_LOWER,
    winsor_upper: float = WINSOR_UPPER,
    in_start: int = IN_SAMPLE_START,
    in_end: int = IN_SAMPLE_END,
) -> pd.DataFrame:
    id_cols = list(id_cols)
    factor_cols = [c for c in merged.columns if c not in id_cols]
    print(f"Z-scoring {len(factor_cols)} factors by (FF49, RETURN_YEAR)")

    out = merged.copy()
    grouped = out.groupby(["FF49", "RETURN_YEAR"])[factor_cols]
    means = grouped.transform("mean")
    stds = grouped.transform("std").replace(0, np.nan)
    out[factor_cols] = (out[factor_cols] - means) / stds
    out[factor_cols] = out[factor_cols].fillna(0)

    is_panel = out.loc[(out["RETURN_YEAR"] >= in_start) & (out["RETURN_YEAR"] <= in_end)]
    lower_q = is_panel[factor_cols].quantile(winsor_lower)
    upper_q = is_panel[factor_cols].quantile(winsor_upper)
    out[factor_cols] = out[factor_cols].clip(lower=lower_q, upper=upper_q, axis=1)

    ret_lo = float(is_panel["RET_ANNUAL"].quantile(winsor_lower))
    ret_hi = float(is_panel["RET_ANNUAL"].quantile(winsor_upper))
    out["RET_ANNUAL"] = out["RET_ANNUAL"].clip(lower=ret_lo, upper=ret_hi)
    print(f"Winsor at IS [{winsor_lower:.0%}, {winsor_upper:.0%}], RET_ANNUAL [{ret_lo:.3f}, {ret_hi:.3f}]")

    return out

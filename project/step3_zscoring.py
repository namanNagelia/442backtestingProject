"""Step 3: Z-score factor columns within (FF49 industry, RETURN_YEAR) groups."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

# Columns that must NOT be z-scored (identifiers, the return target, join keys).
ID_COLS = [
    "permno",
    "yyyymm",
    "YEAR_x",
    "MONTH",
    "RETURN_YEAR",
    "PERMNO_x",
    "YEAR_y",
    "RET_ANNUAL",
    "PERMNO_y",
    "YEAR",
    "PRC_ABS",
    "MARKET_CAP",
    "SICCD",
    "FF49",
]


def zscore_by_industry_year(
    merged: pd.DataFrame,
    id_cols: Iterable[str] = ID_COLS,
) -> pd.DataFrame:
    """Z-score every non-ID column within each (FF49, RETURN_YEAR) bucket.

    - `std == 0` groups become NaN before division (can't normalize a constant).
    - Remaining NaNs in factor columns are filled with 0 (neutral).
    - `RET_ANNUAL` is in `id_cols` and is therefore left untouched.
    """
    id_cols = list(id_cols)
    factor_cols = [c for c in merged.columns if c not in id_cols]
    print(f"Z-scoring {len(factor_cols)} factor columns by (FF49, RETURN_YEAR)...")

    out = merged.copy()
    grouped = out.groupby(["FF49", "RETURN_YEAR"])[factor_cols]
    means = grouped.transform("mean")
    stds = grouped.transform("std").replace(0, np.nan)
    out[factor_cols] = (out[factor_cols] - means) / stds
    out[factor_cols] = out[factor_cols].fillna(0)

    # Sanity: the return target must survive z-scoring unchanged.
    assert (out["RET_ANNUAL"] == merged["RET_ANNUAL"]).all(), "RET_ANNUAL was altered"

    return out

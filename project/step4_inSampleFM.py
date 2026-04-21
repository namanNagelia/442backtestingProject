"""Step 4: In-sample Fama-MacBeth regressions.

For every year in [in_start, in_end] run a cross-sectional OLS of RET_ANNUAL on
every z-scored factor. Aggregate per-year coefficients into an average premium,
standard deviation, and FM t-stat per factor.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
import pandas as pd
import statsmodels.api as sm

from step3_zscoring import ID_COLS

IN_SAMPLE_START = 1986
IN_SAMPLE_END = 2007
MIN_OBS_PER_YEAR = 50


def run_fama_macbeth(
    merged_z: pd.DataFrame,
    in_start: int = IN_SAMPLE_START,
    in_end: int = IN_SAMPLE_END,
    min_obs_per_year: int = MIN_OBS_PER_YEAR,
    id_cols: Iterable[str] = ID_COLS,
) -> pd.DataFrame:
    id_cols = list(id_cols)
    factor_cols = [c for c in merged_z.columns if c not in id_cols]

    in_sample = merged_z[
        (merged_z["RETURN_YEAR"] >= in_start) & (merged_z["RETURN_YEAR"] <= in_end)
    ]
    print(
        f"In-sample rows: {len(in_sample):,} across "
        f"{in_sample['RETURN_YEAR'].nunique()} years "
        f"({in_sample['RETURN_YEAR'].min()}-{in_sample['RETURN_YEAR'].max()})"
    )

    yearly_coefs: dict[str, list[float]] = defaultdict(list)
    years_used = 0

    for year in sorted(in_sample["RETURN_YEAR"].unique()):
        year_data = in_sample[in_sample["RETURN_YEAR"] == year]
        year_data_clean = year_data[factor_cols + ["RET_ANNUAL"]].dropna()
        if len(year_data_clean) < min_obs_per_year:
            continue

        X = sm.add_constant(year_data_clean[factor_cols])
        y = year_data_clean["RET_ANNUAL"]
        model = sm.OLS(y, X).fit()

        for col in factor_cols:
            yearly_coefs[col].append(model.params.get(col, np.nan))
        years_used += 1
        print(f"  Year {year}: {len(year_data_clean):,} obs")

    print(f"Completed FM regressions across {years_used} years")

    rows = []
    for col in factor_cols:
        coefs = [c for c in yearly_coefs[col] if not np.isnan(c)]
        if len(coefs) < 5:
            continue
        avg_premium = float(np.mean(coefs))
        std_premium = float(np.std(coefs, ddof=1))
        t_stat = (
            avg_premium / (std_premium / np.sqrt(len(coefs))) if std_premium > 0 else 0.0
        )
        rows.append(
            {
                "Factor": col,
                "Avg_Premium": avg_premium,
                "Std_Premium": std_premium,
                "t_stat": t_stat,
                "N_Years": len(coefs),
            }
        )

    fm_df = pd.DataFrame(rows)
    fm_df["abs_t"] = fm_df["t_stat"].abs()
    fm_df = fm_df.sort_values("abs_t", ascending=False).reset_index(drop=True)
    print(f"Factors analyzed: {len(fm_df)}")
    return fm_df

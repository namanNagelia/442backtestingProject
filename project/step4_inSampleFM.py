"""Step 4: Multivariate Fama-MacBeth on a pre-registered ~25-factor universe.

Curated list spans canonical anomaly categories (value, profitability, investment,
momentum, quality, risk, issuance, reversal). Selection is a-priori, NOT data-driven,
to defend against snooping bias from the 209-factor OAP zoo.
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

# Pre-registered canonical anomaly universe (defined a-priori from textbook taxonomy).
CANDIDATE_FACTORS = [
    # Value
    "BM", "EP", "SP", "EntMult", "NetPayoutYield",
    # Profitability
    "GP", "OperProf", "RoE",
    # Investment
    "AssetGrowth", "NOA", "Investment", "InvGrowth",
    # Momentum
    "Mom12m", "Mom6m", "IntMom", "ResidualMomentum",
    # Quality / accruals
    "Accruals", "EarningsConsistency", "NumEarnIncrease",
    # Risk / lottery
    "IdioVol3F", "MaxRet", "BetaTailRisk",
    # Issuance
    "ShareIss1Y", "CompEquIss",
    # Reversal
    "MRreversal",
]


def run_fama_macbeth(
    merged_z: pd.DataFrame,
    in_start: int = IN_SAMPLE_START,
    in_end: int = IN_SAMPLE_END,
    min_obs_per_year: int = MIN_OBS_PER_YEAR,
    id_cols: Iterable[str] = ID_COLS,
    candidate_factors: Iterable[str] = CANDIDATE_FACTORS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    factors = [f for f in candidate_factors if f in merged_z.columns]
    missing = [f for f in candidate_factors if f not in merged_z.columns]
    if missing:
        print(f"WARNING: {len(missing)} curated factors not in data: {missing}")

    in_sample = merged_z[
        (merged_z["RETURN_YEAR"] >= in_start) & (merged_z["RETURN_YEAR"] <= in_end)
    ]
    years = sorted(in_sample["RETURN_YEAR"].unique())
    print(f"In-sample: {len(in_sample):,} rows, {len(years)} years ({years[0]}-{years[-1]}), {len(factors)} curated factors")

    yearly: dict[str, list[float]] = defaultdict(list)
    long_rows: list[dict] = []

    for year in years:
        yd = in_sample[in_sample["RETURN_YEAR"] == year]
        clean = yd[factors + ["RET_ANNUAL"]].dropna()
        if len(clean) < min_obs_per_year:
            continue
        X = sm.add_constant(clean[factors])
        y = clean["RET_ANNUAL"]
        try:
            model = sm.OLS(y, X).fit()
        except Exception:
            continue
        for f in factors:
            coef = float(model.params.get(f, float("nan")))
            yearly[f].append(coef)
            long_rows.append({"Year": int(year), "Factor": f, "Coef": coef})

    rows = []
    for f, coefs in yearly.items():
        clean = [c for c in coefs if not np.isnan(c)]
        if len(clean) < 5:
            continue
        avg = float(np.mean(clean))
        std = float(np.std(clean, ddof=1))
        t = avg / (std / np.sqrt(len(clean))) if std > 0 else 0.0
        rows.append({
            "Factor": f, "Avg_Premium": avg, "Std_Premium": std,
            "t_stat": t, "N_Years": len(clean),
        })
    fm_df = pd.DataFrame(rows)
    fm_df["abs_t"] = fm_df["t_stat"].abs()
    fm_df = fm_df.sort_values("abs_t", ascending=False).reset_index(drop=True)

    print(f"Multivariate FM result ({len(fm_df)} factors):")
    print(fm_df[["Factor", "Avg_Premium", "t_stat"]].to_string(index=False))

    yearly_df = pd.DataFrame(long_rows).sort_values(["Factor", "Year"]).reset_index(drop=True)
    return fm_df, yearly_df

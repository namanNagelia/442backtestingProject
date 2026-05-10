"""Step 4: Two-stage Fama-MacBeth on the full 209-factor OAP universe.

Stage 1: univariate yearly OLS for each factor -> FM t-stat -> top 40 by |t|.
Stage 2: multivariate yearly OLS on those 40 jointly -> FM t-stats.
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
UNIVARIATE_TOP_K = 40


def _aggregate_fm(yearly_coefs: dict[str, list[float]]) -> pd.DataFrame:
    rows = []
    for factor, coefs in yearly_coefs.items():
        clean = [c for c in coefs if not np.isnan(c)]
        if len(clean) < 5:
            continue
        avg = float(np.mean(clean))
        std = float(np.std(clean, ddof=1))
        t = avg / (std / np.sqrt(len(clean))) if std > 0 else 0.0
        rows.append({
            "Factor": factor, "Avg_Premium": avg, "Std_Premium": std,
            "t_stat": t, "N_Years": len(clean),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["abs_t"] = df["t_stat"].abs()
    return df.sort_values("abs_t", ascending=False).reset_index(drop=True)


def run_fama_macbeth(
    merged_z: pd.DataFrame,
    in_start: int = IN_SAMPLE_START,
    in_end: int = IN_SAMPLE_END,
    min_obs_per_year: int = MIN_OBS_PER_YEAR,
    id_cols: Iterable[str] = ID_COLS,
    univariate_top_k: int = UNIVARIATE_TOP_K,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    id_cols = list(id_cols)
    factor_cols = [c for c in merged_z.columns if c not in id_cols]

    in_sample = merged_z[
        (merged_z["RETURN_YEAR"] >= in_start) & (merged_z["RETURN_YEAR"] <= in_end)
    ]
    years = sorted(in_sample["RETURN_YEAR"].unique())
    print(f"In-sample: {len(in_sample):,} rows, {len(years)} years ({years[0]}-{years[-1]}), {len(factor_cols)} factors")

    long_rows: list[dict] = []

    # Stage 1: univariate FM
    uni_yearly: dict[str, list[float]] = defaultdict(list)
    for year in years:
        yd = in_sample[in_sample["RETURN_YEAR"] == year]
        if len(yd) < min_obs_per_year:
            continue
        y = yd["RET_ANNUAL"].to_numpy()
        for f in factor_cols:
            x = yd[f].to_numpy()
            X = sm.add_constant(x)
            try:
                coef = float(sm.OLS(y, X).fit().params[1])
            except Exception:
                coef = float("nan")
            uni_yearly[f].append(coef)
            long_rows.append({"Year": int(year), "Factor": f, "Coef": coef, "Stage": "univariate"})

    uni_df = _aggregate_fm(uni_yearly)
    top_k = uni_df.head(univariate_top_k)["Factor"].tolist()
    print(f"Stage 1 univariate: kept top {univariate_top_k} of {len(uni_df)}")

    # Stage 2: multivariate FM on top-K
    multi_yearly: dict[str, list[float]] = defaultdict(list)
    for year in years:
        yd = in_sample[in_sample["RETURN_YEAR"] == year]
        clean = yd[top_k + ["RET_ANNUAL"]].dropna()
        if len(clean) < min_obs_per_year:
            continue
        X = sm.add_constant(clean[top_k])
        y = clean["RET_ANNUAL"]
        try:
            model = sm.OLS(y, X).fit()
        except Exception:
            continue
        for f in top_k:
            coef = float(model.params.get(f, float("nan")))
            multi_yearly[f].append(coef)
            long_rows.append({"Year": int(year), "Factor": f, "Coef": coef, "Stage": "multivariate"})

    multi_df = _aggregate_fm(multi_yearly)
    print(f"Stage 2 multivariate: {len(multi_df)} factors\n{multi_df[['Factor','Avg_Premium','t_stat']].to_string(index=False)}")

    yearly_coefs_df = (
        pd.DataFrame(long_rows)
        .sort_values(["Stage", "Factor", "Year"])
        .reset_index(drop=True)
    )
    return multi_df, yearly_coefs_df

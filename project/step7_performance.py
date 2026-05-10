"""Step 7: Decile sort, value-weighted L/S, CAPM + 4F alphas, IR."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import statsmodels.api as sm

N_BUCKETS = 10
TRANSACTION_COST_ANNUAL = 0.0


def _assign_deciles(group: pd.DataFrame, n_buckets: int) -> pd.Series:
    try:
        return pd.qcut(group["SCORE"], n_buckets, labels=False, duplicates="drop") + 1
    except ValueError:
        return pd.Series(np.nan, index=group.index)


def evaluate(
    scored_oos: pd.DataFrame,
    crsp_monthly: pd.DataFrame,
    ff_all: pd.DataFrame,
    n_buckets: int = N_BUCKETS,
    transaction_cost_annual: float = TRANSACTION_COST_ANNUAL,
) -> Dict[str, pd.DataFrame]:
    scored = scored_oos.copy()
    scored["decile"] = (
        scored.groupby("RETURN_YEAR", group_keys=False)
        .apply(_assign_deciles, n_buckets=n_buckets)
    )
    scored = scored.dropna(subset=["decile"])
    scored["decile"] = scored["decile"].astype(int)

    yearly_dec = (
        scored.groupby(["RETURN_YEAR", "decile"])["RET_ANNUAL"]
        .mean()
        .unstack("decile")
        .sort_index()
    )
    yearly_dec.columns = [f"D{int(c)}" for c in yearly_dec.columns]

    oos_years = scored["RETURN_YEAR"].unique()
    monthly = crsp_monthly.loc[
        crsp_monthly["YEAR"].isin(oos_years),
        ["PERMNO", "YEAR", "YYYYMM", "RET"],
    ].copy()
    decile_map = scored[["permno", "RETURN_YEAR", "decile", "MARKET_CAP"]]
    joined = monthly.merge(
        decile_map,
        left_on=["PERMNO", "YEAR"],
        right_on=["permno", "RETURN_YEAR"],
        how="inner",
    )

    # Value-weighted decile returns by Dec-T market cap (long D10, short D1).
    joined = joined.dropna(subset=["RET", "MARKET_CAP"])
    joined["W_RET"] = joined["RET"] * joined["MARKET_CAP"]
    grp = joined.groupby(["YYYYMM", "decile"])
    monthly_dec = (
        (grp["W_RET"].sum() / grp["MARKET_CAP"].sum())
        .unstack("decile")
        .sort_index()
    )
    monthly_dec.columns = [f"D{int(c)}" for c in monthly_dec.columns]

    top_col, bot_col = f"D{n_buckets}", "D1"
    monthly_port = monthly_dec.copy()
    monthly_port["long"] = monthly_port[top_col]
    monthly_port["short"] = monthly_port[bot_col]
    monthly_port["ls_gross"] = monthly_port["long"] - monthly_port["short"]

    tc_monthly = transaction_cost_annual / 12.0
    monthly_port["ls"] = monthly_port["ls_gross"] - tc_monthly
    monthly_port["cumulative_ls"] = (1.0 + monthly_port["ls"]).cumprod() - 1.0
    monthly_port["cumulative_ls_gross"] = (1.0 + monthly_port["ls_gross"]).cumprod() - 1.0
    monthly_port = monthly_port.reset_index()

    reg = monthly_port.merge(ff_all, on="YYYYMM", how="inner").dropna(
        subset=["ls", "MKT_RF", "SMB", "HML", "UMD", "RF"]
    )
    y = reg["ls"]

    capm = sm.OLS(y, sm.add_constant(reg[["MKT_RF"]])).fit()
    ff4 = sm.OLS(y, sm.add_constant(reg[["MKT_RF", "SMB", "HML", "UMD"]])).fit()

    n_months = int(len(reg))
    raw_mean_m = float(y.mean())
    raw_std_m = float(y.std(ddof=1))
    raw_mean_ann = raw_mean_m * 12
    raw_std_ann = raw_std_m * np.sqrt(12)
    sharpe_ann = raw_mean_ann / raw_std_ann if raw_std_ann > 0 else 0.0

    alpha_capm_m = float(capm.params["const"])
    t_alpha_capm = float(capm.tvalues["const"])
    beta_capm = float(capm.params["MKT_RF"])

    alpha_4f_m = float(ff4.params["const"])
    t_alpha_4f = float(ff4.tvalues["const"])
    resid_std_m = float(ff4.resid.std(ddof=1))
    ir = (alpha_4f_m / resid_std_m) * np.sqrt(12) if resid_std_m > 0 else 0.0

    cum_total = float(monthly_port["cumulative_ls"].iloc[-1]) if len(monthly_port) else 0.0
    hit_rate_m = float((y > 0).mean())

    summary = pd.DataFrame([
        {"metric": "n_months", "value": n_months},
        {"metric": "transaction_cost_annual", "value": transaction_cost_annual},
        {"metric": "raw_mean_monthly", "value": raw_mean_m},
        {"metric": "raw_mean_annualized", "value": raw_mean_ann},
        {"metric": "raw_std_annualized", "value": raw_std_ann},
        {"metric": "sharpe_annualized", "value": sharpe_ann},
        {"metric": "hit_rate_monthly", "value": hit_rate_m},
        {"metric": "cumulative_total", "value": cum_total},
        {"metric": "alpha_capm_monthly", "value": alpha_capm_m},
        {"metric": "alpha_capm_annualized", "value": alpha_capm_m * 12},
        {"metric": "t_alpha_capm", "value": t_alpha_capm},
        {"metric": "beta_capm_mkt", "value": beta_capm},
        {"metric": "r2_capm", "value": float(capm.rsquared)},
        {"metric": "alpha_4f_monthly", "value": alpha_4f_m},
        {"metric": "alpha_4f_annualized", "value": alpha_4f_m * 12},
        {"metric": "t_alpha_4f", "value": t_alpha_4f},
        {"metric": "resid_std_monthly", "value": resid_std_m},
        {"metric": "information_ratio", "value": ir},
        {"metric": "r2_4f", "value": float(ff4.rsquared)},
    ])

    def _coef_table(fit):
        return pd.DataFrame({
            "term": fit.params.index,
            "coef": fit.params.values,
            "t_stat": fit.tvalues.values,
            "p_value": fit.pvalues.values,
        })

    ic = (
        scored.groupby("RETURN_YEAR")[["SCORE", "RET_ANNUAL"]]
        .corr()
        .unstack()["RET_ANNUAL"]["SCORE"]
        .rename("ic")
        .to_frame()
    )
    ic_mean = float(ic["ic"].mean())
    ic_std = float(ic["ic"].std(ddof=1))
    ic_t = ic_mean / (ic_std / np.sqrt(len(ic))) if ic_std > 0 else 0.0
    ic_summary = pd.DataFrame([
        {"metric": "ic_mean", "value": ic_mean},
        {"metric": "ic_std", "value": ic_std},
        {"metric": "ic_t_stat", "value": ic_t},
        {"metric": "ic_hit_rate", "value": float((ic["ic"] > 0).mean())},
    ])

    print(f"Raw: {raw_mean_ann:.2%}/yr (σ {raw_std_ann:.2%}, Sharpe {sharpe_ann:.2f}), cum {cum_total:.2%}")
    print(f"CAPM α: {alpha_capm_m * 12:.2%}/yr (t={t_alpha_capm:.2f}), β={beta_capm:.2f}")
    print(f"4F α: {alpha_4f_m * 12:.2%}/yr (t={t_alpha_4f:.2f}), IR={ir:.2f}, R²={ff4.rsquared:.2f}")

    return {
        "scored_with_deciles": scored.reset_index(drop=True),
        "yearly_decile_returns": yearly_dec.reset_index(),
        "monthly_portfolio": monthly_port,
        "reg_coefs_capm": _coef_table(capm),
        "reg_coefs_4f": _coef_table(ff4),
        "summary": summary,
        "ic_yearly": ic.reset_index(),
        "ic_summary": ic_summary,
    }

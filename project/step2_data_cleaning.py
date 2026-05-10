"""Step 2: Clean CRSP, map FF49, compound annual returns, merge OAP panel."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

START_YEAR = 1985
END_YEAR = 2023
MIN_PRICE = 5.0
MIN_MARKET_CAP_THOUSANDS = 500_000
DELISTING_RETURN = -0.30


def clean_crsp(
    crsp: pd.DataFrame,
    start_year: int = START_YEAR,
    end_year: int = END_YEAR,
    min_price: float = MIN_PRICE,
    min_market_cap_thousands: int = MIN_MARKET_CAP_THOUSANDS,
) -> pd.DataFrame:
    before_rows = len(crsp)
    crsp = crsp.copy()
    crsp.columns = crsp.columns.str.upper()
    crsp["DATE"] = pd.to_datetime(crsp["DATE"])
    crsp["YEAR"] = crsp["DATE"].dt.year
    crsp["MONTH"] = crsp["DATE"].dt.month
    crsp["YYYYMM"] = crsp["YEAR"] * 100 + crsp["MONTH"]

    crsp = crsp[crsp["SHRCD"].isin([10, 11])]
    crsp = crsp[crsp["EXCHCD"].isin([1, 2, 3])]
    crsp["RET"] = pd.to_numeric(crsp["RET"], errors="coerce")
    crsp = crsp[crsp["RET"].notna()]
    crsp = crsp[crsp["RET"] > -1.0]

    crsp["PRC_ABS"] = crsp["PRC"].abs()
    crsp["MARKET_CAP"] = crsp["PRC_ABS"] * crsp["SHROUT"]
    crsp = crsp[crsp["PRC_ABS"] >= min_price]
    crsp = crsp[crsp["MARKET_CAP"] >= min_market_cap_thousands]
    crsp = crsp[(crsp["YEAR"] >= start_year) & (crsp["YEAR"] <= end_year)]

    print(f"CRSP cleaned: {len(crsp):,} rows (removed {before_rows - len(crsp):,})")
    return crsp


def map_ff49(crsp: pd.DataFrame, sic_mapping: pd.DataFrame) -> pd.DataFrame:
    sic_sorted = sic_mapping.sort_values("SIC_start").reset_index(drop=True)
    intervals = pd.IntervalIndex.from_arrays(
        sic_sorted["SIC_start"],
        sic_sorted["SIC_end"],
        closed="both",
    )
    industry_values = sic_sorted["Industry"].to_numpy()

    sic_numeric = pd.to_numeric(crsp["SICCD"], errors="coerce")
    idx = intervals.get_indexer(sic_numeric.fillna(-1).to_numpy())

    out = crsp.copy()
    out["FF49"] = np.where(idx >= 0, industry_values[idx], 49).astype("int16")
    return out


def compute_annual_returns(crsp: pd.DataFrame) -> pd.DataFrame:
    log1p_ret = np.log1p(crsp["RET"].to_numpy())
    annual = (
        pd.DataFrame(
            {
                "PERMNO": crsp["PERMNO"].to_numpy(),
                "YEAR": crsp["YEAR"].to_numpy(),
                "LOG1P_RET": log1p_ret,
            }
        )
        .groupby(["PERMNO", "YEAR"], sort=False, as_index=False)["LOG1P_RET"]
        .sum()
    )
    annual["RET_ANNUAL"] = np.expm1(annual["LOG1P_RET"])
    annual = annual[["PERMNO", "YEAR", "RET_ANNUAL"]]
    print(f"Annual returns: {len(annual):,} stock-years")
    return annual


def build_oap_december(oap_factors: pd.DataFrame) -> pd.DataFrame:
    yyyymm = oap_factors["yyyymm"].to_numpy()
    new_cols = pd.DataFrame(
        {"YEAR": yyyymm // 100, "MONTH": yyyymm % 100},
        index=oap_factors.index,
    )
    oap = pd.concat([oap_factors, new_cols], axis=1)
    oap_dec = oap[oap["MONTH"] == 12].copy()
    oap_dec["RETURN_YEAR"] = oap_dec["YEAR"] + 1
    print(f"OAP December: {len(oap_dec):,} rows")
    return oap_dec


def merge_panel(
    oap_dec: pd.DataFrame,
    annual_returns: pd.DataFrame,
    dec_crsp: pd.DataFrame,
    delisting_return: float = DELISTING_RETURN,
) -> pd.DataFrame:
    merged = pd.merge(
        oap_dec,
        annual_returns,
        left_on=["permno", "RETURN_YEAR"],
        right_on=["PERMNO", "YEAR"],
        how="left",
    )
    n_delisted = int(merged["RET_ANNUAL"].isna().sum())
    merged["RET_ANNUAL"] = merged["RET_ANNUAL"].fillna(delisting_return)
    print(f"Survivorship: {n_delisted:,} delisted stock-years imputed at {delisting_return:.0%}")

    merged = pd.merge(
        merged,
        dec_crsp,
        left_on=["permno", "YEAR_x"],
        right_on=["PERMNO", "YEAR"],
        how="left",
    )
    before = len(merged)
    merged = merged[merged["FF49"].notna()].copy()
    print(f"Merged panel: {len(merged):,} rows (dropped {before - len(merged):,} missing FF49)")
    return merged


def clean_ff_all(ff_factors: pd.DataFrame, ff_mom: pd.DataFrame) -> pd.DataFrame:
    ff_factors = ff_factors.copy()
    ff_factors.columns = ["YYYYMM", "MKT_RF", "SMB", "HML", "RF"]
    ff_factors = ff_factors[ff_factors["YYYYMM"].astype(str).str.len() == 6]
    ff_factors["YYYYMM"] = ff_factors["YYYYMM"].astype(int)
    ff_factors[["MKT_RF", "SMB", "HML", "RF"]] = (
        ff_factors[["MKT_RF", "SMB", "HML", "RF"]].apply(pd.to_numeric, errors="coerce") / 100
    )

    ff_mom = ff_mom.copy()
    ff_mom.columns = ["YYYYMM", "UMD"]
    ff_mom = ff_mom[ff_mom["YYYYMM"].astype(str).str.len() == 6]
    ff_mom["YYYYMM"] = ff_mom["YYYYMM"].astype(int)
    ff_mom["UMD"] = pd.to_numeric(ff_mom["UMD"], errors="coerce") / 100

    ff_all = pd.merge(ff_factors, ff_mom, on="YYYYMM", how="inner")
    print(f"FF factors: {len(ff_all):,} months")
    return ff_all


def clean_and_merge(raw: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    crsp = clean_crsp(raw["crsp"])
    crsp = map_ff49(crsp, raw["sic_mapping"])
    annual_returns = compute_annual_returns(crsp)

    dec_crsp = crsp.loc[
        crsp["MONTH"] == 12,
        ["PERMNO", "YEAR", "PRC_ABS", "MARKET_CAP", "SICCD", "FF49"],
    ].copy()

    oap_dec = build_oap_december(raw["oap_factors"])
    merged = merge_panel(oap_dec, annual_returns, dec_crsp)
    ff_all = clean_ff_all(raw["ff_factors"], raw["ff_mom"])

    return {
        "crsp_cleaned": crsp,
        "annual_returns": annual_returns,
        "dec_crsp": dec_crsp,
        "merged": merged,
        "ff_all": ff_all,
    }

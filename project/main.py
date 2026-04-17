"""Run the full pipeline: load -> clean+merge -> z-score. Writes outputs to cleaned_data/.

Usage:
    python project/main.py
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict

import pandas as pd

# Make `project/` importable whether you run this as a script or as a module.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import step1_data_loading as step1  # noqa: E402
import step2_data_cleaning as step2  # noqa: E402
import step3_zscoring as step3  # noqa: E402


def _missing_data_top(merged: pd.DataFrame, n: int = 30) -> pd.Series:
    factor_cols = [c for c in merged.columns if c not in step3.ID_COLS]
    pct = (merged[factor_cols].isna().sum() / len(merged) * 100).sort_values(ascending=False)
    return pct.head(n)


def run_pipeline(data_dir: str = "data", output_dir: str = "cleaned_data") -> Dict[str, pd.DataFrame]:
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("STEP 1: Loading raw data")
    print("=" * 60)
    raw = step1.load_raw_data(data_dir)

    print("\n" + "=" * 60)
    print("STEP 2: Cleaning + merging panel")
    print("=" * 60)
    cleaned = step2.clean_and_merge(raw)

    print("\n" + "=" * 60)
    print("STEP 3: Z-scoring factors")
    print("=" * 60)
    merged_z = step3.zscore_by_industry_year(cleaned["merged"])
    cleaned["merged_z"] = merged_z

    # Persist artifacts the dashboard reads later.
    merged_path = os.path.join(output_dir, "cleaned_merged_data.csv")
    ff_path = os.path.join(output_dir, "ff_factors_clean.csv")
    zscored_path = os.path.join(output_dir, "cleaned_merged_zscored.csv")

    print("\nWriting outputs...")
    cleaned["merged"].to_csv(merged_path, index=False)
    print(f"  {merged_path}")
    cleaned["ff_all"].to_csv(ff_path, index=False)
    print(f"  {ff_path}")
    merged_z.to_csv(zscored_path, index=False)
    print(f"  {zscored_path}")

    stats = {
        "crsp_cleaned_rows": int(len(cleaned["crsp_cleaned"])),
        "annual_return_stock_years": int(len(cleaned["annual_returns"])),
        "merged_rows": int(len(cleaned["merged"])),
        "merged_zscored_rows": int(len(merged_z)),
        "ff_all_months": int(len(cleaned["ff_all"])),
        "crsp_year_min": int(cleaned["crsp_cleaned"]["YEAR"].min()),
        "crsp_year_max": int(cleaned["crsp_cleaned"]["YEAR"].max()),
        "missing_pct_top30": _missing_data_top(cleaned["merged"]).round(3).to_dict(),
        "bm_mean_after_zscore": float(merged_z["BM"].mean()),
    }
    stats_path = os.path.join(output_dir, "pipeline_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  {stats_path}")

    print("\nPipeline complete.")
    print(f"  BM z-scored mean: {stats['bm_mean_after_zscore']:.6f} (should be ~0)")
    return cleaned


if __name__ == "__main__":
    run_pipeline()

"""Run the full pipeline: load -> clean+merge -> z-score -> FM -> select -> OOS score -> perf.

Usage:
    python project/main.py
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import step1_data_loading as step1  # noqa: E402
import step2_data_cleaning as step2  # noqa: E402
import step3_zscoring as step3  # noqa: E402
import step4_inSampleFM as step4  # noqa: E402
import step5_factor_selection as step5  # noqa: E402
import step6_OOSscoring as step6  # noqa: E402
import step7_performance as step7  # noqa: E402


def run_pipeline(data_dir: str = "data", output_dir: str = "cleaned_data") -> Dict[str, pd.DataFrame]:
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60); print("STEP 1: Loading raw data"); print("=" * 60)
    raw = step1.load_raw_data(data_dir)

    print("\n" + "=" * 60); print("STEP 2: Cleaning + merging panel"); print("=" * 60)
    cleaned = step2.clean_and_merge(raw)

    print("\n" + "=" * 60); print("STEP 3: Z-scoring factors"); print("=" * 60)
    merged_z = step3.zscore_by_industry_year(cleaned["merged"])
    cleaned["merged_z"] = merged_z

    print("\n" + "=" * 60); print("STEP 4: In-sample Fama-MacBeth regressions"); print("=" * 60)
    fm_df, yearly_coefs = step4.run_fama_macbeth(merged_z)
    cleaned["fm_results"] = fm_df
    cleaned["fm_yearly_coefs"] = yearly_coefs

    print("\n" + "=" * 60); print("STEP 5: Factor selection"); print("=" * 60)
    final_factors = step5.select_factors(fm_df)
    cleaned["final_factors"] = final_factors

    print("\n" + "=" * 60); print("STEP 6: Out-of-sample scoring"); print("=" * 60)
    scored = step6.score_oos(merged_z, final_factors)
    cleaned["oos_scores"] = scored

    print("\n" + "=" * 60); print("STEP 7: OOS performance"); print("=" * 60)
    perf = step7.evaluate(scored, cleaned["crsp_cleaned"], cleaned["ff_all"])
    cleaned.update(
        {
            "oos_yearly_decile_returns": perf["yearly_decile_returns"],
            "oos_monthly_portfolio": perf["monthly_portfolio"],
            "oos_reg_coefs_capm": perf["reg_coefs_capm"],
            "oos_reg_coefs_4f": perf["reg_coefs_4f"],
            "oos_summary": perf["summary"],
            "oos_ic_yearly": perf["ic_yearly"],
            "oos_ic_summary": perf["ic_summary"],
            "oos_scored_deciles": perf["scored_with_deciles"],
        }
    )

    paths = {
        "merged": os.path.join(output_dir, "cleaned_merged_data.parquet"),
        "ff": os.path.join(output_dir, "ff_factors_clean.parquet"),
        "zscored": os.path.join(output_dir, "cleaned_merged_zscored.parquet"),
        "fm": os.path.join(output_dir, "fm_regression_results.parquet"),
        "fm_yearly": os.path.join(output_dir, "fm_yearly_coefs.parquet"),
        "final": os.path.join(output_dir, "final_factors.parquet"),
        "oos_scores": os.path.join(output_dir, "oos_scores.parquet"),
        "oos_yearly_deciles": os.path.join(output_dir, "oos_yearly_decile_returns.parquet"),
        "oos_monthly": os.path.join(output_dir, "oos_monthly_portfolio.parquet"),
        "oos_capm": os.path.join(output_dir, "oos_reg_coefs_capm.parquet"),
        "oos_4f": os.path.join(output_dir, "oos_reg_coefs_4f.parquet"),
        "oos_summary": os.path.join(output_dir, "oos_performance_summary.parquet"),
        "oos_ic": os.path.join(output_dir, "oos_ic_yearly.parquet"),
        "oos_ic_summary": os.path.join(output_dir, "oos_ic_summary.parquet"),
    }

    def _downcast(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        f64 = out.select_dtypes(include="float64").columns
        i64 = out.select_dtypes(include="int64").columns
        out[f64] = out[f64].astype("float32")
        out[i64] = out[i64].astype("int32")
        return out

    SAMPLE_N = 30_000
    merged_sample = _downcast(
        cleaned["merged"].sample(n=SAMPLE_N, random_state=42)
        .sort_values("RETURN_YEAR").reset_index(drop=True)
        if len(cleaned["merged"]) > SAMPLE_N else cleaned["merged"]
    )
    zscored_sample = _downcast(
        merged_z.sample(n=SAMPLE_N, random_state=42)
        .sort_values("RETURN_YEAR").reset_index(drop=True)
        if len(merged_z) > SAMPLE_N else merged_z
    )

    print("\nWriting outputs...")
    merged_sample.to_parquet(paths["merged"], index=False, compression="zstd", compression_level=19); print(f"  {paths['merged']}")
    cleaned["ff_all"].to_parquet(paths["ff"], index=False);                           print(f"  {paths['ff']}")
    zscored_sample.to_parquet(paths["zscored"], index=False, compression="zstd", compression_level=19); print(f"  {paths['zscored']}")
    fm_df.to_parquet(paths["fm"], index=False);                                       print(f"  {paths['fm']}")
    yearly_coefs.to_parquet(paths["fm_yearly"], index=False);                         print(f"  {paths['fm_yearly']}")
    final_factors.to_parquet(paths["final"], index=False);                            print(f"  {paths['final']}")
    scored.to_parquet(paths["oos_scores"], index=False);                              print(f"  {paths['oos_scores']}")
    perf["yearly_decile_returns"].to_parquet(paths["oos_yearly_deciles"], index=False); print(f"  {paths['oos_yearly_deciles']}")
    perf["monthly_portfolio"].to_parquet(paths["oos_monthly"], index=False);          print(f"  {paths['oos_monthly']}")
    perf["reg_coefs_capm"].to_parquet(paths["oos_capm"], index=False);                print(f"  {paths['oos_capm']}")
    perf["reg_coefs_4f"].to_parquet(paths["oos_4f"], index=False);                    print(f"  {paths['oos_4f']}")
    perf["summary"].to_parquet(paths["oos_summary"], index=False);                    print(f"  {paths['oos_summary']}")
    perf["ic_yearly"].to_parquet(paths["oos_ic"], index=False);                       print(f"  {paths['oos_ic']}")
    perf["ic_summary"].to_parquet(paths["oos_ic_summary"], index=False);              print(f"  {paths['oos_ic_summary']}")

    summary_kv = dict(zip(perf["summary"]["metric"], perf["summary"]["value"]))
    ic_kv = dict(zip(perf["ic_summary"]["metric"], perf["ic_summary"]["value"]))

    stats = {
        "crsp_cleaned_rows": int(len(cleaned["crsp_cleaned"])),
        "annual_return_stock_years": int(len(cleaned["annual_returns"])),
        "merged_rows": int(len(cleaned["merged"])),
        "merged_zscored_rows": int(len(merged_z)),
        "ff_all_months": int(len(cleaned["ff_all"])),
        "crsp_year_min": int(cleaned["crsp_cleaned"]["YEAR"].min()),
        "crsp_year_max": int(cleaned["crsp_cleaned"]["YEAR"].max()),
        "bm_mean_after_zscore": float(merged_z["BM"].mean()),
        "fm_in_sample_start": int(step4.IN_SAMPLE_START),
        "fm_in_sample_end": int(step4.IN_SAMPLE_END),
        "fm_factors_analyzed": int(len(fm_df)),
        "final_factor_count": int(len(final_factors)),
        "final_top_n": int(step5.TOP_N),
        "final_negative_premium_count": int((final_factors["t_stat"] < 0).sum()),
        "oos_start": int(step6.OOS_START),
        "oos_end": int(step6.OOS_END),
        "oos_rows": int(len(scored)),
        "oos_n_months": int(summary_kv["n_months"]),
        "oos_raw_mean_annualized": float(summary_kv["raw_mean_annualized"]),
        "oos_raw_std_annualized": float(summary_kv["raw_std_annualized"]),
        "oos_sharpe_annualized": float(summary_kv["sharpe_annualized"]),
        "oos_hit_rate_monthly": float(summary_kv["hit_rate_monthly"]),
        "oos_cumulative_total": float(summary_kv["cumulative_total"]),
        "oos_alpha_capm_annualized": float(summary_kv["alpha_capm_annualized"]),
        "oos_t_alpha_capm": float(summary_kv["t_alpha_capm"]),
        "oos_beta_capm_mkt": float(summary_kv["beta_capm_mkt"]),
        "oos_r2_capm": float(summary_kv["r2_capm"]),
        "oos_alpha_4f_annualized": float(summary_kv["alpha_4f_annualized"]),
        "oos_t_alpha_4f": float(summary_kv["t_alpha_4f"]),
        "oos_information_ratio": float(summary_kv["information_ratio"]),
        "oos_r2_4f": float(summary_kv["r2_4f"]),
        "oos_ic_mean": float(ic_kv["ic_mean"]),
        "oos_ic_t_stat": float(ic_kv["ic_t_stat"]),
        "oos_ic_hit_rate": float(ic_kv["ic_hit_rate"]),
    }
    stats_path = os.path.join(output_dir, "pipeline_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  {stats_path}")

    print("\nPipeline complete.")
    print(f"  Final model: {stats['final_factor_count']} factors")
    print(
        f"  OOS raw: {stats['oos_raw_mean_annualized']:.2%}/yr (Sharpe {stats['oos_sharpe_annualized']:.2f}, "
        f"cum {stats['oos_cumulative_total']:.2%})"
    )
    print(
        f"  CAPM α: {stats['oos_alpha_capm_annualized']:.2%}/yr (t={stats['oos_t_alpha_capm']:.2f})"
    )
    print(
        f"  4F α: {stats['oos_alpha_4f_annualized']:.2%}/yr (t={stats['oos_t_alpha_4f']:.2f}), "
        f"IR={stats['oos_information_ratio']:.2f}"
    )
    print(f"  OOS IC: mean {stats['oos_ic_mean']:.3f} (t={stats['oos_ic_t_stat']:.2f})")
    return cleaned


if __name__ == "__main__":
    run_pipeline()

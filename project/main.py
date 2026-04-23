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
        "merged": os.path.join(output_dir, "cleaned_merged_data.csv"),
        "ff": os.path.join(output_dir, "ff_factors_clean.csv"),
        "zscored": os.path.join(output_dir, "cleaned_merged_zscored.csv"),
        "fm": os.path.join(output_dir, "fm_regression_results.csv"),
        "fm_yearly": os.path.join(output_dir, "fm_yearly_coefs.csv"),
        "final": os.path.join(output_dir, "final_factors.csv"),
        "oos_scores": os.path.join(output_dir, "oos_scores.csv"),
        "oos_yearly_deciles": os.path.join(output_dir, "oos_yearly_decile_returns.csv"),
        "oos_monthly": os.path.join(output_dir, "oos_monthly_portfolio.csv"),
        "oos_capm": os.path.join(output_dir, "oos_reg_coefs_capm.csv"),
        "oos_4f": os.path.join(output_dir, "oos_reg_coefs_4f.csv"),
        "oos_summary": os.path.join(output_dir, "oos_performance_summary.csv"),
        "oos_ic": os.path.join(output_dir, "oos_ic_yearly.csv"),
        "oos_ic_summary": os.path.join(output_dir, "oos_ic_summary.csv"),
    }

    print("\nWriting outputs...")
    cleaned["merged"].to_csv(paths["merged"], index=False);              print(f"  {paths['merged']}")
    cleaned["ff_all"].to_csv(paths["ff"], index=False);                   print(f"  {paths['ff']}")
    merged_z.to_csv(paths["zscored"], index=False);                       print(f"  {paths['zscored']}")
    fm_df.to_csv(paths["fm"], index=False);                               print(f"  {paths['fm']}")
    yearly_coefs.to_csv(paths["fm_yearly"], index=False);                 print(f"  {paths['fm_yearly']}")
    final_factors.to_csv(paths["final"], index=False);                    print(f"  {paths['final']}")
    scored.to_csv(paths["oos_scores"], index=False);                      print(f"  {paths['oos_scores']}")
    perf["yearly_decile_returns"].to_csv(paths["oos_yearly_deciles"], index=False); print(f"  {paths['oos_yearly_deciles']}")
    perf["monthly_portfolio"].to_csv(paths["oos_monthly"], index=False); print(f"  {paths['oos_monthly']}")
    perf["reg_coefs_capm"].to_csv(paths["oos_capm"], index=False);       print(f"  {paths['oos_capm']}")
    perf["reg_coefs_4f"].to_csv(paths["oos_4f"], index=False);           print(f"  {paths['oos_4f']}")
    perf["summary"].to_csv(paths["oos_summary"], index=False);            print(f"  {paths['oos_summary']}")
    perf["ic_yearly"].to_csv(paths["oos_ic"], index=False);               print(f"  {paths['oos_ic']}")
    perf["ic_summary"].to_csv(paths["oos_ic_summary"], index=False);      print(f"  {paths['oos_ic_summary']}")

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
        "fm_factors_above_threshold": int((fm_df["abs_t"] >= step5.T_THRESHOLD).sum()),
        "final_factor_count": int(len(final_factors)),
        "final_factor_t_threshold": float(step5.T_THRESHOLD),
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

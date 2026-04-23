"""
Usage:
    streamlit run project/dashboard.py
"""
from __future__ import annotations

import json
import os

import pandas as pd
import plotly.express as px
import streamlit as st

CLEANED_DIR = "cleaned_data"
P = lambda name: os.path.join(CLEANED_DIR, name)

PATHS = {
    "merged": P("cleaned_merged_data.csv"),
    "zscored": P("cleaned_merged_zscored.csv"),
    "stats": P("pipeline_stats.json"),
    "fm": P("fm_regression_results.csv"),
    "fm_yearly": P("fm_yearly_coefs.csv"),
    "final": P("final_factors.csv"),
    "oos_scores": P("oos_scores.csv"),
    "oos_yearly_deciles": P("oos_yearly_decile_returns.csv"),
    "oos_monthly": P("oos_monthly_portfolio.csv"),
    "oos_capm": P("oos_reg_coefs_capm.csv"),
    "oos_4f": P("oos_reg_coefs_4f.csv"),
    "oos_summary": P("oos_performance_summary.csv"),
    "oos_ic": P("oos_ic_yearly.csv"),
}

ID_COLS = [
    "permno", "yyyymm", "YEAR_x", "MONTH", "RETURN_YEAR", "PERMNO_x",
    "YEAR_y", "RET_ANNUAL", "PERMNO_y", "YEAR", "PRC_ABS", "MARKET_CAP",
    "SICCD", "FF49",
]

FF49_NAMES = {
    1: "Agric", 2: "Food", 3: "Soda", 4: "Beer", 5: "Smoke", 6: "Toys",
    7: "Fun", 8: "Books", 9: "Hshld", 10: "Clths", 11: "Hlth", 12: "MedEq",
    13: "Drugs", 14: "Chems", 15: "Rubbr", 16: "Txtls", 17: "BldMt", 18: "Cnstr",
    19: "Steel", 20: "FabPr", 21: "Mach", 22: "ElcEq", 23: "Autos", 24: "Aero",
    25: "Ships", 26: "Guns", 27: "Gold", 28: "Mines", 29: "Coal", 30: "Oil",
    31: "Util", 32: "Telcm", 33: "PerSv", 34: "BusSv", 35: "Hardw", 36: "Softw",
    37: "Chips", 38: "LabEq", 39: "Paper", 40: "Boxes", 41: "Trans", 42: "Whlsl",
    43: "Rtail", 44: "Meals", 45: "Banks", 46: "Insur", 47: "RlEst", 48: "Fin",
    49: "Other",
}

LONG_COLOR = "#2E8B57"
SHORT_COLOR = "#B22222"


@st.cache_data(show_spinner="Loading pipeline outputs...")
def load_all():
    required = ["merged", "zscored", "stats"]
    for k in required:
        if not os.path.exists(PATHS[k]):
            return None
    out = {
        "merged": pd.read_csv(PATHS["merged"], low_memory=False),
        "zscored": pd.read_csv(PATHS["zscored"], low_memory=False),
    }
    with open(PATHS["stats"]) as f:
        out["stats"] = json.load(f)
    for k in (
        "fm", "fm_yearly", "final", "oos_scores",
        "oos_yearly_deciles", "oos_monthly", "oos_capm", "oos_4f",
        "oos_summary", "oos_ic",
    ):
        out[k] = pd.read_csv(PATHS[k]) if os.path.exists(PATHS[k]) else None
    return out


st.set_page_config(page_title="Backtesting Pipeline Dashboard", layout="wide")
st.title("Backtesting Pipeline Dashboard")
st.caption(
    "End-to-end view: cleaned panel → z-scored factors → in-sample Fama-MacBeth → "
    "selected factors → out-of-sample composite score and portfolio performance."
)

data = load_all()
if data is None:
    st.error(
        "Pipeline outputs not found. Run `python project/main.py` first to "
        f"generate files in `{CLEANED_DIR}/`."
    )
    st.stop()

merged = data["merged"]
merged_z = data["zscored"]
stats = data["stats"]
fm_df = data["fm"]
fm_yearly = data["fm_yearly"]
final_factors = data["final"]
oos_scores = data["oos_scores"]
oos_yearly_deciles = data["oos_yearly_deciles"]
oos_monthly = data["oos_monthly"]
oos_capm = data["oos_capm"]
oos_4f = data["oos_4f"]
oos_summary = data["oos_summary"]
oos_ic = data["oos_ic"]

factor_cols = [c for c in merged.columns if c not in ID_COLS]

# ---------- Headline strip ----------
h1, h2, h3, h4, h5 = st.columns(5)
h1.metric("Final factors", stats.get("final_factor_count", "–"))
h2.metric(
    "OOS raw L/S (ann.)",
    f"{stats.get('oos_raw_mean_annualized', 0):.2%}",
    delta=f"Sharpe {stats.get('oos_sharpe_annualized', 0):.2f}",
)
h3.metric(
    "CAPM α (ann.)",
    f"{stats.get('oos_alpha_capm_annualized', 0):.2%}",
    delta=f"t={stats.get('oos_t_alpha_capm', 0):.2f}",
)
h4.metric(
    "4-Factor α (ann.)",
    f"{stats.get('oos_alpha_4f_annualized', 0):.2%}",
    delta=f"t={stats.get('oos_t_alpha_4f', 0):.2f}",
)
h5.metric(
    "Information Ratio",
    f"{stats.get('oos_information_ratio', 0):.2f}",
    delta=f"cum {stats.get('oos_cumulative_total', 0):.1%}",
)
st.caption(
    f"In-sample window: {stats.get('fm_in_sample_start','?')}–{stats.get('fm_in_sample_end','?')}. "
    f"OOS window: {stats.get('oos_start','?')}–{stats.get('oos_end','?')}. "
    f"Panel years: {stats['crsp_year_min']}–{stats['crsp_year_max']}."
)

tab_panel, tab_z, tab_fm, tab_final, tab_oos = st.tabs(
    ["Panel", "Z-scored data", "In-sample FM", "Final factors", "Out-of-sample"]
)

# =======================================================================
# TAB 1 — PANEL
# =======================================================================
with tab_panel:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Stocks per return year")
        per_year = merged.groupby("RETURN_YEAR")["permno"].nunique().reset_index()
        per_year.columns = ["RETURN_YEAR", "num_stocks"]
        fig = px.line(
            per_year, x="RETURN_YEAR", y="num_stocks", markers=True,
            labels={"RETURN_YEAR": "Return year", "num_stocks": "Unique stocks"},
        )
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Annual return distribution")
        ret_df = merged[["RETURN_YEAR", "RET_ANNUAL"]].copy()
        lo, hi = ret_df["RET_ANNUAL"].quantile([0.01, 0.99])
        ret_df["RET_ANNUAL"] = ret_df["RET_ANNUAL"].clip(lo, hi)
        fig = px.box(
            ret_df, x="RETURN_YEAR", y="RET_ANNUAL", points=False,
            labels={"RETURN_YEAR": "Return year", "RET_ANNUAL": "Annual return"},
        )
        fig.add_hline(y=0, line_dash="dot", line_color="gray")
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Clipped to 1st–99th percentile for display.")

    st.subheader("Industry distribution (FF49)")
    ind = merged["FF49"].astype("int64").value_counts().sort_values(ascending=True).reset_index()
    ind.columns = ["FF49", "rows"]
    ind["label"] = ind["FF49"].astype(str) + " — " + ind["FF49"].map(FF49_NAMES).fillna("?")
    fig = px.bar(
        ind, x="rows", y="label", orientation="h",
        labels={"rows": "Stock-year observations", "label": "FF49 industry"},
    )
    fig.update_layout(height=900, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

# =======================================================================
# TAB 2 — Z-SCORED DATA
# =======================================================================
with tab_z:
    st.caption(
        "Factors are z-scored within each (FF49 × return year) bucket. "
        "Remaining NaNs are set to 0 (neutral)."
    )

    default_factor = "BM" if "BM" in factor_cols else factor_cols[0]
    factor = st.selectbox(
        "Factor to inspect", factor_cols, index=factor_cols.index(default_factor),
    )

    raw = merged[factor].dropna()
    zs = merged_z[factor]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Raw mean", f"{raw.mean():.4f}")
    c2.metric("Raw std", f"{raw.std():.4f}")
    c3.metric("Z-score mean", f"{zs.mean():.4f}")
    c4.metric("Z-score std", f"{zs.std():.4f}")

    b_lo, b_hi = raw.quantile([0.01, 0.99])
    a_lo, a_hi = zs.quantile([0.01, 0.99])
    hist_df = pd.concat([
        pd.DataFrame({"value": raw.clip(b_lo, b_hi), "stage": "Before (raw)"}),
        pd.DataFrame({"value": zs.clip(a_lo, a_hi),  "stage": "After (z-score)"}),
    ])
    fig = px.histogram(
        hist_df, x="value", facet_col="stage", nbins=60,
        color="stage",
        color_discrete_map={"Before (raw)": "#6C8EBF", "After (z-score)": "#D6B656"},
    )
    fig.update_xaxes(matches=None, showticklabels=True)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sample of z-scored panel")
    preview_cols = ["permno", "RETURN_YEAR", "FF49", "RET_ANNUAL", factor]
    st.dataframe(merged_z[preview_cols].head(200), use_container_width=True, height=320)
    st.caption(f"Full z-scored panel: {len(merged_z):,} rows × {merged_z.shape[1]} columns.")

# =======================================================================
# TAB 3 — IN-SAMPLE FAMA-MACBETH
# =======================================================================
with tab_fm:
    if fm_df is None:
        st.info("No FM results on disk yet. Re-run `python project/main.py`.")
    else:
        st.caption(
            f"Yearly cross-sectional OLS of RET_ANNUAL on every z-scored factor, "
            f"aggregated over {stats.get('fm_in_sample_start','?')}–"
            f"{stats.get('fm_in_sample_end','?')}. "
            f"{stats.get('fm_factors_analyzed', len(fm_df))} factors analyzed; "
            f"{stats.get('fm_factors_above_threshold','?')} pass |t| ≥ "
            f"{stats.get('final_factor_t_threshold', 1.5)}."
        )

        top_n = st.slider("Show top N factors by |t-stat|", 10, 60, 30, step=5)
        top_fm = fm_df.head(top_n).copy()
        top_fm["direction"] = top_fm["t_stat"].apply(lambda x: "long (+)" if x > 0 else "short (−)")
        t_threshold = stats.get("final_factor_t_threshold", 1.5)

        fig = px.bar(
            top_fm.sort_values("t_stat"),
            x="t_stat", y="Factor", orientation="h",
            color="direction",
            color_discrete_map={"long (+)": LONG_COLOR, "short (−)": SHORT_COLOR},
            labels={"t_stat": "FM t-stat", "Factor": "Factor"},
        )
        fig.add_vline(x=t_threshold, line_dash="dot", line_color="gray")
        fig.add_vline(x=-t_threshold, line_dash="dot", line_color="gray")
        fig.update_layout(
            height=max(400, 22 * top_n), margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            top_fm[["Factor", "Avg_Premium", "t_stat", "N_Years"]]
            .style.format({"Avg_Premium": "{:.4f}", "t_stat": "{:.2f}"}),
            use_container_width=True,
        )

# =======================================================================
# TAB 4 — FINAL FACTORS + per-factor history
# =======================================================================
with tab_final:
    if final_factors is None:
        st.info("No selected-factors file yet. Re-run `python project/main.py`.")
    else:
        n_long = int((final_factors["t_stat"] > 0).sum())
        n_short = int((final_factors["t_stat"] < 0).sum())
        avg_abs_t = float(final_factors["t_stat"].abs().mean())

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total factors", len(final_factors))
        c2.metric("Long legs", n_long)
        c3.metric("Short legs", n_short)
        c4.metric("Avg |t|", f"{avg_abs_t:.2f}")

        st.caption(
            f"Selection: keep factors with |t| ≥ "
            f"{stats.get('final_factor_t_threshold', 1.5)}, then take the top "
            f"{len(final_factors)} by |t|."
        )

        sel = final_factors.copy()
        sel["direction"] = sel["t_stat"].apply(lambda x: "long (+)" if x > 0 else "short (−)")

        fig = px.bar(
            sel.sort_values("t_stat"),
            x="t_stat", y="Factor", orientation="h",
            color="direction",
            color_discrete_map={"long (+)": LONG_COLOR, "short (−)": SHORT_COLOR},
            labels={"t_stat": "FM t-stat", "Factor": "Factor"},
        )
        fig.add_vline(x=0, line_color="gray")
        fig.update_layout(
            height=max(400, 22 * len(sel)), margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            sel[["Factor", "Avg_Premium", "t_stat", "N_Years", "direction"]]
            .style.format({"Avg_Premium": "{:.4f}", "t_stat": "{:.2f}"}),
            use_container_width=True,
        )

        st.subheader("Per-factor in-sample premium path")
        if fm_yearly is None:
            st.info("Yearly FM coefficients not found — re-run the pipeline to generate `fm_yearly_coefs.csv`.")
        else:
            chosen = st.multiselect(
                "Pick factors to plot",
                options=final_factors["Factor"].tolist(),
                default=final_factors["Factor"].head(5).tolist(),
            )
            if chosen:
                sub = fm_yearly[fm_yearly["Factor"].isin(chosen)].copy()
                sub = sub.sort_values(["Factor", "Year"])
                sub["CumCoef"] = sub.groupby("Factor")["Coef"].cumsum()

                c1, c2 = st.columns(2)
                with c1:
                    fig = px.line(
                        sub, x="Year", y="Coef", color="Factor", markers=True,
                        labels={"Coef": "Yearly FM premium"},
                    )
                    fig.add_hline(y=0, line_dash="dot", line_color="gray")
                    fig.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10))
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("Per-year cross-sectional regression coefficient.")
                with c2:
                    fig = px.line(
                        sub, x="Year", y="CumCoef", color="Factor",
                        labels={"CumCoef": "Cumulative sum of premia"},
                    )
                    fig.add_hline(y=0, line_dash="dot", line_color="gray")
                    fig.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10))
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("Cumulative sum — a rough 'equity curve' for each factor's in-sample signal.")

# =======================================================================
# TAB 5 — OUT-OF-SAMPLE
# =======================================================================
with tab_oos:
    if oos_scores is None or oos_yearly_deciles is None or oos_monthly is None:
        st.info("No OOS outputs yet. Re-run `python project/main.py`.")
    else:
        st.caption(
            f"OOS window: {stats.get('oos_start','?')}–{stats.get('oos_end','?')} "
            f"({stats.get('oos_rows', len(oos_scores)):,} stock-years, "
            f"{stats.get('oos_n_months', 0)} monthly obs). "
            f"Each year, stocks are sorted into deciles by composite score; "
            f"D10 is the top decile (long), D1 the bottom (short). "
            f"L/S = D10 − D1 is a self-financing monthly series."
        )

        # ---- Performance metric cards ----
        st.subheader("Headline performance (monthly L/S, regressed on factors)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "Raw return (ann.)",
            f"{stats.get('oos_raw_mean_annualized', 0):.2%}",
            delta=f"σ {stats.get('oos_raw_std_annualized', 0):.2%}",
        )
        c2.metric(
            "CAPM α (ann.)",
            f"{stats.get('oos_alpha_capm_annualized', 0):.2%}",
            delta=f"t={stats.get('oos_t_alpha_capm', 0):.2f}",
        )
        c3.metric(
            "4-Factor α (ann.)",
            f"{stats.get('oos_alpha_4f_annualized', 0):.2%}",
            delta=f"t={stats.get('oos_t_alpha_4f', 0):.2f}",
        )
        c4.metric(
            "Information Ratio",
            f"{stats.get('oos_information_ratio', 0):.2f}",
            delta=f"R²(4F)={stats.get('oos_r2_4f', 0):.2f}",
        )

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Sharpe (ann.)", f"{stats.get('oos_sharpe_annualized', 0):.2f}")
        c6.metric("Hit rate (monthly)", f"{stats.get('oos_hit_rate_monthly', 0):.1%}")
        c7.metric("Cumulative total", f"{stats.get('oos_cumulative_total', 0):.1%}")
        c8.metric("β_MKT (CAPM)", f"{stats.get('oos_beta_capm_mkt', 0):.2f}")

        # ---- Monthly L/S cumulative curve ----
        st.subheader("Monthly L/S cumulative return")
        mp = oos_monthly.copy()
        mp["YYYYMM"] = mp["YYYYMM"].astype(int)
        mp["date"] = pd.to_datetime(mp["YYYYMM"].astype(str), format="%Y%m")
        fig = px.line(
            mp, x="date", y="cumulative_ls",
            labels={"date": "Month", "cumulative_ls": "Cumulative L/S return"},
        )
        fig.add_hline(y=0, line_dash="dot", line_color="gray")
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

        # ---- Regression coefficient tables ----
        st.subheader("Factor regressions (monthly L/S as dependent)")
        reg_c1, reg_c2 = st.columns(2)
        with reg_c1:
            st.markdown("**CAPM: L/S ~ α + β·MKT_RF**")
            if oos_capm is not None:
                st.dataframe(
                    oos_capm.style.format(
                        {"coef": "{:.4f}", "t_stat": "{:.2f}", "p_value": "{:.3f}"}
                    ),
                    use_container_width=True, hide_index=True,
                )
                st.caption(f"R² = {stats.get('oos_r2_capm', 0):.3f}")
        with reg_c2:
            st.markdown("**4-Factor: L/S ~ α + β_MKT + β_SMB + β_HML + β_UMD**")
            if oos_4f is not None:
                st.dataframe(
                    oos_4f.style.format(
                        {"coef": "{:.4f}", "t_stat": "{:.2f}", "p_value": "{:.3f}"}
                    ),
                    use_container_width=True, hide_index=True,
                )
                st.caption(f"R² = {stats.get('oos_r2_4f', 0):.3f}")

        # ---- Yearly returns by decile ----
        st.subheader("Yearly returns by decile")
        bucket_long = oos_yearly_deciles.melt(
            id_vars="RETURN_YEAR", var_name="decile", value_name="avg_return"
        )
        bucket_long["RETURN_YEAR"] = bucket_long["RETURN_YEAR"].astype(int)
        decile_order = sorted(bucket_long["decile"].unique(), key=lambda s: int(s[1:]))
        bucket_long["decile"] = pd.Categorical(bucket_long["decile"], decile_order, ordered=True)
        fig = px.bar(
            bucket_long, x="RETURN_YEAR", y="avg_return", color="decile", barmode="group",
            labels={"RETURN_YEAR": "Return year", "avg_return": "Equal-weight mean return"},
            category_orders={"decile": decile_order},
        )
        fig.add_hline(y=0, line_dash="dot", line_color="gray")
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

        c9, c10 = st.columns(2)
        with c9:
            st.subheader("Avg return per decile (all years)")
            avg_by_dec = bucket_long.groupby("decile", observed=True)["avg_return"].mean().reset_index()
            avg_by_dec["decile"] = pd.Categorical(avg_by_dec["decile"], decile_order, ordered=True)
            avg_by_dec = avg_by_dec.sort_values("decile")
            fig = px.bar(
                avg_by_dec, x="decile", y="avg_return",
                labels={"decile": "Decile", "avg_return": "Mean annual return"},
                color="avg_return", color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
            )
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Monotonicity D1 → D10 is the key sanity check.")

        with c10:
            st.subheader("Composite score distribution")
            sample = oos_scores.sample(min(len(oos_scores), 20000), random_state=1)
            fig = px.histogram(sample, x="SCORE", nbins=60)
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                f"Score range: {oos_scores['SCORE'].min():.1f} to "
                f"{oos_scores['SCORE'].max():.1f}."
            )

        if oos_ic is not None:
            st.subheader("Information coefficient over time (annual)")
            ic = oos_ic.copy()
            ic["RETURN_YEAR"] = ic["RETURN_YEAR"].astype(int)
            fig = px.bar(
                ic, x="RETURN_YEAR", y="ic",
                labels={"RETURN_YEAR": "Return year", "ic": "Pearson IC"},
                color="ic", color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
            )
            fig.add_hline(y=stats.get("oos_ic_mean", 0), line_dash="dash", line_color="black")
            fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                f"Mean IC = {stats.get('oos_ic_mean', 0):.3f} "
                f"(t = {stats.get('oos_ic_t_stat', 0):.2f}), "
                f"hit rate = {stats.get('oos_ic_hit_rate', 0):.1%}. "
                "Dashed line is the OOS mean IC."
            )

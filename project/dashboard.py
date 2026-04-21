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
MERGED_PATH = os.path.join(CLEANED_DIR, "cleaned_merged_data.csv")
ZSCORED_PATH = os.path.join(CLEANED_DIR, "cleaned_merged_zscored.csv")
FF_PATH = os.path.join(CLEANED_DIR, "ff_factors_clean.csv")
STATS_PATH = os.path.join(CLEANED_DIR, "pipeline_stats.json")
FM_PATH = os.path.join(CLEANED_DIR, "fm_regression_results.csv")
FINAL_PATH = os.path.join(CLEANED_DIR, "final_factors.csv")

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


@st.cache_data(show_spinner="Loading pipeline outputs...")
def load_outputs():
    for p in [MERGED_PATH, ZSCORED_PATH, STATS_PATH]:
        if not os.path.exists(p):
            return None
    merged = pd.read_csv(MERGED_PATH, low_memory=False)
    merged_z = pd.read_csv(ZSCORED_PATH, low_memory=False)
    with open(STATS_PATH) as f:
        stats = json.load(f)
    fm_df = pd.read_csv(FM_PATH) if os.path.exists(FM_PATH) else None
    final_factors = pd.read_csv(FINAL_PATH) if os.path.exists(FINAL_PATH) else None
    return merged, merged_z, stats, fm_df, final_factors


st.set_page_config(page_title="Backtesting Pipeline Dashboard", layout="wide")
st.title("Backtesting Pipeline Dashboard")
st.caption(
    "End-to-end view of the panel: raw CRSP → cleaned + merged → z-scored → "
    "Fama-MacBeth premia → final factor selection."
)

outputs = load_outputs()
if outputs is None:
    st.error(
        "Pipeline outputs not found. Run `python project/main.py` first to "
        f"generate files in `{CLEANED_DIR}/`."
    )
    st.stop()

merged, merged_z, stats, fm_df, final_factors = outputs
factor_cols = [c for c in merged.columns if c not in ID_COLS]

# ---------- Top-level headline metrics ----------
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("CRSP rows (cleaned)", f"{stats['crsp_cleaned_rows']:,}")
m2.metric("Stock-years", f"{stats['annual_return_stock_years']:,}")
m3.metric("Merged panel", f"{stats['merged_rows']:,}")
m4.metric("FM factors analyzed", f"{stats.get('fm_factors_analyzed', '–')}")
m5.metric("Final model size", f"{stats.get('final_factor_count', '–')}")
st.caption(
    f"CRSP year range: {stats['crsp_year_min']}–{stats['crsp_year_max']}. "
    f"FF monthly factors: {stats['ff_all_months']:,} months. "
    f"In-sample FM window: {stats.get('fm_in_sample_start','?')}–{stats.get('fm_in_sample_end','?')}."
)

tab_overview, tab_data, tab_zscore, tab_fm, tab_final = st.tabs(
    ["Overview", "Data quality", "Z-scoring", "Fama-MacBeth", "Final model"]
)

# =======================================================================
# TAB 1 — OVERVIEW
# =======================================================================
with tab_overview:
    st.subheader("Panel composition over time")
    per_year = merged.groupby("RETURN_YEAR")["permno"].nunique().reset_index()
    per_year.columns = ["RETURN_YEAR", "num_stocks"]
    fig = px.line(
        per_year, x="RETURN_YEAR", y="num_stocks", markers=True,
        labels={"RETURN_YEAR": "Return year", "num_stocks": "Unique stocks"},
    )
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Industry distribution (FF49)")
    ind = (
        merged["FF49"].astype("int64").value_counts().sort_values(ascending=True).reset_index()
    )
    ind.columns = ["FF49", "rows"]
    ind["industry"] = ind["FF49"].map(FF49_NAMES).fillna(ind["FF49"].astype(str))
    ind["label"] = ind["FF49"].astype(str) + " — " + ind["industry"]
    fig = px.bar(
        ind, x="rows", y="label", orientation="h",
        labels={"rows": "Stock-year observations", "label": "FF49 industry"},
    )
    fig.update_layout(height=900, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

# =======================================================================
# TAB 2 — DATA QUALITY
# =======================================================================
with tab_data:
    st.subheader("Annual return distribution by year")
    st.caption("Box plots are clipped to the 1st–99th percentile for readability.")
    ret_df = merged[["RETURN_YEAR", "RET_ANNUAL"]].copy()
    lo, hi = ret_df["RET_ANNUAL"].quantile([0.01, 0.99])
    ret_df["RET_ANNUAL"] = ret_df["RET_ANNUAL"].clip(lo, hi)
    fig = px.box(
        ret_df, x="RETURN_YEAR", y="RET_ANNUAL", points=False,
        labels={"RETURN_YEAR": "Return year", "RET_ANNUAL": "Annual return"},
    )
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Factor missingness (before z-scoring)")
    st.caption(
        "Top 30 factors by % missing. Z-scoring fills NaN with 0, so these rows "
        "are effectively treated as neutral after the next step."
    )
    miss = (
        merged[factor_cols].isna().sum() / len(merged) * 100
    ).sort_values(ascending=False).head(30).reset_index()
    miss.columns = ["factor", "pct_missing"]
    fig = px.bar(
        miss.iloc[::-1], x="pct_missing", y="factor", orientation="h",
        labels={"pct_missing": "% missing", "factor": "Factor"},
    )
    fig.update_layout(height=700, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

# =======================================================================
# TAB 3 — Z-SCORING
# =======================================================================
with tab_zscore:
    st.caption(
        "Within each (FF49 industry × return year) bucket, each factor is "
        "transformed to mean 0 / std 1. Remaining NaNs are set to 0 (neutral)."
    )

    default_factor = "BM" if "BM" in factor_cols else factor_cols[0]
    factor = st.selectbox(
        "Factor to inspect", factor_cols,
        index=factor_cols.index(default_factor),
    )

    m_before_raw = merged[factor].dropna()
    m_after = merged_z[factor]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Before: mean", f"{m_before_raw.mean():.4f}")
    c2.metric("Before: std",  f"{m_before_raw.std():.4f}")
    c3.metric("After: mean",  f"{m_after.mean():.4f}")
    c4.metric("After: std",   f"{m_after.std():.4f}")

    # Side-by-side histograms using facets so the two scales don't collide.
    b_lo, b_hi = m_before_raw.quantile([0.01, 0.99])
    a_lo, a_hi = m_after.quantile([0.01, 0.99])
    hist_df = pd.concat([
        pd.DataFrame({"value": m_before_raw.clip(b_lo, b_hi), "stage": "Before (raw)"}),
        pd.DataFrame({"value": m_after.clip(a_lo, a_hi),     "stage": "After (z-score)"}),
    ])
    fig = px.histogram(
        hist_df, x="value", facet_col="stage", nbins=60,
        color="stage",
        color_discrete_map={"Before (raw)": "#6C8EBF", "After (z-score)": "#D6B656"},
    )
    fig.update_xaxes(matches=None, showticklabels=True)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_layout(
        height=380, margin=dict(l=10, r=10, t=40, b=10), showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Each distribution is clipped to its own 1st–99th percentile for display, "
        "so the before panel isn't crushed by outliers and the after panel isn't clipped to them."
    )

# =======================================================================
# TAB 4 — FAMA-MACBETH
# =======================================================================
with tab_fm:
    if fm_df is None:
        st.info("No FM results on disk yet. Re-run `python project/main.py`.")
    else:
        st.caption(
            f"Yearly cross-sectional OLS of RET_ANNUAL on every z-scored factor, "
            f"aggregated over {stats.get('fm_in_sample_start', '?')}–"
            f"{stats.get('fm_in_sample_end', '?')}. "
            f"{stats.get('fm_factors_analyzed', len(fm_df))} factors analyzed; "
            f"{stats.get('fm_factors_above_threshold', '?')} pass |t| ≥ "
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
            color_discrete_map={"long (+)": "#2E8B57", "short (−)": "#B22222"},
            labels={"t_stat": "FM t-stat", "Factor": "Factor"},
        )
        fig.add_vline(x=t_threshold, line_dash="dot", line_color="gray")
        fig.add_vline(x=-t_threshold, line_dash="dot", line_color="gray")
        fig.update_layout(
            height=max(400, 22 * top_n), margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Per-factor statistics")
        st.dataframe(
            top_fm[["Factor", "Avg_Premium", "t_stat", "N_Years"]]
            .style.format({"Avg_Premium": "{:.4f}", "t_stat": "{:.2f}"}),
            use_container_width=True,
        )

# =======================================================================
# TAB 5 — FINAL MODEL
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
            f"Selection rule: keep factors with |t| ≥ "
            f"{stats.get('final_factor_t_threshold', 1.5)}, then take the top "
            f"{len(final_factors)} by |t|."
        )

        sel = final_factors.copy()
        sel["direction"] = sel["t_stat"].apply(lambda x: "long (+)" if x > 0 else "short (−)")

        fig = px.bar(
            sel.sort_values("t_stat"),
            x="t_stat", y="Factor", orientation="h",
            color="direction",
            color_discrete_map={"long (+)": "#2E8B57", "short (−)": "#B22222"},
            labels={"t_stat": "FM t-stat", "Factor": "Factor"},
        )
        fig.add_vline(x=0, line_color="gray")
        fig.update_layout(
            height=max(400, 22 * len(sel)), margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Selected factor table")
        st.dataframe(
            sel[["Factor", "Avg_Premium", "t_stat", "N_Years", "direction"]]
            .style.format({"Avg_Premium": "{:.4f}", "t_stat": "{:.2f}"}),
            use_container_width=True,
        )

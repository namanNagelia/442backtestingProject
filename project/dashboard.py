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

ID_COLS = [
    "permno", "yyyymm", "YEAR_x", "MONTH", "RETURN_YEAR", "PERMNO_x",
    "YEAR_y", "RET_ANNUAL", "PERMNO_y", "YEAR", "PRC_ABS", "MARKET_CAP",
    "SICCD", "FF49",
]


@st.cache_data(show_spinner="Loading pipeline outputs...")
def load_outputs():
    for p in [MERGED_PATH, ZSCORED_PATH, STATS_PATH]:
        if not os.path.exists(p):
            return None
    merged = pd.read_csv(MERGED_PATH, low_memory=False)
    merged_z = pd.read_csv(ZSCORED_PATH, low_memory=False)
    with open(STATS_PATH) as f:
        stats = json.load(f)
    return merged, merged_z, stats


st.set_page_config(page_title="Backtesting Pipeline Dashboard", layout="wide")
st.title("Backtesting Pipeline Dashboard")
st.caption(
    "Summary of the cleaned CRSP panel, OAP factor coverage, and z-scored factors.")

outputs = load_outputs()
if outputs is None:
    st.error(
        "Pipeline outputs not found. Run `python project/main.py` first to generate "
        f"files in `{CLEANED_DIR}/`."
    )
    st.stop()

merged, merged_z, stats = outputs
factor_cols = [c for c in merged.columns if c not in ID_COLS]

st.header("1. Pipeline summary")
col1, col2, col3, col4 = st.columns(4)
col1.metric("CRSP rows (cleaned)", f"{stats['crsp_cleaned_rows']:,}")
col2.metric("Stock-years (annual returns)",
            f"{stats['annual_return_stock_years']:,}")
col3.metric("Merged panel rows", f"{stats['merged_rows']:,}")
col4.metric("Z-scored panel rows", f"{stats['merged_zscored_rows']:,}")
st.caption(
    f"Year range in cleaned CRSP: {stats['crsp_year_min']} – {stats['crsp_year_max']}. "
    f"FF monthly factors: {stats['ff_all_months']:,} months."
)

st.header("2. Panel composition")
left, right = st.columns(2)

with left:
    st.subheader("Stocks per return year")
    per_year = merged.groupby("RETURN_YEAR")["permno"].nunique().reset_index()
    per_year.columns = ["RETURN_YEAR", "num_stocks"]
    fig = px.line(per_year, x="RETURN_YEAR", y="num_stocks", markers=True)
    fig.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("FF49 industry distribution (merged panel)")
    ind = merged["FF49"].value_counts().sort_index().reset_index()
    ind.columns = ["FF49", "rows"]
    fig = px.bar(ind, x="FF49", y="rows")
    fig.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

st.header("3. Factor missingness")
st.caption(
    "Top 30 factors by % missing BEFORE z-scoring (z-scoring fills NaN with 0).")
miss = (
    merged[factor_cols].isna().sum() / len(merged) * 100
).sort_values(ascending=False).head(30).reset_index()
miss.columns = ["factor", "pct_missing"]
fig = px.bar(miss.iloc[::-1], x="pct_missing", y="factor", orientation="h")
fig.update_layout(height=700, margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(fig, use_container_width=True)

st.header("4. Z-score sanity checks")
st.caption(
    "After industry-year z-scoring, every factor should be roughly mean≈0 and std≈1 "
    "within the non-zero-filled subset. Pick a factor to see its distribution before/after."
)

default_factor = "BM" if "BM" in factor_cols else factor_cols[0]
factor = st.selectbox("Factor", factor_cols,
                      index=factor_cols.index(default_factor))

m_before = merged[factor].dropna()
m_after = merged_z[factor]
c1, c2, c3 = st.columns(3)
c1.metric("Before: mean", f"{m_before.mean():.4f}")
c2.metric("After: mean",  f"{m_after.mean():.4f}")
c3.metric("After: std",   f"{m_after.std():.4f}")

hist_df = pd.concat([
    pd.DataFrame({"value": m_before, "stage": "before"}),
    pd.DataFrame({"value": m_after,  "stage": "after"}),
])
q01, q99 = hist_df["value"].quantile([0.01, 0.99])
hist_df["value"] = hist_df["value"].clip(q01, q99)
fig = px.histogram(hist_df, x="value", color="stage",
                   barmode="overlay", nbins=80, opacity=0.6)
fig.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(fig, use_container_width=True)

st.header("5. Annual return distribution by year")
ret_df = merged[["RETURN_YEAR", "RET_ANNUAL"]].copy()
lo, hi = ret_df["RET_ANNUAL"].quantile([0.01, 0.99])
ret_df["RET_ANNUAL"] = ret_df["RET_ANNUAL"].clip(lo, hi)
fig = px.box(ret_df, x="RETURN_YEAR", y="RET_ANNUAL", points=False)
fig.update_layout(height=450, margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Returns clipped to 1st–99th percentile for display. Underlying CSV is unclipped."
)

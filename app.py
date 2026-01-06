import streamlit as st
import pandas as pd
from pathlib import Path

# -----------------------------
# Paths to data
# -----------------------------
RAW_PATH = Path("data/telco_churn.csv")
SCORED_PATH = Path("outputs/scored/scored_test_calibrated.csv")
OP_PATH = Path("outputs/tables/threshold_compare.csv")   # operating points table

# -----------------------------
# Cached loaders
# -----------------------------
@st.cache_data
def load_data():
    df_raw = pd.read_csv(RAW_PATH)
    df_scored = pd.read_csv(SCORED_PATH)
    return df_raw, df_scored

@st.cache_data
def load_operating_points():
    """Load threshold comparison table if available."""
    if OP_PATH.exists():
        return pd.read_csv(OP_PATH)
    return None

# -----------------------------
# App layout
# -----------------------------
st.set_page_config(page_title="Telco Churn – Demo", layout="wide")

st.title("Telco Customer Churn – Model Demo")

df_raw, df_scored = load_data()

# ---- Section 1: Overview tables ----
st.subheader("Input data (first 10 rows)")
st.dataframe(df_raw.head(10), use_container_width=True)

st.subheader("Scored output (first 10 rows)")
st.dataframe(df_scored.head(10), use_container_width=True)

st.markdown("---")

# ---- Section 1.5: Model operating points (optional) ----
op_df = load_operating_points()
if op_df is not None:
    st.subheader("Model Operating Points")
    cols_to_show = [
        "Operating Point",
        "Threshold",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "Profit",
    ]
    # keep only columns that actually exist in the CSV
    cols_to_show = [c for c in cols_to_show if c in op_df.columns]

    st.dataframe(op_df[cols_to_show], use_container_width=True)
else:
    st.info(
        "Threshold comparison table not found at "
        "`outputs/tables/threshold_compare.csv`."
    )

st.markdown("---")

# ---- Section 2: Churn-risk explorer ----
st.subheader("Churn-Risk Explorer")

# 1) Choose threshold
st.caption("Select a probability threshold to flag customers as churn-risk.")
threshold = st.slider(
    "Churn probability threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.06,        # your profit-optimal threshold
    step=0.01,
)

# 2) Work on a copy of scored data
df_view = df_scored.copy()

# Try to automatically detect the probability column
CANDIDATE_PROBA_COLS = [
    "proba",
    "y_proba",
    "y_proba_calibrated",
    "y_proba_xgb_calibrated",
    "churn_proba",
    "pred_proba",
]

proba_col = None
for c in CANDIDATE_PROBA_COLS:
    if c in df_view.columns:
        proba_col = c
        break

# If still not found, look for any column name containing "proba" or "prob"
if proba_col is None:
    for c in df_view.columns:
        if "proba" in c.lower() or "prob" in c.lower():
            proba_col = c
            break

if proba_col is None:
    st.error(
        "Could not find a probability column in scored_test_calibrated.csv. "
        "Available columns are:\n\n" + ", ".join(df_view.columns)
    )
else:
    # Binary prediction using chosen threshold
    df_view["churn_proba"] = df_view[proba_col]
    df_view["churn_flag"] = (df_view["churn_proba"] >= threshold).map(
        {True: "Churn risk", False: "Safe"}
    )

    st.write(f"Using column **{proba_col}** as churn probability.")

    # Overall % flagged
    flagged_mask = df_view["churn_flag"] == "Churn risk"
    flagged_n = flagged_mask.sum()
    total_n = len(df_view)
    flagged_pct = flagged_n / total_n * 100

    st.write(
        f"At threshold **{threshold:.2f}**, "
        f"**{flagged_n} / {total_n}** customers "
        f"({flagged_pct:.1f}%) are flagged as **churn-risk**."
    )

    # 3) Show a compact table with key columns, sorted by risk (top 25)
    cols_to_show = []
    for c in ["customerID", "Churn", "y_true"]:
        if c in df_view.columns:
            cols_to_show.append(c)

    cols_to_show += ["churn_proba", "churn_flag"]

    df_top = (
        df_view[cols_to_show]
        .sort_values("churn_proba", ascending=False)
        .head(25)
    )

    st.write("Top customers ranked by churn risk:")
    st.dataframe(df_top, use_container_width=True)

    # ---- NEW: Full retention list download ----
    st.markdown("### Download Retention List")

    retention_df = (
        df_view[flagged_mask][cols_to_show]
        .sort_values("churn_proba", ascending=False)
    )

    st.write(
        f"Retention list contains **{len(retention_df)}** customers "
        f"to target for offers / calls."
    )

    csv_bytes = retention_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download churn-risk customers as CSV",
        data=csv_bytes,
        file_name=f"churn_retention_list_thr_{threshold:.2f}.csv",
        mime="text/csv",
    )

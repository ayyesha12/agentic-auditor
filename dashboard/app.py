import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import sqlite3

DB_PATH = "logs/runs.db"
DRIFT_REPORT_PATH = "logs/drift_report.csv"

st.set_page_config(
    page_title="Agent Auditor Dashboard",
    page_icon="🔍",
    layout="wide"
)

st.title("Agent Auditor — Eval Dashboard")
st.caption("Multi-agent LangGraph pipeline evaluation framework")

# ── Load data ──────────────────────────────────────────────
@st.cache_data(ttl=30)
def load_runs():
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM runs", conn)
    conn.close()
    return df

@st.cache_data(ttl=30)
def load_drift():
    if not os.path.exists(DRIFT_REPORT_PATH):
        return pd.DataFrame()
    return pd.read_csv(DRIFT_REPORT_PATH)

df_runs = load_runs()
df_drift = load_drift()

if df_runs.empty:
    st.warning("No eval runs found. Run `python -m eval.run_eval --version v1` first.")
    st.stop()

# ── Sidebar — version filter ────────────────────────────────
st.sidebar.header("Filters")
versions = sorted(df_runs["version"].unique().tolist())
selected_versions = st.sidebar.multiselect(
    "Filter by version",
    options=versions,
    default=versions
)

if not selected_versions:
    st.warning("Select at least one version from the sidebar.")
    st.stop()

df_filtered = df_runs[df_runs["version"].isin(selected_versions)]

# ── Top metrics ─────────────────────────────────────────────
st.subheader("Overall Metrics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    rule_rate = df_filtered["rule_pass"].mean() * 100
    st.metric("Rule-based pass rate", f"{rule_rate:.1f}%")

with col2:
    judge_rate = df_filtered["judge_pass"].mean() * 100
    st.metric("Judge pass rate", f"{judge_rate:.1f}%")

with col3:
    hit_rate = df_filtered["retrieval_hit"].mean() * 100
    st.metric("Retrieval hit rate", f"{hit_rate:.1f}%")

with col4:
    insuf_rate = df_filtered["insufficient_context"].mean() * 100
    st.metric("Insufficient context rate", f"{insuf_rate:.1f}%")

st.divider()

# ── Version comparison chart ─────────────────────────────────
st.subheader("Pass rates by version")
st.caption("Shows how rule-based and judge pass rates changed across prompt versions")

version_summary = (
    df_filtered
    .groupby("version")[["rule_pass", "judge_pass", "retrieval_hit"]]
    .mean()
    .mul(100)
    .round(1)
)
version_summary.columns = ["Rule Pass %", "Judge Pass %", "Retrieval Hit %"]
st.line_chart(version_summary)

# ── Data-centric experiment table ────────────────────────────
st.subheader("Data-centric experiment — version comparison")
st.caption("v1 = baseline no few-shot | v2 = concise few-shot Planner prompt")

exp_table = (
    df_runs.groupby("version").agg(
        Total=("id", "count"),
        Rule_Pass=("rule_pass", "sum"),
        Judge_Pass=("judge_pass", "sum"),
        Rule_Pct=("rule_pass", lambda x: f"{x.mean()*100:.1f}%"),
        Judge_Pct=("judge_pass", lambda x: f"{x.mean()*100:.1f}%"),
        Retrieval_Pct=("retrieval_hit", lambda x: f"{x.mean()*100:.1f}%")
    ).reset_index()
)
st.dataframe(exp_table, use_container_width=True)

st.divider()

# ── Rule vs Judge disagreement ────────────────────────────────
st.subheader("Rule vs Judge agreement")
st.caption("High disagreement = model answers correctly but keywords don't match")

agree = (df_filtered["rule_pass"] == df_filtered["judge_pass"]).mean() * 100
disagree = 100 - agree
col_a, col_b = st.columns(2)
col_a.metric("Agreement rate", f"{agree:.1f}%")
col_b.metric("Disagreement rate", f"{disagree:.1f}%")

st.divider()

# ── Anomalous runs ────────────────────────────────────────────
st.subheader("Anomalous runs — behavioral drift")

if df_drift.empty:
    st.info("Run `python eval/drift_analysis.py` to generate drift data.")
else:
    df_drift_filtered = df_drift[
        df_drift["version"].isin(selected_versions)
    ] if "version" in df_drift.columns else df_drift

    anomalous = df_drift_filtered[df_drift_filtered["anomaly"] == -1]
    st.write(f"**{len(anomalous)}** anomalous runs out of "
             f"{len(df_drift_filtered)} total")

    if not anomalous.empty:
        display_cols = [
            "id", "version", "question", "answer",
            "rule_pass", "judge_pass",
            "latency_seconds", "anomaly_reason"
        ]
        available = [c for c in display_cols if c in anomalous.columns]
        st.dataframe(
            anomalous[available].reset_index(drop=True),
            use_container_width=True
        )
    else:
        st.success("No anomalous runs detected in selected versions.")

st.divider()

# ── Full run log ───────────────────────────────────────────────
with st.expander("Full run log"):
    st.dataframe(df_filtered.reset_index(drop=True), use_container_width=True)
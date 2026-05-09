"""
Streamlit Dashboard  –  Drowsiness Detection History Viewer
Run: streamlit run streamlit_dashboard.py
"""

import streamlit as st
import json
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

LOG_PATH       = "drowsiness_log.json"
SCREENSHOT_DIR = "screenshots"

st.set_page_config(
    page_title="Drowsiness Monitor Dashboard",
    page_icon="🚗",
    layout="wide",
)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='text-align:center; color:#00C8FF;'>
    🚗 AI Driver Drowsiness Detection — Dashboard
</h1>
""", unsafe_allow_html=True)

# ── Load log ──────────────────────────────────────────────────────────────────
@st.cache_data(ttl=5)
def load_log():
    if not os.path.exists(LOG_PATH):
        return []
    with open(LOG_PATH) as f:
        return json.load(f)

events = load_log()

if not events:
    st.info("No detection events found. Run the main detector first.")
    st.stop()

df = pd.DataFrame(events)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["minute"]    = df["timestamp"].dt.floor("min")

# ── KPI Cards ─────────────────────────────────────────────────────────────────
drowsy_events = df[df["event"] == "DROWSINESS_DETECTED"]
yawn_events   = df[df["event"] == "YAWN_DETECTED"]
awake_events  = df[df["event"] == "DRIVER_AWAKE"]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Events",       len(df))
col2.metric("Drowsiness Alerts",  len(drowsy_events), delta=None)
col3.metric("Yawn Events",        len(yawn_events))
col4.metric("Awake Recoveries",   len(awake_events))

st.divider()

# ── Timeline chart ────────────────────────────────────────────────────────────
st.subheader("📈 Event Timeline")
fig_timeline = px.scatter(
    df,
    x="timestamp",
    y="event",
    color="event",
    color_discrete_map={
        "DROWSINESS_DETECTED": "#FF4444",
        "YAWN_DETECTED":       "#FFB300",
        "DRIVER_AWAKE":        "#00E050",
    },
    title="Detection Events Over Time",
    height=300,
)
fig_timeline.update_traces(marker=dict(size=10))
st.plotly_chart(fig_timeline, use_container_width=True)

# ── EAR trend ─────────────────────────────────────────────────────────────────
if "ear" in df.columns:
    st.subheader("👁️ EAR Trend During Alerts")
    ear_df = drowsy_events[["timestamp", "ear"]].dropna()
    if not ear_df.empty:
        fig_ear = px.line(ear_df, x="timestamp", y="ear", title="EAR at Drowsiness Events")
        fig_ear.add_hline(y=0.25, line_dash="dash", line_color="red", annotation_text="Threshold")
        st.plotly_chart(fig_ear, use_container_width=True)

# ── Hourly heatmap ────────────────────────────────────────────────────────────
st.subheader("⏰ Hourly Drowsiness Frequency")
drowsy_events_copy = drowsy_events.copy()
drowsy_events_copy["hour"] = drowsy_events_copy["timestamp"].dt.hour
hourly = drowsy_events_copy.groupby("hour").size().reset_index(name="count")
fig_hour = px.bar(hourly, x="hour", y="count", labels={"hour": "Hour of Day", "count": "Alerts"},
                  color="count", color_continuous_scale="Reds")
st.plotly_chart(fig_hour, use_container_width=True)

# ── Screenshots gallery ───────────────────────────────────────────────────────
st.subheader("📸 Saved Screenshots")
if os.path.exists(SCREENSHOT_DIR):
    images = sorted(
        [f for f in os.listdir(SCREENSHOT_DIR) if f.endswith(".jpg")],
        reverse=True
    )[:12]
    if images:
        cols = st.columns(4)
        for i, img_name in enumerate(images):
            cols[i % 4].image(
                os.path.join(SCREENSHOT_DIR, img_name),
                caption=img_name, use_container_width=True
            )
    else:
        st.info("No screenshots yet.")
else:
    st.info("Screenshots directory not found.")

# ── Raw log table ─────────────────────────────────────────────────────────────
with st.expander("📋 Raw Event Log"):
    st.dataframe(df.sort_values("timestamp", ascending=False), use_container_width=True)

# ── Auto refresh ─────────────────────────────────────────────────────────────
st.caption("Dashboard auto-refreshes every 5 seconds. Run `streamlit run streamlit_dashboard.py`")
if st.button("🔄 Refresh Now"):
    st.cache_data.clear()
    st.rerun()

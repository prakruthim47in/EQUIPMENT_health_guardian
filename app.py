"""
Equipment Health Guardian — a GenAI-powered industrial monitoring dashboard.

Run with: streamlit run app.py

This is a hackathon starter: simulated multi-machine sensor data ->
IsolationForest anomaly detection -> Gemini API turns flagged anomalies
into plain-English diagnoses, and answers technician chat questions.
"""
import os
import streamlit as st
import pandas as pd

from simulator import generate_fleet, MACHINES, FAULT_PROFILES
from detector import detect_anomalies, latest_status
import ai_assistant

st.set_page_config(page_title="Equipment Health Guardian", layout="wide")

# ---------- Sidebar: setup ----------
st.sidebar.title("⚙️ Equipment Health Guardian")
st.sidebar.caption("GenAI-powered predictive maintenance assistant")

api_key = st.sidebar.text_input("Gemini API key", type="password",
                                 help="Get a free key at aistudio.google.com/apikey")
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key
else:
    try:
        if "GOOGLE_API_KEY" in st.secrets:
            os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        pass
    st.write("This is now inside the if block")  # Pushed to the right!
    st.write("This is also inside")              # Pushed to the right!

st.sidebar.markdown("---")  # Back to the left (outside the if block)    

st.sidebar.markdown("---")
st.sidebar.subheader("Simulate a fault (for demo)")
fault_machine = st.sidebar.selectbox("Machine", ["None"] + MACHINES)
fault_type = st.sidebar.selectbox("Fault type", [f for f in FAULT_PROFILES if f != "none"])
n_points = st.sidebar.slider("History length (readings)", 60, 300, 200, step=20)

fault_map = {}
if fault_machine != "None":
    fault_map[fault_machine] = fault_type

# ---------- Generate + analyze data ----------
@st.cache_data(show_spinner=False)
def get_data(fault_map_key, n_points):
    df = generate_fleet(fault_map=dict(fault_map_key), n_points=n_points)
    return detect_anomalies(df)

df = get_data(tuple(fault_map.items()), n_points)
latest = latest_status(df)
ai_assistant.set_fleet_data(df)

# ---------- Main layout ----------
st.title("🏭 Equipment Health Guardian")
st.caption("Real-time simulated sensor monitoring with GenAI-powered diagnosis")

cols = st.columns(len(MACHINES))
for col, (_, row) in zip(cols, latest.iterrows()):
    with col:
        st.metric(row["machine"], row["status"])
        st.caption(f"Vibration {row['vibration_mm_s']} mm/s · Temp {row['temperature_c']}°C · RPM {row['rpm']}")

st.markdown("---")

selected_machine = st.selectbox("Inspect machine", MACHINES)
machine_df = df[df["machine"] == selected_machine]
machine_latest = latest[latest["machine"] == selected_machine].iloc[0]

chart_col, alert_col = st.columns([2, 1])

with chart_col:
    st.subheader(f"Sensor trend — {selected_machine}")
    chart_df = machine_df.set_index("t")[["vibration_mm_s", "temperature_c", "rpm"]]
    st.line_chart(chart_df)

    anomalies = machine_df[machine_df["is_anomaly"]]
    if len(anomalies) > 0:
        st.warning(f"{len(anomalies)} anomalous readings detected in this machine's recent history.")
    else:
        st.success("No anomalies detected — operating within normal range.")

with alert_col:
    st.subheader("AI Diagnosis")
    if not api_key:
        st.info("Enter your Gemini API key in the sidebar to enable AI diagnosis.")
    elif machine_latest["status"] == "✅ Healthy":
        st.write("Machine is healthy — no diagnosis needed.")
    else:
        recent_history = machine_df.tail(15)[["t"] + ["vibration_mm_s", "temperature_c", "rpm"]].to_string(index=False)
        if st.button("Generate diagnosis"):
            with st.spinner("Analyzing sensor data..."):
                diagnosis = ai_assistant.ask_agent(selected_machine, "Please analyze this machine's telemetry and provide a prioritized action plan.")
            st.markdown(diagnosis)

st.markdown("---")
st.subheader(f"💬 Ask the AI agent about {selected_machine}")
question = st.text_input("Your question", placeholder="e.g. Why is this machine flagged? Is it safe to keep running?")
if st.button("Ask") and question:
    if not api_key:
        st.info("Enter your Gemini API key in the sidebar first.")
    else:
        with st.spinner("Agent is thinking..."):
            answer = ai_assistant.ask_agent(selected_machine, question)
        st.markdown(f"**Agent:** {answer}")
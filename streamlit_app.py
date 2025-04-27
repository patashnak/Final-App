
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model_simulation_functions import simulate, extract_metrics
from datetime import datetime
from pathlib import Path

@st.cache_data
def cached_simulate(params, mode, t_max):
    return simulate(params, mode, t_max)

st.set_page_config(page_title="Astrocyte-Neuron Explorer", layout="wide")
st.title("🧠 Astrocyte-Neuron Inflammation Simulator & Data Explorer")

st.sidebar.header("Model Parameters")
alpha = st.sidebar.slider("α – TNF Sensitivity", 0.1, 3.0, 1.5, 0.1)
beta = st.sidebar.slider("β – Calcium Decay Rate", 0.1, 2.0, 0.8, 0.1)
gamma = st.sidebar.slider("γ – Glutamate Scaling", 0.1, 3.0, 1.2, 0.1)
delta = st.sidebar.slider("δ – Neuron Excitability", 0.1, 3.0, 1.0, 0.1)
epsilon = st.sidebar.slider("ε – Baseline Inhibition", 0.0, 2.0, 0.5, 0.1)
eta = st.sidebar.slider("η – Feedback Strength", 0.0, 3.0, 0.0, 0.1)
t_max = st.sidebar.slider("Simulation Duration (s)", 10, 100, 50, 10)

params = {'alpha': alpha, 'beta': beta, 'gamma': gamma, 'delta': delta, 'epsilon': epsilon, 'eta': eta}
input_type = st.sidebar.selectbox("TNF Input Type", ["acute", "chronic"])

t, Ca_astro, F_neuron = cached_simulate(params, input_type, t_max)
metrics = extract_metrics(t, F_neuron)

metrics["timestamp"] = datetime.now().isoformat()
metrics.update(params)
try:
    log_df = pd.DataFrame([metrics])
    log_df.to_csv("simulation_log.csv", mode="a", header=not Path("simulation_log.csv").exists(), index=False)
except Exception as e:
    st.warning(f"Could not log simulation: {e}")

st.subheader("Astrocyte Calcium (Ca²⁺)")
fig1, ax1 = plt.subplots()
ax1.plot(t, Ca_astro, color='skyblue')
st.pyplot(fig1)

st.subheader("Neuron Firing Rate")
fig2, ax2 = plt.subplots()
ax2.plot(t, F_neuron, color='orange')
st.pyplot(fig2)

st.subheader("Output Metrics")
st.metric("Peak Firing", f"{metrics['peak_firing']:.2f} Hz")
st.metric("AUC", f"{metrics['auc_firing']:.2f}")
st.metric("Time to Peak", f"{metrics['time_to_peak']:.2f} s")
st.metric("Duration >1Hz", f"{metrics['firing_duration']:.2f} s")

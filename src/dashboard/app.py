import streamlit as st
import pandas as pd
import json
from pathlib import Path
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

# 📁 Load available LLM reports
report_dir = Path("src/evaluation/data")
llm_reports = sorted(report_dir.glob("report_batch_*.json"))
default_report = report_dir / "report_batch_3.json"

# 🎛️ Sidebar Controls
st.sidebar.title("Configuration")
scoring_mode = st.sidebar.radio("Scoring Engine", ["Traditional", "Groq LLM"])
selected_report = st.sidebar.selectbox("LLM Report File", llm_reports, index=llm_reports.index(default_report) if default_report in llm_reports else 0)
heatmap_mode = st.sidebar.radio("Heatmap Mode", ["LLM", "Traditional", "Compare", "Delta"])

# 📦 Load Reports
report_path_traditional = report_dir / "real_report.json"

def load_data(path):
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

data_traditional = load_data(report_path_traditional)
data_llm = load_data(selected_report)
data = data_traditional if scoring_mode == "Traditional" else data_llm

# 📊 Define scoring dimensions
score_columns = [
    "Instruction-Following",
    "Coherence & Accuracy",
    "Hallucination Detection",
    "Style Matching",
    "Length Penalty",
    "Assumption Control",
    "Final Score"
]

def to_dataframe(data):
    rows = []
    for r in data:
        s = r["scores"]
        if "coherence_&_accuracy" in s:
            s["coherence_accuracy"] = s["coherence_&_accuracy"]
        rows.append({
            "Agent": r["agent_id"],
            "Instruction-Following": s.get("instruction_following", 0.0),
            "Coherence & Accuracy": s.get("coherence_accuracy", 0.0),
            "Hallucination Detection": s.get("hallucination_detection", 0.0),
            "Style Matching": s.get("style_matching", 0.0),
            "Length Penalty": s.get("length_penalty", 0.0),
            "Assumption Control": s.get("assumption_control", 0.0),
            "Final Score": s.get("final", 0.0),
            "Domain": r.get("domain", "Unknown")
        })
    return pd.DataFrame(rows)

df_traditional = to_dataframe(data_traditional)
df_traditional_scaled = df_traditional.copy()
for col in score_columns:
    if col != "Domain":
        df_traditional_scaled[col] = df_traditional_scaled[col] * 10

df_llm = to_dataframe(data_llm)
df = df_traditional_scaled if scoring_mode == "Traditional" else df_llm

# 🎛️ Domain Filter
domains = sorted(set(r.get("domain", "Unknown") for r in data))
selected_domain = st.sidebar.selectbox("Domain", ["All"] + domains)
if selected_domain != "All":
    df = df[df["Domain"] == selected_domain]

# 🎛️ Leaderboard controls
st.sidebar.title("Leaderboard Filters")
sort_by = st.sidebar.selectbox("Sort by", score_columns)
top_n = st.sidebar.slider("Top N agents", 1, len(df), min(10, len(df)))

# 🏆 Leaderboard
st.title("Agentic Evaluation Framework")
st.markdown(f"**Scoring Mode**: `{scoring_mode}` | **Domain**: `{selected_domain}`")
st.markdown("Explore agent performance across multiple scoring dimensions.")

# 🔍 Agent Explanation Block
st.subheader("🔍 Agent Explanations")
selected_agent = st.selectbox("Select an agent", df["Agent"])
agent_data = next((r for r in data if r["agent_id"] == selected_agent), None)

if agent_data:
    if st.checkbox("Show Prompt & Response"):
        st.markdown("### 📝 Prompt")
        st.code(agent_data.get("prompt", "No prompt available."), language="markdown")
        st.markdown("### 💬 Response")
        st.code(agent_data.get("response", "No response available."), language="markdown")

    explanation = agent_data.get("explanations", {})
    scores = agent_data.get("scores", {})
    if "coherence_&_accuracy" in scores:
        scores["coherence_accuracy"] = scores["coherence_&_accuracy"]
        explanation["coherence_accuracy"] = explanation.get("coherence_&_accuracy", "")

    for dim, text in explanation.items():
        st.markdown(f"**{dim.replace('_', ' ').title()}**: {text}")

    core_dims = [
        "instruction_following",
        "coherence_accuracy",
        "hallucination_detection",
        "style_matching",
        "length_penalty",
        "assumption_control"
    ]
    weakest_dim = min(core_dims, key=lambda d: scores.get(d, 10.0))
    weak_score = scores.get(weakest_dim, 0.0)
    weak_expl = explanation.get(weakest_dim, "No explanation available.")

    st.markdown("---")
    st.markdown(f"**Weakest Dimension**: `{weakest_dim.replace('_', ' ').title()}` – {weak_score}/10")
    st.markdown(f"📉 _{weak_expl}_")

# 📈 Leaderboard Table
df_sorted = df.sort_values(by=sort_by, ascending=False).head(top_n)
st.dataframe(df_sorted.drop(columns=["Domain"]), use_container_width=True)

# 📊 Score Distribution Chart
st.subheader("📊 Score Distribution")
st.bar_chart(df.set_index("Agent")[score_columns[:-1]])

# Heatmap Section
st.subheader("Agent vs Dimension Heatmap")

chunk_size = st.sidebar.slider("Agents per heatmap", 10, 50, 25)
start_idx = st.sidebar.slider("Start index", 0, len(df) - chunk_size, 0)
df_chunk_llm = df_llm.iloc[start_idx:start_idx + chunk_size]
df_chunk_traditional_scaled = df_traditional_scaled.iloc[start_idx:start_idx + chunk_size]

if heatmap_mode == "LLM":
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df_chunk_llm.set_index("Agent")[score_columns[:-1]], annot=True, fmt=".1f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

elif heatmap_mode == "Traditional":
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df_chunk_traditional_scaled.set_index("Agent")[score_columns[:-1]], annot=True, fmt=".1f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

elif heatmap_mode == "Compare":
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### LLM Judge")
        fig1, ax1 = plt.subplots(figsize=(6, 6))
        sns.heatmap(df_chunk_llm.set_index("Agent")[score_columns[:-1]], annot=False, cmap="coolwarm", ax=ax1)
        st.pyplot(fig1)
    with col2:
        st.markdown("### 🧮 Traditional Heuristics")
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        sns.heatmap(df_chunk_traditional_scaled.set_index("Agent")[score_columns[:-1]], annot=False, cmap="coolwarm", ax=ax2)
        st.pyplot(fig2)

elif heatmap_mode == "Delta":
    st.markdown("### Score Delta (LLM - Traditional)")
    df_delta = df_chunk_llm.set_index("Agent")[score_columns[:-1]] - df_chunk_traditional_scaled.set_index("Agent")[score_columns[:-1]]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df_delta, cmap="bwr", center=0, annot=True, fmt=".1f", ax=ax)
    st.pyplot(fig)

# 📈 Final Score Trend
st.subheader("📈 Final Score Trend")
st.line_chart(df["Final Score"])

# 📦 Boxplot
st.subheader("📦 Score Distribution by Dimension")
fig_box = go.Figure()
for col in score_columns[:-1]:
    fig_box.add_trace(go.Box(y=df[col], name=col))
st.plotly_chart(fig_box, use_container_width=True)

# 🕸️ Radar Chart
st.subheader("🕸️ Agent Score Profile")
selected_agent_radar = st.selectbox("Select an agent for radar view", df["Agent"], key="radar_agent_select")
agent_data_radar = next((r for r in data if r["agent_id"] == selected_agent_radar), None)

if agent_data_radar:
    scores = agent_data_radar.get("scores", {})
    if "coherence_&_accuracy" in scores:
        scores["coherence_accuracy"] = scores["coherence_&_accuracy"]

    dimensions = [
        "instruction_following",
        "coherence_accuracy",
        "hallucination_detection",
        "style_matching",
        "length_penalty",
        "assumption_control"
    ]
    values = [scores.get(dim, 0.0) for dim in dimensions]
    values.append(values[0])  # Close the radar loop
    labels = [dim.replace("_", " ").title() for dim in dimensions]
    labels.append(labels[0])

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=values,
        theta=labels,
        fill='toself',
        name=selected_agent_radar,
        line=dict(color='royalblue')
    ))

    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
        margin=dict(t=30, b=30),
        showlegend=False
    )

    st.plotly_chart(fig_radar, use_container_width=True)
else:
    st.warning("⚠️ Could not load agent data for radar view.")


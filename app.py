

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="EnerVision AI",
    page_icon="⚡",
    layout="wide"
)
st.markdown("""
<style>
.block-container {
    max-width: 1400px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

h1 {
    font-size: 42px !important;
    font-weight: 800 !important;
}

h2, h3 {
    font-weight: 700 !important;
}

[data-testid="stMetric"] {
    background: linear-gradient(135deg, #14263b, #1c3850);
    padding: 24px;
    border-radius: 16px;
    border: 1px solid #286078;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}
</style>
""", unsafe_allow_html=True)

# Custom CSS
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #07111f, #101d30);
    color: #ffffff;
}
.block-container {
    padding-top: 2rem;
}
h1, h2, h3 {
    color: #eaf6ff !important;
}
[data-testid="stMetric"] {
    background: #14263b;
    padding: 22px;
    border: 1px solid #25445f;
    border-radius: 15px;
}
[data-testid="stMetricLabel"] {
    color: #9bbbd1 !important;
}
[data-testid="stMetricValue"] {
    color: #42e8c5 !important;
}
section[data-testid="stSidebar"] {
    background: #0b1727;
}
div.stButton > button {
    background: #13bfa5;
    color: #061521;
    border-radius: 10px;
    border: none;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Load data and model
df = pd.read_csv("data/hourly_energy.csv")
df["Time"] = pd.to_datetime(df["Time"])

model = joblib.load("models/demand_model.pkl")
features = joblib.load("models/feature_columns.pkl")
results = pd.read_csv("data/model_results.csv")

# Sidebar
st.sidebar.title("⚡ EnerVision AI")
st.sidebar.caption("Smart Energy Intelligence")

page = st.sidebar.radio(
    "NAVIGATION",
    ["Overview", "Energy Analytics", "AI Forecast", "Model Performance","About Project"]
)

st.sidebar.divider()
st.sidebar.success("ML System Active")
st.sidebar.caption("Dataset: April 2023")

# Header
st.title("⚡ EnerVision AI")
st.write("### Intelligent Energy Consumption & Production Forecasting")
st.caption("Machine Learning Powered Energy Analytics Platform")

st.divider()

# Overview page
if page == "Overview":

    st.subheader("Energy Overview")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Electricity Demand",
        f"{df['Demand'].iloc[-1]:,.0f}",
        "Latest hourly value"
    )

    c2.metric(
        "Wind Production",
        f"{df['Wind'].iloc[-1]:,.0f}",
        "Latest hourly value"
    )

    c3.metric(
        "Solar Production",
        f"{df['Solar'].iloc[-1]:,.0f}",
        "Latest hourly value"
    )

    st.divider()

    st.subheader("Electricity Demand Trend")

    fig = px.area(
        df,
        x="Time",
        y="Demand",
        template="plotly_dark",
        color_discrete_sequence=["#18d6bd"]
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=420
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Renewable Energy Production")

    renewable = df.melt(
        id_vars="Time",
        value_vars=["Wind", "Solar"],
        var_name="Energy Source",
        value_name="Production"
    )

    fig2 = px.line(
        renewable,
        x="Time",
        y="Production",
        color="Energy Source",
        template="plotly_dark",
        color_discrete_map={
            "Wind": "#36a3ff",
            "Solar": "#ffc857"
        }
    )

    fig2.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=400
    )

    st.plotly_chart(fig2, use_container_width=True)
    st.divider()

    st.subheader("⚡ Energy Balance Analysis")

    df["Renewable Production"] = df["Wind"] + df["Solar"]

    df["Other Generation Needed"] = (
        df["Demand"] - df["Renewable Production"]
    )

    c1, c2 = st.columns(2)

    c1.metric(
        "Total Renewable Production",
        f"{df['Renewable Production'].mean():,.0f}",
        "Average hourly"
    )

    c2.metric(
        "Demand minus Wind & Solar",
        f"{df['Other Generation Needed'].mean():,.0f}",
        "Average hourly"
    )

    balance_fig = px.area(
        df,
        x="Time",
        y=["Demand", "Renewable Production"],
        template="plotly_dark",
        title="Demand vs Renewable Production"
    )

    balance_fig.update_layout(height=450)

    st.plotly_chart(balance_fig, use_container_width=True)
# Analytics page
elif page == "Energy Analytics":

    st.subheader("Detailed Energy Analytics")

    st.write("Explore hourly electricity demand and renewable production.")

    start_date = df["Time"].min().date()
    end_date = df["Time"].max().date()

    selected_dates = st.date_input(
        "Select date range",
        value=(start_date, end_date)
    )

    if len(selected_dates) == 2:
        filtered = df[
            (df["Time"].dt.date >= selected_dates[0]) &
            (df["Time"].dt.date <= selected_dates[1])
        ]
    else:
        filtered = df

    fig = px.line(
        filtered,
        x="Time",
        y=["Demand", "Wind", "Solar"],
        template="plotly_dark",
        title="Energy Consumption and Production"
    )

    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Statistical Summary")
    st.dataframe(
        filtered[["Demand", "Wind", "Solar"]].describe(),
        use_container_width=True
    )

# Forecast page

elif page == "AI Forecast":

    st.subheader("🔮 AI Demand Forecast")

    latest = df.iloc[-1]
    previous_24 = df.iloc[-24]

    next_time = latest["Time"] + pd.Timedelta(hours=1)

    input_data = pd.DataFrame([{
        "Hour": next_time.hour,
        "Day": next_time.day,
        "DayOfWeek": next_time.day_of_week,

        "Demand_lag1": latest["Demand"],
        "Wind_lag1": latest["Wind"],
        "Solar_lag1": latest["Solar"],

        "Demand_lag24": previous_24["Demand"],
        "Wind_lag24": previous_24["Wind"],
        "Solar_lag24": previous_24["Solar"]
    }])

    input_data = input_data[features]

    prediction = model.predict(input_data)[0]

    st.write("Next-hour electricity demand prediction")

    st.metric(
        "Predicted Demand",
        f"{prediction:,.2f}"
    )

    st.info(f"Forecast Time: {next_time}")

    st.caption(
        "Gradient Boosting prediction using previous-hour "
        "and previous-24-hour energy data."
    )

    st.subheader("Recent Demand vs Forecast")

    recent = df.tail(24)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=recent["Time"],
        y=recent["Demand"],
        mode="lines",
        name="Actual Demand",
        line=dict(color="#18d6bd", width=3)
    ))

    fig.add_trace(go.Scatter(
        x=[next_time],
        y=[prediction],
        mode="markers",
        name="Predicted Demand",
        marker=dict(color="#ffc857", size=13)
    ))

    fig.update_layout(
        template="plotly_dark",
        height=420,
        paper_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(fig, use_container_width=True)


# Model performance page
elif page == "Model Performance":

    st.subheader("Machine Learning Model Evaluation")

    st.write("Comparison of three trained regression algorithms.")

    fig = px.bar(
        results,
        x="Model",
        y="MAE",
        color="Model",
        text_auto=".0f",
        template="plotly_dark",
        title="Mean Absolute Error (Lower is Better)"
    )

    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Performance Metrics")

    st.dataframe(
        results.round(3),
        use_container_width=True,
        hide_index=True
    )

    best = results.loc[results["MAE"].idxmin()]

    st.success(
        f"Best Model: {best['Model']} | "
        f"R² Score: {best['R2 Score']:.3f}"
    )

# About Project Page
elif page == "About Project":

    st.subheader("📘 About EnerVision AI")

    st.write(
        "EnerVision AI is a machine learning based energy "
        "analytics and demand forecasting platform."
    )

    st.divider()

    st.subheader("🎯 Project Objectives")

    st.markdown("""
    - Analyze electricity consumption patterns.
    - Study renewable energy production from wind and solar.
    - Forecast next-hour electricity demand.
    - Compare different machine learning algorithms.
    - Understand the balance between demand and renewable production.
    """)

    st.divider()

    st.subheader("📂 Dataset Information")

    st.markdown("""
    - Dataset: India Electricity Demand, Solar and Wind Generation
    - Data period used: April 2023
    - Original frequency: 10-second readings
    - Processed frequency: Hourly readings
    - Main features: Demand, Wind and Solar
    - Total hourly records: 720
    """)

    st.divider()

    st.subheader("🤖 Machine Learning Algorithms")

    st.markdown("""
    1. Linear Regression
    2. Random Forest Regressor
    3. Gradient Boosting Regressor
    """)

    st.info(
        "Best performing model: Gradient Boosting "
        "| R² Score: 0.916"
    )

    st.divider()

    st.subheader("🛠️ Technology Stack")

    st.markdown("""
    - Python
    - Pandas and NumPy
    - Scikit-learn
    - Plotly
    - Streamlit
    """)

    st.divider()

    st.subheader("⚠️ Project Limitations")

    st.markdown("""
    - Analysis uses April 2023 data only.
    - Forecast is a next-hour demonstration, not a live prediction.
    - Model performance may differ on unseen future data.
    """)

# Footer
st.divider()
st.caption(
    "EnerVision AI | Energy Forecasting & Analytics | Academic Project"
)

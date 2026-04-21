import time
import warnings

import pandas as pd
import streamlit as st

from src.data_cleaning import load_and_clean_data
from src.model import load_model_artifacts, train_model
from src.pricing import calculate_demand, cancellation_risk, dynamic_price

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Dynamic Pricing Engine", layout="wide")


@st.cache_data
def get_data():
    return load_and_clean_data("data/hotel_bookings.csv")


@st.cache_resource
def get_model_artifacts():
    try:
        return load_model_artifacts("model.pkl")
    except Exception:
        df = get_data()
        return train_model(df)


def format_currency(value):
    return f"Rs. {value:,.2f}"


def explain_recommendation(demand_score, cancel_risk, season_factor, uplift):
    reasons = []

    if demand_score >= 0.5:
        reasons.append("high booking demand")
    elif demand_score <= 0.2:
        reasons.append("soft demand conditions")

    if cancel_risk >= 0.3:
        reasons.append("elevated cancellation risk")

    if season_factor >= 0.3:
        reasons.append("peak season effect")

    if not reasons:
        reasons.append("balanced market conditions")

    if uplift > 0:
        action = "increase"
    elif uplift < 0:
        action = "reduce"
    else:
        action = "keep"

    return f"Recommended to {action} price because of {', '.join(reasons)}."


def build_row(model, feature_columns, row, booking_number):
    features = pd.DataFrame([[row[col] for col in feature_columns]], columns=feature_columns)
    base_price = float(model.predict(features)[0])
    demand_score = calculate_demand(row)
    cancel_risk = cancellation_risk(row)
    season_factor = row["season_factor"]
    dynamic_value = dynamic_price(base_price, demand_score, cancel_risk, season_factor)
    actual_adr = float(row["adr"])
    fixed_price = round(base_price * 1.05, 2)
    error = actual_adr - base_price
    dynamic_gap = dynamic_value - actual_adr
    recommendation = explain_recommendation(demand_score, cancel_risk, season_factor, dynamic_value - base_price)

    return {
        "booking": booking_number,
        "base_price": round(base_price, 2),
        "fixed_price": fixed_price,
        "dynamic_price": round(dynamic_value, 2),
        "actual_adr": round(actual_adr, 2),
        "uplift": round(dynamic_value - base_price, 2),
        "prediction_error": round(error, 2),
        "abs_error": round(abs(error), 2),
        "dynamic_gap": round(dynamic_gap, 2),
        "demand_score": round(demand_score, 2),
        "cancel_risk": round(cancel_risk, 2),
        "season_factor": round(season_factor, 2),
        "recommendation": recommendation,
    }


st.title("Real-Time Hotel Dynamic Pricing Engine")
st.caption("Predictive pricing dashboard with evaluation, scenario testing, strategy comparison, and explainable recommendations.")

df = get_data()
artifacts = get_model_artifacts()
model = artifacts["model"]
feature_columns = artifacts["features"]
metrics = artifacts["metrics"]
importance_df = artifacts["feature_importance"]

st.sidebar.header("Controls")
rows = st.sidebar.slider("Rows to simulate", 10, 100, 50)
speed = st.sidebar.slider("Speed", 0.0, 1.0, 0.2, 0.1)

with st.expander("Methodology", expanded=False):
    st.write(
        """
        - Dataset: Hotel booking records from `data/hotel_bookings.csv`
        - Features used for prediction: lead time, total guests, and stay length
        - Model: XGBoost Regressor trained to estimate ADR
        - Pricing strategies compared: baseline predicted price, fixed markup, and dynamic rule-based pricing
        """
    )

if metrics:
    st.subheader("Model Performance")
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("MAE", metrics["mae"])
    metric_col2.metric("RMSE", metrics["rmse"])
    metric_col3.metric("R2 Score", metrics["r2"])

overview_left, overview_right = st.columns([1.1, 1.9])

with overview_left:
    st.subheader("Feature Importance")
    if importance_df is not None:
        st.bar_chart(importance_df.set_index("feature"))
        st.dataframe(importance_df, use_container_width=True, hide_index=True)
    else:
        st.info("Feature importance will appear after retraining the model.")

with overview_right:
    st.subheader("Scenario Testing")
    st.write("Adjust booking conditions to see how the engine changes its recommendation.")

    scenario_col1, scenario_col2, scenario_col3 = st.columns(3)
    scenario_lead_time = scenario_col1.slider("Lead Time", 0, 365, 30)
    scenario_guests = scenario_col2.slider("Total Guests", 1, 6, 2)
    scenario_stay = scenario_col3.slider("Stay Length", 1, 14, 3)
    scenario_season = st.selectbox("Season", ["Regular", "Peak"])

    scenario_row = pd.Series(
        {
            "lead_time": scenario_lead_time,
            "total_guests": scenario_guests,
            "stay_length": scenario_stay,
            "season_factor": 0.3 if scenario_season == "Peak" else 0.1,
            "adr": 0.0,
        }
    )
    scenario_output = build_row(model, feature_columns, scenario_row, 1)

    scenario_metric1, scenario_metric2, scenario_metric3 = st.columns(3)
    scenario_metric1.metric("Predicted Base Price", format_currency(scenario_output["base_price"]))
    scenario_metric2.metric("Dynamic Price", format_currency(scenario_output["dynamic_price"]))
    scenario_metric3.metric("Price Change", f"{scenario_output['uplift']:+.2f}")
    st.info(scenario_output["recommendation"])

error_section, strategy_section = st.columns(2)

analysis_df = df.head(rows).copy()
analysis_rows = [build_row(model, feature_columns, row, i + 1) for i, (_, row) in enumerate(analysis_df.iterrows())]
analysis_results = pd.DataFrame(analysis_rows)

with error_section:
    st.subheader("Actual vs Predicted Error Analysis")
    avg_abs_error = analysis_results["abs_error"].mean()
    max_error_row = analysis_results.loc[analysis_results["abs_error"].idxmax()]

    error_metric1, error_metric2 = st.columns(2)
    error_metric1.metric("Average Absolute Error", format_currency(avg_abs_error))
    error_metric2.metric("Largest Error", format_currency(max_error_row["abs_error"]))

    error_chart = analysis_results[["booking", "actual_adr", "base_price"]].set_index("booking")
    st.line_chart(error_chart)
    st.dataframe(
        analysis_results[["booking", "actual_adr", "base_price", "prediction_error", "abs_error"]],
        use_container_width=True,
        hide_index=True,
    )

with strategy_section:
    st.subheader("Pricing Strategy Comparison")
    strategy_summary = pd.DataFrame(
        [
            {
                "strategy": "Predicted Base Price",
                "average_price": round(analysis_results["base_price"].mean(), 2),
                "average_gap_vs_actual": round((analysis_results["base_price"] - analysis_results["actual_adr"]).mean(), 2),
            },
            {
                "strategy": "Fixed +5% Price",
                "average_price": round(analysis_results["fixed_price"].mean(), 2),
                "average_gap_vs_actual": round((analysis_results["fixed_price"] - analysis_results["actual_adr"]).mean(), 2),
            },
            {
                "strategy": "Dynamic Price",
                "average_price": round(analysis_results["dynamic_price"].mean(), 2),
                "average_gap_vs_actual": round((analysis_results["dynamic_price"] - analysis_results["actual_adr"]).mean(), 2),
            },
        ]
    )

    st.bar_chart(strategy_summary.set_index("strategy")[["average_price"]])
    st.dataframe(strategy_summary, use_container_width=True, hide_index=True)

st.subheader("Live Simulation")
placeholder = st.empty()
price_chart = []
demand_chart = []
simulation_rows = []

for i in range(rows):
    row = df.iloc[i]
    row_result = build_row(model, feature_columns, row, i + 1)

    price_chart.append(
        {
            "Booking": row_result["booking"],
            "Base Price": row_result["base_price"],
            "Fixed Price": row_result["fixed_price"],
            "Dynamic Price": row_result["dynamic_price"],
            "Actual ADR": row_result["actual_adr"],
        }
    )
    demand_chart.append({"Booking": row_result["booking"], "Demand Score": row_result["demand_score"]})
    simulation_rows.append(row_result)

    simulation_df = pd.DataFrame(simulation_rows)
    avg_uplift = simulation_df["uplift"].mean()
    total_revenue_gain = simulation_df["uplift"].sum()
    high_demand_count = int((simulation_df["demand_score"] >= 0.5).sum())
    risky_bookings = int((simulation_df["cancel_risk"] >= 0.3).sum())

    with placeholder.container():
        summary_cols = st.columns(4)
        summary_cols[0].metric("Base Price", format_currency(row_result["base_price"]))
        summary_cols[1].metric("Demand", f"{row_result['demand_score']:.2f}")
        summary_cols[2].metric("Cancel Risk", f"{row_result['cancel_risk']:.2f}")
        summary_cols[3].metric("Dynamic Price", format_currency(row_result["dynamic_price"]), f"{row_result['uplift']:+.2f}")

        insight_cols = st.columns(4)
        insight_cols[0].metric("Average Uplift", format_currency(avg_uplift))
        insight_cols[1].metric("Revenue Gain", format_currency(total_revenue_gain))
        insight_cols[2].metric("High Demand Bookings", high_demand_count)
        insight_cols[3].metric("High Risk Bookings", risky_bookings)

        st.success(row_result["recommendation"])

        if row_result["demand_score"] > 0.5:
            st.warning("High demand detected: increasing price can improve revenue.")
        elif row_result["demand_score"] < 0.2:
            st.info("Low demand detected: discounting may improve conversions.")

        if row_result["cancel_risk"] > 0.3:
            st.error("High cancellation risk detected for this booking.")

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.subheader("Price Comparison")
            st.line_chart(pd.DataFrame(price_chart).set_index("Booking"))

        with chart_col2:
            st.subheader("Demand Trend")
            st.line_chart(pd.DataFrame(demand_chart).set_index("Booking"))

        st.subheader("Simulation Table")
        st.dataframe(
            simulation_df[
                [
                    "booking",
                    "base_price",
                    "fixed_price",
                    "dynamic_price",
                    "actual_adr",
                    "uplift",
                    "prediction_error",
                    "recommendation",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    if speed > 0:
        time.sleep(speed)

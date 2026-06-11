import pandas as pd
import requests
import streamlit as st

from config import API_BASE_URL


def fetch_dashboard_run(run_id: int):
    """
    Fetch dashboard data for one pipeline run from the FastAPI backend.
    """
    url = f"{API_BASE_URL}/dashboard/runs/{run_id}"

    try:
        response = requests.get(url, timeout=5)
    except requests.exceptions.ConnectionError:
        return None, "Could not connect to the FastAPI server. Make sure it is running."
    except requests.exceptions.Timeout:
        return None, "The API request timed out."

    if response.status_code != 200:
        try:
            error_body = response.json()
        except ValueError:
            error_body = response.text

        return None, f"Request failed with status code {response.status_code}: {error_body}"

    try:
        return response.json(), None
    except ValueError:
        return None, "API returned a successful response, but it was not valid JSON."


def show_summary_metrics(summary: dict):
    """
    Display run-level summary metrics.
    """
    st.subheader("Run Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Predictions", summary["total_predictions"])

    with col2:
        st.metric("Predicted Anomalies", summary["total_anomalies_predicted"])

    with col3:
        st.metric("Row Alerts", summary["total_row_alerts"])

    col4, col5, col6 = st.columns(3)

    with col4:
        st.metric("Alert Events", summary["total_alert_events"])

    with col5:
        st.metric("Critical Events", summary["critical_alert_events"])

    with col6:
        st.metric("Machines With Alerts", summary["machines_with_alerts"])

    max_score = summary["max_anomaly_score"]
    mean_score = summary["mean_anomaly_score"]

    col7, col8 = st.columns(2)

    with col7:
        st.metric(
            "Max Anomaly Score",
            "N/A" if max_score is None else round(max_score, 3),
        )

    with col8:
        st.metric(
            "Mean Anomaly Score",
            "N/A" if mean_score is None else round(mean_score, 3),
        )


def show_distribution_chart(data: list[dict], label_column: str, title: str):
    """
    Display one distribution bar chart from API distribution data.
    """
    st.subheader(title)

    df = pd.DataFrame(data)

    if df.empty:
        st.info(f"No {title.lower()} data available.")
        return

    df = df.set_index(label_column)

    st.bar_chart(df["count"])


def show_top_critical_events(events: list[dict]):
    """
    Display top critical alert events in a table.
    """
    st.subheader("Top Critical Events")

    top_events_df = pd.DataFrame(events)

    if top_events_df.empty:
        st.info("No critical events found for this run.")
        return

    display_columns = [
        "event_id",
        "machine_id",
        "sensor",
        "anomaly_type",
        "start_step",
        "end_step",
        "duration",
        "alert_count",
        "max_severity",
        "max_anomaly_score",
        "first_reason",
    ]

    existing_columns = [
        column for column in display_columns if column in top_events_df.columns
    ]

    top_events_df = top_events_df[existing_columns]

    st.dataframe(
        top_events_df,
        use_container_width=True,
        hide_index=True,
    )


st.set_page_config(
    page_title="Anomaly Detection Dashboard",
    layout="wide",
)

st.title("Industrial Anomaly Detection Dashboard")

st.write(
    "This dashboard calls the FastAPI backend and displays stored pipeline results."
)

run_id = st.number_input(
    "Run ID",
    min_value=1,
    value=1,
    step=1,
)

if st.button("Load Dashboard"):
    data, error = fetch_dashboard_run(int(run_id))

    if error is not None:
        st.error(error)
    else:
        st.success(f"Loaded dashboard data for run {run_id}")

        show_summary_metrics(data["summary"])

        st.divider()

        show_distribution_chart(
            data=data["anomaly_type_distribution"],
            label_column="anomaly_type",
            title="Anomaly Type Distribution",
        )

        show_distribution_chart(
            data=data["sensor_distribution"],
            label_column="sensor",
            title="Sensor Distribution",
        )

        show_distribution_chart(
            data=data["severity_distribution"],
            label_column="severity",
            title="Severity Distribution",
        )

        st.divider()

        show_top_critical_events(data["top_critical_events"])

        with st.expander("Raw API Response"):
            st.json(data)
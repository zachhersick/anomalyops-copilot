import pandas as pd
import pytest

from alerts import (
    ALERT_STATUS_OPEN,
    ALERT_TYPE_MODEL,
    ALERT_TYPE_MODEL_AND_THRESHOLD,
    SEVERITY_CRITICAL,
    SEVERITY_INFO,
    SEVERITY_WARNING,
    build_alert,
    build_alerts,
    check_violation,
    load_predictions,
    run_alert_pipeline,
    save_alerts,
)


def make_prediction_row(
    prediction=1,
    target_sensor="temperature",
    anomaly_score=0.8,
    temperature=90.0,
    pressure=50.0,
    vibration=2.0,
    flow_rate=1.0,
    voltage=120.0,
    current=10.0,
    real_value=1,
):
    return pd.Series(
        {
            "step": 1,
            "machine_id": 1,
            "prediction": prediction,
            "anomaly_score": anomaly_score,
            "target_sensor": target_sensor,
            "anomaly_type": "spike",
            "temperature": temperature,
            "pressure": pressure,
            "vibration": vibration,
            "flow_rate": flow_rate,
            "voltage": voltage,
            "current": current,
            "real_value": real_value,
            "threshold": 0.35,
        }
    )


def make_predictions_df():
    return pd.DataFrame(
        [
            make_prediction_row(
                prediction=1,
                target_sensor="temperature",
                anomaly_score=0.8,
                temperature=90.0,
                real_value=1,
            ),
            make_prediction_row(
                prediction=1,
                target_sensor="pressure",
                anomaly_score=0.95,
                pressure=120.0,
                real_value=1,
            ),
            make_prediction_row(
                prediction=0,
                target_sensor="current",
                anomaly_score=0.1,
                current=10.0,
                real_value=0,
            ),
        ]
    )


def test_check_violation_returns_none_when_sensor_is_safe():
    row = make_prediction_row(
        target_sensor="temperature",
        temperature=90.0,
    )

    severity, reason = check_violation(row, "temperature")

    assert severity is None
    assert reason is None


def test_check_violation_returns_critical_for_high_pressure():
    row = make_prediction_row(
        target_sensor="pressure",
        pressure=120.0,
    )

    severity, reason = check_violation(row, "pressure")

    assert severity == SEVERITY_CRITICAL
    assert "pressure value" in reason
    assert "critical high threshold" in reason


def test_build_alert_creates_model_only_alert():
    row = make_prediction_row(
        target_sensor="temperature",
        anomaly_score=0.8,
        temperature=90.0,
    )

    alert = build_alert(row, alert_id=1)

    assert alert["alert_id"] == 1
    assert alert["sensor"] == "temperature"
    assert alert["sensor_value"] == 90.0
    assert alert["severity"] == SEVERITY_WARNING
    assert alert["alert_type"] == ALERT_TYPE_MODEL
    assert alert["status"] == ALERT_STATUS_OPEN
    assert alert["real_value"] == 1


def test_build_alert_creates_model_and_threshold_alert():
    row = make_prediction_row(
        target_sensor="pressure",
        anomaly_score=0.95,
        pressure=120.0,
    )

    alert = build_alert(row, alert_id=1)

    assert alert["alert_id"] == 1
    assert alert["sensor"] == "pressure"
    assert alert["sensor_value"] == 120.0
    assert alert["severity"] == SEVERITY_CRITICAL
    assert alert["alert_type"] == ALERT_TYPE_MODEL_AND_THRESHOLD
    assert "critical high threshold" in alert["reason"]


def test_build_alert_handles_unknown_sensor():
    row = make_prediction_row(
        target_sensor="unknown_sensor",
        anomaly_score=0.8,
    )

    alert = build_alert(row, alert_id=1)

    assert alert["alert_id"] == 1
    assert alert["sensor"] == "unknown_sensor"
    assert alert["sensor_value"] is None
    assert alert["severity"] == SEVERITY_WARNING
    assert alert["alert_type"] == ALERT_TYPE_MODEL
    assert "not recognized" in alert["reason"]


def test_build_alerts_only_uses_predicted_anomaly_rows():
    predictions_df = make_predictions_df()

    alerts_df = build_alerts(predictions_df)

    assert len(alerts_df) == 2
    assert alerts_df["alert_id"].tolist() == [1, 2]
    assert alerts_df["prediction"].tolist() == [1, 1]
    assert set(alerts_df["sensor"]) == {"temperature", "pressure"}


def test_load_predictions_loads_valid_file(tmp_path):
    predictions_df = make_predictions_df()
    input_path = tmp_path / "predictions.csv"

    predictions_df.to_csv(input_path, index=False)

    loaded_df = load_predictions(input_path)

    assert len(loaded_df) == len(predictions_df)
    assert "prediction" in loaded_df.columns
    assert "anomaly_score" in loaded_df.columns


def test_load_predictions_raises_when_required_column_missing(tmp_path):
    predictions_df = make_predictions_df().drop(columns=["anomaly_score"])
    input_path = tmp_path / "bad_predictions.csv"

    predictions_df.to_csv(input_path, index=False)

    with pytest.raises(ValueError, match="Missing required columns"):
        load_predictions(input_path)


def test_save_alerts_writes_file(tmp_path):
    predictions_df = make_predictions_df()
    alerts_df = build_alerts(predictions_df)

    output_path = tmp_path / "alerts.csv"

    save_alerts(alerts_df, output_path)

    assert output_path.exists()

    saved_df = pd.read_csv(output_path)

    assert len(saved_df) == len(alerts_df)
    assert "alert_id" in saved_df.columns
    assert "severity" in saved_df.columns


def test_run_alert_pipeline_writes_alerts_and_returns_dataframe(tmp_path, capsys):
    predictions_df = make_predictions_df()

    input_path = tmp_path / "predictions.csv"
    output_path = tmp_path / "alerts.csv"

    predictions_df.to_csv(input_path, index=False)

    alerts_df = run_alert_pipeline(
        input_file=input_path,
        output_file=output_path,
    )

    captured = capsys.readouterr()

    assert output_path.exists()
    assert len(alerts_df) == 2
    assert "ALERT SUMMARY" in captured.out
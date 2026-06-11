import pandas as pd

from evaluate import (
    load_evaluation_inputs,
    build_recall_table,
    build_evaluation_tables,
    build_overall_metrics,
    run_evaluation,
)


def make_predictions_df():
    return pd.DataFrame(
        [
            {
                "step": 1,
                "machine_id": 1,
                "real_value": 1,
                "prediction": 1,
                "anomaly_type": "drift",
                "target_sensor": "temperature",
                "threshold": 0.35,
                "temperature_lag_5_autocorr": 0.5,
            },
            {
                "step": 2,
                "machine_id": 1,
                "real_value": 1,
                "prediction": 0,
                "anomaly_type": "drift",
                "target_sensor": "temperature",
                "threshold": 0.35,
                "temperature_lag_5_autocorr": 0.2,
            },
            {
                "step": 3,
                "machine_id": 2,
                "real_value": 1,
                "prediction": 1,
                "anomaly_type": "oscillation",
                "target_sensor": "current",
                "threshold": 0.35,
                "current_lag_5_autocorr": 0.7,
            },
            {
                "step": 4,
                "machine_id": 2,
                "real_value": 0,
                "prediction": 1,
                "anomaly_type": "none",
                "target_sensor": "none",
                "threshold": 0.35,
            },
            {
                "step": 5,
                "machine_id": 2,
                "real_value": 0,
                "prediction": 0,
                "anomaly_type": "none",
                "target_sensor": "none",
                "threshold": 0.35,
            },
        ]
    )


def make_retention_df():
    return pd.DataFrame(
        [
            {
                "anomaly_type": "drift",
                "target_sensor": "temperature",
                "before_drop": 10,
                "after_drop": 8,
                "rows_dropped": 2,
                "pct_kept": 0.8,
            }
        ]
    )


def make_feature_importance_df():
    return pd.DataFrame(
        [
            {"feature": "temperature_delta", "importance": 0.5},
            {"feature": "current_lag_5_autocorr", "importance": 0.3},
        ]
    )


def test_load_evaluation_inputs_loads_all_files(tmp_path):
    predictions_df = make_predictions_df()
    retention_df = make_retention_df()
    feature_importance_df = make_feature_importance_df()

    predictions_path = tmp_path / "predictions.csv"
    retention_path = tmp_path / "feature_row_retention.csv"
    feature_importance_path = tmp_path / "feature_importance.csv"

    predictions_df.to_csv(predictions_path, index=False)
    retention_df.to_csv(retention_path, index=False)
    feature_importance_df.to_csv(feature_importance_path, index=False)

    loaded_predictions, loaded_retention, loaded_importance = load_evaluation_inputs(
        predictions_path=predictions_path,
        retention_path=retention_path,
        feature_importance_path=feature_importance_path,
    )

    assert len(loaded_predictions) == len(predictions_df)
    assert len(loaded_retention) == len(retention_df)
    assert len(loaded_importance) == len(feature_importance_df)


def test_load_evaluation_inputs_handles_missing_feature_importance(tmp_path):
    predictions_df = make_predictions_df()
    retention_df = make_retention_df()

    predictions_path = tmp_path / "predictions.csv"
    retention_path = tmp_path / "feature_row_retention.csv"
    missing_feature_importance_path = tmp_path / "missing_feature_importance.csv"

    predictions_df.to_csv(predictions_path, index=False)
    retention_df.to_csv(retention_path, index=False)

    loaded_predictions, loaded_retention, loaded_importance = load_evaluation_inputs(
        predictions_path=predictions_path,
        retention_path=retention_path,
        feature_importance_path=missing_feature_importance_path,
    )

    assert len(loaded_predictions) == len(predictions_df)
    assert len(loaded_retention) == len(retention_df)
    assert loaded_importance is None


def test_build_recall_table_groups_and_calculates_recall():
    df = make_predictions_df()

    mask = df["real_value"] == 1

    recall_table = build_recall_table(
        df=df,
        mask=mask,
        group_col="anomaly_type",
    )

    assert recall_table.loc["drift", "total"] == 2
    assert recall_table.loc["drift", "correct"] == 1
    assert recall_table.loc["drift", "missed"] == 1
    assert recall_table.loc["drift", "recall"] == 0.5

    assert recall_table.loc["oscillation", "total"] == 1
    assert recall_table.loc["oscillation", "correct"] == 1
    assert recall_table.loc["oscillation", "recall"] == 1.0


def test_build_evaluation_tables_returns_expected_tables():
    df = make_predictions_df()

    tables = build_evaluation_tables(df)

    assert "anomaly_by_type" in tables
    assert "anomaly_by_sensor" in tables
    assert "drift_by_sensor" in tables
    assert "oscillation_by_sensor" in tables
    assert "false_positives" in tables
    assert "false_negatives" in tables

    assert len(tables["false_positives"]) == 1
    assert len(tables["false_negatives"]) == 1

    assert tables["drift_by_sensor"].loc["temperature", "recall"] == 0.5
    assert tables["oscillation_by_sensor"].loc["current", "recall"] == 1.0


def test_build_overall_metrics_returns_expected_values():
    df = make_predictions_df()

    overall_metrics = build_overall_metrics(df)

    assert overall_metrics["accuracy"] == 0.6
    assert overall_metrics["false_positives"] == 1
    assert overall_metrics["false_negatives"] == 1
    assert overall_metrics["true_positives"] == 2
    assert overall_metrics["true_negatives"] == 1
    assert overall_metrics["confusion_matrix"].shape == (2, 2)


def test_run_evaluation_returns_report_objects(tmp_path, capsys):
    predictions_df = make_predictions_df()
    retention_df = make_retention_df()
    feature_importance_df = make_feature_importance_df()

    predictions_path = tmp_path / "predictions.csv"
    retention_path = tmp_path / "feature_row_retention.csv"
    feature_importance_path = tmp_path / "feature_importance.csv"

    predictions_df.to_csv(predictions_path, index=False)
    retention_df.to_csv(retention_path, index=False)
    feature_importance_df.to_csv(feature_importance_path, index=False)

    result = run_evaluation(
        predictions_path=predictions_path,
        retention_path=retention_path,
        feature_importance_path=feature_importance_path,
        debug_oscillation_details=False,
    )

    captured = capsys.readouterr()

    assert "FINAL MODEL EVALUATION" in captured.out
    assert "ROW RETENTION" in captured.out

    assert "predictions_df" in result
    assert "retention_df" in result
    assert "feature_importance_df" in result
    assert "evaluation_tables" in result
    assert "overall_metrics" in result
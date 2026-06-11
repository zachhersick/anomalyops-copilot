import numpy as np
import pandas as pd

from config import MODEL_THRESHOLD
from model import (
    THRESHOLDS,
    find_cols_with_suffixes,
    load_feature_data,
    prepare_model_inputs,
    split_train_test,
    train_model,
    run_threshold_sweep,
    save_predictions,
    save_threshold_results,
    save_feature_importance,
    run_model_pipeline,
)



def make_test_feature_df(num_rows=40):
    """
    Create a small fake feature dataframe that looks like sensor_data_features.csv.
    """
    rows = []

    for i in range(num_rows):
        is_anomaly = 1 if i % 2 == 0 else 0

        rows.append(
            {
                "step": i + 1,
                "machine_id": 1 if i < num_rows // 2 else 2,
                "any_anomaly": is_anomaly,
                "anomaly_type": "spike" if is_anomaly else "none",
                "target_sensor": "temperature" if is_anomaly else "none",

                # Raw-ish sensor values
                "temperature": 100.0 + is_anomaly * 20.0 + i * 0.01,
                "pressure": 50.0 + i * 0.01,
                "vibration": 2.0 + is_anomaly * 0.5,
                "flow_rate": 1.0,
                "voltage": 120.0,
                "current": 10.0,

                # Example engineered features
                "temperature_delta": 0.5 + is_anomaly,
                "temperature_abs_delta": 0.5 + is_anomaly,
                "temperature_roll_mean": 90.0 + is_anomaly * 10.0,
                "temperature_roll_std": 1.0 + is_anomaly,
                "temperature_zscore": float(is_anomaly) * 3.0,
                "temperature_slope_10": 0.1 + is_anomaly,

                # Column that should be dropped by PERMANENT_DROP_SUFFIXES
                "temperature_dir_imbalance_10": 0.99,

                # Per-sensor label column that should not be used as a feature
                "temperature_anomaly": is_anomaly,
            }
        )

    return pd.DataFrame(rows)


def test_find_cols_with_suffixes_returns_matching_columns():
    columns = [
        "temperature_delta",
        "temperature_dir_imbalance_10",
        "pressure_dir_imbalance_20",
        "current_zscore",
    ]

    result = find_cols_with_suffixes(
        columns,
        ["_dir_imbalance_10", "_dir_imbalance_20"],
    )

    assert result == [
        "temperature_dir_imbalance_10",
        "pressure_dir_imbalance_20",
    ]


def test_load_feature_data_drops_nan_and_infinite_rows(tmp_path):
    df = make_test_feature_df(num_rows=6)

    df.loc[0, "temperature_delta"] = np.inf
    df.loc[1, "temperature_delta"] = np.nan

    input_path = tmp_path / "features.csv"
    df.to_csv(input_path, index=False)

    loaded_df = load_feature_data(input_path)

    assert len(loaded_df) == 4
    assert np.isfinite(loaded_df["temperature_delta"]).all()


def test_load_feature_data_raises_when_label_missing(tmp_path):
    df = make_test_feature_df(num_rows=6)
    df = df.drop(columns=["any_anomaly"])

    input_path = tmp_path / "features_missing_label.csv"
    df.to_csv(input_path, index=False)

    try:
        load_feature_data(input_path)
        assert False, "Expected ValueError for missing label column."
    except ValueError as error:
        assert "Missing label column" in str(error)


def test_prepare_model_inputs_splits_features_labels_and_metadata():
    df = make_test_feature_df(num_rows=20)

    X, y, meta = prepare_model_inputs(df)

    assert len(X) == len(df)
    assert len(y) == len(df)
    assert len(meta) == len(df)

    assert "any_anomaly" not in X.columns
    assert "anomaly_type" not in X.columns
    assert "target_sensor" not in X.columns
    assert "temperature_anomaly" not in X.columns
    assert "temperature_dir_imbalance_10" not in X.columns

    assert "temperature_delta" in X.columns
    assert "temperature_zscore" in X.columns

    assert set(["step", "machine_id", "anomaly_type", "target_sensor"]).issubset(
        set(meta.columns)
    )


def test_split_train_test_returns_disjoint_indices():
    df = make_test_feature_df(num_rows=40)
    X, y, meta = prepare_model_inputs(df)

    train_idx, test_idx = split_train_test(df, y)

    assert len(train_idx) + len(test_idx) == len(df)
    assert set(train_idx).isdisjoint(set(test_idx))


def test_train_model_and_threshold_sweep_returns_expected_outputs():
    df = make_test_feature_df(num_rows=40)

    X, y, meta = prepare_model_inputs(df)
    train_idx, test_idx = split_train_test(df, y)

    X_train = X.loc[train_idx]
    X_test = X.loc[test_idx]

    y_train = y.loc[train_idx]
    y_test = y.loc[test_idx]

    meta_test = meta.loc[test_idx].copy()

    trained_model = train_model(X_train, y_train)

    threshold_df, predictions_df = run_threshold_sweep(
        model=trained_model,
        df=df,
        X_test=X_test,
        y_test=y_test,
        meta_test=meta_test,
        test_idx=test_idx,
    )

    assert len(threshold_df) == len(THRESHOLDS)
    assert MODEL_THRESHOLD in threshold_df["threshold"].values

    assert len(predictions_df) == len(test_idx)
    assert "real_value" in predictions_df.columns
    assert "prediction" in predictions_df.columns
    assert "anomaly_score" in predictions_df.columns
    assert "threshold" in predictions_df.columns

    assert set(predictions_df["threshold"].unique()) == {MODEL_THRESHOLD}


def test_save_predictions_and_threshold_results_write_files(tmp_path):
    predictions_df = pd.DataFrame(
        [
            {
                "step": 1,
                "machine_id": 1,
                "real_value": 1,
                "prediction": 1,
                "anomaly_score": 0.95,
                "threshold": MODEL_THRESHOLD,
            }
        ]
    )

    threshold_df = pd.DataFrame(
        [
            {
                "threshold": MODEL_THRESHOLD,
                "accuracy": 1.0,
                "precision": 1.0,
                "f1": 1.0,
                "false_positives": 0,
                "false_negatives": 0,
                "anomaly_recall": 1.0,
                "oscillation_recall": np.nan,
                "current_osc_recall": np.nan,
                "voltage_osc_recall": np.nan,
            }
        ]
    )

    predictions_path = tmp_path / "predictions.csv"
    threshold_path = tmp_path / "outputs" / "threshold_results.csv"

    save_predictions(predictions_df, predictions_path)
    save_threshold_results(threshold_df, threshold_path)

    assert predictions_path.exists()
    assert threshold_path.exists()


def test_save_feature_importance_writes_file(tmp_path):
    df = make_test_feature_df(num_rows=40)

    X, y, meta = prepare_model_inputs(df)
    train_idx, test_idx = split_train_test(df, y)

    model = train_model(X.loc[train_idx], y.loc[train_idx])

    output_path = tmp_path / "feature_importance.csv"

    feature_importance_df = save_feature_importance(
        model=model,
        feature_columns=X.columns,
        output_path=output_path,
    )

    assert output_path.exists()
    assert feature_importance_df is not None
    assert "feature" in feature_importance_df.columns
    assert "importance" in feature_importance_df.columns
    assert len(feature_importance_df) == len(X.columns)


def test_run_model_pipeline_writes_expected_outputs(tmp_path):
    df = make_test_feature_df(num_rows=40)

    input_path = tmp_path / "sensor_data_features.csv"
    predictions_path = tmp_path / "predictions.csv"
    threshold_results_path = tmp_path / "outputs" / "threshold_results.csv"
    feature_importance_path = tmp_path / "feature_importance.csv"

    df.to_csv(input_path, index=False)

    model, threshold_df, predictions_df, feature_importance_df = run_model_pipeline(
        input_csv=input_path,
        predictions_output_path=predictions_path,
        threshold_results_output_path=threshold_results_path,
        feature_importance_output_path=feature_importance_path,
    )

    assert predictions_path.exists()
    assert threshold_results_path.exists()
    assert feature_importance_path.exists()

    assert len(threshold_df) == len(THRESHOLDS)
    assert len(predictions_df) > 0
    assert feature_importance_df is not None
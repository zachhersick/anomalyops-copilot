import pandas as pd

from features import build_features, save_feature_outputs
from generator import generate_sensor_data


def test_build_features_returns_expected_outputs():
    raw_df = generate_sensor_data(
        num_machines=2,
        num_timesteps=80,
        fixed_seed=295,
    )

    features_df, retention_df = build_features(raw_df)

    assert not features_df.empty
    assert not retention_df.empty


def test_build_features_has_required_original_columns():
    raw_df = generate_sensor_data(
        num_machines=2,
        num_timesteps=80,
        fixed_seed=295,
    )

    features_df, retention_df = build_features(raw_df)

    required_columns = {
        "step",
        "machine_id",
        "any_anomaly",
        "target_sensor",
        "anomaly_type",
        "temperature",
        "pressure",
        "vibration",
        "flow_rate",
        "voltage",
        "current",
    }

    assert required_columns.issubset(set(features_df.columns))


def test_build_features_has_engineered_columns():
    raw_df = generate_sensor_data(
        num_machines=2,
        num_timesteps=80,
        fixed_seed=295,
    )

    features_df, retention_df = build_features(raw_df)

    expected_engineered_columns = {
        "temperature_delta",
        "temperature_abs_delta",
        "temperature_roll_mean",
        "temperature_roll_std",
        "temperature_zscore",
        "temperature_5_step_diff",
        "temperature_10_step_diff",
        "temperature_slope_10",
        "temperature_lag_5_autocorr",
        "temperature_center_balance_10",
    }

    assert expected_engineered_columns.issubset(set(features_df.columns))


def test_build_features_drops_initial_rolling_rows():
    raw_df = generate_sensor_data(
        num_machines=2,
        num_timesteps=80,
        fixed_seed=295,
    )

    features_df, retention_df = build_features(raw_df)

    assert len(features_df) < len(raw_df)


def test_build_features_retention_report_has_expected_columns():
    raw_df = generate_sensor_data(
        num_machines=2,
        num_timesteps=80,
        fixed_seed=295,
    )

    features_df, retention_df = build_features(raw_df)

    expected_columns = {
        "anomaly_type",
        "target_sensor",
        "before_drop",
        "after_drop",
        "rows_dropped",
        "pct_kept",
    }

    assert expected_columns.issubset(set(retention_df.columns))


def test_build_features_is_reproducible_for_same_input():
    raw_df = generate_sensor_data(
        num_machines=2,
        num_timesteps=80,
        fixed_seed=295,
    )

    features_df_1, retention_df_1 = build_features(raw_df)
    features_df_2, retention_df_2 = build_features(raw_df)

    pd.testing.assert_frame_equal(features_df_1, features_df_2)
    pd.testing.assert_frame_equal(retention_df_1, retention_df_2)


def test_save_feature_outputs_writes_files(tmp_path):
    raw_df = generate_sensor_data(
        num_machines=2,
        num_timesteps=80,
        fixed_seed=295,
    )

    features_df, retention_df = build_features(raw_df)

    features_output_path = tmp_path / "test_sensor_data_features.csv"
    retention_output_path = tmp_path / "test_feature_row_retention.csv"

    save_feature_outputs(
        features_df=features_df,
        retention_df=retention_df,
        features_output_path=features_output_path,
        retention_output_path=retention_output_path,
    )

    assert features_output_path.exists()
    assert retention_output_path.exists()

    saved_features_df = pd.read_csv(features_output_path)
    saved_retention_df = pd.read_csv(retention_output_path)

    assert len(saved_features_df) == len(features_df)
    assert len(saved_retention_df) == len(retention_df)
import pandas as pd

from generator import generate_sensor_data, sensors


def test_generate_sensor_data_returns_expected_shape():
    df = generate_sensor_data(
        num_machines=2,
        num_timesteps=5,
        fixed_seed=295,
    )

    assert df.shape[0] == 10


def test_generate_sensor_data_has_required_columns():
    df = generate_sensor_data(
        num_machines=2,
        num_timesteps=5,
        fixed_seed=295,
    )

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

    assert required_columns.issubset(set(df.columns))


def test_generate_sensor_data_has_sensor_anomaly_columns():
    df = generate_sensor_data(
        num_machines=2,
        num_timesteps=5,
        fixed_seed=295,
    )

    for sensor in sensors:
        assert f"{sensor}_anomaly" in df.columns


def test_generate_sensor_data_is_reproducible_with_same_seed():
    df_1 = generate_sensor_data(
        num_machines=2,
        num_timesteps=10,
        fixed_seed=295,
    )

    df_2 = generate_sensor_data(
        num_machines=2,
        num_timesteps=10,
        fixed_seed=295,
    )

    pd.testing.assert_frame_equal(df_1, df_2)


def test_generate_sensor_data_step_and_machine_ids_are_correct():
    df = generate_sensor_data(
        num_machines=2,
        num_timesteps=3,
        fixed_seed=295,
    )

    assert df["step"].tolist() == [1, 1, 2, 2, 3, 3]
    assert df["machine_id"].tolist() == [1, 2, 1, 2, 1, 2]
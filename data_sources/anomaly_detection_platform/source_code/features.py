import pandas as pd
import numpy as np

INPUT_PATH = 'outputs/sensor_data_raw.csv'
FEATURES_OUTPUT_PATH = 'outputs/sensor_data_features.csv'
RETENTION_OUTPUT_PATH = 'outputs/feature_row_retention.csv'
window = 10
long_window = 50
eps = 1e-6
osc_window = 20

sensors = ['temperature', 'pressure', 'vibration', 'flow_rate', 'voltage', 'current']

def slope(window_values):
    x = np.arange(len(window_values))
    return np.polyfit(x, window_values, 1)[0]

def build_features(df):
    df = df.copy()
    df.sort_values(by=['machine_id', 'step']).reset_index(drop=True)
    
    for sensor in sensors:
        sensor_group = df.groupby('machine_id', sort=False)[sensor]

        df[f'{sensor}_delta'] = sensor_group.diff()
        df[f'{sensor}_abs_delta'] = df[f'{sensor}_delta'].abs()

        delta_sign = np.sign(df[f'{sensor}_delta'])
        prev_sign = delta_sign.groupby(df['machine_id']).shift(1)

        new_run = (
            prev_sign.isna()
            | (delta_sign == 0)
            | (delta_sign != prev_sign)
        )

        run_id = new_run.groupby(df['machine_id']).cumsum()

        same_dir_run = df.groupby([df['machine_id'], run_id]).cumcount() + 1
        same_dir_run[delta_sign == 0] = 0

        df[f'{sensor}_same_dir_run'] = same_dir_run
        df[f'{sensor}_same_dir_run_10'] = same_dir_run.clip(upper=10)

        sign_flip = (delta_sign * prev_sign < 0).astype(int)
        df[f'{sensor}_sign_change_count'] = (
            sign_flip.groupby(df['machine_id'])
            .rolling(window)
            .sum()
            .reset_index(level=0, drop=True)
        )

        df[f'{sensor}_roll_mean'] = sensor_group.transform(
            lambda s: s.rolling(window).mean()
        )
        df[f'{sensor}_long_roll_mean'] = sensor_group.transform(
            lambda s: s.rolling(long_window).mean()
        )
        df[f'{sensor}_roll_std'] = sensor_group.transform(
            lambda s: s.rolling(window).std()
        )
        df[f'{sensor}_long_roll_std'] = sensor_group.transform(
            lambda s: s.rolling(long_window).std()
        )
        df[f'{sensor}_roll_min'] = sensor_group.transform(
            lambda s: s.rolling(window).min()
        )
        df[f'{sensor}_roll_max'] = sensor_group.transform(
            lambda s: s.rolling(window).max()
        )

        df[f'{sensor}_dev'] = df[sensor] - df[f'{sensor}_roll_mean']

        zscore_denom = df[f'{sensor}_roll_std'].where(
            df[f'{sensor}_roll_std'] > 1e-6, np.nan
        )
        df[f'{sensor}_zscore'] = (
            (df[sensor] - df[f'{sensor}_roll_mean']) / zscore_denom
        ).fillna(0)

        df[f'{sensor}_5_step_diff'] = sensor_group.diff(5)
        df[f'{sensor}_10_step_diff'] = sensor_group.diff(10)

        df[f'{sensor}_roll_avg_delta'] = (
            df.groupby('machine_id', sort=False)[f'{sensor}_delta']
            .rolling(window)
            .mean()
            .reset_index(level=0, drop=True)
        )

        df[f'{sensor}_short_long_mean_diff'] = (
            df[f'{sensor}_roll_mean'] - df[f'{sensor}_long_roll_mean']
        )

        df[f'{sensor}_abs_5_step_diff'] = sensor_group.diff(5).abs()
        df[f'{sensor}_abs_10_step_diff'] = sensor_group.diff(10).abs()
        df[f'{sensor}_roll_range'] = df[f'{sensor}_roll_max'] - df[f'{sensor}_roll_min']

        df[f'{sensor}_roll_mean_abs_delta'] = (
            df.groupby('machine_id', sort=False)[f'{sensor}_abs_delta']
            .rolling(window)
            .mean()
            .reset_index(level=0, drop=True)
        )

        var_denom = df[f'{sensor}_long_roll_std'].where(
            df[f'{sensor}_long_roll_std'] > 1e-6, np.nan
        )
        df[f'{sensor}_variability'] = (
            df[f'{sensor}_roll_std'] / var_denom
        ).fillna(0)

        df[f'{sensor}_slope_10'] = (
            sensor_group
            .rolling(10)
            .apply(slope, raw=True)
            .reset_index(level=0, drop=True)
        )

        df[f'{sensor}_slope_25'] = (
            sensor_group
            .rolling(25)
            .apply(slope, raw=True)
            .reset_index(level=0, drop=True)
        )

        df[f'{sensor}_cum_change_25'] = sensor_group.diff(25)
        df[f'{sensor}_long_base_dev'] = df[sensor] - df[f'{sensor}_long_roll_mean']

        centered_col = f'{sensor}_long_base_dev'
        grouped_centered = df.groupby('machine_id', sort=False)[centered_col]

        # ------------------------------------------------------------------
        # Oscillation-specific features
        # ------------------------------------------------------------------

        # 1) Lag autocorrelation on centered signal
        centered_sq_roll_mean = (
            (df[centered_col] ** 2)
            .groupby(df['machine_id'])
            .rolling(osc_window)
            .mean()
            .reset_index(level=0, drop=True)
        )

        lag_5_product = df[centered_col] * grouped_centered.shift(5)
        lag_10_product = df[centered_col] * grouped_centered.shift(10)

        df[f'{sensor}_lag_5_autocorr'] = (
            lag_5_product
            .groupby(df['machine_id'])
            .rolling(osc_window)
            .mean()
            .reset_index(level=0, drop=True)
            / (centered_sq_roll_mean + eps)
        )

        df[f'{sensor}_lag_10_autocorr'] = (
            lag_10_product
            .groupby(df['machine_id'])
            .rolling(osc_window)
            .mean()
            .reset_index(level=0, drop=True)
            / (centered_sq_roll_mean + eps)
        )

        # 2) Zero-crossing count on centered signal
        centered_sign = np.sign(df[centered_col])
        prev_centered_sign = centered_sign.groupby(df['machine_id']).shift(1)

        centered_zero_cross = (
            (centered_sign != 0)
            & (prev_centered_sign != 0)
            & (centered_sign != prev_centered_sign)
        ).astype(int)

        df[f'{sensor}_centered_zero_cross_count_10'] = (
            centered_zero_cross
            .groupby(df['machine_id'])
            .rolling(10)
            .sum()
            .reset_index(level=0, drop=True)
        )

        df[f'{sensor}_centered_zero_cross_count_20'] = (
            centered_zero_cross
            .groupby(df['machine_id'])
            .rolling(20)
            .sum()
            .reset_index(level=0, drop=True)
        )

        # 3) Direction imbalance
        pos_moves = (df[f'{sensor}_delta'] > 0).astype(int)
        neg_moves = (df[f'{sensor}_delta'] < 0).astype(int)

        pos_moves_10 = (
            pos_moves.groupby(df['machine_id'])
            .rolling(10)
            .sum()
            .reset_index(level=0, drop=True)
        )
        neg_moves_10 = (
            neg_moves.groupby(df['machine_id'])
            .rolling(10)
            .sum()
            .reset_index(level=0, drop=True)
        )

        pos_moves_20 = (
            pos_moves.groupby(df['machine_id'])
            .rolling(20)
            .sum()
            .reset_index(level=0, drop=True)
        )
        neg_moves_20 = (
            neg_moves.groupby(df['machine_id'])
            .rolling(20)
            .sum()
            .reset_index(level=0, drop=True)
        )

        df[f'{sensor}_dir_imbalance_10'] = (pos_moves_10 - neg_moves_10).abs() / 10
        df[f'{sensor}_dir_imbalance_20'] = (pos_moves_20 - neg_moves_20).abs() / 20

        # 4) Trend-vs-oscillation ratio
        df[f'{sensor}_trend_ratio_10'] = (
            df[f'{sensor}_slope_10'].abs() / (df[f'{sensor}_roll_std'] + eps)
        )
        df[f'{sensor}_trend_ratio_25'] = (
            df[f'{sensor}_slope_25'].abs() / (df[f'{sensor}_long_roll_std'] + eps)
        )

        # 5) Center-balance ratio
        centered_mean_10 = (
            grouped_centered
            .rolling(10)
            .mean()
            .reset_index(level=0, drop=True)
        )
        centered_abs_mean_10 = (
            df[centered_col].abs()
            .groupby(df['machine_id'])
            .rolling(10)
            .mean()
            .reset_index(level=0, drop=True)
        )

        centered_mean_20 = (
            grouped_centered
            .rolling(20)
            .mean()
            .reset_index(level=0, drop=True)
        )
        centered_abs_mean_20 = (
            df[centered_col].abs()
            .groupby(df['machine_id'])
            .rolling(20)
            .mean()
            .reset_index(level=0, drop=True)
        )

        df[f'{sensor}_center_balance_10'] = (
            centered_mean_10.abs() / (centered_abs_mean_10 + eps)
        )
        df[f'{sensor}_center_balance_20'] = (
            centered_mean_20.abs() / (centered_abs_mean_20 + eps)
        )

        # ------------------------------------------------------------------
        # Short-window centered signal helpers
        # ------------------------------------------------------------------

        df[f'{sensor}_centered_signal_t'] = df[sensor] - df[f'{sensor}_roll_mean']

        short_centered_group = df.groupby('machine_id', sort=False)[f'{sensor}_centered_signal_t']

        df[f'{sensor}_centered_signal_t-5'] = short_centered_group.shift(5)
        df[f'{sensor}_centered_signal_t-10'] = short_centered_group.shift(10)

    before_counts = df.groupby(['anomaly_type', 'target_sensor']).size().rename('before_drop')

    df = df.dropna()

    after_counts = df.groupby(['anomaly_type', 'target_sensor']).size().rename('after_drop')

    counts = pd.concat([before_counts, after_counts], axis=1).fillna(0)
    counts['before_drop'] = counts['before_drop'].astype(int)
    counts['after_drop'] = counts['after_drop'].astype(int)
    counts['rows_dropped'] = counts['before_drop'] - counts['after_drop']
    counts['pct_kept'] = counts['after_drop'] / counts['before_drop']
    
    counts = counts.reset_index()
    
    return df, counts

def save_feature_outputs(features_df, retention_df, features_output_path=FEATURES_OUTPUT_PATH, retention_output_path=RETENTION_OUTPUT_PATH):
    features_df.to_csv(features_output_path, index=False)
    retention_df.to_csv(retention_output_path, index=False)
    
def main():
    df = pd.read_csv(INPUT_PATH)
    
    features_df, retention_df = build_features(df)
    
    save_feature_outputs(features_df, retention_df)
    
    print(df.shape, features_df.shape, retention_df)
    
if __name__ == "__main__":
    main()
import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from config import OUTPUT_DIR, RANDOM_STATE, TEST_SIZE, MODEL_THRESHOLD


INPUT_CSV = OUTPUT_DIR / 'sensor_data_features.csv'
OUTPUT_DIR.mkdir(exist_ok=True)

LABEL_COL = 'any_anomaly'

ABLATION_GROUPS = {
    'lag_autocorr': [
        '_lag_5_autocorr',
        '_lag_10_autocorr',
    ],
    'zero_cross': [
        '_centered_zero_cross_count_10',
        '_centered_zero_cross_count_20',
    ],
    'center_balance': [
        '_center_balance_10',
        '_center_balance_20',
    ],
    'dir_imbalance': [
        '_dir_imbalance_10',
        '_dir_imbalance_20',
    ],
    'trend_ratio': [
        '_trend_ratio_10',
        '_trend_ratio_25',
    ],
}

PERMANENT_DROP_SUFFIXES = [
    '_dir_imbalance_10',
    '_dir_imbalance_20',
]

ABLATION_RUNS = {
    'final_model': [],
    "without_lag_autocorr": ABLATION_GROUPS["lag_autocorr"],
    "without_zero_cross": ABLATION_GROUPS["zero_cross"],
    "without_center_balance": ABLATION_GROUPS["center_balance"],
    "without_trend_ratio": ABLATION_GROUPS["trend_ratio"],
}


def find_cols_with_suffixes(columns, suffixes):
    cols_to_drop = []

    for col in columns:
        for suffix in suffixes:
            if col.endswith(suffix):
                cols_to_drop.append(col)
                break

    return cols_to_drop


def make_model():
    return RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        class_weight='balanced',
        n_jobs=-1
    )


def safe_recall(correct, total):
    if total == 0:
        return np.nan
    return correct / total


def evaluate_predictions(run_name, y_test, pred_series, meta_test):
    cm = metrics.confusion_matrix(y_test, pred_series, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    anomaly_recall = safe_recall(tp, tp + fn)

    osc_mask = (
        (y_test == 1) &
        (meta_test['anomaly_type'] == 'oscillation')
    )

    current_osc_mask = (
        osc_mask &
        (meta_test['target_sensor'] == 'current')
    )

    voltage_osc_mask = (
        osc_mask &
        (meta_test['target_sensor'] == 'voltage')
    )

    osc_total = osc_mask.sum()
    osc_correct = ((pred_series == 1) & osc_mask).sum()

    current_osc_total = current_osc_mask.sum()
    current_osc_correct = ((pred_series == 1) & current_osc_mask).sum()

    voltage_osc_total = voltage_osc_mask.sum()
    voltage_osc_correct = ((pred_series == 1) & voltage_osc_mask).sum()

    return {
        'run_name': run_name,

        'accuracy': metrics.accuracy_score(y_test, pred_series),

        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'true_positives': tp,

        'anomaly_recall': anomaly_recall,

        'oscillation_correct': osc_correct,
        'oscillation_total': osc_total,
        'oscillation_recall': safe_recall(osc_correct, osc_total),

        'current_osc_correct': current_osc_correct,
        'current_osc_total': current_osc_total,
        'current_osc_recall': safe_recall(current_osc_correct, current_osc_total),

        'voltage_osc_correct': voltage_osc_correct,
        'voltage_osc_total': voltage_osc_total,
        'voltage_osc_recall': safe_recall(voltage_osc_correct, voltage_osc_total),
    }


# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------

df = pd.read_csv(INPUT_CSV)

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna().copy()

if LABEL_COL not in df.columns:
    raise ValueError(f'Missing label column: {LABEL_COL}')


# ---------------------------------------------------------------------
# Build X, y, and metadata
# ---------------------------------------------------------------------

metadata_cols = [
    'step',
    'timestamp',
    'machine_id',
    'anomaly_type',
    'target_sensor',
]

non_feature_cols = [
    'step',
    'timestamp',
    'machine_id',
    LABEL_COL,
    'anomaly_type',
    'target_sensor',
]

# Remove per-sensor label columns from X.
for col in df.columns:
    if col.endswith('_anomaly'):
        non_feature_cols.append(col)

non_feature_cols = [
    col for col in non_feature_cols
    if col in df.columns
]

metadata_cols = [
    col for col in metadata_cols
    if col in df.columns
]

y = df[LABEL_COL]

permanent_drop_cols = find_cols_with_suffixes(
    df.columns,
    PERMANENT_DROP_SUFFIXES
)

drop_cols = non_feature_cols + permanent_drop_cols

X_full = df.drop(columns=drop_cols)

# Keep only numeric features.
X_full = X_full.select_dtypes(include=[np.number])

meta = df[metadata_cols].copy()


# ---------------------------------------------------------------------
# Train/test split
# ---------------------------------------------------------------------

train_idx, test_idx = train_test_split(
    df.index,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

y_train = y.loc[train_idx]
y_test = y.loc[test_idx]

meta_test = meta.loc[test_idx].copy()


# ---------------------------------------------------------------------
# Run ablations
# ---------------------------------------------------------------------

ablation_results = []

for run_name, suffixes_to_drop in ABLATION_RUNS.items():
    cols_to_drop = find_cols_with_suffixes(
        X_full.columns,
        suffixes_to_drop
    )

    X_run = X_full.drop(columns=cols_to_drop)

    X_train = X_run.loc[train_idx]
    X_test = X_run.loc[test_idx]

    model = make_model()
    model.fit(X_train, y_train)

    if hasattr(model, 'predict_proba'):
        anomaly_scores = model.predict_proba(X_test)[:, 1]
    else:
        anomaly_scores = model.predict(X_test)

    predictions = (anomaly_scores >= MODEL_THRESHOLD).astype(int)

    pred_series = pd.Series(
        predictions,
        index=test_idx,
        name='prediction'
    )

    score_series = pd.Series(
        anomaly_scores,
        index=test_idx,
        name='anomaly_score'
    )

    result = evaluate_predictions(
        run_name=run_name,
        y_test=y_test,
        pred_series=pred_series,
        meta_test=meta_test
    )

    result['num_features_used'] = X_run.shape[1]
    result['num_features_removed'] = len(cols_to_drop)
    result['removed_features'] = ', '.join(cols_to_drop)

    ablation_results.append(result)

    # Save predictions for this ablation run.
    predictions_df = df.loc[test_idx].copy()
    predictions_df['real_value'] = y_test
    predictions_df['prediction'] = pred_series
    predictions_df['anomaly_score'] = score_series
    predictions_df['ablation_run'] = run_name

    predictions_df.to_csv(
        OUTPUT_DIR / f'predictions_{run_name}.csv',
        index=False
    )

    # Keep evaluate.py working on the final model run.
    if run_name == 'final_model':
        predictions_df.to_csv(
            OUTPUT_DIR / 'predictions.csv',
            index=False
        )

    # Save feature importances.
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X_run.columns,
            'importance': model.feature_importances_
        })

        feature_importance = feature_importance.sort_values(
            by='importance',
            ascending=False
        )

        feature_importance.to_csv(
            OUTPUT_DIR / f'feature_importance_{run_name}.csv',
            index=False
        )

        # Keep your normal baseline feature importance file.
        if run_name == 'final_model':
            feature_importance.to_csv(
                OUTPUT_DIR / 'feature_importance.csv',
                index=False
            )


# ---------------------------------------------------------------------
# Save and print ablation results
# ---------------------------------------------------------------------

ablation_df = pd.DataFrame(ablation_results)
ablation_df.to_csv(OUTPUT_DIR / 'ablation_results.csv', index=False)

print('\nABLATION RESULTS')
print(
    ablation_df[
        [
            'run_name',
            'accuracy',
            'false_positives',
            'false_negatives',
            'anomaly_recall',
            'oscillation_recall',
            'current_osc_recall',
            'voltage_osc_recall',
            'num_features_used',
            'num_features_removed',
        ]
    ]
)
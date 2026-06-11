from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import json

from datetime import datetime
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from config import OUTPUT_DIR, RANDOM_STATE, TEST_SIZE, MODEL_THRESHOLD, ARTIFACT_DIR


INPUT_CSV = OUTPUT_DIR / "sensor_data_features.csv"

PREDICTIONS_OUTPUT_PATH = OUTPUT_DIR / "predictions.csv"

FEATURE_IMPORTANCE_OUTPUT_PATH = OUTPUT_DIR / "feature_importance.csv"
THRESHOLD_RESULTS_OUTPUT_PATH = OUTPUT_DIR / "threshold_results.csv"

LABEL_COL = "any_anomaly"

THRESHOLDS = [
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
]

ABLATION_GROUPS = {
    "lag_autocorr": [
        "_lag_5_autocorr",
        "_lag_10_autocorr",
    ],
    "zero_cross": [
        "_centered_zero_cross_count_10",
        "_centered_zero_cross_count_20",
    ],
    "center_balance": [
        "_center_balance_10",
        "_center_balance_20",
    ],
    "dir_imbalance": [
        "_dir_imbalance_10",
        "_dir_imbalance_20",
    ],
    "trend_ratio": [
        "_trend_ratio_10",
        "_trend_ratio_25",
    ],
}

PERMANENT_DROP_SUFFIXES = [
    "_dir_imbalance_10",
    "_dir_imbalance_20",
]

ABLATION_RUNS = {
    "final_model": [],
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
        class_weight="balanced",
        n_jobs=-1,
    )

def safe_recall(correct, total):
    if total == 0:
        return np.nan

    return correct / total

def evaluate_predictions(run_name, y_test, pred_series, meta_test, threshold=None):
    cm = metrics.confusion_matrix(y_test, pred_series, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    anomaly_recall = safe_recall(tp, tp + fn)

    precision = metrics.precision_score(
        y_test,
        pred_series,
        zero_division=0,
    )

    f1 = metrics.f1_score(
        y_test,
        pred_series,
        zero_division=0,
    )

    osc_mask = (
        (y_test == 1) &
        (meta_test["anomaly_type"] == "oscillation")
    )

    current_osc_mask = (
        osc_mask &
        (meta_test["target_sensor"] == "current")
    )

    voltage_osc_mask = (
        osc_mask &
        (meta_test["target_sensor"] == "voltage")
    )

    osc_total = osc_mask.sum()
    osc_correct = ((pred_series == 1) & osc_mask).sum()

    current_osc_total = current_osc_mask.sum()
    current_osc_correct = ((pred_series == 1) & current_osc_mask).sum()

    voltage_osc_total = voltage_osc_mask.sum()
    voltage_osc_correct = ((pred_series == 1) & voltage_osc_mask).sum()

    return {
        "run_name": run_name,
        "threshold": threshold,

        "accuracy": metrics.accuracy_score(y_test, pred_series),
        "precision": precision,
        "f1": f1,

        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,

        "anomaly_recall": anomaly_recall,

        "oscillation_correct": osc_correct,
        "oscillation_total": osc_total,
        "oscillation_recall": safe_recall(osc_correct, osc_total),

        "current_osc_correct": current_osc_correct,
        "current_osc_total": current_osc_total,
        "current_osc_recall": safe_recall(current_osc_correct, current_osc_total),

        "voltage_osc_correct": voltage_osc_correct,
        "voltage_osc_total": voltage_osc_total,
        "voltage_osc_recall": safe_recall(voltage_osc_correct, voltage_osc_total),
    }

def load_feature_data(input_csv=INPUT_CSV):
    """
    Load engineered feature data and remove invalid rows.
    """
    df = pd.read_csv(input_csv)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna().copy()

    if LABEL_COL not in df.columns:
        raise ValueError(f"Missing label column: {LABEL_COL}")

    return df

def prepare_model_inputs(df):
    """
    Build X, y, and metadata from the engineered feature dataframe.
    """
    metadata_cols = [
        "step",
        "timestamp",
        "machine_id",
        "anomaly_type",
        "target_sensor",
    ]

    non_feature_cols = [
        "step",
        "timestamp",
        "machine_id",
        LABEL_COL,
        "anomaly_type",
        "target_sensor",
    ]

    for col in df.columns:
        if col.endswith("_anomaly"):
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
        PERMANENT_DROP_SUFFIXES,
    )

    drop_cols = non_feature_cols + permanent_drop_cols

    X = df.drop(columns=drop_cols)

    X = X.select_dtypes(include=[np.number])

    meta = df[metadata_cols].copy()

    return X, y, meta

def split_train_test(df, y):
    """
    Create reproducible train/test indices.
    """
    train_idx, test_idx = train_test_split(
        df.index,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    return train_idx, test_idx

def train_model(X_train, y_train):
    """
    Train and return the final anomaly model.
    """
    model = make_model()
    model.fit(X_train, y_train)

    return model

def run_threshold_sweep(
    model,
    df,
    X_test,
    y_test,
    meta_test,
    test_idx,
):
    """
    Evaluate all thresholds and build the default-threshold predictions dataframe.
    """
    anomaly_scores = model.predict_proba(X_test)[:, 1]

    score_series = pd.Series(
        anomaly_scores,
        index=test_idx,
        name="anomaly_score",
    )

    threshold_results = []
    default_predictions_df = None

    for threshold in THRESHOLDS:
        predictions = (score_series >= threshold).astype(int)

        pred_series = pd.Series(
            predictions,
            index=test_idx,
            name="prediction",
        )

        result = evaluate_predictions(
            run_name=f"threshold_{threshold}",
            y_test=y_test,
            pred_series=pred_series,
            meta_test=meta_test,
            threshold=threshold,
        )

        threshold_results.append(result)

        if threshold == MODEL_THRESHOLD:
            predictions_df = df.loc[test_idx].copy()
            predictions_df["real_value"] = y_test
            predictions_df["prediction"] = pred_series
            predictions_df["anomaly_score"] = score_series
            predictions_df["threshold"] = threshold

            default_predictions_df = predictions_df

    threshold_df = pd.DataFrame(threshold_results)

    if default_predictions_df is None:
        raise ValueError(
            f"DEFAULT_THRESHOLD {MODEL_THRESHOLD} was not found in THRESHOLDS."
        )

    return threshold_df, default_predictions_df

def save_predictions(predictions_df, output_path=PREDICTIONS_OUTPUT_PATH):
    """
    Save default-threshold model predictions.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    predictions_df.to_csv(output_path, index=False)

def save_threshold_results(threshold_df, output_path=THRESHOLD_RESULTS_OUTPUT_PATH):
    """
    Save threshold sweep results.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    threshold_df.to_csv(output_path, index=False)

def save_feature_importance(
    model,
    feature_columns,
    output_path=FEATURE_IMPORTANCE_OUTPUT_PATH,
):
    """
    Save feature importances if the model exposes them.
    """
    if not hasattr(model, "feature_importances_"):
        return None

    feature_importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": model.feature_importances_,
        }
    )

    feature_importance = feature_importance.sort_values(
        by="importance",
        ascending=False,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    feature_importance.to_csv(output_path, index=False)

    return feature_importance

def print_threshold_results(threshold_df):
    """
    Print the threshold comparison report.
    """
    print("\nTHRESHOLD RESULTS")
    print(
        threshold_df[
            [
                "threshold",
                "accuracy",
                "precision",
                "f1",
                "false_positives",
                "false_negatives",
                "anomaly_recall",
                "oscillation_recall",
                "current_osc_recall",
                "voltage_osc_recall",
            ]
        ]
    )

def run_model_pipeline(
    input_csv=INPUT_CSV,
    predictions_output_path=PREDICTIONS_OUTPUT_PATH,
    threshold_results_output_path=THRESHOLD_RESULTS_OUTPUT_PATH,
    feature_importance_output_path=FEATURE_IMPORTANCE_OUTPUT_PATH,
):
    """
    Run the full model training and thresholding pipeline.
    """
    df = load_feature_data(input_csv)

    X, y, meta = prepare_model_inputs(df)

    train_idx, test_idx = split_train_test(df, y)

    y_train = y.loc[train_idx]
    y_test = y.loc[test_idx]

    X_train = X.loc[train_idx]
    X_test = X.loc[test_idx]

    meta_test = meta.loc[test_idx].copy()

    model = train_model(X_train, y_train)

    threshold_df, predictions_df = run_threshold_sweep(
        model=model,
        df=df,
        X_test=X_test,
        y_test=y_test,
        meta_test=meta_test,
        test_idx=test_idx,
    )
    
    ARTIFACT_DIR.mkdir(exist_ok=True)
    
    joblib.dump(model, ARTIFACT_DIR / "model.joblib")
    
    feature_columns = list(X_train.columns)
    
    with open(ARTIFACT_DIR / "feature_columns.json", "w") as file:
        json.dump(feature_columns, file, indent=4)
                
    metadata = {
        "model_type": "RandomForestClassifier",
        "threshold": MODEL_THRESHOLD,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "feature_count": len(feature_columns),
        "training_rows": len(X_train),
        "created_at": datetime.now().isoformat()
    }
    
    with open(ARTIFACT_DIR / "model_metadata.json", "w") as file:
        json.dump(metadata, file, indent=4)

    save_predictions(
        predictions_df=predictions_df,
        output_path=predictions_output_path,
    )

    save_threshold_results(
        threshold_df=threshold_df,
        output_path=threshold_results_output_path,
    )

    feature_importance_df = save_feature_importance(
        model=model,
        feature_columns=X.columns,
        output_path=feature_importance_output_path,
    )

    print_threshold_results(threshold_df)

    print(f"\nSaved predictions to {predictions_output_path}")
    print(f"Saved threshold results to {threshold_results_output_path}")
    print(f"Saved feature importance to {feature_importance_output_path}")

    return model, threshold_df, predictions_df, feature_importance_df

def main():
    run_model_pipeline()

if __name__ == "__main__":
    main()
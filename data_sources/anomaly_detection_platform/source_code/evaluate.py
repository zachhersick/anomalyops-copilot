from pathlib import Path

import pandas as pd
from sklearn import metrics


PREDICTIONS_INPUT_PATH = "outputs/predictions.csv"
RETENTION_INPUT_PATH = "outputs/feature_row_retention.csv"
FEATURE_IMPORTANCE_INPUT_PATH = "outputs/feature_importance.csv"

DEBUG_OSCILLATION_DETAILS = False

OSC_FEATURE_SUFFIXES = [
    "lag_5_autocorr",
    "lag_10_autocorr",
    "centered_zero_cross_count_10",
    "centered_zero_cross_count_20",
    "dir_imbalance_10",
    "dir_imbalance_20",
    "trend_ratio_10",
    "trend_ratio_25",
    "center_balance_10",
    "center_balance_20",
]


def load_evaluation_inputs(
    predictions_path=PREDICTIONS_INPUT_PATH,
    retention_path=RETENTION_INPUT_PATH,
    feature_importance_path=FEATURE_IMPORTANCE_INPUT_PATH,
):
    """
    Load the files needed for the evaluation report.
    """
    predictions_df = pd.read_csv(predictions_path)
    retention_df = pd.read_csv(retention_path)

    try:
        feature_importance_df = pd.read_csv(feature_importance_path)
    except FileNotFoundError:
        feature_importance_df = None

    return predictions_df, retention_df, feature_importance_df


def build_recall_table(df, mask, group_col):
    """
    Build a recall summary table for one filtered subset of prediction rows.
    """
    rows = df[mask]

    summary = rows.groupby(by=group_col).agg(
        total=("prediction", "size"),
        correct=("prediction", lambda x: (x == 1).sum()),
        missed=("prediction", lambda x: (x == 0).sum()),
    )

    summary["recall"] = summary["correct"] / summary["total"]

    return summary


def build_evaluation_tables(predictions_df):
    """
    Build grouped recall tables and false-positive/false-negative dataframes.
    """
    anomaly_mask = predictions_df["real_value"] == 1
    drift_mask = anomaly_mask & (predictions_df["anomaly_type"] == "drift")
    oscillation_mask = anomaly_mask & (predictions_df["anomaly_type"] == "oscillation")

    anomaly_by_type = build_recall_table(
        df=predictions_df,
        mask=anomaly_mask,
        group_col="anomaly_type",
    )

    anomaly_by_sensor = build_recall_table(
        df=predictions_df,
        mask=anomaly_mask,
        group_col="target_sensor",
    )

    drift_by_sensor = build_recall_table(
        df=predictions_df,
        mask=drift_mask,
        group_col="target_sensor",
    )

    oscillation_by_sensor = build_recall_table(
        df=predictions_df,
        mask=oscillation_mask,
        group_col="target_sensor",
    )

    false_positives = predictions_df[
        (predictions_df["real_value"] == 0)
        & (predictions_df["prediction"] == 1)
    ]

    false_negatives = predictions_df[
        (predictions_df["real_value"] == 1)
        & (predictions_df["prediction"] == 0)
    ]

    return {
        "anomaly_by_type": anomaly_by_type,
        "anomaly_by_sensor": anomaly_by_sensor,
        "drift_by_sensor": drift_by_sensor,
        "oscillation_by_sensor": oscillation_by_sensor,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def build_overall_metrics(predictions_df):
    """
    Build confusion matrix and overall classification metrics.
    """
    y_test = predictions_df["real_value"]
    predictions = predictions_df["prediction"]

    confusion_matrix = metrics.confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    )

    tn, fp, fn, tp = confusion_matrix.ravel()

    return {
        "confusion_matrix": confusion_matrix,
        "accuracy": metrics.accuracy_score(y_test, predictions),
        "precision": metrics.precision_score(y_test, predictions, zero_division=0),
        "recall": metrics.recall_score(y_test, predictions, zero_division=0),
        "f1": metrics.f1_score(y_test, predictions, zero_division=0),
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "true_negatives": tn,
    }


def debug_oscillation_sensor(predictions_df, sensor):
    """
    Print feature comparisons for correctly detected vs missed oscillation rows.
    """
    sensor_rows = predictions_df.loc[
        (predictions_df["real_value"] == 1)
        & (predictions_df["anomaly_type"] == "oscillation")
        & (predictions_df["target_sensor"] == sensor)
    ]

    correct_rows = sensor_rows.loc[sensor_rows["prediction"] == 1]
    missed_rows = sensor_rows.loc[sensor_rows["prediction"] == 0]

    compare_cols = [
        f"{sensor}_{suffix}"
        for suffix in OSC_FEATURE_SUFFIXES
        if f"{sensor}_{suffix}" in predictions_df.columns
    ]

    if not compare_cols:
        print(f"\n--- OSCILLATION DEBUG: {sensor} ---")
        print("No matching oscillation feature columns found.")
        return

    print(f"\n--- OSCILLATION DEBUG: {sensor} ---")
    print("correct:", len(correct_rows))
    print("missed:", len(missed_rows))
    print("columns:", compare_cols)

    print("\ncorrect describe")
    print(correct_rows[compare_cols].describe())

    print("\nmissed describe")
    print(missed_rows[compare_cols].describe())

    print("\nmean difference (correct - missed)")
    print(correct_rows[compare_cols].mean() - missed_rows[compare_cols].mean())

    print("\ncorrect NaN counts")
    print(correct_rows[compare_cols].isna().sum())

    print("\nmissed NaN counts")
    print(missed_rows[compare_cols].isna().sum())


def print_evaluation_report(
    predictions_df,
    retention_df,
    feature_importance_df,
    evaluation_tables,
    overall_metrics,
    debug_oscillation_details=DEBUG_OSCILLATION_DETAILS,
):
    """
    Print the full evaluation report.
    """
    false_positives = evaluation_tables["false_positives"]
    false_negatives = evaluation_tables["false_negatives"]

    print("\n==============================")
    print("ROW RETENTION")
    print("==============================")
    print(retention_df)

    print("\n==============================")
    print("FINAL MODEL EVALUATION")
    print("==============================")

    if "threshold" in predictions_df.columns:
        print("threshold:", predictions_df["threshold"].iloc[0])

    print("\nConfusion Matrix")
    print(overall_metrics["confusion_matrix"])

    print("\nOverall Metrics")
    print("accuracy:", overall_metrics["accuracy"])
    print("precision:", overall_metrics["precision"])
    print("recall:", overall_metrics["recall"])
    print("f1:", overall_metrics["f1"])
    print("false positives:", overall_metrics["false_positives"])
    print("false negatives:", overall_metrics["false_negatives"])
    print("true positives:", overall_metrics["true_positives"])
    print("true negatives:", overall_metrics["true_negatives"])

    print("\nRecall by Anomaly Type")
    print(evaluation_tables["anomaly_by_type"].sort_values(by="recall"))

    print("\nRecall by Target Sensor")
    print(evaluation_tables["anomaly_by_sensor"].sort_values(by="recall"))

    print("\nDrift Recall by Target Sensor")
    print(evaluation_tables["drift_by_sensor"].sort_values(by="recall"))

    print("\nOscillation Recall by Target Sensor")
    print(evaluation_tables["oscillation_by_sensor"].sort_values(by="recall"))

    print("\nFalse Positive Count:", len(false_positives))
    print("False Negative Count:", len(false_negatives))

    print("\nFalse Negatives by Anomaly Type")
    print(false_negatives["anomaly_type"].value_counts())

    print("\nFalse Negatives by Target Sensor")
    print(false_negatives["target_sensor"].value_counts())

    if feature_importance_df is not None:
        print("\nTop 20 Feature Importances")
        print(feature_importance_df.head(20))
    else:
        print("\nNo feature_importance.csv found.")

    if debug_oscillation_details:
        debug_oscillation_sensor(predictions_df, "current")
        debug_oscillation_sensor(predictions_df, "voltage")


def run_evaluation(
    predictions_path=PREDICTIONS_INPUT_PATH,
    retention_path=RETENTION_INPUT_PATH,
    feature_importance_path=FEATURE_IMPORTANCE_INPUT_PATH,
    debug_oscillation_details=DEBUG_OSCILLATION_DETAILS,
):
    """
    Run the full evaluation report pipeline.
    """
    predictions_df, retention_df, feature_importance_df = load_evaluation_inputs(
        predictions_path=predictions_path,
        retention_path=retention_path,
        feature_importance_path=feature_importance_path,
    )

    evaluation_tables = build_evaluation_tables(predictions_df)
    overall_metrics = build_overall_metrics(predictions_df)

    print_evaluation_report(
        predictions_df=predictions_df,
        retention_df=retention_df,
        feature_importance_df=feature_importance_df,
        evaluation_tables=evaluation_tables,
        overall_metrics=overall_metrics,
        debug_oscillation_details=debug_oscillation_details,
    )

    return {
        "predictions_df": predictions_df,
        "retention_df": retention_df,
        "feature_importance_df": feature_importance_df,
        "evaluation_tables": evaluation_tables,
        "overall_metrics": overall_metrics,
    }


def main():
    run_evaluation()


if __name__ == "__main__":
    main()
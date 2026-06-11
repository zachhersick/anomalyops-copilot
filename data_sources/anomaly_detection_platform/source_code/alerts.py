import pandas as pd

#model prediction + anomaly score + sensor value + safety value + safety rule = alert record

INPUT_FILE = 'outputs/predictions.csv'
OUTPUT_FILE = 'outputs/alerts.csv'

SENSORS = [
    'temperature',
    'pressure',
    'vibration',
    'flow_rate',
    'voltage',
    'current'
]

req_cols = [
    'step',
    'machine_id',
    'prediction',
    'anomaly_score',
    'target_sensor',
    'anomaly_type',
    'temperature',
    'pressure',
    'vibration',
    'flow_rate',
    'voltage',
    'current'
]

eval_cols = [
    'real_value',
    'threshold',
    '*_anomaly',
]

SEVERITY_INFO = 'INFO'
SEVERITY_WARNING = 'WARNING'
SEVERITY_CRITICAL = 'CRITICAL'

ALERT_TYPE_MODEL = 'model_anomaly'
ALERT_TYPE_THRESHOLD = 'threshold_violation'
ALERT_TYPE_MODEL_AND_THRESHOLD = 'model_and_threshold'

ALERT_STATUS_OPEN = 'OPEN'

SAFETY_THRESHOLDS = {
    'temperature': {
        'critical_low': 40.0,
        'warning_low': 50.0,
        'warning_high': 110.0,
        'critical_high': 125.0,
        'unit': 'F',
    },

    'pressure': {
        'critical_low': 15.0,
        'warning_low': 25.0,
        'warning_high': 90.0,
        'critical_high': 105.0,
        'unit': 'PSI',
    },

    'vibration': {
        'warning_high': 12.0,
        'critical_high': 16.0,
        'unit': 'mm/s',
    },

    'flow_rate': {
        'critical_low': 0.20,
        'warning_low': 0.40,
        'warning_high': 2.30,
        'critical_high': 2.75,
        'unit': 'm^3/s',
    },

    'voltage': {
        'critical_low': 90.0,
        'warning_low': 105.0,
        'warning_high': 135.0,
        'critical_high': 145.0,
        'unit': 'V',
    },

    'current': {
        'critical_low': 2.0,
        'warning_low': 4.0,
        'warning_high': 23.0,
        'critical_high': 28.0,
        'unit': 'A',
    },
}

SEVERITY_PRIORITY = {
    SEVERITY_INFO: 1,
    SEVERITY_WARNING: 2,
    SEVERITY_CRITICAL: 3,
}

REASON_TEMPLATES = {
    'critical_high': (
        '{sensor} value {value:.3f} {unit} exceeded critical high threshold '
        '{threshold:.3f} {unit}.'
    ),

    'critical_low': (
        '{sensor} value {value:.3f} {unit} fell below critical low threshold '
        '{threshold:.3f} {unit}.'
    ),

    'warning_high': (
        '{sensor} value {value:.3f} {unit} exceeded warning high threshold '
        '{threshold:.3f} {unit}.'
    ),

    'warning_low': (
        '{sensor} value {value:.3f} {unit} fell below warning low threshold '
        '{threshold:.3f} {unit}.'
    ),

    'model_high_score': (
        'Model predicted anomaly on {sensor} with high anomaly score '
        '{score:.3f}. Sensor value was {value:.3f} {unit}.'
    ),

    'model_medium_score': (
        'Model predicted anomaly on {sensor} with anomaly score '
        '{score:.3f}. Sensor value was {value:.3f} {unit}.'
    ),

    'model_low_score': (
        'Model predicted anomaly on {sensor} with low anomaly score '
        '{score:.3f}. Sensor value was {value:.3f} {unit}.'
    ),

    'unknown_sensor': (
        'Model predicted anomaly, but target sensor "{sensor}" was not recognized.'
    ),
}

THRESHOLD_SEVERITY = {
    'critical_high': SEVERITY_CRITICAL,
    'critical_low': SEVERITY_CRITICAL,
    'warning_high': SEVERITY_WARNING,
    'warning_low': SEVERITY_WARNING,
}

THRESHOLD_CHECK_ORDER = [
    'critical_high',
    'critical_low',
    'warning_high',
    'warning_low',
]

WARNING_SCORE_THRESHOLD = 0.35
HIGH_SCORE_THRESHOLD = 0.70
CRITICAL_SCORE_THRESHOLD = 0.90

def check_violation(row, sensor):
    if sensor not in SAFETY_THRESHOLDS:
        reason = REASON_TEMPLATES['unknown_sensor'].format(
            sensor=sensor,
        )
        return None, reason
    
    value = row[sensor]
    thresholds = SAFETY_THRESHOLDS[sensor]
    unit = thresholds['unit']
    
    for threshold_key in THRESHOLD_CHECK_ORDER:
        if threshold_key not in thresholds:
            continue
        
        threshold_value = thresholds[threshold_key]
        
        if threshold_key.endswith('high') and value > threshold_value:
            severity = THRESHOLD_SEVERITY[threshold_key]
            reason = REASON_TEMPLATES[threshold_key].format(
                sensor=sensor,
                value=value,
                threshold=threshold_value,
                unit=unit
            )
            return severity, reason
            
        if threshold_key.endswith('low') and value < threshold_value:
            severity = THRESHOLD_SEVERITY[threshold_key]
            reason = REASON_TEMPLATES[threshold_key].format(
                sensor=sensor,
                value=value,
                threshold=threshold_value,
                unit=unit
            )
            return severity, reason

    return None, None

def assign_model_severity(score):
    if score >= HIGH_SCORE_THRESHOLD:
        return SEVERITY_WARNING
    return SEVERITY_INFO

def build_model_reason(row, sensor):
    value = row[sensor]
    score = row['anomaly_score']
    unit = SAFETY_THRESHOLDS[sensor]['unit']
    
    if score >= HIGH_SCORE_THRESHOLD:
        template_key = 'model_high_score'
    
    elif score >= WARNING_SCORE_THRESHOLD:
        template_key = 'model_medium_score'
    
    else:
        template_key = 'model_low_score'
        
    reason = REASON_TEMPLATES[template_key].format(
        sensor=sensor,
        score=score,
        value=value,
        unit=unit
    )
        
    return reason
    
def build_alert(row, alert_id):
    sensor = row['target_sensor']
    
    if sensor not in SENSORS:
        reason = REASON_TEMPLATES['unknown_sensor'].format(
            sensor=sensor
        )
        
        alert = {
            'alert_id': alert_id,
            'step': row['step'],
            'machine_id': row['machine_id'],
            'sensor': sensor,
            'sensor_value': None,
            'prediction': row['prediction'],
            'anomaly_score': row['anomaly_score'],
            'severity': SEVERITY_WARNING,
            'alert_type': ALERT_TYPE_MODEL,
            'reason': reason,
            'status': ALERT_STATUS_OPEN,
            'anomaly_type': row['anomaly_type'],
            'real_value': row['real_value'] if 'real_value' in row.index else None,
        }
        
        return alert
    
    value = row[sensor]
    score = row['anomaly_score']
    
    threshold_severity, threshold_reason = check_violation(row, sensor)
    
    if threshold_severity is not None:
        severity = threshold_severity
        alert_type = ALERT_TYPE_MODEL_AND_THRESHOLD
        reason = threshold_reason
    
    else:
        severity = assign_model_severity(score)
        alert_type = ALERT_TYPE_MODEL
        reason = build_model_reason(row, sensor)
        
    alert = {
        'alert_id': alert_id,
        'step': row['step'],
        'machine_id': row['machine_id'],
        'sensor': sensor,
        'sensor_value': value,
        'prediction': row['prediction'],
        'anomaly_score': score,
        'severity': severity,
        'alert_type': alert_type,
        'reason': reason,
        'status': ALERT_STATUS_OPEN,
        'anomaly_type': row['anomaly_type'],
        'real_value': row['real_value'] if 'real_value' in row.index else None,
    }
        
    return alert

def print_alert_summary(alerts):
    print("\n==============================")
    print("ALERT SUMMARY")
    print("==============================")

    print("\nShape")
    print(alerts.shape)

    if alerts.empty:
        print("\nNo alerts generated.")
        return

    print("\nSeverity Counts")
    print(alerts["severity"].value_counts())

    print("\nAlert Type Counts")
    print(alerts["alert_type"].value_counts())

    print("\nSensor Counts")
    print(alerts["sensor"].value_counts())

    print("\nTop Critical Alerts")
    critical_alerts = alerts[alerts["severity"] == SEVERITY_CRITICAL]
    print(
        critical_alerts
        .sort_values(by="anomaly_score", ascending=False)
        .head(10)
    )

def load_predictions(input_file=INPUT_FILE):
    df = pd.read_csv(input_file)

    missing_cols = [
        col for col in req_cols
        if col not in df.columns
    ]

    if missing_cols:
        raise ValueError(f'Missing required columns: {missing_cols}')
    
    return df

def build_alerts(predictions_df):
    pred_anom_rows = predictions_df[
        (predictions_df['prediction'] == 1)
    ]

    alertsList = []
    alert_id = 1

    for index, row in pred_anom_rows.iterrows():
        alert = build_alert(row, alert_id)
    
        if alert is not None:
            alertsList.append(alert)
            alert_id += 1
    
    alerts = pd.DataFrame(alertsList)
    
    return alerts

def save_alerts(alerts_df, output_file=OUTPUT_FILE):    
    alerts_df.to_csv(output_file, index=False)

def run_alert_pipeline(input_file=INPUT_FILE, output_file=OUTPUT_FILE):
    predictions_df = load_predictions(input_file)
    alerts_df = build_alerts(predictions_df)
    save_alerts(alerts_df, output_file)
    print_alert_summary(alerts_df)
    
    return alerts_df

def main():
    run_alert_pipeline()

if __name__ == "__main__":
    main()
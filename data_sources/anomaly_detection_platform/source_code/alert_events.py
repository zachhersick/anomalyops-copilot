import pandas as pd

INPUT_FILE = 'outputs/alerts.csv'
OUTPUT_FILE = 'outputs/alert_events.csv'

MAX_STEP_GAP = 3
SEVERITY_PRIORITY = {
    'INFO': 1,
    'WARNING': 2,
    'CRITICAL': 3
}

input_req_cols = [
    'alert_id',
    'step',
    'machine_id',
    'sensor',
    'sensor_value',
    'prediction',
    'anomaly_score',
    'severity',
    'alert_type',
    'reason',
    'status',
    'anomaly_type',
    'real_value'
]

output_cols = [
    'event_id',
    'machine_id',
    'sensor',
    'anomaly_type',
    'start_step',
    'end_step',
    'duration',
    'alert_count',
    'max_severity',
    'max_severity_reason',
    'max_anomaly_score',
    'mean_anomaly_score',
    'min_sensor_value',
    'max_sensor_value',
    'first_reason',
    'status',
    'real_value',
]

def load_alerts(input_file=INPUT_FILE):
    df = pd.read_csv(input_file)
    for col in input_req_cols:
        if col not in df.columns:
            raise KeyError(f'Column not found: {col}')
    
    return df

def get_severity_priority(severity):
    if severity in SEVERITY_PRIORITY:
        return SEVERITY_PRIORITY[severity]
    else:
        return 0
    
def same_event(current_event, row):
    step_gap = row['step'] - current_event['end_step']
    if (current_event['machine_id'] == row['machine_id'] and
        current_event['sensor'] == row['sensor'] and
        current_event['anomaly_type'] == row['anomaly_type'] and
        step_gap <= MAX_STEP_GAP and
        step_gap >= 0
    ):
        return True
    else:
        return False
    
def safe_min(current_min, new_value):
    if pd.isna(current_min):
        return new_value

    if pd.isna(new_value):
        return current_min

    return min(current_min, new_value)
    
def safe_max(current_max, new_value):
    if pd.isna(current_max):
        return new_value

    if pd.isna(new_value):
        return current_max

    return max(current_max, new_value)
    
def start_event(row, event_id):
    event = {}
    
    event['machine_id'] = row['machine_id']
    event['event_id'] = event_id
    event['sensor'] = row['sensor']
    event['anomaly_type'] = row['anomaly_type']
    
    event['start_step'] = row['step']
    event['end_step'] = row['step']
    
    event['duration'] = 1
    event['alert_count'] = 1
    
    event['max_severity'] = row['severity']
    event['max_severity_reason'] = row['reason']
    
    event['max_anomaly_score'] = row['anomaly_score']
    event['score_sum'] = row['anomaly_score']
    event['mean_anomaly_score'] = row['anomaly_score']
    
    event['min_sensor_value'] = row['sensor_value']
    event['max_sensor_value'] = row['sensor_value']
    
    event['first_reason'] = row['reason']
    event['status'] = row['status']
    event['real_value'] = row['real_value']
    
    return event

def update_event(event, row):
    row_priority = get_severity_priority(row['severity'])
    event_priority = get_severity_priority(event['max_severity'])
    
    event['end_step'] = row['step']
    
    event['alert_count'] += 1
    
    event['duration'] = event['end_step'] - event['start_step'] + 1
    
    if row['anomaly_score'] > event['max_anomaly_score']:
        event['max_anomaly_score'] = row['anomaly_score']
        
    event['score_sum'] = event['score_sum'] + row['anomaly_score']
    
    event['mean_anomaly_score'] = event['score_sum'] / event['alert_count']
    
    event['min_sensor_value'] = safe_min(
        event['min_sensor_value'],
        row['sensor_value']
    )
    
    event['max_sensor_value'] = safe_max(
        event['max_sensor_value'],
        row['sensor_value']
    )
    
    if row_priority > event_priority:
        event['max_severity'] = row['severity']
        event['max_severity_reason'] = row['reason']
        
    return event

def finalize_event(event):
    del event['score_sum']
    return event

def print_event_summary(events_df):
    print("\n==============================")
    print("EVENT SUMMARY")
    print("==============================")

    print("\nShape")
    print(events_df.shape)

    if events_df.empty:
        print("\nNo alert events generated.")
        return

    print("\nMax Severity Counts")
    print(events_df["max_severity"].value_counts())

    print("\nAnomaly Type Counts")
    print(events_df["anomaly_type"].value_counts())

    print("\nSensor Counts")
    print(events_df["sensor"].value_counts())

    print("\nTop 10 Longest Events")
    print(events_df.sort_values(by="duration", ascending=False).head(10))

    print("\nTop 10 Highest Score Events")
    print(events_df.sort_values(by="max_anomaly_score", ascending=False).head(10))

    print("\nCritical Events")
    print(events_df[events_df["max_severity"] == "CRITICAL"])
    
def group_alert_events(alerts):
    alerts = alerts.sort_values(by=['machine_id', 'sensor', 'anomaly_type', 'step'])

    events = []
    current_event = None
    event_id = 1

    for i, row in alerts.iterrows():
        if current_event is None:
            current_event = start_event(row, event_id)
            continue
        
        if same_event(current_event, row):
            current_event = update_event(current_event, row)
            
        else:
            finalized_event = finalize_event(current_event)
            events.append(finalized_event)
            
            event_id += 1
            current_event = start_event(row, event_id)        
        
    if current_event is not None:
        finalized_event = finalize_event(current_event)
        events.append(finalized_event)
        
    if not events:
        return pd.DataFrame(columns=output_cols)
        
    events_df = pd.DataFrame(events)

    missing_output_cols = []

    for col in output_cols:
        if col not in events_df.columns:
            missing_output_cols.append(col)

    if missing_output_cols:
        raise ValueError(f"Missing output columns: {missing_output_cols}")

    events_df = events_df[output_cols]
    
    return events_df

def save_alert_events(events_df, output_file=OUTPUT_FILE):
    events_df.to_csv(output_file, index=False)
    
def run_alert_event_pipeline(input_file=INPUT_FILE, output_file=OUTPUT_FILE):
    alerts_df = load_alerts(input_file)

    events_df = group_alert_events(alerts_df)

    print_event_summary(events_df)

    save_alert_events(events_df, output_file)
    
    return events_df
    
def main():
    run_alert_event_pipeline()
    
    
if __name__ == "__main__":
    main()
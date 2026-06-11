from generator import main as run_generator
from features import main as run_features
from model import main as run_model
from evaluate import main as run_evaluate
from alerts import main as run_alerts
from alert_events import main as run_alert_events

PIPELINE_STEPS = [
    ("Generate synthetic sensor data", run_generator),
    ("Build features", run_features),
    ("Train model and write predictions", run_model),
    ("Evaluate model", run_evaluate),
    ("Generate row-level alerts", run_alerts),
    ("Group alert events", run_alert_events),
]

def run_step(step_name, step_function):
    print("\n==============================")
    print(f"Running {step_name}")
    print("==============================")

    step_function()
    
def run_pipeline(steps=PIPELINE_STEPS):
    for step_name, step_function in steps:
        try:
            run_step(step_name, step_function)
        except Exception as error:
            print("\nPipeline failed.")
            print(f"Failed step: {step_name}")
            print(f"Error: {error}")
            raise

    print("\n==============================")
    print("Pipeline completed successfully")
    print("==============================")
        
def main():
    run_pipeline()

if __name__ == "__main__":
    main()
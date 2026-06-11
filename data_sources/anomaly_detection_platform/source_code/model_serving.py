import joblib
import json
import pandas as pd

from config import ARTIFACT_DIR


def load_model_artifacts():
    model = joblib.load(ARTIFACT_DIR / "model.joblib")

    with open(ARTIFACT_DIR / "feature_columns.json") as file:
        feature_columns = json.load(file)
    
    with open(ARTIFACT_DIR / "model_metadata.json") as file:
        metadata = json.load(file)
        
    return model, feature_columns, metadata


def prepare_feature_row(feature_dict, feature_columns):
    df = pd.DataFrame([feature_dict])
    
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
            
    df = df[feature_columns]

    return df


def predict_feature_row(feature_dict):
    model, feature_columns, metadata = load_model_artifacts()
    
    df = prepare_feature_row(feature_dict, feature_columns)
    
    score = float(model.predict_proba(df)[0,1])
    
    prediction = int(score >= metadata['threshold'])
    
    return {
        "prediction": prediction,
        "anomaly_score": score,
        "threshold": metadata["threshold"],
        "is_anomaly": bool(prediction),
    }
    
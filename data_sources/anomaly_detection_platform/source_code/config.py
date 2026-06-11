from pathlib import Path
import os

DB_PATH = Path(os.getenv("DB_PATH", "anomaly_detection.db"))

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

MODEL_THRESHOLD = float(os.getenv("MODEL_THRESHOLD", 0.35))

RANDOM_STATE = int(os.getenv("RANDOM_STATE", 42))
TEST_SIZE = float(os.getenv("TEST_SIZE", 0.2))

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs"))
ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", "artifacts"))
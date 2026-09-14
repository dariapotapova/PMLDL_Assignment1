"""
Stage 2: Model Engineering — Palmer Penguins.
Feature engineering, training, evaluation, MLflow logging, packaging.
Preprocessing and the model form ONE sklearn Pipeline, so the saved
artifact applies identical transformations at inference time.
"""

import os

import json
import logging
from datetime import datetime
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from mlflow.tracking import MlflowClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "model.pkl"
MLFLOW_DIR = PROJECT_ROOT / "mlruns"
MLFLOW_DB_PATH = MLFLOW_DIR / "mlflow.db"

# Tracking URI comes from the environment when running under Airflow
# (Postgres); local runs fall back to SQLite. Configuration outside code.
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{MLFLOW_DB_PATH.as_posix()}",
)
TARGET = "species"
NUMERIC = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
CATEGORICAL = ["island", "sex"]
FEATURES = NUMERIC + CATEGORICAL
RANDOM_STATE = 42

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    log.info("Loaded train=%d rows, test=%d rows", len(train), len(test))
    return train, test


def build_pipeline() -> Pipeline:
    """Preprocessing + model in a single object.

    - Transformers are fitted on TRAIN only -> no data leakage;
    - The pickle carries preprocessing too -> the API never duplicates
      transformation logic (no train/serve skew).
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC),
            # handle_unknown="ignore": an unseen category at inference time
            # (e.g. an unexpected value sent to the API) becomes zeros, not a crash.
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def evaluate(pipeline: Pipeline, test: pd.DataFrame) -> tuple[dict, str]:
    X_test, y_test = test[FEATURES], test[TARGET]
    preds = pipeline.predict(X_test)

    # macro-F1 weights classes equally: with imbalanced classes, accuracy
    # alone can hide poor performance on the smallest class.
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "f1_macro": f1_score(y_test, preds, average="macro"),
    }
    report = classification_report(y_test, preds)
    log.info("Test metrics: %s", {k: round(v, 4) for k, v in metrics.items()})
    return metrics, report


def log_to_mlflow(pipeline: Pipeline, train: pd.DataFrame, test: pd.DataFrame,
                  metrics: dict, report: str) -> str:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    report_path = MODELS_DIR / "classification_report.txt"
    report_path.write_text(report)
    metrics_path = MODELS_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("penguins")
    # Create the experiment explicitly with an absolute artifact location,
    # so artifacts land in <repo>/mlruns regardless of the working directory
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    if client.get_experiment_by_name("penguins") is None:
        client.create_experiment("penguins", artifact_location=MLFLOW_DIR.as_uri())
    mlflow.set_experiment("penguins")

    with mlflow.start_run(run_name=f"train-{datetime.now():%Y%m%d-%H%M%S}") as run:
        mlflow.log_params({
            "model": "LogisticRegression",
            "n_train_rows": len(train),
            "n_test_rows": len(test),
            "n_features": len(NUMERIC) + 5,
            "random_state": RANDOM_STATE,
        })
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(report_path), artifact_path="reports")
        mlflow.log_artifact(str(metrics_path), artifact_path="reports")
        mlflow.sklearn.log_model(pipeline, artifact_path="model")
        run_id = run.info.run_id

    log.info("MLflow run logged: %s", run_id)
    return run_id


def package_and_verify(pipeline: Pipeline) -> None:
    # Package the WHOLE pipeline (preprocessing + model) as one artifact
    # for the API container.
    joblib.dump(pipeline, MODEL_PATH)
    log.info("Model saved: %s (%.1f KB)", MODEL_PATH, MODEL_PATH.stat().st_size / 1024)

    # Self-check: reload the artifact and predict one row with the exact
    # field structure the API will receive — a broken pickle is caught
    # here, before deployment, not inside the container.
    loaded = joblib.load(MODEL_PATH)
    sample = pd.DataFrame([{
        "bill_length_mm": 39.1, "bill_depth_mm": 18.7,
        "flipper_length_mm": 181.0, "body_mass_g": 3750.0,
        "island": "Torgersen", "sex": "male",
    }])
    probs = loaded.predict_proba(sample)[0]
    log.info("Verify sample -> %s", dict(zip(loaded.classes_, probs.round(3))))


def main() -> None:
    train, test = load_data()
    pipeline = build_pipeline()

    log.info("Training...")
    pipeline.fit(train[FEATURES], train[TARGET])

    metrics, report = evaluate(pipeline, test)
    log_to_mlflow(pipeline, train, test, metrics, report)
    package_and_verify(pipeline)
    log.info("Stage 2 done.")


if __name__ == "__main__":
    main()
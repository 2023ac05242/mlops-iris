import os
import io
import csv
import time
import sqlite3
import threading
from datetime import datetime
from typing import Optional

import joblib
import pandas as pd
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

from fastapi import (
    FastAPI,
    HTTPException,
    UploadFile,
    File,
    BackgroundTasks,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import (
    BaseModel,
    Field,
    field_validator,
    ValidationInfo,
    model_validator,
)
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, Gauge


# =========================
# FastAPI app setup
# =========================
app = FastAPI(
    title="Iris Classifier API",
    description=(
        "Predicts Iris species using either a baked pickle or the MLflow registry (@production)."
    ),
    version="1.0.0",
)

# Prometheus (make /metrics visible in Swagger)
instrumentator = Instrumentator().instrument(app).expose(
    app,
    include_in_schema=True,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# Config & globals
# =========================
# 1) Fast path: baked pickle (preferred for instant boot)
PICKLE_PATH = os.getenv("PICKLE_PATH", "baked_models/iris_best.pkl")

# 2) MLflow settings (used only if pickle missing)
#    Outside Docker (host): http://localhost:5000
#    Inside Docker (container): set env MLFLOW_TRACKING_URI=http://mlflow-server:5000
MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    "http://host.docker.internal:5000",
).strip()
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
_client = MlflowClient()

# Optionally override which registry entry to load
# If MODEL_URI is set, it wins (e.g., models:/iris-best/15 or models:/iris-best@production)
# Else, if MODEL_NAME is set, it loads models:/<MODEL_NAME>@production
# Else, it auto-discovers the first model with alias "production"
MODEL_URI_ENV = os.getenv("MODEL_URI")       # full MLflow model URI (highest priority)
MODEL_NAME_ENV = os.getenv("MODEL_NAME")     # registry name (with '@production')

# App-level model objects/flags
model = None             # the loaded model (pyfunc or sklearn)
model_name = "pending"   # human-readable name
_model_loaded = False    # flips True when a model is ready to serve

# Logs DB
LOG_DB_PATH = os.getenv("LOG_DB_PATH", "logs/logs.db")
log_dir = os.path.dirname(LOG_DB_PATH)
if log_dir:
    os.makedirs(log_dir, exist_ok=True)

# Staged data for retraining
STAGED_DATA_PATH = os.getenv("STAGED_DATA_PATH", "data/new_data.csv")
staged_dir = os.path.dirname(STAGED_DATA_PATH)
if staged_dir:
    os.makedirs(staged_dir, exist_ok=True)


# =========================
# Prometheus custom metrics
# =========================
PREDICTION_COUNTER = Counter(
    "iris_predictions_total",
    "Number of predictions served",
    ["status", "model_name"],
)

PREDICTION_LATENCY = Histogram(
    "iris_prediction_latency_seconds",
    "Latency of prediction endpoint",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)

MODEL_LOADED_GAUGE = Gauge(
    "iris_model_loaded",
    "1 if a model is loaded and ready, else 0",
)


def set_model_loaded_metric():
    MODEL_LOADED_GAUGE.set(1 if _model_loaded and model is not None else 0)


# =========================
# Utilities
# =========================
def init_logging_db():
    conn = sqlite3.connect(LOG_DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS logs (
            timestamp TEXT,
            input TEXT,
            prediction TEXT,
            status TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def get_production_model_info() -> Optional[tuple[str, int]]:
    """
    Find the first registered model having alias 'production'.
    Returns (name, version) or None if not found.
    """
    for rm in _client.search_registered_models():
        try:
            mv = _client.get_model_version_by_alias(rm.name, "production")
            return rm.name, int(mv.version)
        except Exception:
            continue
    return None


def resolve_model_uri() -> str:
    """
    Determine which MLflow model URI to load, based on env overrides.
    """
    if MODEL_URI_ENV:
        return MODEL_URI_ENV

    if MODEL_NAME_ENV:
        return f"models:/{MODEL_NAME_ENV}@production"

    info = get_production_model_info()
    if not info:
        raise RuntimeError(
            "❌ No model with alias @production found in MLflow registry."
        )
    name, _version = info
    return f"models:/{name}@production"


def load_model_now(uri: str):
    """
    Load MLflow model immediately (blocking).
    """
    print(f"Loading model from: {uri}")
    return mlflow.pyfunc.load_model(uri)


def try_load_baked_model() -> bool:
    """
    Attempt to load the baked sklearn pickle first (fast path).
    """
    global model, model_name, _model_loaded
    if os.path.exists(PICKLE_PATH):
        try:
            print(f"Loading native sklearn model from {PICKLE_PATH}")
            model = joblib.load(PICKLE_PATH)
            model_name = "iris-best-local"
            _model_loaded = True
            set_model_loaded_metric()
            return True
        except Exception as e:
            print(f"Failed to load baked model at {PICKLE_PATH}: {e}")
    return False


def lazy_load_mlflow_model_async():
    """
    If no baked model, load from MLflow in background to avoid blocking app startup.
    """
    global model, model_name, _model_loaded
    try:
        uri = resolve_model_uri()
        m = load_model_now(uri)
        model = m
        model_name = uri.replace("models:/", "")  # derive a friendly name from URI
        _model_loaded = True
        set_model_loaded_metric()
        print("MLflow model load complete.")
    except Exception as e:
        print(f"MLflow model load failed: {e}")
        set_model_loaded_metric()


def append_rows_to_csv(df: pd.DataFrame, path: str):
    exists = os.path.exists(path)
    df.to_csv(path, mode="a", header=not exists, index=False)


# Initialize DB and load model (baked first, else MLflow in background)
init_logging_db()
set_model_loaded_metric()
if not try_load_baked_model():
    threading.Thread(target=lazy_load_mlflow_model_async, daemon=True).start()


# =========================
# Schemas
# =========================
class IrisInput(BaseModel):
    sepal_length: float = Field(
        ...,
        gt=0,
        le=10,
        description="Sepal length in cm (0,10]",
        examples=[5.1],
    )
    sepal_width: float = Field(
        ...,
        gt=0,
        le=10,
        description="Sepal width in cm (0,10]",
        examples=[3.5],
    )
    petal_length: float = Field(
        ...,
        gt=0,
        le=10,
        description="Petal length in cm (0,10]",
        examples=[1.4],
    )
    petal_width: float = Field(
        ...,
        gt=0,
        le=10,
        description="Petal width in cm (0,10]",
        examples=[0.2],
    )

    @field_validator("*")
    @classmethod
    def check_reasonable_range(cls, v, info: ValidationInfo):
        if v > 10:
            raise ValueError(f"{info.field_name} seems too large: {v}")
        return v

    @model_validator(mode="after")
    def cross_field_rules(self):
        if self.petal_length <= self.petal_width:
            raise ValueError("petal_length must be greater than petal_width.")
        if self.sepal_length <= self.petal_length:
            raise ValueError("sepal_length must exceed petal_length.")
        return self


class PredictionOut(BaseModel):
    prediction: int = Field(..., description="Predicted class (0, 1, 2)")
    model_name: str = Field(..., description="Model identifier used for this prediction")


# =========================
# Endpoints
# =========================
@app.get("/", tags=["General"], summary="Welcome")
def root():
    return {
        "message": "Welcome to the Iris Classification API",
        "model_name": model_name,
        "note": "Send 4 Iris features to /predict",
    }


@app.get("/health", tags=["General"], summary="Health check")
def health():
    return {
        "status": "ok",
        "tracking_uri": MLFLOW_TRACKING_URI,
        "model_loaded": _model_loaded,
        "model_name": model_name,
        "pickle_path": PICKLE_PATH if os.path.exists(PICKLE_PATH) else None,
        "staged_data_path": STAGED_DATA_PATH if os.path.exists(STAGED_DATA_PATH) else None,
        "log_db_path": LOG_DB_PATH,
    }


@app.post(
    "/predict",
    tags=["Prediction"],
    summary="Predict Iris class",
    response_model=PredictionOut,
)
def predict(input_data: IrisInput):
    start = time.perf_counter()
    if not _model_loaded or model is None:
        PREDICTION_COUNTER.labels(status="not_ready", model_name=model_name).inc()
        raise HTTPException(
            status_code=503,
            detail="Model not loaded yet. Try again shortly.",
        )

    try:
        input_df = pd.DataFrame(
            [
                {
                    "sepal_length": input_data.sepal_length,
                    "sepal_width": input_data.sepal_width,
                    "petal_length": input_data.petal_length,
                    "petal_width": input_data.petal_width,
                }
            ]
        )
        prediction = model.predict(input_df)

        # Log success
        conn = sqlite3.connect(LOG_DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO logs (timestamp, input, prediction, status) "
            "VALUES (?, ?, ?, ?)",
            (
                datetime.utcnow().isoformat(),
                input_df.to_json(),
                str(prediction[0]),
                "success",
            ),
        )
        conn.commit()
        conn.close()

        # Ensure int for JSON
        try:
            pred_val = int(prediction[0])
        except Exception:
            pred_val = int(getattr(prediction[0], "item", lambda: prediction[0])())

        PREDICTION_COUNTER.labels(status="success", model_name=model_name).inc()
        return {"prediction": pred_val, "model_name": model_name}

    except Exception as e:
        PREDICTION_COUNTER.labels(status="error", model_name=model_name).inc()

        conn = sqlite3.connect(LOG_DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO logs (timestamp, input, prediction, status) "
            "VALUES (?, ?, ?, ?)",
            (
                datetime.utcnow().isoformat(),
                str(input_data),
                "error",
                "failure",
            ),
        )
        conn.commit()
        conn.close()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        PREDICTION_LATENCY.observe(time.perf_counter() - start)


@app.post(
    "/retrain",
    tags=["Maintenance"],
    summary="Trigger model retraining (optional)",
)
def retrain_model():
    """
    Optional retrain hook: runs your train script on the same container.
    Note: if your serving image lacks training deps/data, disable or remove this.
    """
    try:
        os.system("python src/train_models.py")
        return {"message": "Model retraining initiated successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")


# =========================
# Part 5: Logging & Monitoring
# =========================
@app.get(
    "/logs",
    tags=["Logging & Monitoring"],
    summary="List latest prediction logs (JSON)",
    description="Returns the 100 most recent prediction logs stored in SQLite.",
)
def get_logs():
    conn = sqlite3.connect(LOG_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM logs ORDER BY timestamp DESC LIMIT 100")
    rows = cursor.fetchall()
    conn.close()
    return {
        "logs": [
            {
                "timestamp": r[0],
                "input": r[1],
                "prediction": r[2],
                "status": r[3],
            }
            for r in rows
        ]
    }


@app.get(
    "/logs.csv",
    tags=["Logging & Monitoring"],
    summary="Download logs as CSV",
    description="Streams the 100 most recent logs as a CSV file.",
)
def download_logs_csv():
    conn = sqlite3.connect(LOG_DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT timestamp, input, prediction, status "
        "FROM logs ORDER BY timestamp DESC LIMIT 100"
    )
    rows = cursor.fetchall()
    conn.close()

    def iter_csv():
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["timestamp", "input", "prediction", "status"])
        for r in rows:
            writer.writerow(r)
        yield buf.getvalue()

    headers = {"Content-Disposition": 'attachment; filename="logs.csv"'}
    return StreamingResponse(iter_csv(), media_type="text/csv", headers=headers)


@app.get(
    "/logs.db",
    tags=["Logging & Monitoring"],
    summary="Download raw SQLite database",
    description="Downloads the raw SQLite file that stores all logs.",
)
def download_logs_db():
    if not os.path.exists(LOG_DB_PATH):
        raise HTTPException(status_code=404, detail="logs.db not found.")
    return FileResponse(
        LOG_DB_PATH,
        media_type="application/octet-stream",
        filename=os.path.basename(LOG_DB_PATH),
    )


@app.delete(
    "/logs",
    tags=["Logging & Monitoring"],
    summary="Clear all logs",
    description="Deletes all rows from the logs table.",
)
def clear_logs():
    conn = sqlite3.connect(LOG_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM logs")
    conn.commit()
    conn.close()
    return {"message": "All logs cleared."}


# =========================
# Data ingestion + retrain on staged data
# =========================
@app.post(
    "/ingest-json",
    tags=["Maintenance"],
    summary="Ingest new labeled data (JSON)",
    description="Array of rows with Iris features plus 'label' (0/1/2).",
)
def ingest_json(rows: list[dict]):
    required = {"sepal_length", "sepal_width", "petal_length", "petal_width", "label"}
    clean = []
    for r in rows:
        if not required.issubset(r):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Each row must include sepal_length, sepal_width, "
                    "petal_length, petal_width, label."
                ),
            )
        _ = IrisInput(
            sepal_length=float(r["sepal_length"]),
            sepal_width=float(r["sepal_width"]),
            petal_length=float(r["petal_length"]),
            petal_width=float(r["petal_width"]),
        )
        clean.append(
            {
                "sepal_length": float(r["sepal_length"]),
                "sepal_width": float(r["sepal_width"]),
                "petal_length": float(r["petal_length"]),
                "petal_width": float(r["petal_width"]),
                "label": int(r["label"]),
            }
        )
    df = pd.DataFrame(clean)
    append_rows_to_csv(df, STAGED_DATA_PATH)
    return {"message": "Rows ingested.", "rows": len(df), "path": STAGED_DATA_PATH}


@app.post(
    "/ingest-csv",
    tags=["Maintenance"],
    summary="Ingest new labeled data (CSV)",
    description="CSV with columns: sepal_length,sepal_width,petal_length,petal_width,label",
)
def ingest_csv(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
    finally:
        file.file.close()

    expected = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "label",
    ]
    if list(df.columns) != expected:
        raise HTTPException(
            status_code=400,
            detail=f"CSV must have columns: {','.join(expected)}",
        )

    for _, row in df.iterrows():
        _ = IrisInput(
            sepal_length=float(row["sepal_length"]),
            sepal_width=float(row["sepal_width"]),
            petal_length=float(row["petal_length"]),
            petal_width=float(row["petal_width"]),
        )
    df["label"] = df["label"].astype(int)

    append_rows_to_csv(df, STAGED_DATA_PATH)
    return {"message": "CSV ingested.", "rows": len(df), "path": STAGED_DATA_PATH}


@app.post(
    "/retrain-on-staged",
    tags=["Maintenance"],
    summary="Retrain model using staged data",
    description="Runs training script with STAGED_DATA_PATH as extra data.",
)
def retrain_on_staged(background_tasks: BackgroundTasks):
    if not os.path.exists(STAGED_DATA_PATH):
        raise HTTPException(status_code=404, detail="No staged data found.")

    def _run():
        cmd = (
            "python src/train_models.py "
            f"--extra_data {STAGED_DATA_PATH}"
        )
        os.system(cmd)

    background_tasks.add_task(_run)
    return {"message": "Retraining started in background."}

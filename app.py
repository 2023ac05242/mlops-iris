import os
import sqlite3
import datetime
import threading
from typing import Optional

import joblib
import pandas as pd
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, ValidationInfo
from prometheus_fastapi_instrumentator import Instrumentator


# =========================
# FastAPI app setup
# =========================
app = FastAPI(
    title="Iris Classifier API",
    description=(
        "Predicts Iris species using either a baked pickle or the MLflow registry "
        "(@production)"
    ),
    version="1.0.0",
)

# Prometheus
instrumentator = Instrumentator().instrument(app).expose(app)

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
MODEL_URI_ENV = os.getenv("MODEL_URI")  # full MLflow model URI (highest priority)
MODEL_NAME_ENV = os.getenv("MODEL_NAME")  # registry name (with '@production')

# App-level model objects/flags
model = None            # the loaded model (pyfunc or sklearn)
model_name = "pending"  # human-readable name
_model_loaded = False   # flips True when a model is ready to serve

# Logs DB
LOG_DB_PATH = "logs.db"


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
        # exact URI provided
        return MODEL_URI_ENV

    if MODEL_NAME_ENV:
        # use @production alias for the provided name
        return f"models:/{MODEL_NAME_ENV}@production"

    # auto-discover any model with @production alias
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
    print(f"📦 Loading model from: {uri}")
    return mlflow.pyfunc.load_model(uri)


def try_load_baked_model() -> bool:
    """
    Attempt to load the baked sklearn pickle first (fast path).
    """
    global model, model_name, _model_loaded
    if os.path.exists(PICKLE_PATH):
        try:
            print(f"⚡ Loading native sklearn model from {PICKLE_PATH}")
            model = joblib.load(PICKLE_PATH)
            model_name = "iris-best-local"
            _model_loaded = True
            return True
        except Exception as e:
            print(f"⚠️ Failed to load baked model at {PICKLE_PATH}: {e}")
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
        # derive a friendly name from URI
        model_name = uri.replace("models:/", "")
        _model_loaded = True
        print("✅ MLflow model load complete.")
    except Exception as e:
        print(f"❌ MLflow model load failed: {e}")


# Initialize DB and load model (baked first, else MLflow in background)
init_logging_db()
if not try_load_baked_model():
    # no baked model -> kick off background MLflow load
    threading.Thread(
        target=lazy_load_mlflow_model_async,
        daemon=True,
    ).start()


# =========================
# Request schema
# =========================
class IrisInput(BaseModel):
    sepal_length: float = Field(..., gt=0, description="Length of sepal in cm")
    sepal_width: float = Field(..., gt=0, description="Width of sepal in cm")
    petal_length: float = Field(..., gt=0, description="Length of petal in cm")
    petal_width: float = Field(..., gt=0, description="Length of petal in cm")

    @field_validator("*")
    @classmethod
    def check_reasonable_range(cls, v, info: ValidationInfo):
        if v > 10:
            raise ValueError(f"{info.field_name} seems too large: {v}")
        return v


# =========================
# Endpoints
# =========================
@app.get("/")
def root():
    return {
        "message": "Welcome to the Iris Classification API",
        "model_name": model_name,
        "note": "Send 4 Iris features to /predict",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "tracking_uri": MLFLOW_TRACKING_URI,
        "model_loaded": _model_loaded,
        "model_name": model_name,
        "pickle_path": PICKLE_PATH if os.path.exists(PICKLE_PATH) else None,
    }


@app.post("/predict")
def predict(input_data: IrisInput):
    if not _model_loaded or model is None:
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
        # Both sklearn and pyfunc models support .predict on a DataFrame
        prediction = model.predict(input_df)

        # Log success
        conn = sqlite3.connect(LOG_DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO logs (timestamp, input, prediction, status) "
            "VALUES (?, ?, ?, ?)",
            (
                datetime.datetime.utcnow().isoformat(),
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
            # some pyfunc models may return numpy scalar
            pred_val = int(getattr(prediction[0], "item", lambda: prediction[0])())

        return {"prediction": pred_val, "model_name": model_name}

    except Exception as e:
        # Log failure
        conn = sqlite3.connect(LOG_DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO logs (timestamp, input, prediction, status) "
            "VALUES (?, ?, ?, ?)",
            (
                datetime.datetime.utcnow().isoformat(),
                str(input_data),
                "error",
                "failure",
            ),
        )
        conn.commit()
        conn.close()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retrain")
def retrain_model():
    """
    Optional retrain hook: runs your train script on the same container.
    Note: if your serving image doesn't include training deps/data, you can disable/remove this.
    """
    try:
        os.system("python src/train_models.py")
        return {"message": "✅ Model retraining initiated successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")


@app.get("/logs")
def get_logs():
    conn = sqlite3.connect(LOG_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM logs ORDER BY timestamp DESC LIMIT 100")
    logs = cursor.fetchall()
    conn.close()
    return {
        "logs": [
            {"timestamp": row[0], "input": row[1], "prediction": row[2], "status": row[3]}
            for row in logs
        ]
    }

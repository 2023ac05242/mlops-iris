import os
import sqlite3
import datetime
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, ValidationInfo

import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

from prometheus_fastapi_instrumentator import Instrumentator

# ---------------------
# FastAPI app
# ---------------------
app = FastAPI(
    title="Iris Classifier API",
    description="Predicts Iris species using the best model from MLflow registry",
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

# ---------------------
# MLflow setup
# ---------------------
# In Docker network: http://mlflow-server:5000
# On host:           http://localhost:5000
mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
mlflow.set_tracking_uri(mlflow_tracking_uri)
client = MlflowClient()

# ---------------------
# Logging DB (local file in container/host)
# ---------------------
LOG_DB_PATH = "logs.db"

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

init_logging_db()

# ---------------------
# Model resolution / loading
# ---------------------
def resolve_model_uri() -> str:
    # 1) Explicit URI wins (e.g., models:/iris-best@production)
    explicit_uri = os.getenv("MODEL_URI")
    if explicit_uri:
        print(f"📦 Loading explicit MODEL_URI: {explicit_uri}")
        return explicit_uri

    # 2) Specific name + @production
    override_name = os.getenv("MODEL_NAME")
    if override_name:
        uri = f"models:/{override_name}@production"
        print(f"📦 Loading overridden model: {uri}")
        return uri

    # 3) Stable default name
    try:
        client.get_model_version_by_alias("iris-best", "production")
        print("✅ Auto-loading: models:/iris-best@production")
        return "models:/iris-best@production"
    except Exception:
        pass

    # 4) Fallback: first registry model that has @production
    for rm in client.search_registered_models():
        try:
            client.get_model_version_by_alias(rm.name, "production")
            print(f"✅ Fallback auto-loading: models:/{rm.name}@production")
            return f"models:/{rm.name}@production"
        except Exception:
            continue

    raise RuntimeError("❌ No model with alias @production found in any registered model.")

def load_model_and_name():
    uri = resolve_model_uri()
    model = mlflow.pyfunc.load_model(uri)
    name = "iris-best" if "iris-best@" in uri else os.getenv("MODEL_NAME") or uri.replace("models:/", "")
    return model, name

model, model_name = load_model_and_name()

# ---------------------
# Schema
# ---------------------
class IrisInput(BaseModel):
    sepal_length: float = Field(..., gt=0, description="Length of sepal in cm")
    sepal_width: float  = Field(..., gt=0, description="Width of sepal in cm")
    petal_length: float = Field(..., gt=0, description="Length of petal in cm")
    petal_width: float  = Field(..., gt=0, description="Width of petal in cm")

    @field_validator("*")
    @classmethod
    def check_reasonable_range(cls, v, info: ValidationInfo):
        if v > 10:
            raise ValueError(f"{info.field_name} seems too large: {v}")
        return v

# ---------------------
# Routes
# ---------------------
@app.get("/health")
def health():
    return {"status": "ok", "model_name": model_name}

@app.get("/")
def root():
    return {
        "message": "Welcome to the Iris Classification API",
        "model_name": model_name,
        "note": "Send 4 Iris features to /predict",
    }

@app.post("/predict")
def predict(input_data: IrisInput):
    try:
        input_df = pd.DataFrame([{
            "sepal_length": input_data.sepal_length,
            "sepal_width": input_data.sepal_width,
            "petal_length": input_data.petal_length,
            "petal_width": input_data.petal_width,
        }])
        prediction = model.predict(input_df)

        # Log success
        conn = sqlite3.connect(LOG_DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO logs (timestamp, input, prediction, status) VALUES (?, ?, ?, ?)",
            (
                datetime.datetime.utcnow().isoformat(),
                input_df.to_json(),
                str(prediction[0]),
                "success",
            ),
        )
        conn.commit()
        conn.close()

        # If your sklearn model predicts class indices, cast to int
        return {"prediction": int(prediction[0]), "model_name": model_name}

    except Exception as e:
        # Log failure
        conn = sqlite3.connect(LOG_DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO logs (timestamp, input, prediction, status) VALUES (?, ?, ?, ?)",
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
    try:
        # This runs inside the container filesystem.
        # Ensure your code & data are in the image or mounted as a volume if you call this.
        os.system("python src/train_models.py")
        return {"message": "✅ Model retraining initiated successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")

@app.get("/logs")
def get_logs():
    conn = sqlite3.connect(LOG_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM logs ORDER BY timestamp DESC LIMIT 100")
    rows = cursor.fetchall()
    conn.close()
    return {
        "logs": [
            {"timestamp": r[0], "input": r[1], "prediction": r[2], "status": r[3]}
            for r in rows
        ]
    }

if __name__ == "__main__":
    import uvicorn
    # For local debug; Dockerfile/CMD can invoke uvicorn directly too.
    uvicorn.run("app:app", host="0.0.0.0", port=8000)

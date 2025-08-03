import os
import mlflow
import mlflow.pyfunc
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from mlflow.tracking import MlflowClient
import pandas as pd

# --- FastAPI app setup ---
app = FastAPI(
    title="Iris Classifier API",
    description="Predicts Iris species using the @production model from MLflow registry",
    version="1.0.0"
)

# --- Add CORS middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MLflow Setup ---
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://host.docker.internal:5000"))
client = MlflowClient()

# --- Auto-load best model from @production alias ---
def get_production_model_info():
    for rm in client.search_registered_models():
        try:
            mv = client.get_model_version_by_alias(rm.name, "production")
            return rm.name, mv.version
        except Exception:
            continue
    raise RuntimeError("❌ No model with alias @production found.")

def load_production_model():
    override_name = os.getenv("MODEL_NAME")
    if override_name:
        model_uri = f"models:/{override_name}@production"
        print(f"📦 Loading overridden model: {model_uri}")
        return mlflow.pyfunc.load_model(model_uri), override_name
    else:
        model_name, version = get_production_model_info()
        model_uri = f"models:/{model_name}@production"
        print(f"✅ Auto-loading model from: {model_uri}")
        return mlflow.pyfunc.load_model(model_uri), model_name

model, model_name = load_production_model()

# --- Input schema for Iris features ---
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def root():
    return {
        "message": "Welcome to the Iris Classification API",
        "model_name": model_name,
        "note": "Send 4 Iris features to /predict"
    }



@app.post("/predict")
def predict(input_data: IrisInput):
    try:
        input_df = pd.DataFrame([{
            "sepal_length": input_data.sepal_length,
            "sepal_width": input_data.sepal_width,
            "petal_length": input_data.petal_length,
            "petal_width": input_data.petal_width
        }])
        prediction = model.predict(input_df)
        return {
            "prediction": int(prediction[0]),
            "model_name": model_name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

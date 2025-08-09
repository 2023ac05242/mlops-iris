import os
import json
import time
import pandas as pd
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, RocCurveDisplay
)

# ---------------------
# Config
# ---------------------
CONFIG_PATH = "config.json"
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    experiment_name = config.get("experiment_name", "Iris-Classification")
    override_model_name = config.get("model_name", None)
else:
    experiment_name = "Iris-Classification"
    override_model_name = None

# ---------------------
# MLflow setup
# ---------------------
# When training on host: http://localhost:5000
# When training inside Docker on same network as mlflow:
# http://mlflow-server:5000
mlflow.set_tracking_uri(
    os.environ.get(
        "MLFLOW_TRACKING_URI",
        "http://localhost:5000"))
mlflow.set_experiment(experiment_name)


def safe_log_params(params: dict):
    for k, v in params.items():
        try:
            json.dumps(v)
            mlflow.log_param(k, v)
        except Exception:
            mlflow.log_param(k, str(v))


# ---------------------
# Load data
# ---------------------
csv_path = "data/processed/iris_processed.csv"
df = pd.read_csv(csv_path)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

if not pd.api.types.is_numeric_dtype(y):
    y = LabelEncoder().fit_transform(y)

classes_sorted = sorted(set(y))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.7, random_state=42
)
y_test_binarized = label_binarize(y_test, classes=classes_sorted)

# ---------------------
# Models
# ---------------------
models = {
    "logistic_regression": LogisticRegression(max_iter=200),
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
}

metrics_dict = {}
model_objects = {}

print("\n📊 Model Evaluation Results:")
for name, model in models.items():
    with mlflow.start_run(run_name=name) as run:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average="macro", zero_division=0)
        rec = recall_score(y_test, preds, average="macro", zero_division=0)
        f1 = f1_score(y_test, preds, average="macro", zero_division=0)
        auc = roc_auc_score(y_test_binarized, proba, multi_class="ovr")

        # metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc", auc)

        # params
        mlflow.log_param("model_type", name)
        if hasattr(model, "get_params"):
            safe_log_params(model.get_params())

        # confusion matrix
        cm = confusion_matrix(y_test, preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title(f"Confusion Matrix - {name}")
        cm_path = f"confusion_matrix_{name}.png"
        plt.savefig(cm_path, bbox_inches="tight")
        mlflow.log_artifact(cm_path)
        plt.close()

        # ROC curves (one-vs-rest)
        for i in range(y_test_binarized.shape[1]):
            RocCurveDisplay.from_predictions(
                y_test_binarized[:, i],
                proba[:, i],
                name=f"Class {i} vs Rest",
                plot_chance_level=(i == 0),
            )
        plt.title(f"ROC Curve - {name}")
        plt.legend(loc="lower right")
        roc_path = f"roc_curve_{name}.png"
        plt.savefig(roc_path, bbox_inches="tight")
        mlflow.log_artifact(roc_path)
        plt.close()

        # model artifact
        input_example = X_test[:1]
        signature = infer_signature(X_test, model.predict(X_test))
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
        )

        metrics_dict[name] = {
            "run_id": run.info.run_id,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": auc,
        }
        model_objects[name] = model

        print(f"\n🧠 Model: {name}")
        print(f"  Run ID: {run.info.run_id}")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall: {rec:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  ROC AUC: {auc:.4f}")

# ---------------------
# Select best model (by F1)
# ---------------------
best_model_name = max(metrics_dict, key=lambda k: metrics_dict[k]["f1"])
best_model = model_objects[best_model_name]
best_metrics = metrics_dict[best_model_name]
best_run_id = best_metrics["run_id"]

print("\n✅ Best model selected:")
print(f"Model: {best_model_name}")
print(f"Run ID: {best_run_id}")
print(f"Accuracy: {best_metrics['accuracy']:.4f}")
print(f"Precision: {best_metrics['precision']:.4f}")
print(f"Recall: {best_metrics['recall']:.4f}")
print(f"F1 Score: {best_metrics['f1']:.4f}")
print(f"ROC AUC: {best_metrics['roc_auc']:.4f}")

# ---------------------
# Register and promote
# ---------------------
client = MlflowClient()
model_uri = f"runs:/{best_run_id}/model"

# 1) Stable name
stable_name = "iris-best"
try:
    client.create_registered_model(stable_name)
except Exception:
    pass

print(f"\n📦 Registering best model to '{stable_name}'")
version = client.create_model_version(
    name=stable_name,
    source=model_uri,
    run_id=best_run_id)

print("⏳ Waiting for model version to be READY...")
while True:
    mv = client.get_model_version(name=stable_name, version=version.version)
    if mv.status == "READY":
        break
    time.sleep(1)

client.set_registered_model_alias(
    name=stable_name,
    alias="production",
    version=version.version)
print(f"🚀 Promoted {stable_name} version {version.version} to @production")

# 2) Winner's own name
winner_name = override_model_name or best_model_name
try:
    client.create_registered_model(winner_name)
except Exception:
    pass

winner_ver = client.create_model_version(
    name=winner_name, source=model_uri, run_id=best_run_id)

print("⏳ Waiting for winner model version to be READY...")
while True:
    mv2 = client.get_model_version(
        name=winner_name,
        version=winner_ver.version)
    if mv2.status == "READY":
        break
    time.sleep(1)

client.set_registered_model_alias(
    name=winner_name,
    alias="production",
    version=winner_ver.version)
print(f"🏷️ Also set {winner_name}@production -> v{winner_ver.version}")

# ---------------------
# (Optional) Re-log best model summary
# ---------------------
with mlflow.start_run(run_name="best_model_saved") as run:
    for metric_name, metric_value in best_metrics.items():
        if metric_name != "run_id":
            mlflow.log_metric(f"best_{metric_name}", metric_value)

    mlflow.log_param("model_type", best_model_name)
    if hasattr(best_model, "get_params"):
        safe_log_params(best_model.get_params())

    signature = infer_signature(X_test, best_model.predict(X_test))
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="best_model",
        signature=signature,
        input_example=X_test[:1],
    )

# ---------------------
# Persist plain-text summary
# ---------------------
with open("best_model_info.txt", "w") as f:
    f.write(f"Best Model: {best_model_name}\n")
    f.write(f"Run ID: {best_run_id}\n")
    for k, v in best_metrics.items():
        if k != "run_id":
            f.write(f"{k.replace('_', ' ').capitalize()}: {v:.4f}\n")
    f.write(f"Registered Model Name (stable): {stable_name}\n")
    f.write(f"Also Aliased Winner Name: {winner_name}\n")
    f.write(f"Model URI: {model_uri}\n")
    f.write("Promoted to Alias: @production\n")

print("📝 Saved best model details to best_model_info.txt")
print("✅ Done.")

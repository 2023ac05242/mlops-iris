import os
import json
import pandas as pd
import time
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, RocCurveDisplay
)
from sklearn.preprocessing import label_binarize
from mlflow.models.signature import infer_signature

# --- CONFIG ---
CONFIG_PATH = "config.json"
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    experiment_name = config.get("experiment_name", "Iris-Classification")
    override_model_name = config.get("model_name", None)
else:
    experiment_name = "Iris-Classification"
    override_model_name = None

# --- MLflow Setup ---
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment(experiment_name)

def safe_log_params(params):
    for key, val in params.items():
        try:
            json.dumps(val)
            mlflow.log_param(key, val)
        except (TypeError, OverflowError):
            mlflow.log_param(key, str(val))


# --- Load processed data from CSV ---
csv_path = "data/processed/iris_processed.csv"
df = pd.read_csv(csv_path)

# Assume last column is target
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Handle string labels (e.g., setosa, versicolor...) if needed
if not pd.api.types.is_numeric_dtype(y):
    from sklearn.preprocessing import LabelEncoder
    y = LabelEncoder().fit_transform(y)

y_binarized = label_binarize(y, classes=sorted(set(y)))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.7, random_state=42
)
y_test_binarized = label_binarize(y_test, classes=sorted(set(y)))

# --- Models ---
models = {
    "logistic_regression": LogisticRegression(max_iter=200),
    "random_forest": RandomForestClassifier(n_estimators=100)
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
        prec = precision_score(y_test, preds, average='macro')
        rec = recall_score(y_test, preds, average='macro')
        f1 = f1_score(y_test, preds, average='macro')
        auc = roc_auc_score(y_test_binarized, proba, multi_class="ovr")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)

        mlflow.log_param("model_type", name)
        if hasattr(model, "get_params"):
            safe_log_params(model.get_params())

        cm = confusion_matrix(y_test, preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        cm_path = f"confusion_matrix_{name}.png"
        plt.title(f"Confusion Matrix - {name}")
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)
        plt.close()

        for i in range(y_test_binarized.shape[1]):
            RocCurveDisplay.from_predictions(
                y_test_binarized[:, i],
                proba[:, i],
                name=f"Class {i} vs Rest",
                plot_chance_level=(i == 0)
            )
        plt.title(f"ROC Curve - {name}")
        plt.legend(loc="lower right")
        roc_path = f"roc_curve_{name}.png"
        plt.savefig(roc_path)
        mlflow.log_artifact(roc_path)
        plt.close()

        input_example = X_test[:1]
        signature = infer_signature(X_test, model.predict(X_test))
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=input_example
        )

        metrics_dict[name] = {
            "run_id": run.info.run_id,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": auc
        }
        model_objects[name] = model

        print(f"\n🧠 Model: {name}")
        print(f"  Run ID: {run.info.run_id}")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall: {rec:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  ROC AUC: {auc:.4f}")

# --- Select Best Model ---
best_model_name = max(metrics_dict, key=lambda k: metrics_dict[k]["f1_score"])
best_model = model_objects[best_model_name]
best_metrics = metrics_dict[best_model_name]
best_run_id = best_metrics["run_id"]

print("\n✅ Best model selected:")
print(f"Model: {best_model_name}")
print(f"Run ID: {best_run_id}")
print(f"Accuracy: {best_metrics['accuracy']:.4f}")
print(f"Precision: {best_metrics['precision']:.4f}")
print(f"Recall: {best_metrics['recall']:.4f}")
print(f"F1 Score: {best_metrics['f1_score']:.4f}")
print(f"ROC AUC: {best_metrics['roc_auc']:.4f}")

# --- Register and Promote ---
client = MlflowClient()
registered_name = override_model_name or best_model_name
model_uri = f"runs:/{best_run_id}/model"

print(f"\n📦 Registering best model: {registered_name}")
model_version = mlflow.register_model(model_uri=model_uri, name=registered_name)

print("⏳ Waiting for model version to be READY...")
while True:
    model_info = client.get_model_version(name=registered_name, version=model_version.version)
    if model_info.status == "READY":
        break
    time.sleep(1)

client.set_registered_model_alias(name=registered_name, alias="production", version=model_version.version)
print(f"🚀 Promoted {registered_name} version {model_version.version} to @production")

# --- Re-log best model ---
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
        name="best_model",
        signature=signature,
        input_example=X_test[:1]
    )

# --- Save summary ---
with open("best_model_info.txt", "w") as f:
    f.write(f"Best Model: {best_model_name}\n")
    f.write(f"Run ID: {best_run_id}\n")
    for k, v in best_metrics.items():
        if k != "run_id":
            f.write(f"{k.replace('_', ' ').capitalize()}: {v:.4f}\n")
    f.write(f"Registered Model Name: {registered_name}\n")
    f.write(f"Model URI: {model_uri}\n")
    f.write(f"Promoted to Alias: @production\n")

print("📝 Saved best model details to best_model_info.txt")
print("✅ Best model also saved again under a new run: 'best_model_saved'")

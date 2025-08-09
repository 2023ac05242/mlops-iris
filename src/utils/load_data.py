import os
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest

RAW_DIR = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)


def load_and_save_iris():
    # Step 1: Load Iris dataset
    iris = load_iris(as_frame=True)
    df = iris.frame

    # Step 2: Save raw data
    raw_path = os.path.join(RAW_DIR, "iris_raw.csv")
    df.to_csv(raw_path, index=False)

    # Step 3: Rename columns
    df.columns = [col.lower().replace(" (cm)", "").replace(" ", "_")
                  for col in df.columns]

    # Step 4: Missing value imputation (just to simulate pipeline)
    imputer = SimpleImputer(strategy='mean')
    features = df.drop("target", axis=1)
    features_imputed = pd.DataFrame(
        imputer.fit_transform(features),
        columns=features.columns)

    # Step 5: Outlier removal using Isolation Forest
    iso = IsolationForest(contamination=0.05, random_state=42)
    yhat = iso.fit_predict(features_imputed)
    mask = yhat == 1
    features_imputed = features_imputed[mask]
    target = df["target"][mask]

    # Step 6: Feature scaling
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_imputed)

    # Step 7: Create final dataframe
    df_final = pd.DataFrame(features_scaled, columns=features.columns)
    df_final["target"] = target.values

    # Step 8: Save processed data
    processed_path = os.path.join(PROCESSED_DIR, "iris_processed.csv")
    df_final.to_csv(processed_path, index=False)

    return df_final


def load_preprocessed_data():
    processed_path = os.path.join(PROCESSED_DIR, "iris_processed.csv")
    df = pd.read_csv(processed_path)

    X = df.drop("target", axis=1)
    y = df["target"]

    return X, y


if __name__ == "__main__":
    df = load_and_save_iris()
    print("✅ Preprocessing complete. Sample:")
    print(df.head())

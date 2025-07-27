# src/load_data.py

from sklearn.datasets import load_iris
import pandas as pd
import os

def load_and_save_iris():
    iris = load_iris(as_frame=True)
    df = iris.frame
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/iris.csv", index=False)
    print("✅ Iris dataset saved to data/iris.csv")

if __name__ == "__main__":
    load_and_save_iris()

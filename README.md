# 🏗️ Iris Classifier MLOps Project

The service predicts Iris species via a FastAPI API, logs requests to SQLite, and exposes Prometheus metrics that Grafana visualizes.

## 🖼️ Architecture

![Architecture Diagram](docs/iris_architecture_graphviz.png)

---

## 📂 Folder Structure (important only)

```text
mlops-iris/
├── app.py                      # FastAPI prediction API
├── README.md                   # Project documentation
├── baked_models/                # Pre-trained model(s) & metadata
│   ├── iris_best.pkl
│   └── metadata.json
├── artifacts/                   # Evaluation outputs (reports/plots)
│   ├── logistic_regression/
│   └── random_forest/
├── data/                        # Datasets
│   ├── iris.csv
│   ├── raw/
│   └── processed/
├── monitoring-stack/            # Prometheus & Grafana stack
│   ├── docker-compose.yml
│   ├── prometheus.yml
│   ├── grafana-dashboard-iris.json
│   └── grafana/                  # provisioning & dashboards
├── src/                         # Training & pipeline code
│   ├── train_models.py
│   └── utils/
│       └── load_data.py
├── docs/                        # Diagrams & docs
│   └── iris_architecture_graphviz.png
└── .github/workflows/           # CI/CD
    └── main.yml
```

---

## 🚀 Deployment

Refer to the project documentation for Docker deployment, monitoring stack setup, and CI/CD details.


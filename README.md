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

## ⚡ Quick Start

### 1️⃣ Clone the repository
```bash
git clone <YOUR_GITHUB_REPO_URL>
cd mlops-iris
```

### 2️⃣ Create and activate virtual environment
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Train and save the best model
```bash
python src/train_models.py
```

### 5️⃣ Run locally (FastAPI + baked model)
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Open API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📊 Monitoring (Prometheus + Grafana)

### Start the monitoring stack
```bash
cd monitoring-stack
docker compose up -d
```

- **Prometheus**: [http://localhost:9090](http://localhost:9090)  
- **Grafana**: [http://localhost:3000](http://localhost:3000) (default login: `admin` / `admin`)  

Import the dashboard file: `monitoring-stack/grafana-dashboard-iris.json`

---

## 🔄 CI/CD Pipeline

This project includes a GitHub Actions workflow:
- Runs linting (`flake8`)
- Runs unit tests (if available)
- Builds and pushes Docker image to Docker Hub
- Optionally deploys container

Workflow file: `.github/workflows/main.yml`

---

## 📬 API Endpoints

| Method | Endpoint       | Description |
|--------|---------------|-------------|
| GET    | `/`            | Root endpoint with welcome message |
| GET    | `/health`      | Health check with model status |
| POST   | `/predict`     | Make a prediction |
| POST   | `/retrain`     | Trigger model retraining |
| GET    | `/logs`        | Retrieve last 100 prediction logs |
| GET    | `/metrics`     | Prometheus metrics |

---

## 📜 License

This project is licensed under the MIT License.

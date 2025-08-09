@echo off
setlocal enabledelayedexpansion

REM ==============================
REM Config
REM ==============================
REM Optional arg 1 = image tag (e.g., sha-abc1234). Defaults to "latest".
set "IMAGE_TAG=%~1"
if "%IMAGE_TAG%"=="" set "IMAGE_TAG=latest"

REM Docker Hub username (env wins)
if "%DOCKER_USERNAME%"=="" set "DOCKER_USERNAME=amitk2501"

set "REPO=%DOCKER_USERNAME%/iris-predictor-app"
set "IMAGE=%REPO%:%IMAGE_TAG%"

REM MLflow server config (persisted on host)
set "MLFLOW_URL=http://localhost:5000"
set "ART_DIR=%CD%\mlruns"
set "DB_DIR=%CD%\mlflow_db"
set "BACKEND_DB_URI=sqlite:////mlflow_db/mlflow.db"
set "MLFLOW_IMAGE=ghcr.io/mlflow/mlflow:latest"

echo ============================
echo   Local CI/CD (push + train + deploy)
echo ============================
echo Repo   : %REPO%
echo Tag    : %IMAGE_TAG%
echo Image  : %IMAGE%
echo MLflow : %MLFLOW_URL%
echo ArtDir : %ART_DIR%
echo DB Dir : %DB_DIR%
echo.

REM ===== 0) Git push (only if there are local changes) =====
set "BRANCH="
for /f "delims=" %%b in ('git rev-parse --abbrev-ref HEAD 2^>nul') do set "BRANCH=%%b"
if "%BRANCH%"=="" goto SKIP_GIT

echo On branch %BRANCH%

REM Detect changes safely (no tricky flags)
set "HAS_CHANGES="
for /f "delims=" %%L in ('git status --porcelain 2^>nul') do set "HAS_CHANGES=1"

if defined HAS_CHANGES (
  echo Staging and committing local changes...
  git add -A
  git commit -m "chore: local changes before deploy"
) else (
  echo No local changes to commit.
)

echo Pushing to origin/%BRANCH%...
git push origin %BRANCH%

:SKIP_GIT

REM ===== 1) Docker prechecks =====
docker version >nul 2>&1
if %ERRORLEVEL% neq 0 (
  echo Docker is not running. Start Docker Desktop and retry.
  exit /b 1
)

if not exist "%ART_DIR%" mkdir "%ART_DIR%"
if not exist "%DB_DIR%" mkdir "%DB_DIR%"

docker network inspect ml-network >nul 2>&1
if %ERRORLEVEL% neq 0 docker network create ml-network >nul

REM ===== 2) Start MLflow server (persisted DB + artifacts) if needed =====
for /f "tokens=*" %%N in ('docker ps --format "{{.Names}}" ^| findstr /i "^mlflow-server$"') do set "FOUND_MLFLOW=1"
if not defined FOUND_MLFLOW (
  echo Starting mlflow-server...
  docker rm -f mlflow-server >nul 2>&1
  docker run -d --name mlflow-server --network ml-network -p 5000:5000 ^
    -v "%ART_DIR%:/mlruns" ^
    -v "%DB_DIR%:/mlflow_db" ^
    %MLFLOW_IMAGE% ^
    mlflow server --host 0.0.0.0 --port 5000 ^
    --backend-store-uri %BACKEND_DB_URI% ^
    --artifacts-destination /mlruns --serve-artifacts >nul
) else (
  echo mlflow-server already running.
)

REM ===== 3) Wait for MLflow to be ready =====
set RETRIES=30
:wait_mlflow
curl -fsS %MLFLOW_URL% >nul 2>&1
if %ERRORLEVEL% neq 0 (
  set /a RETRIES-=1
  if %RETRIES% leq 0 (
    echo MLflow did not become ready in time.
    exit /b 1
  )
  timeout /t 1 >nul
  goto :wait_mlflow
)
echo MLflow OK.

REM ===== 4) Train locally (Conda env) and update iris-best@production =====
echo Training models in conda env mlops-iris ...
set "MLFLOW_TRACKING_URI=%MLFLOW_URL%"
conda run -n mlops-iris python src\train_models.py
if %ERRORLEVEL% neq 0 (
  echo Training failed.
  exit /b 1
)

REM ===== 5) Pull latest image from Docker Hub =====
echo Pulling image %IMAGE% ...
docker pull %IMAGE%
if %ERRORLEVEL% neq 0 (
  echo Failed to pull %IMAGE%. If you just pushed code, wait for GitHub Actions to finish building the image, then run deploy.bat again.
  exit /b 1
)

REM ===== 6) Restart API container =====
docker rm -f iris-predictor-app >nul 2>&1
docker run -d --name iris-predictor-app --network ml-network -p 8000:8000 ^
  -e MLFLOW_TRACKING_URI=http://mlflow-server:5000 ^
  -e MODEL_URI=models:/iris-best@production ^
  %IMAGE% >nul

timeout /t 2 >nul

REM ===== 7) Health check (/health) =====
curl -fsS http://localhost:8000/health >nul 2>&1
if %ERRORLEVEL% neq 0 (
  echo Health check failed. Recent logs:
  docker logs --tail=200 iris-predictor-app
  exit /b 1
)

echo.
echo ===== /health =====
curl -s http://localhost:8000/health
echo.

REM ===== 8) Print model version + run id + F1 from MLflow (jq-free) =====
echo ===== Model info from MLflow =====
python -c "import json,urllib.request; base='http://localhost:5000'; \
def show(name): \
    try: \
        mv=json.load(urllib.request.urlopen(f'{base}/api/2.0/mlflow/model-versions/get-by-alias?name={name}&alias=production'))['model_version']; \
        ver=mv['version']; run_id=mv['run_id']; \
        run=json.load(urllib.request.urlopen(f'{base}/api/2.0/mlflow/runs/get?run_id={run_id}'))['run']; \
        metrics={m['key']:m['value'] for m in run['data']['metrics']}; \
        f1=metrics.get('f1') or metrics.get('best_f1') or metrics.get('f1_score'); \
        print(f'{name}@production -> version {ver}, run {run_id}, f1={f1}') \
    except Exception as e: \
        print(f'{name}@production -> not set ({e.__class__.__name__})'); \
[show(n) for n in ['iris-best','logistic_regression','random_forest']]"
echo.

echo Deployment complete.
endlocal
exit /b 0

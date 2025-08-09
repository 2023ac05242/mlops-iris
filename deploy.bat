@echo off
setlocal enabledelayedexpansion

REM ---- Config ----
set "IMAGE_TAG=%~1"
if "%IMAGE_TAG%"=="" set "IMAGE_TAG=latest"
if "%DOCKER_USERNAME%"=="" set "DOCKER_USERNAME=amitk2501"
set "REPO=%DOCKER_USERNAME%/iris-predictor-app"
set "IMAGE=%REPO%:%IMAGE_TAG%"
set "MLFLOW_URL=http://localhost:5000"
set "ART_DIR=%CD%\mlruns"
set "DB_DIR=%CD%\mlflow_db"
set "BACKEND_DB_URI=sqlite:////mlflow_db/mlflow.db"
set "MLFLOW_IMAGE=ghcr.io/mlflow/mlflow:latest"

echo ============================
echo   Local CI/CD (train + deploy)
echo ============================
echo Repo   : %REPO%
echo Tag    : %IMAGE_TAG%
echo Image  : %IMAGE%
echo MLflow : %MLFLOW_URL%
echo ArtDir : %ART_DIR%
echo DB Dir : %DB_DIR%
echo.

REM ---- Prechecks ----
docker version >nul 2>&1
if errorlevel 1 goto :DOCKER_NOT_RUNNING

if not exist "%ART_DIR%" mkdir "%ART_DIR%"
if not exist "%DB_DIR%" mkdir "%DB_DIR%"

docker network inspect ml-network >nul 2>&1
if errorlevel 1 docker network create ml-network >nul

REM ---- Start MLflow if not running ----
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
)

REM ---- Wait for MLflow ----
set RETRIES=30
:WAIT_ML
curl -fsS %MLFLOW_URL% >nul 2>&1
if errorlevel 1 (
  set /a RETRIES-=1
  if %RETRIES% LEQ 0 goto :MLFLOW_TIMEOUT
  timeout /t 1 >nul
  goto :WAIT_ML
)
echo MLflow OK.

REM ---- Train ----
echo Training models in conda env mlops-iris ...
set "MLFLOW_TRACKING_URI=%MLFLOW_URL%"
conda run -n mlops-iris python src\train_models.py
if errorlevel 1 goto :TRAIN_FAIL

REM ---- Deploy ----
echo Pulling image %IMAGE% ...
docker pull %IMAGE%
if errorlevel 1 goto :PULL_FAIL

docker rm -f iris-predictor-app >nul 2>&1
docker run -d --name iris-predictor-app --network ml-network -p 8000:8000 ^
  -e MLFLOW_TRACKING_URI=http://mlflow-server:5000 ^
  -e MODEL_URI=models:/iris-best@production ^
  %IMAGE% >nul

timeout /t 2 >nul

REM ---- Health check ----
curl -fsS http://localhost:8000/health >nul 2>&1
if errorlevel 1 goto :HEALTH_FAIL

echo.
echo ===== /health =====
curl -s http://localhost:8000/health
echo.

REM ---- Model info ----
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

echo ✅ Deployment complete.
goto :EOF

:DOCKER_NOT_RUNNING
echo Docker is not running. Start Docker Desktop and retry.
exit /b 1

:MLFLOW_TIMEOUT
echo MLflow did not become ready in time.
exit /b 1

:TRAIN_FAIL
echo Training failed. Check errors above.
exit /b 1

:PULL_FAIL
echo Pull failed. Try again later.
exit /b 1

:HEALTH_FAIL
echo Health check failed. Recent logs:
docker logs --tail=200 iris-predictor-app
exit /b 1

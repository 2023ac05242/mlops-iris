@echo off
setlocal enabledelayedexpansion

echo ============================
echo   Local CI/CD (push + train + deploy)
echo ============================
set "REPO=amitk2501/iris-predictor-app"
set "TAG=latest"
set "IMAGE=%REPO%:%TAG%"
set "MLFLOW=http://localhost:5000"
set "ARTDIR=%CD%\mlruns"
set "DBDIR=%CD%\mlflow_db"

echo Repo   : %REPO%
echo Tag    : %TAG%
echo Image  : %IMAGE%
echo MLflow : %MLFLOW%
echo ArtDir : %ARTDIR%
echo DB Dir : %DBDIR%

REM ===== 0) Git push (only if there are local changes) =====
set "BRANCH="
for /f %%b in ('git branch --show-current 2^>nul') do set "BRANCH=%%b"
if "%BRANCH%"=="" goto SKIP_GIT

echo On branch %BRANCH%
REM Check if there are changes; if yes, commit
git diff-index --quiet HEAD -- 2>nul
if not "%ERRORLEVEL%"=="0" (
  echo Staging & committing local changes...
  git add .
  git commit -m "chore: local changes before deploy"
)
echo Pushing to origin/%BRANCH%...
git push origin %BRANCH%

:SKIP_GIT

REM ===== 1) Train models locally =====
call conda activate mlops-iris
python src/train_models.py

REM ===== 2) Build & push Docker image =====
docker build -t %IMAGE% .
docker push %IMAGE%

REM ===== 3) Stop existing container if running =====
docker rm -f iris-predictor-app 2>nul

REM ===== 4) Run container locally =====
docker run -d --name iris-predictor-app ^
  --network ml-network ^
  -p 8000:8000 ^
  -e MLFLOW_TRACKING_URI=%MLFLOW% ^
  %IMAGE%

REM ===== 5) Wait for container to start =====
echo Waiting for API to start...
timeout /t 10 >nul

REM ===== 6) Health check =====
echo Checking /health endpoint...
curl -s http://localhost:8000/health && echo:

REM ===== 7) Get model version + F1 from MLflow =====
echo Fetching model version and F1 score from MLflow...
for /f "tokens=* usebackq" %%i in (`curl -s %MLFLOW%/api/2.0/mlflow/registered-models/get-latest-versions -H "Content-Type: application/json" -d "{\"name\": \"iris-best\", \"stages\": [\"None\"], \"aliases\": [\"production\"]}" ^| jq -r ".model_versions[0].version"`) do set MODEL_VER=%%i

for /f "tokens=* usebackq" %%i in (`curl -s %MLFLOW%/api/2.0/mlflow/runs/get?run_id=%MODEL_VER% ^| jq -r ".run.data.metrics.f1"`) do set MODEL_F1=%%i

echo Model Version: %MODEL_VER%
echo Model F1 Score: %MODEL_F1%

echo === Deployment Complete ===
endlocal

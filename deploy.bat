@echo off
setlocal enabledelayedexpansion

REM =========================================
REM Config (you can tweak these)
REM =========================================
REM Pass image tag as 1st arg (e.g., sha-abc1234). Defaults to "latest".
set "IMAGE_TAG=%~1"
if "%IMAGE_TAG%"=="" set "IMAGE_TAG=latest"

REM Use your Docker Hub username from env if available
if "%DOCKER_USERNAME%"=="" (
  REM Fallback if env var not set — change this to your Docker Hub username
  set "DOCKER_USERNAME=amitk2501"
)

set "REPO=%DOCKER_USERNAME%/iris-predictor-app"
set "IMAGE=%REPO%:%IMAGE_TAG%"
set "MLFLOW_IMAGE=ghcr.io/mlflow/mlflow:latest"

echo.
echo ============================
echo  Local Deploy (CMD/Anaconda)
echo ============================
echo Image: %IMAGE%
echo Repo : %REPO%
echo.

REM -----------------------------------------
REM Check Docker is available
REM -----------------------------------------
docker version >nul 2>&1
if errorlevel 1 (
  echo ❌ Docker does not seem to be running. Please start Docker Desktop and retry.
  exit /b 1
)

REM -----------------------------------------
REM Ensure network exists
REM -----------------------------------------
docker network inspect ml-network >nul 2>&1
if errorlevel 1 (
  echo Creating docker network: ml-network
  docker network create ml-network >nul
)

REM -----------------------------------------
REM Start MLflow server if not running
REM -----------------------------------------
for /f "tokens=*" %%N in ('docker ps --format "{{.Names}}" ^| findstr /i "^mlflow-server$"') do set "FOUND_MLFLOW=1"
if not defined FOUND_MLFLOW (
  echo Starting mlflow-server...
  docker rm -f mlflow-server >nul 2>&1
  docker run -d --name mlflow-server --network ml-network -p 5000:5000 ^
    -v "%CD%\mlruns:/mlruns" ^
    %MLFLOW_IMAGE% ^
    mlflow server --host 0.0.0.0 --port 5000 ^
    --backend-store-uri sqlite:///mlflow.db ^
    --artifacts-destination /mlruns --serve-artifacts >nul
)

REM -----------------------------------------
REM Pull and (re)start API
REM -----------------------------------------
echo Pulling %IMAGE% ...
docker pull %IMAGE%
if errorlevel 1 (
  echo ❌ Failed to pull %IMAGE%. Make sure CI pushed it to Docker Hub.
  exit /b 1
)

echo Restarting iris-predictor-app...
docker rm -f iris-predictor-app >nul 2>&1
docker run -d --name iris-predictor-app --network ml-network -p 8000:8000 ^
  -e MLFLOW_TRACKING_URI=http://mlflow-server:5000 ^
  -e MODEL_URI=models:/iris-best@production ^
  %IMAGE% >nul

REM -----------------------------------------
REM Health check
REM -----------------------------------------
echo Waiting for API to start...
timeout /t 2 >nul
curl -fsS http://localhost:8000/health >nul 2>&1
if errorlevel 1 (
  echo ❌ Health check failed. Last logs:
  docker logs --tail=200 iris-predictor-app
  exit /b 1
)

echo ✅ Deployed OK!
curl -s http://localhost:8000/health
echo.
endlocal
exit /b 0

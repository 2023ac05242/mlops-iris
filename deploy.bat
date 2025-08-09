@echo off
setlocal ENABLEDELAYEDEXPANSION

REM ====== config ======
set "IMAGE=amitk2501/iris-predictor-app:latest"
set "CONTAINER=iris-predictor-app"
set "PORT=8000"
set "PICKLE_PATH=baked_models/iris_best.pkl"

echo ============================
echo   Local Deploy (baked model)
echo ============================
echo Image     : %IMAGE%
echo Container : %CONTAINER%
echo Port      : %PORT%
echo Pickle    : %PICKLE_PATH%
echo.

REM Stop/remove any previous container
docker rm -f %CONTAINER% >nul 2>&1

REM (Optional) pull latest
REM docker pull %IMAGE% >nul

REM Run container (serve baked model; no MLflow needed)
echo Starting container...
docker run -d --name %CONTAINER% -p %PORT%:8000 ^
  -e PICKLE_PATH=%PICKLE_PATH% ^
  %IMAGE% >nul
IF ERRORLEVEL 1 (
  echo ❌ Failed to start container. Ensure the image exists: %IMAGE%
  goto :fail
)

echo.
echo Waiting for /health and model_loaded=true (up to 60s)...
set "HEALTH="
for /l %%i in (1,1,60) do (
  >nul timeout /t 1
  for /f "usebackq delims=" %%H in (`curl -s http://localhost:%PORT%/health`) do set "HEALTH=%%H"
  echo !HEALTH! | findstr /c:"\"status\":\"ok\"" >nul && (
    echo !HEALTH! | findstr /c:"\"model_loaded\":true" >nul && goto :health_ok
  )
  if %%i lss 60 echo  . waiting (%%i/60)
)

echo ❌ Health check did not reach model_loaded=true in time.
echo --- /health response ---
echo %HEALTH%
echo --- recent logs ---
docker logs --tail=200 %CONTAINER%
echo --- checking baked model inside container ---
docker exec %CONTAINER% sh -lc "ls -l %PICKLE_PATH% || ls -l baked_models || echo 'pickle not found'"
goto :fail

:health_ok
echo ✅ Health OK:
echo %HEALTH%
echo.

echo Warming up with a sample prediction...
curl -s -X POST "http://localhost:%PORT%/predict" ^
     -H "Content-Type: application/json" ^
     -d "{\"sepal_length\":5.1,\"sepal_width\":3.5,\"petal_length\":1.4,\"petal_width\":0.2}"
echo.

echo 🎉 Deployment complete.
goto :done

:fail
endlocal
exit /b 1

:done
endlocal
exit /b 0

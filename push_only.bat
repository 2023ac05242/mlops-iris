@echo off
setlocal enabledelayedexpansion

echo ============================
echo   Git Push (safe)
echo ============================

set "MSG=%~1"
if "%MSG%"=="" set "MSG=chore: local changes before deploy"

echo On branch:
git branch --show-current

echo.
echo Adding changes...
git add .

echo.
echo Committing with message: "!MSG!"
git commit -m "!MSG!"
if errorlevel 1 (
    echo ⚠️ Commit failed. Possibly no changes to commit.
    goto push
)

:push
echo.
echo Pushing to origin/main...
git push origin main
if errorlevel 1 (
    echo ❌ Push failed.
    exit /b 1
) else (
    echo ✅ Push successful.
)

endlocal

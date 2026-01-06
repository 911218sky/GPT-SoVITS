@echo off

echo ==========================================
echo   GPT-SoVITS Docker Stop Tool
echo ==========================================
echo.

REM Change to project root directory
cd /d "%~dp0.."

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not running. Please start Docker Desktop first.
    pause
    exit /b 1
)

echo [INFO] Stopping container...
docker compose stop GPT-SoVITS-CU128

echo.
echo [DONE] Container stopped.
echo.
pause

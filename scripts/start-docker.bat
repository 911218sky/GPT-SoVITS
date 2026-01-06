@echo off

echo ==========================================
echo   GPT-SoVITS Docker Launcher
echo   Default: GPT-SoVITS-CU128
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

echo [INFO] Pulling the latest Docker image...
docker compose pull GPT-SoVITS-CU128

echo.
echo [INFO] Starting GPT-SoVITS-CU128 container...
docker compose up GPT-SoVITS-CU128

pause

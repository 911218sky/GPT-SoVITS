@echo off
chcp 65001 >nul

echo ==========================================
echo   GPT-SoVITS Docker 啟動器
echo   預設使用: GPT-SoVITS-CU128
echo ==========================================
echo.

REM 檢查 Docker 是否正在運行
docker info >nul 2>&1
if errorlevel 1 (
    echo [錯誤] Docker 未運行，請先啟動 Docker Desktop
    pause
    exit /b 1
)

echo [資訊] 正在拉取最新的 Docker 映像...
docker compose pull GPT-SoVITS-CU128

echo.
echo [資訊] 正在啟動 GPT-SoVITS-CU128 容器...
docker compose up GPT-SoVITS-CU128

pause

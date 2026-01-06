@echo off
chcp 65001 >nul

echo ==========================================
echo   GPT-SoVITS API Server
echo   Default Port: 9880
echo ==========================================
echo.

set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
cd /d "%SCRIPT_DIR%"
set "PATH=%SCRIPT_DIR%\runtime;%PATH%"

echo [INFO] Starting API Server...
echo [INFO] API will be available at: http://127.0.0.1:9880
echo [INFO] Press Ctrl+C to stop the server
echo.

runtime\python.exe api_v2.py -a 127.0.0.1 -p 9880

pause

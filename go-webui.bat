set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
cd /d "%SCRIPT_DIR%"
set "PATH=%SCRIPT_DIR%\runtime\python;%SCRIPT_DIR%\runtime\python\Scripts;%PATH%"
runtime\python\python.exe -I webui.py zh_CN
pause

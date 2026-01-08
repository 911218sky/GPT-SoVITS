$ErrorActionPreference = "SilentlyContinue"
chcp 65001
Set-Location $PSScriptRoot
$envPath = Join-Path $PSScriptRoot "runtime\python"
$env:PATH = "$envPath;$envPath\Scripts;$env:PATH"
& "$envPath\python.exe" -I "$PSScriptRoot\webui.py" zh_CN
pause

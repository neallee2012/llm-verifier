@echo off
setlocal
cd /d %~dp0
set "LOG_DIR=%~dp0logs"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
powershell -NoProfile -Command "python -m uvicorn app.main:app --reload | Tee-Object -FilePath '%LOG_DIR%\server.log' -Append; exit $LASTEXITCODE"
endlocal

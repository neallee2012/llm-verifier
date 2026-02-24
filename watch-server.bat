@echo off
setlocal
cd /d %~dp0
set "POLL_SECONDS=5"
set "LOG_DIR=%~dp0logs"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

:loop
powershell -NoProfile -Command "python -m uvicorn app.main:app --reload | Tee-Object -FilePath '%LOG_DIR%\server.log' -Append; exit $LASTEXITCODE"
set "EXITCODE=%ERRORLEVEL%"

echo Server stopped (exit %EXITCODE%). Waiting %POLL_SECONDS% seconds before restart.
set /a "WAIT=%POLL_SECONDS%"
:wait
if %WAIT% LEQ 0 goto loop
timeout /t 1 /nobreak >nul
set /a "WAIT=%WAIT%-1"

goto loop
